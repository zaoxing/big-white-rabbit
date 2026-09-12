"""Thin argv dispatcher.

Mirrors Big White Rabbit's ``cli.py`` shape (parse a subcommand, lazily import the module that
implements it) so the Phase 2 control-plane port can slot ``serve``/``shell``/``ctl`` in
next to these without restructuring. Phase 0 ships only ``generate`` and ``info``.
"""

from __future__ import annotations

import argparse
import sys
import time


def _add_model_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--model", "-m", required=required, help="model weights: a .gguf file for --engine metal, a directory for --engine mlx")
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="layers on the Metal backend (-1 = all, the unified-memory default)")
    p.add_argument("--ctx-size", "-c", type=int, default=4096, help="context length")
    p.add_argument("--no-flash-attn", action="store_true", help="disable flash attention")


def _cmd_info(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="bwr info", description="Print GGUF/model metadata.")
    _add_model_args(ap, required=True)
    args = ap.parse_args(argv)

    from . import Model, ModelParams

    mp = ModelParams()
    mp.n_gpu_layers = args.n_gpu_layers
    # Metadata only: simulate allocations instead of paying for the weights.
    mp.no_alloc = True
    model = Model(args.model, mp)

    print(f"desc          : {model.desc}")
    print(f"arch          : {model.meta_val('general.architecture')}")
    print(f"params        : {model.n_params / 1e9:.2f} B")
    print(f"size          : {model.size_bytes / 1024**3:.2f} GiB")
    print(f"n_layer       : {model.n_layer}")
    print(f"n_embd        : {model.n_embd}")
    print(f"n_vocab       : {model.n_vocab}")
    print(f"n_ctx_train   : {model.n_ctx_train}")
    for key in ("general.name", "general.quantization_version", "general.file_type"):
        val = model.meta_val(key)
        if val:
            print(f"{key:<14}: {val}")
    return 0


def _cmd_generate(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="bwr generate", description="Generate from a prompt.")
    _add_model_args(ap, required=True)
    ap.add_argument("--prompt", "-p", default="Explain what a mixture-of-experts model is, briefly.")
    ap.add_argument("--max-tokens", "-n", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    args = ap.parse_args(argv)

    from . import Context, ContextParams, Model, ModelParams, SamplerParams, generate

    mp = ModelParams()
    mp.n_gpu_layers = args.n_gpu_layers

    t0 = time.monotonic()
    model = Model(args.model, mp)
    t_load = time.monotonic() - t0

    cp = ContextParams()
    cp.n_ctx = args.ctx_size
    cp.flash_attn = not args.no_flash_attn

    sp = SamplerParams()
    sp.temp = args.temp

    ctx = Context(model, cp, sp)

    print(f"[loaded {model.desc} in {t_load:.2f}s]\n", file=sys.stderr)
    print(args.prompt, end="", flush=True)

    n_tok = 0
    t1 = time.monotonic()
    for piece in generate(model, ctx, args.prompt, max_tokens=args.max_tokens):
        print(piece, end="", flush=True)
        n_tok += 1
    elapsed = time.monotonic() - t1

    rate = n_tok / elapsed if elapsed > 0 else 0.0
    print(f"\n\n[{n_tok} tokens in {elapsed:.2f}s = {rate:.1f} tok/s]", file=sys.stderr)
    return 0


def _cmd_serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="bwr serve", description="Serve an OpenAI-compatible API over a GGUF model."
    )
    _add_model_args(ap, required=False)
    ap.add_argument("--recipe", default=None, metavar="PATH|27b|30b",
                    help="ready-to-use recipe: path to JSON or shorthand '27b' (MLX 13.2 tok/s) / '30b' (Metal 57 tok/s, prefix-cache 200× on 21k); see models/recipes/")
    ap.add_argument("--receipt", default=None, metavar="PATH|27b|30b",
                    help=argparse.SUPPRESS)  # deprecated alias for --recipe
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--n-ubatch", type=int, default=None,
                    help="micro-batch (default = n-batch; 512 is the tuned value for 30B MoE)")
    ap.add_argument("--n-seq-max", type=int, default=8,
                    help="max concurrent requests (each gets ctx-size/n-seq-max tokens "
                         "of KV unless --kv-unified)")
    ap.add_argument("--n-threads", type=int, default=0,
                    help="CPU threads (0 = auto; 10 is tuned for 30B MoE on M1 Max)")
    ap.add_argument("--n-threads-batch", type=int, default=0,
                    help="CPU threads for batch/prefill (0 = auto; keep 0, nonzero measured -7%%)")
    ap.add_argument("--speculative", action="store_true",
                    help="n-gram speculative decoding (greedy only; +5-7%% on repetitive text)")
    ap.add_argument("--spec-max-drafts", type=int, default=4,
                    help="max n-gram drafts per step (4 tuned; 8 collapses acceptance)")
    ap.add_argument("--mlx-prefix-cache", action="store_true",
                    help="MLX exact-prefix prompt cache: repeat prompts skip prefill "
                         "(agent-loop TTFT; 2K 28s->~1s measured)")
    ap.add_argument("--mlx-prefix-cache-size", type=int, default=2,
                    help="max cached MLX prompt snapshots, LRU (default 2)")
    ap.add_argument("--kv-unified", action="store_true",
                    help="share one KV buffer across sequences; required for partial "
                         "prefix copies (see docs/llamacpp-notes.md)")
    ap.add_argument("--served-model-name", default=None,
                    help="name reported by /v1/models (defaults to the GGUF's own)")
    ap.add_argument("--draft-model", default=None, metavar="GGUF",
                    help="draft model for draft-model speculation (SPEC-draft-model.md); "
                         "its weights stay resident alongside the target's")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="auto-pin repeated prompt prefixes to skip re-prefill "
                         "across requests (SPEC-prefix-cache.md)")
    ap.add_argument("--record-experts", action="store_true",
                    help="record MoE router distributions (SPEC-residency.md); single-seq only, profiling cost")
    ap.add_argument("--ssd-hotlist", action="store_true",
                    help="SSD hotlist LRU over routed experts (SPEC-ssd-hotlist.md, ds4-inspired); needs --record-experts")
    ap.add_argument("--ssd-hotlist-k", type=int, default=32,
                    help="resident experts per layer for hotlist (default 32)")
    ap.add_argument("--ssd-hotlist-bytes", default=None, metavar="BYTES",
                    help="byte budget alternative to --ssd-hotlist-k, e.g. 32GB or 1073741824 (ds4: --ssd-streaming-cache-experts)")
    ap.add_argument("--engine", default="mlx", choices=("metal", "mlx"),
                    help="inference backend (default mlx; 'metal' serves a GGUF via llama.cpp)")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    # Explicitly-passed flags beat recipe values (fix: recipe used to
    # clobber them silently). Long opts map directly; shorts cover -m/-c.
    explicit: set[str] = set()
    for tok in argv:
        if tok.startswith("--"):
            explicit.add(tok[2:].replace("-", "_").split("=")[0])
        elif tok in ("-m", "-c"):
            explicit.add({"-m": "model", "-c": "ctx_size"}[tok])

    # --recipe shorthand: 27b → models/recipes/27b.json, 30b → 30b.json (with --receipt deprecated alias)
    recipe_arg = args.recipe if args.recipe is not None else args.receipt
    if recipe_arg is not None:
        import json, pathlib
        recipe_path = recipe_arg
        if recipe_path in ("27b", "27B", "qwen27b", "27"):
            recipe_path = "models/recipes/27b.json"
        elif recipe_path in ("30b", "30B", "qwen30b", "30", "moe"):
            recipe_path = "models/recipes/30b.json"
        p = pathlib.Path(recipe_path)
        if not p.exists():
            print(f"bwr: recipe {recipe_arg!r} not found at {p}", file=sys.stderr)
            return 2
        data = json.loads(p.read_text())
        # Recipe keys with their argparse dests (n_ctx maps to ctx_size).
        # Unknown keys are rejected: silently dropping a perf-critical knob
        # (as happened with flash_attn) is worse than failing fast.
        known: dict[str, str] = {
            "model": "model", "engine": "engine", "ctx_size": "ctx_size",
            "n_ctx": "ctx_size", "n_batch": "n_batch", "n_ubatch": "n_ubatch",
            "n_seq_max": "n_seq_max", "n_threads": "n_threads",
            "n_threads_batch": "n_threads_batch", "kv_unified": "kv_unified",
            "flash_attn": "no_flash_attn",
            "speculative": "speculative", "spec_max_drafts": "spec_max_drafts",
            "prefix_cache": "prefix_cache",
            "prefix_cache_pins": "prefix_cache_pins",
            "prefix_cache_min_tokens": "prefix_cache_min_tokens",
            "mlx_prefix_cache": "mlx_prefix_cache",
            "mlx_prefix_cache_size": "mlx_prefix_cache_size",
            "mlx_kv_bits": "mlx_kv_bits",
            "comment": "", "bench": "", "fallback_gguf": "",
            "fallback_engine": "", "mlx_fallback": "", "mlx_engine": "",
        }
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            print(f"bwr: recipe {p} has unknown keys: {', '.join(unknown)}",
                  file=sys.stderr)
            return 2
        for k, dest in known.items():
            if not dest or k not in data or dest in explicit:
                continue
            if k == "flash_attn":
                # Inverted polarity: recipe true == flag absent.
                args.no_flash_attn = not data[k]
            else:
                setattr(args, dest, data[k])

    if args.model is None:
        print("bwr serve: --model is required unless --recipe/--receipt is given", file=sys.stderr)
        return 2

    # Draft-model speculation and n-gram speculation are mutually exclusive
    # (MetalEngine refuses both). An explicitly passed --draft-model is the
    # rarer, deliberate choice, so it wins over recipe-enabled spec with a
    # warning; explicitly passing both is a genuine conflict and still errors.
    if args.draft_model and args.speculative:
        if "speculative" not in explicit:
            print("bwr: --draft-model disables recipe-enabled n-gram speculation",
                  file=sys.stderr)
            args.speculative = False

    try:
        from .server.launch import serve
    except ImportError as exc:  # fastapi/uvicorn are the [serve] extra, not a core dep
        print(
            f"bwr serve needs the server extras: pip install 'big-white-rabbit[serve]' ({exc})",
            file=sys.stderr,
        )
        return 1

    # Byte budget string -> int (ds4: "32GB" or "4000" slots; we support bytes)
    ssd_bytes = None
    if args.ssd_hotlist_bytes is not None:
        s = str(args.ssd_hotlist_bytes).strip().upper()
        mult = 1
        for suffix, factor in [("GB", 1<<30), ("G", 1<<30), ("MB", 1<<20), ("M", 1<<20), ("KB", 1<<10), ("K", 1<<10)]:
            if s.endswith(suffix):
                mult = factor
                s = s[:-len(suffix)]
                break
        try:
            ssd_bytes = int(float(s) * mult)
        except ValueError:
            print(f"bwr: invalid --ssd-hotlist-bytes {args.ssd_hotlist_bytes!r}", file=sys.stderr)
            return 2
        # byte budget implies hotlist on
        args.ssd_hotlist = True
        if not args.record_experts:
            args.record_experts = True

    serve(
        args.model,
        host=args.host,
        port=args.port,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.ctx_size,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_seq_max=args.n_seq_max,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads_batch,
        flash_attn=not args.no_flash_attn,
        kv_unified=args.kv_unified,
        served_model_name=args.served_model_name,
        log_level=args.log_level,
        draft_model_path=args.draft_model,
        speculative=args.speculative,
        spec_max_drafts=args.spec_max_drafts,
        mlx_prefix_cache=args.mlx_prefix_cache,
        mlx_prefix_cache_size=args.mlx_prefix_cache_size,
        mlx_kv_bits=getattr(args, "mlx_kv_bits", None),
        prefix_cache=args.prefix_cache,
        prefix_cache_pins=getattr(args, "prefix_cache_pins", 2),
        prefix_cache_min_tokens=getattr(args, "prefix_cache_min_tokens", 256),
        record_experts=args.record_experts,
        ssd_hotlist=args.ssd_hotlist,
        ssd_hotlist_k=args.ssd_hotlist_k,
        ssd_hotlist_bytes=ssd_bytes,
        engine=args.engine,
    )
    return 0


_COMMANDS = {
    "info": _cmd_info,
    "generate": _cmd_generate,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: bwr <{'|'.join(_COMMANDS)}> [options]", file=sys.stderr)
        return 0 if argv else 2

    cmd, rest = argv[0], argv[1:]
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"bwr: unknown command {cmd!r}; expected one of {', '.join(_COMMANDS)}", file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
