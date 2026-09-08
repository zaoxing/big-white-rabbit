"""Thin argv dispatcher.

Mirrors FreeToken's ``cli.py`` shape (parse a subcommand, lazily import the module that
implements it) so the Phase 2 control-plane port can slot ``serve``/``shell``/``ctl`` in
next to these without restructuring. Phase 0 ships only ``generate`` and ``info``.
"""

from __future__ import annotations

import argparse
import sys
import time


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", required=True, help="path to a .gguf model file")
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="layers on the Metal backend (-1 = all, the unified-memory default)")
    p.add_argument("--ctx-size", "-c", type=int, default=4096, help="context length")
    p.add_argument("--no-flash-attn", action="store_true", help="disable flash attention")


def _cmd_info(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ftm info", description="Print GGUF/model metadata.")
    _add_model_args(ap)
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
    ap = argparse.ArgumentParser(prog="ftm generate", description="Generate from a prompt.")
    _add_model_args(ap)
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


_COMMANDS = {
    "info": _cmd_info,
    "generate": _cmd_generate,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: ftm <{'|'.join(_COMMANDS)}> [options]", file=sys.stderr)
        return 0 if argv else 2

    cmd, rest = argv[0], argv[1:]
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"ftm: unknown command {cmd!r}; expected one of {', '.join(_COMMANDS)}", file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
