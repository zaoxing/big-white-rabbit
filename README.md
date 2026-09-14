# Big White Rabbit

The whole motivation is to explore the optimizations of model serving engine for Apple Silicon.

An edge-native MoE serving engine for Apple Silicon — the ideas behind
[FreeToken](https://github.com/FlashML-org/FreeToken) (bandwidth-adaptive MoE placement,
semantic-aware KV caching, an Anthropic/OpenAI-compatible API for coding agents) rebuilt on
Apple's [MLX](https://github.com/ml-explore/mlx) framework, with the original
[llama.cpp](https://github.com/ggml-org/llama.cpp) Metal/ggml backend still available.

> **Status: serving.** OpenAI- *and* Anthropic-compatible APIs with continuous
> batching and tool calling. MLX is the default backend (measured ~1.24x the
> Metal path in-harness, single-stream plain decode); `--engine metal` keeps
> the llama.cpp path with speculation, prefix caching, and MoE residency.
> Expert residency control exists; no prefetch policy or semantic KV caching yet.

## Architecture

```
   HTTP (OpenAI / Anthropic compatible)
               │
   ┌──────────▼───────────┐
   │  control plane       │  FastAPI routes + Pydantic schemas, single process
   │  (python/…/server)   │  (no ZMQ: FreeToken's multi-process design exists for
   └──────────┬───────────┘   multi-GPU CUDA contexts, which UMA does not need)
               │ in-process call
   ┌──────────▼───────────┐
   │  engine              │  MLXEngine (default): one mlx-lm generator per
   │  (python/…/engine)   │  request, stepped in lockstep
   │                      │  MetalEngine (--engine metal): admission table +
   │                      │  step loop over llama_batch, continuous batching
   └──────────┬───────────┘
               │ mlx-lm  │  pybind11 (metal path only)
   ┌──────────▼───────────┐
   │  MLX / Metal        │  Apple frameworks; llama.cpp/ggml vendored as a
   │  backends           │  submodule for the metal path
   └──────────────────────┘
```

## Requirements

- Apple Silicon Mac, macOS 14+ (developed on M1 Max / 64 GB)
- Python 3.11+ — **not** the macOS system `python3` (a stub): `brew install python@3.11`, or python.org
- Xcode command-line tools: `xcode-select --install` (compiler for the metal-path extension build, which runs on every install)
- ~20 GB free disk per 27B-class model; 24 GB+ free RAM to serve one at `n_ctx 8192` (smaller Macs: `bwr host` / `--no-adapt` in [Host adaptation](#host-adaptation--tuning) auto-clamps `n_ctx` to fit)

## Quick start

```bash
# 1. Clone (submodules carry llama.cpp for the metal backend)
git clone --recurse-submodules <this repo> && cd big-white-rabbit

# 2. Fresh venv with a new pip (system pip is routinely too old)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[serve]"
# ^ builds the metal extension too (~10-20 min first time, cached after);
#   the default MLX backend needs no build of its own.

# 3. Fetch weights — ONE of:
# MLX (default backend), e.g. 27B 4-bit (~16 GB, multi-file: repeat per file into one dir)
curl -L -o qwen38-mlx-4bit/model-00001-of-00003.safetensors \
  https://huggingface.co/orcarouter/Qwen3.8-27B-MLX/resolve/main/4-bit/model-00001-of-00003.safetensors
# GGUF (metal backend), e.g. 30B MoE (~17 GB, single file)
curl -L -o models/qwen3-30b.gguf \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf

# 4. Serve (default port 1919) — ready-to-use recipes
bwr serve --recipe 27b   # 27B dense qwen35 hybrid: MLX 10.9 tok/s, 4bit 15G, n_ctx 8192 (fallback GGUF: bwr serve --recipe 27b --engine metal)
bwr serve --recipe 30b   # 30B-A3B MoE qwen3moe: Metal 58.2 tok/s (vs MLX 15.97), 4bit 17G, n_ctx 8192, prefix-cache 200× on 21k, spec +7.5% on rep
# or explicit:
bwr serve -m models/Qwen3.8-27B-MLX-4bit --engine mlx --ctx-size 8192
bwr serve -m models/Qwen3-30B-A3B-Q4_K_M.gguf --engine metal --ctx-size 8192 --n-seq-max 2 --kv-unified --prefix-cache --speculative

# 5. Check it answers
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' -d \
  '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```
bwr info     -m /path/to/model.gguf
bwr host     -m /path/to/model.gguf -c 8192   # Mac/chip/GPU/RAM probe + ctx fit (also see --no-adapt below)
bwr tune -m models/Qwen3.8-27B-Q4_K_M.gguf --depths off,2,4   # per-machine n-gram depth tuning (report-only, Metal+MLX)
```

Point either an OpenAI or an Anthropic client at it — one server, one loaded model,
both protocols:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:1919/v1", api_key="none")
print(c.chat.completions.create(
    model="local", messages=[{"role": "user", "content": "hi"}]
).choices[0].message.content)

from anthropic import Anthropic
a = Anthropic(base_url="http://127.0.0.1:1919", api_key="none")
print(a.messages.create(
    model="local", max_tokens=64, messages=[{"role": "user", "content": "hi"}]
).content[0].text)
```

| Endpoint | Protocol |
|---|---|
| `POST /v1/chat/completions` | OpenAI (streaming + tools) |
| `POST /v1/messages` | Anthropic Messages (streaming + tools) |
| `GET /v1/models`, `GET /health` | — |

Both surfaces share one prompt format and one tool-call parser: the client's choice of
API never reaches the model, which sees only the format its chat template was trained on.

On `--engine metal`, `--n-seq-max` is how many requests decode concurrently. With the
default split KV buffer each sequence gets `ctx-size / n-seq-max` tokens of context, so
raising concurrency shrinks per-request context; `/health` reports both `n_ctx` and the
per-sequence `n_ctx_seq`. Pass `--kv-unified` to share one buffer instead. The MLX
backend serves requests from independent generators (batching parity is follow-up work).

## Ready-to-use recipes (bench on M1 Max 64GB; `bench` in each JSON)

| Recipe | Model | Engine | `n_ctx` | `tok/s` | Notes |
|---|---|---|---|---|---|
| `bwr serve --recipe 27b` | `Qwen3.8-27B-Uncensored-MLX-4bit/4-bit` `qwen35` hybrid 64L `248320` | `mlx` | `8192` | `10.9` | `4bit 15G` fallback `Q4_K_M` `9.85` `metal`; `spec` off (hybrid); `mlx_prefix_cache` on (repeat 2K TTFT 28s→0.07s, parity exact) |
| `bwr serve --recipe 30b` | `Qwen3-30B-A3B-Q4_K_M.gguf` `qwen3moe` 48L `151936` | `metal` | `8192` `n_seq_max=2 kv_unified` | `58.2` | `vs MLX 15.97` `3.6×`; `spec +7.5%` rep (`drafts=4`), `prefix-cache 123.22s→0.62s 200×` on `21k`, `MLX fallback` `models/Qwen3-30B-A3B-4bit` |

Recipes are `models/recipes/27b.json` / `30b.json` (JSON `EngineConfig` + `model` + `bench`); `bwr serve --recipe 27b --port 1919` or `bwr serve --recipe models/recipes/30b.json` (explicit `CLI` wins; `--receipt` is a deprecated alias). Q4_K_M remains the 30B recipe: Q8_0 fits at 8192 (30.25 GiB) but measured 34% slower decode on this Mac (llama-bench `tg128` 61.2→45.45; `bwr generate` 58.5→41.0 tok/s).

## Host adaptation & tuning

Recipes are tuned on a 64 GB M1 Max. `bwr serve` / `generate` / `tune` probe the host
(`hw.model`, `machdep.cpu.brand_string`, `hw.ncpu`, `hw.memsize`, GPU cores from
`system_profiler`) and clamp `--ctx-size` down `8192→4096→2048→1024→512` until
`weights + 4 GiB OS reserve + KV` fit. Use the probe directly:

```bash
bwr host                          # caps only (chip, CPUs, GPU cores, RAM)
bwr host -m models/Qwen3-30B-A3B-Q4_K_M.gguf -c 8192   # fit verdict for a model
```

Explicit `--ctx-size` still wins but fails fast with an actionable error when it
cannot fit (weights >75% of RAM also refuse). Add `--no-adapt` to any of
`serve` / `generate` / `tune` to take responsibility for the raw value.
Bias is conservative: under-clamping wastes context, over-clamping OOMs — worst
case you lose a ladder rung, not the process.

`bwr tune` is a per-machine, report-only tuner for `speculative` n-gram depth
(depths `off,2,4`; 5% noise band): one engine per candidate, median of `--reps`.

```bash
bwr tune -m models/Qwen3.8-27B-Q4_K_M.gguf --engine metal --depths off,2,4 --reps 3 --max-tokens 128
bwr tune --engine mlx -m models/Qwen3.8-27B-MLX-4bit --depths off,2,4
# verdict: keep baseline | --speculative --spec-max-drafts N (+% vs AR)
```

## Optional features (all default off unless noted)

| Flag / knob | What | Notes |
|---|---|---|
| `--engine mlx` (default) / `metal` | Inference backend | Metal keeps the llama.cpp path below |
| `EngineConfig(speculative=True)` | N-gram speculative decoding | Both backends, greedy only; ~1.5x on repetitive text, parity on prose; use `bwr tune` to pick depth per machine |
| `--draft-model GGUF` | Draft-model speculation (metal) | Attention targets only; refused on hybrids |
| `--prefix-cache` | Pin repeated prompt prefixes, skip re-prefill (metal) | Attention only; ~20x TTFT win measured |
| `mlx_kv_bits=8` / `EngineConfig(mlx_kv_bits=8)` | Quantized KV cache (MLX qwen35) | 2x KV headroom for long context; default `f16` (None); live-KV `q8` measured neutral on decode |
| `ModelParams(expert_weights="cpu")` | MoE expert weights on CPU (metal) | Residency knob only — no prefetch policy yet |
| `bwr host` / `--no-adapt` | Host RAM fit for `n_ctx` | Auto-clamp down ladder as above; explicit wins, `--no-adapt` disables |
| `--model-dir DIR` | Multi-model pool: lazy load + LRU eviction | Instead of `-m`; nothing loads until a request names a model. See below |
| `--mlx-mtp` | MTP-head speculation (MLX) | Output-identical but **slower** on mlx-lm 0.31.3 — the trunk forward is linear in rows. Off for a reason; see `SPEC-mlx-mtp-draft.md` |

## Multi-model pool & web UI

`--model-dir` serves every model under a directory instead of one fixed model.
Engines load lazily on first request and the pool evicts least-recently-used
ones to stay inside a memory budget, so startup is instant no matter how many
27Bs are on disk:

```bash
bwr serve --model-dir models --port 1919
curl localhost:1919/v1/models                 # everything discovered
curl localhost:1919/health                    # what is resident right now
```

Discovery takes MLX weight directories, `.gguf` files, and one level of
nesting (`<repo>/4-bit/`, the shape several published repos use). Requests
route by the `model` field; with exactly one model any name works, matching
the single-model server's behaviour.

The budget is a **policy guardrail sized from file bytes, not a prediction of
RAM** — MLX mmaps weights, so three 27Bs "resident" measured 2.2 GB RSS here.
What it really prevents is page-cache thrash when alternating large models
(~25% throughput drift, measured). Details in `python/bwr/engine/pool.py`.

A web UI (chat + dashboard) is served at `/admin`, and a macOS menubar app
lives in `apps/bwr-mac`. Both are derived from
[oMLX](https://github.com/jundot/omlx) under Apache-2.0 — see
`python/bwr/server/webui/__init__.py` and `apps/bwr-mac/PROVENANCE.md` for
what was changed, and `vendor/` for the licence. oMLX-only features
(benchmark suites, model downloaders, ANE tuning) report `supported: false`
rather than pretending.

## License

Apache-2.0. Vendors llama.cpp (MIT) as a submodule and adapts portions of FreeToken
(Apache-2.0); see [NOTICE](NOTICE).
