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
- ~20 GB free disk per 27B-class model; 24 GB+ free RAM to serve one

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

# 4. Serve (default port 1919)
bwr serve -m qwen38-mlx-4bit --ctx-size 8192
# metal instead: bwr serve -m models/qwen3-30b.gguf --engine metal --n-seq-max 8

# 5. Check it answers
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' -d \
  '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```
bwr info     -m /path/to/model.gguf
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

## Optional features (all default off unless noted)

| Flag / knob | What | Notes |
|---|---|---|
| `--engine mlx` (default) / `metal` | Inference backend | Metal keeps the llama.cpp path below |
| `EngineConfig(speculative=True)` | N-gram speculative decoding | Both backends, greedy only; ~1.5x on repetitive text, parity on prose |
| `--draft-model GGUF` | Draft-model speculation (metal) | Attention targets only; refused on hybrids |
| `--prefix-cache` | Pin repeated prompt prefixes, skip re-prefill (metal) | Attention only; ~20x TTFT win measured |
| `EngineConfig(kv_cache="q8_0")` | Quantized KV cache (metal) | 2x KV headroom for long context; default `f16`; `q4_*` not recommended |
| `ModelParams(expert_weights="cpu")` | MoE expert weights on CPU (metal) | Residency knob only — no prefetch policy yet |

## License

Apache-2.0. Vendors llama.cpp (MIT) as a submodule and adapts portions of FreeToken
(Apache-2.0); see [NOTICE](NOTICE).
