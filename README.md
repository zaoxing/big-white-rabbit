# FreeToken-Mac

An edge-native MoE serving engine for Apple Silicon — the ideas behind
[FreeToken](https://github.com/FlashML-org/FreeToken) (bandwidth-adaptive MoE placement,
semantic-aware KV caching, an Anthropic/OpenAI-compatible API for coding agents) rebuilt on
Apple's [MLX](https://github.com/ml-explore/mlx) framework, with the original
[llama.cpp](https://github.com/ggml-org/llama.cpp) Metal/ggml backend still available.

> **Status: serving.** OpenAI- *and* Anthropic-compatible APIs with continuous
> batching and tool calling. MLX is the default backend (measured ~1.24x the
> Metal path in-harness, single-stream plain decode); `--engine metal` keeps
> the llama.cpp path with speculation, prefix caching, and MoE placement.
> No MoE placement policy or semantic KV caching yet.

## Why this is a rewrite, not a port

FreeToken is CUDA-only by construction: `nvcc`-JIT'd `.cu` kernels, `torch.cuda` throughout,
and CUDA-graph capture baked directly into its attention/MoE backend base classes. There is no
vendor abstraction to plug a Metal backend into.

More importantly, FreeToken's central idea does not transplant. Its `q*` policy solves a
**capacity + PCIe-transfer-cost** problem: VRAM is a small pool physically separate from host
RAM, so experts are streamed across a bus. Apple Silicon has **one physical DRAM pool** shared
by CPU and GPU — there is no transfer to amortize. The binding constraints become:

1. whether total resident memory (weights + KV + activations) fits the RAM budget without OS
   paging pressure or Metal working-set stalls — the primary driver, and
2. which physical engine (GPU shader cores vs. CPU cores) computes each expert matmul — a
   softer, mostly tie-break factor.

So `q*-Mac` is a re-derived cost model, not translated code. What *does* carry over cleanly is
FreeToken's hardware-agnostic Python: the FastAPI route/schema layer, the tokenizer, GGUF
metadata reading, and the radix prefix-cache bookkeeping behind semantic-anchor caching.

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

- Apple Silicon Mac (developed on M1 Max / 64 GB), macOS 14+
- Xcode command-line tools, CMake 3.21+ (metal path only)
- Python 3.10+

## Build

```bash
git clone --recurse-submodules <this repo> && cd FreeToken-Mac
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[serve]"
```

(`mlx`/`mlx-lm` ship as core dependencies — no extra is needed for the default backend.)

## Run

```bash
# Serve an OpenAI-compatible API (default port 1919) off MLX weights
ftm serve -m /path/to/model-mlx --ctx-size 8192

# Or the llama.cpp Metal backend off a GGUF
ftm serve -m /path/to/model.gguf --engine metal --ctx-size 8192 --n-seq-max 8

# Generate straight from the CLI (llama.cpp path)
ftm generate -m /path/to/model.gguf -p "Hello" -n 64
ftm info     -m /path/to/model.gguf
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

## License

Apache-2.0. Vendors llama.cpp (MIT) as a submodule and adapts portions of FreeToken
(Apache-2.0); see [NOTICE](NOTICE).
