# FreeToken-Mac

An edge-native MoE serving engine for Apple Silicon — the ideas behind
[FreeToken](https://github.com/FlashML-org/FreeToken) (bandwidth-adaptive MoE placement,
semantic-aware KV caching, an Anthropic/OpenAI-compatible API for coding agents) rebuilt on
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s Metal/ggml backend.

> **Status: Phase 0 (bring-up).** Not usable yet.

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
   │  MetalEngine         │  admission table + step loop over llama_batch
   │  (python/…/engine)   │  continuous batching
   └──────────┬───────────┘
              │ pybind11
   ┌──────────▼───────────┐
   │  csrc/               │  hand-written binding; buffer_control.cpp is the one
   │                      │  seam that reaches ggml backend-buffer APIs
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │  llama.cpp / ggml    │  submodule, Metal backend
   └──────────────────────┘
```

## Requirements

- Apple Silicon Mac (developed on M1 Max / 64 GB), macOS 14+
- Xcode command-line tools, CMake 3.21+
- Python 3.10+

## Build

```bash
git clone --recurse-submodules <this repo> && cd FreeToken-Mac
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## License

Apache-2.0. Vendors llama.cpp (MIT) as a submodule and adapts portions of FreeToken
(Apache-2.0); see [NOTICE](NOTICE).
