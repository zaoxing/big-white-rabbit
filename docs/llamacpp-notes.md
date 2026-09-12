# llama.cpp integration notes

Hard-won facts about the vendored llama.cpp (submodule pinned at `050dde50`, Sept 2026)
that are not obvious from its headers, plus the open issues later phases inherit. Written
down because every one of these cost real debugging time.

## llama.cpp signals bad input with `abort()`, not exceptions

`GGML_ASSERT` expands to `GGML_ABORT` — it is **not** `NDEBUG`-gated
(`ggml/include/ggml.h:288`). A bad argument does not throw; it kills the process. Under
the planned single-process server (see the Big White Rabbit design) that means one malformed
request takes down the API server along with the engine.

**Rule: every Python-reachable binding bounds-checks its own arguments and raises.** Three
separate reachable aborts were found during Phase 1, each by a wider sweep than the last:

| Entry point | Trigger | Guard |
|---|---|---|
| `llama_decode` | `n_tokens > n_batch` | `Context::decode`, `decode_seq0` |
| `llama_memory_seq_cp` | partial range across different seq_ids without `kv_unified` | `Context::memory_seq_cp` |
| `llama_memory_seq_rm` / `_keep` | `seq_id` out of `[0, n_seq_max)` (`-1` is a legal wildcard for `_rm`) | `validate_seq_id` |
| `llama_init_from_model` | effective `n_batch < n_seq_max` (asserts inside `output_reserve`) | `Context` constructor, predicts the clamps |
| `llama_sampler_sample` | no logits exist yet (`llama-sampler.cpp:940`) | `sample_last`, `sample_seq` |

When adding a binding, sweep for **both** argument-shaped asserts and *state-dependent*
ones (sampling before a decode, ops after a failed decode). The last row above was the
subtle one: it is reachable during `MetalEngine`'s own chunked prefill, because
intermediate chunks decode with zero logits rows.

## Effective geometry is not requested geometry

Three separate clamps, all silent:

- **`n_ctx` is rounded UP** (KV padding).
- **`n_batch` is clamped DOWN** to the *requested* `n_ctx` under causal attention.
  Requesting `n_ctx=32, n_batch=512` yields `n_ctx=256, n_batch=32`.
- **`n_ctx_seq` is the per-sequence capacity.** With `kv_unified=false`,
  `n_ctx_seq = n_ctx / n_seq_max`, and the *reported* `n_ctx` is then inflated back to
  `n_ctx_seq * n_seq_max`. So `EngineConfig(n_ctx=512, n_seq_max=2)` reports
  `ctx.n_ctx == 512` while each sequence really holds 256.

Always budget against `ctx.n_batch` / `ctx.n_ctx_seq`, never against what you asked for.
Budgeting a single request against `ctx.n_ctx` overstates its room by a factor of
`n_seq_max` and admits prompts that only fail later, mid-decode, with a KV-slot error.

## `kv_unified` gates partial-range `seq_cp` — this matters for the semantic-anchor cache

`llama_memory_seq_cp` with a partial range is legal **only** when the context is created
with `kv_unified=true`. Otherwise `n_stream == n_seq_max`, the copy takes llama.cpp's
cross-stream path, and `llama-kv-cache.cpp:506` asserts `is_full`.

This is load-bearing for the planned semantic-anchor / radix KV cache, whose central
primitive is "copy prefix `[0, anchor_pos)` onto a new sequence". That design **requires
`kv_unified=true`**. Verified working: a partial copy of the first 3 of 5 cells reaches the
same greedy token as a full prefill, while a no-copy control diverges.

## Renamed / removed APIs (vs. most online examples)

- `llama_kv_cache_seq_*` → **`llama_memory_seq_*`**, reached via `llama_get_memory(ctx)`.
- `llama_model_params.use_mmap` / `use_mlock` → **`load_mode`** enum (which also adds
  `DIRECT_IO`), plus a separate **`lazy_mode`** for on-demand row reads.
- `no_alloc` is llama.cpp's *memory-fit pass*: it maps nothing and only simulates
  allocations, which is exactly what a placement/sizing solver wants — but it asserts
  against mmap, so a sizing load must force `load_mode=NONE` and `lazy_mode=OFF`.

## Build

- **Link llama/ggml statically** (`BUILD_SHARED_LIBS=OFF`). A shared build leaves the
  extension chasing `@rpath/libllama.0.dylib`, which the wheel does not ship; with conda
  on `PATH` that rpath resolves into the conda prefix and fails to load.
- **Keep `GGML_METAL_EMBED_LIBRARY=ON`** (the Apple default). It embeds shader *source* and
  compiles at runtime, so no Metal offline toolchain is needed. The `OFF` path shells out
  to `xcrun metal` at build time and fails on machines without that separately-downloaded
  component.

## GPU resources must be released explicitly, not left to the collector

ggml frees the Metal device from a **C++ static destructor** at process exit
(`__cxa_finalize` → `ggml_metal_device_free`) and asserts its residency sets are empty
(`ggml-metal-device.m:1021`). Anything still holding Metal buffers at that moment
`abort()`s the process with SIGABRT — *after* a clean shutdown, so every request
succeeds, the logs look healthy, and the exit code still says crash.

Python's garbage collector is not a sufficient answer. A FastAPI app keeps the model and
engine alive through cycles in its route closures, and those cycles can outlive the
interpreter's last collection; `del app; gc.collect()` did **not** clear it. Two things
therefore have explicit `close()` methods, and both must be called on shutdown:

1. `Context.close()` — the KV cache and compute buffers. Called by
   `AsyncEngine.stop()`, which the app's lifespan runs.
2. `Model.close()` — the GPU-resident **weights**, which is what the residency sets
   actually hold. Called by `serve()` after uvicorn returns.

Order matters: close every context before its model, since contexts keep only a
`shared_ptr` to the wrapper, not to the `llama_model` behind it. After `close()` every
method on either handle raises rather than dereferencing a freed pointer.

The symptom to recognise: a suite or server that passes everything and exits **134**.

## Open issue carried forward

**`sample_last()` after a failed decode is safe only because of belt 1.** `decode_raw` now
clears the logits mask when `llama_decode` returns non-zero, so the first guard rejects
sampling. That matters because the second guard —
`llama_get_logits_ith(ctx, -1) == nullptr` — is **build-dependent**: that function returns
`nullptr` only under `NDEBUG` and calls `GGML_ABORT("fatal error")`
(`llama-context.cpp:900`) in a Debug build. The shipped build is Release/NDEBUG, so there
is no reachable abort today, but **do not remove the mask-clearing on the assumption that
the null-check covers it** — a `--config Debug` build would then crash.
