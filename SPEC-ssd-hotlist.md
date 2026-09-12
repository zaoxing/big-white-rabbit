# Spec: SSD expert hotlist (qstar, ds4-inspired)

Status: DRAFT 2026-09-10 — awaiting human approval to implement.
Phase: `qstar` after `residency` (T10b merged as `expert_activations`).

## Objective

Bounded expert cache for MoE models larger than RAM, borrowing `ds4`'s
`--ssd-streaming` design: keep a hot set of routed experts resident,
fetch misses from GGUF on demand. Trades speed for capacity; does not
remove memory for non-routed weights, activations, KV. Single-process,
no distribution.

## Findings (ours + ds4)

- Ours (30B, 48 layers x 128 experts, `~0.91MB`/slab, `~116MB`/tensor):
  `top8 mass 0.20` too diffuse for frequency prediction, but per-layer
  `LRU K=32 -> 0.76, K=64 -> 0.91` — plain LRU wins. Per-slab `mmap` read
  `0.35ms` (`2.6 GB/s` random), so `384 touches/token * 0.24 miss @ K=32
  = 32ms/token -> ~31 tok/s` ceiling (viable). `ds4` SSD doc confirms
  `generation more sensitive than prefill`, `non-routed weights always
  resident`, byte-budget auto-fit (`--ssd-streaming-cache-experts 32GB`).

- `ds4` borrow list (ranked):
  1. **Hotlist + byte budget** (`ds4_ssd.c`, `ds4_streaming_hotlist.inc`):
     `LRU(t)` over routed experts, `K/layer` or `bytes` budget, `full-layers`
     option, auto-fit to remaining RAM.
  2. **Resident non-routed invariant** (`attention/shared`) — we already have
     via `expert_weights` placement, just never evict them.
  3. **Opportunistic sampling** for `temp>0` spec — accept greedy-matching
     drafts directly, resume sampling on mismatch (ours skips `temp>0`).
  4. Process: `speed-bench/*.svg` recorded baselines (cheap to add).

- What NOT to borrow: bespoke `C/Metal/CUDA/ROCm` no-GGML core — ideas,
  not code. `ds4` supports only `DeepSeek/GLM` narrowly; we stay general.

## Design (proposed, incremental)

- **T11a — hotlist tracker (no I/O yet)**: `engine/hotlist.py::ExpertHotlist`
  `per-layer LRU(K)` driven by `expert_activations()` after each decode.
  `update(frames) -> {misses, hits}` + `hit_rate()`. No buffer mutation —
  measures the residency that *would* exist, validates `K` vs hit rate on
  live traffic before touching memory. `MetalEngine` gains optional
  `hotlist: ExpertHotlist | None` (default `None`, zero overhead when off).

- **T11b — SSD fetch (later)**: `Context` gains per-expert buffer
  swap (`Metal` shared buffers are CPU-writable on UMA — `memcpy` slab from
  `mmap`'d GGUF into buffer; offset = `tensor_offset + expert_idx * slab`).
  Miss path blocks decode (like `ds4`); `T11a` hit rate tells us how often.

- **Config**: `EngineConfig(ssd_hotlist: bool=False, ssd_hotlist_k: int=32)`
  or byte budget variant — bikeshed after `T11a` hit-rate data.

- **Gate**: `record_experts` must be on + `n_seq_max==1` (same guard as
  `T10b`); multi-seq hotlist is later.

## Testing strategy

- Unit: `LRU(K)` eviction, `byte->K` conversion, `hit_rate` empty.
- Integration (30B, `BWR_MOE_MODEL`-gated): deterministic routing ->
  deterministic hotlist, `K=64` hit `>0.85` on 100-token run, miss count
  matches `L1` reuse distance.

## Open questions

1. `K/layer` or `bytes` API first? `K` is simpler to test; `bytes` auto-fit
   matches `ds4` UX — propose `K` now, `bytes` as `T11b` sugar.
2. Proceed with `T11a` tracker only (no I/O, no risk to generation)?
