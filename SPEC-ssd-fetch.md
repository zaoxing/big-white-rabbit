# Spec: SSD fetch (slab swap)

Status: DRAFT 2026-09-10 — awaiting human approval to implement.
Phase: `qstar` fetch, after `hotlist` (T11a+b) + `byte-budget`.

## Objective

On hotlist miss, fetch the expert slab from GGUF on SSD into the Metal
shared buffer before the layer's MoE op. Turns the hotlist from measurement
into capacity: `K=32` (~1.4GB) serves like `K=128` at `0.76` hit, with `32ms`
miss cost vs `60ms` decode.

## Findings (ours)

- Hotlist hit `0.76 @ K=32 / 0.91 @ K=64` already proven on live traffic.
- Per-expert slab `~0.91MB`, `0.35ms` random `mmap` read (`2.6 GB/s`).
  `384 touches/token * 0.24 miss = 32ms` ceiling, `12ms` at `K=64`.
- Metal weight buffers are `MTLStorageModeShared` on UMA (CPU-writable,
  no copy) — confirmed via `ggml-metal` buffer creation path inspection.
  Slab offset = `tensor_offset + expert_idx * slab_stride` (row-major,
  `128` experts contiguous).

## Design (proposed)

- **C++**: `Context` gains `fetch_expert(layer, expert_idx)` — `mmap`'d
  GGUF `pread` `0.91MB` into `Metal` buffer at computed offset,
  `mtlBufferDidModifyRange` + `ggml_backend_synchronize` if needed.
  `Model` exposes `expert_slab_offset(layer, expert_idx) -> (offset, bytes)`.

- **Python**: `ExpertHotlist` on miss calls `ctx.fetch_expert(layer, eid)`
  before the layer's MoE. `ssd_fetch` flag enables the path; `byte-budget`
  `K` already auto-fits. `health` adds `fetches` + `fetch_ms`.

- **Fallback**: miss fetch failure degrades to `CPU` expert (already
  measured `11x` slower) rather than abort — like `ds4`'s `CPU` non-routed
  path.

- **Gate**: `record_experts + ssd_hotlist` (same as hotlist), single-seq
  only. `n_seq_max>1` is later.

## Testing

- Unit: offset math `layer*expert -> byte` vs GGUF tensor table, `fetch`
  idempotent, `hit->no fetch`, `miss->fetch + resident`.
- Integration (30B): `K=32` run identical tokens to `K=128` (fetch restores
  exact weights), hit `0.76`, fetches `~92/token`.

## Risks

- `Metal` buffer writability assumed `Shared` — verify in `T11c` spike
  (write 1 byte, read back). If `Private`, need `blit` staging.
- `GGUF` `mmap` eviction under pressure — `ds4` warns `more cache helps
  only while working set fits`; same here.

## Open questions

1. Proceed with `T11c` fetch spike (single slab `memcpy` + readback)?
