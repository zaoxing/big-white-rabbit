# Spec: KV snapshot (ds4 /save) — disk-persisted prompt cache

Status: DRAFT 2026-09-10 — awaiting human approval to implement.
Phase: `serve-moe` extension, after `hotlist` (T11a/b).

## Objective

Resume a prior conversation without re-prefilling the prompt: save the
`KV` + `recurrent` state for a `seq_id` to disk, restore it later in
`O(ms)` instead of `O(seconds)` prefill. `ds4` does `~/.ds4/kvcache`
with `/save` `/switch` `/del` `/strip`. Good for agent loops where the
same system prompt + history repeats.

## Findings (ours + ds4)

- Ours: `30B` prefill `235s` for `21k` tokens; `Metal` `KV` is `300MB`
  at `2k` ctx, `~1.2GB` at `8k` with `n_seq_max=8`. `ds4` snapshots
  contain `KV` + `recurrent` + `graph` state; ours would need `KV` +
  `recurrent` + `sampler` + `n_pos`.
- `ds4` borrow: `kvcache` dir, `save`/`switch`/`del`/`strip` (keep text,
  drop `KV`), compatible snapshots avoid rebuild, `strip` saves text only.

## Design (proposed, minimal)

- `engine/kv_snapshot.py::KVSnapStore(dir)` — `save(engine, rid, name)`,
  `load(engine, name) -> rid`, `list()`, `del()`, `strip()`. File per
  snapshot: `msgpack` of `prompt tokens + output tokens + n_pos +
  KV cells` via `llama_memory_seq_rm`/`cp` primitives (no new `C++`).

- **T12a — snapshot primitive**: `Context` gains `kv_snapshot(seq_id)` and
  `kv_restore(seq_id, snapshot)` via `llama_memory` helpers (already
  exposed as `memory_seq_*`). No new `C++`.

- **T12b — store + CLI**: `ftm save` / `ftm load` subcommands, `Engine`
  integration, `health` snapshot count.

- **Gate**: `n_seq_max==1` like `record_experts`; `KV` size check before
  restore (refuse if `n_ctx` too small).

## Testing

- Unit: `save` then `load` identical `KV` (drain, save, new engine, load,
  generate — tokens identical).
- Integration: `30B` `21k` prompt `save`/`load` wall `235s -> <0.5s`.

## Open questions

1. Disk budget? `ds4` caps at `8GB`; propose `4GB` default.
2. Proceed with `T12a` primitive?
