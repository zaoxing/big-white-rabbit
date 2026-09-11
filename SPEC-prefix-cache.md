# Spec: prefix-cache TTFT (T8)

Status: APPROVED 2026-09-10 (human: auto-pin, save spec OK).
Phase: implementation on branch `feat/prefix-cache` (worktree `../FreeToken-Mac-wt-prefix`).

## Objective

Kill repeated-prefill cost: a 21k-token prefill costs ~235s on the 27B, and
agent loops resend the same system prompt every turn. Reuse its KV.

## Design: auto-pin + full-copy fork (no geometry change)

True KV *sharing* needs `kv_unified=true` (engine-wide geometry change --
deferred to T9). Phase 1 COPIES: full-range `seq_cp` is legal on split KV,
so a pinned slot's cells move to a new sequence with plain existing ops.

- **Pin creation (automatic)**: when a request with a fully-prefilled prompt
  of length >= `prefix_cache_min_tokens` retires, its slot converts to a pin
  instead of freeing: key = full prompt token tuple, value = seq_id. Exact-
  match dedupe (refresh LRU position, keep old slot); over capacity ->
  evict oldest (full rm + slot back to the free pool).
- **Why retire-time is sound**: the pin key is the PROMPT, but the slot may
  also hold generation cells. Forks only ever use a shared run K with
  K <= len(prompt tokens), then truncate everything from K-1 -- generation
  cells always fall in the truncated region. And speculative rewinds only
  ever truncate suffixes at/after the prompt end, so prompt cells are intact
  for any fully-prefilled request. The one hard guard: pin ONLY fully
  prefilled requests (`n_prefilled >= len(prompt)`); a cancelled mid-prefill
  request must never pin.
- **Fork (admission)**: longest pin with common-prefix length K where
  savings = K - 1 >= `prefix_cache_min_tokens` (the -1 is the backoff below).
  Full-copy pin -> new slot, suffix-rm from K - 1, set
  `n_prefilled = n_pos = K - 1`.
- **Backoff by one (load-bearing)**: the last covered token is RE-decoded by
  the normal chunked-prefill path so its logits row exists for sampling the
  continuation. Re-decoding an already-resident cell would abort (llama
  requires strictly consecutive writes), hence the slot is truncated one
  short. Costs exactly 1 prefill token per fork; negligible against min_tokens.
- **Slot pressure**: pins hold seq_ids outside the free pool. Admission with
  no free slot evicts non-matched pins LRU-first; if the MATCHED pin itself
  must go, the fork is dropped and admission proceeds plain (never crash,
  never strand). Requests always outrank pins; empty pool + no pins ->
  `SeqIdExhausted` exactly as today.
- **Untouched**: sampler chains (fresh per request), stop/EOG/max-tokens
  (generation-side), n-gram tables (seeded from the full prompt), draft
  prepare (cross-model, full prefill), read window, cancel path (fork state
  is ordinary request state by then).
- **Config**: `prefix_cache: bool = False`, `prefix_cache_pins: int = 2`,
  `prefix_cache_min_tokens: int = 256` (>= 1). Counters: `prefix_cache_hits`,
  `prefix_cache_tokens_saved`. Serve flag `--prefix-cache`.

## Testing strategy

- Correctness: pinned vs unpinned drains byte-identical (tokens + reason)
  on shared-prefix/divergent-suffix prompts, exact-match reruns, stop-seq
  prompts, and cancel-mid-flight.
- Efficacy: prefill `decode_calls` collapse + `hits`/`tokens_saved` counters.
- Lifecycle: min_tokens gate (short shared run -> no fork), pin bound under
  rotation, n_seq_max=1 eviction path, fork-then-cancel hygiene.
- Suite green (zero pins = today's paths exactly).

## Task breakdown

- [ ] T8a (M): pin store + admission fork + retire/evict + tests + serve flag.
- [ ] T8b (S): TTFT bench evidence, verifier, STATE.md.

L2 rules: worktree, <= 3 attempts per item, verifier after implementation,
no push without human approval.
