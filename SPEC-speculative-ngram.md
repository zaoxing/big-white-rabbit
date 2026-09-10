# Spec: greedy n-gram speculative decoding

Status: APPROVED 2026-09-10 (human: `spec_max_drafts` default 4 OK, save spec OK, start T1).
Phase: T1 in progress on branch `feat/spec-ngram-t1` (worktree `../FreeToken-Mac-wt-specngram`).

## Objective

Raise single-stream decode speed on the 27B (measured 8.2 tok/s; 8B does 36.5 tok/s)
with no new model file and no C++ changes, by drafting repetitive token runs from
an n-gram table and verifying them inside the single `llama_decode` the engine
already issues per step.

## Assumptions

1. Phase 1 is greedy only (`temp <= 0`): the binding exposes sampled tokens, not
   logits (`csrc/context.h`), and correct non-greedy speculation needs logits.
   Non-greedy requests silently use the current path.
2. Default off (`EngineConfig.speculative=False`): opt-in per deployment.
3. No `server/` changes in Phase 1: engine-level only, counters observable on
   the engine.
4. Correctness invariant: speculation is byte-identical to the non-spec path
   under greedy sampling.

## Why this design

- The binding already has every primitive: multi-token batches with per-row
  logits (`Batch.add(tok, pos, seq, want_logits)`), `sample_seq(seq, row)`,
  `accept_seq(seq, tok)` ("for externally chosen tokens, e.g. forced prefixes" --
  the replay-after-reject case), `memory_seq_rm(seq, p0, p1)` for KV rewind,
  `reset_seq_sampler`.
- At the first draft mismatch the sampled token IS the true next token, so every
  verify step makes progress >= 1 token; worst case degrades to ~today's speed.
- Draft-model path deferred: needs a second context plus a matching-vocab draft
  GGUF we do not have (27B is `qwen35`/248k vocab; local 8B/0.5B do not match).

## Algorithm (per generating request, greedy only)

1. Draft: look up trailing context in the per-request n-gram table (order 3),
   take up to `spec_max_drafts` tokens while `n_batch` budget allows.
2. Pack `[next_token + drafts]` at successive positions, logits on every row,
   into the step's single batch.
3. Verify: sample rows in order. Accepted prefix flows through the existing
   `_advance_row` logic (EOG / stop / `max_tokens` / context checks unchanged).
4. On first mismatch at position k: `reset_seq_sampler` + `accept_seq` replay of
   the accepted prefix, `memory_seq_rm(seq, k, -1)`, rewind `n_pos`; the
   mismatched sample becomes `next_token`.
5. Accepted tokens feed the n-gram table (prompt seeds it at admission).

## Config

- `EngineConfig.speculative: bool = False`, `spec_max_drafts: int = 4`
- Counters: `spec_drafted`, `spec_accepted` (+ acceptance-rate property)

## Testing strategy

- Invariant: same prompt, spec on/off -> identical `tokens_of` (small test model).
- Speedup: repetitive prompt -> fewer C++-counted `decode_calls` per token.
- Fallback: `temp > 0` byte-identical with flag on; stop/EOG/`max_tokens`
  exactness under speculation.
- Unit tests for the table (pure Python, no model).
- Full suite stays green:
  `FTM_TEST_MODEL=models/qwen2.5-0.5b-instruct-q4_k_m.gguf .venv/bin/python -m pytest tests/ -q`

## Task breakdown

- [x] T0: spec approved + saved here.
- [ ] T1 (S): `NgramTable` in `engine/` + unit tests. Verify: new test file green.
- [ ] T2 (M): refactor `_advance_row` into sample + `advance_token(req, token)`.
      Acceptance: suite green, zero behavior change.
- [x] T3 (M): draft + verify + rewind in `MetalEngine.step` (greedy-gated,
      budget-aware, counters). DONE commit `1aca139`, verifier APPROVE.
      Measured on 0.5B periodic prompt: 16 -> 4 decode calls, 12/12 accepted
      (rate 1.0), byte-identical. One fix during T3: n_pos off-by-one
      (pos_base+len, not +1+len) caught by llama position check.
- [x] Checkpoint: acceptance-rate numbers on small model before big models.
      0.5B periodic: 16 -> 4 calls, rate 1.0. 8B repetitive: 64 -> 14 calls,
      31.2 -> 48.6 tok/s, rate 1.0. 8B natural: 62 vs 64 calls, 39.1 -> 38.1
      tok/s (noise), rate 0.286, identical.
- [x] T4 (S): 8B/27B benchmark. DONE with BLOCKER (see below). 27B repetitive:
      64 -> 14 calls, 8.5 -> 14.1 tok/s (1.66x), rate 1.0, identical.
      27B natural + spec ABORTED pre-fix (inclusive-rewind fix + all-wrong
      regression test in `d6e7061`); post-fix the blocker is structural (hybrid
      rewind needs n_rs_seq), so 27B+spec now raises ValueError at construction.
- [x] T5a (M): rewind foundation. DONE commit `6acbe9c`, verifier APPROVE.
      `n_rs_seq` plumbed, `seq_rm` verdict surfaced (bool + raise at rewind
      site), capability gate (detect hybrid via list, confirm via readback).
      220 green. 27B validation: readback 5 as designed; repetitive 8.5->12.8
      tok/s (64->14 calls, rate 1.0); natural now COMPLETES identical
      (was: abort), 64 calls, drafted 3 / accepted 0. Deviation from proposal:
      bool-return + raise-at-callsite instead of blanket C++ throw (safer for
      the retire hot path).
- [x] T5b (L): native MTP verdict -- MEASURED via upstream `llama-server`
      (built from our pin in `build/upstream/`, no repo source changes), and
      DECLINED for engine integration. 4B: natural 51.5->59.6 tok/s (1.16x),
      repetitive 50.8->64.9 (1.28x, acceptance 1.0). 27B: repetitive
      11.0->14.9 (1.35x, acceptance 1.0) but natural 11.1->9.2 (0.83x LOSS,
      acceptance 0.47) -- reproduces upstream issue #23752 on this M1 Max.
      Our n-gram matches MTP where MTP wins (27B rep ~13-15 tok/s both) and
      never loses (natural parity). Conclusion: MTP wiring (load_mtp + MTP
      draft ctx + embd batches + nextn hooks + continuous-batching adaptation
      of upstream's contiguous-batch state machine) costs more than it can
      return on this hardware. Keep n-gram; revisit if Metal MTP overhead
      improves upstream. No OOM seen (duplicate-buffer watch item moot).
- [ ] T5 (S): verifier pass + `STATE.md` evidence (per item above).

L2 rules: worktree per attempt, <= 3 attempts per item, verifier sub-agent after
implementation, no push without human approval.

## Risks

| Risk | Mitigation |
|---|---|
| Low acceptance on prose -> net slower | Default off; T4 measures; per-request auto-disable is a follow-up |
| Sampler replay divergence | Greedy-only; `accept_seq` replay is order-exact for standard chains |
| Prefill budget contention | Drafts only from spare `n_batch`; FCFS prefill priority unchanged |

## Open questions (resolved)

1. `spec_max_drafts` default 4 -- APPROVED.
2. Save spec in repo -- APPROVED (this file).
