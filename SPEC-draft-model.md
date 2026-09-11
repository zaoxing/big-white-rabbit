# Spec: draft-model speculation (T7)

Status: APPROVED (human: "do 1, 4 first", item 1).
Phase: implementation on branch `feat/draft-model` (worktree `../FreeToken-Mac-wt-draft`).

## Objective

Pair the 27B target with the 4B draft (both qwen35, byte-identical
tokenizers, verified 2026-09-10) so general prose -- where n-gram acceptance
is ~0.3 -- speculates from a real model. Expected economics (4B @42 tok/s
solo): each step costs ~1 target decode + (D+1) small draft decodes for up to
D+2 tokens.

## Design (decided)

- New `engine/draft.py::DraftEngine`: owns a draft Model's Context + Batch +
  per-request maps (`seq`, `pos`, `pending_row`). Greedy sampler chains only.
- The draft context mirrors the target token stream at IDENTICAL positions
  (prompt, then every target-accepted token). No n-gram tables involved.
- `prepare(rid, seq_id, prompt)`: full-clear slot, greedy chain, chunked
  prefill with logits on the last token only (pending logits = continuation).
- `propose(rid, max_d)`: sample pending logits, then autoregress (decode each
  sample with logits for the next). Ends with pending logits. Runs draft
  decodes immediately (caller = worker thread, single owner).
- `sync(rid, confirmed_pos, tail)`: partial-rm from `confirmed_pos` (loud on
  failure, same contract as target rewind), decode tail tokens (logits on
  last). Draft ctx gets `n_rs_seq = max_drafts + 1` like the target (hybrid
  drafts need it too).
- `release(rid)`: full clear + sampler reset + map drops (tolerant of unknown
  ids, mirroring `_retire`).
- Key invariant (why sync is always exactly one tail decode): `_verify_rows`
  breaks at the first mismatch, so `accepted[matched:]` is always exactly the
  single mismatch/bonus sample at `pos_base + 1 + matched`. Matched drafts
  already sit at the right draft cells.
- `MetalEngine(..., draft_model: Model | None)`: builds DraftEngine sized
  from the live target context (see geometry below), mutually
  exclusive with `speculative` (ValueError). No tables in draft mode.
  `_plan_drafts` branches on source; `_verify` returns
  `(outputs, accepted, matched)`; step syncs live requests only; `_retire`
  releases draft state. `EngineConfig.draft_max_drafts = 4`.
- Draft ctx geometry: `n_ctx` = target TOTAL `n_ctx` (per-stream room must
  cover the target per-seq span), `n_seq_max` = target's (seq_ids shared),
  `n_batch` = target effective `n_batch` (prompt chunking).
- `ftm serve --draft-model PATH`: second Model load (documented RAM cost),
  closed after uvicorn alongside the target. `AsyncEngine.stop` also closes
  the draft context (same Metal-teardown abort class as the main context).
- Correctness is vocab-independent: mismatched-vocab drafts just reject, so
  the 0.5B self-draft pair exercises the full loop fast in tests; the
  27B<-4B pair is for economics (manual bench).

## Testing strategy

- `tests/test_draft_model.py`: DraftEngine mechanics on FTM_TEST_MODEL
  (prepare/propose determinism, sync advance, release/re-prepare, max_d=0,
  unknown-rid KeyError) + integration with a second same-file Model
  (invariant on/off identical + reason, calls<=, drafted>0, mutual
  exclusion).
- Full suite green; 27B<-4B bench evidence (rep + nat, identical check).

## Task breakdown

- [x] T7a: `engine/draft.py` + mechanics tests. DONE, 6/6 first-try green.
- [x] T7b: MetalEngine integration + serve/launch/CLI + integration tests.
      DONE commit `f1ca0f2`, verifier APPROVE. 0.5B self-draft: invariant
      holds, calls<=, mutual exclusion enforced. 1 fix used (target snapshots
      in draft mode) + 1 lint (noqa marker).
- [x] T7c: 27B<-4B bench + gate. DONE with BLOCKER: repetitive 8.7->9.7 tok/s
      (rate 0.93, identical); natural DIVERGES deterministically (bisected to
      foreign-decodes + multi-row-target cross-context interference, upstream
      #20075 class, pin predates fix; 4B/8B unaffected). Gate added:
      `check_draft_target` refuses hybrid targets (27B refusal verified live);
      attention path stays enabled. Attempts on T7 correctness: 2 used.

L2 rules: worktree, <= 3 attempts per item, verifier after implementation,
no push without human approval.
