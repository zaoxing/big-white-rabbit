# Spec: plain-MLX engine track (M10)

Status: SCOPED 2026-09-10 (human: MTP dropped for Metal, plain performance endorsed).
Phase: awaiting implementation approval on branch `feat/mlx-engine`.

## Decision record

- MLX spike: plain 13.8 vs 9.8 tok/s steady (1.41x), deterministic,
  acceptable quality across quants. MTP on M1 Max ≈ parity (bf16 drafter,
  mlx-lm can't load qwen3_5_mtp) -- DROPPED, not wired, not revisited
  without new evidence. No speculation in the MLX track Phase 1.
- Canonical format question resolved separately: on GO, MLX artifacts become
  canonical and GGUFs are archived off-machine (no permanent duplication).

## Design (proposed)

- New `python/freetoken_mac/engine/mlx_engine.py::MLXEngine`: wraps
  `mlx_lm` load/stream in OUR engine interface (admit/step/stream shapes,
  request IDs, finish reasons, stop sequences) so the FastAPI routes,
  schemas, and tool-call parser survive untouched. `mlx_lm` becomes an
  optional dependency (`pip install freetoken-mac[mlx]`), same pattern as
  `[serve]`; import must fail with a clear message, never at module import.
- Out of scope Phase 1: batching parity (single-stream first; mlx-lm
  `BatchGenerator` evaluated in Phase 2), n-gram/MTP speculation,
  prefix cache, MoE placement, KV quant (unsupported on vision arch
  anyway -- hybrid text archs TBD by measurement).
- Config: `EngineConfig(engine: str = "metal")` selects backend;
  `build_app` constructs the matching engine. Default stays `metal`
  (no behavior change for existing users).

## Testing strategy

- Correctness: the 6-prompt battery, temp-0, MLX vs Metal engine text
  equality bar = same as spike (coherent; byte-equality NOT required
  across different quants -- compare within MLX 4-bit runs for
  determinism, and eyeball quality).
- Perf: same-prompt tok/s both backends on 27B (and 8B); ship bar is the
  spike's 1.3x holding inside our harness (not just mlx-lm CLI).
- Full suite green with BOTH backends selectable; Metal suite unchanged
  (MLX tests skip without the extra + weights, same pattern as
  FTM_TEST_MODEL/FTM_MOE_MODEL).

## Task breakdown

- [ ] M10a (M): scratch-venv parity probe (already done in spike -- fold
      numbers in, no-op).
- [x] M10b (L): `MLXEngine` shim + config plumbing + optional dep. DONE.
- [x] M10c (M): server wiring (`--engine` flag, `build_app`). DONE.
- [x] M10d (S): correctness + perf evidence in-harness, verifier, STATE.md.
      DONE commit `134f3d4`, verifier APPROVE (245 metal green + 7 MLX
      green). In-harness head-to-head (27B, 48 toks x6): 1/6 SAME, rest
      coherent quant flips; speedups 1.09-1.31x, mean 1.24x (CLI-level 1.41x
      shrinks through per-token stepping on both sides). Ship-bar note: 1.3x
      narrowly missed on the mean; min 1.09x, no regressions anywhere.

L2 rules: worktree, <= 3 attempts per item, verifier after implementation,
no push without human approval.

## Risks

| Risk | Mitigation |
|---|---|
| mlx-lm API churn | Pin versions; shim isolates us |
| M1 bf16 gaps in future models | 4-bit artifacts; re-verify per model |
| Perf inside our harness < CLI numbers | Ship bar measured in-harness, not CLI |
| Two backends to maintain | Metal stays default; MLX opt-in until proven |

## Open questions

1. Proceed with M10b implementation? (awaiting human approval)
2. Keep the 17GB spike artifacts as the dev weights, or re-fetch on demand?
   (Propose: keep until Phase 1 green, then decide canonical.)

## M12: n-gram speculation on MLX (approved)

N-gram drafts transfer verbatim (`NgramTable` is backend-free); only the
verify loop is new, because mlx-lm 0.31.3 has NO hybrid rewind: `ArraysCache`
defines no trim/snapshot/restore (`_BaseCache.is_trimmable()` is False), so
`trim_prompt_cache` returns 0 on our hybrid. Design from that fact:

- Full match: cache advanced exactly the accepted stream -- nothing to do
  (upstream's own "full-acceptance zero overhead" observation).
- Any mismatch: RECOMPUTE -- fresh cache, re-feed prompt + accepted
  (exact by construction, never lossy). Cost is O(prefix) per mismatch.
- Auto-fallback: per-request rolling acceptance disables drafting below
  0.5 after >= 8 drafted OR >= 3 recomputes (the second clause catches
  sparse-draft prose that never packs enough to trip the first). Bounds
  hostile-text cost to a few prefills; repetitive text never trips it.
  Fallback disables DRAFTING only -- the manual loop is sticky, because a
  stream generator would restart from the prompt (measured duplicate-prefix
  bug). Temp>0 or flag-off requests never enter the manual loop at all.
- Greedy only (temp <= 0 like Metal); stream path untouched when the flag
  is off or the request isn't greedy. Drafts capped by remaining budget.
- Manual loop owns its cache per spec request (`model.make_cache()`,
  chunked prefill at 512 mirroring mlx-lm, `mx.eval` discipline per the
  pinned source); EOG/stop/cap reuse `_feed`; counters
  `spec_drafted/spec_accepted/spec_recomputes` mirror the Metal names.
- Correctness invariant (same as T3): recompute is exact, so output is
  byte-identical at ANY acceptance rate -- including the all-wrong case.

## M11: MLX as default (approved: code + docs)

`EngineConfig.engine` and `ftm serve --engine` default to `mlx`; `mlx`/`mlx-lm`
moved from the `[mlx]` extra to core dependencies (extra removed); README
rewritten MLX-first. All pre-existing server/engine tests pin
`engine="metal"` explicitly (13 call sites + bare `build_app(model)`); the
new-default assertion lives in `test_speculative_verify.py`. Serve extras
(`fastapi` etc.) unchanged.
