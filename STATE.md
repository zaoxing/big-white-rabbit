# Loop State — FreeToken-Mac

Last run: 2026-09-10 (M12 MLX speculation, human-approved) — verifier APPROVE, branch `feat/mlx-spec` (worktree), no push
- M12 DONE: n-gram verify on MLX (recompute-on-mismatch, sticky loop, fallback, 14 green). 27B: rep 1.15x identical, prose 0.75x identical (default-off contains it). 3 behavior fixes along the way (piece-sig, base-emit, sticky-loop); attempt budget exhausted, tuning deferred.. kv-cache-quant disregarded (deleted).
- Default MLX serve VERIFIED live: /health 200 + chat completion 200 (32 toks @ 9.7 tok/s served). Fixed two default-path gaps found live: health geometry + chat renderer duck-typing (both 500'd before). Note: thinking trace leaks into output (model default, needs enable_thinking=False passthrough — follow-up, not filed).
- A/B served greedy-64 (same prompts): 27B nat 11.0 vs 9.7 (0.88x), rep 11.1 vs 8.9 (0.80x); 30B nat 62.7 vs 48.2 (0.77x), rep 62.5 vs 45.3 (0.72x). Gap widens where decodes are fast (Python per-step overhead). MLX backend at 1.24x ours would sit above upstream on dense.
- Fix: M11 merge left a duplicate EngineConfig in launch.serve that discarded --engine (metal serve impossible); found live during A/B, fixed + verified by the A/B runs themselves. README quickstart added (Mac deps).
- M11 DONE: EngineConfig/CLI default mlx, mlx core deps, README MLX-first, 13 test sites pinned metal + new-default assert (247+5 green, 252 with weights).
- M10 DONE: MLXEngine + `--engine` + build_app/launch wiring (245 metal + 7 MLX green). In-harness: 1.24x mean (1.09–1.31x), coherent, deterministic. Ship call open: bar was 1.3x.
- MLX spike DONE (scratch venv mlx 0.32.2, orcarouter 27B 4-bit + mtp drafter): plain steady-state 13.8 tok/s vs ours 9.8 (1.41x); deterministic; 2/6 byte-identical to our Q4_K_M, rest coherent early-flips (different quant; one Chinese code-mix smell). MTP gate FAILED on M1 Max: mlx-lm can't load qwen3_5_mtp; via mlx-vlm it runs (60% accept) but totals ≈ plain (bf16 drafter, no bf16 GPU on M1 — warned upstream). Rewrite unjustified on 1.4x alone; unblockers: fp16-converted drafter retest, or M1-relevant upstream fixes.
- T9 DONE: `ModelParams(expert_weights)` metal/cpu + tests (247 + 4 green). 30B A/B: identical outputs, metal 0.2s vs cpu 2.2s (11x gap = residency target). Next: `residency` (activation tracking) → `qstar`.
- moe-bringup DONE: `models/Qwen3-30B-A3B-Q4_K_M.gguf` (17.28 GiB, qwen3moe, 30.53B total/~3B active, 48 layers, vocab 151936, n_ctx_train 40960). `ftm generate` (ctx 2048): 32 tokens @ **35.2 tok/s**, coherent self-ID. (gpt-oss-120B skipped: every GGUF quant ~63GB, cannot load resident on 64GB.)
- Next per capability map: `placement` (tensor_buft_overrides) → `residency` (activation tracking) → `qstar` (validated under simulated constraint) → `serve-moe`.
- T8 DONE: auto-pin + full-copy fork + `--prefix-cache` (233 green). 0.5B 2k-prefix TTFT 0.81s→0.04s (20x), 2000 saved, identical. Hybrid fork DIVERGES nondeterministically (4B+27B measured) → construction gate (attention exact, triple-fork proven); snapshots knob dropped as unneeded.
- T7 DONE (branch `feat/draft-model`, verifier APPROVE, 232 green): DraftEngine + integration + serve flag. 27B BLOCKER: deterministic divergence on natural (bisected to cross-context interference, upstream #20075 class) → `check_draft_target` refuses hybrid targets (live-verified); attention path enabled. N-gram path unaffected (no foreign decodes).
- T6 DONE (+long-ctx follow-up): q8_0 6/6 identical incl. 79-token long-ctx run, but decode SLOWER on Metal (21k ctx: f16 9.3/7.9 vs q8_0 5.3, thermals accounted) — capacity lever only, adopt solely anti-OOM; default f16. q4 diverges + slower, not adopted. 228 green, verifier APPROVE.

## Speculative n-gram decoding (L2, SPEC-speculative-ngram.md APPROVED 2026-09-10)

- T1 DONE: `NgramTable` (`engine/ngram.py`) + `tests/test_speculative_ngram.py` (10 passed) + exports. Worktree `../FreeToken-Mac-wt-specngram`, additive-only edits to existing files (export lines).
- T2 DONE: `_advance_row` → sample + `_advance_token(req, token)` split (`metal_engine.py`, 12+/1-, docstring only addition). Commit `9315de8`. Verifier: APPROVE (pure split, no other callers, suite re-ran 210 passed in 16.29s, no push).
- T3 DONE: draft + verify + rewind in `MetalEngine.step` (greedy-gated, spare-rows-only, counters `spec_drafted`/`spec_accepted` + rate). Commit `1aca139`. Evidence (worktree, 0.5B periodic prompt): full suite `215 passed in ~17s` (verifier re-ran 215 in 16.93s); 16→4 decode calls, 12/12 accepted, byte-identical incl. finish reason. 1 fix used (n_pos off-by-one, caught by llama consecutive-position check). Verifier: APPROVE, no push.
- T4 DONE with BLOCKER: 8B repetitive 64→14 calls, 31.2→48.6 tok/s (1.47x), rate 1.0, identical; 8B natural 62 vs 64 calls, parity tok/s, rate 0.286, identical; 27B repetitive 64→14 calls, 8.5→14.1 tok/s (1.66x), identical. 27B natural + spec aborted: two fixes in `d6e7061` (inclusive rewind bound; `_WrongTable` all-mismatch regression test) then root cause found STRUCTURAL — binding sets n_rs_seq=0 and ignores seq_rm's bool, so partial rewinds silently no-op on hybrid/recurrent memory (qwen35) and abort the next decode; existing code only ever full-clears, which is why it never surfaced. Guard added: 27B+spec raises ValueError at construction (fail loud). Suite `218 passed` (verifier re-ran, APPROVE). Attempts on speculation correctness: 3 used (n_pos, rewind bound, guard) — no further code without human direction.
- T5a DONE (commit `6acbe9c`, verifier APPROVE, 220 green): n_rs_seq plumbing + surfaced seq_rm verdict + capability gate (bool-return + raise-at-callsite, not blanket throw). 27B validation: readback 5; repetitive 8.5→12.8 tok/s (64→14 calls, rate 1.0, identical); natural COMPLETES identical (was abort), drafted 3/accepted 0. Blocker lifted.
- T5b VERDICT (measured, no repo source changes): native MTP via upstream llama-server from our pin — 4B natural 1.16x / rep 1.28x; 27B rep 1.35x (14.9 tok/s) but natural 0.83x LOSS (9.2 vs 11.1, acc 0.47), reproducing #23752 on this M1 Max. N-gram matches MTP where it wins and never loses → MTP engine integration DECLINED (cost >> return on this hardware). Servers stopped. T5a remains the shippable piece.
- Draft model acquired 2026-09-10: `models/Qwen3.8-4B-Q4_K_M.gguf` (2.6G, `empero-ai/Qwen3.8-4B-Distill-GGUF`, distill of Qwen3.8 into Qwen3.5-4B arch). Verified: arch qwen35, vocab 248320, n_ctx_train 262144 (all match 27B); 8/8 probe tokenizations byte-identical (EN/code/digits/Chinese/tool-call/agent text); EOG agreement; solo 42.4 tok/s. Bonus find: 27B GGUF carries a native MTP head (`blk.64.nextn.*`, 65 blocks). NOT downloaded: DSpark/DFlash drafters (safetensors, need SGLang/vLLM + target feature taps + CUDA — unusable here). Remaining before use: engine draft-model path (not implemented) + T5 n_rs_seq (draft is also qwen35 hybrid).
- Evidence (worktree, `FTM_TEST_MODEL=<abs path to 0.5b gguf>`, finder-shim): full suite `210 passed in 16.69s` (200 existing + 10 new). Main-repo baseline same day: `200 passed in 16.80s`. 1 test-expectation fix during T1 (order-1 max_tokens=2), no source bug. Verifier: APPROVE.
- Next: T2 (`_advance_row` → sample + `advance_token` refactor, zero-behavior-change) on human go-ahead. No push (awaiting approval per constraints).

## Perf triage L1 (2026-09-10, report-only — no source edits)

- Baseline (measured, same command as prior runs): `FTM_TEST_MODEL=models/qwen2.5-0.5b-instruct-q4_k_m.gguf .venv/bin/python -m pytest tests/ -q --durations=10` → `200 passed in 17.47s` (prior: 18.29s, within noise).
- Slowest: `test_both_surfaces_serve_correctly_after_many_requests` 1.63s, `test_per_decode_scan_cost_is_independent_of_requests_served` 0.71s (both integration/model-bound, expected).
- Skill sections N/A for this repo (no frontend, DB, bundle, images, React): LCP/INP/CLS, bundle-size, image-optimization, re-render checks do not apply to a FastAPI+Metal inference server.
- Candidates examined, no fix proposed (L1):
  - Double tokenization per request (`common.count_tokens:112` then `metal_engine.add_request:123`) — only credible candidate; needs L2 measurement (tokenize 8k prompt, 1x vs 2x) before any change.
  - `pop(0)` / `in`-scan on `_free_seq_ids`, double-scan of `_states` in `step()` — bounded by `n_seq_max` (default 8), negligible vs `llama_decode` cost; optimizing now would be premature per skill. Already guarded by `test_per_decode_scan_cost_is_independent_of_requests_served`.
  - `StopSequenceFilter` buffer is bounded (trimmed to partial-prefix `hold` each `push`), not unbounded — no leak.
  - AsyncEngine empty-plan busy-spin analyzed unreachable (admission guarantees `room > remaining` while prefilling); no evidence, no change.
- Skill gap: `references/performance-checklist.md` (See Also link) does not exist — only `SKILL.md` in skill dir.
- Next: nothing unless human enables L2 with a specific symptom + budget (e.g. p95 latency target).

## High Priority (loop is acting or waiting on human)

1. **PROJECT FINISHED — all loopable items done, verified 2026-09-10**
   - Why: `e4f2075` foundry/fleet/memory + `gate.yaml` already committed (STATE's prior "review + commit" note was stale — `git log` shows `e4f2075`, `git status` clean). Final `FTM_TEST_MODEL=... pytest tests/ -q` → `200 passed in 18.29s`. Doctor `100/100 L3 healthy`, `gate=true`. Original polish loop 4/4 ALL GREEN (`.loop/state.md:47`), L2 TDD 7 cycles, whole-system parallel, fleet 8 agents — all committed (`de30ebf`, `096d555`, `e6bdba4`, `e8f9e62`, `e4f2075`).
   - Next: Nothing required. Optional only on explicit opt-in: `.github/workflows`, `harness run --goal "Verify harness wiring"`. Do NOT push (no remote, `loop-constraints.md:8`).
   - Evidence: `200 passed`, doctor `exit 0`, `harness-foundry validate` → `Stack is valid`.

## Watch List — remaining after whole-system (2 items, low risk)

- **Launch serve teardown** (`launch.py serve()` hand-rolled copy in `tests/test_slot_release_on_abnormal_exit.py:88`) — out-of-scope, needs process-level harness (no TDD without `UF_HIDDEN` hazard).
- **Deferred non-loopable** (`.loop/backlog.md:42` 4 unreferenced defs in `anthropic_schemas.py`, `metal_engine.py:167` list-comp discard, `UF_HIDDEN` env) — needs-human / no-machine-verify.
- **Budget:** `loop-budget.md:8` 100k/day, 2 runs/day, realistic `23k/run`. Spend today `~52k` (8 cycles), under 80%.
- **Scaffold added:** `docs/safety.md:1`, `.github/PULL_REQUEST_TEMPLATE.md:1`, `.github/ISSUE_TEMPLATE.md:1`, `patterns/registry.yaml:1`, plus now `.foundry/`, `memory-tiers.md`, `memory-budget.md`, `fleet-registry.md`, `fleet-inbox.md`, `gate.yaml` — doctor `p50` now only workflows / `harness run` (optional).

## Recent Noise (ignored this run)

- `.loop/evidence/` 9 logs already GREEN, `criteria.sha256:1` unchanged (`efa1129...`).
- `tests/test_phase0_smoke.py` → `9 skipped` without model — expected, not actionable.
- `.loop/polish-seen.md` findings already triaged into `.loop/backlog.md`; no new commits on `main` since `097b84f` besides uncommitted diff.
- No open PRs/issues to triage (no remote).

## Loop Mode

- L2 **ENABLED** 2026-09-10, whole-system completed via parallel agents (TDD RED→GREEN→REFACTOR, `skills/parallel-subagents` + `incremental-implementation`). Each cycle: failing test → minimal fix → `pytest tests/ -q`. Work on `main`, verifier `full_suite` 196 passed. `AGENTS.md:16` worktree + 3-attempt cap respected (0 escalations).

---
Run log: `loop-run-log.md:21`. Constraints: `loop-constraints.md:1` 7 rules. Budget: `loop-budget.md:1` L1 0 / L2 2. Next: commit whole-system batch → `npx @cobusgreyling/loop badge .`.

### TDD Evidence (prove-it, fleet 8 agents)

- `tests/test_token_count_overcount.py:1` — 3 FAILED→5 PASSED, fix `app.py:308`/`anthropic_api.py:305`.
- `tests/test_error_reason_mapping.py:1` — 2 FAILED→4 PASSED, fix `server/reasons.py:1`.
- `tests/test_noqa_markers_extended.py:1` — 1 FAILED→1 PASSED, extended as/tuple regex.
- `server/params.py:1` + `server/reasons.py:1` + `server/common.py:1` DRY — 189→200 passed (guard).
- `tests/test_common_helpers.py:1` — 4 new, all PASSED first run (block parity, SSE headers, flags, 400/503).
- `tests/test_remaining_gaps.py:98` — vacuous `==4 or >=1` → exact `==4`, still PASSED (cap binds).
- `tools.py:244,259` — delimiter breakout MITIGATED (reviewer + security APPROVE), normal bytes identical.
- Fleet: wave-1 code-reviewer/security/test/explore/perf (5 parallel) → wave-2 re-review 3x APPROVE.
