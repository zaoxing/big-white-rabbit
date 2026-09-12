# Spec: expert residency (activation tracking)

Status: PROPOSED 2026-09-10 — awaiting human approval.
Phase: `residency` capability-map module, after `placement` (T9, merged).

## Objective

Observe WHICH experts fire per token on the 30B MoE, building the activation
dataset that `qstar` (prefetch policy) will decide on. No placement changes,
no policy — measurement only.

## Findings (verified against our pin before speccing)

- The pin exposes NO runtime routing API (no expert-usage counters, no
  router-logit getters; hyperparams describe only the static shape).
- `build_moe_ffn` (llama-graph.cpp) does NOT name the router distribution
  tensor — only weights and the final output carry names. Name-based
  capture needs a one-line vendor touch.
- The available hook is `ggml_backend_sched_eval_callback` (`cb_eval`):
  per-graph-node callback, backend-agnostic, with tensor pointers. Copying
  the router distribution costs ~128 floats/layer/token (24KB/token on the
  30B) — negligible next to a decode.
- 30B tensor inventory (measured): 579 tensors, 144 expert tensors
  (`blk.N.ffn_{gate,up,down}_exps`, ~16.7GB), router weights
  `blk.N.ffn_gate_inp` x48.

## Design (proposed)

- **T10a — mechanism spike (measure first)**: name the router probs tensor
  in the vendored graph builder (`moe_probs-{il}`), attach `cb_eval` in a
  scratch C++ harness (NOT the engine yet), record top-k + full distribution
  for ~100 tokens on the 30B, and answer: (1) does capture work, (2) what is
  the skew (how concentrated is routing?), (3) per-step overhead. If skew is
  flat, STOP the track — streaming can't win.
- **T10b — binding** (DONE): `Context` gains activation recording behind a
  flag (default off, zero overhead when off): opt-in `record_experts` on the
  context params, per-decode callback into a drained-on-read frame list,
  Python accessor returning per-request activation lists. NO vendor patch
  was needed (tensors pre-named); single-sequence guard fails loud.
- **T10c — engine + tests** (DONE): `MetalEngine` exposes
  `expert_activations(rid)`; 7 tests pin defaults, shape + softmax
  exactness, determinism, consume semantics, output identity, seq guard,
  unknown rid. Verifier APPROVE; full suite green.

## Open questions

1. Metal buffer mapping: deferred — moot for the LRU design (no madvise
   until true SSD eviction; revisit when qstar needs it).
2. T10a PASSED — proceed to T10b binding?

## T10a verdict (measured 2026-09-10, 30B, 100 gen tokens)

- Capture works first try, NO vendor patch needed: router tensors are
  pre-named `ffn_moe_probs-{il}`, n_expert=128 on all 48 layers; cb_eval
  protocol (ask=true to keep, must return true after) verified empirically.
- Skew: avg top-1 0.043, top-8 0.197 (uniform: 0.008/0.062) — too diffuse
  for frequency prediction.
- Temporal locality SAVES the track (per-layer LRU sim on observed top-8):
  K=8 -> 0.302, K=16 -> 0.540, K=32 -> 0.758, K=64 -> 0.909. No predictor
  needed: resident set = LRU hot experts (K=32/layer ~= 4GB on 30B).
- Overhead: cb_eval active costs visible throughput (batching breaks at
  every router node) — production recording must sample, never always-on.

## Out of scope

Prefetch/evict policy (`qstar`), per-expert placement changes, serve
wiring, non-MoE models (router tensors don't exist on dense).
