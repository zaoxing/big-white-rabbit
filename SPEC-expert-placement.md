# Spec: expert placement plumbing (T9)

Status: APPROVED 2026-09-10 (human: --n-cpu-moe form deferred).
Phase: implementation on branch `feat/expert-placement` (worktree `../FreeToken-Mac-wt-place`).

## Objective

Expose per-tensor residency control so MoE experts can live outside Metal
buffers -- the prerequisite for all later streaming work. No policy yet,
just the knob + proof it works.

## Design (verified against our pin before speccing)

- **C++**: `ModelParams.expert_weights: {"metal" (default), "cpu"}`.
  `"cpu"` installs upstream's canned override (`\.ffn_(up|down|gate|gate_up)_`
  `(ch|)exps` -> CPU buft, the same regex as `--cpu-moe`; patterns are
  `std::regex` matched by substring search). Override structs + pattern
  strings live as `Model` members (common keeps them in params for the same
  lifetime reason; the loader only reads them during load, but member
  lifetime is free insurance). `has_tensor_overrides` side effects checked:
  only disables multi-device pipeline parallelism -- inert on single-device
  Mac. No `--n-cpu-moe` partial form yet (deferred to residency phase).
- **Python**: `ModelParams(expert_weights=...)` passthrough (plain string;
  validated in C++ before touching llama.cpp).
- **Scope**: model load only. No engine/context/server changes. No prefetch,
  no eviction, no policy.

## Testing strategy

- Unit (suite): default loads unchanged; invalid value rejected with a named
  error before load.
- Correctness proof (30B, FTM_MOE_MODEL): same prompt, all-Metal vs
  experts-CPU -> **byte-identical tokens** (placement must never change
  numerics; CPU compute is only slower).
- Efficacy baseline: wall time + tok/s both ways. Expect CPU-experts much
  slower -- that gap IS the T10 optimization target, recorded here.
- Arch drift guard: the 30B file carries 144 expert tensors (48 layers x 3,
  ~16.7 of 17.3GB); the behavioral gap (much slower iff matched) fails loud
  if a future arch renames them.
- Full suite green (default path untouched).

## Task breakdown

- [ ] T9a (M): C++ field + override install + lifetime handling + rebuild.
- [ ] T9b (S): Python passthrough + validation + placement tests + 30B A/B.
- [x] T9c (S): verifier pass + STATE.md. DONE: verifier APPROVE (247
      re-ran green + 4 MoE tests with evidence this session).

L2 rules: worktree per attempt, <= 3 attempts per item, verifier after
implementation, no push without human approval.

## Risks

| Risk | Mitigation |
|---|---|
| CPU-buft + mmap warning path | Measure default AUTO as-is; report, don't chase |
| Override silently matches nothing on drift | Behavioral gap (much slower iff matched) fails loud |
| C++ rebuild time | One build; worktree per rules |

## Open questions (resolved)

1. `--n-cpu-moe N` partial form now? WAIT -- deferred to residency phase.
