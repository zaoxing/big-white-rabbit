# Safety — FreeToken-Mac Loop Gates

This file is the human-owned denylist and auto-merge policy for the loop. The loop reads `loop-constraints.md` and `AGENTS.md` each run; this document is the rationale behind those rules and the place to review before enabling L3.

## Denylisted Paths

Never edit without human approval (see `loop-constraints.md:12` and `AGENTS.md:14`):

- `.env`, `.env.*`, `auth/`, `payments/`, `secrets/`, `credentials/`
- `.loop/` (contract, evidence, state history — read-only for loop)
- `third_party/`, `models/*.gguf`, `build/`, `.venv/`
- `infrastructure/`, `deploy/`, `*.tf`, `k8s/`

## Auto-Merge Policy

- **Never auto-merge to `main`**. Every fix is proposed as a patch or draft PR and waits for human `APPROVE` (see `AGENTS.md:16` maker/checker).
- Max 3 fix attempts per item; after that escalate to human (see `loop-constraints.md:20`).
- Never close an issue or PR without human approval (`loop-constraints.md:25`).

## Push & Merge Gates

- `loop-constraints.md:8` — Don't push before telling me.
- Use `git worktree` for every L2 code-changing attempt (see `LOOP.md:14`).
- CI must be green: `FTM_TEST_MODEL=... pytest tests/ -q` ≥ `194 passed` (current) and `npx @cobusgreyling/loop doctor .` `L3 healthy`.

## MCP / Connector Scopes

- L1 report-only: no MCP required, read-only `bash` + `edit` with `ask` (see `opencode.json:8` `loop-triage` permissions).
- L2+: GitHub MCP may read CI/issues and comment, but scoped to `read + comment` until trusted. Never grant `push` or `merge` to the agent.
- Least-privilege: `verifier` has `edit: deny` (`opencode.json:28`), `loop-triage` has `edit: ask`.

## Kill Switch

- Label or file flag `loop-pause-all` immediately stops all loops (see `loop-budget.md:19`).
- Resume only after human clears flag in `STATE.md`.

## Budget

- `loop-budget.md:8` `100k/day`, `2 runs/day`, `0 (L1)/2 (L2)` spawns.
- At 80% switch to report-only; at 100% exit (`skills/loop-budget`).

## Review

- This file is reviewed before enabling `foundry`/`fleet`/`memory` tiers (see `install-loop` stop condition: score ≥80 + human opt-in).
