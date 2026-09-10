# Fleet Registry — FreeToken-Mac (daily-triage, opencode)

Least-privilege populations mirroring `opencode.json` agents. Week-one is
report-only; implementer/verifier activate only after human enables L2 per item.

## Agents

- **Agent ID:** `loop-triage`
  - **Role:** Report-only daily triage. Reads STATE.md, updates High Priority / Watch List. No source edits in L1.
  - **Allowed Tools:** `read`, `bash` (ask), `edit` (ask)
  - **Capabilities:** triage commits/issues/CI, STATE.md updates

- **Agent ID:** `implementer`
  - **Role:** L2 minimal scoped fix inside an isolated git worktree. One fix per run.
  - **Allowed Tools:** `read`, `bash` (ask), `edit` (ask)
  - **Capabilities:** minimal fix, documented tests, stop on denylist (gate.yaml)

- **Agent ID:** `verifier`
  - **Role:** Checker for L2+ diffs. APPROVE or REJECT only, no edits.
  - **Allowed Tools:** `read`, `bash` (ask), `edit` (deny)
  - **Capabilities:** diff review, test-evidence check vs docs/safety.md
