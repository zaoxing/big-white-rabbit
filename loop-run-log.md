# Loop Run Log — YOUR_PROJECT

Append one entry per run. Prune entries older than 30 days.

## Format

```json
{
  "run_id": "2026-06-09T08:15:00Z",
  "pattern": "daily-triage",
  "duration_s": 45,
  "items_found": 4,
  "actions_taken": 1,
  "escalations": 0,
  "tokens_estimate": 52000,
  "outcome": "report-only | fix-proposed | escalated | no-op"
}
```

## Recent Runs

<!-- Loop appends below this line -->
- 2026-09-10T04:16:20Z — daily-triage — report-only — 2 found, 0 taken, 0 escalations, ~8k tokens, 45s

```json
{
  "run_id": "2026-09-10T04:16:20Z",
  "pattern": "daily-triage",
  "duration_s": 45,
  "items_found": 2,
  "actions_taken": 0,
  "escalations": 0,
  "tokens_estimate": 8000,
  "outcome": "report-only"
}
```
- 2026-09-10T04:45:00Z — daily-triage — L2 TDD 7 cycles — 6 found, 7 fixed, 0 escalations, ~45k tokens, 900s

```json
{
  "run_id": "2026-09-10T04:45:00Z",
  "pattern": "daily-triage",
  "duration_s": 900,
  "items_found": 6,
  "actions_taken": 7,
  "escalations": 0,
  "tokens_estimate": 45000,
  "outcome": "fix-proposed"
}
```