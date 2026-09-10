# Pull Request

## What

## Why

## Tests

- [ ] `FTM_TEST_MODEL=models/qwen2.5-0.5b-instruct-q4_k_m.gguf ./.venv/bin/python -m pytest tests/ -q` → `194 passed`
- [ ] `npx @cobusgreyling/loop doctor .` → `L3 healthy`

## Safety

- [ ] No denylisted paths (`.env`, `auth/`, `payments/`, `secrets/`, `credentials/`, `.loop/`)
- [ ] No auto-merge; human review required (see `docs/safety.md`)
- [ ] One fix per PR, verifier `APPROVE` attached if L2
