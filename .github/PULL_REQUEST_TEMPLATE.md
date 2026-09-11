# Pull Request

## What

## Why

## Tests

- [ ] `FTM_TEST_MODEL=models/qwen2.5-0.5b-instruct-q4_k_m.gguf ./.venv/bin/python -m pytest tests/ -q` green (add `FTM_MLX_MODEL=<dir>` / `FTM_MOE_MODEL=<file>` to cover those paths)

## Safety

- [ ] No secrets or credentials (`.env`, `auth/`, `payments/`, `secrets/`, `credentials/`)
- [ ] No auto-merge; human review required
- [ ] One change per PR
