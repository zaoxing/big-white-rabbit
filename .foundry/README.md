# Harness stack (from loop-engineering)

Scaffolded by `loop-init --with-foundry` for pattern **daily-triage** (preset: **minimal**).

```
loop-engineering  →  harness-foundry  →  outerloop
   (patterns)         (runtime)          (governance)
```

## Next

```bash
npx @cobusgreyling/harness-foundry validate
npx @cobusgreyling/harness-foundry run --goal "Verify harness wiring"
npx @cobusgreyling/harness-foundry evolve report --session <id>
```

Showcase: https://github.com/cobusgreyling/harness-foundry/blob/main/docs/showcase.md
