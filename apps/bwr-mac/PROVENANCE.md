# Big White Rabbit — macOS menubar app

## Provenance

Derived from the oMLX macOS app (https://github.com/jundot/omlx,
`apps/omlx-mac`), licensed under the Apache License 2.0. The full licence
text is at `vendor/LICENSE.omlx-Apache-2.0` in this repository.

Apache-2.0 section 4(b) requires a derivative to state its changes. They are:

- **Rebranded in full.** Section 6 grants no trademark licence, so the app
  does not ship under the licensor's name or marks. `oMLX` → `BigWhiteRabbit`
  throughout, including the Xcode project, target, entitlements and tests.
  App icon and menubar glyphs are original artwork (see
  `Resources/Assets.xcassets`).
- **Repointed at bwr.** The app was forked via an intermediate `bwrx` naming
  pass; that is gone. 1,274 identifiers were renamed `bwrx` → `bwr` because
  the app now drives *this* repository's server, not a fork of oMLX's.
- **Default port 1919**, bwr's default, rather than oMLX's 8000.

## What it talks to

`python/bwr/server/webui/routes.py` — bwr's own adapter. The app drives a
small surface, all of it implemented there:

| endpoint | backing |
|---|---|
| `/health` | engine counters |
| `/admin/api/stats` | pool residency + per-model counters |
| `/admin/api/server-info` | version, model count, uptime |
| `/admin/api/global-settings` | auth disabled (bwr checks no key) |
| `/admin/dashboard`, `/admin/chat` | vendored UI pages |
| `/admin/auto-login`, `/admin/api/login` | no-ops; bwr has no auth |
| `/admin/api/activity` | empty — bwr keeps counters, not a request log |

Features the app inherited from oMLX that bwr has no equivalent for
(benchmark suites, ANE tuning, HuggingFace/ModelScope downloads, prompt
profiles, preset bundles, cluster) answer `supported: false` on GET and 501
on write, so those panels render empty rather than throwing. See the
`_UNSUPPORTED` list in `routes.py`.

## Building

Needs **full Xcode**, not just Command Line Tools:

```bash
xcodebuild -project apps/bwr-mac/BigWhiteRabbit.xcodeproj -list
xcodebuild -project apps/bwr-mac/BigWhiteRabbit.xcodeproj -scheme BigWhiteRabbit build
```

**Not verified in this repository.** It has only been checked structurally —
all 129 project file references resolve and no `bwrx`/`omlx` identifiers
remain. Nobody has compiled or run it here, because this machine has only
CommandLineTools installed. Treat "it builds" as untested.

Then start the server it expects:

```bash
bwr serve --model-dir models --port 1919
```
