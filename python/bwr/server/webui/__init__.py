"""Web UI for big-white-rabbit, mounted at /admin.

Provenance
==========

The templates and static assets under this package are derived from oMLX
(https://github.com/jundot/omlx, Apache-2.0; full licence text in
vendor/LICENSE.omlx-Apache-2.0). Changes made, per Apache-2.0 section 4(b):

- oMLX's name and marks are REPLACED throughout, not merely hidden. Apache-2.0
  section 6 grants no trademark licence, so a derivative must not ship under
  the licensor's brand. The rabbit marks in `static/*.svg` are original to
  this project.
- Client-side storage keys renamed `omlx-*` -> `bwr-*`, so a browser can hold
  state for both this and a real oMLX install without collision.
- The bundled webfonts (~11 MB) were dropped; the UI falls back to the system
  font stack.
- The server side is NOT oMLX's: `routes.py` here is a small original adapter
  over bwr's own engine, not a copy of `omlx/admin/routes.py`.

Source comments in the vendored JavaScript that cite `omlx/...` paths are
deliberately preserved as provenance.

Scope
=====

This serves the chat UI against bwr's engine. oMLX-specific surfaces
(cluster, OQ manager, ANE tuning, benchmark suites, model downloads) have no
bwr equivalent; `routes.py` answers those with explicit empty/unsupported
payloads so the page degrades instead of erroring. See `routes.py` for the
per-endpoint status.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

__all__ = ["PACKAGE_DIR", "TEMPLATES_DIR", "STATIC_DIR", "build_router"]


def build_router(*args, **kwargs):
    """Lazy re-export so importing this package costs no fastapi import."""
    from .routes import build_router as _build

    return _build(*args, **kwargs)
