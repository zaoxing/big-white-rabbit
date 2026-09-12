"""HTTP control plane: a single-process, dual-protocol server (OpenAI and Anthropic) over one AsyncEngine."""

from __future__ import annotations

__all__ = ["build_app", "serve"]


def __getattr__(name: str):
    # Lazy so `import bwr` does not require fastapi/uvicorn to be installed;
    # the engine is usable on its own (pyproject exposes the server as the [serve] extra).
    if name == "build_app":
        from .app import build_app

        return build_app
    if name == "serve":
        from .launch import serve

        return serve
    raise AttributeError(name)
