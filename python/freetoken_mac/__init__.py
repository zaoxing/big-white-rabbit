"""FreeToken-Mac: edge-native MoE serving on Apple Silicon.

Phase 0 surface: load a GGUF model onto the Metal backend and generate from it.
The engine/control-plane layers (``freetoken_mac.engine``, ``freetoken_mac.server``)
arrive in Phases 1-2.
"""

from __future__ import annotations

from ._freetoken_metal import (  # noqa: F401
    Context,
    ContextParams,
    LazyMode,
    LoadMode,
    Model,
    ModelParams,
    SamplerParams,
    backend_init,
)
from .generate import generate

__version__ = "0.0.1.dev0"

__all__ = [
    "Context",
    "ContextParams",
    "LazyMode",
    "LoadMode",
    "Model",
    "ModelParams",
    "SamplerParams",
    "backend_init",
    "generate",
    "__version__",
]
