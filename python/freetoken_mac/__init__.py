"""FreeToken-Mac: edge-native MoE serving on Apple Silicon.

Phase 0 surface: load a GGUF model onto the Metal backend and generate from it.
Phase 1 adds ``freetoken_mac.engine``: N requests interleaved through one
``llama_decode`` per step behind a pluggable ``AdmissionPolicy``. The
control-plane layer (``freetoken_mac.server``) arrives in Phase 2.
"""

from __future__ import annotations

from ._freetoken_metal import (  # noqa: F401
    Batch,
    Context,
    ContextParams,
    LazyMode,
    LoadMode,
    Model,
    ModelParams,
    SamplerParams,
    backend_init,
)
from .engine import (  # noqa: F401
    AdmissionPolicy,
    DraftEngine,
    EngineConfig,
    FCFSPolicy,
    MetalEngine,
    MLXEngine,
    NgramTable,
    RequestParams,
    RequestState,
    SeqIdExhausted,
    StepBudget,
    StepOutput,
    StepPlan,
)
from .generate import generate

__version__ = "0.0.1.dev0"

__all__ = [
    "AdmissionPolicy",
    "Batch",
    "Context",
    "ContextParams",
    "DraftEngine",
    "EngineConfig",
    "FCFSPolicy",
    "LazyMode",
    "LoadMode",
    "MetalEngine",
    "MLXEngine",
    "Model",
    "ModelParams",
    "NgramTable",
    "RequestParams",
    "RequestState",
    "SamplerParams",
    "SeqIdExhausted",
    "StepBudget",
    "StepOutput",
    "StepPlan",
    "backend_init",
    "generate",
    "__version__",
]
