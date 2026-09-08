"""`ftm serve` -- load a model, build the app, run uvicorn.

Replaces FreeToken's ``server/launch.py``, which spawns scheduler/tokenizer/detokenizer
processes and wires them over ZMQ. There is nothing to spawn here: one process owns the
model, and the engine thread is started by the app's lifespan hook.
"""

from __future__ import annotations

from .._freetoken_metal import Model, ModelParams
from ..engine.config import EngineConfig


def serve(
    model_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 1919,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    n_batch: int = 512,
    n_seq_max: int = 8,
    kv_unified: bool = False,
    served_model_name: str | None = None,
    log_level: str = "info",
) -> None:
    import uvicorn

    from .app import build_app

    mp = ModelParams()
    mp.n_gpu_layers = n_gpu_layers
    model = Model(model_path, mp)

    config = EngineConfig(
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_ubatch=n_batch,
        n_seq_max=n_seq_max,
        kv_unified=kv_unified,
    )
    app = build_app(model, config, served_model_name=served_model_name)

    try:
        # Single worker on purpose: the model and its KV cache live in this process, so a
        # second worker would load a second copy of the weights and halve the RAM budget.
        uvicorn.run(app, host=host, port=port, log_level=log_level, workers=1)
    finally:
        # Release the weights explicitly. The app's lifespan already closed the context;
        # the model is separate and, held alive by FastAPI's route closures, may outlive
        # Python's last collection -- at which point ggml's Metal device destructor
        # asserts on its non-empty residency sets and abort()s a process that had already
        # shut down cleanly. See docs/llamacpp-notes.md.
        model.close()
