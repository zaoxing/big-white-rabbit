"""`bwr serve` -- load a model, build the app, run uvicorn.

Replaces Big White Rabbit's ``server/launch.py``, which spawns scheduler/tokenizer/detokenizer
processes and wires them over ZMQ. There is nothing to spawn here: one process owns the
model, and the engine thread is started by the app's lifespan hook.
"""

from __future__ import annotations

from .._bwr_metal import Model, ModelParams
from ..engine.config import EngineConfig


def serve(
    model_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 1919,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    n_batch: int = 512,
    n_ubatch: int | None = None,
    n_seq_max: int = 8,
    n_threads: int = 0,
    n_threads_batch: int = 0,
    kv_unified: bool = False,
    served_model_name: str | None = None,
    log_level: str = "info",
    draft_model_path: str | None = None,
    speculative: bool = False,
    spec_max_drafts: int = 4,
    prefix_cache: bool = False,
    prefix_cache_pins: int = 2,
    prefix_cache_min_tokens: int = 256,
    record_experts: bool = False,
    ssd_hotlist: bool = False,
    ssd_hotlist_k: int = 32,
    ssd_hotlist_bytes: int | None = None,
    engine: str = "metal",
) -> None:
    import uvicorn

    from .app import build_app

    config = EngineConfig(
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_ubatch=n_ubatch if n_ubatch is not None else n_batch,
        n_seq_max=n_seq_max,
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
        kv_unified=kv_unified,
        speculative=speculative,
        spec_max_drafts=spec_max_drafts,
        prefix_cache=prefix_cache,
        prefix_cache_pins=prefix_cache_pins,
        prefix_cache_min_tokens=prefix_cache_min_tokens,
        record_experts=record_experts,
        ssd_hotlist=ssd_hotlist,
        ssd_hotlist_k=ssd_hotlist_k,
        ssd_hotlist_bytes=ssd_hotlist_bytes,
        engine=engine,
    )
    if engine == "mlx":
        # model_path names an MLX weights directory here, not a GGUF: no
        # llama Model to load (MLXEngine loads it lazily itself).
        app = build_app(
            None, config,
            served_model_name=served_model_name, mlx_model_path=model_path,
        )
        uvicorn.run(app, host=host, port=port, log_level=log_level, workers=1)
        return

    mp = ModelParams()
    mp.n_gpu_layers = n_gpu_layers
    model = Model(model_path, mp)
    # A second resident model: the drafter's weights live alongside the
    # target's, so budget ~target + draft weights before enabling this.
    draft_model = Model(draft_model_path, mp) if draft_model_path else None

    app = build_app(
        model, config,
        served_model_name=served_model_name, draft_model=draft_model
    )

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
        if draft_model is not None:
            draft_model.close()
