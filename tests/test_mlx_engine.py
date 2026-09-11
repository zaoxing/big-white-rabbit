"""MLX backend (SPEC-mlx-engine.md, M10b/M10c).

Skips unless the mlx extra is installed AND FTM_MLX_MODEL names an MLX
weights directory. Text-equality with the Metal backend is NOT asserted:
different precisions may legally flip greedy argmaxes (same standard as the
KV-quant battery). What is asserted: interface parity (admit/step/drain/
cancel/accessors), finish-reason validity, stop-sequence truncation, and
AsyncEngine compatibility (including context close).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest

import freetoken_mac as ftm
from freetoken_mac.engine import EngineConfig, RequestParams
from freetoken_mac.engine.mlx_engine import MLXEngine

MODEL_PATH = os.environ.get("FTM_MLX_MODEL")

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None,
    reason="pip install 'freetoken-mac[mlx]' to run MLX tests",
)
needs_weights = pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None
    or not MODEL_PATH
    or not os.path.isdir(MODEL_PATH or ""),
    reason="need mlx installed and FTM_MLX_MODEL to run MLX backend tests",
)

N_TOKENS = 8


def greedy(max_tokens: int = N_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=True)


def test_bad_engine_name_rejected_without_weights() -> None:
    from freetoken_mac.server.app import build_app

    with pytest.raises(ValueError, match="engine"):
        build_app(None, EngineConfig(engine="bogus"))  # type: ignore[arg-type]


def test_mlx_engine_requires_path() -> None:
    from freetoken_mac.server.app import build_app

    with pytest.raises(ValueError, match="mlx_model_path"):
        build_app(None, EngineConfig(engine="mlx"))  # type: ignore[arg-type]


@needs_weights
def test_generate_to_cap() -> None:
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    rid = engine.add_request("Hello, what model are you?", greedy())
    outs = list(engine.drain())
    assert len(engine.tokens_of(rid)) == N_TOKENS
    assert engine.state(rid).finish_reason == "length"
    assert all(o.request_id == rid for o in outs)
    assert not engine.has_work


@needs_weights
def test_cancel() -> None:
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    rid = engine.add_request("Count: 1 2 3", greedy(64))
    engine.step()
    assert engine.cancel(rid) is True
    assert engine.cancel(rid) is False
    assert not engine.has_work


@needs_weights
def test_stop_sequence_truncates() -> None:
    """Two-phase like the Metal stop proof: observe text, then stop inside it.
    Temp-0 determinism reproduces the span, so the delimiter must hit; asserted
    on retirement reason and early shortfall (same contract as MetalEngine --
    no assertion about response text, which can't tell "stopped" from "cut").
    """
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    probe = engine.add_request(
        "Explain what mixture-of-experts means briefly.",
        RequestParams(temp=0.0, max_tokens=32, stop_at_eog=True),
    )
    list(engine.drain())
    full = engine.text_of(probe)
    assert len(full) > 20
    stop = full[10:20]
    rid = engine.add_request(
        "Explain what mixture-of-experts means briefly.",
        RequestParams(temp=0.0, max_tokens=32, stop_at_eog=True, stop=(stop,)),
    )
    list(engine.drain())
    state = engine.state(rid)
    assert state.finish_reason == "stop_sequence"
    assert state.n_generated < 32


@needs_weights
def test_admission_validation() -> None:
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    with pytest.raises(ValueError, match="max_tokens"):
        engine.add_request("hi", RequestParams(temp=0.0, max_tokens=0))
    with pytest.raises(ValueError, match="zero tokens"):
        engine.add_request("", greedy())
    with pytest.raises(ValueError, match="capacity"):
        tiny = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=16))
        tiny.add_request("word " * 100, greedy())
    with pytest.raises(KeyError):
        engine.tokens_of(999)


@needs_weights
def test_async_engine_compat() -> None:
    """AsyncEngine drives MLXEngine end to end, including context close."""

    async def scenario() -> tuple[list[int], bool]:
        from freetoken_mac.engine.async_engine import AsyncEngine

        engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
        async_engine = AsyncEngine(engine)
        await async_engine.start()
        try:
            rid = await async_engine.submit("Count: 1 2 3", greedy())
            seen: list[int] = []
            async for out in async_engine.stream(rid):
                seen.append(out.token)
                if out.finished:
                    assert out.finish_reason == "length"
            async_engine.release(rid)
            return seen, engine.ctx.closed
        finally:
            await async_engine.stop()

    seen, closed_before_stop = asyncio.run(scenario())
    assert len(seen) == N_TOKENS
    assert closed_before_stop is False


@needs_weights
def test_chat_completion_end_to_end() -> None:
    """The reported default-serve 500: chat rendering needs the tokenizer,
    not the llama Model. Proves a completion serves over HTTP."""
    from fastapi.testclient import TestClient

    from freetoken_mac.server.app import build_app

    app = build_app(
        None, EngineConfig(n_ctx=4096, engine="mlx"),
        served_model_name="t", mlx_model_path=MODEL_PATH,
    )
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "t",
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["message"]["content"].strip() != ""
        assert body["usage"]["completion_tokens"] >= 1


@needs_weights
def test_health_geometry() -> None:
    """The exact /health failure: geometry + counters must exist so serve
    boots (default engine) instead of 500ing."""
    from fastapi.testclient import TestClient

    from freetoken_mac.server.app import build_app

    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    assert engine.ctx.n_ctx == 4096
    assert engine.ctx.n_ctx_seq == 4096
    assert engine.ctx.n_seq_max == 8
    assert engine.ctx.decode_calls == 0
    assert engine.n_free_seq_slots == 8
    rid = engine.add_request("hi", greedy())
    list(engine.drain())
    assert engine.ctx.decode_calls == N_TOKENS

    app = build_app(
        None, EngineConfig(n_ctx=4096, engine="mlx"),
        served_model_name="t", mlx_model_path=MODEL_PATH,
    )
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["n_ctx"] == 4096
        assert body["free_seq_slots"] == 8
