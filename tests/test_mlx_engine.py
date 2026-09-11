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


REPETITIVE = "Count: " + " ".join(str(i % 10) for i in range(48))
REP_TOKENS = 16


def _rep_params(max_tokens: int = REP_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False)


@pytest.fixture(scope="module")
def plain_engine() -> MLXEngine:
    return MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))


@pytest.fixture(scope="module")
def spec_engine() -> MLXEngine:
    return MLXEngine(
        MODEL_PATH, EngineConfig(n_ctx=4096, speculative=True)
    )


def _run(engine: MLXEngine, prompt: str = REPETITIVE) -> int:
    rid = engine.add_request(prompt, _rep_params())
    list(engine.drain())
    return rid


@needs_weights
def test_spec_on_off_byte_identical(
    plain_engine: MLXEngine, spec_engine: MLXEngine
) -> None:
    """The invariant, MLX-flavored: speculation must be byte-identical."""
    plain_id = _run(plain_engine)
    spec_id = _run(spec_engine)
    assert spec_engine.tokens_of(spec_id) == plain_engine.tokens_of(plain_id)
    assert (
        spec_engine.state(spec_id).finish_reason
        == plain_engine.state(plain_id).finish_reason
    )


@needs_weights
def test_speculative_engages(spec_engine: MLXEngine) -> None:
    """Drafts packed and accepted on repetitive text; rate in bounds."""
    before_drafted = spec_engine.spec_drafted
    rid = _run(spec_engine)
    assert spec_engine.spec_drafted > before_drafted
    assert spec_engine.tokens_of(rid) is not None
    rate = spec_engine.spec_acceptance_rate
    assert rate is not None and 0.0 <= rate <= 1.0


@needs_weights
def test_spec_fallback_all_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every draft wrong: output still identical, recomputes ran, and the
    request fell back to plain single-token evals."""
    from freetoken_mac.engine import mlx_engine as engine_module

    probe = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    probe_id = _run(probe)
    expected = probe.tokens_of(probe_id)
    prompt_ids = probe.state(probe_id).prompt
    vocab = probe._tokenizer.vocab_size
    hist = prompt_ids + expected
    wrong = {
        tuple(hist[max(0, i - 2) : i]): (tok + 1) % vocab
        for i, tok in enumerate(hist)
    }

    real_table = engine_module.NgramTable

    class WrongTable(real_table):  # type: ignore[valid-type, misc]
        def predict(self, context, max_tokens: int) -> list[int]:  # type: ignore[override]
            token = wrong.get(tuple(context[-(self.order - 1) :]))
            return [token] * max_tokens if token is not None else []

    monkeypatch.setattr(engine_module, "NgramTable", WrongTable)
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096, speculative=True))
    rid = _run(engine)
    assert engine.tokens_of(rid) == expected
    assert engine.spec_recomputes > 0, "no recompute ran on all-mismatch"
    assert engine.spec_fallbacks == 1, "fallback never engaged"


@needs_weights
def test_speculation_defaults_off(plain_engine: MLXEngine) -> None:
    assert plain_engine._spec_tables == {}
    assert plain_engine.spec_drafted == 0
    assert plain_engine.spec_acceptance_rate is None


@needs_weights
def test_nongreedy_ignores_spec() -> None:
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096, speculative=True))
    rid = engine.add_request(
        REPETITIVE, RequestParams(temp=1.0, max_tokens=8, stop_at_eog=False)
    )
    list(engine.drain())
    assert engine.spec_drafted == 0
    assert len(engine.tokens_of(rid)) == 8
    assert engine.state(rid).finish_reason == "length"


@needs_weights
def test_eog_reason_and_no_eos_leak() -> None:
    """A natural end must retire eog (not length), with no end-token piece
    text leaking into output. Regression: the EOS set used to miss
    <|endoftext|>, leaking its piece and mislabeling the finish."""
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    rid = engine.add_request(
        "Say hi in one sentence.",
        RequestParams(temp=0.0, max_tokens=64, stop_at_eog=True),
    )
    list(engine.drain())
    assert engine.state(rid).finish_reason == "eog"
    text = engine.text_of(rid)
    assert "<|endoftext|>" not in text
    assert "<|im_end|>" not in text


@needs_weights
def test_thinking_disabled_by_default() -> None:
    """Server-shaped prompt (template-rendered) must produce no think block:
    thinking is off unless a future API opts in."""
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=8192))
    prompt = engine.apply_chat_template([("user", "Say hi in one sentence.")])
    rid = engine.add_request(
        prompt, RequestParams(temp=0.0, max_tokens=32, stop_at_eog=True)
    )
    list(engine.drain())
    assert "<think>" not in engine.text_of(rid)
