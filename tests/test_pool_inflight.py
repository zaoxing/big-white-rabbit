"""The pool's in-flight accounting as the request paths actually exercise it.

`ModelPool.request_started` / `request_finished` are what make
`_unload_locked` wait for a live generation instead of stopping its engine.
tests/test_pool.py proves the pool honours them; this file proves the SERVER
calls them -- a real `ModelPool` and a real `AsyncEngine` over a fake engine
that emits one token per step, driven through the FastAPI app, so that an
eviction triggered by a second request is seen by the first one as either
"all my tokens" (correct) or "a truncated 200" (the bug).

The same app serves the Anthropic surface in pool mode, which is the other
thing that only shows up with a pool: the renderer is per request there,
not per server.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import types

import pytest
from bwr.engine.config import EngineConfig
from bwr.engine.metal_engine import StepOutput

GIB = 1024**3
N_TOKENS = 20


class _FakeRaw:
    """A synchronous engine that produces `N_TOKENS` pieces, one per step.

    Only the worker thread calls `add_request` / `step` / `cancel`, as with
    the real engines; the renderer methods run on the loop thread and touch
    nothing shared.
    """

    def __init__(self, entry, step_delay: float = 0.005) -> None:
        self.entry = entry
        self.step_delay = step_delay
        self._next = 0
        self._emitted: dict[int, int] = {}
        self._lock = threading.Lock()
        self.ctx = types.SimpleNamespace(
            close=lambda: None, n_ctx=4096, n_ctx_seq=4096, n_seq_max=8,
            decode_calls=0,
        )

    # renderer -------------------------------------------------------------

    def apply_chat_template(self, pairs, add_generation_prompt):
        return "\n".join(f"{r}: {t}" for r, t in pairs)

    def tokenize(self, prompt, *, add_special=True, parse_special=True):
        return list(range(len(prompt.split())))

    # engine ---------------------------------------------------------------

    def add_request(self, prompt, params, *, add_special=True) -> int:
        with self._lock:
            rid = self._next
            self._next += 1
            self._emitted[rid] = 0
            return rid

    @property
    def has_work(self) -> bool:
        with self._lock:
            return bool(self._emitted)

    def step(self) -> list[StepOutput]:
        time.sleep(self.step_delay)
        out: list[StepOutput] = []
        with self._lock:
            for rid in list(self._emitted):
                self._emitted[rid] += 1
                n = self._emitted[rid]
                done = n >= N_TOKENS
                out.append(
                    StepOutput(
                        request_id=rid, token=n, piece="t ",
                        finished=done, finish_reason="eog" if done else None,
                    )
                )
                if done:
                    del self._emitted[rid]
        return out

    def cancel(self, rid: int) -> bool:
        with self._lock:
            return self._emitted.pop(rid, None) is not None


@pytest.fixture
def pool_app(tmp_path):
    """A pool app whose budget fits ONE of two equally sized models."""
    pytest.importorskip("httpx")
    from bwr.server.app import build_app

    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    app = build_app(None, EngineConfig(), model_dir=str(tmp_path))
    pool = app.state.pool
    pool._factory = lambda entry, _cfg: _FakeRaw(entry)
    for mid in ("a", "b"):
        pool._entries[mid].size_bytes = 10 * GIB
    pool.budget_bytes = 15 * GIB
    return app


async def _post(app, path: str, body: dict):
    import httpx

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.post(path, json=body)


def _chat(model: str, *, stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 64,
        "stream": stream,
    }


def _messages(model: str, *, stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 64,
        "stream": stream,
    }


def _run_eviction_race(app, first: dict, second: dict, first_path: str):
    """Start `first`, then `second` (which evicts first's model) mid-stream.

    Returns (first response, second response, pool) once both are done.
    """

    async def go():
        async with app.router.lifespan_context(app):
            t_first = asyncio.create_task(_post(app, first_path, first))
            # Let A be admitted and produce a few tokens before B arrives.
            for _ in range(200):
                await asyncio.sleep(0.005)
                raw = app.state.pool.engine_for("a")
                if raw is not None and raw._next > 0 and raw._emitted.get(0, 0) >= 3:
                    break
            r_second = await _post(app, "/v1/chat/completions", second)
            r_first = await t_first
            return r_first, r_second

    return asyncio.run(go())


# -- finding 1: the request paths bracket the pool's in-flight count ----------


def test_openai_completion_survives_an_eviction_by_another_model(pool_app):
    """Request A (model a, 20 tokens) must get all 20 tokens even though
    request B (model b) arrives mid-generation and needs a's budget: the
    pool waits for A to drain before stopping a's engine. Without the server
    bracketing the request, `in_flight` is 0, the engine is stopped under A,
    and A gets a truncated 200 with finish_reason "stop"."""
    r_a, r_b = _run_eviction_race(
        pool_app, _chat("a"), _chat("b"), "/v1/chat/completions"
    )
    assert r_a.status_code == 200, r_a.text
    assert r_b.status_code == 200, r_b.text
    body = r_a.json()
    assert body["usage"]["completion_tokens"] == N_TOKENS, body
    assert body["choices"][0]["message"]["content"] == "t " * N_TOKENS


def test_openai_stream_survives_an_eviction_by_another_model(pool_app):
    r_a, r_b = _run_eviction_race(
        pool_app, _chat("a", stream=True), _chat("b"), "/v1/chat/completions"
    )
    assert r_a.status_code == 200 and r_b.status_code == 200
    frames = [ln for ln in r_a.text.split("\n") if ln.startswith("data: ")]
    content = "".join(
        json.loads(ln[6:])["choices"][0].get("delta", {}).get("content", "")
        for ln in frames if ln != "data: [DONE]"
    )
    assert content == "t " * N_TOKENS
    assert frames[-1] == "data: [DONE]"


def test_anthropic_message_survives_an_eviction_by_another_model(pool_app):
    r_a, r_b = _run_eviction_race(
        pool_app, _messages("a"), _chat("b"), "/v1/messages"
    )
    assert r_a.status_code == 200, r_a.text
    assert r_b.status_code == 200
    body = r_a.json()
    assert body["usage"]["output_tokens"] == N_TOKENS, body
    assert body["content"][0]["text"] == "t " * N_TOKENS


def test_anthropic_stream_survives_an_eviction_by_another_model(pool_app):
    r_a, r_b = _run_eviction_race(
        pool_app, _messages("a", stream=True), _chat("b"), "/v1/messages"
    )
    assert r_a.status_code == 200 and r_b.status_code == 200
    assert r_a.text.count('"text_delta"') == N_TOKENS
    assert '"output_tokens": 20' in r_a.text or '"output_tokens":20' in r_a.text
    assert "event: message_stop" in r_a.text


def test_in_flight_is_released_on_every_exit(pool_app):
    """After a request on each surface finishes, nothing is left counted as
    in flight -- otherwise the first unload would hang forever."""

    async def go():
        async with app.router.lifespan_context(app):
            for path, body in (
                ("/v1/chat/completions", _chat("a")),
                ("/v1/chat/completions", _chat("a", stream=True)),
                ("/v1/messages", _messages("a")),
                ("/v1/messages", _messages("a", stream=True)),
            ):
                r = await _post(app, path, body)
                assert r.status_code == 200, (path, r.text)
            rec = app.state.pool._loaded["a"]
            assert rec.in_flight == 0
            assert rec.drained.is_set()
            assert await asyncio.wait_for(app.state.pool.unload("a"), 2.0) is True

    app = pool_app
    asyncio.run(go())


# -- finding 2: /v1/messages renders with the RESOLVED model in pool mode ----


def test_anthropic_messages_works_in_pool_mode(pool_app):
    """Pool mode has no server-wide renderer; the route must resolve first
    and render with the model the request picked. It used to render before
    resolving, against a renderer of None, and 500 on every request."""

    async def go():
        async with pool_app.router.lifespan_context(pool_app):
            return await _post(pool_app, "/v1/messages", _messages("a"))

    r = asyncio.run(go())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "a"
    assert body["usage"]["input_tokens"] == 2  # "user: go"
    assert body["content"][0]["text"] == "t " * N_TOKENS
