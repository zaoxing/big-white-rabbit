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
    """A synchronous engine that produces `N_TOKENS` pieces, one per step,
    then an eog retire of its own.

    As with the real engines, the retire is not a generated token: it
    carries no text and comes one step after the last piece. Only the
    worker thread calls `add_request` / `step` / `cancel`, as with the real
    engines; the renderer methods run on the loop thread and touch nothing
    shared.
    """

    def __init__(self, entry, step_delay: float = 0.005) -> None:
        self.entry = entry
        self.step_delay = step_delay
        self._next = 0
        self._emitted: dict[int, int] = {}
        # Requests that reached their last token. A request whose engine
        # was stopped under it never gets here.
        self.finished: set[int] = set()
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
                done = n > N_TOKENS
                out.append(
                    StepOutput(
                        request_id=rid, token=n, piece="" if done else "t ",
                        finished=done, finish_reason="eog" if done else None,
                    )
                )
                if done:
                    del self._emitted[rid]
                    self.finished.add(rid)
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


_EVERY_SURFACE = [
    ("/v1/chat/completions", _chat("a")),
    ("/v1/chat/completions", _chat("a", stream=True)),
    ("/v1/messages", _messages("a")),
    ("/v1/messages", _messages("a", stream=True)),
]


@pytest.mark.parametrize("path,body", _EVERY_SURFACE)
def test_in_flight_is_released_when_submit_503s(pool_app, monkeypatch, path, body):
    """The bracket opens right after `resolve` and must close on the ERROR
    exit too: a submit the engine refuses (no free sequence slot, mapped to
    503) raises before any generator exists, so the handler's own finally is
    the only thing that can release it. Without that release the 503 leaves
    `in_flight` at 1 for a request that never ran, and the next unload of
    the model -- eviction, admin, context benchmark -- waits forever."""
    from bwr.engine.async_engine import AsyncEngine

    async def boom(self, prompt, params):
        raise RuntimeError("no free seq")

    monkeypatch.setattr(AsyncEngine, "submit", boom)

    async def go():
        async with pool_app.router.lifespan_context(pool_app):
            r = await _post(pool_app, path, body)
            assert r.status_code == 503, (path, r.text)
            rec = pool_app.state.pool._loaded["a"]
            try:
                assert rec.in_flight == 0, f"in_flight leaked: {rec.in_flight}"
            finally:
                # A leak would also hang the lifespan's shutdown (the pool
                # waits on `drained` under its lock); reset so the failure
                # is reported rather than timed out.
                rec.in_flight = 0
                rec.drained.set()
            assert await asyncio.wait_for(pool_app.state.pool.unload("a"), 2.0) is True

    asyncio.run(go())


@pytest.mark.parametrize("path,body", _EVERY_SURFACE)
def test_client_disconnect_cancels_and_releases(pool_app, monkeypatch, path, body):
    """A client that hangs up mid-generation stops costing decodes: every
    consumer loop polls `is_disconnected` per token, cancels the engine's
    request and leaves -- and on the streaming paths that early return still
    reaches the generator's finally, which hands the bracket back. No test
    polled the hang-up branch before, so dropping the poll (or the cancel,
    or the release behind it) left the suite green."""
    from starlette.requests import Request

    polls = {"n": 0}

    async def gone(self):  # connected on the first poll, gone from the second
        polls["n"] += 1
        return polls["n"] >= 2

    monkeypatch.setattr(Request, "is_disconnected", gone)

    async def go():
        async with pool_app.router.lifespan_context(pool_app):
            r = await _post(pool_app, path, body)
            raw = pool_app.state.pool.engine_for("a")
            for _ in range(100):
                if not raw._emitted:
                    break
                await asyncio.sleep(0.01)
            assert raw._emitted == {}, "engine request was not cancelled"
            rec = pool_app.state.pool._loaded["a"]
            assert rec.in_flight == 0
            return r

    r = asyncio.run(go())
    assert polls["n"] >= 2, "the hang-up was never polled"
    assert r.text.count("t ") < N_TOKENS, r.text


def test_a_benchmark_row_survives_an_eviction_by_another_model(pool_app):
    """The throughput benchmark issues requests through `resolve` like a
    handler does, so it must bracket them with the pool's in-flight count
    like a handler does. It used to skip the bracket: a request for model b
    mid-row evicted a, the engine was stopped under the row after a few
    tokens, and the run still reported "completed" with a rate computed
    from a fraction of the work."""

    async def go():
        app = pool_app
        async with app.router.lifespan_context(app):
            bench = app.state.bench
            started = bench.start(
                "a", prompt_lengths=[8], generation_length=N_TOKENS,
                batch_sizes=[1],
            )
            raw_a = None
            for _ in range(200):
                await asyncio.sleep(0.005)
                raw_a = app.state.pool.engine_for("a")
                if (raw_a is not None and raw_a._next > 0
                        and raw_a._emitted.get(0, 0) >= 3):
                    break
            assert raw_a is not None and raw_a._next > 0, "bench never started"
            r_b = await _post(app, "/v1/chat/completions", _chat("b"))
            for _ in range(400):
                body = bench.get(started["bench_id"])
                if body["status"] != "running":
                    break
                await asyncio.sleep(0.005)
            return raw_a, r_b, bench.get(started["bench_id"])

    raw_a, r_b, run = asyncio.run(go())
    assert run["status"] == "completed", run
    assert run["results"][0]["error"] is None, run
    # Every request the row issued ran to its last token.
    assert raw_a.finished == set(range(raw_a._next)), (raw_a.finished, raw_a._next)
    # And the bracket was released: b was served after a drained.
    assert r_b.status_code == 200, r_b.text
    assert r_b.json()["usage"]["completion_tokens"] == N_TOKENS


# -- a profile's cap is a default for BOTH spellings of the request's cap ----


def test_a_profile_cap_fills_only_a_request_that_sent_neither_spelling():
    """An exposed profile's max_tokens is a default, not an override, and the
    request has two spellings for its cap. The builder used to test only the
    legacy field before applying the profile, so an explicit
    `max_completion_tokens: 400` lost to a profile's 50 -- the one spelling
    `resolved_max_tokens` says wins lost to a default."""
    from bwr.server.app import DEFAULT_MAX_TOKENS, _request_params
    from bwr.server.schemas import ChatCompletionRequest

    def req(**kw):
        return ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}], **kw
        )

    profile = {"max_tokens": 50}
    assert _request_params(req(max_completion_tokens=400), profile).max_tokens == 400
    assert _request_params(req(max_tokens=400), profile).max_tokens == 400
    assert (
        _request_params(req(max_completion_tokens=400, max_tokens=7), profile).max_tokens
        == 400
    )
    # Nothing explicit: the profile fills the gap, ahead of the server default.
    assert _request_params(req(), profile).max_tokens == 50
    assert _request_params(req(), {}).max_tokens == DEFAULT_MAX_TOKENS


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


# -- finding 3: Metal's prefix counter reaches the stats ----------------------


def test_prefix_tokens_reads_the_metal_counter_too():
    from bwr.server.app import _prefix_tokens

    mlx_like = types.SimpleNamespace(prefix_hit_tokens=7)
    metal_like = types.SimpleNamespace(prefix_cache_hits=1, prefix_cache_tokens_saved=11)
    neither = object()
    assert _prefix_tokens(mlx_like) == 7
    assert _prefix_tokens(metal_like) == 11
    assert _prefix_tokens(neither) == 0
    # The AsyncEngine wrapper is unwrapped the same way for both.
    assert _prefix_tokens(types.SimpleNamespace(engine=metal_like)) == 11


# -- the two surfaces answer the same pool conditions the same way -----------


def _both(app, model: str, *, stream: bool = False):
    """One request per surface for `model`, as (openai, anthropic) responses."""

    async def go():
        async with app.router.lifespan_context(app):
            r_o = await _post(app, "/v1/chat/completions", _chat(model, stream=stream))
            r_a = await _post(app, "/v1/messages", _messages(model, stream=stream))
            return r_o, r_a

    return asyncio.run(go())


def test_unknown_model_is_a_400_on_both_surfaces(pool_app):
    """The pool's "no such model" is a client error, and the OpenAI route says
    so with a 400 naming the available ids. /v1/messages let the same
    exception escape as a 500, so an Anthropic client typo'ing a model id was
    told the server was broken."""
    r_o, r_a = _both(pool_app, "nope")
    assert r_o.status_code == 400, r_o.text
    assert r_a.status_code == 400, r_a.text
    assert r_o.json()["detail"] == "unknown model 'nope'; available: a, b"
    assert r_a.json()["detail"] == r_o.json()["detail"]


def test_anthropic_stream_names_the_model_the_request_resolved_to(pool_app):
    """`message_start.message.model` must be the model the request picked, as
    the non-streaming body and the OpenAI stream already report. The streaming
    path was handed the registration-time name instead, which in pool mode is
    the first model in the directory, so a stream for "b" opened as "a"."""
    r_o, r_a = _both(pool_app, "b", stream=True)
    assert r_o.status_code == 200 and r_a.status_code == 200, r_a.text
    first = r_a.text.split("\n\n")[0]
    assert first.startswith("event: message_start")
    start = json.loads(first.split("\ndata: ", 1)[1])
    assert start["message"]["model"] == "b"
    chunk = json.loads(r_o.text.split("\n\n")[0][6:])
    assert chunk["model"] == start["message"]["model"]


# -- /v1/messages traffic reaches the server's stats ---------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_requests_are_counted_in_server_stats(pool_app, stream):
    """Every finished request feeds /admin/api/stats and the per-model usage
    breakdown. The Anthropic route recorded nothing, so a server serving only
    Anthropic clients reported zero requests and zero tokens."""
    stats = pool_app.state.stats
    stats.clear()

    async def go():
        async with pool_app.router.lifespan_context(pool_app):
            return await _post(pool_app, "/v1/messages", _messages("a", stream=stream))

    r = asyncio.run(go())
    assert r.status_code == 200, r.text
    snap = stats.snapshot()
    assert snap["total_requests"] == 1
    assert snap["total_prompt_tokens"] == 2  # "user: go"
    assert snap["total_completion_tokens"] == N_TOKENS
    assert snap["avg_generation_tps"] > 0
    by_id = {m["model_id"]: m for m in stats.usage()["models"]}
    assert by_id["a"]["requests"] == 1
    assert by_id["a"]["total_tokens"] == 2 + N_TOKENS


# -- an exposed profile's sampling defaults apply on /v1/messages too ----------


@pytest.fixture
def captured_params(pool_app):
    """The RequestParams each admitted request handed the engine, in order."""
    seen: list = []

    class _Capturing(_FakeRaw):
        def add_request(self, prompt, params, *, add_special=True) -> int:
            seen.append(params)
            return super().add_request(prompt, params, add_special=add_special)

    pool_app.state.pool._factory = lambda entry, _cfg: _Capturing(entry)
    return seen


def test_exposed_profile_is_a_default_on_the_anthropic_surface(pool_app, captured_params):
    """`a:precise` names model `a` plus a sampling bundle. The OpenAI route
    applies the bundle as DEFAULTS (an explicit request value still wins);
    the Anthropic route resolved the id, bound the overlay to a throwaway and
    sampled with the engine's defaults, so the profile a client picked from
    the model list did nothing there."""
    pool_app.state.profiles.create_profile(
        "a", "precise",
        settings={"temperature": 0.11, "top_k": 7, "top_p": 0.5},
        expose_as_model=True,
    )

    async def go():
        async with pool_app.router.lifespan_context(pool_app):
            r1 = await _post(pool_app, "/v1/chat/completions", _chat("a:precise"))
            r2 = await _post(pool_app, "/v1/messages", _messages("a:precise"))
            r3 = await _post(
                pool_app, "/v1/messages", {**_messages("a:precise"), "temperature": 0.9}
            )
            r4 = await _post(pool_app, "/v1/messages", _messages("a"))
            return r1, r2, r3, r4

    r1, r2, r3, r4 = asyncio.run(go())
    for r in (r1, r2, r3, r4):
        assert r.status_code == 200, r.text
    assert r2.json()["model"] == "a"
    openai, anthropic, explicit, plain = captured_params
    assert (anthropic.temp, anthropic.top_k, anthropic.top_p) == (0.11, 7, 0.5)
    assert (anthropic.temp, anthropic.top_k, anthropic.top_p) == (
        openai.temp, openai.top_k, openai.top_p
    )
    # A profile is a default, not an override.
    assert explicit.temp == pytest.approx(0.9)
    assert (explicit.top_k, explicit.top_p) == (7, 0.5)
    # And a plain id is untouched by it.
    assert plain.temp != pytest.approx(0.11)
