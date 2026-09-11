"""A request that ends abnormally must be RETIRED, not merely forgotten.

`AsyncEngine.release()` used to only pop the delivery channel (`_streams`). But the
seq_id -- the finite resource, `n_seq_max` of them -- is returned to the pool by
`MetalEngine._retire` and by nothing else. So a request whose stream ended in an error,
or whose handler was killed mid-flight, stayed in `MetalEngine._states` unfinished:

  * its slot was gone for the life of the process, and
  * `MetalEngine.has_work` is "any request not finished", so the worker thread kept
    re-running the same failing `step()` forever -- one core pinned, tens of thousands
    of wasted `llama_decode` calls, and every later request eventually 503-ing.

These tests force the abnormal exit by injecting a failing `step`, which is the only way
to make the assertion deterministic: with the failure still injected NOTHING can retire
normally, so `free_seq_slots` returning to `n_seq_max` can only mean the request was
actually cancelled.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

import freetoken_mac as ftm

MODEL_PATH = os.environ.get("FTM_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_TEST_MODEL to a .gguf path to run these",
)


def _boom() -> None:
    """Stands in for a step that cannot succeed (a KV-slot failure, a policy overrun).

    Bound as an INSTANCE attribute, so it shadows `MetalEngine.step` for one engine only
    and takes no `self`.
    """
    raise RuntimeError("injected step failure")


@pytest.fixture(scope="module")
def model() -> ftm.Model:
    m = ftm.Model(MODEL_PATH, ftm.ModelParams())
    yield m
    # Release the weights before interpreter exit or ggml's Metal device destructor
    # aborts the process (exit 134). See docs/llamacpp-notes.md.
    m.close()


# --- the engine seam ----------------------------------------------------------------


def test_release_cancels_a_request_the_engine_never_retired(model: ftm.Model) -> None:
    """The unit-level statement of the bug: release() on an errored request must retire it."""
    from freetoken_mac.engine.async_engine import AsyncEngine
    from freetoken_mac.engine.metal_engine import MetalEngine

    async def run() -> tuple[int, int, str | None]:
        engine = MetalEngine(
            model, ftm.EngineConfig(n_ctx=1024, n_batch=256, n_ubatch=256, n_seq_max=2)
        )
        eng = AsyncEngine(engine)
        await eng.start()
        try:
            engine.step = _boom
            rid = await eng.submit("Count slowly:", ftm.RequestParams(max_tokens=64))
            assert engine.n_free_seq_slots == engine.ctx.n_seq_max - 1, "admission took no slot"

            with pytest.raises(RuntimeError):
                async for _ in eng.stream(rid):
                    pass

            eng.release(rid)
            # The cancel is a command to the worker thread, so give it a bounded window.
            # It cannot be satisfied by the request completing: step() still raises.
            for _ in range(200):
                if engine.n_free_seq_slots == engine.ctx.n_seq_max:
                    break
                await asyncio.sleep(0.02)
            return (
                engine.n_free_seq_slots,
                engine.ctx.n_seq_max,
                engine.state(rid).finish_reason,
            )
        finally:
            await eng.stop()

    free, n_seq_max, reason = asyncio.run(run())
    assert free == n_seq_max, f"release() leaked the slot: {free}/{n_seq_max} free"
    assert reason == "cancelled", f"request was not retired (finish_reason={reason!r})"


def test_release_after_normal_completion_does_not_cancel(model: ftm.Model) -> None:
    """The other half of the contract: a request that finished on its own is already
    retired, so release() must NOT post a cancel for it -- its recorded reason has to
    stay the engine's own ("length" here), not become "cancelled"."""
    from freetoken_mac.engine.async_engine import AsyncEngine
    from freetoken_mac.engine.metal_engine import MetalEngine

    async def run() -> tuple[int, int, str | None]:
        engine = MetalEngine(
            model, ftm.EngineConfig(n_ctx=1024, n_batch=256, n_ubatch=256, n_seq_max=2)
        )
        eng = AsyncEngine(engine)
        await eng.start()
        try:
            rid = await eng.submit(
                "The capital of France is", ftm.RequestParams(max_tokens=4, temp=0.0)
            )
            async for _ in eng.stream(rid):
                pass
            eng.release(rid)
            await asyncio.sleep(0.2)  # a stray cancel would have landed by now
            return (
                engine.n_free_seq_slots,
                engine.ctx.n_seq_max,
                engine.state(rid).finish_reason,
            )
        finally:
            await eng.stop()

    free, n_seq_max, reason = asyncio.run(run())
    assert free == n_seq_max
    assert reason == "length", f"normal completion was re-labelled {reason!r}"


# --- both HTTP surfaces -------------------------------------------------------------


@pytest.fixture(scope="module")
def served(model: ftm.Model):
    tc = pytest.importorskip("fastapi.testclient")
    from freetoken_mac.server.app import build_app

    app = build_app(
        model, ftm.EngineConfig(n_ctx=2048, n_batch=256, n_ubatch=256, n_seq_max=4, engine="metal")
    )
    # raise_server_exceptions=False: the engine failure must surface the way a real
    # client sees it (a 500), instead of being re-raised into the test body.
    with tc.TestClient(app, raise_server_exceptions=False) as c:
        yield app, c
    assert app.state.engine.engine.ctx.closed, "lifespan must close the context"


def _request(surface: str, *, stream: bool) -> tuple[str, dict]:
    if surface == "openai":
        return "/v1/chat/completions", {
            "model": "local",
            "messages": [{"role": "user", "content": "Count: 1 2 3"}],
            "max_tokens": 32,
            "stream": stream,
        }
    return "/v1/messages", {
        "model": "local",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Count: 1 2 3"}],
        "stream": stream,
    }


def _free_slots(client) -> int:
    return client.get("/health").json()["free_seq_slots"]


def _wait_for_slots(client, expected: int, timeout: float = 10.0) -> int:
    """Poll /health until every slot is back, or give up and report what we saw."""
    deadline = time.monotonic() + timeout
    seen = _free_slots(client)
    while seen != expected and time.monotonic() < deadline:
        time.sleep(0.05)
        seen = _free_slots(client)
    return seen


def _serve_one(client, surface: str) -> None:
    """A plain request must still be served after the failure is lifted."""
    url, payload = _request(surface, stream=False)
    payload["max_tokens"] = 4
    r = client.post(url, json=payload)
    assert r.status_code == 200, f"engine unusable after the failure: {r.text}"


@pytest.mark.parametrize("surface", ["openai", "anthropic"])
def test_errored_non_streaming_request_frees_its_slot(served, surface, monkeypatch) -> None:
    app, client = served
    engine = app.state.engine.engine
    n_seq_max = engine.ctx.n_seq_max
    assert _wait_for_slots(client, n_seq_max) == n_seq_max, "test began with a leaked slot"

    monkeypatch.setattr(engine, "step", _boom)
    url, payload = _request(surface, stream=False)
    r = client.post(url, json=payload)
    assert r.status_code == 500, f"expected the injected failure to surface: {r.status_code}"

    # Asserted while step() still raises, so the only way back to n_seq_max is a cancel.
    free = _wait_for_slots(client, n_seq_max)
    assert free == n_seq_max, f"{surface} leaked a slot: {free}/{n_seq_max} free"

    monkeypatch.undo()
    _serve_one(client, surface)


@pytest.mark.parametrize("surface", ["openai", "anthropic"])
def test_errored_streaming_request_frees_its_slot(served, surface, monkeypatch) -> None:
    app, client = served
    engine = app.state.engine.engine
    n_seq_max = engine.ctx.n_seq_max
    assert _wait_for_slots(client, n_seq_max) == n_seq_max, "test began with a leaked slot"

    monkeypatch.setattr(engine, "step", _boom)
    url, payload = _request(surface, stream=True)
    try:
        with client.stream("POST", url, json=payload) as r:
            for _ in r.iter_lines():
                pass
    except Exception:
        # The injected failure tears the response down mid-body (the headers and the
        # opening frame are already sent). That broken stream IS the abnormal exit under
        # test; what must hold is the state it leaves behind, asserted below.
        pass

    free = _wait_for_slots(client, n_seq_max)
    assert free == n_seq_max, f"{surface} stream leaked a slot: {free}/{n_seq_max} free"

    monkeypatch.undo()
    _serve_one(client, surface)
