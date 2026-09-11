"""Phase 2: chat templating, the async engine bridge, and the HTTP surface."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

import freetoken_mac as ftm

MODEL_PATH = os.environ.get("FTM_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_TEST_MODEL to a .gguf path to run these",
)


@pytest.fixture(scope="module")
def model() -> ftm.Model:
    m = ftm.Model(MODEL_PATH, ftm.ModelParams())
    yield m
    # Release the weights before the interpreter exits. ggml frees the Metal device from
    # a static destructor and asserts its residency sets are empty; a model still alive
    # then abort()s the process (exit 134) after the suite has already passed.
    m.close()


# --- metadata strings longer than any fixed buffer ----------------------------------


def test_long_metadata_value_is_read_in_full(model: ftm.Model) -> None:
    """llama.cpp's string getters are snprintf-backed: they return the length the value
    WOULD need, not what was written. Reading that many bytes out of a fixed buffer is a
    stack over-read, and clamping to the buffer edge splits multi-byte UTF-8. Chat
    templates are multi-KiB, so this is the routine case, not an edge case."""
    tmpl = model.meta_val("tokenizer.chat_template")
    assert tmpl, "the test model is expected to embed a chat template"
    # Comfortably past the 1 KiB buffer the first implementation used.
    assert len(tmpl) > 1024
    # A truncated read would have split a UTF-8 sequence or lost the tail; a full read
    # round-trips and keeps the template's own closing structure.
    assert tmpl == model.chat_template
    assert tmpl.encode("utf-8").decode("utf-8") == tmpl


# --- chat templating ---------------------------------------------------------------


def test_apply_chat_template_renders_roles(model: ftm.Model) -> None:
    out = model.apply_chat_template(
        [("system", "You are terse."), ("user", "Hi")], True
    )
    assert "You are terse." in out
    assert "Hi" in out
    # add_assistant=True must leave the prompt ready for the model to continue from,
    # i.e. ending at the assistant turn rather than closing it.
    assert out.rstrip().endswith("assistant") or out.endswith("\n")

    without = model.apply_chat_template([("user", "Hi")], False)
    assert len(without) < len(out)


def test_apply_chat_template_rejects_empty(model: ftm.Model) -> None:
    with pytest.raises(ValueError, match="no messages"):
        model.apply_chat_template([], True)


# --- async engine bridge -----------------------------------------------------------


def _engine(model: ftm.Model, **kw: int) -> ftm.MetalEngine:
    cfg = ftm.EngineConfig(n_ctx=kw.get("n_ctx", 1024), n_batch=kw.get("n_batch", 256),
                           n_ubatch=kw.get("n_batch", 256), n_seq_max=kw.get("n_seq_max", 4))
    return ftm.MetalEngine(model, cfg)


def test_async_engine_streams_one_request(model: ftm.Model) -> None:
    from freetoken_mac.engine.async_engine import AsyncEngine

    async def run() -> list[str]:
        eng = AsyncEngine(_engine(model))
        await eng.start()
        try:
            rid = await eng.submit("The capital of France is", ftm.RequestParams(max_tokens=6))
            return [o.piece async for o in eng.stream(rid)]
        finally:
            await eng.stop()

    pieces = asyncio.run(run())
    assert pieces
    assert "".join(pieces).strip()


def test_async_engine_interleaves_concurrent_requests(model: ftm.Model) -> None:
    """Three coroutines streaming at once must each get their own tokens, and the
    engine must fold them into shared decodes rather than serialising."""
    from freetoken_mac.engine.async_engine import AsyncEngine

    prompts = ["The capital of France is", "Count: 1 2 3", "def add(a, b):"]

    async def run() -> tuple[list[str], int]:
        eng = AsyncEngine(_engine(model))
        await eng.start()
        try:
            rids = [
                await eng.submit(p, ftm.RequestParams(max_tokens=6, temp=0.0))
                for p in prompts
            ]

            async def collect(rid: int) -> str:
                return "".join([o.piece async for o in eng.stream(rid)])

            texts = await asyncio.gather(*(collect(r) for r in rids))
            return list(texts), eng.engine.ctx.decode_calls
        finally:
            await eng.stop()

    texts, decode_calls = asyncio.run(run())
    assert all(t.strip() for t in texts), f"a request produced nothing: {texts}"
    assert len(set(texts)) == len(texts), f"distinct prompts gave identical output: {texts}"
    # 3 requests x 6 tokens = 18 sequence-steps. Serialising would need ~18+ decodes;
    # batching folds them, so this must be well under that.
    assert decode_calls < 18, f"expected batched decodes, got {decode_calls}"


def test_async_engine_cancel_stops_a_stream(model: ftm.Model) -> None:
    from freetoken_mac.engine.async_engine import AsyncEngine

    async def run() -> int:
        eng = AsyncEngine(_engine(model))
        await eng.start()
        try:
            rid = await eng.submit("Count slowly:", ftm.RequestParams(max_tokens=64))
            seen = 0
            async for _ in eng.stream(rid):
                seen += 1
                if seen == 3:
                    await eng.cancel(rid)
            return seen
        finally:
            await eng.stop()

    seen = asyncio.run(run())
    # The stream must END after the cancel, well short of max_tokens=64.
    assert 3 <= seen < 64, f"cancel did not stop the stream: {seen} tokens"


def test_async_engine_survives_a_bad_request(model: ftm.Model) -> None:
    """A rejected request must not kill the worker thread -- the whole point of the
    single-process design is that one bad request cannot take the server down."""
    from freetoken_mac.engine.async_engine import AsyncEngine

    async def run() -> str:
        eng = AsyncEngine(_engine(model, n_ctx=512, n_seq_max=2))
        await eng.start()
        try:
            with pytest.raises(ValueError):
                await eng.submit([1] * 10_000, ftm.RequestParams(max_tokens=4))
            # The engine is still alive and serving.
            rid = await eng.submit("Hello", ftm.RequestParams(max_tokens=4))
            return "".join([o.piece async for o in eng.stream(rid)])
        finally:
            await eng.stop()

    assert asyncio.run(run()).strip()


# --- HTTP surface -------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(model: ftm.Model):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from freetoken_mac.server.app import build_app

    app = build_app(model, ftm.EngineConfig(n_ctx=1024, n_batch=256, n_ubatch=256, n_seq_max=4, engine="metal"))
    with fastapi_testclient.TestClient(app) as c:
        yield c
    # Leaving the context block ran the lifespan's shutdown, which closes the context.
    assert app.state.engine.engine.ctx.closed, "lifespan must close the context"


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_endpoint(client) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"] and body["data"][0]["object"] == "model"


def test_chat_completion_non_streaming(client) -> None:
    r = client.post("/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 8,
        "temperature": 0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"].strip()
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completion_streaming_sse(client) -> None:
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "Count: 1 2 3"}],
        "max_tokens": 6,
        "stream": True,
    }) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln]

    assert lines[-1] == "data: [DONE]", f"stream must end with [DONE], got {lines[-1]!r}"

    chunks = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # The first chunk announces the role; later chunks carry content deltas.
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.strip(), "stream produced no content"
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_streaming_and_non_streaming_agree(client) -> None:
    """Greedy decoding must give the same text through both paths -- otherwise one of
    them is mishandling the token stream."""
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    whole = client.post("/v1/chat/completions", json=payload).json()
    content = whole["choices"][0]["message"]["content"]

    with client.stream("POST", "/v1/chat/completions", json={**payload, "stream": True}) as r:
        lines = [ln for ln in r.iter_lines() if ln and ln != "data: [DONE]"]
    streamed = "".join(
        json.loads(ln[len("data: "):])["choices"][0]["delta"].get("content", "")
        for ln in lines
    )
    assert streamed == content


def test_bad_request_is_a_4xx_not_a_crash(client) -> None:
    # Missing messages entirely.
    assert client.post("/v1/chat/completions", json={"model": "local"}).status_code == 422
    # Empty message list.
    r = client.post("/v1/chat/completions", json={"model": "local", "messages": []})
    assert 400 <= r.status_code < 500, r.text
    # And the server is still healthy afterwards.
    assert client.get("/health").status_code == 200


# --- deterministic teardown ---------------------------------------------------------


def test_server_process_exits_cleanly() -> None:
    """Serve a request in a fresh process and require exit code 0.

    This is an at-exit behaviour, so it can only be tested by exiting: ggml frees the
    Metal device from a C++ static destructor and asserts its residency sets are empty
    (ggml-metal-device.m:1021). A context or model still holding GPU buffers at that
    moment abort()s with SIGABRT (134) -- AFTER a clean shutdown, so every request
    succeeds and the process still reports a crash to whatever supervises it.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        import freetoken_mac as ftm
        from freetoken_mac.server.app import build_app
        from fastapi.testclient import TestClient

        model = ftm.Model({MODEL_PATH!r}, ftm.ModelParams())
        app = build_app(model, ftm.EngineConfig(engine="metal"))
        with TestClient(app) as c:
            r = c.post("/v1/chat/completions", json={{
                "model": "m",
                "messages": [{{"role": "user", "content": "hi"}}],
                "max_tokens": 4,
            }})
            assert r.status_code == 200, r.text
        model.close()
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, timeout=300
    )
    assert proc.returncode == 0, (
        f"server process exited {proc.returncode} "
        f"(134 = SIGABRT, i.e. GPU resources outlived the Metal device)\n"
        f"{proc.stderr.decode()[-2000:]}"
    )


def test_closed_handles_raise_instead_of_crashing() -> None:
    """close() must make later use an exception, not a null-deref inside llama.cpp."""
    model = ftm.Model(MODEL_PATH, ftm.ModelParams())
    ctx = ftm.Context(model, ftm.ContextParams())

    ctx.close()
    assert ctx.closed
    ctx.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        ctx.decode_seq0([1])
    with pytest.raises(RuntimeError, match="closed"):
        _ = ctx.n_ctx

    model.close()
    assert model.closed
    model.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        model.tokenize("hi")
    # repr must stay safe on a closed handle -- debuggers and logs call it.
    assert "closed" in repr(model)
