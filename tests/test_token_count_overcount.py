"""RED test for completion_tokens overcount on natural end.

Baseline at 097b84f (and current HEAD) counts the terminal empty-piece eog frame
as a completion token. The engine does NOT count it (it retires before appending),
so `n_completion = len(pieces)` is off by one on every naturally-ended turn.

This test is the Prove-It pattern: it must FAIL before the fix and PASS after.

Bug: `python/bwr/server/app.py:310` and `anthropic_api.py:308`
increment `n_completion` for every `StepOutput`, including the final
`StepOutput(piece="", finished=True, finish_reason="eog")` whose piece is empty.
Observed: 10 vs engine `n_generated=9` (backlog).

Fix: do not count an empty terminal piece (or more generally, a finished
output whose piece is empty and reason is eog) toward `completion_tokens` /
`output_tokens` in both non-streaming and streaming paths, for both protocols.
"""

from __future__ import annotations

import asyncio
import json
import time
import types

import bwr
import pytest
from bwr.engine.metal_engine import StepOutput

# Use a tiny canned generation: two real pieces + one empty eog terminal.
# This mirrors the real engine's StepOutputs for a natural end.
CANNED_TEXT_PIECES = ["Hello", " world"]
EOG_TOKEN = 2  # arbitrary, is_eog check is on token id via model, but canned bypasses that


def _canned_eog_stream(text_pieces, reason="eog"):
    """Yield pieces, then an empty terminal frame with `reason`.

    The final yield is the bug trigger: empty piece, finished, eog.
    A correct server must NOT count this as a completion token.
    """
    async def stream(request_id: int):
        for i, piece in enumerate(text_pieces):
            # non-terminal pieces
            yield StepOutput(request_id, 100 + i, piece, False, None)
        # terminal empty piece — what the real engine does for eog
        yield StepOutput(request_id, EOG_TOKEN, "", True, reason)

    return stream


@pytest.fixture(scope="module")
def model():
    import os

    path = os.environ.get("BWR_TEST_MODEL")
    if not path:
        pytest.skip("BWR_TEST_MODEL not set")
    m = bwr.Model(path, bwr.ModelParams())
    yield m
    m.close()


@pytest.fixture(scope="module")
def served(model):
    tc = pytest.importorskip("fastapi.testclient")
    from bwr.server.app import build_app

    app = build_app(
        model, bwr.EngineConfig(n_ctx=4096, n_batch=512, n_ubatch=512, n_seq_max=2, engine="metal")
    )
    with tc.TestClient(app) as c:
        yield app, c
    assert app.state.engine.engine.ctx.closed


@pytest.fixture
def client(served):
    return served[1]


def test_openai_natural_end_does_not_count_empty_eog_token(served, monkeypatch) -> None:
    """OpenAI non-streaming: completion_tokens must be 2, not 3."""
    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", _canned_eog_stream(CANNED_TEXT_PIECES, "eog"))

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "dummy"}],
            "temperature": 0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 2 pieces of real content, terminal empty eog must not be counted
    assert body["usage"]["completion_tokens"] == len(CANNED_TEXT_PIECES), body["usage"]
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + len(CANNED_TEXT_PIECES)
    assert body["choices"][0]["finish_reason"] == "stop"
    # content is the joined pieces (empty terminal contributes nothing)
    assert body["choices"][0]["message"]["content"] == "".join(CANNED_TEXT_PIECES)


def test_openai_streaming_natural_end_does_not_count_empty_eog_token(served, monkeypatch) -> None:
    """OpenAI streaming: final usage must also be 2."""
    import json

    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", _canned_eog_stream(CANNED_TEXT_PIECES, "eog"))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "dummy"}],
            "temperature": 0,
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
    # find the last chunk that carries finish_reason
    final = [c for c in chunks if c["choices"][0].get("finish_reason") is not None]
    assert len(final) == 1
    # Streaming OpenAI does not emit usage in this impl, but the chunk's text
    # accumulation should be 2 tokens worth; we verify via the non-streaming path
    # above and via Anthropic streaming which does emit usage.
    assert final[0]["choices"][0]["finish_reason"] == "stop"


def test_anthropic_natural_end_does_not_count_empty_eog_token(served, monkeypatch) -> None:
    """Anthropic non-streaming: output_tokens must be 2, not 3."""
    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", _canned_eog_stream(CANNED_TEXT_PIECES, "eog"))

    r = client.post(
        "/v1/messages",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "dummy"}],
            "max_tokens": 64,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["usage"]["output_tokens"] == len(CANNED_TEXT_PIECES), body["usage"]
    assert body["stop_reason"] == "end_turn"


def test_anthropic_streaming_natural_end_does_not_count_empty_eog_token(served, monkeypatch) -> None:
    """Anthropic streaming: message_delta usage must be 2."""
    import json

    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", _canned_eog_stream(CANNED_TEXT_PIECES, "eog"))

    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "dummy"}],
            "max_tokens": 64,
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        raw = [ln for ln in r.iter_lines() if ln]
    datas = [json.loads(ln[len("data: "):]) for ln in raw if ln.startswith("data: ")]
    deltas = [d for d in datas if d.get("type") == "message_delta"]
    assert len(deltas) == 1
    assert deltas[0]["usage"]["output_tokens"] == len(CANNED_TEXT_PIECES), deltas[0]["usage"]
    assert deltas[0]["delta"]["stop_reason"] == "end_turn"


def test_cap_path_still_counts_exactly(served, monkeypatch) -> None:
    """Positive control: a cap-bound generation MUST still count all tokens.

    The bug is only for empty eog; a length retirement has a non-empty piece and
    must be counted. This guards the fix from under-counting.
    """

    def canned_length(request_id: int):
        async def stream(rid: int):
            # 5 tokens, last one retires for length with non-empty piece
            for i in range(4):
                yield StepOutput(rid, 100 + i, f"t{i}", False, None)
            yield StepOutput(rid, 104, "t4", True, "length")

        return stream

    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", canned_length(0))

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "dummy"}],
            "temperature": 0,
        },
    )
    assert r.status_code == 200
    assert r.json()["usage"]["completion_tokens"] == 5
    assert r.json()["choices"][0]["finish_reason"] == "length"

    monkeypatch.setattr(app.state.engine, "stream", canned_length(0))
    r = client.post(
        "/v1/messages",
        json={"model": "local", "messages": [{"role": "user", "content": "dummy"}], "max_tokens": 64},
    )
    assert r.status_code == 200
    assert r.json()["usage"]["output_tokens"] == 5


# -- an eog retire that flushed a held partial character ----------------------
#
# Both engines retire on eog WITHOUT appending the eog token to output_tokens
# (mlx_engine._feed, metal_engine._advance), but the retire's piece is whatever
# `_flush_piece` held: a character spelled over several tokens that was cut
# short by the eog comes out as "�". The old predicate counted the retire
# whenever its piece was non-empty, so such a turn reported one token more
# than the engine generated. The rule is: an eog retire is never a generated
# token, whatever its piece. These tests need no model: a fake raw engine
# behind the pool feeds the frames.

FLUSHED_PIECES = ["Hel", "lo"]
FLUSHED_RETIRE = "�"


class _FlushingEog:
    """A raw engine that yields `FLUSHED_PIECES`, then an eog retire carrying
    a replacement character (a held partial character the eog cut short)."""

    def __init__(self, entry) -> None:
        self.entry = entry
        self._next = 0
        self._emitted: dict[int, int] = {}
        self.ctx = types.SimpleNamespace(
            close=lambda: None, n_ctx=4096, n_ctx_seq=4096, n_seq_max=8,
            decode_calls=0,
        )

    def apply_chat_template(self, pairs, add_generation_prompt):
        return "\n".join(f"{r}: {t}" for r, t in pairs)

    def tokenize(self, prompt, *, add_special=True, parse_special=True):
        return list(range(len(prompt.split())))

    def add_request(self, prompt, params, *, add_special=True) -> int:
        rid = self._next
        self._next += 1
        self._emitted[rid] = 0
        return rid

    @property
    def has_work(self) -> bool:
        return bool(self._emitted)

    def step(self) -> list[StepOutput]:
        time.sleep(0.002)
        out: list[StepOutput] = []
        for rid in list(self._emitted):
            n = self._emitted[rid]
            self._emitted[rid] = n + 1
            if n < len(FLUSHED_PIECES):
                out.append(StepOutput(rid, 100 + n, FLUSHED_PIECES[n], False, None))
            else:
                out.append(StepOutput(rid, EOG_TOKEN, FLUSHED_RETIRE, True, "eog"))
                del self._emitted[rid]
        return out

    def cancel(self, rid: int) -> bool:
        return self._emitted.pop(rid, None) is not None


@pytest.fixture
def flushing_app(tmp_path):
    pytest.importorskip("httpx")
    from bwr.server.app import build_app

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.json").write_text("{}")
    app = build_app(None, bwr.EngineConfig(), model_dir=str(tmp_path))
    app.state.pool._factory = lambda entry, _cfg: _FlushingEog(entry)
    return app


def _post_in_lifespan(app, path: str, body: dict):
    import httpx

    async def go():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.post(path, json=body)

    return asyncio.run(go())


def _body(*, stream: bool) -> dict:
    return {
        "model": "a",
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 64,
        "stream": stream,
    }


def test_openai_flushed_eog_retire_is_not_a_completion_token(flushing_app) -> None:
    r = _post_in_lifespan(flushing_app, "/v1/chat/completions", _body(stream=False))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["usage"]["completion_tokens"] == len(FLUSHED_PIECES), body["usage"]
    # The flushed remainder still reaches the body: it is text, not a token.
    assert body["choices"][0]["message"]["content"] == "".join(FLUSHED_PIECES) + FLUSHED_RETIRE
    assert body["choices"][0]["finish_reason"] == "stop"


def test_openai_stream_flushed_eog_retire_is_not_a_completion_token(flushing_app) -> None:
    """The OpenAI stream emits no usage block; its count reaches the server
    stats, which is where a dashboard would see the extra token."""
    stats = flushing_app.state.stats
    stats.clear()
    r = _post_in_lifespan(flushing_app, "/v1/chat/completions", _body(stream=True))
    assert r.status_code == 200, r.text
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == "".join(FLUSHED_PIECES) + FLUSHED_RETIRE
    assert stats.snapshot()["total_completion_tokens"] == len(FLUSHED_PIECES)


def test_anthropic_flushed_eog_retire_is_not_an_output_token(flushing_app) -> None:
    r = _post_in_lifespan(flushing_app, "/v1/messages", _body(stream=False))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["usage"]["output_tokens"] == len(FLUSHED_PIECES), body["usage"]
    assert body["content"][0]["text"] == "".join(FLUSHED_PIECES) + FLUSHED_RETIRE
    assert body["stop_reason"] == "end_turn"


def test_anthropic_stream_flushed_eog_retire_is_not_an_output_token(flushing_app) -> None:
    r = _post_in_lifespan(flushing_app, "/v1/messages", _body(stream=True))
    assert r.status_code == 200, r.text
    datas = [
        json.loads(ln[len("data: "):])
        for ln in r.text.splitlines()
        if ln.startswith("data: ")
    ]
    deltas = [d for d in datas if d.get("type") == "message_delta"]
    assert len(deltas) == 1
    assert deltas[0]["usage"]["output_tokens"] == len(FLUSHED_PIECES), deltas[0]["usage"]


def test_bench_row_flushed_eog_retire_is_not_a_generated_token(flushing_app) -> None:
    """The bench row counts with the same rule. The count is not reported
    directly, but `gen_tps = n / decode_s` and `tpot_ms = decode_s / (n - 1)`
    share the one `decode_s`, so their product is `n / (n - 1)` whatever the
    timing: 2.0 for the two generated tokens, 1.5 if the retire were counted.
    """

    async def go():
        app = flushing_app
        async with app.router.lifespan_context(app):
            bench = app.state.bench
            started = bench.start("a", prompt_lengths=[8], generation_length=8, batch_sizes=[1])
            for _ in range(400):
                body = bench.get(started["bench_id"])
                if body["status"] != "running":
                    break
                await asyncio.sleep(0.005)
            return bench.get(started["bench_id"])

    run = asyncio.run(go())
    assert run["status"] == "completed", run
    row = run["results"][0]
    assert row["error"] is None, row
    n = len(FLUSHED_PIECES)
    assert row["gen_tps"] * row["tpot_ms"] / 1000 == pytest.approx(n / (n - 1)), row
