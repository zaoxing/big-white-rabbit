"""RED test for completion_tokens overcount on natural end.

Baseline at 097b84f (and current HEAD) counts the terminal empty-piece eog frame
as a completion token. The engine does NOT count it (it retires before appending),
so `n_completion = len(pieces)` is off by one on every naturally-ended turn.

This test is the Prove-It pattern: it must FAIL before the fix and PASS after.

Bug: `python/freetoken_mac/server/app.py:310` and `anthropic_api.py:308`
increment `n_completion` for every `StepOutput`, including the final
`StepOutput(piece="", finished=True, finish_reason="eog")` whose piece is empty.
Observed: 10 vs engine `n_generated=9` (backlog).

Fix: do not count an empty terminal piece (or more generally, a finished
output whose piece is empty and reason is eog) toward `completion_tokens` /
`output_tokens` in both non-streaming and streaming paths, for both protocols.
"""

from __future__ import annotations

import pytest

import freetoken_mac as ftm
from freetoken_mac.engine.metal_engine import StepOutput

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

    path = os.environ.get("FTM_TEST_MODEL")
    if not path:
        pytest.skip("FTM_TEST_MODEL not set")
    m = ftm.Model(path, ftm.ModelParams())
    yield m
    m.close()


@pytest.fixture(scope="module")
def served(model):
    tc = pytest.importorskip("fastapi.testclient")
    from freetoken_mac.server.app import build_app

    app = build_app(
        model, ftm.EngineConfig(n_ctx=4096, n_batch=512, n_ubatch=512, n_seq_max=2, engine="metal")
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
