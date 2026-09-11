"""`stop` (OpenAI) and `stop_sequences` (Anthropic) must actually stop the turn.

Both schemas accepted the field and both dropped it. Verified on the running server
before this file existed: `{"stop": ["4"], "max_tokens": 30}` on `/v1/chat/completions`
came back as `'1\\n2\\n3\\n4\\n5\\n6\\n7\\n8'` -- the delimiter and everything past it --
with `finish_reason: "stop"`, and `{"stop_sequences": ["Paris"]}` on `/v1/messages` came
back as `'Paris, the city of'` with `stop_reason: "max_tokens"`. An agent loop that
delimits its turns with a stop string therefore mis-parses every turn, and pays for the
tokens it then throws away.

What is asserted here, and why each assertion is the one that would have failed:

  * the delimiter is absent from the response and the reason is right, on BOTH surfaces
    and on BOTH the streaming and non-streaming path;
  * Anthropic's `stop_sequence` field NAMES the match (`anthropic_schemas` has always
    declared `stop_reason: "stop_sequence"` as legal, and nothing could produce it);
  * a delimiter SPLIT ACROSS TOKEN PIECES is still caught -- the tokeniser does not
    align to it, so this is the case a per-piece implementation gets wrong while every
    single-piece test passes;
  * streamed text equals non-streamed text for the same request, which is what a
    hold-back bug breaks (a piece already sent cannot be un-sent);
  * a stop string that never appears changes nothing, byte for byte;
  * generation STOPS rather than being truncated -- asserted on the engine's own
    retirement reason, its token count and `ctx.decode_calls`, because "stopped paying"
    is the actual point and no string assertion can see it.

Generations are CANNED for the wire-format cases (the engine's `stream` is replaced, as
`test_anthropic_tool_parsing_off.py` does): whether the 0.5B model emits a given
delimiter is a coin flip, and these are statements about the routes. The engine and
end-to-end cases below use the real model, with the expected text DERIVED from the
model's own unconstrained greedy output rather than hardcoded, so they assert against
the authority instead of against a transcription of it.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

import freetoken_mac as ftm
from freetoken_mac.engine.config import (
    StopSequenceFilter,
    first_stop_match,
    normalize_stops,
    partial_stop_len,
)

MODEL_PATH = os.environ.get("FTM_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_TEST_MODEL to a .gguf path to run these",
)

# The engine's own retirement reason for "the client's delimiter arrived". Named once
# here because BOTH protocol maps have to learn it: a reason mapped in one and not the
# other is a response whose stop_reason contradicts its text.
ENGINE_REASON = "stop_sequence"

# A ReAct-shaped turn: the delimiter is multi-character and mid-text, which is what an
# agent loop actually sends.
DELIMITER = "\nObservation:"
CANNED = "Thought: I should look it up.\nObservation: rain\nThought: done"
EXPECTED = "Thought: I should look it up."
assert CANNED.startswith(EXPECTED) and DELIMITER in CANNED  # keeps the fixture honest


# --- the shared rule (no model, no server) ------------------------------------------


def test_normalize_stops_accepts_both_spellings_and_drops_the_unusable() -> None:
    assert normalize_stops(None) == ()
    # OpenAI allows a bare string as well as a list, and means the same by both.
    assert normalize_stops("STOP") == ("STOP",)
    assert normalize_stops(["a", "b"]) == ("a", "b")
    assert normalize_stops([]) == ()
    # An empty string sits at offset 0 of every text: honouring it would truncate every
    # response to nothing, so it is dropped rather than obeyed.
    assert normalize_stops("") == ()
    assert normalize_stops(["", "a", ""]) == ("a",)
    # Duplicates cannot change where generation ends; order is otherwise preserved so
    # the reported match is stable.
    assert normalize_stops(["b", "a", "b"]) == ("b", "a")


def test_first_stop_match_picks_the_sequence_that_completes_earliest() -> None:
    assert first_stop_match("hello", ()) is None
    assert first_stop_match("hello", ("zz",)) is None

    m = first_stop_match("one two three", ("two",))
    assert (m.text, m.start, m.end) == ("two", 4, 7)

    # Earliest end, not earliest start: at the character that completed "ab", "abc" was
    # still unwritten, so the model would never have been asked for the "c".
    assert first_stop_match("xabc", ("abc", "ab")).text == "ab"
    # Multiple delimiters, the earliest match wins whatever order they were given in.
    assert first_stop_match("aaa STOP bbb END", ("END", "STOP")).text == "STOP"
    assert first_stop_match("aaa END bbb STOP", ("STOP", "END")).text == "END"


def test_partial_stop_len_holds_back_exactly_the_undecidable_tail() -> None:
    assert partial_stop_len("", ("ab",)) == 0
    assert partial_stop_len("xya", ("ab",)) == 1  # "a" may still become "ab"
    assert partial_stop_len("xyab", ("ab",)) == 0  # a complete match is not "partial"
    assert partial_stop_len("x\nObserv", (DELIMITER,)) == len("\nObserv")
    # The longest pending prefix across all sequences, since any of them could complete.
    assert partial_stop_len("zzab", ("abc", "zzz")) == 2


def test_filter_detects_a_delimiter_split_across_pieces_and_leaks_nothing() -> None:
    """The case a per-piece implementation gets wrong.

    Fed one character at a time -- the worst tokenisation there is -- the filter must
    never emit a byte of the delimiter, and must stop exactly at it.
    """
    f = StopSequenceFilter((DELIMITER,))
    emitted: list[str] = []
    stopped_at = None
    for i, ch in enumerate(CANNED):
        text, hit = f.push(ch)
        emitted.append(text)
        if hit:
            stopped_at = i
            break
    assert stopped_at is not None, "the delimiter was never detected"
    assert "".join(emitted) == EXPECTED
    assert f.stopped and f.matched == DELIMITER
    # It stopped at the character that COMPLETED the delimiter, not later.
    assert stopped_at == CANNED.index(DELIMITER) + len(DELIMITER) - 1
    # Nothing more is emitted once the turn has ended, and the flush adds nothing.
    assert f.push(" more text") == ("", False)
    assert f.flush() == ""


def test_filter_releases_a_partial_that_never_completes() -> None:
    f = StopSequenceFilter((DELIMITER,))
    assert f.push("done.\nObs") == ("done.", False), "the tail is still undecidable"
    assert f.flush() == "\nObs", "a partial that never completed is real content"
    assert not f.stopped and f.matched is None


def test_filter_without_stops_is_a_byte_for_byte_pass_through() -> None:
    """The wire-compatibility guarantee: no `stop` means no behaviour change at all."""
    f = StopSequenceFilter()
    assert f.enabled is False
    for piece in ("\nObs", "ervation", ":", ""):
        assert f.push(piece) == (piece, False)
    assert f.flush() == ""
    assert not f.stopped and f.matched is None


def test_both_protocol_maps_learned_the_new_engine_reason() -> None:
    """A new engine reason mapped on one surface only is a silent protocol divergence."""
    from freetoken_mac.server.anthropic_api import _STOP_REASONS
    from freetoken_mac.server.app import _FINISH_REASONS

    assert _FINISH_REASONS[ENGINE_REASON] == "stop"
    assert _STOP_REASONS[ENGINE_REASON] == "stop_sequence"


# --- fixtures -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> ftm.Model:
    m = ftm.Model(MODEL_PATH, ftm.ModelParams())
    yield m
    # Release the weights before interpreter exit or ggml's Metal device destructor
    # aborts the process (exit 134). See docs/llamacpp-notes.md.
    m.close()


@pytest.fixture(scope="module")
def served(model: ftm.Model):
    tc = pytest.importorskip("fastapi.testclient")
    from freetoken_mac.server.app import build_app

    app = build_app(
        model, ftm.EngineConfig(n_ctx=4096, n_batch=512, n_ubatch=512, n_seq_max=2, engine="metal")
    )
    with tc.TestClient(app) as c:
        yield app, c
    assert app.state.engine.engine.ctx.closed, "lifespan must close the context"


def _canned_stream(text: str, *, chunk: int = 3):
    """Replace AsyncEngine.stream with one replaying `text` in fixed-size pieces.

    `chunk=3` cuts the 13-character delimiter into five pieces at offsets nothing in the
    server can predict, which is how a real tokeniser delivers it.
    """
    from freetoken_mac.engine.metal_engine import StepOutput

    pieces = [text[i : i + chunk] for i in range(0, len(text), chunk)] or [""]

    async def stream(request_id: int):
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            yield StepOutput(request_id, 0, piece, last, "eog" if last else None)

    return stream


@pytest.fixture
def canned(served, monkeypatch):
    """Every request served while this is active replays CANNED in 3-char pieces."""
    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", _canned_stream(CANNED))
    return client


# --- wire helpers -------------------------------------------------------------------


def _oai_body(client, **extra) -> dict:
    r = client.post("/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "Look up the weather."}],
        "max_tokens": 64,
        "temperature": 0,
        **extra,
    })
    assert r.status_code == 200, r.text
    return r.json()


def _oai_chunks(client, **extra) -> list[dict]:
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "Look up the weather."}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
        **extra,
    }) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln]
    assert lines[-1] == "data: [DONE]", lines[-3:]
    return [json.loads(ln[len("data: "):]) for ln in lines[:-1]]


def _oai_streamed_text(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)


def _msg_body(client, **extra) -> dict:
    r = client.post("/v1/messages", json={
        "model": "local",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Look up the weather."}],
        "temperature": 0,
        **extra,
    })
    assert r.status_code == 200, r.text
    return r.json()


def _msg_events(client, **extra) -> tuple[list[str], list[dict]]:
    with client.stream("POST", "/v1/messages", json={
        "model": "local",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Look up the weather."}],
        "temperature": 0,
        "stream": True,
        **extra,
    }) as r:
        assert r.status_code == 200
        raw = [ln for ln in r.iter_lines() if ln]
    names = [ln[len("event: "):] for ln in raw if ln.startswith("event: ")]
    datas = [json.loads(ln[len("data: "):]) for ln in raw if ln.startswith("data: ")]
    return names, datas


def _msg_streamed_text(datas: list[dict]) -> str:
    return "".join(
        d["delta"]["text"] for d in datas
        if d.get("type") == "content_block_delta"
        and d["delta"].get("type") == "text_delta"
    )


def _msg_text(body: dict) -> str:
    return "".join(b["text"] for b in body["content"] if b["type"] == "text")


# --- OpenAI surface -----------------------------------------------------------------


def test_openai_non_streaming_truncates_at_the_delimiter(canned) -> None:
    body = _oai_body(canned, stop=[DELIMITER])
    choice = body["choices"][0]
    assert choice["message"]["content"] == EXPECTED
    assert DELIMITER not in choice["message"]["content"]
    assert "Observation" not in choice["message"]["content"]
    assert choice["finish_reason"] == "stop"


def test_openai_streaming_truncates_at_the_delimiter(canned) -> None:
    chunks = _oai_chunks(canned, stop=[DELIMITER])
    assert _oai_streamed_text(chunks) == EXPECTED
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # No frame may carry any part of the delimiter, not even a prefix of it: the client
    # already has every frame that was sent.
    for c in chunks:
        content = c["choices"][0]["delta"].get("content", "")
        assert "\n" not in content, f"a delimiter fragment reached the client: {content!r}"


def test_openai_stop_may_be_a_bare_string(canned) -> None:
    """OpenAI types `stop` as `string | string[]`; both mean the same thing."""
    as_string = _oai_body(canned, stop=DELIMITER)
    as_list = _oai_body(canned, stop=[DELIMITER])
    assert as_string["choices"][0]["message"]["content"] == EXPECTED
    assert (
        as_string["choices"][0]["message"]["content"]
        == as_list["choices"][0]["message"]["content"]
    )
    assert as_string["choices"][0]["finish_reason"] == "stop"


def test_openai_earliest_of_several_stops_wins(canned) -> None:
    """Two delimiters, both present: the turn ends at the first one to complete."""
    later = "\nThought: done"
    assert CANNED.index(DELIMITER) < CANNED.index(later)
    for order in ([DELIMITER, later], [later, DELIMITER]):
        body = _oai_body(canned, stop=order)
        assert body["choices"][0]["message"]["content"] == EXPECTED, order


def test_openai_stop_that_never_appears_changes_nothing(canned) -> None:
    """The wire-compatibility guarantee, checked against the same generation."""
    without = _oai_body(canned)
    with_stop = _oai_body(canned, stop=["\nNEVER APPEARS:"])
    for key in ("message", "finish_reason"):
        assert with_stop["choices"][0][key] == without["choices"][0][key]
    assert with_stop["choices"][0]["message"]["content"] == CANNED
    assert with_stop["usage"]["completion_tokens"] == without["usage"]["completion_tokens"]

    s_without = _oai_streamed_text(_oai_chunks(canned))
    s_with = _oai_streamed_text(_oai_chunks(canned, stop=["\nNEVER APPEARS:"]))
    assert s_with == s_without == CANNED


def test_openai_streaming_equals_non_streaming(canned) -> None:
    body = _oai_body(canned, stop=[DELIMITER])
    chunks = _oai_chunks(canned, stop=[DELIMITER])
    assert _oai_streamed_text(chunks) == body["choices"][0]["message"]["content"]
    assert (
        chunks[-1]["choices"][0]["finish_reason"] == body["choices"][0]["finish_reason"]
    )


# --- Anthropic surface --------------------------------------------------------------


def test_anthropic_non_streaming_truncates_and_names_the_match(canned) -> None:
    body = _msg_body(canned, stop_sequences=[DELIMITER])
    assert _msg_text(body) == EXPECTED
    assert body["stop_reason"] == "stop_sequence"
    # The field the client reads to learn WHICH delimiter ended the turn. It was always
    # null before, including on the turns that ran past a delimiter.
    assert body["stop_sequence"] == DELIMITER


def test_anthropic_streaming_truncates_and_names_the_match(canned) -> None:
    _names, datas = _msg_events(canned, stop_sequences=[DELIMITER])
    assert _msg_streamed_text(datas) == EXPECTED
    delta = [d for d in datas if d.get("type") == "message_delta"][-1]
    assert delta["delta"]["stop_reason"] == "stop_sequence"
    assert delta["delta"]["stop_sequence"] == DELIMITER
    for d in datas:
        if d.get("type") == "content_block_delta" and d["delta"].get("type") == "text_delta":
            assert "\n" not in d["delta"]["text"], d


def test_anthropic_streaming_equals_non_streaming(canned) -> None:
    body = _msg_body(canned, stop_sequences=[DELIMITER])
    names, datas = _msg_events(canned, stop_sequences=[DELIMITER])
    assert _msg_streamed_text(datas) == _msg_text(body)
    delta = [d for d in datas if d.get("type") == "message_delta"][-1]
    assert delta["delta"]["stop_reason"] == body["stop_reason"]
    assert delta["delta"]["stop_sequence"] == body["stop_sequence"]
    assert names[-1] == "message_stop", names


def test_anthropic_stop_that_never_appears_changes_nothing(canned) -> None:
    without = _msg_body(canned)
    with_stop = _msg_body(canned, stop_sequences=["\nNEVER APPEARS:"])
    assert _msg_text(with_stop) == _msg_text(without) == CANNED
    assert with_stop["stop_reason"] == without["stop_reason"] == "end_turn"
    # Still null when nothing matched -- the field only ever names a real match.
    assert with_stop["stop_sequence"] is None
    assert with_stop["usage"]["output_tokens"] == without["usage"]["output_tokens"]

    _n, datas = _msg_events(canned, stop_sequences=["\nNEVER APPEARS:"])
    assert _msg_streamed_text(datas) == CANNED
    delta = [d for d in datas if d.get("type") == "message_delta"][-1]
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["delta"]["stop_sequence"] is None


def test_both_surfaces_truncate_the_same_generation_identically(canned) -> None:
    oai = _oai_body(canned, stop=[DELIMITER])["choices"][0]["message"]["content"]
    anth = _msg_text(_msg_body(canned, stop_sequences=[DELIMITER]))
    assert oai == anth == EXPECTED


# --- the delimiter must survive the tool parser sitting on the same stream ----------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
CALL_CANNED = (
    'Thought: look it up.\n<tool_call>\n{"name": "get_weather", '
    '"arguments": {"city": "London"}}\n</tool_call>\nObservation: rain'
)


def test_stop_filter_composes_with_the_tool_parser(served, monkeypatch) -> None:
    """Both stages withhold text on the same stream; the composition must hold.

    The delimiter sits AFTER a complete tool call, so the call must still be parsed and
    the trailing delimiter must still cut the turn -- and neither stage may leak the
    text the other was holding.
    """
    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", _canned_stream(CALL_CANNED))

    body = _oai_body(client, stop=[DELIMITER], tools=[WEATHER_TOOL])
    choice = body["choices"][0]
    calls = choice["message"]["tool_calls"]
    assert calls and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "London"}
    assert choice["message"]["content"] == "Thought: look it up."
    assert "Observation" not in (choice["message"]["content"] or "")
    assert choice["finish_reason"] == "tool_calls"

    chunks = _oai_chunks(client, stop=[DELIMITER], tools=[WEATHER_TOOL])
    assert _oai_streamed_text(chunks) == choice["message"]["content"]
    streamed_calls = [
        tc for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert [tc["function"]["name"] for tc in streamed_calls] == ["get_weather"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_a_delimiter_inside_a_tool_call_still_ends_the_turn(served, monkeypatch) -> None:
    """The delimiter is matched on the model's RAW output, which is what the engine
    scans -- so one that lands inside `<tool_call>` syntax stops the turn there rather
    than being hidden by the parser. The block is then unterminated, and the parser's
    existing rule hands it back as text."""
    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", _canned_stream(CALL_CANNED))

    body = _oai_body(client, stop=["get_weather"], tools=[WEATHER_TOOL])
    choice = body["choices"][0]
    assert not choice["message"].get("tool_calls")
    assert "get_weather" not in choice["message"]["content"]
    assert choice["message"]["content"].startswith("Thought: look it up.")
    assert choice["finish_reason"] == "stop"
    assert _oai_streamed_text(
        _oai_chunks(client, stop=["get_weather"], tools=[WEATHER_TOOL])
    ) == choice["message"]["content"]


# --- generation must actually STOP, not merely be truncated -------------------------

ENGINE_PROMPT = "Write a numbered list of five fruits:\n1."
ENGINE_MAX_TOKENS = 40


def _fresh_engine(model: ftm.Model) -> ftm.MetalEngine:
    return ftm.MetalEngine(model, ftm.EngineConfig(n_ctx=1024, n_seq_max=1))


def _pick_delimiter(text: str, size: int = 5) -> tuple[str, int]:
    """A delimiter taken from the model's OWN output, plus where it first occurs.

    Derived rather than hardcoded: the expectation for the constrained run is then the
    unconstrained run's own prefix, so the test cannot pass by asserting that the code
    agrees with a transcription of what the model once said.
    """
    for start in range(len(text) // 4, len(text) - size - 3):
        candidate = text[start : start + size]
        if not candidate.strip():
            continue
        first = text.find(candidate)
        # Must leave real text before it (so truncation is observable) and real text
        # after it (so stopping early is observable).
        if 2 <= first and first + size <= len(text) - 3:
            return candidate, first
    pytest.fail(f"no usable delimiter inside the model's output: {text!r}")


def test_engine_stops_decoding_at_a_stop_sequence(model: ftm.Model) -> None:
    """The proof that this is a stop, not a truncation.

    `stop_at_eog=False` and a fixed cap make the unconstrained run exactly
    ENGINE_MAX_TOKENS tokens long, so any shortfall in the constrained run is the stop
    sequence doing its job. Asserted on the engine's retirement reason, its token count
    and `ctx.decode_calls` -- the C++-side count of real llama_decode calls -- because
    no assertion about the response text can tell "stopped" from "cut down afterwards".
    """
    params = dict(temp=0.0, max_tokens=ENGINE_MAX_TOKENS, stop_at_eog=False)

    engine = _fresh_engine(model)
    try:
        rid = engine.add_request(ENGINE_PROMPT, ftm.RequestParams(**params))
        baseline = "".join(o.piece for o in engine.drain())
        assert engine.state(rid).n_generated == ENGINE_MAX_TOKENS
        assert engine.state(rid).finish_reason == "length"
        full_decodes = engine.ctx.decode_calls
    finally:
        engine.ctx.close()

    delimiter, cut = _pick_delimiter(baseline)

    engine = _fresh_engine(model)
    try:
        rid = engine.add_request(
            ENGINE_PROMPT, ftm.RequestParams(stop=(delimiter,), **params)
        )
        pieces = [o.piece for o in engine.drain()]
        state = engine.state(rid)
        stopped_decodes = engine.ctx.decode_calls
    finally:
        engine.ctx.close()

    assert state.finish_reason == ENGINE_REASON, (
        f"expected the engine to retire on {delimiter!r}; got {state.finish_reason!r}"
    )
    assert state.n_generated < ENGINE_MAX_TOKENS, (
        "the engine generated its whole budget, i.e. nothing actually stopped"
    )
    assert stopped_decodes < full_decodes, (
        f"{stopped_decodes} decodes vs {full_decodes} unconstrained: the delimiter has "
        "to save decode calls, not just characters"
    )
    # The same filter the routes run, over the same pieces, reproduces the wire text --
    # and it is the unconstrained run's own prefix, so the two runs agree about the
    # text up to the delimiter.
    stopper = StopSequenceFilter((delimiter,))
    emitted = []
    for piece in pieces:
        text, hit = stopper.push(piece)
        emitted.append(text)
        if hit:
            break
    assert "".join(emitted) == baseline[:cut]
    assert stopper.matched == delimiter


def test_async_engine_reports_the_stop_reason_through_its_stream(model: ftm.Model) -> None:
    """The reason has to survive the asyncio hop, since that is what the routes read."""
    from freetoken_mac.engine.async_engine import AsyncEngine

    async def run(stop: tuple[str, ...]) -> tuple[list[str], str | None]:
        eng = AsyncEngine(_fresh_engine(model))
        await eng.start()
        try:
            rid = await eng.submit(
                ENGINE_PROMPT,
                ftm.RequestParams(
                    temp=0.0, max_tokens=ENGINE_MAX_TOKENS, stop_at_eog=False, stop=stop
                ),
            )
            pieces: list[str] = []
            reason: str | None = None
            async for out in eng.stream(rid):
                pieces.append(out.piece)
                if out.finished:
                    reason = out.finish_reason
            return pieces, reason
        finally:
            await eng.stop()

    baseline, reason = asyncio.run(run(()))
    assert reason == "length"
    delimiter, _cut = _pick_delimiter("".join(baseline))

    pieces, reason = asyncio.run(run((delimiter,)))
    assert reason == ENGINE_REASON
    assert len(pieces) < len(baseline), "the stopped run must be shorter in TOKENS"


def test_real_model_end_to_end_openai(served) -> None:
    """One end-to-end pass on the real model: no canned stream, no hardcoded text."""
    _app, client = served
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": "List five fruits, one per line."}],
        "max_tokens": 48,
        "temperature": 0,
        "seed": 1234,
    }
    base = client.post("/v1/chat/completions", json=payload)
    assert base.status_code == 200, base.text
    base_body = base.json()
    full = base_body["choices"][0]["message"]["content"]
    delimiter, cut = _pick_delimiter(full)

    stopped = client.post("/v1/chat/completions", json={**payload, "stop": [delimiter]})
    assert stopped.status_code == 200, stopped.text
    body = stopped.json()
    choice = body["choices"][0]
    assert choice["message"]["content"] == full[:cut]
    assert delimiter not in choice["message"]["content"]
    assert choice["finish_reason"] == "stop"
    # Generation stopped: fewer tokens were billed than the unconstrained run needed.
    assert 0 < body["usage"]["completion_tokens"] < base_body["usage"]["completion_tokens"]

    with client.stream(
        "POST", "/v1/chat/completions", json={**payload, "stop": [delimiter], "stream": True}
    ) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln and ln != "data: [DONE]"]
    streamed = "".join(
        json.loads(ln[len("data: "):])["choices"][0]["delta"].get("content", "")
        for ln in lines
    )
    assert streamed == choice["message"]["content"]


def test_real_model_end_to_end_anthropic(served) -> None:
    _app, client = served
    payload = {
        "model": "local",
        "max_tokens": 48,
        "messages": [{"role": "user", "content": "List five fruits, one per line."}],
        "temperature": 0,
    }
    base = client.post("/v1/messages", json=payload)
    assert base.status_code == 200, base.text
    base_body = base.json()
    full = _msg_text(base_body)
    delimiter, cut = _pick_delimiter(full)

    r = client.post("/v1/messages", json={**payload, "stop_sequences": [delimiter]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert _msg_text(body) == full[:cut]
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == delimiter
    assert 0 < body["usage"]["output_tokens"] < base_body["usage"]["output_tokens"]

    with client.stream(
        "POST", "/v1/messages",
        json={**payload, "stop_sequences": [delimiter], "stream": True},
    ) as r:
        assert r.status_code == 200
        raw = [ln for ln in r.iter_lines() if ln]
    datas = [json.loads(ln[len("data: "):]) for ln in raw if ln.startswith("data: ")]
    assert _msg_streamed_text(datas) == _msg_text(body)
    delta = [d for d in datas if d.get("type") == "message_delta"][-1]
    assert delta["delta"]["stop_reason"] == "stop_sequence"
    assert delta["delta"]["stop_sequence"] == delimiter
