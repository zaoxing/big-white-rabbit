"""`max_tokens` must be validated the same way on both surfaces, and the cap must bind.

Three defects, all verified against the running server before this file existed:

  * `{"max_tokens": -5}` on `/v1/chat/completions` returned **200** with a ONE-token
    body (`content: "Paris"`, `completion_tokens: 1`) and `finish_reason: "length"`.
    `-5` is truthy, so `resolved_max_tokens`'s `or` chain passed it straight into the
    engine, where `_advance`'s `if req.n_generated >= req.params.max_tokens` is
    `1 >= -5` -- true after the very first token.
  * `{"max_tokens": 0}` is FALSY, so the same `or` chain silently substituted
    `DEFAULT_MAX_TOKENS = 512`: an explicit cap replaced by a server default, and the
    client got a full-length generation it did not ask for. Same hole for
    `max_completion_tokens: 0`, which additionally deferred to `max_tokens`.
  * `/v1/messages` has always answered `max_tokens <= 0` with a 400, so the two
    surfaces disagreed about the same field.

And a testing gap behind all three: nothing in the suite bounded `completion_tokens` by
what was REQUESTED, and every `finish_reason` assertion was a disjunction over the whole
legal set (`in ("stop", "length")` / `in ("end_turn", "max_tokens", ...)`), so
`_FINISH_REASONS` and `_STOP_REASONS` could not fail whatever they mapped to. Nothing
sent `max_completion_tokens` at all, which is the ONLY spelling recent OpenAI SDKs use.

What is asserted here, and why each assertion is the one that would have failed:

  * a non-positive cap in either spelling is a 4xx on BOTH surfaces (it was a 200 with
    a one-token body, or a 200 with 512 tokens of allowance, on one of them);
  * `resolved_max_tokens` selects on PRESENCE, not truthiness -- an explicit 0 can
    never become 512, and `max_completion_tokens: 0` never defers to `max_tokens`;
  * the cap binds EXACTLY: `max_tokens: 5` yields `completion_tokens == 5`, pinned
    against the same prompt's unconstrained length so the test cannot pass by the model
    happening to stop on its own;
  * each retirement CAUSE maps to its one expected reason (cap -> length/max_tokens,
    natural end -> stop/end_turn, delimiter -> stop/stop_sequence), rather than to
    "something in the legal set";
  * a normal positive cap is untouched (the positive control: the fix must not turn a
    served request into a 4xx).

The cap cases use the REAL model, because "exactly five tokens" is a statement about
the engine's retirement arithmetic. The reason-mapping cases use CANNED generations
(`AsyncEngine.stream` replaced, as `test_anthropic_tool_parsing_off.py` and
`test_stop_sequences.py` do), because whether a 0.5B model retires for a given cause on
a given prompt is a coin flip and these are statements about the mapping tables.
"""

from __future__ import annotations

import json
import os

import pytest

import freetoken_mac as ftm
from freetoken_mac.server.anthropic_api import _STOP_REASONS
from freetoken_mac.server.app import DEFAULT_MAX_TOKENS, _FINISH_REASONS
from freetoken_mac.server.schemas import ChatCompletionRequest, max_tokens_error

MODEL_PATH = os.environ.get("FTM_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_TEST_MODEL to a .gguf path to run these",
)

# The exact cap under test. Small enough that the model cannot plausibly finish first on
# the prompt below, and the unconstrained run is asserted to be longer than it so the
# comparison is against the model's own behaviour rather than a guess about it.
CAP = 5

# A prompt with a long, obvious continuation: an unconstrained greedy run must exceed
# CAP tokens, or the "exactly CAP" assertions would pass for the wrong reason.
LONG_PROMPT = "Count from 1 to 40, one number per line, nothing else."

# The engine's own retirement reasons, and the cause each one names. Written out here
# rather than imported so that a change to a mapping table has to be made twice --
# once in the server, once as a deliberate edit to this contract.
CAP_REASON = "length"
NATURAL_END_REASON = "eog"
DELIMITER_REASON = "stop_sequence"

DELIMITER = "\nObservation:"
CANNED = "Thought: look it up.\nObservation: rain"
assert DELIMITER in CANNED  # keeps the fixture honest


# --- the rule, with no model and no server ------------------------------------------


def test_max_tokens_error_only_rejects_a_cap_the_client_actually_sent() -> None:
    """`None` means "no cap of mine"; a sent cap must be >= 1."""
    assert max_tokens_error(("max_tokens", None)) is None
    assert max_tokens_error(("max_tokens", 1)) is None
    assert max_tokens_error(("max_tokens", 512)) is None

    # 0 is the case the `or` chain swallowed, and -5 the case that reached the engine.
    for bad in (0, -1, -5):
        detail = max_tokens_error(("max_tokens", bad))
        assert detail is not None, f"{bad} must be rejected"
        # The error names the offending field, which is the whole point of validating at
        # the protocol boundary rather than letting the engine's arithmetic decide.
        assert "max_tokens" in detail
        assert str(bad) in detail


def test_max_tokens_error_names_the_first_offending_field_in_precedence_order() -> None:
    """Both spellings are checked, and the one that WINS is reported first.

    `max_completion_tokens` takes precedence in `resolved_max_tokens`, so when it is the
    bad one it is the field the client has to fix -- naming `max_tokens` there would
    send them to edit the field that was never going to be used.
    """
    detail = max_tokens_error(
        ("max_completion_tokens", 0), ("max_tokens", 64)
    )
    assert detail is not None and "max_completion_tokens" in detail

    detail = max_tokens_error(("max_completion_tokens", None), ("max_tokens", -5))
    assert detail is not None and "max_tokens" in detail


def test_resolved_max_tokens_selects_on_presence_not_truthiness() -> None:
    """The falsy-0 hole, asserted directly on the resolver.

    Before the fix this was `max_completion_tokens or max_tokens or default`, so every
    line below that involves a 0 returned the wrong number: an explicit 0 became
    DEFAULT_MAX_TOKENS, and `max_completion_tokens: 0` silently deferred to its sibling.
    """

    def req(**kw) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}], **kw
        )

    # Neither sent: the server default, which is the only case that may fall through.
    assert req().resolved_max_tokens(DEFAULT_MAX_TOKENS) == DEFAULT_MAX_TOKENS
    # Either one sent alone.
    assert req(max_tokens=7).resolved_max_tokens(DEFAULT_MAX_TOKENS) == 7
    assert (
        req(max_completion_tokens=7).resolved_max_tokens(DEFAULT_MAX_TOKENS) == 7
    )
    # Both sent: the newer spelling wins, as OpenAI's own precedence has it.
    assert (
        req(max_completion_tokens=3, max_tokens=7).resolved_max_tokens(
            DEFAULT_MAX_TOKENS
        )
        == 3
    )
    # An explicit 0 must never become the default. It is rejected at the route (below),
    # but the resolver must be right on its own terms: this is what made a 0 unnoticeable
    # in the first place.
    assert req(max_tokens=0).resolved_max_tokens(DEFAULT_MAX_TOKENS) == 0
    assert (
        req(max_completion_tokens=0).resolved_max_tokens(DEFAULT_MAX_TOKENS) == 0
    )
    # ... and a 0 in the winning field must not defer to the losing one.
    assert (
        req(max_completion_tokens=0, max_tokens=7).resolved_max_tokens(
            DEFAULT_MAX_TOKENS
        )
        == 0
    )
    # A negative value survives the resolver unchanged too, so the route is the only
    # thing standing between it and the engine -- and the route is asserted below.
    assert req(max_tokens=-5).resolved_max_tokens(DEFAULT_MAX_TOKENS) == -5


def test_the_reason_tables_map_every_engine_cause_to_one_protocol_reason() -> None:
    """The tables themselves, key by key.

    Both surfaces have to learn every engine reason: one mapped in one table and not the
    other is a response whose stop_reason contradicts its text. The wire-level tests
    below cover the three causes that can be provoked; this covers the rest, and pins
    the specific value instead of "a member of the legal set".
    """
    assert _FINISH_REASONS[CAP_REASON] == "length"
    assert _STOP_REASONS[CAP_REASON] == "max_tokens"
    assert _FINISH_REASONS[NATURAL_END_REASON] == "stop"
    assert _STOP_REASONS[NATURAL_END_REASON] == "end_turn"
    assert _FINISH_REASONS[DELIMITER_REASON] == "stop"
    assert _STOP_REASONS[DELIMITER_REASON] == "stop_sequence"
    # Out of KV room is a cap the SERVER imposed, not a completed turn: telling a client
    # "stop" would have it treat a truncated answer as final.
    assert _FINISH_REASONS["context"] == "length"
    assert _STOP_REASONS["context"] == "max_tokens"
    # A cancelled turn has no better protocol word than "the turn ended".
    assert _FINISH_REASONS["cancelled"] == "stop"
    assert _STOP_REASONS["cancelled"] == "end_turn"
    # Both tables must agree on WHICH causes exist, or a reason added to one surface
    # silently falls through to the other's default.
    assert set(_FINISH_REASONS) == set(_STOP_REASONS)


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
        model, ftm.EngineConfig(n_ctx=4096, n_batch=512, n_ubatch=512, n_seq_max=2)
    )
    with tc.TestClient(app) as c:
        yield app, c
    assert app.state.engine.engine.ctx.closed, "lifespan must close the context"


@pytest.fixture
def client(served):
    return served[1]


def _canned_stream(text: str, reason: str, *, chunk: int = 3):
    """Replace AsyncEngine.stream with one replaying `text` and retiring for `reason`.

    Pieces are fixed-size rather than token-aligned, which is what a real tokeniser
    delivers as far as the routes can tell.
    """
    from freetoken_mac.engine.metal_engine import StepOutput

    pieces = [text[i : i + chunk] for i in range(0, len(text), chunk)] or [""]

    async def stream(request_id: int):
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            yield StepOutput(request_id, 0, piece, last, reason if last else None)

    return stream


@pytest.fixture
def canned(served, monkeypatch):
    """`canned(reason)` makes every subsequent request retire for that engine reason."""
    app, client = served

    def install(reason: str, text: str = CANNED):
        monkeypatch.setattr(app.state.engine, "stream", _canned_stream(text, reason))
        return client

    return install


# --- wire helpers -------------------------------------------------------------------


def _oai(client, **extra):
    return client.post("/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "temperature": 0,
        **extra,
    })


def _msg(client, **extra):
    return client.post("/v1/messages", json={
        "model": "local",
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "temperature": 0,
        **extra,
    })


def _oai_ok(client, **extra) -> dict:
    r = _oai(client, **extra)
    assert r.status_code == 200, r.text
    return r.json()


def _msg_ok(client, **extra) -> dict:
    r = _msg(client, **extra)
    assert r.status_code == 200, r.text
    return r.json()


def _oai_chunks(client, **extra) -> list[dict]:
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "temperature": 0,
        "stream": True,
        **extra,
    }) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln]
    assert lines[-1] == "data: [DONE]", lines[-3:]
    return [json.loads(ln[len("data: "):]) for ln in lines[:-1]]


def _oai_stream_finish_reason(chunks: list[dict]) -> str | None:
    reasons = [
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c["choices"][0].get("finish_reason") is not None
    ]
    assert len(reasons) == 1, f"expected exactly one final frame, got {reasons}"
    return reasons[0]


def _msg_events(client, **extra) -> list[dict]:
    with client.stream("POST", "/v1/messages", json={
        "model": "local",
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "temperature": 0,
        "stream": True,
        **extra,
    }) as r:
        assert r.status_code == 200
        raw = [ln for ln in r.iter_lines() if ln]
    return [json.loads(ln[len("data: "):]) for ln in raw if ln.startswith("data: ")]


def _msg_delta(datas: list[dict]) -> dict:
    deltas = [d for d in datas if d.get("type") == "message_delta"]
    assert len(deltas) == 1, f"expected one message_delta, got {len(deltas)}"
    return deltas[0]


# --- non-positive is a 4xx, on both surfaces ----------------------------------------


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("bad", [0, -1, -5])
def test_openai_rejects_a_non_positive_cap(client, field, bad) -> None:
    """The reported defect: `max_tokens: -5` was a 200 with a one-token body."""
    r = _oai(client, **{field: bad})
    assert 400 <= r.status_code < 500, (
        f"{field}={bad} must be a 4xx; got {r.status_code}: {r.text}"
    )
    # Route-level, so the same status the Anthropic surface has always returned.
    assert r.status_code == 400, r.text
    assert field in r.text, r.text


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_anthropic_rejects_a_non_positive_cap(client, bad) -> None:
    r = _msg(client, max_tokens=bad)
    assert 400 <= r.status_code < 500, (
        f"max_tokens={bad} must be a 4xx; got {r.status_code}: {r.text}"
    )
    assert r.status_code == 400, r.text
    assert "max_tokens" in r.text, r.text


def test_a_bad_cap_in_the_losing_field_is_still_a_4xx(client) -> None:
    """`max_completion_tokens: 8, max_tokens: -5` must not be served.

    The resolver would never look at `max_tokens` here, so the value could reach nothing
    -- but a request carrying a nonsensical cap is a client bug worth naming, and
    serving it silently is how `-5` survived long enough to reach the engine through the
    other field.
    """
    r = _oai(client, max_completion_tokens=8, max_tokens=-5)
    assert r.status_code == 400, r.text
    assert "max_tokens" in r.text


def test_the_server_is_healthy_after_every_rejection(client) -> None:
    for payload in ({"max_tokens": 0}, {"max_tokens": -5}, {"max_completion_tokens": 0}):
        assert _oai(client, **payload).status_code == 400
    assert _msg(client, max_tokens=0).status_code == 400
    assert client.get("/health").status_code == 200
    # And a good request still serves, i.e. no slot was burned by the rejections.
    assert _oai_ok(client, max_tokens=4)["usage"]["completion_tokens"] >= 1


def test_both_surfaces_answer_a_non_positive_cap_identically(client) -> None:
    """The disagreement itself: same field, same value, same status."""
    oai = _oai(client, max_tokens=0)
    msg = _msg(client, max_tokens=0)
    assert oai.status_code == msg.status_code == 400, (oai.text, msg.text)


# --- the cap binds exactly ----------------------------------------------------------


def test_openai_cap_binds_exactly(client) -> None:
    """`max_tokens: CAP` yields exactly CAP completion tokens and `length`.

    The unconstrained run is measured first so the assertion is against the model's own
    behaviour: if it stopped on its own inside CAP tokens, "exactly CAP" would be an
    accident rather than the cap binding, and this test would be vacuous.
    """
    free = _oai_ok(client, max_tokens=64)
    assert free["usage"]["completion_tokens"] > CAP, (
        "prompt finishes inside the cap, so the cap assertion would be vacuous: "
        f"{free['usage']}"
    )

    body = _oai_ok(client, max_tokens=CAP)
    assert body["usage"]["completion_tokens"] == CAP, body["usage"]
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + CAP
    assert body["choices"][0]["finish_reason"] == "length"


def test_openai_max_completion_tokens_binds_exactly(client) -> None:
    """The field recent OpenAI SDKs send, which nothing in the suite exercised."""
    body = _oai_ok(client, max_completion_tokens=CAP)
    assert body["usage"]["completion_tokens"] == CAP, body["usage"]
    assert body["choices"][0]["finish_reason"] == "length"


def test_max_completion_tokens_wins_over_max_tokens_on_the_wire(client) -> None:
    """Precedence, asserted where it is observable rather than only in the resolver."""
    body = _oai_ok(client, max_completion_tokens=CAP, max_tokens=64)
    assert body["usage"]["completion_tokens"] == CAP, body["usage"]
    assert body["choices"][0]["finish_reason"] == "length"


def test_openai_streaming_cap_binds_exactly(client) -> None:
    chunks = _oai_chunks(client, max_tokens=CAP)
    assert _oai_stream_finish_reason(chunks) == "length"


def test_anthropic_cap_binds_exactly(client) -> None:
    free = _msg_ok(client, max_tokens=64)
    assert free["usage"]["output_tokens"] > CAP, (
        f"prompt finishes inside the cap: {free['usage']}"
    )

    body = _msg_ok(client, max_tokens=CAP)
    assert body["usage"]["output_tokens"] == CAP, body["usage"]
    assert body["stop_reason"] == "max_tokens"
    # A turn cut off by the cap did not end at a delimiter.
    assert body["stop_sequence"] is None


def test_anthropic_streaming_cap_binds_exactly(client) -> None:
    delta = _msg_delta(_msg_events(client, max_tokens=CAP))
    assert delta["usage"]["output_tokens"] == CAP, delta["usage"]
    assert delta["delta"]["stop_reason"] == "max_tokens"


def test_a_cap_of_one_yields_exactly_one_token(client) -> None:
    """The boundary the engine's `>=` gets wrong in the other direction.

    `max_tokens: 1` is the smallest legal cap, and the value `-5` was accidentally
    behaving as. It must produce one token because it was asked to, not because the
    comparison happened to be true.
    """
    body = _oai_ok(client, max_tokens=1)
    assert body["usage"]["completion_tokens"] == 1, body["usage"]
    assert body["choices"][0]["finish_reason"] == "length"


# --- each cause maps to its one reason ----------------------------------------------


def test_hitting_the_cap_reports_length_and_max_tokens(canned) -> None:
    client = canned(CAP_REASON)
    assert _oai_ok(client, max_tokens=8)["choices"][0]["finish_reason"] == "length"
    assert _msg_ok(client, max_tokens=8)["stop_reason"] == "max_tokens"


def test_a_natural_end_reports_stop_and_end_turn(canned) -> None:
    client = canned(NATURAL_END_REASON)
    assert _oai_ok(client, max_tokens=64)["choices"][0]["finish_reason"] == "stop"
    body = _msg_ok(client, max_tokens=64)
    assert body["stop_reason"] == "end_turn"
    assert body["stop_sequence"] is None


def test_a_delimiter_reports_stop_and_stop_sequence(canned) -> None:
    """The cause item 1 made reachable; asserted here so the two tables stay pinned
    together against the same three causes."""
    client = canned(NATURAL_END_REASON)
    oai = _oai_ok(client, max_tokens=64, stop=[DELIMITER])
    assert oai["choices"][0]["finish_reason"] == "stop"
    assert DELIMITER not in oai["choices"][0]["message"]["content"]

    body = _msg_ok(client, max_tokens=64, stop_sequences=[DELIMITER])
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == DELIMITER


def test_streamed_reasons_match_the_non_streamed_ones(canned) -> None:
    """A reason that differs between the paths is a client that sees two different
    turns for one request."""
    for reason, oai_expected, msg_expected in (
        (CAP_REASON, "length", "max_tokens"),
        (NATURAL_END_REASON, "stop", "end_turn"),
    ):
        client = canned(reason)
        assert _oai_stream_finish_reason(_oai_chunks(client, max_tokens=8)) == (
            oai_expected
        )
        assert _msg_delta(_msg_events(client, max_tokens=8))["delta"][
            "stop_reason"
        ] == msg_expected


# --- positive control ---------------------------------------------------------------


def test_a_normal_positive_cap_is_untouched(client) -> None:
    """The fix must not make a served request into a 4xx, on either surface.

    Byte-shape as well as status: the response keys a client reads are all still there
    and `completion_tokens` is inside the requested bound.
    """
    body = _oai_ok(client, max_tokens=16)
    assert set(body) >= {"id", "object", "created", "model", "choices", "usage"}
    assert body["object"] == "chat.completion"
    assert 1 <= body["usage"]["completion_tokens"] <= 16
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    assert isinstance(body["choices"][0]["message"]["content"], str)

    msg = _msg_ok(client, max_tokens=16)
    assert set(msg) >= {"id", "type", "role", "content", "model", "stop_reason", "usage"}
    assert 1 <= msg["usage"]["output_tokens"] <= 16

    # No cap at all still works and stays inside the server default.
    plain = _oai_ok(client, messages=[{"role": "user", "content": "hi"}])
    assert 1 <= plain["usage"]["completion_tokens"] <= DEFAULT_MAX_TOKENS


def test_engine_admission_refuses_a_non_positive_cap(model: ftm.Model) -> None:
    """Belt and braces, at the layer where `-5` actually did its damage.

    `_advance` compares `n_generated >= max_tokens`, so a non-positive cap retires the
    request after one token and reports `length` -- a plausible-looking 200 rather than
    an error. Admission is where the engine already refuses an empty or oversized
    prompt, so it is where this belongs: any future route, and any direct engine caller,
    gets the same refusal instead of a one-token generation.
    """
    from freetoken_mac.engine.config import RequestParams
    from freetoken_mac.engine.metal_engine import MetalEngine

    engine = MetalEngine(
        model, ftm.EngineConfig(n_ctx=1024, n_batch=256, n_ubatch=256, n_seq_max=2)
    )
    try:
        free_before = engine.n_free_seq_slots
        for bad in (0, -1, -5):
            with pytest.raises(ValueError, match="max_tokens"):
                engine.add_request("hi", RequestParams(max_tokens=bad))
        # A refused admission must not leak the sequence slot it had popped.
        assert engine.n_free_seq_slots == free_before
        # And a good cap is still admitted afterwards.
        engine.add_request("hi", RequestParams(max_tokens=1))
    finally:
        engine.ctx.close()
