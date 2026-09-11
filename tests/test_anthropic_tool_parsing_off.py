"""`/v1/messages` must not invent tool calls the client never offered.

The Anthropic route computed `known = tool_names(...) if (tools and mode != "none") else
None` and handed that straight to the shared parser. But `None` meant "accept ANY tool
name" inside the parser, so the sentinel the route used for "do not parse" switched
parsing fully ON with the name check disabled. Consequences on the wire:

  * a request with NO `tools` at all -- e.g. one replaying a prior tool exchange, which
    `_message_pairs` renders back into `<tool_call>` syntax and which therefore primes
    the model to continue in that style -- could come back with a `tool_use` block and
    `stop_reason: "tool_use"`, naming a tool the client had not declared;
  * `tool_choice: {"type": "none"}` was strictly worse: the client stated it would not
    dispatch calls, and the surface parsed anyway.

The OpenAI route on the same generation returns those bytes as literal text with
`finish_reason: "stop"`, so the two surfaces disagreed about the same model output.

The generation is CANNED here (the engine's stream is replaced) rather than coaxed out of
the 0.5B model: whether it emits a call is a coin flip, and this is a statement about the
route, not about the weights. Each "off" case is paired with the identical generation
through a request that DID offer the tool, so an assertion cannot pass just because
nothing was parseable.
"""

from __future__ import annotations

import json
import os

import pytest

import freetoken_mac as ftm

MODEL_PATH = os.environ.get("FTM_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_TEST_MODEL to a .gguf path to run these",
)

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

# Exactly what the model emitted in the reproduction: one well-formed call, in Qwen2.5's
# own syntax, nothing else.
CALL_TEXT = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'
)


def _canned_stream(text: str, *, chunk: int = 7):
    """Replace AsyncEngine.stream with one that replays `text` in small pieces.

    Pieces deliberately do not respect the tag boundaries, so the parser is exercised the
    way real token pieces exercise it (`<tool`, `_call`, `>`).
    """
    from freetoken_mac.engine.metal_engine import StepOutput

    pieces = [text[i : i + chunk] for i in range(0, len(text), chunk)] or [""]

    async def stream(request_id: int):
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            yield StepOutput(request_id, 0, piece, last, "eog" if last else None)

    return stream


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


@pytest.fixture
def canned(served, monkeypatch):
    """Every request served while this is active replays CALL_TEXT."""
    app, client = served
    monkeypatch.setattr(app.state.engine, "stream", _canned_stream(CALL_TEXT))
    return client


def _messages(client, **extra) -> dict:
    r = client.post("/v1/messages", json={
        "model": "local",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "What is the weather in London?"}],
        "temperature": 0,
        **extra,
    })
    assert r.status_code == 200, r.text
    return r.json()


def _events(client, **extra) -> tuple[list[str], list[dict]]:
    with client.stream("POST", "/v1/messages", json={
        "model": "local",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "What is the weather in London?"}],
        "temperature": 0,
        "stream": True,
        **extra,
    }) as r:
        assert r.status_code == 200
        raw = [ln for ln in r.iter_lines() if ln]
    names = [ln[len("event: "):] for ln in raw if ln.startswith("event: ")]
    datas = [json.loads(ln[len("data: "):]) for ln in raw if ln.startswith("data: ")]
    return names, datas


def _streamed_text(datas: list[dict]) -> str:
    return "".join(
        d["delta"]["text"] for d in datas
        if d.get("type") == "content_block_delta"
        and d["delta"].get("type") == "text_delta"
    )


# --- the parser's contract ----------------------------------------------------------


def test_parser_off_state_has_exactly_one_meaning() -> None:
    """`known_names=None` is "off", and off means pass-through -- not "accept anything"."""
    from freetoken_mac.server.tools import ToolCallStreamParser, parse_tool_calls

    off = parse_tool_calls(CALL_TEXT, None)
    assert off.tool_calls == [], "parsing is off, so nothing may be extracted"
    assert off.content == CALL_TEXT, "off must hand back the bytes untouched"

    parser = ToolCallStreamParser(None)
    assert parser.enabled is False
    # Off withholds nothing: the chunk boundaries the caller fed in are preserved, which
    # is what keeps a no-tools SSE stream identical to the pre-tools one.
    assert parser.push("<tool") == ("<tool", [])
    assert parser.push("_call>x") == ("_call>x", [])
    assert parser.flush() == ("", [])
    assert parser.n_calls == 0

    # And the same generation with the tool actually offered still parses.
    on = parse_tool_calls(CALL_TEXT, {"get_weather"})
    assert [c.name for c in on.tool_calls] == ["get_weather"]
    assert ToolCallStreamParser({"get_weather"}).enabled is True


def test_route_derives_the_off_state_for_both_no_tools_and_choice_none() -> None:
    from freetoken_mac.server.anthropic_api import _openai_tool_choice, _parsing_names

    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    assert _parsing_names([], None) is None, "no tools offered -> parsing off"
    assert _parsing_names([], _openai_tool_choice({"type": "auto"})) is None
    assert _parsing_names(tools, _openai_tool_choice({"type": "none"})) is None
    assert _parsing_names(tools, None) == {"get_weather"}
    assert _parsing_names(tools, _openai_tool_choice({"type": "any"})) == {"get_weather"}


# --- non-streaming ------------------------------------------------------------------


def test_messages_without_tools_returns_call_syntax_as_text(canned) -> None:
    body = _messages(canned)  # no `tools` on the request
    assert not [b for b in body["content"] if b["type"] == "tool_use"], (
        "a request that offered no tools must not come back with a tool_use block"
    )
    assert [b["type"] for b in body["content"]] == ["text"]
    assert body["content"][0]["text"] == CALL_TEXT
    assert body["stop_reason"] == "end_turn"


def test_messages_tool_choice_none_returns_call_syntax_as_text(canned) -> None:
    body = _messages(canned, tools=[WEATHER_TOOL], tool_choice={"type": "none"})
    assert not [b for b in body["content"] if b["type"] == "tool_use"], (
        'tool_choice "none" means the client will not dispatch calls'
    )
    assert body["content"][0]["text"] == CALL_TEXT
    assert body["stop_reason"] != "tool_use"


def test_messages_with_tools_parses_the_same_generation(canned) -> None:
    """The control: identical bytes, but the tool WAS offered, so it is a real call."""
    body = _messages(canned, tools=[WEATHER_TOOL])
    uses = [b for b in body["content"] if b["type"] == "tool_use"]
    assert len(uses) == 1, body
    assert uses[0]["name"] == "get_weather"
    assert uses[0]["input"] == {"city": "London"}
    assert uses[0]["id"].startswith("toolu_")
    assert body["stop_reason"] == "tool_use"


def test_undeclared_tool_name_is_still_text_when_tools_are_offered(served, monkeypatch) -> None:
    """The name check itself must keep working: a call naming a tool that was not offered
    is not dispatchable, so it stays text even though parsing is on."""
    app, client = served
    other = CALL_TEXT.replace("get_weather", "rm_rf")
    monkeypatch.setattr(app.state.engine, "stream", _canned_stream(other))
    body = _messages(client, tools=[WEATHER_TOOL])
    assert not [b for b in body["content"] if b["type"] == "tool_use"]
    assert body["content"][0]["text"] == other
    assert body["stop_reason"] != "tool_use"


# --- streaming ----------------------------------------------------------------------


def test_messages_stream_without_tools_returns_call_syntax_as_text(canned) -> None:
    names, datas = _events(canned)
    opened = [
        d["content_block"]["type"] for d in datas
        if d.get("type") == "content_block_start"
    ]
    assert "tool_use" not in opened, f"stream opened a tool_use block: {opened}"
    assert _streamed_text(datas) == CALL_TEXT
    delta = [d for d in datas if d.get("type") == "message_delta"][-1]
    assert delta["delta"]["stop_reason"] != "tool_use"
    assert names[-1] == "message_stop"


def test_messages_stream_tool_choice_none_returns_call_syntax_as_text(canned) -> None:
    _names, datas = _events(canned, tools=[WEATHER_TOOL], tool_choice={"type": "none"})
    opened = [
        d["content_block"]["type"] for d in datas
        if d.get("type") == "content_block_start"
    ]
    assert "tool_use" not in opened
    assert _streamed_text(datas) == CALL_TEXT
    delta = [d for d in datas if d.get("type") == "message_delta"][-1]
    assert delta["delta"]["stop_reason"] != "tool_use"


def test_messages_stream_with_tools_parses_the_same_generation(canned) -> None:
    _names, datas = _events(canned, tools=[WEATHER_TOOL])
    starts = [
        d for d in datas
        if d.get("type") == "content_block_start"
        and d["content_block"]["type"] == "tool_use"
    ]
    assert len(starts) == 1, datas
    assert starts[0]["content_block"]["name"] == "get_weather"
    assert _streamed_text(datas) == "", "call syntax must not also leak as text"
    delta = [d for d in datas if d.get("type") == "message_delta"][-1]
    assert delta["delta"]["stop_reason"] == "tool_use"


# --- the two surfaces must agree ----------------------------------------------------


def test_both_surfaces_treat_the_off_state_identically(canned) -> None:
    """The OpenAI route already returned these bytes as text with finish_reason "stop";
    that is the contract the Anthropic route now honours as well."""
    oai = canned.post("/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "What is the weather in London?"}],
        "max_tokens": 16,
    })
    assert oai.status_code == 200, oai.text
    choice = oai.json()["choices"][0]
    assert choice["message"]["content"] == CALL_TEXT
    assert not choice["message"].get("tool_calls")
    assert choice["finish_reason"] == "stop"

    body = _messages(canned)
    anthropic_text = "".join(b["text"] for b in body["content"] if b["type"] == "text")
    assert anthropic_text == choice["message"]["content"]
