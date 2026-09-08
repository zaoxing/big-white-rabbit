"""Anthropic Messages API surface (/v1/messages).

The load-bearing property is that both surfaces share one prompt format and one parser:
a request expressed in Anthropic's shape must produce the SAME prompt as the equivalent
OpenAI request, because the model's chat template was trained on one format and the
client's choice of API must not leak into it.
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


@pytest.fixture(scope="module")
def model() -> ftm.Model:
    m = ftm.Model(MODEL_PATH, ftm.ModelParams())
    yield m
    # Release the weights before interpreter exit or ggml's Metal device destructor
    # aborts the process (exit 134). See docs/llamacpp-notes.md.
    m.close()


@pytest.fixture(scope="module")
def client(model: ftm.Model):
    tc = pytest.importorskip("fastapi.testclient")
    from freetoken_mac.server.app import build_app

    app = build_app(
        model, ftm.EngineConfig(n_ctx=4096, n_batch=512, n_ubatch=512, n_seq_max=2)
    )
    with tc.TestClient(app) as c:
        yield c
    assert app.state.engine.engine.ctx.closed, "lifespan must close the context"


# --- prompt equivalence with the OpenAI surface -------------------------------------


def test_prompt_matches_openai_surface_for_equivalent_request(model: ftm.Model) -> None:
    """The whole point of translating at the edges: an Anthropic request and the
    equivalent OpenAI request must render the IDENTICAL prompt. If these diverge, one of
    the two surfaces is feeding the model text it was not trained on."""
    from freetoken_mac.server import anthropic_api as AA
    from freetoken_mac.server.app import _message_pairs as oai_pairs
    from freetoken_mac.server.app import render_pairs
    from freetoken_mac.server.anthropic_schemas import MessagesRequest
    from freetoken_mac.server.schemas import ChatCompletionRequest
    from freetoken_mac.server.tools import inject_tools

    anth = MessagesRequest(
        model="m",
        max_tokens=16,
        system="You are terse.",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[WEATHER_TOOL],
    )
    a_pairs = AA._message_pairs(anth)
    a_pairs = inject_tools(a_pairs, AA._openai_tools(anth), None)
    a_prompt = render_pairs(model, a_pairs)

    oai = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Weather in Paris?"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": WEATHER_TOOL["name"],
                    "description": WEATHER_TOOL["description"],
                    "parameters": WEATHER_TOOL["input_schema"],
                },
            }
        ],
    )
    o_pairs = inject_tools(oai_pairs(oai), oai.tools, oai.tool_choice)
    o_prompt = render_pairs(model, o_pairs)

    assert a_prompt == o_prompt, "Anthropic and OpenAI surfaces rendered different prompts"
    # And the tools block really is present, so this is not two identically-empty prompts.
    assert "<tools>" in a_prompt and "get_weather" in a_prompt


def test_system_is_a_top_level_field_not_a_message(model: ftm.Model) -> None:
    """Anthropic puts `system` outside `messages`; it must still land in the system turn."""
    from freetoken_mac.server import anthropic_api as AA
    from freetoken_mac.server.anthropic_schemas import MessagesRequest

    req = MessagesRequest(
        model="m", max_tokens=8, system="SENTINEL_SYS",
        messages=[{"role": "user", "content": "hi"}],
    )
    pairs = AA._message_pairs(req)
    assert pairs[0] == ("system", "SENTINEL_SYS")

    # Block-list spelling of `system` must flatten the same way.
    req2 = MessagesRequest(
        model="m", max_tokens=8,
        system=[{"type": "text", "text": "SENTINEL_SYS"}],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert AA._message_pairs(req2)[0] == ("system", "SENTINEL_SYS")


def test_tool_use_and_tool_result_blocks_round_trip(model: ftm.Model) -> None:
    """A replayed tool exchange must reach the prompt in the model's own syntax."""
    from freetoken_mac.server import anthropic_api as AA
    from freetoken_mac.server.anthropic_schemas import MessagesRequest

    req = MessagesRequest(
        model="m", max_tokens=8,
        messages=[
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Paris"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"},
            ]},
        ],
        tools=[WEATHER_TOOL],
    )
    pairs = AA._message_pairs(req)
    roles = [r for r, _ in pairs]
    assert roles == ["user", "assistant", "user"]
    # The assistant turn carries the call in <tool_call> syntax with an OBJECT argument.
    assert "<tool_call>" in pairs[1][1]
    assert '"name": "get_weather"' in pairs[1][1]
    assert '"city": "Paris"' in pairs[1][1]
    # The result is wrapped as the template wraps it.
    assert "<tool_response>" in pairs[2][1] and "18C" in pairs[2][1]


def test_consecutive_tool_results_collapse_into_one_turn(model: ftm.Model) -> None:
    from freetoken_mac.server import anthropic_api as AA
    from freetoken_mac.server.anthropic_schemas import MessagesRequest

    req = MessagesRequest(
        model="m", max_tokens=8,
        messages=[
            {"role": "user", "content": "go"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "A"},
                {"type": "tool_result", "tool_use_id": "b", "content": "B"},
            ]},
        ],
    )
    pairs = AA._message_pairs(req)
    tool_turns = [t for r, t in pairs if "<tool_response>" in t]
    assert len(tool_turns) == 1, "two results must share one user turn"
    assert tool_turns[0].count("<tool_response>") == 2


def test_tool_choice_translation(model: ftm.Model) -> None:
    """Anthropic's `any`/`tool` map onto the OpenAI spellings inject_tools understands."""
    from freetoken_mac.server.anthropic_api import _openai_tool_choice
    from freetoken_mac.server.tools import resolve_tool_choice

    assert resolve_tool_choice(_openai_tool_choice(None)) == ("auto", None)
    assert resolve_tool_choice(_openai_tool_choice({"type": "auto"})) == ("auto", None)
    assert resolve_tool_choice(_openai_tool_choice({"type": "none"})) == ("none", None)
    assert resolve_tool_choice(_openai_tool_choice({"type": "any"})) == ("required", None)
    assert resolve_tool_choice(
        _openai_tool_choice({"type": "tool", "name": "get_weather"})
    ) == ("required", "get_weather")


# --- HTTP surface -------------------------------------------------------------------


def test_messages_non_streaming(client) -> None:
    r = client.post("/v1/messages", json={
        "model": "local",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "temperature": 0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert isinstance(body["content"], list) and body["content"], "content must be blocks"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"].strip()
    assert body["stop_reason"] in ("end_turn", "max_tokens", "tool_use", "stop_sequence")
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] > 0
    assert body["id"].startswith("msg_")


def test_max_tokens_is_required(client) -> None:
    """Anthropic requires it; omitting it must be a 422 naming the field, not a default."""
    r = client.post("/v1/messages", json={
        "model": "local",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 422
    assert "max_tokens" in r.text


def test_messages_streaming_event_sequence(client) -> None:
    """The Anthropic stream is NAMED events in a required order; the SDK dispatches on
    the event name, so a stream that merely carries valid JSON decodes to nothing."""
    with client.stream("POST", "/v1/messages", json={
        "model": "local",
        "max_tokens": 12,
        "messages": [{"role": "user", "content": "Count: 1 2 3"}],
        "stream": True,
    }) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = [ln for ln in r.iter_lines() if ln]

    events = [ln[len("event: "):] for ln in raw if ln.startswith("event: ")]
    datas = [json.loads(ln[len("data: "):]) for ln in raw if ln.startswith("data: ")]
    assert len(events) == len(datas), "every event needs its data line"

    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert events[-2] == "message_delta"
    assert "content_block_start" in events
    assert "content_block_delta" in events
    assert "content_block_stop" in events
    # Blocks must open before they delta and close after.
    assert events.index("content_block_start") < events.index("content_block_delta")

    start = datas[0]
    assert start["message"]["id"].startswith("msg_")
    assert start["message"]["usage"]["input_tokens"] > 0

    text = "".join(
        d["delta"]["text"] for d in datas
        if d.get("type") == "content_block_delta" and d["delta"].get("type") == "text_delta"
    )
    assert text.strip(), "stream produced no text"
    assert datas[-2]["delta"]["stop_reason"] in ("end_turn", "max_tokens", "tool_use")
    assert datas[-2]["usage"]["output_tokens"] > 0


def test_streaming_and_non_streaming_agree(client) -> None:
    payload = {
        "model": "local",
        "max_tokens": 12,
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "temperature": 0,
    }
    whole = client.post("/v1/messages", json=payload).json()
    text_blocks = [b["text"] for b in whole["content"] if b["type"] == "text"]
    expected = "".join(text_blocks)

    with client.stream("POST", "/v1/messages", json={**payload, "stream": True}) as r:
        datas = [
            json.loads(ln[len("data: "):]) for ln in r.iter_lines()
            if ln.startswith("data: ")
        ]
    streamed = "".join(
        d["delta"]["text"] for d in datas
        if d.get("type") == "content_block_delta" and d["delta"].get("type") == "text_delta"
    )
    assert streamed == expected


def test_messages_with_tools_is_well_formed(client) -> None:
    """Conditional on purpose: a 0.5B model may or may not choose to call. What must
    hold is that IF it calls, the block is well-formed and stop_reason agrees."""
    r = client.post("/v1/messages", json={
        "model": "local",
        "max_tokens": 48,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "temperature": 0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    uses = [b for b in body["content"] if b["type"] == "tool_use"]
    if uses:
        assert body["stop_reason"] == "tool_use"
        for u in uses:
            assert u["id"].startswith("toolu_")
            assert u["name"] == "get_weather"
            # Anthropic's `input` is an OBJECT, unlike OpenAI's JSON-string arguments.
            assert isinstance(u["input"], dict)
    else:
        assert body["stop_reason"] in ("end_turn", "max_tokens")
        assert any(b["type"] == "text" for b in body["content"])
    # Either way no call syntax may leak into a text block.
    for b in body["content"]:
        if b["type"] == "text":
            assert "<tool_call>" not in b["text"]


def test_tool_choice_none_suppresses_the_tools_block(client) -> None:
    r = client.post("/v1/messages", json={
        "model": "local",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": {"type": "none"},
        "temperature": 0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert not [b for b in body["content"] if b["type"] == "tool_use"]
    assert body["stop_reason"] != "tool_use"


def test_bad_requests_are_4xx_and_server_survives(client) -> None:
    # Empty messages.
    assert client.post("/v1/messages", json={
        "model": "local", "max_tokens": 8, "messages": []
    }).status_code == 400
    # Non-positive max_tokens.
    assert client.post("/v1/messages", json={
        "model": "local", "max_tokens": 0,
        "messages": [{"role": "user", "content": "hi"}],
    }).status_code == 400
    # A role Anthropic does not allow.
    assert client.post("/v1/messages", json={
        "model": "local", "max_tokens": 8,
        "messages": [{"role": "system", "content": "hi"}],
    }).status_code == 422
    # Garbage tool_choice degrades rather than 500s.
    assert client.post("/v1/messages", json={
        "model": "local", "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [WEATHER_TOOL], "tool_choice": {"type": "wat"},
    }).status_code in (200, 422)
    assert client.get("/health").status_code == 200


def test_openai_surface_still_works(client) -> None:
    """Mounting /v1/messages on the same app must not disturb the OpenAI routes."""
    r = client.post("/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 6,
    })
    assert r.status_code == 200, r.text
    assert r.json()["object"] == "chat.completion"
