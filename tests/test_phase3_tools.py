"""Phase 3: OpenAI-compatible tool calling.

The test model is Qwen2.5-**0.5B**. Whether it decides to emit a tool call for a given
prompt is not reliable, so "ask for the weather and assert a call comes back" would fail
for reasons unrelated to this code and pass by luck. Everything deterministic is
therefore tested directly:

* **prompt rendering** -- a pure function of the request,
* **output parsing** -- fed the exact byte shape Qwen2.5 emits, plus malformed variants
  and incremental chunkings,
* **HTTP plumbing** -- a real request with `tools`, asserting the response is valid
  either way and that `tool_calls`, *if* present, are well-formed.
"""

from __future__ import annotations

import json
import os
import re

import pytest

import bwr as bwr
from bwr.server import tools as T

MODEL_PATH = os.environ.get("BWR_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_TEST_MODEL to a .gguf path to run these",
)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}},
}
TOOL_NAMES = {"get_weather", "get_time"}


def _call_block(name: str, args: str) -> str:
    """A `<tool_call>` block in exactly the shape the Qwen2.5 template asks for."""
    return f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'


# --- the expected wire format is read out of the GGUF, not restated here --------------
#
# The instruction text is not ours to choose: it is fixed by the chat template embedded
# in the model, because that is what the weights were trained against. A test that
# spelled the strings out again could only ever confirm that the implementation equals
# the test author's transcription -- and a transcription error would be ratified rather
# than caught (it was: the example call object's braces are *doubled* in the template,
# because there they sit inside a jinja string literal and so are not `{{ }}`
# delimiters). So the expectation is extracted from `Model.chat_template` at test time.
#
# No jinja engine is needed: the pieces are plain string literals in the template source,
# concatenated by `{{- "..." }}` statements, so lifting the literals and undoing their
# backslash escapes is enough.

_JINJA_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\"}


def _unescape(literal: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(literal):
        ch = literal[i]
        if ch == "\\" and i + 1 < len(literal):
            nxt = literal[i + 1]
            out.append(_ESCAPES.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _template_literal(template: str, marker: str) -> str:
    """The one double-quoted jinja string literal in `template` containing `marker`."""
    assert template, "the test model's GGUF must carry a chat template"
    hits = [
        _unescape(m.group(1))
        for m in _JINJA_STRING.finditer(template)
        if marker in _unescape(m.group(1))
    ]
    assert len(hits) == 1, f"{marker!r}: expected 1 literal, found {len(hits)}"
    return hits[0]


def _template_tools_block(template: str) -> tuple[str, str]:
    """`(prefix, suffix)` of the tools block exactly as the template renders them.

    The template's trailing `<|im_end|>\\n` is dropped: it closes the system *turn*, which
    `llama_chat_apply_template` emits itself, so it is not part of the message text.
    """
    prefix = _template_literal(template, "# Tools")
    suffix = _template_literal(template, "For each function call")
    turn_end = "<|im_end|>\n"
    assert suffix.endswith(turn_end)
    return prefix, suffix[: -len(turn_end)]


# --- ids are random; tests compare everything else -----------------------------------


def _ids():
    n = iter(range(1000))
    return lambda: f"call_{next(n):04d}"


@pytest.fixture(scope="module")
def model() -> bwr.Model:
    m = bwr.Model(MODEL_PATH, bwr.ModelParams())
    yield m
    # Release the weights before the interpreter exits: ggml frees the Metal device from
    # a static destructor and asserts its residency sets are empty, so a live model then
    # abort()s the process (134) after the suite has already passed.
    m.close()


# ===================================================================================
# prompt injection
# ===================================================================================


def test_prompt_injection_appends_to_existing_system(model: bwr.Model) -> None:
    """The declarations must land *after* existing system content, in the model's own
    wire format -- reproduced byte-for-byte, because the format is fixed by the GGUF's
    embedded template and the weights were trained against it.

    The expectation is therefore *derived from that template*, read live out of the GGUF,
    not restated from the implementation. Restating it would make this assertion
    tautological on exactly the byte range where a transcription slip hides."""
    prefix, suffix = _template_tools_block(model.chat_template)

    # What the authority actually says, called out because it is the easy thing to get
    # wrong: the example call object is wrapped in DOUBLED braces (a jinja string literal
    # passes `{{`/`}}` through untouched), and the surrounding block is not.
    assert '{{"name": <function-name>, "arguments": <args-json-object>}}' in suffix
    assert '\n{"name": <function-name>' not in suffix  # i.e. never the single-brace form
    assert prefix.startswith("\n\n# Tools\n\n") and prefix.endswith("<tools>")

    pairs = [("system", "You are terse."), ("user", "Weather in Paris?")]
    out = T.inject_tools(pairs, [WEATHER_TOOL, TIME_TOOL], "auto")

    assert [r for r, _ in out] == ["system", "user"]
    assert out[1] == ("user", "Weather in Paris?")

    system = out[0][1]
    expected = (
        "You are terse."
        + prefix
        # `{%- for tool in tools %}{{- "\n" }}{{- tool | tojson }}{%- endfor %}`:
        # the whole tool object per line, `ensure_ascii=False` as transformers configures
        # jinja's tojson.
        + "\n"
        + json.dumps(WEATHER_TOOL, ensure_ascii=False)
        + "\n"
        + json.dumps(TIME_TOOL, ensure_ascii=False)
        + suffix
    )
    assert system == expected
    # And the module constants are the template's own strings, so a future edit to them
    # fails here instead of quietly changing what the model is shown.
    assert (T.TOOLS_PREFIX, T.TOOLS_SUFFIX) == (prefix, suffix)


def test_prompt_injection_creates_system_when_absent() -> None:
    """A conversation with no system turn still has to be told about the tools."""
    out = T.inject_tools([("user", "hi")], [WEATHER_TOOL], "auto")
    assert [r for r, _ in out] == ["system", "user"]
    assert out[1] == ("user", "hi")
    # No system content to append to, so the block stands alone (its leading blank lines
    # trimmed) rather than inheriting one vendor's identity string.
    assert out[0][1].startswith("# Tools\n\nYou may call one or more functions")
    assert '"name": "get_weather"' in out[0][1]
    assert out[0][1].endswith("</tool_call>")


def test_prompt_injection_none_omits_tools_block() -> None:
    """`tool_choice="none"` means the client will not dispatch a call, so telling the
    model about the tools can only waste a turn."""
    pairs = [("system", "You are terse."), ("user", "hi")]
    assert T.inject_tools(pairs, [WEATHER_TOOL], "none") == pairs
    assert "# Tools" not in "".join(t for _, t in T.inject_tools(pairs, [WEATHER_TOOL], "none"))

    # Absent, "auto" and unknown spellings all inject; only "none" opts out.
    for choice in (None, "auto", "AUTO", "something-new"):
        assert "# Tools" in T.inject_tools(pairs, [WEATHER_TOOL], choice)[0][1]

    # "required" and a named choice cannot be enforced without a decoding grammar, so
    # they are expressed in the prompt -- and only outside the verbatim block.
    forced = T.inject_tools(pairs, [WEATHER_TOOL], "required")[0][1]
    assert forced.startswith(T.inject_tools(pairs, [WEATHER_TOOL], "auto")[0][1])
    assert "must call at least one" in forced

    named = T.inject_tools(
        pairs,
        [WEATHER_TOOL, TIME_TOOL],
        {"type": "function", "function": {"name": "get_time"}},
    )[0][1]
    assert 'must call the function "get_time"' in named
    # A named choice still declares every tool, so the parser can recognise any of them.
    assert '"name": "get_weather"' in named


def test_prompt_injection_survives_the_real_chat_template(model: bwr.Model) -> None:
    """llama_chat_apply_template has no tools argument (it pattern-matches ~56 templates
    instead of running jinja), so the block has to survive as message *text*."""
    pairs = T.inject_tools(
        [("user", "Weather in Paris?")], [WEATHER_TOOL], "auto"
    )
    prompt = model.apply_chat_template(pairs, True)

    assert "<|im_start|>system" in prompt
    assert "# Tools" in prompt
    assert '"name": "get_weather"' in prompt
    # The tools block belongs to the system turn: it must close before the user turn.
    assert prompt.index("# Tools") < prompt.index("Weather in Paris?")
    assert prompt.index("</tools>") < prompt.index("<|im_start|>user")
    # And the prompt is still left open for the assistant to continue from.
    assert prompt.rstrip().endswith("assistant")

    without = model.apply_chat_template([("user", "Weather in Paris?")], True)
    assert "# Tools" not in without


def test_prompt_injection_round_trips_assistant_calls_and_tool_results() -> None:
    """An agent loop replays its history. The assistant turn that made the calls has to
    go back in the model's syntax, and a result has to go back as a tool_response --
    otherwise the second turn of a tool conversation is missing its own first half."""
    rendered = T.render_assistant_turn(
        "Let me look.",
        [
            {
                "id": "call_1",
                "type": "function",
                # OpenAI replays arguments as a JSON *string*; the template wants the
                # object, so it must be re-inflated.
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    )
    assert rendered == (
        "Let me look."
        '\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    )
    # What we render must be what we parse: the round trip has to close.
    reparsed = T.parse_tool_calls(rendered, TOOL_NAMES, id_factory=_ids())
    assert reparsed.content == "Let me look."
    assert [(c.name, c.arguments) for c in reparsed.tool_calls] == [
        ("get_weather", '{"city": "Paris"}')
    ]

    # A call-only turn (`content: null`, the usual case) begins *at* the block: the
    # template emits the content only under `{%- if message.content %}`, so there is no
    # leading newline for the templater's `<|im_start|>assistant\n` to double.
    assert T.render_assistant_turn(
        "", [{"function": {"name": "get_time", "arguments": "{}"}}]
    ) == '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'

    # Junk in replayed history must not raise -- it degrades.
    assert T.render_assistant_turn("hi", [{"function": {"name": "f"}}]).endswith(
        '{"name": "f", "arguments": {}}\n</tool_call>'
    )
    assert T.render_assistant_turn("hi", [{"nope": 1}]) == "hi"
    assert T.render_assistant_turn("hi", None) == "hi"

    assert T.render_tool_response("22C") == "<tool_response>\n22C\n</tool_response>"


def test_prompt_injection_groups_consecutive_tool_results(model: bwr.Model) -> None:
    """Parallel calls come back as several `role: "tool"` messages in a row. The template
    opens `<|im_start|>user` only when the previous message was not a tool and closes it
    only when the next one is not, so a whole run is ONE user turn carrying several
    <tool_response> blocks -- not one turn each."""
    from bwr.server.app import _message_pairs
    from bwr.server.schemas import ChatCompletionRequest

    req = ChatCompletionRequest(
        model="local",
        messages=[
            {"role": "user", "content": "Weather and time in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "22C, sunny"},
            {"role": "tool", "tool_call_id": "call_b", "content": "13:05"},
        ],
        tools=[WEATHER_TOOL, TIME_TOOL],
    )
    pairs = _message_pairs(req)
    assert [r for r, _ in pairs] == ["user", "assistant", "user"]
    assert pairs[2][1] == (
        "<tool_response>\n22C, sunny\n</tool_response>\n"
        "<tool_response>\n13:05\n</tool_response>"
    )

    # Byte-identical to the template's own tool branch once the real templater has added
    # the turn markers: `'<|im_start|>user'`, then `'\n<tool_response>\n' + content +
    # '\n</tool_response>'` per result, then a single `'<|im_end|>\n'`.
    prompt = model.apply_chat_template(pairs, True)
    assert (
        "<|im_start|>user\n<tool_response>\n22C, sunny\n</tool_response>\n"
        "<tool_response>\n13:05\n</tool_response><|im_end|>\n"
    ) in prompt
    assert prompt.count("<|im_start|>user") == 2  # the question, plus one results turn

    # A non-tool message between two results breaks the run, as the template's
    # `messages[loop.index0 - 1].role != "tool"` guard does.
    split = ChatCompletionRequest(
        model="local",
        messages=[
            {"role": "tool", "tool_call_id": "call_a", "content": "22C"},
            {"role": "user", "content": "and the time?"},
            {"role": "tool", "tool_call_id": "call_b", "content": "13:05"},
        ],
        tools=[WEATHER_TOOL],
    )
    assert [t for _, t in _message_pairs(split)] == [
        "<tool_response>\n22C\n</tool_response>",
        "and the time?",
        "<tool_response>\n13:05\n</tool_response>",
    ]

    # Without `tools` but WITH tool history, the history is still a tool
    # exchange — a follow-up turn may omit the declaration but the prior
    # assistant/tool turns must still be rendered as <tool_response> so the
    # model sees the same shape it was trained on. This fixes the
    # OpenAI/Anthropic divergence (see backlog).
    plain = ChatCompletionRequest(
        model="local", messages=[m.model_dump() for m in req.messages]
    )
    plain_pairs = _message_pairs(plain)
    assert [r for r, _ in plain_pairs] == ["user", "assistant", "user"]
    assert plain_pairs[2][1] == pairs[2][1]  # same grouping as with tools


# ===================================================================================
# parsing: one call
# ===================================================================================


def test_parse_single_tool_call() -> None:
    text = _call_block("get_weather", '{"city": "Paris"}')
    out = T.parse_tool_calls(text, TOOL_NAMES, id_factory=_ids())

    assert len(out.tool_calls) == 1
    call = out.tool_calls[0]
    assert call.name == "get_weather"
    assert call.index == 0
    assert call.id == "call_0000"
    # A response that is only a call has no content at all -- null, not "".
    assert out.content is None


def test_parse_single_arguments_is_a_json_string_not_an_object() -> None:
    """OpenAI sends `function.arguments` as a string and every client calls json.loads on
    it; handing back an object breaks them all."""
    out = T.parse_tool_calls(
        _call_block("get_weather", '{"city": "Paris", "unit": "C"}'),
        TOOL_NAMES,
        id_factory=_ids(),
    )
    args = out.tool_calls[0].arguments
    assert isinstance(args, str)
    assert json.loads(args) == {"city": "Paris", "unit": "C"}

    # A call with no arguments still gets a valid JSON object string, not "" or null.
    no_args = T.parse_tool_calls(
        '<tool_call>\n{"name": "get_time"}\n</tool_call>', TOOL_NAMES, id_factory=_ids()
    )
    assert json.loads(no_args.tool_calls[0].arguments) == {}


# ===================================================================================
# parsing: several calls, and surrounding text
# ===================================================================================


def test_parse_multi_and_text_keeps_surrounding_text() -> None:
    text = (
        "I'll check both.\n"
        + _call_block("get_weather", '{"city": "Paris"}')
        + "\n"
        + _call_block("get_time", "{}")
        + "\nBack in a moment."
    )
    out = T.parse_tool_calls(text, TOOL_NAMES, id_factory=_ids())

    assert [(c.index, c.name, c.arguments) for c in out.tool_calls] == [
        (0, "get_weather", '{"city": "Paris"}'),
        (1, "get_time", "{}"),
    ]
    # Every id distinct: they are what a client echoes back as tool_call_id.
    assert len({c.id for c in out.tool_calls}) == 2
    assert out.content is not None
    assert "I'll check both." in out.content
    assert "Back in a moment." in out.content
    # No trace of the call syntax leaks into content.
    assert "<tool_call>" not in out.content
    assert "get_weather" not in out.content


def test_parse_multi_and_text_content_is_none_without_text() -> None:
    """Only the padding the model puts around its calls, so there is no content."""
    text = (
        "\n"
        + _call_block("get_weather", '{"city": "Paris"}')
        + "\n"
        + _call_block("get_weather", '{"city": "Rome"}')
        + "\n"
    )
    out = T.parse_tool_calls(text, TOOL_NAMES, id_factory=_ids())
    assert len(out.tool_calls) == 2
    assert out.content is None

    # But with no calls at all, text passes through byte-for-byte, trailing newline and
    # all -- a plain answer must not be reshaped just because tools were offered.
    plain = T.parse_tool_calls("Paris is in France.\n", TOOL_NAMES, id_factory=_ids())
    assert plain.tool_calls == []
    assert plain.content == "Paris is in France.\n"
    assert T.parse_tool_calls("", TOOL_NAMES).content == ""


# ===================================================================================
# parsing: streaming
# ===================================================================================


def _stream(text: str, chunks: list[str] | None = None, names=TOOL_NAMES):
    """Feed `text` through the incremental parser and assemble the deltas the way an SSE
    client does: concatenate content, group tool calls by `index`."""
    parser = T.ToolCallStreamParser(names, id_factory=_ids())
    pieces = chunks if chunks is not None else list(text)
    content_parts: list[str] = []
    slots: dict[int, dict] = {}

    def take(text_delta, calls):
        if text_delta:
            content_parts.append(text_delta)
        for c in calls:
            slot = slots.setdefault(c.index, {"id": None, "name": None, "arguments": ""})
            slot["id"] = slot["id"] or c.id
            slot["name"] = slot["name"] or c.name
            slot["arguments"] += c.arguments

    for piece in pieces:
        take(*parser.push(piece))
    take(*parser.flush())

    content = "".join(content_parts)
    calls = [slots[i] for i in sorted(slots)]
    return (content if (content or not calls) else None), calls


def _one_shot(text: str, names=TOOL_NAMES):
    out = T.parse_tool_calls(text, names, id_factory=_ids())
    return out.content, [
        {"id": c.id, "name": c.name, "arguments": c.arguments} for c in out.tool_calls
    ]


def test_parse_streaming_char_by_char_matches_one_shot() -> None:
    """Assembling the deltas has to reconstruct exactly what the non-streaming path
    returns -- for every prefix, since generation can stop anywhere."""
    text = (
        "Checking.\n"
        + _call_block("get_weather", '{"city": "Paris"}')
        + "\n"
        + _call_block("get_time", "{}")
    )
    assert _stream(text) == _one_shot(text)

    for cut in range(len(text) + 1):
        prefix = text[:cut]
        assert _stream(prefix) == _one_shot(prefix), f"diverged at cut={cut}: {prefix!r}"


def test_parse_streaming_splits_tags_across_chunks() -> None:
    """Tokenisers split `<tool_call>` into pieces (`<tool`, `_call`, `>`). A parser that
    decided per piece would leak the opening tag into content and then miss the call."""
    text = _call_block("get_weather", '{"city": "Paris"}')
    chunks = ["<tool", "_call", ">", "\n", '{"name":', ' "get_weather",', ' "arguments":',
              ' {"city":', ' "Paris"}}', "\n</tool", "_call", ">"]
    assert "".join(chunks) == text
    content, calls = _stream(text, chunks)
    assert content is None
    assert [(c["name"], c["arguments"]) for c in calls] == [
        ("get_weather", '{"city": "Paris"}')
    ]
    assert _stream(text, chunks) == _one_shot(text)


def test_parse_streaming_deltas_carry_index() -> None:
    """`index` is how a client accumulates fragments into the right call."""
    parser = T.ToolCallStreamParser(TOOL_NAMES, id_factory=_ids())
    seen: list[T.ParsedToolCall] = []
    for piece in [
        "Sure.",
        _call_block("get_weather", '{"city": "Paris"}'),
        _call_block("get_time", "{}"),
    ]:
        _, calls = parser.push(piece)
        seen.extend(calls)
    assert [c.index for c in seen] == [0, 1]
    assert parser.n_calls == 2

    # And the wire frames the route builds keep it through `exclude_none` serialisation,
    # which is what strips absent delta fields.
    from bwr.server.app import _tool_call_delta
    from bwr.server.schemas import ChunkChoice, Delta

    payload = ChunkChoice(
        delta=Delta(tool_calls=[_tool_call_delta(seen[1])])
    ).model_dump(exclude_none=True)
    entry = payload["delta"]["tool_calls"][0]
    assert entry["index"] == 1
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "get_time"
    assert entry["function"]["arguments"] == "{}"
    assert "content" not in payload["delta"]


# ===================================================================================
# parsing: robustness
# ===================================================================================


def test_parse_malformed_json_degrades_to_text() -> None:
    """GGML_ASSERT calls abort() and the server is one process: a bad call has to become
    text, never an exception."""
    broken = '<tool_call>\n{"name": "get_weather", "arguments": {"city": }\n</tool_call>'
    out = T.parse_tool_calls(broken, TOOL_NAMES, id_factory=_ids())
    assert out.tool_calls == []
    # Nothing is dropped: the block comes back verbatim so the caller can see it.
    assert out.content == broken

    for junk in (
        "<tool_call>\nnot json at all\n</tool_call>",
        "<tool_call>\n[1, 2, 3]\n</tool_call>",
        "<tool_call>\n{}\n</tool_call>",
        '<tool_call>\n{"name": 42}\n</tool_call>',
        '<tool_call>\n{"arguments": {}}\n</tool_call>',
        "<tool_call>\n\n</tool_call>",
    ):
        got = T.parse_tool_calls(junk, TOOL_NAMES, id_factory=_ids())
        assert got.tool_calls == [], junk
        assert got.content == junk, junk
        assert _stream(junk) == _one_shot(junk), junk


def test_parse_malformed_doubled_braces_are_the_models_own_instruction() -> None:
    """The block the model is *shown* wraps its example object in doubled braces (they
    are inside a jinja string literal in the template, so they render verbatim -- see
    `test_prompt_injection_appends_to_existing_system`). A 0.5B model copies that shape
    literally, and the copy is not valid JSON. Refusing it would hand the client
    `<tool_call>` text for a call the model plainly made, so the duplicated outermost
    brace is dropped -- and nothing else is."""
    for body in (
        '{{"name": "get_weather", "arguments": {"city": "Paris"}}',   # doubled open only
        '{{"name": "get_weather", "arguments": {"city": "Paris"}}}',  # doubled both ends
    ):
        text = f"<tool_call>\n{body}\n</tool_call>"
        out = T.parse_tool_calls(text, TOOL_NAMES, id_factory=_ids())
        assert [(c.name, c.arguments) for c in out.tool_calls] == [
            ("get_weather", '{"city": "Paris"}')
        ], body
        assert out.content is None, body
        # The repair lives in the shared state machine, so streaming cannot diverge.
        assert _stream(text) == _one_shot(text), body

    # The leniency is exactly one brace deep and never invents a parse: everything that
    # still is not an object degrades to text as before.
    for junk in (
        '<tool_call>\n{{{"name": "get_weather"}}}\n</tool_call>',
        '<tool_call>\n{{"name": "get_weather", "arguments": {"city": }}\n</tool_call>',
        '<tool_call>\n{{"nope": 1}}\n</tool_call>',
        '<tool_call>\n{{"name": "rm_rf"}}\n</tool_call>',  # still an unoffered tool
    ):
        got = T.parse_tool_calls(junk, TOOL_NAMES, id_factory=_ids())
        assert got.tool_calls == [], junk
        assert got.content == junk, junk
        assert _stream(junk) == _one_shot(junk), junk


def test_parse_malformed_unterminated_block_degrades_to_text() -> None:
    """max_tokens can cut a response mid-call. The partial block is still tokens the
    client paid for, so it is returned as text rather than swallowed."""
    text = 'Checking.\n<tool_call>\n{"name": "get_weather", "argum'
    out = T.parse_tool_calls(text, TOOL_NAMES, id_factory=_ids())
    assert out.tool_calls == []
    assert out.content is not None
    assert "Checking." in out.content
    assert '{"name": "get_weather", "argum' in out.content

    # A good call followed by a truncated one keeps the good one.
    mixed = _call_block("get_time", "{}") + "\n<tool_call>\n{\"name\": \"get_wea"
    got = T.parse_tool_calls(mixed, TOOL_NAMES, id_factory=_ids())
    assert [c.name for c in got.tool_calls] == ["get_time"]
    assert got.content is not None and '"get_wea' in got.content
    assert _stream(mixed) == _one_shot(mixed)


def test_parse_malformed_unknown_tool_degrades_to_text() -> None:
    """A call naming a tool that was never offered is not dispatchable, so it is not a
    tool call -- emitting it would make the client key an unknown function."""
    text = _call_block("rm_rf", '{"path": "/"}')
    out = T.parse_tool_calls(text, TOOL_NAMES, id_factory=_ids())
    assert out.tool_calls == []
    assert out.content == text

    # Offered tools are still recognised in the same response.
    mixed = text + "\n" + _call_block("get_time", "{}")
    got = T.parse_tool_calls(mixed, TOOL_NAMES, id_factory=_ids())
    assert [c.name for c in got.tool_calls] == ["get_time"]
    assert got.content is not None and "rm_rf" in got.content

    # CONTRACT CHANGE: known_names=None used to mean "accept anything", while both
    # routes were already using None to mean the opposite -- "do not parse at all" (see
    # app._parsing_names). One sentinel with two opposite meanings let /v1/messages,
    # which forwarded it without a guard, return a tool_use block for a request that
    # offered no tools. None now means OFF and nothing else, so an unoffered call is
    # text through this path too, and "parse but accept any name" no longer exists.
    off = T.parse_tool_calls(text, None, id_factory=_ids())
    assert off.tool_calls == []
    assert off.content == text


def test_parse_malformed_never_raises_on_fuzzed_input() -> None:
    """Sweep every prefix of a hostile string through both paths."""
    hostile = (
        "<tool_call></tool_call>"
        "</tool_call>text<tool_call><tool_call>\n"
        '{"name": "get_weather", "arguments": "not-an-object"}\n</tool_call>'
        '<tool_call>\n{"name": "get_time", "arguments": null}\n</tool_call>'
        "<tool_call>\n{\"name\": \"\xe2\x98\x83\"}\n</tool_call>"
        "<tool_call"
    )
    for cut in range(len(hostile) + 1):
        prefix = hostile[:cut]
        one = _one_shot(prefix)
        assert _stream(prefix) == one, f"diverged at cut={cut}"
        content, calls = one
        for call in calls:
            assert call["name"] in TOOL_NAMES
            assert isinstance(call["arguments"], str)
        if calls and content is not None:
            assert content.strip(), "content must be null, not whitespace"

    # Arguments the model sent as a bare string are passed through as-is; the client's
    # json.loads decides, we do not guess.
    weird = T.parse_tool_calls(
        '<tool_call>\n{"name": "get_time", "arguments": "raw"}\n</tool_call>',
        TOOL_NAMES,
        id_factory=_ids(),
    )
    assert weird.tool_calls[0].arguments == "raw"


# ===================================================================================
# HTTP surface
# ===================================================================================


@pytest.fixture(scope="module")
def client(model: bwr.Model):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from bwr.server.app import build_app

    # A tools block is a few hundred tokens on its own, and n_ctx_seq is n_ctx/n_seq_max
    # (see docs/llamacpp-notes.md), so the per-sequence budget has to be real here.
    app = build_app(
        model, bwr.EngineConfig(n_ctx=4096, n_batch=512, n_ubatch=512, n_seq_max=2, engine="metal")
    )
    with fastapi_testclient.TestClient(app) as c:
        yield c
    assert app.state.engine.engine.ctx.closed, "lifespan must close the context"


def _check_tool_calls_shape(calls: list[dict]) -> None:
    for i, call in enumerate(calls):
        assert call["type"] == "function"
        assert isinstance(call["id"], str) and call["id"]
        fn = call["function"]
        assert fn["name"] in TOOL_NAMES
        assert isinstance(fn["arguments"], str), "arguments must be a JSON string"
        json.loads(fn["arguments"])  # must be parseable, as a client would


TOOL_PAYLOAD = {
    "model": "local",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris right now?"},
    ],
    "tools": [WEATHER_TOOL, TIME_TOOL],
    "tool_choice": "auto",
    "max_tokens": 64,
    "temperature": 0,
}


def test_http_tool_calls_non_streaming(client) -> None:
    """Deliberately conditional: a 0.5B model's decision to call is not a fact about this
    code. What IS asserted unconditionally is that the response cannot be malformed."""
    r = client.post("/v1/chat/completions", json=TOOL_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"

    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    calls = message.get("tool_calls")
    finish = body["choices"][0]["finish_reason"]

    if calls:
        _check_tool_calls_shape(calls)
        assert finish == "tool_calls"
        # Only calls and nothing else means null content, not "".
        assert message["content"] is None or message["content"].strip()
        assert "<tool_call>" not in (message["content"] or "")
    else:
        assert finish in ("stop", "length")
        assert message["content"] is not None
    assert body["usage"]["total_tokens"] > 0
    assert client.get("/health").status_code == 200


def test_http_tool_calls_streaming(client) -> None:
    with client.stream("POST", "/v1/chat/completions", json={**TOOL_PAYLOAD, "stream": True}) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    slots: dict[int, dict] = {}
    for c in chunks:
        for entry in c["choices"][0]["delta"].get("tool_calls", []):
            assert "index" in entry, "a tool-call delta without index cannot be assembled"
            slot = slots.setdefault(
                entry["index"], {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
            )
            slot["id"] = slot["id"] or entry.get("id")
            fn = entry.get("function") or {}
            slot["function"]["name"] = slot["function"]["name"] or fn.get("name")
            slot["function"]["arguments"] += fn.get("arguments") or ""

    finish = chunks[-1]["choices"][0]["finish_reason"]
    if slots:
        _check_tool_calls_shape([slots[i] for i in sorted(slots)])
        assert finish == "tool_calls"
    else:
        assert finish in ("stop", "length")
    # No call syntax leaks into the content deltas either way.
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "<tool_call>" not in text


def test_http_tool_streaming_matches_non_streaming(client) -> None:
    """Greedy decoding, same prompt: assembling the SSE deltas must reproduce the
    non-streaming body. This is what proves the two paths share a parser."""
    whole = client.post("/v1/chat/completions", json=TOOL_PAYLOAD).json()
    message = whole["choices"][0]["message"]

    with client.stream("POST", "/v1/chat/completions", json={**TOOL_PAYLOAD, "stream": True}) as r:
        lines = [ln for ln in r.iter_lines() if ln and ln != "data: [DONE]"]
    chunks = [json.loads(ln[len("data: "):]) for ln in lines]

    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    slots: dict[int, dict] = {}
    for c in chunks:
        for entry in c["choices"][0]["delta"].get("tool_calls", []):
            slot = slots.setdefault(entry["index"], {"name": None, "arguments": ""})
            fn = entry.get("function") or {}
            slot["name"] = slot["name"] or fn.get("name")
            slot["arguments"] += fn.get("arguments") or ""

    assembled_calls = [slots[i] for i in sorted(slots)]
    expected_calls = [
        {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
        for c in (message.get("tool_calls") or [])
    ]
    assert assembled_calls == expected_calls
    assert (content if (content or not assembled_calls) else None) == message["content"]
    assert whole["choices"][0]["finish_reason"] == chunks[-1]["choices"][0]["finish_reason"]


def test_http_tool_choice_none_is_served_plainly(client) -> None:
    """With calls forbidden the request is an ordinary completion: no tools block in the
    prompt, and no tool_calls key in the response."""
    r = client.post(
        "/v1/chat/completions",
        json={**TOOL_PAYLOAD, "tool_choice": "none", "max_tokens": 8},
    )
    assert r.status_code == 200, r.text
    message = r.json()["choices"][0]["message"]
    assert message.get("tool_calls") is None
    assert message["content"] is not None
    assert r.json()["choices"][0]["finish_reason"] in ("stop", "length")


def test_http_tool_result_round_trip_is_served(client) -> None:
    """The second turn of an agent loop: history contains the assistant's calls and a
    tool result. It must render and serve, not 422 or 500."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "22C, sunny"},
            ],
            "tools": [WEATHER_TOOL],
            "max_tokens": 24,
            "temperature": 0,
        },
    )
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    if choice["message"].get("tool_calls"):
        _check_tool_calls_shape(choice["message"]["tool_calls"])
        assert choice["finish_reason"] == "tool_calls"
    else:
        assert choice["message"]["content"] is not None
    assert client.get("/health").status_code == 200


def test_http_tool_bad_tools_payload_is_a_4xx_not_a_crash(client) -> None:
    """A malformed `tools` array is the client's error; the process must survive it."""
    for bad in ([{"type": "function"}], [{"type": "function", "function": {}}], ["nope"]):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": bad,
                "max_tokens": 4,
            },
        )
        assert 400 <= r.status_code < 500, f"{bad} -> {r.status_code}: {r.text}"
    assert client.get("/health").status_code == 200
