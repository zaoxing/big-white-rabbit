"""Tool calling: prompt injection and tool-call extraction.

Two facts about llama.cpp shape everything here.

**1. `llama_chat_apply_template` cannot render tools.** Its signature is
`(tmpl, chat, n_msg, add_ass, buf, length)` -- there is no tools argument, and it
pattern-matches ~56 known templates instead of evaluating jinja. So the declarations
cannot be handed to the templater the way `transformers` hands them to jinja; they have
to be *injected into the message text* before templating.

**2. The wire format is not ours to choose.** It is fixed by the template embedded in the
model's own GGUF, because that is what the weights were trained against. The strings
below are transcribed verbatim out of Qwen2.5-Instruct's `tokenizer.chat_template`:
tools go into the system message as a `<tools>` block, calls come back as
`<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>`. Deviating would still
"work" as prompt engineering, but the model would be reading something it never saw.

Parsing is a re-implementation of the behaviour of FreeToken's `Qwen25Detector`
(`freetoken/server/function_call_parser.py`, Apache-2.0; see NOTICE) against the same
format. No code is copied: FreeToken streams *partial* JSON arguments and needs
`partial_json_parser` plus a 900-line reasoning parser for it, whereas here the one-shot
and streaming paths are deliberately the *same* state machine (`parse_tool_calls` is one
`push` plus a `flush`) so that assembling the SSE deltas cannot drift from the
non-streaming body.

Robustness is a hard requirement, not a nicety: `GGML_ASSERT` calls `abort()` and the
server is a single process, so a malformed call must degrade, never raise. Anything that
does not parse as a call naming an *offered* tool is passed through as ordinary text.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..engine.config import partial_stop_len

# --- the format, verbatim from Qwen2.5's chat_template ------------------------------

TOOLS_PREFIX = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query."
    "\n\nYou are provided with function signatures within <tools></tools> XML tags:"
    "\n<tools>"
)
TOOLS_SUFFIX = (
    "\n</tools>\n\nFor each function call, return a json object with function name and"
    " arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>'
)
# The braces around that example object are **doubled** on purpose. In the template they
# sit inside a jinja *string literal*
# (`{{- "...\\n<tool_call>\\n{{\\"name\\": ...}}\\n</tool_call><|im_end|>\\n" }}`), so they
# are not `{{ }}` delimiters and pass through the renderer verbatim: every runtime that
# renders this GGUF's chat_template (transformers, vLLM, SGLang) shows the model the
# doubled form, and that is what it was trained on. Single braces would read as a
# plausible near-miss while silently being a different instruction --
# `tests/test_phase3_tools.py` derives its expectation from `Model.chat_template` so that
# this cannot be re-transcribed wrongly.

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def new_call_id() -> str:
    """OpenAI's `call_...` id shape; clients echo it back as `tool_call_id`."""
    return f"call_{uuid.uuid4().hex[:24]}"


# --- request-side helpers -----------------------------------------------------------


def _as_dict(obj: Any) -> dict[str, Any]:
    """Accept either a pydantic model or a plain dict for a tool declaration.

    Tests and internal callers pass dicts; the route passes `schemas.Tool`. Dropping
    None-valued fields matters: `transformers` renders the tool dict the client actually
    sent, so emitting `"description": null` would put noise in the prompt that no real
    OpenAI server puts there.
    """
    if isinstance(obj, Mapping):
        raw = dict(obj)
    else:
        dump = getattr(obj, "model_dump", None)
        if dump is None:
            raise TypeError(f"not a tool declaration: {obj!r}")
        raw = dump(exclude_none=True)
    return {k: v for k, v in raw.items() if v is not None}


def resolve_tool_choice(tool_choice: Any) -> tuple[str, str | None]:
    """Normalise `tool_choice` to `(mode, forced_name)`.

    Anything unrecognised falls back to `"auto"` rather than 4xx-ing: an unknown value
    should behave as if it had not been sent (the pre-tools behaviour), which is also
    what keeps a client sending a newer OpenAI spelling from breaking outright.
    """
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        mode = tool_choice.strip().lower()
        return (mode, None) if mode in ("auto", "none", "required") else ("auto", None)
    # {"type": "function", "function": {"name": ...}}
    holder = tool_choice if isinstance(tool_choice, Mapping) else None
    if holder is not None:
        fn = holder.get("function")
    else:
        fn = getattr(tool_choice, "function", None)
    name = None
    if isinstance(fn, Mapping):
        name = fn.get("name")
    elif fn is not None:
        name = getattr(fn, "name", None)
    if isinstance(name, str) and name:
        return "required", name
    return "auto", None


def tool_names(tools: Sequence[Any] | None) -> set[str]:
    """The set of function names the caller actually offered.

    Used to reject a hallucinated call: a model naming a tool that was never declared
    has not made a call the client can dispatch, so it is text, not a `tool_calls` entry.
    """
    names: set[str] = set()
    for tool in tools or ():
        try:
            fn = _as_dict(tool).get("function")
        except TypeError:
            continue
        if isinstance(fn, Mapping) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def render_tools_block(
    tools: Sequence[Any],
    *,
    mode: str = "auto",
    forced_name: str | None = None,
) -> str:
    """The `<tools>...</tools>` declaration block, ready to append to system content.

    One tool object per line as JSON, exactly as the template's
    `{%- for tool in tools %}{{- "\\n" }}{{- tool | tojson }}` does -- the *whole* tool
    (`{"type": "function", "function": {...}}`), not just the inner function, because
    that is what jinja is handed and therefore what the model was trained on.
    """
    parts = [TOOLS_PREFIX]
    for tool in tools:
        try:
            payload = _as_dict(tool)
        except TypeError:
            continue
        # ensure_ascii=False and default separators match transformers' `tojson` filter.
        parts.append("\n" + json.dumps(payload, ensure_ascii=False))
    parts.append(TOOLS_SUFFIX)
    # There is no way to *constrain* decoding to a call here (that needs a grammar), so
    # "required" is expressed the only way the prompt can express it. Kept outside the
    # verbatim block so the trained-on text stays byte-exact.
    if mode == "required":
        if forced_name:
            parts.append(
                f'\n\nYou must call the function "{forced_name}" '
                "before replying to the user."
            )
        else:
            parts.append(
                "\n\nYou must call at least one of the provided functions "
                "before replying to the user."
            )
    return "".join(parts)


def inject_tools(
    pairs: Sequence[tuple[str, str]],
    tools: Sequence[Any] | None,
    tool_choice: Any = None,
) -> list[tuple[str, str]]:
    """Return `(role, text)` pairs with the tool declarations folded into the system turn.

    Appended after any existing system content; a conversation with no system message
    gets one. `tool_choice="none"` injects nothing at all -- telling a model about tools
    it is forbidden to call is the one way to guarantee a wasted turn.
    """
    out = [(role, text) for role, text in pairs]
    mode, forced_name = resolve_tool_choice(tool_choice)
    if not tools or mode == "none":
        return out
    block = render_tools_block(tools, mode=mode, forced_name=forced_name)
    if out and out[0][0] == "system":
        role, text = out[0]
        out[0] = (role, text + block)
    else:
        # No system turn to append to. The Qwen template would substitute its own
        # "You are Qwen, created by Alibaba Cloud." here; hardcoding one vendor's
        # identity into a server that serves arbitrary GGUFs would be worse than the
        # leading blank lines, so the block is used on its own with them trimmed.
        out.insert(0, ("system", block.lstrip("\n")))
    return out


def render_assistant_turn(text: str, tool_calls: Sequence[Any] | None) -> str:
    """Re-render a past assistant turn's `tool_calls` back into `<tool_call>` blocks.

    A client replaying its history sends the calls back as structured JSON; the model
    only understands them in its own syntax, so the round trip has to be closed here or
    the assistant's side of a multi-turn tool exchange vanishes from the prompt.
    """
    if not tool_calls:
        return text
    blocks: list[str] = []
    for call in tool_calls:
        try:
            payload = _as_dict(call)
        except TypeError:
            continue
        fn = payload.get("function")
        if not isinstance(fn, Mapping):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            # OpenAI sends arguments as a JSON *string*; the template's `| tojson` wants
            # the object, so re-inflate it and fall back to the raw text if it is not
            # valid JSON (a half-written call in replayed history must not 500).
            try:
                args_json = json.dumps(json.loads(args), ensure_ascii=False)
            except (TypeError, ValueError):
                args_json = args if args.strip() else "{}"
        elif args is None:
            args_json = "{}"
        else:
            try:
                args_json = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args_json = "{}"
        # json.dumps neutralizes `"` / `\` in replayed names so a crafted
        # history entry cannot break the call object (normal names unchanged).
        blocks.append(
            f"{TOOL_CALL_OPEN}\n{{\"name\": {json.dumps(name)}, \"arguments\": {args_json}}}"
            f"\n{TOOL_CALL_CLOSE}"
        )
    if not blocks:
        return text
    body = "\n".join(blocks)
    # The template guards the content with `{%- if message.content %}{{- '\\n' + ... }}`
    # and only then emits `'\n<tool_call>'` per call, so a call-only assistant turn has
    # NO leading newline: `<|im_start|>assistant\n<tool_call>`. Emitting one would put a
    # blank line where the model was trained to see the call begin.
    return f"{text}\n{body}" if text else body


def render_tool_response(text: str) -> str:
    """Wrap a `role: "tool"` result the way the template does, as a user turn.

    Neutralizes `</tool_response` in the result so a tool output cannot close
    the block early and inject fake turns. Normal results (no delimiter) are
    byte-identical.
    """
    safe = text.replace("</tool_response", "<\\/tool_response")
    return f"{TOOL_RESPONSE_OPEN}\n{safe}\n{TOOL_RESPONSE_CLOSE}"


# --- response-side parsing ----------------------------------------------------------


@dataclass
class ParsedToolCall:
    """One extracted call. `arguments` is a JSON *string*, as OpenAI sends it."""

    id: str
    name: str
    arguments: str
    index: int


@dataclass
class ParsedOutput:
    content: str | None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)


def _partial_prefix_len(buf: str, token: str) -> int:
    """Length of the longest suffix of `buf` that is a proper prefix of `token`.

    Tokenisers split `<tool_call>` across pieces (`<tool`, `_call`, `>`), so a streaming
    parser that decided on each piece in isolation would leak the opening tag into
    `content` and then fail to recognise the call.

    The arithmetic is `engine.config.partial_stop_len`, which stop-sequence detection
    needs for exactly the same reason against a different set of strings. One
    implementation, so the two hold-back decisions cannot drift; the name stays because
    what it means *here* is a partial tag, not a partial delimiter.
    """
    return partial_stop_len(buf, (token,))


def _load_call_object(body: str) -> dict[str, Any] | None:
    """A block body as a JSON object, or None if it is not one.

    One narrow repair: the instruction text the model is shown wraps its example call in
    **doubled** braces (`{{"name": ...}}` -- see `TOOLS_SUFFIX`, where they are verbatim
    from the template), and a small model copies that shape literally, emitting either
    `{{...}}` or `{{...}` inside an otherwise perfect `<tool_call>` block. That is not
    valid JSON, so without this a call the model plainly made would be handed back to the
    client as text. Only a duplicated *outermost* brace is dropped, and only if the result
    parses to an object: nothing is guessed at, and anything else still degrades to text.
    """
    candidates = [body]
    if body.startswith("{{"):
        candidates.append(body[1:])
        if body.endswith("}}"):
            candidates.append(body[1:-1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


class ToolCallStreamParser:
    """Incremental `<tool_call>` extractor. Also backs the one-shot path.

    Text is emitted as soon as it cannot become part of a call. Whitespace-only runs are
    held back and dropped if the response turns out to be calls only -- that is what
    makes `content` `null` for a pure tool-call response through *both* paths, without
    the streaming client having to know the rule.

    `known_names` is the set of names the client OFFERED, and `None` means tool parsing
    is OFF: every byte the model produced is content, `<tool_call>` syntax included.
    There is deliberately no spelling of "parse, but accept any name". `None` used to
    mean exactly that -- while both routes were already using `None` to mean the
    opposite, "do not parse" -- so one sentinel carried two contradictory meanings, and
    the surface that forwarded it without a second guard (`/v1/messages`) emitted
    `tool_use` blocks for a request that offered no tools at all. Accepting only declared
    names is the whole point of the check (an undispatchable call is not a call), so the
    ambiguity is resolved by dropping the meaning nothing legitimately wanted.
    """

    def __init__(
        self,
        known_names: Iterable[str] | None = None,
        *,
        id_factory: Callable[[], str] = new_call_id,
    ) -> None:
        self._known: set[str] | None = None if known_names is None else set(known_names)
        self._id_factory = id_factory
        self._buffer = ""
        self._pending_ws = ""
        self._in_call = False
        self._n_calls = 0

    @property
    def enabled(self) -> bool:
        """False when this parser is the pass-through (`known_names=None`)."""
        return self._known is not None

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def _emit_text(self, s: str) -> str:
        if not s:
            return ""
        core = s.rstrip()
        if not core:
            # Might be the gap in front of a `<tool_call>`; decide later.
            self._pending_ws += s
            return ""
        out = self._pending_ws + core
        self._pending_ws = s[len(core):]
        return out

    def _build_call(self, inner: str) -> ParsedToolCall | None:
        """Turn a block body into a call, or None to mean "this was not a call"."""
        body = inner.strip()
        if not body:
            return None
        obj = _load_call_object(body)
        if obj is None:
            return None
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            return None
        if self._known is None or name not in self._known:
            # Never offered (or parsing is off): not dispatchable, so it is text.
            return None
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if args is None:
            args_json = "{}"
        elif isinstance(args, str):
            args_json = args
        else:
            try:
                args_json = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                return None
        call = ParsedToolCall(
            id=self._id_factory(), name=name, arguments=args_json, index=self._n_calls
        )
        self._n_calls += 1
        return call

    def push(self, new_text: str) -> tuple[str, list[ParsedToolCall]]:
        """Feed generated text; return `(text_delta, calls_completed_by_this_chunk)`."""
        if self._known is None:
            # Parsing off: pass the chunk straight through, byte for byte and with the
            # caller's own chunk boundaries. Nothing is buffered, so a request that
            # offered no tools streams exactly as it did before tool calling existed.
            return new_text, []
        if not new_text:
            return "", []
        self._buffer += new_text
        texts: list[str] = []
        calls: list[ParsedToolCall] = []

        while True:
            if not self._in_call:
                i = self._buffer.find(TOOL_CALL_OPEN)
                if i >= 0:
                    seg = self._buffer[:i]
                    self._buffer = self._buffer[i + len(TOOL_CALL_OPEN):]
                    texts.append(self._emit_text(seg))
                    self._in_call = True
                    continue
                hold = _partial_prefix_len(self._buffer, TOOL_CALL_OPEN)
                safe = self._buffer[: len(self._buffer) - hold] if hold else self._buffer
                self._buffer = self._buffer[len(safe):]
                texts.append(self._emit_text(safe))
                break

            j = self._buffer.find(TOOL_CALL_CLOSE)
            if j < 0:
                break  # hold the whole body until the block closes
            inner = self._buffer[:j]
            self._buffer = self._buffer[j + len(TOOL_CALL_CLOSE):]
            self._in_call = False
            call = self._build_call(inner)
            if call is None:
                # Not a usable call: hand the block back verbatim as text. `_emit_text`
                # replays any whitespace held in front of it, so nothing is lost.
                texts.append(
                    self._emit_text(TOOL_CALL_OPEN + inner + TOOL_CALL_CLOSE)
                )
            else:
                calls.append(call)
                self._pending_ws = ""  # gap before a real call is not content

        return "".join(texts), calls

    def flush(self) -> tuple[str, list[ParsedToolCall]]:
        """Finish the response: release held text, degrade an unterminated block."""
        if self._known is None:
            return "", []  # parsing off: push() held nothing back
        texts: list[str] = []
        if self._in_call:
            # `<tool_call>` that never closed -- generation was cut off by max_tokens or
            # end-of-generation. Show it as text rather than dropping the tokens.
            raw, self._buffer, self._in_call = TOOL_CALL_OPEN + self._buffer, "", False
            texts.append(self._emit_text(raw))
        elif self._buffer:
            buf, self._buffer = self._buffer, ""
            texts.append(self._emit_text(buf))
        if self._pending_ws:
            # Trailing whitespace is real content in a plain answer, but only padding
            # around a call.
            if self._n_calls == 0:
                texts.append(self._pending_ws)
            self._pending_ws = ""
        return "".join(texts), []


def parse_tool_calls(
    text: str,
    known_names: Iterable[str] | None = None,
    *,
    id_factory: Callable[[], str] = new_call_id,
) -> ParsedOutput:
    """One-shot parse: the streaming state machine fed a single chunk.

    Sharing the implementation is the point -- it is what guarantees that assembling the
    SSE deltas reproduces the non-streaming body instead of merely resembling it. That
    includes the off state: `known_names=None` yields `content == text` and no calls
    here, exactly as `push` passes the chunk through there.
    """
    parser = ToolCallStreamParser(known_names, id_factory=id_factory)
    text_a, calls_a = parser.push(text)
    text_b, calls_b = parser.flush()
    content = text_a + text_b
    calls = calls_a + calls_b
    # `null`, not `""`, when the model produced calls and nothing else: an OpenAI client
    # branches on `message.content is None` to decide it must dispatch tools.
    return ParsedOutput(content=content if (content or not calls) else None, tool_calls=calls)
