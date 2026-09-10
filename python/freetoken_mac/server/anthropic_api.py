"""The Anthropic Messages API surface: POST /v1/messages.

Shares everything below the protocol with the OpenAI surface -- the same engine, the same
prompt injection, and the same tool-call parser. The translation happens at the edges:

  * Anthropic tool declarations (`{name, description, input_schema}`) are converted to
    the OpenAI shape before rendering, so the `<tools>` block the model sees is
    byte-identical whichever API the client used. The model's chat template was trained
    on the OpenAI-shaped object, so the wire format of the REQUEST must not leak into
    the prompt.
  * Tool calls come back out of the shared parser with `arguments` as a JSON string
    (OpenAI's spelling) and are inflated to an `input` object here (Anthropic's).

That is what keeps the two surfaces from drifting: one prompt format, one parser, two
serialisations.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request

from .._freetoken_metal import Model
from ..engine.async_engine import AsyncEngine
from ..engine.config import RequestParams, StopSequenceFilter, normalize_stops
from . import anthropic_schemas as A
from .common import block_text, count_tokens, sse_response, submit_request, text_from_blocks
from .params import build_params
from .schemas import max_tokens_error
from .tools import (
    ParsedToolCall,
    ToolCallStreamParser,
    inject_tools,
    parse_tool_calls,
    render_assistant_turn,
    render_tool_response,
    resolve_tool_choice,
    tool_names,
)

# Single source -- see server/reasons.py. Re-exported so
# `from server.anthropic_api import _STOP_REASONS` keeps working.
from .reasons import STOP_REASONS as _STOP_REASONS  # noqa: F401


def _system_text(system: str | list[dict[str, Any]] | None) -> str:
    """Flatten the top-level `system` field, which may be a string or a block list."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    # Delegates to the shared helper so the predicate cannot drift (see server/common.py).
    return text_from_blocks(system)


def _tool_result_text(block: dict[str, Any]) -> str:
    """The text of a tool_result block, whose content may itself be blocks."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return text_from_blocks(content)
    if content is None:
        return ""
    # A client may put a bare object there; serialise rather than drop the result.
    return json.dumps(content, ensure_ascii=False)


def _message_pairs(req: A.MessagesRequest) -> list[tuple[str, str]]:
    """Anthropic messages -> `(role, text)` pairs for the chat templater.

    Two shapes need real translation rather than flattening:
      * an ASSISTANT turn's `tool_use` blocks must be re-rendered as `<tool_call>` syntax,
        or the assistant's side of a tool exchange vanishes from the prompt;
      * a USER turn's `tool_result` blocks become `<tool_response>` wrapped text, and a
        run of them collapses into ONE user turn, as the template does.
    """
    pairs: list[tuple[str, str]] = []
    system = _system_text(req.system)
    if system:
        pairs.append(("system", system))

    for msg in req.messages:
        if isinstance(msg.content, str):
            pairs.append((msg.role, msg.content))
            continue

        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[str] = []
        for block in msg.content:
            # Text blocks delegate to the shared predicate so the
            # ``type in (None, "text")`` check lives in one place.
            t = block_text(block) if isinstance(block, dict) else None
            if t is not None:
                texts.append(t)
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                name = block.get("name")
                if isinstance(name, str) and name:
                    # Hand the shared renderer the OpenAI shape it expects; it accepts an
                    # object for `arguments` and serialises it itself.
                    tool_calls.append(
                        {
                            "function": {
                                "name": name,
                                "arguments": block.get("input") or {},
                            }
                        }
                    )
            elif btype == "tool_result":
                tool_results.append(render_tool_response(_tool_result_text(block)))

        body = "".join(texts)
        if msg.role == "assistant":
            rendered = render_assistant_turn(body, tool_calls)
            if rendered:
                pairs.append(("assistant", rendered))
        else:
            # Results first: they answer the previous assistant turn, and the template
            # emits them as their own user turn.
            if tool_results:
                merged = "\n".join(tool_results)
                if body:
                    merged = f"{merged}\n{body}"
                pairs.append(("user", merged))
            elif body:
                pairs.append(("user", body))
    return pairs


def _openai_tool_choice(choice: Any) -> Any:
    """Anthropic's tool_choice -> the OpenAI spelling.

    Translating rather than branching keeps ONE injection path: `inject_tools` derives
    the mode itself via `tools.resolve_tool_choice`, so re-deriving it here would be a
    second copy of that logic waiting to disagree with the first.

    Anthropic says `any` for "call something" and `tool` + `name` for "call this one",
    where OpenAI says `required` and a named function. Unknown spellings fall through to
    `auto`, matching the OpenAI surface's permissiveness.
    """
    if choice is None:
        return None
    holder = choice if isinstance(choice, dict) else choice.model_dump(exclude_none=True)
    ctype = str(holder.get("type") or "auto").lower()
    if ctype == "none":
        return "none"
    if ctype == "any":
        return "required"
    if ctype == "tool":
        name = holder.get("name")
        if isinstance(name, str) and name:
            return {"type": "function", "function": {"name": name}}
        return "required"
    return "auto"


def _openai_tools(req: A.MessagesRequest) -> list[dict[str, Any]]:
    return [t.to_openai_shape() for t in (req.tools or [])]


def _parsing_names(
    oai_tools: list[dict[str, Any]], oai_choice: Any
) -> set[str] | None:
    """Tool names to accept in the output, or None meaning "do not parse at all".

    The same rule as the OpenAI surface's `app._parsing_names`, on the translated
    request: no `tools` means this is not a tool turn, and `tool_choice: none` means the
    client will not dispatch calls -- either way `<tool_call>` text the model produced
    anyway (replaying a prior tool exchange primes it to) is just text, and turning it
    into a `tool_use` block would hand the client a call it never offered to make.

    Follow-up contract: history (`tool_use` / `tool_result` blocks) still renders into
    the prompt even when `tools` is omitted, so the model sees context; new output is
    NOT parsed unless tools are declared this turn.

    `None` is safe to pass straight to the parser: it is the parser's own spelling of
    "off". It used to also be its spelling of "accept ANY name", which is how this route
    ended up emitting tool_use blocks for a request with no tools at all.
    """
    if not oai_tools:
        return None
    mode, _forced = resolve_tool_choice(oai_choice)
    if mode == "none":
        return None
    return tool_names(oai_tools)


def _request_params(req: A.MessagesRequest) -> RequestParams:
    return build_params(
        max_tokens=req.max_tokens,
        stop=normalize_stops(req.stop_sequences),
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )


def _input_object(call: ParsedToolCall) -> dict[str, Any]:
    """`arguments` (a JSON string, OpenAI's spelling) -> `input` (an object).

    A call whose arguments are not a JSON object still has to serialise: Anthropic's
    `input` is typed as an object, so an unparseable body is surfaced under a key rather
    than dropped or 500-ing.
    """
    try:
        parsed = json.loads(call.arguments)
    except (TypeError, ValueError):
        return {"_raw": call.arguments}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def register_anthropic_routes(
    router: APIRouter,
    *,
    model: Model,
    async_engine: AsyncEngine,
    model_name: str,
    render_prompt,
) -> None:
    """Mount /v1/messages. `render_prompt` is injected rather than reimplemented so the
    Anthropic surface cannot drift from the OpenAI one on prompt construction."""

    @router.post("/v1/messages")
    async def create_message(req: A.MessagesRequest, http_request: Request):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        # The same rule the OpenAI surface applies, from the same implementation: two
        # copies of "the cap must be positive" would be free to drift, which is exactly
        # how the two surfaces came to disagree about it (this route 400'd; the other
        # served a one-token 200). The status stays 400, as it has always been here.
        cap_error = max_tokens_error(("max_tokens", req.max_tokens))
        if cap_error is not None:
            raise HTTPException(status_code=400, detail=cap_error)

        pairs = _message_pairs(req)
        if not pairs:
            raise HTTPException(status_code=400, detail="messages contained no content")

        oai_tools = _openai_tools(req)
        oai_choice = _openai_tool_choice(req.tool_choice)
        # inject_tools is a no-op for "none" or an empty tool list, so this is the same
        # single path the OpenAI route takes.
        pairs = inject_tools(pairs, oai_tools, oai_choice)
        known = _parsing_names(oai_tools, oai_choice)

        prompt = render_prompt(pairs)
        params = _request_params(req)
        n_prompt = count_tokens(model, prompt)
        request_id = await submit_request(async_engine, prompt, params)

        # Read back off the params the ENGINE was given, so the sequences the engine
        # stops on and the ones the wire truncates on cannot be two different lists.
        stops = params.stop
        if req.stream:
            return sse_response(
                _stream(
                    async_engine,
                    request_id,
                    model_name,
                    known,
                    n_prompt,
                    http_request,
                    stops,
                ),
            )

        # Same filter, same pieces, same order as the streaming path below: that is what
        # makes the assembled body equal the streamed text rather than resemble it.
        stopper = StopSequenceFilter(stops)
        try:
            parts: list[str] = []
            n_completion = 0
            reason = "eog"
            async for out in async_engine.stream(request_id):
                if await http_request.is_disconnected():
                    await async_engine.cancel(request_id)
                    break
                if not (out.finished and out.piece == "" and out.finish_reason == "eog"):
                    n_completion += 1
                emittable, hit = stopper.push(out.piece)
                parts.append(emittable)
                if hit:
                    reason = "stop_sequence"
                    break  # `release` cancels anything still in flight
                if out.finished:
                    reason = out.finish_reason or "eog"
            else:
                parts.append(stopper.flush())
        finally:
            async_engine.release(request_id)

        raw = "".join(parts)
        try:
            parsed = parse_tool_calls(raw, known, id_factory=A.new_tool_use_id)
        except Exception:  # noqa: BLE001 - a parse failure must degrade, not 500
            parsed = None

        blocks: list[A.TextBlock | A.ToolUseBlock] = []
        stop_reason = _STOP_REASONS.get(reason, "end_turn")
        if parsed is not None and parsed.tool_calls:
            if parsed.content:
                blocks.append(A.TextBlock(text=parsed.content))
            for call in parsed.tool_calls:
                blocks.append(
                    A.ToolUseBlock(id=call.id, name=call.name, input=_input_object(call))
                )
            stop_reason = "tool_use"
        else:
            text = parsed.content if parsed is not None else raw
            # Anthropic always returns at least one block; an empty generation is an
            # empty text block, not an empty list.
            blocks.append(A.TextBlock(text=text or ""))

        return A.MessagesResponse(
            content=blocks,
            model=model_name,
            stop_reason=stop_reason,
            # Which sequence matched, the field Anthropic clients read to learn which
            # delimiter ended the turn. Only ever set together with the matching
            # stop_reason: a turn that ended as tool_use did not end at a delimiter.
            stop_sequence=stopper.matched if stop_reason == "stop_sequence" else None,
            usage=A.Usage(input_tokens=n_prompt, output_tokens=n_completion),
        )


async def _stream(
    engine: AsyncEngine,
    request_id: int,
    model_name: str,
    known: set[str] | None,
    n_prompt: int,
    http_request: Request,
    stops: tuple[str, ...] = (),
) -> AsyncIterator[bytes]:
    """Emit the Anthropic event sequence.

    Text is streamed incrementally through the shared parser (so `<tool_call>` syntax is
    withheld rather than leaking into a text block); `known=None` makes that parser a
    pass-through, because it means the client offered no tools -- or said it will not
    dispatch calls -- so there is nothing to withhold. Tool calls are emitted as their
    own blocks once complete: their arguments are only known to be well-formed after the
    closing tag, and streaming `input_json_delta` fragments before that would force a
    client to handle a call we might still reject.
    """
    message_id = A.new_message_id()
    parser = ToolCallStreamParser(known, id_factory=A.new_tool_use_id)
    # Upstream of the tool parser, for the reason spelled out in app._sse: a stop
    # sequence is defined over the model's raw output (which is what the engine scans),
    # and filtering before parsing means the parser never holds a byte the client must
    # not receive.
    stopper = StopSequenceFilter(stops)

    index = 0
    text_open = False
    n_completion = 0
    stop_reason = "end_turn"

    try:
        yield A.sse(
            "message_start",
            A.MessageStartEvent(
                message=A.MessageStartMessage(
                    id=message_id,
                    model=model_name,
                    usage=A.Usage(input_tokens=n_prompt, output_tokens=0),
                )
            ).model_dump(),
        )

        def open_text() -> bytes:
            return A.sse(
                "content_block_start",
                A.ContentBlockStartEvent(
                    index=index, content_block=A.TextBlock(text="")
                ).model_dump(),
            )

        def emit_call(call: ParsedToolCall, idx: int) -> list[bytes]:
            """A whole tool_use block: start, one input_json_delta, stop."""
            frames = [
                A.sse(
                    "content_block_start",
                    A.ContentBlockStartEvent(
                        index=idx,
                        content_block=A.ToolUseBlock(id=call.id, name=call.name, input={}),
                    ).model_dump(),
                ),
                A.sse(
                    "content_block_delta",
                    A.ContentBlockDeltaEvent(
                        index=idx,
                        delta=A.InputJsonDelta(
                            partial_json=json.dumps(
                                _input_object(call), ensure_ascii=False
                            )
                        ),
                    ).model_dump(),
                ),
                A.sse(
                    "content_block_stop",
                    A.ContentBlockStopEvent(index=idx).model_dump(),
                ),
            ]
            return frames

        async for out in engine.stream(request_id):
            if await http_request.is_disconnected():
                await engine.cancel(request_id)
                return

            if not (out.finished and out.piece == "" and out.finish_reason == "eog"):
                n_completion += 1
            piece, hit = stopper.push(out.piece)
            try:
                text, calls = parser.push(piece)
            except Exception:  # noqa: BLE001 - degrade to raw text
                text, calls = piece, []

            if text:
                if not text_open:
                    yield open_text()
                    text_open = True
                yield A.sse(
                    "content_block_delta",
                    A.ContentBlockDeltaEvent(
                        index=index, delta=A.TextDelta(text=text)
                    ).model_dump(),
                )

            for call in calls:
                if text_open:
                    yield A.sse(
                        "content_block_stop",
                        A.ContentBlockStopEvent(index=index).model_dump(),
                    )
                    text_open = False
                    index += 1
                for frame in emit_call(call, index):
                    yield frame
                index += 1

            if hit:
                # Through the table, as the engine's own reason is: one name for one
                # condition, whichever layer noticed it.
                stop_reason = _STOP_REASONS["stop_sequence"]
                break  # `release` in the finally cancels anything still in flight
            if out.finished:
                stop_reason = _STOP_REASONS.get(out.finish_reason or "eog", "end_turn")

        # Release the tail: first whatever the stop filter withheld for a match that
        # never completed, then whatever the tool parser is still holding (a partial tag
        # that never completed). Filter before parser here too, so the last bytes go
        # through the same two stages in the same order as every byte before them.
        try:
            held = stopper.flush()
        except Exception:  # noqa: BLE001 - a held tail must not tear down the response
            held = ""
        try:
            tail, calls = parser.push(held)
            flushed, more = parser.flush()
            tail, calls = tail + flushed, calls + more
        except Exception:  # noqa: BLE001
            tail, calls = held, []
        if tail:
            if not text_open:
                yield open_text()
                text_open = True
            yield A.sse(
                "content_block_delta",
                A.ContentBlockDeltaEvent(
                    index=index, delta=A.TextDelta(text=tail)
                ).model_dump(),
            )
        if text_open:
            yield A.sse(
                "content_block_stop", A.ContentBlockStopEvent(index=index).model_dump()
            )
            text_open = False
            index += 1
        for call in calls:
            for frame in emit_call(call, index):
                yield frame
            index += 1

        # No block at all (empty generation): Anthropic still sends one text block.
        if index == 0 and parser.n_calls == 0:
            yield open_text()
            yield A.sse(
                "content_block_stop", A.ContentBlockStopEvent(index=0).model_dump()
            )

        if parser.n_calls:
            stop_reason = "tool_use"

        yield A.sse(
            "message_delta",
            A.MessageDeltaEvent(
                delta=A.MessageDeltaBody(
                    stop_reason=stop_reason,
                    # As in the non-streaming body: named only when the delimiter is
                    # what ended the turn.
                    stop_sequence=(
                        stopper.matched if stop_reason == "stop_sequence" else None
                    ),
                ),
                usage={"output_tokens": n_completion},
            ).model_dump(),
        )
        yield A.sse("message_stop", A.MessageStopEvent().model_dump())
    finally:
        engine.release(request_id)
