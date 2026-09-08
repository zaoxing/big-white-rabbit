"""Single-process FastAPI app over one AsyncEngine.

No ZMQ, no worker processes. FreeToken's multi-process topology exists to give each
tensor-parallel CUDA rank its own process and to keep a stream-owning scheduler off the
event loop; on unified memory there is one GPU and one context, and the decode loop is
already isolated on a thread (see engine/async_engine.py). So the whole server is one
process, and the "submit to backend" step is a direct call instead of an IPC hop.

Crash isolation is the tradeoff: a segfault in the engine takes the API down with it.
That is answered at the OS level (launchd/systemd restarting `ftm serve`), not by
rebuilding FreeToken's in-app supervisor -- which is also why every binding raises
instead of aborting (see docs/llamacpp-notes.md).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .._freetoken_metal import Model
from ..engine.async_engine import AsyncEngine
from ..engine.config import EngineConfig, RequestParams
from ..engine.metal_engine import MetalEngine
from .schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCall,
    FunctionCallDelta,
    ModelCard,
    ModelList,
    ResponseMessage,
    ToolCall,
    ToolCallDelta,
    Usage,
    _rid,
)
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

DEFAULT_MAX_TOKENS = 512

# The engine reports why a request retired in its own vocabulary; the OpenAI schema
# admits only stop / length / tool_calls / content_filter, and an SDK client validating
# the field will reject anything else. Map at the protocol boundary rather than
# renaming the engine's reasons, which are more precise and worth keeping internally.
_FINISH_REASONS = {
    "eog": "stop",        # model emitted end-of-generation
    "length": "length",   # hit max_tokens
    "context": "length",  # ran out of per-sequence KV room
    "cancelled": "stop",  # client hung up or explicit cancel
}


def _openai_finish_reason(reason: str | None) -> str:
    return _FINISH_REASONS.get(reason or "", "stop")


def _message_text(msg: ChatMessage) -> str:
    """Flatten OpenAI content parts to text.

    Clients send either a bare string or a list of typed parts; anything that is not a
    text part (an image, an audio clip) has no meaning for a text-only engine, so it is
    dropped rather than stringified into the prompt as JSON noise.
    """
    if msg.content is None:
        return ""
    if isinstance(msg.content, str):
        return msg.content
    out: list[str] = []
    for part in msg.content:
        if isinstance(part, dict) and part.get("type") in (None, "text"):
            text = part.get("text")
            if isinstance(text, str):
                out.append(text)
    return "".join(out)


def _message_pairs(req: ChatCompletionRequest) -> list[tuple[str, str]]:
    """Flatten the conversation to `(role, text)` for the templater.

    The tool-conversation rendering is gated on `req.tools`: without a declaration this
    is not a tool exchange, so the messages are flattened exactly as Phase 2 did.
    """
    tool_turn = bool(req.tools)
    messages = list(req.messages)
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if tool_turn and m.role == "tool":
            # llama.cpp's chatml renderer would emit `<|im_start|>tool`, a role the
            # weights never saw; the template folds results into a user turn instead --
            # and folds a *run* of consecutive results into a single one, opening
            # `<|im_start|>user` only when the previous message was not a tool and
            # closing it only when the next one is not. A parallel-tool-call turn
            # therefore arrives as one user message holding several <tool_response>
            # blocks; emitting one turn per result would show the model a conversation
            # shape it was never trained on.
            run: list[str] = []
            while i < len(messages) and messages[i].role == "tool":
                run.append(render_tool_response(_message_text(messages[i])))
                i += 1
            pairs.append(("user", "\n".join(run)))
            continue
        text = _message_text(m)
        if tool_turn and m.role == "assistant" and m.tool_calls:
            text = render_assistant_turn(text, m.tool_calls)
        pairs.append((m.role, text))
        i += 1
    return pairs


def render_pairs(model: Model, pairs: list[tuple[str, str]]) -> str:
    """Apply the model's chat template to prepared `(role, text)` pairs.

    Shared with the Anthropic surface (server/anthropic_api.py), which prepares its own
    pairs but must not reimplement templating or the error mapping.
    """
    try:
        return model.apply_chat_template(pairs, True)
    except (RuntimeError, ValueError) as exc:
        # No template, or one llama.cpp cannot apply. That is a request-level problem the
        # caller can act on (send a raw prompt / use another model), not a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _render_prompt(model: Model, req: ChatCompletionRequest) -> str:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    pairs = _message_pairs(req)
    if req.tools:
        # Tools cannot be handed to llama_chat_apply_template (no argument for them, and
        # it pattern-matches templates instead of running jinja), so they go into the
        # system message text before templating. See server/tools.py.
        pairs = inject_tools(pairs, req.tools, req.tool_choice)
    return render_pairs(model, pairs)


def _request_params(req: ChatCompletionRequest) -> RequestParams:
    params = RequestParams(max_tokens=req.resolved_max_tokens(DEFAULT_MAX_TOKENS))
    if req.temperature is not None:
        params.temp = req.temperature
    if req.top_p is not None:
        params.top_p = req.top_p
    if req.top_k is not None:
        params.top_k = req.top_k
    if req.seed is not None:
        params.seed = req.seed
    return params


def _parsing_names(req: ChatCompletionRequest) -> set[str] | None:
    """Tool names to accept in the output, or None meaning "do not parse at all".

    Not parsing when no tools were offered is deliberate: a request without `tools` must
    behave exactly as it did before tool calling existed, even if the model spontaneously
    emits something that looks like a call. Same for `tool_choice="none"` -- the client
    said it will not dispatch calls, so `<tool_call>` text is just text.

    `None` is the parser's own spelling of "off" (see tools.ToolCallStreamParser), so it
    is handed over as-is rather than branched on here; the Anthropic surface derives the
    same value the same way in `anthropic_api._parsing_names`.
    """
    if not req.tools:
        return None
    mode, _ = resolve_tool_choice(req.tool_choice)
    if mode == "none":
        return None
    return tool_names(req.tools)


def _tool_call(call: ParsedToolCall) -> ToolCall:
    return ToolCall(
        id=call.id, function=FunctionCall(name=call.name, arguments=call.arguments)
    )


def _tool_call_delta(call: ParsedToolCall) -> ToolCallDelta:
    """One delta carrying a whole call.

    OpenAI may spread a call over several deltas (id/name first, then argument
    fragments); sending it in one is a valid degenerate case of the same protocol -- a
    client accumulating by `index` gets the identical result -- and it avoids emitting
    argument fragments before the JSON is known to be well-formed, which is what would
    force a client to handle a call we later decide was malformed.
    """
    return ToolCallDelta(
        index=call.index,
        id=call.id,
        function=FunctionCallDelta(name=call.name, arguments=call.arguments),
    )


def build_app(
    model: Model,
    config: EngineConfig | None = None,
    *,
    served_model_name: str | None = None,
) -> FastAPI:
    engine = MetalEngine(model, config or EngineConfig())
    async_engine = AsyncEngine(engine)
    model_name = served_model_name or model.meta_val("general.name") or "local"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The engine thread starts here rather than in build_app so it is bound to the
        # loop that will actually serve requests.
        await async_engine.start()
        try:
            yield
        finally:
            await async_engine.stop()

    app = FastAPI(title="FreeToken-Mac", lifespan=lifespan)
    app.state.engine = async_engine

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model": model_name,
            "n_ctx": engine.ctx.n_ctx,
            "n_ctx_seq": engine.ctx.n_ctx_seq,
            "n_seq_max": engine.ctx.n_seq_max,
            "free_seq_slots": engine.n_free_seq_slots,
            "decode_calls": engine.ctx.decode_calls,
        }

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, http_request: Request):
        if req.n is not None and req.n != 1:
            raise HTTPException(status_code=400, detail="only n=1 is supported")

        prompt = _render_prompt(model, req)
        params = _request_params(req)
        known_tools = _parsing_names(req)
        n_prompt = len(model.tokenize(prompt, add_special=True, parse_special=True))

        try:
            request_id = await async_engine.submit(prompt, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            # seq_id exhaustion: the server is at capacity, which is a 503, not a bug.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if req.stream:
            return StreamingResponse(
                _sse(async_engine, request_id, model_name, http_request, known_tools),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            pieces: list[str] = []
            finish_reason = "stop"
            async for out in async_engine.stream(request_id):
                pieces.append(out.piece)
                if out.finished:
                    finish_reason = _openai_finish_reason(out.finish_reason)
        finally:
            async_engine.release(request_id)

        n_completion = len(pieces)
        text = "".join(pieces)
        content: str | None = text
        tool_calls: list[ToolCall] | None = None
        try:
            # `known_tools is None` means parsing is off, which the parser implements as a
            # pass-through (content == text, no calls) -- so the pre-tools response shape
            # is preserved without a second guard restating what "off" means here.
            parsed = parse_tool_calls(text, known_tools)
        except Exception:
            # Nothing in the parser is supposed to raise, but the whole server is one
            # process: falling back to the raw text costs a tool call, whereas a 500
            # here would also mean the engine slot was burned for nothing.
            parsed = None
        if parsed is not None:
            content = parsed.content
            if parsed.tool_calls:
                tool_calls = [_tool_call(c) for c in parsed.tool_calls]
                # The engine's reasons are eog/length/context/cancelled; "tool_calls"
                # exists only at the protocol boundary, so it is set here.
                finish_reason = "tool_calls"
        return ChatCompletion(
            model=model_name,
            choices=[
                Choice(
                    message=ResponseMessage(content=content, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=n_prompt,
                completion_tokens=n_completion,
                total_tokens=n_prompt + n_completion,
            ),
        )

    # --- Anthropic Messages surface -------------------------------------------------
    # Mounted on the same app so one server serves both protocols against one loaded
    # model. `render_pairs` is passed in rather than reimplemented: prompt construction
    # is the one thing the two surfaces must never disagree about.
    from fastapi import APIRouter

    from .anthropic_api import register_anthropic_routes

    anthropic_router = APIRouter()
    register_anthropic_routes(
        anthropic_router,
        model=model,
        async_engine=async_engine,
        model_name=model_name,
        render_prompt=lambda pairs: render_pairs(model, pairs),
    )
    app.include_router(anthropic_router)

    return app


async def _sse(
    engine: AsyncEngine,
    request_id: int,
    model_name: str,
    http_request: Request,
    known_tools: set[str] | None = None,
) -> AsyncIterator[bytes]:
    """Emit OpenAI-shaped SSE frames, then `[DONE]`."""
    completion_id = _rid("chatcmpl")
    # Same state machine the non-streaming path runs, so the assembled deltas cannot
    # drift from the whole-response body. `known_tools is None` means tool parsing is off
    # for this request, and the parser answers that by passing every chunk through
    # untouched -- so this path needs no "no parser" variant of itself.
    parser = ToolCallStreamParser(known_tools)

    def frame(choice: ChunkChoice) -> bytes:
        chunk = ChatCompletionChunk(id=completion_id, model=model_name, choices=[choice])
        # exclude_none matches OpenAI's wire shape: an absent delta field is omitted, not
        # sent as null. Emitting {"content": null} makes the common client idiom
        # `delta.get("content", "")` yield None and blow up on concatenation.
        # finish_reason is re-added explicitly below because OpenAI *does* send it as
        # null on every non-final chunk, and some clients switch on its presence.
        payload = chunk.model_dump(exclude_none=True)
        for ch in payload["choices"]:
            ch.setdefault("finish_reason", None)
            ch.setdefault("delta", {})
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    def tool_frames(piece: str, *, final: bool) -> list[bytes]:
        """Run `piece` through the parser and turn its output into frames.

        Text may lag the tokens here: a chunk that could still turn out to be the start
        of `<tool_call>` is withheld until it is decided, which is the only way a call
        split across token pieces can be recognised at all. With parsing off nothing is
        withheld, so each piece becomes exactly one content frame.
        """
        try:
            text, calls = parser.push(piece)
            if final:
                tail, _ = parser.flush()
                text += tail
        except Exception:
            # See the non-streaming path: degrade to raw text rather than tearing down
            # the response (or the process) over a parse.
            text, calls = piece, []
        out: list[bytes] = []
        if text:
            out.append(frame(ChunkChoice(delta=Delta(content=text))))
        for call in calls:
            out.append(
                frame(ChunkChoice(delta=Delta(tool_calls=[_tool_call_delta(call)])))
            )
        return out

    try:
        # First frame announces the role and carries no content, as OpenAI does.
        yield frame(ChunkChoice(delta=Delta(role="assistant")))
        async for out in engine.stream(request_id):
            # A client that hangs up mid-stream should stop costing us decodes.
            if await http_request.is_disconnected():
                await engine.cancel(request_id)
                return
            if out.piece or out.finished:
                for f in tool_frames(out.piece, final=out.finished):
                    yield f
            if out.finished:
                reason = _openai_finish_reason(out.finish_reason)
                if parser.n_calls:
                    reason = "tool_calls"
                yield frame(ChunkChoice(delta=Delta(), finish_reason=reason))
        yield b"data: [DONE]\n\n"
    finally:
        engine.release(request_id)
