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
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
    _rid,
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


def _render_prompt(model: Model, req: ChatCompletionRequest) -> str:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    pairs = [(m.role, _message_text(m)) for m in req.messages]
    try:
        return model.apply_chat_template(pairs, True)
    except (RuntimeError, ValueError) as exc:
        # No template, or one llama.cpp cannot apply. That is a request-level problem the
        # caller can act on (send a raw prompt / use another model), not a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
                _sse(async_engine, request_id, model_name, http_request),
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
        return ChatCompletion(
            model=model_name,
            choices=[
                Choice(
                    message=ResponseMessage(content="".join(pieces)),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=n_prompt,
                completion_tokens=n_completion,
                total_tokens=n_prompt + n_completion,
            ),
        )

    return app


async def _sse(
    engine: AsyncEngine,
    request_id: int,
    model_name: str,
    http_request: Request,
) -> AsyncIterator[bytes]:
    """Emit OpenAI-shaped SSE frames, then `[DONE]`."""
    completion_id = _rid("chatcmpl")

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

    try:
        # First frame announces the role and carries no content, as OpenAI does.
        yield frame(ChunkChoice(delta=Delta(role="assistant")))
        async for out in engine.stream(request_id):
            # A client that hangs up mid-stream should stop costing us decodes.
            if await http_request.is_disconnected():
                await engine.cancel(request_id)
                return
            if out.piece:
                yield frame(ChunkChoice(delta=Delta(content=out.piece)))
            if out.finished:
                yield frame(
                    ChunkChoice(
                        delta=Delta(),
                        finish_reason=_openai_finish_reason(out.finish_reason),
                    )
                )
        yield b"data: [DONE]\n\n"
    finally:
        engine.release(request_id)
