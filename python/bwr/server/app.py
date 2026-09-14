"""Single-process FastAPI app over one AsyncEngine.

No ZMQ, no worker processes. Big White Rabbit's multi-process topology exists to give each
tensor-parallel CUDA rank its own process and to keep a stream-owning scheduler off the
event loop; on unified memory there is one GPU and one context, and the decode loop is
already isolated on a thread (see engine/async_engine.py). So the whole server is one
process, and the "submit to backend" step is a direct call instead of an IPC hop.

Crash isolation is the tradeoff: a segfault in the engine takes the API down with it.
That is answered at the OS level (launchd/systemd restarting `bwr serve`), not by
rebuilding Big White Rabbit's in-app supervisor -- which is also why every binding raises
instead of aborting (see docs/llamacpp-notes.md).
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import APIRouter, FastAPI, HTTPException, Request

from .anthropic_api import register_anthropic_routes
from .common import count_tokens, sse_response, submit_request, text_from_blocks
from .params import build_params

from .._bwr_metal import Model
from ..engine.async_engine import AsyncEngine
from ..engine.config import (
    EngineConfig,
    RequestParams,
    StopSequenceFilter,
    normalize_stops,
)
from ..engine.metal_engine import MetalEngine
from ..engine.pool import ModelPoolError
from ..engine.mlx_engine import MLXEngine
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

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 512

# Single source for engine->protocol reason mapping (see server/reasons.py).
# Re-exported here so `from server.app import _FINISH_REASONS` keeps working
# for tests and so the two surfaces cannot drift.
from .reasons import FINISH_REASONS as _FINISH_REASONS  # noqa: F401


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
    # Delegates to the shared helper so the ``type in (None, "text")``
    # predicate cannot drift between surfaces (see server/common.py).
    return text_from_blocks(msg.content)  # type: ignore[arg-type]


def _message_pairs(req: ChatCompletionRequest) -> list[tuple[str, str]]:
    """Flatten the conversation to `(role, text)` for the templater.

    Tool rendering is gated on `req.tools` OR replayed tool history: a follow-up
    turn may omit `tools` yet still carry `tool` roles / `tool_calls`, and those
    must still render as `<tool_response>` / `<tool_call>` so the model sees the
    trained shape. A pure chat (no `tools`, no `tool` roles, no `tool_calls`)
    still flattens exactly as Phase 2 did.
    """
    has_tool_history = any(
        mm.role == "tool" or (mm.role == "assistant" and mm.tool_calls)
        for mm in req.messages
    )
    tool_turn = bool(req.tools) or has_tool_history
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


def render_pairs(
    model: Model | MLXEngine, pairs: list[tuple[str, str]]
) -> str:
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


def _render_prompt(model: Model | MLXEngine, req: ChatCompletionRequest) -> str:
    """Render a ChatCompletionRequest to the raw prompt string for the engine."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    pairs = _message_pairs(req)
    if req.tools:
        # Tools cannot be handed to llama_chat_apply_template (no argument for them, and
        # it pattern-matches templates instead of running jinja), so they go into the
        # system message text before templating. See server/tools.py.
        pairs = inject_tools(pairs, req.tools, req.tool_choice)
    return render_pairs(model, pairs)


# Public alias -- backlog asks to settle on one name; keep both so callers
# and tests can use either spelling without churn.
render_prompt = _render_prompt


def _request_params(req: ChatCompletionRequest) -> RequestParams:
    # Shared builder so a new sampling field cannot be added to one surface and
    # forgotten on the other (see server/params.py).
    return build_params(
        max_tokens=req.resolved_max_tokens(DEFAULT_MAX_TOKENS),
        # `stop` may be a bare string or a list of them; the engine takes the normalised
        # tuple so it can stop decoding past the delimiter instead of merely having its
        # output truncated here.
        stop=normalize_stops(req.stop),
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
    )


def _parsing_names(req: ChatCompletionRequest) -> set[str] | None:
    """Tool names to accept in the output, or None meaning "do not parse at all".

    Not parsing when no tools were offered is deliberate: a request without `tools` must
    behave exactly as it did before tool calling existed, even if the model spontaneously
    emits something that looks like a call. Same for `tool_choice="none"` -- the client
    said it will not dispatch calls, so `<tool_call>` text is just text.

    Follow-up contract (intentional asymmetry with `_message_pairs`): history still
    renders as `<tool_call>` / `<tool_response>` when `tools` is omitted but prior
    turns carry `tool_calls` / `tool` roles, so the model sees the trained shape;
    however new output is NOT parsed into calls unless `tools` is declared this turn.
    Parsing a call the client never offered would hand it a dispatch it cannot honor.

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
    model: Model | None,
    config: EngineConfig | None = None,
    *,
    served_model_name: str | None = None,
    draft_model: Model | None = None,
    mlx_model_path: str | None = None,
    model_dir: str | None = None,
    preload: str | None = None,
) -> FastAPI:
    config = config or EngineConfig()
    # Pool mode: `model_dir` replaces the single model/mlx_model_path pair.
    # Engines load lazily per request, so startup does not pay for weights
    # nobody has asked for yet (see engine/pool.py).
    pool = None
    if model_dir is not None:
        if model is not None or mlx_model_path is not None:
            raise ValueError(
                "model_dir is mutually exclusive with model/mlx_model_path"
            )
        from ..engine.pool import ModelPool

        pool = ModelPool(model_dir, config)
    if pool is None and config.engine not in ("metal", "mlx"):
        raise ValueError(
            f"engine must be 'metal' or 'mlx'; got {config.engine!r}"
        )
    if pool is not None:
        engine = None
        async_engine = None
        renderer = None
        ids = [m["id"] for m in pool.list()]
        model_name = served_model_name or (ids[0] if ids else "none")
    elif config.engine == "mlx":
        if draft_model is not None:
            raise ValueError("draft_model is a Metal-backend option; unset it with engine='mlx'")
        if mlx_model_path is None:
            raise ValueError("engine='mlx' needs mlx_model_path (directory of MLX weights)")
        from ..engine.mlx_engine import MLXEngine

        engine = MLXEngine(mlx_model_path, config)
        model_name = served_model_name or mlx_model_path.rstrip("/").rsplit("/", 1)[-1] or "local"
    else:
        if model is None:
            raise ValueError("engine='metal' needs a loaded Model")
        engine = MetalEngine(model, config, draft_model=draft_model)
        model_name = served_model_name or model.meta_val("general.name") or "local"
    if pool is None:
        # Prompt rendering + token counting go through whichever object owns a
        # tokenizer: the llama Model, or the MLXEngine itself (same two methods).
        renderer = model if model is not None else engine
        async_engine = AsyncEngine(engine)

    async def resolve(requested: str | None):
        """(model_id, engine, renderer) for one request.

        Constant in single-model mode; in pool mode this is where a load can
        happen, so it is awaited inside the handler rather than captured at
        registration.
        """
        if pool is None:
            return model_name, async_engine, renderer
        mid, eng = await pool.acquire(requested)
        raw = pool.engine_for(mid)
        # MetalEngine delegates tokenization to its Model; MLXEngine owns it.
        return mid, eng, (getattr(raw, "model", None) or raw)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The engine thread starts here rather than in build_app so it is bound to the
        # loop that will actually serve requests.
        if async_engine is not None:
            await async_engine.start()
        if preload and pool is not None:
            # Pool mode loads lazily, so the FIRST request pays the whole
            # 15-19 GiB weight load. Preloading moves that cost to startup,
            # which is what a desktop app wants: the server reports healthy
            # only once it can actually answer quickly.
            #
            # A failure here must NOT abort startup -- a bad --preload name
            # or a model too large for the host would otherwise leave the
            # user with no server at all rather than a working one missing a
            # warm model.
            target = preload
            if preload == "first":
                # NOT None: resolve(None) only succeeds when the pool holds
                # exactly one model, so passing None here failed with
                # "unknown model None" on any real multi-model directory.
                listed = pool.list()
                target = listed[0]["id"] if listed else None
            try:
                if target is None:
                    raise ValueError(f"no models found under {model_dir}")
                mid, _engine = await pool.acquire(target)
                logger.info("preloaded %s", mid)
            except Exception as exc:  # noqa: BLE001 - reported; must not kill the server
                logger.warning("preload of %r failed: %s", preload, exc)
        try:
            yield
        finally:
            if pool is not None:
                await pool.stop_all()
            if async_engine is not None:
                await async_engine.stop()

    app = FastAPI(title="Big White Rabbit", lifespan=lifespan)
    app.state.engine = async_engine
    app.state.pool = pool
    app.state.resolve = resolve

    # Web UI at /admin (derived from oMLX, Apache-2.0 -- see
    # bwr/server/webui/__init__.py for provenance and changes). Optional: a
    # trimmed install without jinja2 still serves the API.
    try:
        from .webui.routes import mount as _mount_webui

        _mount_webui(app, engine, model_name, pool)
        app.state.webui = True
    except ImportError:  # noqa: BLE001 - jinja2/static deps absent; API still serves
        app.state.webui = False

    @app.get("/health")
    async def health() -> dict[str, object]:
        if pool is not None:
            # Read-only: never triggers a load, so /health stays cheap even
            # when nothing is resident.
            return {
                "status": "ok",
                "mode": "pool",
                "models": len(pool.list()),
                "loaded": pool.loaded_ids,
                "resident_bytes": pool.resident_bytes,
                "budget_bytes": pool.budget_bytes,
            }
        body: dict[str, object] = {
            "status": "ok",
            "model": model_name,
            "n_ctx": engine.ctx.n_ctx,
            "n_ctx_seq": engine.ctx.n_ctx_seq,
            "n_seq_max": engine.ctx.n_seq_max,
            "free_seq_slots": engine.n_free_seq_slots,
            "decode_calls": engine.ctx.decode_calls,
        }
        hotlist = getattr(engine, "_hotlist", None)
        if hotlist is not None:
            body["hotlist"] = {
                "hits": hotlist.hits,
                "misses": hotlist.misses,
                "hit_rate": hotlist.hit_rate(),
            }
        return body

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        if pool is not None:
            return ModelList(data=[ModelCard(id=m["id"]) for m in pool.list()])
        return ModelList(data=[ModelCard(id=model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, http_request: Request):
        if req.n is not None and req.n != 1:
            raise HTTPException(status_code=400, detail="only n=1 is supported")
        # Before anything is rendered or admitted: a non-positive cap is unsatisfiable,
        # and the engine's `n_generated >= max_tokens` would read it as "retire after
        # one token" and answer with a plausible-looking 200. Same check, same 400, as
        # /v1/messages -- see schemas.max_tokens_error for why it is here and not on the
        # model.
        cap_error = req.cap_error()
        if cap_error is not None:
            raise HTTPException(status_code=400, detail=cap_error)

        try:
            served, engine_r, renderer_r = await resolve(req.model)
        except ModelPoolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        prompt = _render_prompt(renderer_r, req)
        params = _request_params(req)
        known_tools = _parsing_names(req)
        n_prompt = count_tokens(renderer_r, prompt)
        request_id = await submit_request(engine_r, prompt, params)

        if req.stream:
            return sse_response(
                _sse(
                    engine_r,
                    request_id,
                    served,
                    http_request,
                    known_tools,
                    params.stop,
                ),
            )

        # The same filter object the SSE path runs, fed the same pieces in the same
        # order, so the assembled body cannot drift from the streamed text -- the reason
        # `parse_tool_calls` is the streaming parser fed one chunk, applied to the other
        # thing that now withholds text. With no `stop` it is a pass-through, so this
        # path is byte-identical to what it produced before.
        stopper = StopSequenceFilter(params.stop)
        try:
            parts: list[str] = []
            n_completion = 0
            finish_reason = "stop"
            async for out in engine_r.stream(request_id):
                if await http_request.is_disconnected():
                    await engine_r.cancel(request_id)
                    break
                # The engine's natural-end retirement (eog) yields an empty
                # piece and does not append to output_tokens / n_generated,
                # so counting it would make `completion_tokens` off by one
                # (10 vs 9). Only count tokens that carried text.
                if not (out.finished and out.piece == "" and out.finish_reason == "eog"):
                    n_completion += 1
                emittable, hit = stopper.push(out.piece)
                parts.append(emittable)
                if hit:
                    # Stop consuming: the engine retires on this same token, and on a
                    # stream that does not (a replayed one) `release` cancels it. The
                    # reason goes through the same table as the engine's own, so the
                    # two cannot name this condition differently.
                    finish_reason = _openai_finish_reason("stop_sequence")
                    break
                if out.finished:
                    finish_reason = _openai_finish_reason(out.finish_reason)
            else:
                parts.append(stopper.flush())
        finally:
            engine_r.release(request_id)

        text = "".join(parts)
        content: str | None = text
        tool_calls: list[ToolCall] | None = None
        try:
            # `known_tools is None` means parsing is off, which the parser implements as a
            # pass-through (content == text, no calls) -- so the pre-tools response shape
            # is preserved without a second guard restating what "off" means here.
            parsed = parse_tool_calls(text, known_tools)
        except Exception:  # noqa: BLE001 - a parse failure must degrade, not 500
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
            # the model actually served, which in pool mode is not the
            # startup default
            model=served,
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
    anthropic_router = APIRouter()
    register_anthropic_routes(
        anthropic_router,
        model=renderer,
        async_engine=async_engine,
        model_name=model_name,
        resolve=resolve,
        render_prompt=lambda pairs: render_pairs(renderer, pairs),
    )
    app.include_router(anthropic_router)

    return app


async def _sse(
    engine: AsyncEngine,
    request_id: int,
    model_name: str,
    http_request: Request,
    known_tools: set[str] | None = None,
    stops: tuple[str, ...] = (),
) -> AsyncIterator[bytes]:
    """Emit OpenAI-shaped SSE frames, then `[DONE]`."""
    completion_id = _rid("chatcmpl")
    # Two things now withhold text on this one stream, and the order is load-bearing:
    # the stop filter runs FIRST, upstream of the tool parser. A stop sequence is
    # defined over the model's raw output -- that is what the engine scans to stop
    # decoding -- so scanning anything else here could disagree with the engine about
    # where the turn ended. Feeding the parser only post-filter text also means it never
    # sees a byte the client must not receive, so nothing ever has to be un-sent; each
    # stage withholds only its own trailing partial, and the composition is a plain
    # pipeline of prefix-preserving filters.
    stopper = StopSequenceFilter(stops)
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
        except Exception:  # noqa: BLE001 - degrade to raw text rather than tearing down
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
            text, hit = stopper.push(out.piece)
            final = hit or out.finished
            if final and not hit:
                # Generation ended without the delimiter: release whatever the filter
                # was holding back for a match that never completed.
                text += stopper.flush()
            if text or final:
                for f in tool_frames(text, final=final):
                    yield f
            if final:
                reason = _openai_finish_reason(
                    "stop_sequence" if hit else out.finish_reason
                )
                if parser.n_calls:
                    reason = "tool_calls"
                yield frame(ChunkChoice(delta=Delta(), finish_reason=reason))
            if hit:
                break  # `release` in the finally cancels whatever is still in flight
        yield b"data: [DONE]\n\n"
    finally:
        engine.release(request_id)
