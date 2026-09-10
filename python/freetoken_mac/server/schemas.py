"""OpenAI-compatible request/response schemas.

Field names and shapes follow the OpenAI chat-completions API so existing clients (the
`openai` SDK, Claude Code, Codex, curl) work unchanged. Modelled on FreeToken's
``server/api_models.py``, which is pure pydantic with no engine coupling; trimmed here to
the surface Phase 2 actually serves, plus the Phase 3 tool-calling fields. Both OpenAI
and Anthropic surfaces are implemented.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_serializer


def _rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


# Public alias: _rid was private yet imported by app.py (see backlog).
# Keep both spellings so external importers and internal code stay compatible.
def rid(prefix: str) -> str:  # noqa: D401
    return _rid(prefix)


# --- sampling-parameter bounds -------------------------------------------------------
#
# The engine's SamplerParams is `uint32_t seed` / `int32_t top_k` in C++ (csrc/context.h),
# and pybind11 raises TypeError on a value that does not fit. That conversion happens
# deep inside admission (RequestParams.to_sampler_params, called by
# MetalEngine.add_request), which is no place for a client's out-of-range number to
# land -- it used to surface as a 500. The bound therefore belongs at the protocol
# boundary, and it is declared on the request MODEL rather than checked in the route so
# that every path parsing one of these bodies (streaming, non-streaming, and any route
# added later) is covered by construction, and so the error names the offending field
# the way a missing `messages` already does. The Anthropic surface reuses these, for the
# same reason it reuses one prompt renderer: two copies of a bound would drift.

SEED_MAX = 2**32 - 1  # uint32_t
TOP_K_MIN = -(2**31)  # int32_t
TOP_K_MAX = 2**31 - 1
# llama.cpp's LLAMA_DEFAULT_SEED -- "draw a seed for me" (see engine/config.py).
LLAMA_DEFAULT_SEED = 0xFFFFFFFF


def validate_seed(value: int | None) -> int | None:
    """Bound `seed` to uint32, translating the `-1` idiom instead of refusing it.

    BEHAVIOUR CHOICE: `seed: -1` is llama.cpp's and ollama's spelling of "give me a
    random seed", and clients send it as a matter of course, so it is NORMALISED to
    LLAMA_DEFAULT_SEED -- which is exactly what the engine's own default means. Any other
    value outside the C++ field's range is a client bug with no defensible reading (which
    seed did they mean?), so it is rejected rather than silently truncated.
    """
    if value is None:
        return None
    if value == -1:
        return LLAMA_DEFAULT_SEED
    if not 0 <= value <= SEED_MAX:
        raise ValueError(
            f"seed must be between 0 and {SEED_MAX} inclusive, or -1 for a random seed; "
            f"got {value}"
        )
    return value


def validate_top_k(value: int | None) -> int | None:
    """Bound `top_k` to int32.

    Negative values stay legal: `top_k <= 0` is llama.cpp's "no top-k truncation", and
    -1 is the conventional spelling of it, so only values the C++ field cannot hold are
    refused.
    """
    if value is None:
        return None
    if not TOP_K_MIN <= value <= TOP_K_MAX:
        raise ValueError(
            f"top_k must be between {TOP_K_MIN} and {TOP_K_MAX} inclusive; got {value}"
        )
    return value


# --- the generation cap --------------------------------------------------------------
#
# PLACEMENT: checked in the ROUTE and answered with 400, deliberately NOT declared on
# the request model like the seed/top_k bounds above. Three reasons:
#
#   * those bounds exist because the value cannot FIT the engine's C++ field, which is
#     a constraint on the VALUE and so belongs on the field. "Generate at least one
#     token" is a constraint on the REQUEST: there is no generation that satisfies a cap
#     of zero, whatever type holds it.
#   * the OpenAI surface has two spellings of the cap and a precedence rule between them
#     (`resolved_max_tokens`), so the check has to see both fields at once and has to
#     name whichever one actually wins. A `field_validator` sees one field in isolation.
#   * 400 is the status both the real OpenAI API and this server's own /v1/messages
#     already return for it (`anthropic_api.create_message` has always had the check),
#     and tests/test_phase4_anthropic.py pins `max_tokens: 0 -> 400`. A validator would
#     make that a 422 and contradict an existing assertion for no gain.
#
# The rule itself lives here, next to the resolution it guards, so the two surfaces
# share one implementation of it -- the same reason they share `validate_top_k`.

MIN_MAX_TOKENS = 1


def max_tokens_error(*fields: tuple[str, int | None]) -> str | None:
    """The 400 detail for a non-positive generation cap, or None if the request is fine.

    Takes `(field_name, value)` pairs in PRECEDENCE order, so the field named in the
    error is the one that would actually have been used -- pointing a client at
    `max_tokens` when `max_completion_tokens` is the bad one sends them to edit a field
    that was never going to be read.

    Only a field the client actually sent is checked: `None` means "no cap of mine, use
    the server default", which is not an error.
    """
    for name, value in fields:
        if value is not None and value < MIN_MAX_TOKENS:
            return (
                f"{name} must be >= {MIN_MAX_TOKENS}; got {value}. No generation "
                f"satisfies a non-positive cap."
            )
    return None


class FunctionDef(BaseModel):
    """A tool's declaration: what the client is offering the model."""

    name: str
    description: str | None = None
    # A JSON Schema object. Kept opaque: it is forwarded into the prompt verbatim, and
    # validating a client's schema here would reject valid drafts we do not model.
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class NamedFunction(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    """`tool_choice={"type": "function", "function": {"name": "..."}}`."""

    type: Literal["function"] = "function"
    function: NamedFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    # OpenAI allows content parts (a list of {type, text, ...}) as well as a bare string;
    # normalisation to text happens in the route, not here, so the schema stays faithful.
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    # An agent loop replays its history: the assistant turn that made the calls, then a
    # `role: "tool"` result per call. Left as raw dicts on purpose -- clients differ on
    # whether `arguments` comes back as a JSON string or an object, and a strict model
    # would 422 the second turn of a conversation the first turn invited.
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    # OpenAI's newer name for max_tokens; accepted so recent clients work.
    max_completion_tokens: int | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    tools: list[Tool] | None = None
    # "auto" | "none" | "required" | {"type": "function", "function": {"name": ...}}.
    # Typed as a bare `str` rather than a Literal so an unknown spelling degrades to the
    # default (see tools.resolve_tool_choice) instead of 422-ing a request that would
    # otherwise have been served.
    tool_choice: str | NamedToolChoice | None = None

    # Range-checked here so an out-of-range value is a 4xx naming the field rather than a
    # TypeError raised inside engine admission. See validate_seed / validate_top_k.
    _bound_seed = field_validator("seed")(validate_seed)
    _bound_top_k = field_validator("top_k")(validate_top_k)

    def cap_error(self) -> str | None:
        """The 400 detail for this request's cap, or None if it is usable.

        Named differently from the module-level `max_tokens_error` it delegates to so
        that neither shadows the other: this one knows which of the request's two cap
        fields wins, and passes them in that order so the error names the field the
        client actually has to fix.
        """
        return max_tokens_error(
            ("max_completion_tokens", self.max_completion_tokens),
            ("max_tokens", self.max_tokens),
        )

    def resolved_max_tokens(self, default: int) -> int:
        """Which cap applies: the newer spelling, then the older, then the default.

        Selected on PRESENCE, not truthiness. This used to be
        `max_completion_tokens or max_tokens or default`, and 0 is falsy: an explicit
        `max_tokens: 0` was silently replaced by the server default (verified live: a
        request capped at 0 came back with 111 completion tokens), and
        `max_completion_tokens: 0` silently deferred to `max_tokens` even though the
        newer spelling is supposed to win. A cap of 0 is rejected upstream by
        `max_tokens_error`, but the resolver must be right on its own terms -- a
        resolver that cannot represent "the client sent 0" is what made that value
        invisible in the first place.
        """
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        if self.max_tokens is not None:
            return self.max_tokens
        return default


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class FunctionCall(BaseModel):
    name: str
    # A JSON *string*, not an object -- OpenAI sends it that way and every client does
    # `json.loads(call.function.arguments)`.
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_serializer(mode="wrap")
    def _omit_empty_tool_calls(self, handler):  # type: ignore[no-untyped-def]
        """Drop `tool_calls` entirely when there are none.

        OpenAI omits the key rather than sending null, and a plain completion must stay
        byte-identical to what Phase 2 emitted. `content` is *not* elided the same way:
        null content is meaningful there (it says "this turn is only tool calls").
        """
        data = handler(self)
        if data.get("tool_calls", "") is None:
            data.pop("tool_calls", None)
        return data


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = None


class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: _rid("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage


class FunctionCallDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    # `index` is what lets a client accumulate fragments into the right call; it is the
    # one field that must survive on every tool-call delta.
    index: int
    id: str | None = None
    type: Literal["function"] | None = "function"
    function: FunctionCallDelta | None = None


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "freetoken-mac"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
