"""OpenAI-compatible request/response schemas.

Field names and shapes follow the OpenAI chat-completions API so existing clients (the
`openai` SDK, Claude Code, Codex, curl) work unchanged. Modelled on FreeToken's
``server/api_models.py``, which is pure pydantic with no engine coupling; trimmed here to
the surface Phase 2 actually serves, plus the Phase 3 tool-calling fields. The Anthropic
surface lands later.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer


def _rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


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

    def resolved_max_tokens(self, default: int) -> int:
        return self.max_completion_tokens or self.max_tokens or default


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
