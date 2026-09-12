"""Anthropic Messages API schemas.

Shapes follow the Anthropic Messages API so the `anthropic` SDK and Claude Code can
point at this server unchanged. Four differences from the OpenAI surface drive the whole
translation layer:

  1. `system` is a TOP-LEVEL field, not a message with `role: "system"`.
  2. `max_tokens` is REQUIRED, not optional.
  3. Content is a typed block LIST (`text` / `tool_use` / `tool_result`), not a string
     plus a sibling `tool_calls` array.
  4. A tool call's `input` is an OBJECT; OpenAI's `arguments` is a JSON string.

Adapted in shape from Big White Rabbit's `server/anthropic_models.py`, which is pure pydantic
with no engine coupling; trimmed to the surface this server actually serves.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# One bound, two surfaces. `top_k` reaches the same `int32_t` in the engine's
# SamplerParams whichever API it arrived through, so the range check is imported rather
# than restated -- a second copy would be free to drift from the first.
from .schemas import validate_top_k


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def new_tool_use_id() -> str:
    # Anthropic's tool_use ids are `toolu_`-prefixed; clients echo them back verbatim in
    # tool_result blocks, so only the prefix convention matters, not the body.
    return f"toolu_{uuid.uuid4().hex[:24]}"


# --- request blocks ------------------------------------------------------------------


class TextBlockParam(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlockParam(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlockParam(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    # A result may be a bare string or its own list of blocks; kept permissive because a
    # client that sent a call is entitled to send its result back in either spelling.
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


class MessageParam(BaseModel):
    # Anthropic messages carry only user/assistant. `system` is top-level, and tool
    # results arrive as a USER message whose content holds tool_result blocks.
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class ToolParam(BaseModel):
    """An Anthropic tool declaration: `{name, description?, input_schema}`.

    Note `input_schema` where OpenAI says `parameters`. `to_openai_shape` converts, so
    the prompt the model sees is the OpenAI-shaped object its chat template was trained
    on regardless of which API the client used.
    """

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None

    def to_openai_shape(self) -> dict[str, Any]:
        fn: dict[str, Any] = {"name": self.name}
        if self.description is not None:
            fn["description"] = self.description
        if self.input_schema is not None:
            fn["parameters"] = self.input_schema
        return {"type": "function", "function": fn}


class ToolChoiceParam(BaseModel):
    # "auto" | "any" | "tool" (+ name) | "none"
    type: Literal["auto", "any", "tool", "none"] = "auto"
    name: str | None = None


class MessagesRequest(BaseModel):
    model: str | None = None
    messages: list[MessageParam]
    # Required by the Anthropic API. Keeping it required means a client that omits it
    # gets a 422 naming the field, which is what the real API does.
    max_tokens: int
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[ToolParam] | None = None
    tool_choice: ToolChoiceParam | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    # Out-of-range top_k must be a 4xx naming the field, not a TypeError from inside
    # engine admission. (This surface has no `seed`; the Messages API does not define
    # one, and an unknown field is ignored, so it never reaches the sampler.)
    _bound_top_k = field_validator("top_k")(validate_top_k)


# --- response blocks ----------------------------------------------------------------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    id: str = Field(default_factory=new_message_id)
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ToolUseBlock]
    model: str
    # "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: Usage


# --- streaming events ---------------------------------------------------------------
#
# The Anthropic stream is a sequence of NAMED SSE events, not one repeated chunk shape:
#
#   message_start -> (content_block_start, content_block_delta*, content_block_stop)*
#                 -> message_delta -> message_stop
#
# Each content block gets its own index, and a tool_use block streams its arguments as
# `input_json_delta.partial_json` fragments rather than as a finished object.


class MessageStartMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[Any] = Field(default_factory=list)
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: Usage


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: MessageStartMessage


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: TextBlock | ToolUseBlock


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: TextDelta | InputJsonDelta


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaBody(BaseModel):
    stop_reason: str | None = None
    stop_sequence: str | None = None


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: MessageDeltaBody
    usage: dict[str, int]


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"


def sse(event_name: str, payload: dict[str, Any]) -> bytes:
    """One Anthropic SSE frame: a named event plus its JSON data.

    The `event:` line is load-bearing here, unlike the OpenAI stream -- the SDK dispatches
    on the event name, so omitting it yields a stream that parses as JSON but decodes to
    nothing.
    """
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    ).encode()


def anthropic_model_card(model_id: str) -> dict[str, Any]:
    return {
        "type": "model",
        "id": model_id,
        "display_name": model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
