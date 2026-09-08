"""OpenAI-compatible request/response schemas.

Field names and shapes follow the OpenAI chat-completions API so existing clients (the
`openai` SDK, Claude Code, Codex, curl) work unchanged. Modelled on FreeToken's
``server/api_models.py``, which is pure pydantic with no engine coupling; trimmed here to
the surface Phase 2 actually serves. Tool calling and the Anthropic surface land later.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def _rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    # OpenAI allows content parts (a list of {type, text, ...}) as well as a bare string;
    # normalisation to text happens in the route, not here, so the schema stays faithful.
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


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

    def resolved_max_tokens(self, default: int) -> int:
        return self.max_completion_tokens or self.max_tokens or default


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None


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


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


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
