"""Shared helpers for both protocol surfaces.

Two duplications lived across ``app.py`` and ``anthropic_api.py``:

* **Route preamble** -- both ``/v1/chat/completions`` and ``/v1/messages``
  did the same four steps verbatim: ``model.tokenize`` to count prompt
  tokens, ``async_engine.submit`` with ``ValueError -> 400`` /
  ``RuntimeError -> 503`` mapping, and the ``text/event-stream``
  ``Cache-Control``/``X-Accel-Buffering`` headers for the streaming
  response. A new header or error case added to one surface and not the
  other would drift, which is how the two surfaces came to disagree about
  the generation cap (see ``schemas.max_tokens_error``).

* **text_from_blocks** -- the loop that flattens a list of typed blocks
  to text (``type in (None, "text")``) was copied 4x: ``app._message_text``
  plus ``anthropic_api._system_text``, ``_tool_result_text`` (inner
  ``content`` list) and the ``text`` branch of ``_message_pairs``.
  One function owns the predicate so the four call sites cannot drift.

This module is the single source for both, following the same pattern as
``server.reasons`` (stop-reason maps) and ``server.params`` (RequestParams
construction). Callers keep their protocol-specific wrappers (e.g. how the
prompt is rendered) and delegate only the shared tail.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# --- SSE framing ----------------------------------------------------------------

SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}
SSE_MEDIA_TYPE: str = "text/event-stream"


def sse_response(iterator: AsyncIterator[bytes]) -> StreamingResponse:
    """Wrap an async SSE iterator with the shared media type and headers.

    Both surfaces use ``text/event-stream`` with ``no-cache`` and
    ``X-Accel-Buffering: no`` so a proxy does not buffer the stream.
    One helper owns the headers so a new header cannot be added to one
    route and forgotten on the other.
    """
    return StreamingResponse(
        iterator,
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


# --- text_from_blocks -----------------------------------------------------------


def text_from_blocks(
    blocks: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> str:
    """Flatten a list of typed blocks to text.

    Anthropic's ``system`` and ``content`` fields and OpenAI's
    ``content`` parts both allow a list of ``{type, text}`` blocks where
    anything that is not a text block (image, audio) has no meaning for a
    text-only engine. The predicate is ``type in (None, "text")``: an
    absent ``type`` is the legacy spelling of a text part (see
    ``app._message_text``), so it is treated as text rather than dropped.

    Non-dict entries and non-string ``text`` values are skipped rather
    than stringified into the prompt as JSON noise.
    """
    if not blocks:
        return ""
    out: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") in (None, "text"):
            text = block.get("text")
            if isinstance(text, str):
                out.append(text)
    return "".join(out)


def block_text(block: dict[str, Any]) -> str | None:
    """Text of a single block if it is a text block, else None.

    Convenience for the branching loop in ``_message_pairs`` where the
    iterator must also handle ``tool_use`` / ``tool_result`` blocks. The
    predicate is identical to ``text_from_blocks`` so the two cannot
    disagree.
    """
    if isinstance(block, dict) and block.get("type") in (None, "text"):
        text = block.get("text")
        if isinstance(text, str):
            return text
    return None


# --- route preamble -------------------------------------------------------------


def count_tokens(model: Any, prompt: str) -> int:
    """Count prompt tokens using the model's tokenizer.

    Both surfaces tokenize with ``add_special=True, parse_special=True``
    so the count matches what the engine will actually encode. One helper
    owns the flags so a new flag cannot be added to one route and not
    the other.
    """
    return len(model.tokenize(prompt, add_special=True, parse_special=True))


async def submit_request(engine: Any, prompt: str, params: Any) -> int:
    """Submit a prompt to the engine, mapping admission errors to HTTP.

    ``ValueError`` (bad request, e.g. prompt too long) becomes 400 and
    ``RuntimeError`` (exhausted sequence slots) becomes 503. The mapping
    is owned here so the two surfaces cannot name the same condition
    differently, and so a future engine error case has to be added in one
    place.
    """
    try:
        return await engine.submit(prompt, params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
