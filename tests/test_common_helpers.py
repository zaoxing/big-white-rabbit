"""Unit coverage for server/common.py single-source helpers.

Pins the exact contracts Agent A consolidated so a future drift fails fast:
- block_text matches text_from_blocks([b]) predicate (type in (None,"text"))
- non-dict / non-string text values are skipped, not stringified
- sse_response media type + headers
- count_tokens uses add_special/parse_special flags
- submit_request maps ValueError->400, RuntimeError->503
"""

from __future__ import annotations

import pytest


def test_block_text_matches_text_from_blocks() -> None:
    from freetoken_mac.server.common import block_text, text_from_blocks

    cases = [
        ({"text": "hi"}, "hi"),
        ({"type": "text", "text": "hi"}, "hi"),
        ({"type": "image_url", "image_url": {"url": "x"}}, None),
        ({"type": "image", "text": "x"}, None),
        ({"type": "text", "text": 123}, None),
        ({"type": "text"}, None),
    ]
    for block, expected in cases:
        assert block_text(block) == expected, block
        # Consistency: text_from_blocks([b]) is expected or "" when None
        flat = text_from_blocks([block])
        assert flat == (expected or ""), (block, flat)

    # Non-dict entries are skipped, not stringified as JSON noise
    assert text_from_blocks([{"type": "text", "text": "a"}, "nope", 123, None]) == "a"
    assert text_from_blocks(None) == ""
    assert text_from_blocks([]) == ""


def test_sse_response_headers() -> None:
    from freetoken_mac.server.common import SSE_HEADERS, SSE_MEDIA_TYPE, sse_response

    async def empty():
        if False:
            yield b""

    resp = sse_response(empty())
    assert resp.media_type == SSE_MEDIA_TYPE == "text/event-stream"
    for k in ("Cache-Control", "X-Accel-Buffering"):
        assert k in SSE_HEADERS
        assert k in resp.headers


def test_count_tokens_uses_special_flags() -> None:
    from freetoken_mac.server.common import count_tokens

    seen: dict = {}

    class FakeModel:
        def tokenize(self, prompt, add_special=False, parse_special=False):
            seen["add_special"] = add_special
            seen["parse_special"] = parse_special
            return [1, 2, 3]

    assert count_tokens(FakeModel(), "hi") == 3
    assert seen == {"add_special": True, "parse_special": True}


def test_submit_request_maps_errors() -> None:
    import asyncio

    from fastapi import HTTPException

    from freetoken_mac.server.common import submit_request

    class OkEngine:
        async def submit(self, prompt, params):
            return 7

    assert asyncio.run(submit_request(OkEngine(), "hi", object())) == 7

    class BadEngine:
        async def submit(self, prompt, params):
            raise ValueError("prompt too long")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(submit_request(BadEngine(), "hi", object()))
    assert exc.value.status_code == 400

    class FullEngine:
        async def submit(self, prompt, params):
            raise RuntimeError("no slots")

    with pytest.raises(HTTPException) as exc2:
        asyncio.run(submit_request(FullEngine(), "hi", object()))
    assert exc2.value.status_code == 503
