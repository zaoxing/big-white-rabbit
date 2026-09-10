"""Shared RequestParams construction for both protocol surfaces.

Both `app._request_params` and `anthropic_api._request_params` were
building the same `RequestParams` shape with duplicated field-by-field
logic. A new sampling field added to one surface and not the other would
silently drift, which is how `seed` was already OpenAI-only. One helper
owns the mapping, the two thin wrappers supply the protocol-specific
extraction.
"""

from __future__ import annotations

from ..engine.config import RequestParams


def build_params(
    *,
    max_tokens: int,
    stop: tuple[str, ...],
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
) -> RequestParams:
    params = RequestParams(max_tokens=max_tokens, stop=stop)
    if temperature is not None:
        params.temp = temperature
    if top_p is not None:
        params.top_p = top_p
    if top_k is not None:
        params.top_k = top_k
    if seed is not None:
        params.seed = seed
    return params
