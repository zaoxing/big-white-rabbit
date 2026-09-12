"""MLX exact-prefix cache: repeat prompt skips prefill with identical tokens.

Skips unless mlx is installed AND BWR_MLX_MODEL names an MLX directory
(same gate as test_mlx_engine.py).
"""

from __future__ import annotations

import os
import time

import pytest

MODEL_PATH = os.environ.get("BWR_MLX_MODEL")

try:
    import mlx.core  # noqa: F401
    from bwr.engine import EngineConfig, MLXEngine, RequestParams

    _MLX_OK = True
except ImportError:
    _MLX_OK = False

pytestmark = pytest.mark.skipif(
    not _MLX_OK or not MODEL_PATH or not os.path.isdir(MODEL_PATH or ""),
    reason="need mlx installed and BWR_MLX_MODEL to run MLX prefix-cache tests",
)

PROMPT = "The quick brown fox jumps over the lazy dog. " * 40  # ~360 tokens


def _once(eng, prompt=PROMPT, max_tokens=8):
    t0 = time.monotonic()
    rid = eng.add_request(
        prompt, RequestParams(max_tokens=max_tokens, temp=0.0, stop_at_eog=False)
    )
    it = eng.drain()
    next(it)
    t1 = time.monotonic()
    n = 1
    for _ in it:
        n += 1
    return rid, t1 - t0, n


def test_repeat_prompt_skips_prefill_with_identical_tokens():
    cfg = EngineConfig(
        n_ctx=4096,
        n_seq_max=1,
        speculative=False,
        mlx_prefix_cache=True,
        mlx_prefix_cache_size=2,
        prefix_cache_min_tokens=16,
    )
    eng = MLXEngine(MODEL_PATH, cfg)
    rid1, ttft1, _ = _once(eng)
    toks1 = eng.tokens_of(rid1)
    assert eng.prefix_misses == 1 and eng.prefix_hits == 0

    rid2, ttft2, _ = _once(eng)
    toks2 = eng.tokens_of(rid2)
    assert eng.prefix_hits == 1 and eng.prefix_misses == 1
    assert toks2 == toks1
    assert ttft2 < 0.5 * ttft1
    eng.ctx.close()


def test_cache_disabled_by_default():
    cfg = EngineConfig(n_ctx=4096, n_seq_max=1, speculative=False)
    eng = MLXEngine(MODEL_PATH, cfg)
    _once(eng)
    _once(eng)
    assert eng.prefix_hits == 0 and eng.prefix_misses == 0
    assert len(eng._prefix) == 0
    eng.ctx.close()


def test_short_prompt_below_min_tokens_not_cached():
    cfg = EngineConfig(
        n_ctx=4096,
        n_seq_max=1,
        speculative=False,
        mlx_prefix_cache=True,
        prefix_cache_min_tokens=10000,
    )
    eng = MLXEngine(MODEL_PATH, cfg)
    _once(eng)
    _once(eng)
    assert len(eng._prefix) == 0
    eng.ctx.close()
