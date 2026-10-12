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


def test_a_sampled_hit_does_not_replay_the_greedy_first_token() -> None:
    """A prefix entry stores the argmax AND the last logits. A greedy repeat
    replays the argmax; a sampled repeat draws from the logits. Before this,
    every temp>0 repeat of a cached prompt opened with the greedy token."""
    import mlx.core as mx
    from bwr.engine.mlx_engine import MLXEngine

    prompt = "Count: " + " ".join(str(i % 10) for i in range(300))
    eng = MLXEngine(MODEL_PATH, EngineConfig(
        n_ctx=4096, mlx_prefix_cache=True, prefix_cache_min_tokens=16,
    ))
    try:
        greedy = RequestParams(temp=0.0, max_tokens=4, stop_at_eog=False)
        rid = eng.add_request(prompt, greedy)
        list(eng.drain())
        first_greedy = eng.tokens_of(rid)[0]
        assert eng.prefix_misses == 1
        # A hot entry carries the logits row the sampled path draws from.
        entry = next(iter(eng._prefix.values()))
        assert isinstance(entry[4], mx.array)
        # Greedy repeat: hit, same first token.
        rid = eng.add_request(prompt, greedy)
        list(eng.drain())
        assert eng.prefix_hits == 1 and eng.tokens_of(rid)[0] == first_greedy
        # Sampled repeats: hit, drawn from the logits -- they must at least
        # run, and with a hot temperature not all land on the argmax.
        hot = RequestParams(temp=2.0, top_k=0, top_p=1.0, max_tokens=4, stop_at_eog=False)
        firsts = set()
        for _ in range(12):
            rid = eng.add_request(prompt, hot)
            list(eng.drain())
            firsts.add(eng.tokens_of(rid)[0])
        assert eng.prefix_hits == 13
        assert len(firsts) > 1, firsts
    finally:
        eng.ctx.close()


def _cold_tokens(prompt: str, n: int) -> list[int]:
    from bwr.engine.mlx_engine import MLXEngine

    eng = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    try:
        rid = eng.add_request(prompt, RequestParams(temp=0.0, max_tokens=n, stop_at_eog=False))
        list(eng.drain())
        return list(eng.tokens_of(rid))
    finally:
        eng.ctx.close()


def test_a_prompt_that_extends_a_stored_one_prefills_only_the_new_part() -> None:
    """Partial hit: the stored cache covers a prefix of the new prompt, so
    only the remainder is prefilled. Output must match a cold prefill."""
    from bwr.engine.mlx_engine import MLXEngine

    base = "Count: " + " ".join(str(i % 10) for i in range(300))
    longer = base + " Now continue the count:"
    eng = MLXEngine(MODEL_PATH, EngineConfig(
        n_ctx=4096, mlx_prefix_cache=True, prefix_cache_min_tokens=16,
    ))
    try:
        greedy = RequestParams(temp=0.0, max_tokens=6, stop_at_eog=False)
        rid = eng.add_request(base, greedy)
        list(eng.drain())
        assert eng.prefix_misses == 1 and eng.prefix_partial_hits == 0
        rid = eng.add_request(longer, greedy)
        list(eng.drain())
        assert eng.prefix_partial_hits == 1
        # The finished first request stored prompt + its answer, which is
        # the longest stored prefix of `longer` only if the answer happens
        # to match -- it does not here, so the match is the base prompt.
        assert eng.prefix_hit_tokens >= len(eng.ctx.tokenize(base)) if hasattr(eng.ctx, "tokenize") else True
        assert list(eng.tokens_of(rid)) == _cold_tokens(longer, 6)
    finally:
        eng.ctx.close()


def test_the_next_chat_turn_extends_the_finished_conversation() -> None:
    """A finished request stores its cache under prompt + answer. A prompt
    that continues that exact sequence (the next turn) is a partial hit
    covering the whole previous conversation."""
    from bwr.engine.mlx_engine import MLXEngine

    base = "Count: " + " ".join(str(i % 10) for i in range(300))
    eng = MLXEngine(MODEL_PATH, EngineConfig(
        n_ctx=4096, mlx_prefix_cache=True, prefix_cache_min_tokens=16,
    ))
    try:
        rid = eng.add_request(base, RequestParams(temp=0.0, max_tokens=8, stop_at_eog=False))
        list(eng.drain())
        answer = list(eng.tokens_of(rid))
        prompt_ids = list(eng._tokenizer.encode(base))
        conversation = prompt_ids + answer
        assert tuple(conversation) in eng._prefix, "retire did not store the conversation"
        extra = list(eng._tokenizer.encode(" and then some more"))
        rid = eng.add_request(conversation + extra, RequestParams(temp=0.0, max_tokens=4, stop_at_eog=False))
        list(eng.drain())
        assert eng.prefix_partial_hits == 1
        assert eng.prefix_hit_tokens == len(conversation)
        assert list(eng.tokens_of(rid)) == _cold_tokens(conversation + extra, 4)
    finally:
        eng.ctx.close()


def test_a_speculating_request_stores_its_cache_under_the_tokens_it_holds() -> None:
    """With n-gram drafts the cache advances by whole draft rows, not one
    lookahead token at a time; the end-of-request entry must still be
    keyed by prompt + answer. Keyed short, an extension of the
    conversation partial-hits a cache holding more than the key says and
    decodes against positions the prompt never had."""
    from bwr.engine.mlx_engine import MLXEngine

    base = "Count: " + " ".join(str(i % 10) for i in range(300))
    eng = MLXEngine(MODEL_PATH, EngineConfig(
        n_ctx=4096, speculative=True, mlx_prefix_cache=True,
        prefix_cache_min_tokens=16,
    ))
    try:
        rid = eng.add_request(base, RequestParams(temp=0.0, max_tokens=16, stop_at_eog=False))
        list(eng.drain())
        assert eng.spec_accepted > 0, "no draft accepted; the cache never jumped"
        conversation = list(eng._tokenizer.encode(base)) + list(eng.tokens_of(rid))
        assert tuple(conversation) in eng._prefix, "retire stored a key that is not prompt + answer"
        extra = list(eng._tokenizer.encode(" and then some more"))
        rid = eng.add_request(conversation + extra, RequestParams(temp=0.0, max_tokens=4, stop_at_eog=False))
        list(eng.drain())
        assert eng.prefix_partial_hits == 1
        assert list(eng.tokens_of(rid)) == _cold_tokens(conversation + extra, 4)
    finally:
        eng.ctx.close()


def test_a_one_token_repeat_of_a_cached_prompt_finishes_on_the_hit() -> None:
    """The hit's replayed base token is the whole answer when max_tokens is
    1: `_feed` retires the request inside `_prefix_hit`, and the lookahead
    that follows must notice (next_token is None by then) rather than queue
    a forward for a finished request."""
    from bwr.engine.mlx_engine import MLXEngine

    eng = MLXEngine(MODEL_PATH, EngineConfig(
        n_ctx=4096, mlx_prefix_cache=True, prefix_cache_min_tokens=16,
    ))
    try:
        rid1, _, _ = _once(eng, max_tokens=8)
        first = eng.tokens_of(rid1)[0]
        rid2, _, n = _once(eng, max_tokens=1)
        assert eng.prefix_hits == 1 and n == 1
        assert eng.state(rid2).finish_reason == "length"
        assert eng.tokens_of(rid2) == [first]
    finally:
        eng.ctx.close()
