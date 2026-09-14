"""Continuous batching on the MLX backend (engine/mlx_batch.py).

Two things are worth pinning. Membership bookkeeping is pure logic and is
tested with stub caches. Output parity needs real weights: a batch whose
output depends on who else is in flight would be worse than no batching, and
that is exactly the failure the cache-convention bugs produced while this was
being written.
"""

from __future__ import annotations

import pathlib

import pytest

from bwr.engine.config import EngineConfig

MODEL = pathlib.Path("models/Qwen3.8-27B-Uncensored-MLX-4bit/4-bit")
needs_weights = pytest.mark.skipif(
    not (MODEL / "config.json").is_file(), reason=f"{MODEL} not present"
)


# -- membership bookkeeping (no weights) -------------------------------------


class _StubCache:
    """Enough surface for join/leave: merge returns a marker, filter records."""

    def __init__(self, tag):
        self.tag = tag
        self.filtered = None

    @classmethod
    def merge(cls, caches):
        m = cls("+".join(str(c.tag) for c in caches))
        return m

    def filter(self, idx):
        self.filtered = idx


def _batch(monkeypatch):
    from bwr.engine import mlx_batch

    # join() on a non-empty batch goes through _concat_batched; for
    # bookkeeping tests replace it with something that just records.
    monkeypatch.setattr(
        mlx_batch, "_concat_batched", lambda a, b: _StubCache(f"{a.tag}|{b.tag}")
    )
    return mlx_batch.MLXBatch(model=object())


def test_new_batch_is_empty(monkeypatch):
    b = _batch(monkeypatch)
    assert b.empty and len(b) == 0
    assert b.index_of(1) is None


def test_join_assigns_rows_in_order(monkeypatch):
    b = _batch(monkeypatch)
    b.join(7, [_StubCache("a")])
    b.join(9, [_StubCache("b")])
    assert b.rows == [7, 9]
    assert b.index_of(7) == 0 and b.index_of(9) == 1
    assert len(b) == 2


def test_first_member_is_still_merged(monkeypatch):
    """A batch of one must have the same shape as a batch of many, or the
    single-member path diverges structurally and hides bugs."""
    b = _batch(monkeypatch)
    b.join(1, [_StubCache("solo")])
    assert b.caches[0].tag == "solo"      # went through merge(), not raw


def test_double_join_is_rejected(monkeypatch):
    b = _batch(monkeypatch)
    b.join(1, [_StubCache("a")])
    with pytest.raises(ValueError, match="already batched"):
        b.join(1, [_StubCache("b")])


def test_leave_drops_the_row_and_filters(monkeypatch):
    b = _batch(monkeypatch)
    b.join(1, [_StubCache("a")])
    b.join(2, [_StubCache("b")])
    b.leave([1])
    assert b.rows == [2]
    assert b.caches[0].filtered is not None, "surviving rows must be filtered"


def test_leave_of_last_member_empties_the_batch(monkeypatch):
    b = _batch(monkeypatch)
    b.join(1, [_StubCache("a")])
    b.leave([1])
    assert b.empty and b.caches is None


def test_leave_of_unknown_request_is_a_noop(monkeypatch):
    b = _batch(monkeypatch)
    b.join(1, [_StubCache("a")])
    b.leave([99])
    assert b.rows == [1]


def test_step_rejects_a_token_count_mismatch(monkeypatch):
    b = _batch(monkeypatch)
    b.join(1, [_StubCache("a")])
    b.join(2, [_StubCache("b")])
    with pytest.raises(ValueError, match="one token per row"):
        b.step([5])          # two rows, one token


def test_step_on_empty_batch_is_an_error(monkeypatch):
    b = _batch(monkeypatch)
    with pytest.raises(RuntimeError, match="empty batch"):
        b.step([])


# -- config guard (no weights) -----------------------------------------------


@pytest.mark.parametrize(
    "kw", [dict(speculative=True), dict(mlx_mtp=True)],
)
def test_batch_refuses_to_combine_with_per_request_features(kw, tmp_path):
    """Those features own a per-request cache; batching owns one shared
    batched cache. Silently picking a winner would be worse than refusing."""
    from bwr.engine.mlx_engine import MLXEngine

    with pytest.raises(ValueError, match="mutually exclusive"):
        MLXEngine(str(tmp_path), EngineConfig(engine="mlx", mlx_batch=True, **kw))


def test_batch_does_not_refuse_the_prefix_cache(tmp_path):
    """These two DO compose: a prefix hit yields a single-sequence cache
    covering the prompt, which is exactly what join() consumes. The failure
    here would be an exclusion error; a missing-weights error means the
    config was accepted and the load got further."""
    from bwr.engine.mlx_engine import MLXEngine

    with pytest.raises(Exception) as exc:
        MLXEngine(str(tmp_path), EngineConfig(
            engine="mlx", mlx_batch=True, mlx_prefix_cache=True))
    assert "mutually exclusive" not in str(exc.value)


# -- output parity (weights required) ----------------------------------------


@needs_weights
def test_incremental_join_matches_mlx_lm_merge():
    """A mid-flight join must produce exactly the cache mlx-lm's all-at-once
    merge would. Getting the offset/left_padding convention wrong here
    corrupts only the SHORTER rows, which is how it hides."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import ArraysCache

    from bwr.engine.mlx_batch import _concat_batched

    model, tok = load(str(MODEL))
    tgt = model.language_model
    prompts = ["Explain in one sentence what a B-tree is.\n", "Short.\n"]

    per_seq = []
    for p in prompts:
        c = tgt.make_cache()
        mx.eval(tgt(mx.array(tok.encode(p))[None], cache=c))
        per_seq.append(c)

    mismatches = []
    for layer in range(len(per_seq[0])):
        ref = type(per_seq[0][layer]).merge([c[layer] for c in per_seq])
        inc = type(per_seq[0][layer]).merge([per_seq[0][layer]])
        inc = _concat_batched(inc, type(per_seq[1][layer]).merge([per_seq[1][layer]]))
        if isinstance(ref, ArraysCache):
            same = all(
                (x is None and y is None)
                or (x.shape == y.shape
                    and float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)))) == 0)
                for x, y in zip(ref.cache, inc.cache)
            )
        else:
            rk, _, ro, rp = ref.state
            ik, _, io, ip = inc.state
            same = (
                rk.shape == ik.shape
                and ro.tolist() == io.tolist()
                and rp.tolist() == ip.tolist()
                and float(mx.max(mx.abs(rk.astype(mx.float32) - ik.astype(mx.float32)))) == 0
            )
        if not same:
            mismatches.append(layer)
    assert not mismatches, f"join diverged from merge at layers {mismatches[:5]}"


@needs_weights
def test_batched_decode_is_output_identical():
    """Three concurrent requests must each produce exactly what they produce
    alone. The reference is a DIRECT greedy loop, not the engine's non-batch
    path -- that one uses mlx-lm's stream generator, whose numeric drift
    looks like a batching bug and already caused one false alarm."""
    import mlx.core as mx
    from mlx_lm import load

    from bwr.engine.config import RequestParams
    from bwr.engine.mlx_engine import MLXEngine

    prompts = ["Explain in one sentence what a B-tree is.\n",
               "Name three sorting algorithms.\n",
               "Write a haiku about caching.\n"]
    n = 12

    model, tok = load(str(MODEL))
    tgt = model.language_model

    def solo(prompt):
        cache = tgt.make_cache()
        logits = tgt(mx.array(tok.encode(prompt))[None], cache=cache)
        t = int(mx.argmax(logits[0, -1]).item())
        out = [t]
        for _ in range(n - 1):
            logits = tgt(mx.array([[t]]), cache=cache)
            t = int(mx.argmax(logits[0, -1]).item())
            out.append(t)
        return out

    ref = [solo(p) for p in prompts]

    eng = MLXEngine(str(MODEL), EngineConfig(engine="mlx", n_ctx=4096, mlx_batch=True))
    try:
        rids = [
            eng.add_request(p, RequestParams(max_tokens=n, temp=0.0, stop_at_eog=False))
            for p in prompts
        ]
        emitted = []
        for out in eng.drain():
            if out.token >= 0:
                emitted.append(out.request_id)
        for i, rid in enumerate(rids):
            assert list(eng.tokens_of(rid))[:n] == ref[i], f"seq{i} diverged"
        # every live row emits once per step, so the first rows interleave
        assert len(set(emitted[:6])) > 1, "requests did not interleave"
    finally:
        eng.ctx.close()


@needs_weights
def test_prefix_cache_composes_with_batching():
    """A repeated prompt must still skip prefill while batching is on, and
    a cache-hit request must batch correctly alongside a fresh one."""
    import time

    import mlx.core as mx
    from mlx_lm import load

    from bwr.engine.config import RequestParams
    from bwr.engine.mlx_engine import MLXEngine

    long_prompt = ("Review this module and continue it.\n\n" + "".join(
        f"def helper_{i}(x):\n    return x * {i} + 1\n\n" for i in range(120)))
    other = "Name three sorting algorithms.\n"
    n = 6

    model, tok = load(str(MODEL))
    tgt = model.language_model

    def solo(prompt):
        cache = tgt.make_cache()
        logits = tgt(mx.array(tok.encode(prompt))[None], cache=cache)
        t = int(mx.argmax(logits[0, -1]).item())
        out = [t]
        for _ in range(n - 1):
            logits = tgt(mx.array([[t]]), cache=cache)
            t = int(mx.argmax(logits[0, -1]).item())
            out.append(t)
        return out

    ref_long, ref_other = solo(long_prompt), solo(other)

    eng = MLXEngine(str(MODEL), EngineConfig(
        engine="mlx", n_ctx=8192, mlx_batch=True, mlx_prefix_cache=True))
    try:
        def run(prompt):
            rid = eng.add_request(
                prompt, RequestParams(max_tokens=n, temp=0.0, stop_at_eog=False))
            t0 = time.perf_counter()
            first = None
            for out in eng.drain():
                if out.request_id == rid and out.token >= 0 and first is None:
                    first = time.perf_counter()
            return (first - t0) * 1000, list(eng.tokens_of(rid))

        cold_ms, cold = run(long_prompt)
        warm_ms, warm = run(long_prompt)
        assert cold[:n] == ref_long[:n] and warm[:n] == ref_long[:n]
        assert eng.prefix_hits >= 1, "repeat did not hit the prefix cache"
        assert warm_ms < cold_ms / 10, (
            f"repeat admission {warm_ms:.0f}ms vs cold {cold_ms:.0f}ms -- "
            "prefill was not skipped"
        )

        # a cache-hit request batched with a fresh one: both must be correct
        r1 = eng.add_request(
            long_prompt, RequestParams(max_tokens=n, temp=0.0, stop_at_eog=False))
        r2 = eng.add_request(
            other, RequestParams(max_tokens=n, temp=0.0, stop_at_eog=False))
        for _ in eng.drain():
            pass
        assert list(eng.tokens_of(r1))[:n] == ref_long[:n]
        assert list(eng.tokens_of(r2))[:n] == ref_other[:n]
    finally:
        eng.ctx.close()


@needs_weights
def test_batching_degrades_quantized_kv_to_f16_with_a_warning(caplog):
    """mlx-lm has no batched QuantizedKVCache, but quantized KV buys headroom
    rather than speed -- so batching drops to f16 and says so, instead of
    refusing a config the user reasonably expected to work."""
    import logging

    from bwr.engine.mlx_engine import MLXEngine
    from mlx_lm.models.cache import QuantizedKVCache

    with caplog.at_level(logging.WARNING, logger="bwr.engine.mlx_engine"):
        eng = MLXEngine(str(MODEL), EngineConfig(
            engine="mlx", n_ctx=4096, mlx_batch=True, mlx_kv_bits=8))
    try:
        assert any("mlx_kv_bits" in r.message for r in caplog.records), \
            "the downgrade must be announced, not silent"
        # and it must actually be f16: no quantized entries in a fresh cache
        cache = eng._make_cache()
        assert not any(isinstance(c, QuantizedKVCache) for c in cache)
        # the caller's config is untouched -- it may be shared across engines
        assert eng.config.mlx_kv_bits == 8
    finally:
        eng.ctx.close()


@needs_weights
def test_quantized_kv_still_works_without_batching():
    """The downgrade is scoped to batching; on its own the flag still bites."""
    from bwr.engine.mlx_engine import MLXEngine
    from mlx_lm.models.cache import QuantizedKVCache

    eng = MLXEngine(str(MODEL), EngineConfig(engine="mlx", n_ctx=4096, mlx_kv_bits=8))
    try:
        assert any(isinstance(c, QuantizedKVCache) for c in eng._make_cache())
    finally:
        eng.ctx.close()


@needs_weights
@pytest.mark.parametrize(
    "kw", [{}, dict(mlx_prefix_cache=True), dict(mlx_batch=True)],
)
def test_eog_marker_never_reaches_the_response_text(kw):
    """The end-of-turn token is a control token, not text.

    Regression: the manual-loop paths (prefix cache, batching) retired on EOG
    while passing the decoded piece, so a literal "<|im_end|>" landed in the
    response body. The stream path never had it -- mlx-lm breaks before
    yielding EOS -- which is exactly why it went unnoticed. MetalEngine
    retires with an empty piece; MLX now matches.
    """
    from bwr.engine.config import RequestParams
    from bwr.engine.mlx_engine import MLXEngine

    eng = MLXEngine(str(MODEL), EngineConfig(engine="mlx", n_ctx=4096, **kw))
    try:
        rid = eng.add_request(
            "Reply with exactly: recipe ok\n",
            RequestParams(max_tokens=16, temp=0.0),
        )
        text = "".join(
            o.piece for o in eng.drain() if o.request_id == rid and o.token >= 0
        )
        assert eng.state(rid).finish_reason == "eog", "test needs an EOG stop"
        for marker in ("<|im_end|>", "<|endoftext|>"):
            assert marker not in text, f"{marker} leaked into output: {text[-40:]!r}"
    finally:
        eng.ctx.close()
