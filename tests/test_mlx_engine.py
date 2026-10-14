"""MLX backend (SPEC-mlx-engine.md, M10b/M10c).

Skips unless the mlx extra is installed AND BWR_MLX_MODEL names an MLX
weights directory. Text-equality with the Metal backend is NOT asserted:
different precisions may legally flip greedy argmaxes (same standard as the
KV-quant battery). What is asserted: interface parity (admit/step/drain/
cancel/accessors), finish-reason validity, stop-sequence truncation, and
AsyncEngine compatibility (including context close).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest
from bwr.engine import EngineConfig, RequestParams
from bwr.engine.mlx_engine import MLXEngine, _MLXContext

MODEL_PATH = os.environ.get("BWR_MLX_MODEL")

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None,
    reason="pip install 'big-white-rabbit[mlx]' to run MLX tests",
)
needs_weights = pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None
    or not MODEL_PATH
    or not os.path.isdir(MODEL_PATH or ""),
    reason="need mlx installed and BWR_MLX_MODEL to run MLX backend tests",
)

N_TOKENS = 8


def greedy(max_tokens: int = N_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=True)


def test_bad_engine_name_rejected_without_weights() -> None:
    from bwr.server.app import build_app

    with pytest.raises(ValueError, match="engine"):
        build_app(None, EngineConfig(engine="bogus"))  # type: ignore[arg-type]


def test_mlx_engine_requires_path() -> None:
    from bwr.server.app import build_app

    with pytest.raises(ValueError, match="mlx_model_path"):
        build_app(None, EngineConfig(engine="mlx"))  # type: ignore[arg-type]


@needs_weights
def test_generate_to_cap() -> None:
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    rid = engine.add_request("Hello, what model are you?", greedy())
    outs = list(engine.drain())
    assert len(engine.tokens_of(rid)) == N_TOKENS
    assert engine.state(rid).finish_reason == "length"
    assert all(o.request_id == rid for o in outs)
    assert not engine.has_work


@needs_weights
def test_cancel() -> None:
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    rid = engine.add_request("Count: 1 2 3", greedy(64))
    engine.step()
    assert engine.cancel(rid) is True
    assert engine.cancel(rid) is False
    assert not engine.has_work


@needs_weights
def test_stop_sequence_truncates() -> None:
    """Two-phase like the Metal stop proof: observe text, then stop inside it.
    Temp-0 determinism reproduces the span, so the delimiter must hit; asserted
    on retirement reason and early shortfall (same contract as MetalEngine --
    no assertion about response text, which can't tell "stopped" from "cut").
    """
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    probe = engine.add_request(
        "Explain what mixture-of-experts means briefly.",
        RequestParams(temp=0.0, max_tokens=32, stop_at_eog=True),
    )
    list(engine.drain())
    full = engine.text_of(probe)
    assert len(full) > 20
    stop = full[10:20]
    rid = engine.add_request(
        "Explain what mixture-of-experts means briefly.",
        RequestParams(temp=0.0, max_tokens=32, stop_at_eog=True, stop=(stop,)),
    )
    list(engine.drain())
    state = engine.state(rid)
    assert state.finish_reason == "stop_sequence"
    assert state.n_generated < 32


@needs_weights
def test_admission_validation() -> None:
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    with pytest.raises(ValueError, match="max_tokens"):
        engine.add_request("hi", RequestParams(temp=0.0, max_tokens=0))
    with pytest.raises(ValueError, match="zero tokens"):
        engine.add_request("", greedy())
    with pytest.raises(ValueError, match="capacity"):
        tiny = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=16))
        tiny.add_request("word " * 100, greedy())
    with pytest.raises(KeyError):
        engine.tokens_of(999)


@needs_weights
def test_async_engine_compat() -> None:
    """AsyncEngine drives MLXEngine end to end, including context close."""

    async def scenario() -> tuple[list[int], bool]:
        from bwr.engine.async_engine import AsyncEngine

        engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
        async_engine = AsyncEngine(engine)
        await async_engine.start()
        try:
            rid = await async_engine.submit("Count: 1 2 3", greedy())
            seen: list[int] = []
            async for out in async_engine.stream(rid):
                seen.append(out.token)
                if out.finished:
                    assert out.finish_reason == "length"
            async_engine.release(rid)
            return seen, engine.ctx.closed
        finally:
            await async_engine.stop()

    seen, closed_before_stop = asyncio.run(scenario())
    assert len(seen) == N_TOKENS
    assert closed_before_stop is False


@needs_weights
def test_chat_completion_end_to_end() -> None:
    """The reported default-serve 500: chat rendering needs the tokenizer,
    not the llama Model. Proves a completion serves over HTTP."""
    from bwr.server.app import build_app
    from fastapi.testclient import TestClient

    app = build_app(
        None, EngineConfig(n_ctx=4096, engine="mlx"),
        served_model_name="t", mlx_model_path=MODEL_PATH,
    )
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "t",
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["message"]["content"].strip() != ""
        assert body["usage"]["completion_tokens"] >= 1


@needs_weights
def test_health_geometry() -> None:
    """The exact /health failure: geometry + counters must exist so serve
    boots (default engine) instead of 500ing."""
    from bwr.server.app import build_app
    from fastapi.testclient import TestClient

    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    assert engine.ctx.n_ctx == 4096
    assert engine.ctx.n_ctx_seq == 4096
    assert engine.ctx.n_seq_max == 8
    assert engine.ctx.decode_calls == 0
    assert engine.n_free_seq_slots == 8
    engine.add_request("hi", greedy())
    list(engine.drain())
    assert engine.ctx.decode_calls == N_TOKENS

    app = build_app(
        None, EngineConfig(n_ctx=4096, engine="mlx"),
        served_model_name="t", mlx_model_path=MODEL_PATH,
    )
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["n_ctx"] == 4096
        assert body["free_seq_slots"] == 8


REPETITIVE = "Count: " + " ".join(str(i % 10) for i in range(48))
REP_TOKENS = 16


def _rep_params(max_tokens: int = REP_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False)


@pytest.fixture(scope="module")
def plain_engine() -> MLXEngine:
    return MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))


@pytest.fixture(scope="module")
def spec_engine() -> MLXEngine:
    return MLXEngine(
        MODEL_PATH, EngineConfig(n_ctx=4096, speculative=True)
    )


def _run(engine: MLXEngine, prompt: str = REPETITIVE) -> int:
    rid = engine.add_request(prompt, _rep_params())
    list(engine.drain())
    return rid


@needs_weights
def test_spec_on_off_byte_identical(
    plain_engine: MLXEngine, spec_engine: MLXEngine
) -> None:
    """The invariant, MLX-flavored: speculation must be byte-identical."""
    plain_id = _run(plain_engine)
    spec_id = _run(spec_engine)
    assert spec_engine.tokens_of(spec_id) == plain_engine.tokens_of(plain_id)
    assert (
        spec_engine.state(spec_id).finish_reason
        == plain_engine.state(plain_id).finish_reason
    )


@needs_weights
def test_speculative_engages(spec_engine: MLXEngine) -> None:
    """Drafts packed and accepted on repetitive text; rate in bounds."""
    before_drafted = spec_engine.spec_drafted
    rid = _run(spec_engine)
    assert spec_engine.spec_drafted > before_drafted
    assert spec_engine.tokens_of(rid) is not None
    rate = spec_engine.spec_acceptance_rate
    assert rate is not None and 0.0 <= rate <= 1.0


@needs_weights
def test_spec_fallback_all_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every draft wrong: output still identical, recomputes ran, and the
    request fell back to plain single-token evals."""
    from bwr.engine import mlx_engine as engine_module

    probe = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    probe_id = _run(probe)
    expected = probe.tokens_of(probe_id)
    prompt_ids = probe.state(probe_id).prompt
    vocab = probe._tokenizer.vocab_size
    hist = prompt_ids + expected
    wrong = {
        tuple(hist[max(0, i - 2) : i]): (tok + 1) % vocab
        for i, tok in enumerate(hist)
    }

    real_table = engine_module.NgramTable

    class WrongTable(real_table):  # type: ignore[valid-type, misc]
        def predict(self, context, max_tokens: int) -> list[int]:  # type: ignore[override]
            token = wrong.get(tuple(context[-(self.order - 1) :]))
            return [token] * max_tokens if token is not None else []

    monkeypatch.setattr(engine_module, "NgramTable", WrongTable)
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096, speculative=True))
    rid = _run(engine)
    assert engine.tokens_of(rid) == expected
    assert engine.spec_recomputes > 0, "no recompute ran on all-mismatch"
    assert engine.spec_fallbacks == 1, "fallback never engaged"


@needs_weights
def test_speculation_defaults_off(plain_engine: MLXEngine) -> None:
    assert plain_engine._spec_tables == {}
    assert plain_engine.spec_drafted == 0
    assert plain_engine.spec_acceptance_rate is None


@needs_weights
def test_nongreedy_ignores_spec() -> None:
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096, speculative=True))
    rid = engine.add_request(
        REPETITIVE, RequestParams(temp=1.0, max_tokens=8, stop_at_eog=False)
    )
    list(engine.drain())
    assert engine.spec_drafted == 0
    assert len(engine.tokens_of(rid)) == 8
    assert engine.state(rid).finish_reason == "length"


@needs_weights
def test_eog_reason_and_no_eos_leak() -> None:
    """A natural end must retire eog (not length), with no end-token piece
    text leaking into output. Regression: the EOS set used to miss
    <|endoftext|>, leaking its piece and mislabeling the finish."""
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    rid = engine.add_request(
        "Say hi in one sentence.",
        RequestParams(temp=0.0, max_tokens=64, stop_at_eog=True),
    )
    list(engine.drain())
    assert engine.state(rid).finish_reason == "eog"
    text = engine.text_of(rid)
    assert "<|endoftext|>" not in text
    assert "<|im_end|>" not in text


@needs_weights
def test_thinking_disabled_by_default() -> None:
    """Server-shaped prompt (template-rendered) must produce no think block:
    thinking is off unless a future API opts in."""
    engine = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=8192))
    prompt = engine.apply_chat_template([("user", "Say hi in one sentence.")])
    rid = engine.add_request(
        prompt, RequestParams(temp=0.0, max_tokens=32, stop_at_eog=True)
    )
    list(engine.drain())
    assert "<think>" not in engine.text_of(rid)


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None, reason="mlx_lm not installed"
)
def test_manual_loop_cache_for_a_model_without_make_cache() -> None:
    """Only hybrid models define make_cache; a plain-attention model (the
    30B-A3B MoE) relies on mlx-lm's per-layer default. The stream path
    always used the builder; the manual loop called the method directly
    and broke every such model the day greedy requests moved onto it."""
    from mlx_lm.models.cache import KVCache

    class _PlainAttentionModel:
        def __init__(self):
            self.layers = [object(), object(), object()]

    eng = MLXEngine.__new__(MLXEngine)
    eng._model = _PlainAttentionModel()
    eng.config = EngineConfig(engine="mlx")
    cache = eng._make_cache()
    assert len(cache) == 3 and all(type(c) is KVCache for c in cache)


# --- stub-level tests of the manual loop's bookkeeping ---------------------
#
# A fake model whose "next token" is always `token + 1`, driving the real
# verify loop through a cache that records exactly what was fed into it.
# No weights: what these pin is the engine's own accounting (which tokens
# the cache holds, what the prefix cache stores), not the model.

needs_mlx_lib = pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None, reason="mlx_lm not installed"
)

_VOCAB = 64


class _TrackingCache:
    """A KVCache stand-in: `offset` is the number of tokens held, and
    rewinding it (what `_restore_cache` does) drops the tokens past it."""

    def __init__(self) -> None:
        self.tokens: list[int] = []

    @property
    def offset(self) -> int:
        return len(self.tokens)

    @offset.setter
    def offset(self, n: int) -> None:
        del self.tokens[n:]


class _SuccessorModel:
    """argmax(logits for token t) == t + 1, at every position."""

    def __call__(self, ids, cache=None):
        import mlx.core as mx

        toks = [int(t) for t in ids[0].tolist()]
        cache[0].tokens.extend(toks)
        logits = mx.zeros((1, len(toks), _VOCAB))
        for i, t in enumerate(toks):
            logits[0, i, (t + 1) % _VOCAB] = 1.0
        return logits


class _Tokenizer:
    eos_token_ids = None
    eos_token_id = None

    def decode(self, ids) -> str:
        return " ".join(str(int(t)) for t in ids)


class _FixedTable:
    """An n-gram table stand-in proposing a scripted draft sequence."""

    order = 3

    def __init__(self, drafts: list[int]) -> None:
        self.drafts = drafts

    def predict(self, context, max_tokens: int) -> list[int]:
        return list(self.drafts[:max_tokens])

    def update_stream(self, tokens) -> None:
        pass


def _stub_engine(config: EngineConfig) -> MLXEngine:
    """An MLXEngine without weights: every per-request table the manual
    loop and the prefix cache touch, set by hand as __init__ would."""
    from collections import OrderedDict

    from bwr.engine.spec_gate import SpecThroughputGate

    eng = MLXEngine.__new__(MLXEngine)
    eng.config = config
    eng._kv_bits = config.mlx_kv_bits
    eng._model = _SuccessorModel()
    eng._tokenizer = _Tokenizer()
    eng._make_cache = lambda: [_TrackingCache()]
    eng._mtp = None
    eng._mtp_caches = {}
    eng._mtp_hidden = {}
    eng._batch = None
    eng._stop_filters = {}
    eng._detoks = {}
    eng._states = {}
    eng._retired = OrderedDict()
    eng.ctx = _MLXContext(eng)
    eng._decode_calls = 0
    eng._spec_tables = {}
    eng._caches = {}
    eng._spec_ok = {}
    eng._spec_stats = {}
    eng._pending = {}
    eng._batch_pending = {}
    eng.spec_drafted = eng.spec_accepted = eng.spec_recomputes = eng.spec_fallbacks = 0
    eng.spec_gate = SpecThroughputGate()
    eng._prefix = OrderedDict()
    eng.prefix_hits = eng.prefix_partial_hits = eng.prefix_hit_tokens = eng.prefix_misses = 0
    eng._cache_tokens = {}
    eng._next_request_id = 0
    return eng


def _admit(eng: MLXEngine, prompt: list[int], max_tokens: int, drafts: list[int] | None):
    from bwr.engine.batching import RequestState

    rid = eng._next_request_id
    eng._next_request_id += 1
    req = RequestState(
        request_id=rid, seq_id=0, prompt=list(prompt),
        params=RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False),
    )
    eng._states[rid] = req
    eng._spec_ok[rid] = True
    eng._spec_stats[rid] = [0, 0, 0]
    if drafts is not None:
        eng._spec_tables[rid] = _FixedTable(drafts)
    return req


@needs_mlx_lib
def test_the_retention_window_covers_at_least_a_full_batch() -> None:
    """tests/test_states_pruning.py's pin, for MLX: every request of a batch
    is read after IT ends while its peers are still running, so the window
    of finished requests kept readable must be at least n_seq_max wide.
    `max(8, MIN_RETAINED_FINISHED)` was a tautology (both are 8), so a
    16-wide batch retiring in one step lost its first eight."""
    n_seq_max = 16
    eng = _stub_engine(EngineConfig(n_ctx=4096, n_seq_max=n_seq_max))
    assert eng.retained_finished_limit >= eng.ctx.n_seq_max
    reqs = [_feed_only(eng, [1, 2, 3], max_tokens=1) for _ in range(n_seq_max)]
    for req in reqs:
        assert eng._feed(req, 4, "4").finished
    assert eng.n_retained_finished == n_seq_max
    for req in reqs:
        assert eng.state(req.request_id).finish_reason == "length"
        assert eng.tokens_of(req.request_id) == [4]


def _spec_cfg() -> EngineConfig:
    return EngineConfig(
        engine="mlx", n_ctx=4096, speculative=True, spec_max_drafts=3,
        mlx_prefix_cache=True, prefix_cache_min_tokens=1,
    )


@needs_mlx_lib
@pytest.mark.parametrize(
    "drafts, expect_recompute",
    [([5, 6, 7], False), ([5, 9, 7], True)],
    ids=["full-match", "mismatch"],
)
def test_spec_round_keeps_cache_tokens_in_step_with_the_cache(
    drafts: list[int], expect_recompute: bool
) -> None:
    """With drafts in play the cache advances by the draft rows (full
    match) or by the re-fed accepted prefix (mismatch), and `_cache_tokens`
    must follow: it is the key the end-of-request prefix entry is stored
    under. Recorded only by the lookahead, the key came out shorter than the
    cache, and an extension of it then partial-hit a cache holding more
    tokens than the key said -- wrong output, silently."""
    eng = _stub_engine(_spec_cfg())
    # Prompt 1 2 3 -> base 4; successor drafts 5 6 7 all match; 5 9 7 fails
    # at the second row, accepting 5 and the bonus 6.
    req = _admit(eng, [1, 2, 3], max_tokens=8, drafts=drafts)
    eng._start_spec(req)
    cache = eng._caches[req.request_id][0]
    assert cache.tokens == [1, 2, 3, 4]  # prompt + lookahead of the base
    eng._verify(req)
    assert (eng.spec_recomputes > 0) is expect_recompute
    held = list(cache.tokens)
    recorded = [int(t) if isinstance(t, int) else int(t.item()) for t in eng._cache_tokens[req.request_id]]
    assert recorded == held
    # The round after retire stores the cache under the tokens it holds.
    req.params = RequestParams(temp=0.0, max_tokens=req.n_generated, stop_at_eog=False)
    eng._retire(req, "length")
    (key, entry), = eng._prefix.items() if len(eng._prefix) == 1 else [max(eng._prefix.items(), key=lambda kv: len(kv[0]))]
    assert list(key) == held
    assert entry[0][0].tokens == held


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"mlx_kv_bits": 3}, "mlx_kv_bits must be 0, 4 or 8"),
        ({"spec_max_drafts": -1}, "spec_max_drafts must be >= 0"),
        ({"prefix_cache_min_tokens": 0}, "prefix_cache_min_tokens must be >= 1"),
    ],
    ids=["kv-bits", "spec-max-drafts", "prefix-min-tokens"],
)
def test_bad_knobs_are_refused_before_the_weights_load(
    monkeypatch: pytest.MonkeyPatch, bad: dict, message: str
) -> None:
    """Validation runs in the constructor ahead of load(): a bad value
    fails in milliseconds naming the knob, not after the weights are in
    (mlx_kv_bits used to surface at the first request's cache build)."""
    from bwr.engine import mlx_engine as engine_module

    loaded = []

    def fake_require():
        def load(path):
            loaded.append(path)
            raise AssertionError("load() ran before validation")

        return load, None

    monkeypatch.setattr(engine_module, "_require_mlx", fake_require)
    with pytest.raises(ValueError, match=message):
        MLXEngine("/nonexistent/weights", EngineConfig(engine="mlx", **bad))
    assert loaded == []


# --- _retire / _prefix_insert_final ----------------------------------------


def _finished_request(eng: MLXEngine, tokens: list[int], nxt: int):
    """A request whose cache holds `tokens` with a lookahead pending whose
    argmax is `nxt` -- the state `_retire` finds at the end of a request."""
    import mlx.core as mx

    req = _admit(eng, tokens[:2], max_tokens=4, drafts=None)
    rid = req.request_id
    cache = _TrackingCache()
    cache.tokens = list(tokens)
    eng._caches[rid] = [cache]
    eng._cache_tokens[rid] = [tokens[0], *(mx.array(t) for t in tokens[1:])]
    logits = mx.zeros((1, 1, _VOCAB))
    logits[0, 0, nxt] = 1.0
    eng._pending[rid] = (mx.argmax(logits[0, -1]), logits, None)
    return req


@needs_mlx_lib
def test_a_normal_retire_stores_the_cache_under_the_tokens_it_holds() -> None:
    import mlx.core as mx

    eng = _stub_engine(_spec_cfg())
    req = _finished_request(eng, [1, 2, 3, 4], nxt=5)
    cache = eng._caches[req.request_id][0]
    eng._retire(req, "length")
    assert list(eng._prefix) == [(1, 2, 3, 4)]
    stored, base, mcache, hidden, last = eng._prefix[(1, 2, 3, 4)]
    assert base == 5
    assert int(mx.argmax(last).item()) == 5
    assert stored[0].tokens == [1, 2, 3, 4] and stored[0] is not cache
    assert mcache is None and hidden is None
    # Everything per-request is released.
    assert req.request_id not in eng._caches
    assert req.request_id not in eng._cache_tokens
    assert req.request_id not in eng._pending
    assert req.finished and req.finish_reason == "length"


@needs_mlx_lib
def test_an_error_retire_stores_nothing() -> None:
    eng = _stub_engine(_spec_cfg())
    req = _finished_request(eng, [1, 2, 3, 4], nxt=5)
    eng._retire(req, "error")
    assert eng._prefix == {}
    assert req.request_id not in eng._caches


@needs_mlx_lib
def test_a_disabled_prefix_cache_stores_nothing_at_retire() -> None:
    cfg = _spec_cfg()
    cfg.mlx_prefix_cache = False
    eng = _stub_engine(cfg)
    req = _finished_request(eng, [1, 2, 3, 4], nxt=5)
    eng._retire(req, "length")
    assert eng._prefix == {}


@needs_mlx_lib
@pytest.mark.parametrize("missing", ["_caches", "_cache_tokens", "_pending"])
def test_retire_without_cache_tokens_or_pending_stores_nothing(missing: str) -> None:
    """A request that never reached the lookahead (admitted, not stepped;
    or failed mid-prefill) has nothing to store, and must not raise."""
    eng = _stub_engine(_spec_cfg())
    req = _finished_request(eng, [1, 2, 3, 4], nxt=5)
    getattr(eng, missing).pop(req.request_id)
    eng._retire(req, "cancelled")
    assert eng._prefix == {}


@needs_mlx_lib
def test_retire_below_min_tokens_stores_nothing() -> None:
    cfg = _spec_cfg()
    cfg.prefix_cache_min_tokens = 5
    eng = _stub_engine(cfg)
    req = _finished_request(eng, [1, 2, 3, 4], nxt=5)
    eng._retire(req, "length")
    assert eng._prefix == {}


# --- _prefix_lookup ----------------------------------------------------------


def _store(eng: MLXEngine, key: tuple[int, ...], base: int = 0) -> None:
    import mlx.core as mx

    cache = _TrackingCache()
    cache.tokens = list(key)
    eng._prefix[key] = ([cache], base, None, None, mx.zeros((_VOCAB,)))


def _probe(eng: MLXEngine, prompt: list[int]):
    return eng._prefix_lookup(_admit(eng, prompt, max_tokens=4, drafts=None))


@needs_mlx_lib
def test_an_exact_hit_returns_a_copy_of_the_stored_cache() -> None:
    eng = _stub_engine(_spec_cfg())
    _store(eng, (1, 2, 3), base=4)
    found = _probe(eng, [1, 2, 3])
    assert found is not None
    cache, base, entry, matched = found
    assert base == 4 and matched == 3
    assert entry is eng._prefix[(1, 2, 3)]
    assert cache[0].tokens == [1, 2, 3] and cache[0] is not entry[0][0]
    assert (eng.prefix_hits, eng.prefix_partial_hits, eng.prefix_hit_tokens) == (1, 0, 3)


@needs_mlx_lib
def test_a_strict_prefix_of_a_stored_entry_misses() -> None:
    """The stored cache holds MORE than the prompt; recurrent state cannot
    be cut back, so it is unusable, however tempting."""
    eng = _stub_engine(_spec_cfg())
    _store(eng, (1, 2, 3, 4, 5))
    assert _probe(eng, [1, 2, 3]) is None
    assert (eng.prefix_hits, eng.prefix_misses) == (0, 1)


@needs_mlx_lib
def test_the_longest_extendable_entry_wins_and_becomes_most_recent() -> None:
    eng = _stub_engine(_spec_cfg())
    _store(eng, (1, 2, 3, 4))
    _store(eng, (1, 2))
    _store(eng, (9, 9, 9))
    found = _probe(eng, [1, 2, 3, 4, 5, 6])
    assert found is not None
    cache, base, entry, matched = found
    assert base is None and matched == 4
    assert cache[0].tokens == [1, 2, 3, 4]
    assert entry is eng._prefix[(1, 2, 3, 4)]
    assert list(eng._prefix)[-1] == (1, 2, 3, 4)
    assert (eng.prefix_hits, eng.prefix_partial_hits, eng.prefix_hit_tokens) == (1, 1, 4)


@needs_mlx_lib
def test_a_diverging_prompt_misses() -> None:
    eng = _stub_engine(_spec_cfg())
    _store(eng, (1, 2, 3, 4))
    assert _probe(eng, [1, 2, 9, 4, 5]) is None
    assert (eng.prefix_hits, eng.prefix_misses) == (0, 1)


# --- the finishing token reaches the output ---------------------------------


def _feed_only(eng: MLXEngine, prompt: list[int], max_tokens: int, stop=()):
    """A request whose tokens will be fed by hand (no forward passes)."""
    from bwr.engine.config import StopSequenceFilter

    req = _admit(eng, prompt, max_tokens=max_tokens, drafts=None)
    if stop:
        eng._stop_filters[req.request_id] = StopSequenceFilter(list(stop))
    return req


@needs_mlx_lib
def test_the_capping_token_is_retired_with_its_token_and_piece() -> None:
    """MetalEngine retires on "length" with the token that hit the cap and
    its piece; the routes build the text from StepOutput.piece, so an
    empty-piece retire drops the final token from the response while the
    usage still counts it."""
    eng = _stub_engine(EngineConfig(n_ctx=4096))
    req = _feed_only(eng, [1, 2, 3], max_tokens=1)
    out = eng._feed(req, 4, "4")
    assert out.finished and out.finish_reason == "length"
    assert (out.token, out.piece) == (4, "4")
    assert eng.text_of(req.request_id) == "4"


@needs_mlx_lib
def test_the_stop_completing_token_is_retired_with_its_token_and_piece() -> None:
    """Same for "stop_sequence": the route-side StopSequenceFilter needs the
    completing piece to recognise the delimiter and drop its withheld prefix.
    Retired without it, the prefix leaks on flush."""
    eng = _stub_engine(EngineConfig(n_ctx=4096))
    req = _feed_only(eng, [1, 2, 3], max_tokens=8, stop=["4 5"])
    first = eng._feed(req, 4, "4 ")
    assert not first.finished
    out = eng._feed(req, 5, "5")
    assert out.finished and out.finish_reason == "stop_sequence"
    assert (out.token, out.piece) == (5, "5")


@needs_weights
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_joined_pieces_are_the_text_at_the_cap(batch: bool) -> None:
    eng = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096, mlx_batch=batch))
    try:
        rid = eng.add_request("Hello, what model are you?", greedy())
        outs = list(eng.drain())
        assert eng.state(rid).finish_reason == "length"
        assert len(eng.tokens_of(rid)) == N_TOKENS
        assert "".join(o.piece for o in outs) == eng.text_of(rid)
        assert outs[-1].token == eng.tokens_of(rid)[-1]
    finally:
        eng.ctx.close()


@needs_weights
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_joined_pieces_are_the_text_at_a_stop_sequence(batch: bool) -> None:
    prompt = "Count: " + " ".join(str(i % 10) for i in range(48))
    eng = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096, mlx_batch=batch))
    try:
        rid = eng.add_request(
            prompt, RequestParams(temp=0.0, max_tokens=32, stop_at_eog=False, stop=("3 4",))
        )
        outs = list(eng.drain())
        assert eng.state(rid).finish_reason == "stop_sequence", eng.text_of(rid)
        assert "".join(o.piece for o in outs) == eng.text_of(rid)
        assert outs[-1].token == eng.tokens_of(rid)[-1]
    finally:
        eng.ctx.close()


# --- the context cap applies on every path ----------------------------------


@needs_mlx_lib
def test_feed_retires_on_context_when_prompt_plus_output_fills_n_ctx() -> None:
    """The n_ctx check lived in the verify loop only; the batch step and the
    prefix hit account their tokens through _feed, which checked EOG, stop
    and max_tokens and nothing else, so a batched request ran past n_ctx
    and reported "length". The check belongs in the one accounting
    function, as in MetalEngine, after the stop and cap rules."""
    eng = _stub_engine(EngineConfig(n_ctx=5))
    req = _feed_only(eng, [1, 2, 3], max_tokens=100)
    assert not eng._feed(req, 4, "4").finished
    out = eng._feed(req, 5, "5")
    assert out.finished and out.finish_reason == "context"
    assert (out.token, out.piece) == (5, "5")
    assert eng.tokens_of(req.request_id) == [4, 5]
    assert eng.text_of(req.request_id) == "4 5"


@needs_weights
def test_batch_path_retires_on_context_where_the_single_path_does() -> None:
    n_ctx, n_prompt = 64, 50
    probe = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=n_ctx))
    words = "Count: " + " ".join(str(i % 10) for i in range(60))
    prompt = list(probe._tokenizer.encode(words, add_special_tokens=True))[:n_prompt]
    probe.ctx.close()
    params = RequestParams(temp=0.0, max_tokens=100, stop_at_eog=False)
    seen = {}
    for batch in (False, True):
        eng = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=n_ctx, mlx_batch=batch))
        try:
            rid = eng.add_request(prompt, params)
            outs = list(eng.drain())
            assert eng.state(rid).finish_reason == "context", batch
            assert "".join(o.piece for o in outs) == eng.text_of(rid)
            seen[batch] = len(eng.tokens_of(rid))
        finally:
            eng.ctx.close()
    assert seen[False] == seen[True] == n_ctx - n_prompt, seen


# --- a character split across tokens reaches the pieces whole ----------------


CJK_PROMPT = "Say hello in Japanese (日本語で): "


@needs_weights
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_joined_pieces_are_the_text_when_characters_span_tokens(batch: bool) -> None:
    """SmolLM spells a CJK character as three single-byte tokens; decoded one
    at a time each came out as U+FFFD, so the routes (which build the body
    from the pieces) streamed '����し��し' where text_of said '你好したし'."""
    eng = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096, mlx_batch=batch))
    try:
        rid = eng.add_request(CJK_PROMPT, greedy(24))
        outs = list(eng.drain())
        text = eng.text_of(rid)
        assert any(ord(c) > 0x7F for c in text), text  # the prompt did its job
        joined = "".join(o.piece for o in outs)
        assert joined == text
        # A cap can land mid-character; only such a trailing remainder may be
        # a replacement character, and the retire carries it as its piece.
        assert "\ufffd" not in joined.rstrip("\ufffd")
        # An eog retire is never a generated token, whatever piece it flushed;
        # a length retire is one (its token is appended before the retire).
        assert len([o for o in outs if not (o.finished and o.finish_reason == "eog")]) == len(
            eng.tokens_of(rid)
        )
    finally:
        eng.ctx.close()


@needs_weights
def test_direct_feed_of_split_characters_joins_to_the_decode() -> None:
    eng = MLXEngine(MODEL_PATH, EngineConfig(n_ctx=4096))
    try:
        ids = eng.tokenize("🦀 日本語", add_special=False)
        assert len(ids) > 3  # byte-fallback: several tokens per character
        rid = eng.add_request("x", greedy(len(ids)))
        req = eng.state(rid)
        pieces = [eng._piece(req, t) for t in ids]
        assert "".join(pieces) + eng._flush_piece(req) == "🦀 日本語"
        assert "\ufffd" not in "".join(pieces)
        eng.cancel(rid)
    finally:
        eng.ctx.close()
