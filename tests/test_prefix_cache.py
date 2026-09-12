"""Prefix-cache TTFT (SPEC-prefix-cache.md, T8a).

Needs BWR_TEST_MODEL like the other engine tests; skips without it. The
shared stem ("word " * N) tokenises identically in both prompts, so the fork
covers all but the tail token; the suite pins exactness (forked == plain) and
efficacy (fewer prefill decodes, hit counters) separately.
"""

from __future__ import annotations

import os

import pytest

import bwr as bwr
from bwr.engine import EngineConfig, MetalEngine, RequestParams

MODEL_PATH = os.environ.get("BWR_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_TEST_MODEL to a .gguf path to run prefix-cache tests",
)

STEM = "word " * 300
PROMPT_A = STEM + "alpha"
PROMPT_B = STEM + "beta"
N_TOKENS = 8


def cache_config(**overrides) -> EngineConfig:
    base = {
        "n_ctx": 1024,
        "n_batch": 64,
        # n_seq_max=2: fork tests need one slot for the pin plus one live
        # request (a single slot forces pin eviction on every admission --
        # see test_single_slot_eviction_path).
        "n_seq_max": 2,
        "prefix_cache": True,
    }
    base.update(overrides)
    return EngineConfig(**base)


def plain_config(**overrides) -> EngineConfig:
    base = {"n_ctx": 1024, "n_batch": 64, "n_seq_max": 2}
    base.update(overrides)
    return EngineConfig(**base)


def greedy(max_tokens: int = N_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False)


@pytest.fixture(scope="module")
def model() -> bwr.Model:
    return bwr.Model(MODEL_PATH, bwr.ModelParams())


def run(engine: MetalEngine, prompt: str, params: RequestParams | None = None) -> int:
    rid = engine.add_request(prompt, params or greedy())
    list(engine.drain())
    return rid


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_disabled_by_default(model: bwr.Model) -> None:
    engine = MetalEngine(model, plain_config())
    run(engine, PROMPT_A)
    assert engine._pins == {}
    assert engine.prefix_cache_hits == 0
    assert engine.prefix_cache_tokens_saved == 0


def test_shared_prefix_forks_identical(model: bwr.Model) -> None:
    """Forked output == plain output, with fewer prefill decodes."""
    cached = MetalEngine(model, cache_config())
    run(cached, PROMPT_A)
    assert len(cached._pins) == 1
    before = cached.ctx.decode_calls
    rid_b = run(cached, PROMPT_B)

    control = MetalEngine(model, plain_config())
    before_control = control.ctx.decode_calls
    rid_c = run(control, PROMPT_B)
    control_calls = control.ctx.decode_calls - before_control

    assert cached.tokens_of(rid_b) == control.tokens_of(rid_c)
    assert cached.state(rid_b).finish_reason == control.state(rid_c).finish_reason
    assert cached.prefix_cache_hits == 1
    # Suffix is one token + backoff one: far fewer prefill decodes.
    assert cached.ctx.decode_calls - before < control_calls


def test_exact_rerun_skips_prefill(model: bwr.Model) -> None:
    cached = MetalEngine(model, cache_config())
    run(cached, PROMPT_A)
    first_calls = cached.ctx.decode_calls
    rid = run(cached, PROMPT_A)
    second_calls = cached.ctx.decode_calls - first_calls
    # Backoff re-decodes exactly one token whose sample is already output #1,
    # so the rerun costs one prefill-as-generation call plus the rest.
    assert second_calls == N_TOKENS
    assert second_calls < first_calls
    assert cached.prefix_cache_hits == 1
    assert len(cached.tokens_of(rid)) == N_TOKENS


def test_below_threshold_no_fork(model: bwr.Model) -> None:
    cached = MetalEngine(model, cache_config())
    run(cached, "word " * 20 + "alpha")
    run(cached, "word " * 20 + "beta")
    assert cached.prefix_cache_hits == 0
    assert cached._pins == {}


def test_eviction_bounded(model: bwr.Model) -> None:
    cached = MetalEngine(
        model, cache_config(prefix_cache_pins=1, n_seq_max=2)
    )
    control = MetalEngine(model, plain_config(n_seq_max=2))
    for tail in ("alpha", "beta", "gamma"):
        prompt = STEM + tail
        rid = run(cached, prompt)
        ref = run(control, prompt)
        assert len(cached._pins) <= 1
        assert cached.tokens_of(rid) == control.tokens_of(ref)


def test_single_slot_eviction_path(model: bwr.Model) -> None:
    """n_seq_max=1: the matched pin itself must go, falling back to plain."""
    cached = MetalEngine(model, cache_config(n_seq_max=1))
    run(cached, PROMPT_A)
    assert len(cached._pins) == 1
    control = MetalEngine(model, plain_config(n_seq_max=1))
    rid_b = run(cached, PROMPT_B)
    rid_c = run(control, PROMPT_B)
    assert cached.tokens_of(rid_b) == control.tokens_of(rid_c)
    assert cached.prefix_cache_hits == 0
    assert len(cached._pins) == 1  # B pinned in turn


def test_cancel_hygiene(model: bwr.Model) -> None:
    cached = MetalEngine(model, cache_config())
    run(cached, PROMPT_A)
    rid_b = cached.add_request(PROMPT_B, greedy())
    for _ in range(3):
        cached.step()
    assert cached.cancel(rid_b) is True
    control = MetalEngine(model, plain_config())
    rid_c = run(control, PROMPT_B)
    rid_d = run(cached, PROMPT_B)
    assert cached.tokens_of(rid_d) == control.tokens_of(rid_c)
    assert cached.prefix_cache_hits == 2


def test_stop_filter_agrees_under_fork(model: bwr.Model) -> None:
    """Whatever the stop filter decides, forked and plain runs must agree exactly:
    same tokens, same reason. (Also covers pinning after a stop retire.)"""
    tail = "Say the word banana. banana banana banana"
    params = RequestParams(temp=0.0, max_tokens=32, stop_at_eog=False, stop=("banana",))
    cached = MetalEngine(model, cache_config())
    run(cached, STEM + tail)
    rid_b = run(cached, STEM + tail, params)
    control = MetalEngine(model, plain_config())
    rid_c = run(control, STEM + tail, params)
    assert cached.tokens_of(rid_b) == control.tokens_of(rid_c)
    assert cached.state(rid_b).finish_reason == control.state(rid_c).finish_reason
    assert cached.prefix_cache_hits == 1


def test_forks_are_deterministic(model: bwr.Model) -> None:
    """Triple-fork proof: three forks from one pin plus the plain control all
    agree exactly. On hybrids this fails (nondeterministic copy/rewind --
    hence the construction gate); on attention it must hold bit-for-bit."""
    cached = MetalEngine(model, cache_config())
    run(cached, PROMPT_A)
    outs = [run(cached, PROMPT_A) for _ in range(3)]
    control = MetalEngine(model, plain_config())
    ref = run(control, PROMPT_A)
    assert cached.prefix_cache_hits == 3
    for rid in outs:
        assert cached.tokens_of(rid) == control.tokens_of(ref)


def test_config_validation(model: bwr.Model) -> None:
    with pytest.raises(ValueError, match="prefix_cache_pins"):
        MetalEngine(model, cache_config(prefix_cache_pins=-1))
    with pytest.raises(ValueError, match="prefix_cache_min_tokens"):
        MetalEngine(model, cache_config(prefix_cache_min_tokens=0))


def test_hybrid_arch_rejected_at_construction(model: bwr.Model) -> None:
    """Fork copies diverge nondeterministically on hybrids (T8b evidence):
    fail loud here. Pure gate-logic pins below; this proves the wiring."""
    from bwr.engine.metal_engine import check_prefix_cache_arch

    for arch in ("qwen35", "qwen35moe", "lfm2", "deepseek4"):
        with pytest.raises(ValueError, match="hybrid"):
            check_prefix_cache_arch(arch)
def test_hybrid_target_refused(
    model: bwr.Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: stubbed hybrid arch string trips the gate in the
    constructor (attention weights underneath, so no other gate fires)."""
    real_meta = bwr.Model.meta_val

    def fake_meta(self: bwr.Model, key: str) -> str:
        if key == "general.architecture":
            return "qwen35"
        return real_meta(self, key)

    monkeypatch.setattr(bwr.Model, "meta_val", fake_meta)
    with pytest.raises(ValueError, match="hybrid"):
        MetalEngine(model, cache_config())


def test_fork_failure_falls_back_to_plain(
    model: bwr.Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed truncate (the hybrid-without-snapshots shape) must fall back
    to a plain prefill: identical output, no hit counted. The stub fails
    partial removes only; full clears pass through untouched."""
    cached = MetalEngine(model, cache_config())
    run(cached, PROMPT_A)
    assert len(cached._pins) == 1
    real_rm = bwr.Context.memory_seq_rm

    def flaky_rm(self: bwr.Context, seq_id: int, p0: int = -1, p1: int = -1) -> bool:
        if p0 is not None and p0 >= 0:
            return False
        return real_rm(self, seq_id, p0, p1)

    monkeypatch.setattr(bwr.Context, "memory_seq_rm", flaky_rm)
    control = MetalEngine(model, plain_config())
    rid_b = run(cached, PROMPT_B)
    rid_c = run(control, PROMPT_B)
    assert cached.tokens_of(rid_b) == control.tokens_of(rid_c)
    assert cached.state(rid_b).finish_reason == control.state(rid_c).finish_reason
    assert cached.prefix_cache_hits == 0
    assert cached.prefix_cache_tokens_saved == 0
