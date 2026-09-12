"""Hotlist tracker — LRU hit-rate validation (SPEC-ssd-hotlist.md, T11a).

Unit tests are pure (no model). Integration uses BWR_MOE_MODEL-gated live
expert_activations() to prove determinism + LRU hit rates matching the
probe's 0.76 @ K=32 / 0.91 @ K=64 on real traffic before any I/O.
"""

from __future__ import annotations

import os

import pytest

import bwr as bwr
from bwr.engine import EngineConfig, MetalEngine
from bwr.engine.hotlist import ExpertHotlist

MODEL_PATH = os.environ.get("BWR_MOE_MODEL")

pytestmark_moe = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_MOE_MODEL to a MoE .gguf path to run hotlist integration",
)


def test_lru_eviction() -> None:
    hl = ExpertHotlist(k_per_layer=2, top_k=1)
    # Layer 0: experts [0], [1], [2] -> evicts 0, resident [1,2] MRU order
    hl.update([{"layer": 0, "tokens": [[1.0, 0.0, 0.0]]}])
    hl.update([{"layer": 0, "tokens": [[0.0, 1.0, 0.0]]}])
    hl.update([{"layer": 0, "tokens": [[0.0, 0.0, 1.0]]}])
    assert hl.resident(0) == [1, 2]
    assert hl.hits == 0 and hl.misses == 3


def test_hit_rate_and_resident_order() -> None:
    hl = ExpertHotlist(k_per_layer=2, top_k=1)
    hl.update([{"layer": 5, "tokens": [[1.0, 0.0]]}])
    hl.update([{"layer": 5, "tokens": [[1.0, 0.0]]}])  # hit, moves to MRU
    hl.update([{"layer": 5, "tokens": [[0.0, 1.0]]}])
    hl.update([{"layer": 5, "tokens": [[1.0, 0.0]]}])  # hit again
    assert hl.hit_rate() == pytest.approx(0.5)
    assert hl.resident(5) == [1, 0]  # 1 was LRU, 0 promoted to MRU


def test_top_k_routing() -> None:
    hl = ExpertHotlist(k_per_layer=4, top_k=2)
    # Row: expert 2 prob 0.5, expert 0 prob 0.4 -> top-2 = [2,0]
    hl.update([{"layer": 0, "tokens": [[0.4, 0.1, 0.5]]}])
    assert set(hl.resident(0)) == {0, 2}
    # Next token routes [1,2] -> 2 is hit, 1 is miss
    result = hl.update([{"layer": 0, "tokens": [[0.1, 0.4, 0.3]]}])
    assert result["hits"] == 1 and result["misses"] == 1
    assert (0, 1) in result["missed"]  # type: ignore[index]


def test_clear_and_validation() -> None:
    hl = ExpertHotlist(k_per_layer=2, top_k=1)
    hl.update([{"layer": 0, "tokens": [[1.0, 0.0]]}])
    hl.clear()
    assert hl.hit_rate() is None
    assert hl.resident(0) == []
    with pytest.raises(ValueError):
        ExpertHotlist(k_per_layer=0)
    with pytest.raises(ValueError):
        ExpertHotlist(top_k=0)


def test_multi_layer_isolation() -> None:
    hl = ExpertHotlist(k_per_layer=1, top_k=1)
    hl.update([{"layer": 0, "tokens": [[1.0, 0.0]]}])
    hl.update([{"layer": 1, "tokens": [[0.0, 1.0]]}])
    assert hl.resident(0) == [0]
    assert hl.resident(1) == [1]


def test_engine_hotlist_wiring() -> None:
    """Engine with ssd_hotlist auto-feeds hotlist from expert_activations."""
    cfg = EngineConfig(
        n_ctx=512, n_seq_max=1, record_experts=True,
        ssd_hotlist=True, ssd_hotlist_k=32, ssd_hotlist_top_k=8,
    )
    model = bwr.Model(MODEL_PATH, bwr.ModelParams())
    engine = MetalEngine(model, cfg)
    assert engine._hotlist is not None
    rid = engine.add_request("Count: 1 2 3", bwr.RequestParams(max_tokens=8, stop_at_eog=False, temp=0.0))
    list(engine.drain())
    # Hotlist auto-fed: should have hits/misses, manual drain now empty
    assert engine._hotlist.hits + engine._hotlist.misses > 0
    assert engine._hotlist.hit_rate() is not None
    assert engine.expert_activations(rid) == []  # consumed by hotlist
    model.close()


def test_hotlist_needs_recording() -> None:
    with pytest.raises(ValueError, match="record_experts"):
        EngineConfig(n_ctx=512, n_seq_max=1, ssd_hotlist=True, record_experts=False).validate_hotlist()
    # Engine should also refuse at construction
    cfg = EngineConfig(n_ctx=512, n_seq_max=1, ssd_hotlist=True, record_experts=False)
    model = bwr.Model(MODEL_PATH, bwr.ModelParams())
    with pytest.raises(ValueError, match="record_experts"):
        MetalEngine(model, cfg)
    model.close()


@pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_MOE_MODEL to a MoE .gguf path to run hotlist integration",
)
def test_live_determinism() -> None:
    """Same prompt -> same routing -> same hotlist sequence."""
    cfg = EngineConfig(n_ctx=512, n_seq_max=1, record_experts=True)

    def run() -> ExpertHotlist:
        model = bwr.Model(MODEL_PATH, bwr.ModelParams())
        engine = MetalEngine(model, cfg)
        hl = ExpertHotlist(k_per_layer=32, top_k=8)
        rid = engine.add_request("Count: 1 2 3", bwr.RequestParams(max_tokens=16, stop_at_eog=False, temp=0.0))
        # Drain in steps so we interleave hotlist updates with decodes (real usage)
        while engine._states:
            for out in engine.step():
                pass
            hl.update(engine.expert_activations(rid) if rid in engine._states or rid in engine._retired else [])
        # Final drain for last decode's frames (retired request still readable)
        hl.update(engine.expert_activations(rid))
        model.close()
        return hl

    a = run()
    b = run()
    assert a.hits == b.hits and a.misses == b.misses
    # Spot-check one layer's resident set (deterministic LRU order)
    assert a.resident(0) == b.resident(0)


@pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_MOE_MODEL to a MoE .gguf path to run hotlist integration",
)
def test_live_hit_rate_bands() -> None:
    """Reproduce probe's LRU hit rates on live traffic (the streaming gate)."""
    model = bwr.Model(MODEL_PATH, bwr.ModelParams())
    cfg = EngineConfig(n_ctx=512, n_seq_max=1, record_experts=True)
    engine = MetalEngine(model, cfg)
    rid = engine.add_request(
        "Explain what a mixture-of-experts model is, briefly.",
        bwr.RequestParams(max_tokens=100, stop_at_eog=False, temp=0.0),
    )
    # Per-step drain: Context clears frames each decode, so collecting once
    # at the end would see only the last token's 48 frames.
    hl32 = ExpertHotlist(k_per_layer=32, top_k=8)
    hl64 = ExpertHotlist(k_per_layer=64, top_k=8)
    while engine.has_work:
        engine.step()
        frames = engine.expert_activations(rid)
        hl32.update(frames)
        hl64.update(frames)
    model.close()

    for hl, floor in [(hl32, 0.65), (hl64, 0.85)]:
        assert hl.hit_rate() is not None and hl.hit_rate() >= floor, f"K={hl.k_per_layer} hit {hl.hit_rate():.3f} < {floor}"
