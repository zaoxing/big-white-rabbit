"""Unit coverage for engine/ngram.py (SPEC-speculative-ngram.md, T1).

Pure-Python table rules -- no model, no Context -- so every case here runs
without weights: lookup hits/misses, chained draft walks, most-recent-wins,
bounded eviction, stream feeding, and constructor guards.
"""

from __future__ import annotations

import pytest


def test_empty_table_predicts_nothing() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    assert NgramTable().predict([1, 2, 3], 4) == []
    assert len(NgramTable()) == 0


def test_single_observation_then_predict() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    t = NgramTable(order=3)
    t.add([10, 20], 30)
    assert t.predict([10, 20], 4) == [30]
    # Longer context is trimmed to the trailing order-1 tokens.
    assert t.predict([99, 10, 20], 4) == [30]
    # Anything else misses.
    assert t.predict([10, 21], 4) == []
    assert t.predict([10], 4) == []


def test_draft_walk_chains_through_table() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    t = NgramTable(order=3)
    t.update_stream([1, 2, 3, 4, 1, 2])
    # (1,2)->3, (2,3)->4, (3,4)->1, (4,1)->2, so from (1,2): 3,4,1,2.
    assert t.predict([1, 2], 4) == [3, 4, 1, 2]
    # Walk stops at the first miss: (2,2) was never observed.
    assert t.predict([2, 2], 4) == []


def test_walk_respects_max_tokens() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    t = NgramTable(order=2)
    t.update_stream([5, 6, 5, 6, 5, 6])
    assert t.predict([5], 2) == [6, 5]
    assert t.predict([5], 1) == [6]
    assert t.predict([5], 0) == []
    assert t.predict([5], -3) == []


def test_most_recent_continuation_wins() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    t = NgramTable(order=3)
    t.add([1, 2], 3)
    t.add([1, 2], 7)
    assert t.predict([1, 2], 2) == [7]
    assert len(t) == 1


def test_eviction_drops_oldest_entry() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    t = NgramTable(order=2, max_entries=2)
    t.add([1], 10)
    t.add([2], 20)
    t.add([3], 30)
    assert len(t) == 2
    assert t.predict([1], 1) == []
    assert t.predict([2], 1) == [20]
    assert t.predict([3], 1) == [30]


def test_reobserved_context_survives_eviction() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    t = NgramTable(order=2, max_entries=2)
    t.add([1], 10)
    t.add([2], 20)
    t.add([1], 11)  # refresh: (1,) is newest again
    t.add([3], 30)  # evicts (2,), the true oldest
    assert t.predict([1], 1) == [11]
    assert t.predict([2], 1) == []
    assert t.predict([3], 1) == [30]


def test_order_one_predicts_last_seen() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    t = NgramTable(order=1)
    t.update_stream([4, 9, 4])
    assert t.predict([12345], 2) == [4, 4]
    assert t.predict([12345], 1) == [4]
    assert len(t) == 1


def test_clear_forgets_everything() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    t = NgramTable()
    t.update_stream([1, 2, 3])
    assert len(t) > 0
    t.clear()
    assert len(t) == 0
    assert t.predict([1, 2], 3) == []


def test_constructor_guards() -> None:
    from freetoken_mac.engine.ngram import NgramTable

    with pytest.raises(ValueError):
        NgramTable(order=0)
    with pytest.raises(ValueError):
        NgramTable(max_entries=0)


def test_context_params_rs_default_zero() -> None:
    """Rollback snapshots stay off unless the engine asks (T5a)."""
    from freetoken_mac import ContextParams

    params = ContextParams()
    assert params.n_rs_seq == 0
    params.n_rs_seq = 5
    assert params.n_rs_seq == 5


def test_spec_arch_gate_rejects_hybrids_without_rollback() -> None:
    """Known hybrids with zero snapshots in effect must fail loud (their
    partial rewinds cannot work): T4 blocker regression test."""
    from freetoken_mac.engine.metal_engine import check_speculative_arch

    for arch in ("qwen35", "qwen35moe", "lfm2", "deepseek4"):
        with pytest.raises(ValueError, match="hybrid"):
            check_speculative_arch(arch, 0)


def test_spec_arch_gate_allows_supported_and_unknown() -> None:
    """Hybrids WITH snapshots pass (capability, not denylist, gates), as do
    attention architectures and missing / future arch strings."""
    from freetoken_mac.engine.metal_engine import check_speculative_arch

    check_speculative_arch("qwen35", 5)
    for arch in ("qwen2", "qwen3", "llama", None, "", "something-new"):
        check_speculative_arch(arch, 0)
