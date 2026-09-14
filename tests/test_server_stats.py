"""Request counters behind /admin/api/stats and /admin/api/usage."""

from __future__ import annotations

from bwr.server.stats import ServerStats


def test_snapshot_has_every_key_statsdto_requires():
    """StatsDTO's fields are non-optional, and Swift discards the whole
    response over one missing key -- so the zero state must still be the
    full shape, not an empty dict."""
    snap = ServerStats().snapshot()
    for key in ("total_requests", "total_prompt_tokens", "total_completion_tokens",
                "total_tokens_served", "total_cached_tokens", "cache_efficiency",
                "avg_prefill_tps", "avg_generation_tps", "uptime_seconds"):
        assert key in snap


def test_totals_accumulate():
    s = ServerStats()
    s.record(prompt_tokens=10, completion_tokens=5)
    s.record(prompt_tokens=20, completion_tokens=7)
    snap = s.snapshot()
    assert snap["total_requests"] == 2
    assert snap["total_prompt_tokens"] == 30
    assert snap["total_completion_tokens"] == 12
    assert snap["total_tokens_served"] == 42


def test_cache_efficiency_is_a_token_ratio():
    s = ServerStats()
    s.record(prompt_tokens=100, completion_tokens=1, cached_tokens=25)
    assert s.snapshot()["cache_efficiency"] == 0.25


def test_zero_prompt_tokens_is_zero_efficiency_not_a_crash():
    assert ServerStats().snapshot()["cache_efficiency"] == 0.0


def test_a_cached_prompt_does_not_enter_the_prefill_average():
    """A prefix-cache hit skips prefill entirely. Averaging its ~0s in would
    report a prefill rate the machine never achieved."""
    s = ServerStats()
    s.record(prompt_tokens=100, completion_tokens=1,
             prefill_seconds=1.0, decode_seconds=1.0)
    baseline = s.snapshot()["avg_prefill_tps"]
    s.record(prompt_tokens=100, completion_tokens=1, cached_tokens=100,
             prefill_seconds=0.0001, decode_seconds=1.0)
    assert s.snapshot()["avg_prefill_tps"] == baseline


def test_prefill_and_decode_rates_are_measured_separately():
    s = ServerStats()
    s.record(prompt_tokens=1000, completion_tokens=10,
             prefill_seconds=1.0, decode_seconds=2.0)
    snap = s.snapshot()
    assert snap["avg_prefill_tps"] == 1000.0
    assert snap["avg_generation_tps"] == 5.0


def test_usage_breaks_totals_down_per_model():
    s = ServerStats()
    s.record(prompt_tokens=10, completion_tokens=2, model="a")
    s.record(prompt_tokens=30, completion_tokens=4, model="b")
    s.record(prompt_tokens=10, completion_tokens=1, model="a")
    usage = s.usage()
    by_id = {m["model_id"]: m for m in usage["models"]}
    assert by_id["a"]["requests"] == 2
    assert by_id["a"]["total_tokens"] == 23
    assert by_id["b"]["requests"] == 1
    assert usage["totals"]["requests"] == 3


def test_usage_reports_no_time_series_rather_than_inventing_one():
    """bwr keeps no per-day history. An empty heatmap beside correct totals
    is honest; synthesised buckets would not be."""
    s = ServerStats()
    s.record(prompt_tokens=1, completion_tokens=1, model="a")
    assert s.usage()["heatmap"] == []
    assert s.usage()["available"] is True


def test_clear_resets_totals_and_the_per_model_breakdown():
    s = ServerStats()
    s.record(prompt_tokens=10, completion_tokens=2, model="a")
    s.clear()
    assert s.snapshot()["total_requests"] == 0
    assert s.usage()["models"] == []


def test_stats_never_claim_to_be_persisted():
    """There is no stats database, so "all time" cannot outlive the process."""
    assert ServerStats().snapshot()["persisted"] is False
