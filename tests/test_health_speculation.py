"""/health's speculation block: what the gate is doing, without engine
internals. Pure shape tests on stub engines."""

from __future__ import annotations

from bwr.engine.spec_gate import SpecThroughputGate
from bwr.server.app import speculation_summary
from bwr.tune import render


class _Counters:
    spec_drafted = 40
    spec_accepted = 30
    spec_recomputes = 2
    spec_fallbacks = 1

    @property
    def spec_acceptance_rate(self):
        return self.spec_accepted / self.spec_drafted


class _WithGate(_Counters):
    def __init__(self):
        self.spec_gate = SpecThroughputGate(min_samples=1)
        self.spec_gate.record(True, 4, 0.5)
        self.spec_gate.record(False, 1, 0.1)


def test_engines_without_counters_report_nothing():
    assert speculation_summary(object()) is None


def test_counters_without_a_gate_have_no_gate_block():
    body = speculation_summary(_Counters())
    assert body["drafted"] == 40 and body["acceptance_rate"] == 0.75
    assert "gate" not in body


def test_gate_block_says_whether_drafts_are_paying():
    body = speculation_summary(_WithGate())
    gate = body["gate"]
    assert gate["drafting"] is False  # 8 tok/s drafting vs 10 plain
    assert gate["draft_tps"] == 8.0 and gate["plain_tps"] == 10.0
    assert gate["probes"] == 0


def test_tune_report_shows_the_gate_column():
    rows = [
        {"depth": None, "tok_s": 100.0, "accept": None, "gate": None},
        {"depth": 4, "tok_s": 99.0, "accept": 0.4, "gate": "plain"},
    ]
    out = render(rows, None)
    assert "gate" in out.splitlines()[0]
    assert "plain" in out
