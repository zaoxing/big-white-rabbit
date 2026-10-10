"""SpecThroughputGate: pure-logic tests, no engine."""

from __future__ import annotations

import pytest
from bwr.engine.spec_gate import SpecThroughputGate


def _steps(gate: SpecThroughputGate, n: int, *, draft_tps: float, plain_tps: float):
    """Drive n steps. Each step asks the gate, then reports a measurement
    in the mode it chose, at the given rate for that mode. Returns the
    modes chosen, in order."""
    chosen = []
    for _ in range(n):
        d = gate.wants_drafts()
        chosen.append(d)
        gate.record(d, tokens=4 if d else 1, seconds=(4 if d else 1) / (draft_tps if d else plain_tps))
    return chosen


def test_alternates_until_both_modes_have_evidence():
    """Plain evidence must not wait on the probe cadence: with four
    samples needed and a probe every sixteen steps, the first verdict
    would be 64 steps out -- half a 128-token request drafting blind."""
    gate = SpecThroughputGate(probe_every=16, min_samples=2)
    assert gate.drafting is True  # the default until there is evidence
    assert gate.draft_tps is None and gate.plain_tps is None
    chosen = _steps(gate, 4, draft_tps=8, plain_tps=11)
    assert chosen == [True, False, True, False]
    assert gate.drafting is False  # verdict after four steps, not 32
    assert gate.probes == 0


def test_probes_back_off_while_the_verdict_holds():
    """A probe costs several plain steps on MLX; one in sixteen forever
    would give back a tenth of what the gate saves. Confirmed verdicts
    double the wait, capped."""
    gate = SpecThroughputGate(probe_every=4, min_samples=1, max_probe_every=16)
    chosen = _steps(gate, 40, draft_tps=10, plain_tps=10)
    # Steps 0-1 alternate for evidence. Then a probe at 4; the counter
    # restarts when the probe records, so the next comes 8 steps later at
    # 13, then 16 later (capped) at 30, and the one after would be 47.
    assert [i for i, d in enumerate(chosen) if not d] == [1, 4, 13, 30]
    assert gate.probes == 3


def test_a_probe_that_flips_the_verdict_resets_the_cadence():
    gate = SpecThroughputGate(probe_every=4, min_samples=1, max_probe_every=64, alpha=1.0)
    _steps(gate, 20, draft_tps=10, plain_tps=5)   # drafting; cadence grows
    assert gate.drafting is True
    assert gate._cadence > 4
    # Plain is now much faster: the next probe flips the verdict.
    while gate.probes < 3:
        _steps(gate, 1, draft_tps=10, plain_tps=50)
    assert gate.drafting is False
    assert gate._cadence == 4


def test_switches_to_plain_when_drafting_measures_slower():
    """The MLX shape: four rows cost three, acceptance is middling, so a
    drafting step emits fewer tokens per second than a plain one."""
    gate = SpecThroughputGate(probe_every=4, min_samples=2, alpha=0.5)
    _steps(gate, 12, draft_tps=8, plain_tps=11)
    assert gate.drafting is False
    assert gate.plain_tps > gate.draft_tps
    # And stays plain apart from the occasional probe.
    assert _steps(gate, 8, draft_tps=8, plain_tps=11).count(True) <= 1


def test_switches_back_when_drafts_start_paying():
    """Repetitive text after prose: the probes carry the news."""
    gate = SpecThroughputGate(probe_every=4, min_samples=2, alpha=0.5)
    _steps(gate, 12, draft_tps=8, plain_tps=11)
    assert gate.drafting is False
    _steps(gate, 24, draft_tps=16, plain_tps=11)
    assert gate.drafting is True


def test_empty_or_instant_steps_are_not_evidence():
    gate = SpecThroughputGate(min_samples=1)
    gate.record(True, tokens=0, seconds=0.1)
    gate.record(True, tokens=3, seconds=0.0)
    assert gate.draft_tps is None


def test_record_counts_the_step_even_when_it_is_not_evidence():
    """Probing is on a step cadence; a step that measured nothing still
    moves the schedule, or an engine that never emits would never probe."""
    gate = SpecThroughputGate(probe_every=2, min_samples=1)
    _steps(gate, 2, draft_tps=10, plain_tps=10)  # evidence for both
    gate.record(True, tokens=0, seconds=0.0)  # step 2, nothing measured
    gate.record(True, tokens=0, seconds=0.0)  # step 3
    assert gate.wants_drafts() is False  # step 4: a probe, on schedule


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0}, {"alpha": 1.5}, {"probe_every": 1}, {"min_samples": 0},
    {"probe_every": 8, "max_probe_every": 4},
])
def test_rejects_degenerate_parameters(kwargs):
    with pytest.raises(ValueError):
        SpecThroughputGate(**kwargs)
