"""SpecThroughputGate: draft only while drafting measures faster.

Speculation pays when the tokens a verify step emits outrun the extra cost
of its rows. On llama.cpp/Metal a four-row verify costs little more than a
one-row decode; on MLX the quantized matmul scales almost linearly in rows
(SPEC-mlx-mtp-draft.md), so the same acceptance rate that wins on Metal
loses on MLX. A fixed acceptance threshold cannot know which machine it is
on. This gate does not predict: it times the engine's own verify steps and
compares tokens per second with drafts against tokens per second without.

Two exponential moving averages, one per mode, fed by every verify step.
Whichever mode is NOT current gets one probe step every ``probe_every``
steps, so a workload that changes shape (prose after a repetitive block,
or the reverse) is noticed rather than locked out. Until both averages
have ``min_samples`` behind them the gate drafts, which is the behaviour
it replaces. The gate is engine-wide, so what one request learned the next
one keeps -- the per-request fallback in the verify loop relearned from
zero on every request and paid for the lesson each time.
"""

from __future__ import annotations


class SpecThroughputGate:
    def __init__(
        self,
        *,
        alpha: float = 0.1,
        probe_every: int = 16,
        min_samples: int = 4,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        if probe_every < 2:
            raise ValueError(f"probe_every must be >= 2; got {probe_every}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1; got {min_samples}")
        self._alpha = alpha
        self._probe_every = probe_every
        self._min_samples = min_samples
        self._tps: dict[bool, float | None] = {True: None, False: None}
        self._samples: dict[bool, int] = {True: 0, False: 0}
        self._steps = 0
        self.probes = 0

    # -- inspection --------------------------------------------------------

    @property
    def draft_tps(self) -> float | None:
        return self._tps[True]

    @property
    def plain_tps(self) -> float | None:
        return self._tps[False]

    @property
    def drafting(self) -> bool:
        """The mode the evidence favours (ignoring probes)."""
        if (
            self._samples[True] < self._min_samples
            or self._samples[False] < self._min_samples
        ):
            return True
        d, p = self._tps[True], self._tps[False]
        assert d is not None and p is not None
        return d >= p

    # -- decisions ---------------------------------------------------------

    def wants_drafts(self) -> bool:
        """Whether the NEXT verify step should draft.

        Until both averages have ``min_samples`` behind them the modes
        alternate, so the first verdict is in on the order of eight steps
        rather than after ``min_samples * probe_every`` of them (a 128-token
        request spent half its length drafting at a loss before this).
        After that, every ``probe_every`` steps the answer is the opposite
        of the favoured mode, so both averages keep seeing fresh evidence.
        """
        short = [
            mode for mode in (True, False)
            if self._samples[mode] < self._min_samples
        ]
        if short:
            # The mode with less evidence; drafts first on a tie.
            return min(short, key=lambda m: (self._samples[m], not m))
        if self._steps and self._steps % self._probe_every == 0:
            self.probes += 1
            return not self.drafting
        return self.drafting

    def record(self, drafted: bool, tokens: int, seconds: float) -> None:
        """One finished verify step: whether it drafted (at least one draft
        row went into the forward), how many tokens it emitted, and the
        wall time its forwards took, including any mismatch re-feed."""
        self._steps += 1
        if tokens <= 0 or seconds <= 0:
            return
        tps = tokens / seconds
        prev = self._tps[drafted]
        self._tps[drafted] = (
            tps if prev is None else prev + self._alpha * (tps - prev)
        )
        self._samples[drafted] += 1
