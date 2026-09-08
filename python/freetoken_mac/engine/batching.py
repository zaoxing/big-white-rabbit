"""Admission seam: what goes into the next single ``llama_decode`` call.

``MetalEngine`` knows how to *build and run* one batch; it does not decide what
belongs in it. That decision is an ``AdmissionPolicy``, so Phase 3/4 (RAM-budget
admission, prefix-aware reordering, expert-aware grouping) plug in here without
touching the decode path. ``FCFSPolicy`` is the deliberately dumb default:
prefill first, then one decode slot per generating sequence, both cut off at the
effective ``n_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from .config import RequestParams


@dataclass
class RequestState:
    """Live bookkeeping for one admitted request. Owned by the engine; policies see
    it read-only."""

    request_id: int
    seq_id: int
    prompt: list[int]
    params: RequestParams
    n_prefilled: int = 0
    # Next KV position to write for this sequence == tokens already in its KV.
    n_pos: int = 0
    # Token sampled last step, waiting to be decoded to produce the next logits.
    next_token: int | None = None
    n_generated: int = 0
    finished: bool = False
    finish_reason: str | None = None
    output_tokens: list[int] = field(default_factory=list)

    @property
    def n_prompt_remaining(self) -> int:
        return len(self.prompt) - self.n_prefilled

    @property
    def is_prefilling(self) -> bool:
        return not self.finished and self.n_prompt_remaining > 0

    @property
    def is_generating(self) -> bool:
        return not self.finished and self.n_prompt_remaining == 0 and self.next_token is not None


@dataclass(frozen=True)
class StepBudget:
    """Effective, post-clamp limits for one step. Read off the live context, never
    from EngineConfig."""

    n_batch: int
    # Per-sequence room (ctx.n_ctx_seq), NOT the total (ctx.n_ctx = n_ctx_seq*n_seq_max).
    n_ctx_seq: int
    free_seq_slots: int


@dataclass(frozen=True)
class PrefillChunk:
    request_id: int
    n_tokens: int


@dataclass(frozen=True)
class StepPlan:
    """The contents of the next batch: prompt chunks plus one-token continuations.
    ``n_tokens`` must not exceed the budget's ``n_batch`` -- the engine rejects a
    plan that does rather than letting llama.cpp's assert abort the process."""

    prefill: tuple[PrefillChunk, ...] = ()
    decode: tuple[int, ...] = ()

    @property
    def n_tokens(self) -> int:
        return sum(c.n_tokens for c in self.prefill) + len(self.decode)

    @property
    def is_empty(self) -> bool:
        return self.n_tokens == 0


@runtime_checkable
class AdmissionPolicy(Protocol):
    """The pluggable scheduling seam."""

    def choose_step_batch(
        self,
        pending: Sequence[RequestState],
        in_flight: Sequence[RequestState],
        budget: StepBudget,
    ) -> StepPlan:
        """Pick this step's batch contents from the requests still needing prefill
        (``pending``) and those generating (``in_flight``)."""
        ...


class FCFSPolicy:
    """First-come-first-served, prefill-priority, chunked to the effective n_batch."""

    def choose_step_batch(
        self,
        pending: Sequence[RequestState],
        in_flight: Sequence[RequestState],
        budget: StepBudget,
    ) -> StepPlan:
        remaining = budget.n_batch
        chunks: list[PrefillChunk] = []
        for req in pending:
            if remaining <= 0:
                break
            # A prompt longer than n_batch is split across steps; a chunk is also
            # capped so the sequence never runs past the context.
            room = max(0, budget.n_ctx_seq - req.n_pos)
            take = min(req.n_prompt_remaining, remaining, room)
            if take <= 0:
                continue
            chunks.append(PrefillChunk(req.request_id, take))
            remaining -= take

        decode: list[int] = []
        for req in in_flight:
            if remaining <= 0:
                break
            decode.append(req.request_id)
            remaining -= 1

        return StepPlan(prefill=tuple(chunks), decode=tuple(decode))
