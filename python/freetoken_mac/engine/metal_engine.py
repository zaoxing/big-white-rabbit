"""MetalEngine: N requests interleaved through one ``llama_decode`` per step.

Phase 0's loop was one sequence, one token, one decode. Here every admitted
request owns a ``seq_id`` (from a free list bounded by the context's effective
``n_seq_max``) and its own sampler chain, and each ``step()`` packs prompt chunks
plus one continuation token per generating sequence into a single ``llama_batch``
and issues exactly one decode. ``ctx.decode_calls`` (counted in C++) is the proof.
Thin glue on purpose: llama.cpp already does the batching, and *what* goes into
the next batch is the ``AdmissionPolicy``'s call, not this class's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .._freetoken_metal import Batch, Context, Model
from .batching import AdmissionPolicy, FCFSPolicy, RequestState, StepBudget, StepPlan
from .config import EngineConfig, RequestParams


@dataclass(frozen=True)
class StepOutput:
    """One token produced for one request by one step."""

    request_id: int
    token: int
    piece: str
    finished: bool
    finish_reason: str | None = None


class SeqIdExhausted(RuntimeError):
    """More concurrent requests than the context's n_seq_max.

    Raised instead of admitting the request: llama.cpp rejects a batch that names a
    seq_id >= n_seq_max, and the surrounding failure modes are aborts, so refusing
    admission in Python is the only survivable answer.
    """


class MetalEngine:
    """Continuous-batching engine over a single llama.cpp Metal context."""

    def __init__(
        self,
        model: Model,
        config: EngineConfig | None = None,
        policy: AdmissionPolicy | None = None,
        *,
        default_params: RequestParams | None = None,
    ) -> None:
        self.model = model
        self.config = config or EngineConfig()
        self.policy: AdmissionPolicy = policy or FCFSPolicy()
        self.default_params = default_params or RequestParams()
        self.ctx = Context(model, self.config.to_context_params())

        # Effective geometry only: n_ctx rounded up, n_batch clamped down, per-sequence
        # room = n_ctx_seq (see StepBudget) -- never the requested values.
        self._batch = Batch(self.ctx.n_batch, 1)
        self._free_seq_ids: list[int] = list(range(self.ctx.n_seq_max))
        self._states: dict[int, RequestState] = {}
        self._next_request_id = 0

    # --- admission --------------------------------------------------------------

    def add_request(
        self,
        prompt: str | Sequence[int],
        params: RequestParams | None = None,
        *,
        add_special: bool = True,
    ) -> int:
        """Admit a request and return its id. Raises before touching llama.cpp on an
        empty prompt, a prompt that cannot fit the context, or seq_id exhaustion."""
        if isinstance(prompt, str):
            tokens = list(self.model.tokenize(prompt, add_special=add_special, parse_special=True))
        else:
            tokens = list(prompt)
        if not tokens:
            raise ValueError("prompt tokenized to zero tokens")
        # Per-sequence capacity, not the context total: n_ctx is n_ctx_seq * n_seq_max,
        # so budgeting against it admits prompts that only fail later inside decode.
        if len(tokens) >= self.ctx.n_ctx_seq:
            raise ValueError(
                f"prompt is {len(tokens)} tokens, per-sequence capacity is "
                f"{self.ctx.n_ctx_seq} (n_ctx {self.ctx.n_ctx} / n_seq_max "
                f"{self.ctx.n_seq_max})"
            )
        if not self._free_seq_ids:
            raise SeqIdExhausted(
                f"all {self.ctx.n_seq_max} sequence slots are in use "
                "(raise EngineConfig.n_seq_max or wait for a request to finish)"
            )

        seq_id = self._free_seq_ids.pop(0)
        request_id = self._next_request_id
        self._next_request_id += 1

        # A retired predecessor may have left KV behind on this slot.
        self.ctx.memory_seq_rm(seq_id, -1, -1)
        rp = params or self.default_params
        self.ctx.set_seq_sampler(seq_id, rp.to_sampler_params())

        self._states[request_id] = RequestState(
            request_id=request_id, seq_id=seq_id, prompt=tokens, params=rp
        )
        return request_id

    def cancel(self, request_id: int) -> bool:
        """Drop a request mid-flight. Its KV and sampler go away and its seq_id returns
        to the pool; peers in the same batch are untouched. False if already finished."""
        req = self._states[request_id]
        if req.finished:
            return False
        self._retire(req, "cancelled")
        return True

    # --- stepping ---------------------------------------------------------------

    def step(self) -> list[StepOutput]:
        """Run one batched decode. Exactly one ``llama_decode`` call, whatever the
        in-flight count; returns the tokens sampled for the rows that carried logits."""
        pending = [r for r in self._states.values() if r.is_prefilling]
        in_flight = [r for r in self._states.values() if r.is_generating]
        if not pending and not in_flight:
            return []

        budget = StepBudget(
            n_batch=self.ctx.n_batch,
            n_ctx_seq=self.ctx.n_ctx_seq,
            free_seq_slots=len(self._free_seq_ids),
        )
        plan = self.policy.choose_step_batch(pending, in_flight, budget)
        if plan.is_empty:
            return []
        if plan.n_tokens > budget.n_batch:
            raise ValueError(
                f"policy planned {plan.n_tokens} tokens but n_batch is {budget.n_batch}"
            )

        rows, commits = self._fill_batch(plan)
        # THE decode. Position/prefill bookkeeping is committed only after it returns,
        # so a rejected batch leaves every request exactly where it was.
        self.ctx.decode(self._batch)
        for req, n_prefilled, n_pos in commits:
            req.n_prefilled = n_prefilled
            req.n_pos = n_pos

        return [self._advance(req, row) for req, row in rows]

    def _fill_batch(
        self, plan: StepPlan
    ) -> tuple[list[tuple[RequestState, int]], list[tuple[RequestState, int, int]]]:
        self._batch.clear()
        rows: list[tuple[RequestState, int]] = []
        commits: list[tuple[RequestState, int, int]] = []

        for chunk in plan.prefill:
            req = self._states[chunk.request_id]
            end = req.n_prefilled + chunk.n_tokens
            is_last_chunk = end >= len(req.prompt)
            for j, tok in enumerate(req.prompt[req.n_prefilled : end]):
                # Only the final prompt token needs logits; intermediate chunks are
                # pure KV fill, which is what makes chunked prefill cheap.
                want = is_last_chunk and j == chunk.n_tokens - 1
                row = self._batch.add(tok, req.n_pos + j, req.seq_id, want)
                if want:
                    rows.append((req, row))
            commits.append((req, end, req.n_pos + chunk.n_tokens))

        for request_id in plan.decode:
            req = self._states[request_id]
            assert req.next_token is not None
            rows.append((req, self._batch.add(req.next_token, req.n_pos, req.seq_id, True)))
            commits.append((req, req.n_prefilled, req.n_pos + 1))

        return rows, commits

    def _advance(self, req: RequestState, row: int) -> StepOutput:
        # Per-sequence chain: this request's params and accepted-token history only.
        token = self.ctx.sample_seq(req.seq_id, row)
        if req.params.stop_at_eog and self.model.is_eog(token):
            return self._retire(req, "eog", token=token)

        req.output_tokens.append(token)
        req.n_generated += 1
        req.next_token = token
        piece = self.model.token_to_piece(token)

        if req.n_generated >= req.params.max_tokens:
            return self._retire(req, "length", token=token, piece=piece)
        if req.n_pos >= self.ctx.n_ctx_seq:
            return self._retire(req, "context", token=token, piece=piece)
        return StepOutput(req.request_id, token, piece, False, None)

    def _retire(self, req: RequestState, reason: str, token: int = -1, piece: str = "") -> StepOutput:
        req.finished = True
        req.finish_reason = reason
        req.next_token = None
        # Free the slot before the sampler, so an exception cannot leak the seq_id.
        self.ctx.memory_seq_rm(req.seq_id, -1, -1)
        if req.seq_id not in self._free_seq_ids:
            self._free_seq_ids.append(req.seq_id)
        self.ctx.reset_seq_sampler(req.seq_id)
        return StepOutput(req.request_id, token, piece, True, reason)

    # --- inspection -------------------------------------------------------------

    @property
    def has_work(self) -> bool:
        return any(not r.finished for r in self._states.values())

    @property
    def n_free_seq_slots(self) -> int:
        return len(self._free_seq_ids)

    def state(self, request_id: int) -> RequestState:
        return self._states[request_id]

    def tokens_of(self, request_id: int) -> list[int]:
        return list(self._states[request_id].output_tokens)

    def text_of(self, request_id: int) -> str:
        return self.model.detokenize(self._states[request_id].output_tokens)

    def drain(self, max_steps: int = 100_000) -> Iterable[StepOutput]:
        """Step until nothing is in flight, yielding every token produced."""
        for _ in range(max_steps):
            if not self.has_work:
                return
            yield from self.step()
