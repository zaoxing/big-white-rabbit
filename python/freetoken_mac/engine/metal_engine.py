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
from .config import EngineConfig, RequestParams, StopSequenceFilter


@dataclass(frozen=True)
class StepOutput:
    """One token produced for one request by one step."""

    request_id: int
    token: int
    piece: str
    finished: bool
    finish_reason: str | None = None


# Floor on how many finished requests stay readable through `state` / `tokens_of` /
# `text_of` after they retire. The window is `max(this, ctx.n_seq_max)`: it must cover at
# least a full batch's worth of retirements, because every request in flight together is
# read after IT ends while its peers are still running, and it must not be 1 or 2 just
# because someone configured a narrow context. See MetalEngine._archive.
MIN_RETAINED_FINISHED = 8


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
        # THE HOT PATH. In-flight requests ONLY: `step` scans this twice per decode
        # (is_prefilling / is_generating) and `has_work` once more, so its size is what
        # sets the per-token bookkeeping cost. Holding finished requests here made that
        # cost grow with the total number of requests ever served -- throughput decaying
        # with uptime -- so `_retire` moves the entry OUT, into `_retired`. Bounded by
        # n_seq_max, hence by concurrency, never by history.
        self._states: dict[int, RequestState] = {}
        # The read window: recently retired requests, newest last, capped at
        # `_retired_limit`. The engine's read accessors are called AFTER a request
        # finishes (the routes, and tests, ask for `tokens_of` / `finish_reason` on a
        # request that has just ended), so retiring cannot mean forgetting; but retaining
        # forever is the leak -- each entry pins a whole prompt token list. A bounded
        # ring is the one policy that serves both, and it covers every caller, including
        # a direct `drain()` user who never goes near AsyncEngine.release.
        # Insertion-ordered dict, so evicting the oldest is `next(iter(...))`.
        self._retired: dict[int, RequestState] = {}
        self._retired_limit = max(self.ctx.n_seq_max, MIN_RETAINED_FINISHED)
        # Stop-sequence scanners, one per request that asked for one. Kept beside the
        # states rather than on RequestState because a scanner is engine-owned mutable
        # state that no AdmissionPolicy has any business reading, and because this way a
        # request without stop sequences allocates nothing at all. `_retire` drops the
        # entry -- and `_advance` retires on ANY failure -- so `set(_stop_filters)` is
        # always a subset of `set(_states)`: a scanner cannot outlive its request.
        self._stop_filters: dict[int, StopSequenceFilter] = {}
        self._next_request_id = 0

    # --- admission --------------------------------------------------------------

    def add_request(
        self,
        prompt: str | Sequence[int],
        params: RequestParams | None = None,
        *,
        add_special: bool = True,
    ) -> int:
        """Admit a request and return its id. Raises before touching llama.cpp on a
        non-positive max_tokens, an empty prompt, a prompt that cannot fit the context,
        or seq_id exhaustion."""
        # First, because it is the only check that needs neither the tokeniser nor the
        # context: `_advance` retires on `n_generated >= params.max_tokens` and samples
        # before it checks, so a non-positive cap does not mean "generate nothing" -- it
        # means "generate exactly one token and call it `length`", a plausible-looking
        # result for an impossible request. The protocol surfaces reject it first
        # (server/schemas.max_tokens_error); this is what makes the refusal hold for a
        # direct engine caller and for any route added later.
        rp = params or self.default_params
        if rp.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be >= 1; got {rp.max_tokens} (no generation "
                f"satisfies a non-positive cap)"
            )
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

        # From here to the _states registration below, the slot is owned by nobody:
        # _retire is the ONLY thing that returns a seq_id to the pool and it needs a
        # RequestState to do it, so anything that raises in this window would lose the
        # slot permanently -- one leaked slot per failed admission, n_seq_max of them and
        # the engine refuses every further request. Rolling the pop back on ANY exception
        # keeps that true for lines added here later, not just today's (a params object
        # whose sampler values do not fit the C++ SamplerParams raises from
        # to_sampler_params).
        try:
            # A retired predecessor may have left KV behind on this slot.
            self.ctx.memory_seq_rm(seq_id, -1, -1)
            self.ctx.set_seq_sampler(seq_id, rp.to_sampler_params())
        except BaseException:  # noqa: BLE001 - re-raised; this only undoes the pop
            # Back to the front, so a rejected request leaves the pool as it found it.
            # Pure list surgery: nothing here can raise and strand the slot again. Any
            # KV or sampler left on the slot is harmless -- the next admission clears the
            # KV and overwrites the sampler before use.
            self._free_seq_ids.insert(0, seq_id)
            raise

        self._states[request_id] = RequestState(
            request_id=request_id, seq_id=seq_id, prompt=tokens, params=rp
        )
        if rp.stop:
            self._stop_filters[request_id] = StopSequenceFilter(rp.stop)
        return request_id

    def cancel(self, request_id: int) -> bool:
        """Drop a request mid-flight. Its KV and sampler go away and its seq_id returns
        to the pool; peers in the same batch are untouched. False if already finished."""
        req = self._lookup(request_id)
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
        """Sample one row and account for it, retiring the request if anything fails.

        The failure path exists because there is no way to resume a request whose
        sampling raised: the token may or may not have been accepted into its sampler
        chain, and `step` has already committed the KV positions. Leaving it in flight
        was the worse answer on every count -- its seq_id was never returned, its
        `StopSequenceFilter` stayed in `_stop_filters`, and `has_work` stayed true, so
        the worker thread re-ran the same failing step forever (the pinned-core mode
        097b84f fixed for abandoned streams). Retiring frees the slot, drops both maps
        and leaves the request readable with `finish_reason == "error"`. The exception is
        re-raised, so nothing is swallowed and the reason never reaches the wire: `step`
        builds its result list from these calls, so a raising `_advance` returns no
        StepOutput at all.
        """
        try:
            return self._advance_row(req, row)
        except BaseException:  # noqa: BLE001 - re-raised; must retire slot on any failure
            try:
                self._retire(req, "error")
            except BaseException:  # noqa: BLE001 - must never mask the real failure
                # A cleanup that itself failed costs one seq_id; reporting the wrong
                # exception would cost the diagnosis of every one of these.
                pass
            raise

    def _advance_row(self, req: RequestState, row: int) -> StepOutput:
        # Per-sequence chain: this request's params and accepted-token history only.
        token = self.ctx.sample_seq(req.seq_id, row)
        if req.params.stop_at_eog and self.model.is_eog(token):
            return self._retire(req, "eog", token=token)

        req.output_tokens.append(token)
        req.n_generated += 1
        req.next_token = token
        piece = self.model.token_to_piece(token)

        stop_filter = self._stop_filters.get(req.request_id)
        if stop_filter is not None:
            # Detection is on the accumulated piece text, not on this piece: the
            # tokeniser does not align to the sequence (see config.StopSequenceFilter).
            # The piece is reported UNTRUNCATED -- StepOutput.piece stays "the text of
            # the token that was decoded", which is what keeps it consistent with
            # `tokens_of`/`text_of` and the KV -- and the caller runs the same filter to
            # decide what goes on the wire. What this check buys is the part only the
            # engine can do: no further token is decoded, so a request whose delimiter
            # has arrived stops costing decodes.
            _emittable, hit = stop_filter.push(piece)
            if hit:
                return self._retire(req, "stop_sequence", token=token, piece=piece)

        # Checked after the stop sequence on purpose: if one token both completes a
        # delimiter and hits the cap, the turn ended because the delimiter arrived --
        # the text is truncated at it either way, and "length" would tell the client to
        # continue a turn that is already complete.
        if req.n_generated >= req.params.max_tokens:
            return self._retire(req, "length", token=token, piece=piece)
        if req.n_pos >= self.ctx.n_ctx_seq:
            return self._retire(req, "context", token=token, piece=piece)
        return StepOutput(req.request_id, token, piece, False, None)

    def _retire(self, req: RequestState, reason: str, token: int = -1, piece: str = "") -> StepOutput:
        req.finished = True
        req.finish_reason = reason
        req.next_token = None
        self._stop_filters.pop(req.request_id, None)
        # Off the hot path and into the read window, before anything that can raise: a
        # failed KV or sampler reset must not leave a finished request being scanned by
        # every subsequent decode (or, worse, keeping `has_work` true forever).
        self._states.pop(req.request_id, None)
        self._archive(req)
        # Free the slot before the sampler, so an exception cannot leak the seq_id.
        self.ctx.memory_seq_rm(req.seq_id, -1, -1)
        if req.seq_id not in self._free_seq_ids:
            self._free_seq_ids.append(req.seq_id)
        self.ctx.reset_seq_sampler(req.seq_id)
        return StepOutput(req.request_id, token, piece, True, reason)

    def _archive(self, req: RequestState) -> None:
        """Put a finished request in the read window, evicting the oldest if it is full.

        Pure dict surgery on purpose -- `_retire` calls it before the context calls that
        can raise, so nothing here may raise either.
        """
        self._retired[req.request_id] = req
        while len(self._retired) > self._retired_limit:
            # Insertion order == retirement order, so this is the least recently
            # finished request, i.e. the one a caller is least likely to still want.
            del self._retired[next(iter(self._retired))]

    # --- inspection -------------------------------------------------------------

    @property
    def has_work(self) -> bool:
        # `_states` holds only in-flight requests, so this is bounded by n_seq_max. The
        # `finished` test is kept as a belt-and-braces guard: were a retired state ever
        # left here, "has work" would spin the worker thread on a request nobody can
        # advance, which is a pinned core rather than a wrong answer.
        return any(not r.finished for r in self._states.values())

    @property
    def n_free_seq_slots(self) -> int:
        return len(self._free_seq_ids)

    @property
    def n_in_flight(self) -> int:
        """Requests on the hot path, i.e. how much `step` scans per decode."""
        return len(self._states)

    @property
    def n_retained_finished(self) -> int:
        """Finished requests still readable through the accessors below."""
        return len(self._retired)

    @property
    def retained_finished_limit(self) -> int:
        """Cap on `n_retained_finished`: the width of the post-completion read window."""
        return self._retired_limit

    def _lookup(self, request_id: int) -> RequestState:
        """The state for a request, in flight or recently retired.

        Raises KeyError once the request has fallen out of the read window, and says
        which mistake it was: a pruned id and an id that never existed are both
        unanswerable, but only one of them means "you waited too long". Returning an
        empty result instead would be indistinguishable from a request that legitimately
        generated nothing, which is the failure mode a caller could not detect.
        """
        req = self._states.get(request_id)
        if req is None:
            req = self._retired.get(request_id)
        if req is None:
            if 0 <= request_id < self._next_request_id:
                raise KeyError(
                    f"request {request_id} finished and is no longer retained "
                    f"(only the last {self._retired_limit} finished requests are)"
                )
            raise KeyError(f"unknown request {request_id}")
        return req

    def state(self, request_id: int) -> RequestState:
        return self._lookup(request_id)

    def tokens_of(self, request_id: int) -> list[int]:
        return list(self._lookup(request_id).output_tokens)

    def text_of(self, request_id: int) -> str:
        return self.model.detokenize(self._lookup(request_id).output_tokens)

    def drain(self, max_steps: int = 100_000) -> Iterable[StepOutput]:
        """Step until nothing is in flight, yielding every token produced."""
        for _ in range(max_steps):
            if not self.has_work:
                return
            yield from self.step()
