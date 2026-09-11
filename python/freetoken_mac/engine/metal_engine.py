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
from .ngram import NgramTable


@dataclass(frozen=True)
class StepOutput:
    """One token produced for one request by one step."""

    request_id: int
    token: int
    piece: str
    finished: bool
    finish_reason: str | None = None


@dataclass
class _VerifyEntry:
    """One generating request's speculation work for this step: its base row,
    its draft rows, the drafted tokens, and the KV position the base was
    packed at (drafts sit at the successive positions)."""

    req: RequestState
    base_row: int
    draft_rows: list[int]
    drafts: list[int]
    pos_base: int


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


# Architectures whose memory is hybrid attention + recurrent state (mirrors
# llama.cpp's llm_arch_supports_rs_rollback list). A speculation mismatch
# rewinds a KV suffix, which on these models needs recurrent-state rollback
# snapshots (llama n_rs_seq). This list DETECTS known hybrids; the n_rs_seq
# readback CONFIRMS support (llama.cpp clamps unsupported archs to 0) -- so a
# future hybrid with rollback support passes without a code change, and only a
# hybrid with provably no rollback is refused.
_HYBRID_RECURRENT_ARCHES = frozenset(
    {
        "qwen35",
        "qwen35moe",
        "qwen4exp",
        "deepseek4",
        "nemotron_h",
        "nemotron_h_moe",
        "lfm2",
        "lfm2moe",
        "bailingmoe3",
    }
)


def check_speculative_arch(arch: str | None, n_rs_seq: int) -> None:
    """Refuse speculation on hybrids WITHOUT rollback support (see above).

    Unknown or missing arch strings pass, as do hybrids whose context reports
    snapshots in effect: only a KNOWN hybrid with provably no rollback is
    rejected, so neither future attention nor future rollback-capable hybrid
    architectures are blocked by this list.
    """
    if arch in _HYBRID_RECURRENT_ARCHES and n_rs_seq <= 0:
        raise ValueError(
            f"speculative decoding is not supported on hybrid architecture "
            f"{arch!r}: mismatch rewinds need recurrent-state rollback "
            f"snapshots (llama n_rs_seq), and this context has none in effect "
            f"(SPEC-speculative-ngram.md T5a)"
        )


def check_prefix_cache_arch(arch: str | None) -> None:
    """Refuse the prefix cache on hybrid/recurrent architectures (see above).

    Measured 2026-09-10: forked generations diverge NONDETERMINISTICALLY on
    4B and 27B qwen35 (fork1 != fork2 != control on identical inputs), while
    attention forks are exactly deterministic (triple-fork proof in tests).
    Forking copies hybrid memory cells and truncates the suffix -- one of
    those two ops is unfaithful on this backend (upstream #20075 class; our
    pin predates the fix). Unknown archs pass (fail-open for future attention
    designs); a hybrid with a proven-clean fork can be allow-listed with
    evidence, not assumptions.
    """
    if arch in _HYBRID_RECURRENT_ARCHES:
        raise ValueError(
            f"prefix caching is disabled on hybrid architecture {arch!r}: "
            f"fork copies diverge nondeterministically on this backend "
            f"(SPEC-prefix-cache.md T8b evidence; upstream llama.cpp #20075 "
            f"class). Re-verify after a pin bump."
        )


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
        if self.config.spec_max_drafts < 0:
            raise ValueError(
                f"spec_max_drafts must be >= 0; got {self.config.spec_max_drafts}"
            )
        if self.config.prefix_cache_pins < 0:
            raise ValueError(
                f"prefix_cache_pins must be >= 0; got {self.config.prefix_cache_pins}"
            )
        if self.config.prefix_cache_min_tokens < 1:
            raise ValueError(
                f"prefix_cache_min_tokens must be >= 1; got "
                f"{self.config.prefix_cache_min_tokens}"
            )
        self.policy: AdmissionPolicy = policy or FCFSPolicy()
        self.default_params = default_params or RequestParams()
        context_params = self.config.to_context_params()
        if self.config.speculative:
            # Mismatch rewinds span at most spec_max_drafts suffix cells; the
            # recurrent rollback needs at least that many snapshots (T5a).
            context_params.n_rs_seq = self.config.spec_max_drafts + 1
        self.ctx = Context(model, context_params)
        arch = model.meta_val("general.architecture")
        if self.config.speculative:
            # After allocating the context: fail loud on hybrids whose context
            # reports no rollback snapshots (llama.cpp clamps unsupported
            # archs to 0), not mid-generation on the first rewind.
            check_speculative_arch(arch, self.ctx.n_rs_seq)
        if self.config.prefix_cache:
            # Fail loud on hybrids: fork copies + truncates hybrid memory,
            # which diverges nondeterministically on this backend (measured
            # on 4B/27B qwen35; upstream #20075 class). Attention KV forks
            # exactly (proven by triple-fork determinism tests).
            check_prefix_cache_arch(arch)
        # Pinning also stays off unless pins are configured: with zero pins
        # the store could never hold anything and lookups would only waste
        # scans. Attention always qualifies; hybrids never reach here.
        self._prefix_cache_active = bool(
            self.config.prefix_cache and self.config.prefix_cache_pins > 0
        )

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
        # N-gram draft tables, one per request. Same ownership story as the
        # stop filters: engine-owned, invisible to policies, dropped on retire
        # (and therefore on ANY `_advance`/`_verify` failure). Only populated
        # when speculation is enabled, so a default engine allocates nothing
        # extra per request.
        self._spec_tables: dict[int, NgramTable] = {}
        # Lifetime counters behind `spec_acceptance_rate` (T4 tunes on these).
        self.spec_drafted = 0
        self.spec_accepted = 0
        # Pinned prefix slots (SPEC-prefix-cache.md): token-tuple -> seq_id in
        # insertion (LRU) order. Pins hold seq_ids OUTSIDE the free pool, so
        # every admission and retire accounts for both pools together; the
        # invariant is len(pins) + len(free) + in-flight == n_seq_max. Only
        # populated when prefix_cache is on.
        self._pins: dict[tuple[int, ...], int] = {}
        self.prefix_cache_hits = 0
        self.prefix_cache_tokens_saved = 0
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
        # Longest qualifying pin BEFORE touching slots: eviction below must
        # spare the fork source (or drop the fork if its slot is needed).
        match = self._find_pin(tokens)
        if not self._free_seq_ids:
            self._evict_pin(exclude=match[0] if match is not None else None)
        if not self._free_seq_ids:
            if match is not None:
                # The only pin is the match itself: free its slot and fall
                # back to a plain prefill rather than strand the request.
                self._evict_pin(exclude=None)
                match = None
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
        if match is not None and match[0] in self._pins:
            # Fork the pinned prefix (best-effort: falls back to plain
            # prefill, which is exactly the no-cache path, so correctness
            # never depends on the fork succeeding).
            try:
                self._attempt_fork(request_id, seq_id, match)
            except BaseException:  # noqa: BLE001 - re-raised; slot returns via cancel
                self.cancel(request_id)
                raise
        if rp.stop:
            self._stop_filters[request_id] = StopSequenceFilter(rp.stop)
        if self.config.speculative:
            # Seed with the prompt so the first generation steps can already
            # draft runs the prompt itself contains (repeated instructions,
            # few-shot examples, boilerplate). Past the try/except above, so a
            # table here cannot strand a seq_id: `_retire` drops it.
            table = NgramTable()
            table.update_stream(tokens)
            self._spec_tables[request_id] = table
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

        # Drafts ride spare batch rows only: the policy plans prompt chunks plus
        # one continuation token per sequence exactly as before, and speculation
        # fills whatever rows that leaves empty. A disabled engine (or a step
        # with no spare rows) plans zero drafts, i.e. today's batch exactly.
        drafts = self._plan_drafts(plan, budget.n_batch - plan.n_tokens)
        entries, commits = self._fill_batch(plan, drafts)
        # THE decode. Position/prefill bookkeeping is committed only after it returns,
        # so a rejected batch leaves every request exactly where it was.
        self.ctx.decode(self._batch)
        for req, n_prefilled, n_pos in commits:
            req.n_prefilled = n_prefilled
            req.n_pos = n_pos

        outputs: list[StepOutput] = []
        for entry in entries:
            if isinstance(entry, _VerifyEntry):
                outputs.extend(self._verify(entry))
            else:
                req, row = entry
                outputs.append(self._advance(req, row))
        return outputs

    def _fill_batch(
        self, plan: StepPlan, drafts: dict[int, list[int]] | None = None
    ) -> tuple[
        list[tuple[RequestState, int] | _VerifyEntry],
        list[tuple[RequestState, int, int]],
    ]:
        self._batch.clear()
        entries: list[tuple[RequestState, int] | _VerifyEntry] = []
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
                    entries.append((req, row))
            commits.append((req, end, req.n_pos + chunk.n_tokens))

        for request_id in plan.decode:
            req = self._states[request_id]
            assert req.next_token is not None
            draft_tokens = (drafts or {}).get(request_id, [])
            if not draft_tokens:
                entries.append(
                    (req, self._batch.add(req.next_token, req.n_pos, req.seq_id, True))
                )
                commits.append((req, req.n_prefilled, req.n_pos + 1))
                continue
            # Base plus drafts at successive positions, logits on every row:
            # row 0 continues the base token, row k > 0 continues draft k-1.
            # The commit runs past every packed row; `_verify` rewinds it (and
            # the KV) to the first mismatch.
            base_row = self._batch.add(req.next_token, req.n_pos, req.seq_id, True)
            draft_rows = [
                self._batch.add(tok, req.n_pos + 1 + k, req.seq_id, True)
                for k, tok in enumerate(draft_tokens)
            ]
            entries.append(
                _VerifyEntry(req, base_row, draft_rows, draft_tokens, req.n_pos)
            )
            commits.append((req, req.n_prefilled, req.n_pos + 1 + len(draft_tokens)))

        return entries, commits

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
        return self._advance_token(req, self.ctx.sample_seq(req.seq_id, row))

    def _advance_token(self, req: RequestState, token: int) -> StepOutput:
        """Account for one already-sampled (or verified) token.

        Split from ``_advance_row`` for speculative decoding: the verify step
        samples draft rows itself and feeds each accepted token here, so every
        finish rule below (EOG, stop sequence, cap, context) applies to
        speculated tokens exactly as it does to normally decoded ones. Callers
        feeding a token from any source other than this request's own sampler
        chain must have replayed the chain to match (see ``accept_seq``).
        """
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

    def _plan_drafts(self, plan: StepPlan, spare: int) -> dict[int, list[int]]:
        """Draft continuations for greedy generating requests from spare rows.

        Read-only with respect to the plan: the policy's tokens are untouched
        and drafts only fill rows the plan left empty, so prefill keeps its
        priority. A request drafts only while it is greedy (verification
        without logits cannot do better than greedy matching), its table
        predicts something, and its own sequence has room for the extra
        positions.
        """
        drafts: dict[int, list[int]] = {}
        if not self.config.speculative or spare <= 0:
            return drafts
        for request_id in plan.decode:
            if spare <= 0:
                break
            req = self._states.get(request_id)
            if req is None or req.finished or req.params.temp > 0:
                continue
            table = self._spec_tables.get(request_id)
            if table is None:
                continue
            room = self.ctx.n_ctx_seq - req.n_pos - 1
            allow = min(self.config.spec_max_drafts, spare, room)
            if allow <= 0:
                continue
            history = req.prompt + req.output_tokens
            context = history[-(table.order - 1) :] if table.order > 1 else []
            predicted = table.predict(context, allow)
            if predicted:
                drafts[request_id] = predicted
                spare -= len(predicted)
        return drafts

    def _verify(self, entry: _VerifyEntry) -> list[StepOutput]:
        """Verify one request's drafts; same retire-on-failure contract as
        ``_advance`` (a request that fails mid-verify cannot resume: its KV
        positions are committed and its sampler chain may hold a sampled token
        the caller never saw)."""
        try:
            return self._verify_rows(entry)
        except BaseException:  # noqa: BLE001 - re-raised; must retire slot on any failure
            try:
                self._retire(entry.req, "error")
            except BaseException:  # noqa: BLE001 - must never mask the real failure
                pass
            raise

    def _verify_rows(self, entry: _VerifyEntry) -> list[StepOutput]:
        req, drafts = entry.req, entry.drafts
        n_drafts = len(drafts)
        # Sample every row in order: row 0 continues the base token, row k > 0
        # continues draft k-1. Sampling accepts into the chain, so a matched
        # prefix leaves the chain exactly where the accepted output says it is
        # -- no replay needed. Rows past the first mismatch are never sampled:
        # they were conditioned on a rejected prefix, and sampling them would
        # pollute the chain with tokens the request must not keep.
        accepted: list[int] = []
        matched = 0
        for k in range(n_drafts + 1):
            row = entry.base_row if k == 0 else entry.draft_rows[k - 1]
            token = self.ctx.sample_seq(req.seq_id, row)
            accepted.append(token)
            if k < n_drafts:
                if token != drafts[k]:
                    break
                matched += 1
            # k == n_drafts is the bonus row past the last draft: conditioned
            # on a fully matched prefix, so always valid output.
        self.spec_drafted += n_drafts
        self.spec_accepted += matched
        # The commit ran n_pos past every packed row; the accepted tokens only
        # fill positions pos_base+1 .. pos_base+len(accepted) (the base row
        # sits at pos_base, already covered by the pre-step n_pos). Rewind the
        # KV from the last accepted position on any mismatch: that cell holds
        # the rejected draft, and the following step re-decodes the accepted
        # token there (overwriting it). The stale cell must go regardless --
        # llama requires each decode to start strictly consecutive with the KV
        # (Y = X + 1), so it aborts the next decode otherwise. A full match
        # needs no rewind: the KV holds exactly the accepted tokens.
        final_n_pos = entry.pos_base + len(accepted)
        if len(accepted) < n_drafts + 1:
            if not self.ctx.memory_seq_rm(req.seq_id, final_n_pos, -1):
                # The packed rows past final_n_pos are stale, and so is every
                # position check from here on: retire LOUD via _verify's
                # contract rather than desync into the next decode's abort.
                raise RuntimeError(
                    f"speculative rewind failed for request {req.request_id}: "
                    f"partial KV removal of [{final_n_pos}, inf) unsupported "
                    f"(hybrid without rollback snapshots?)"
                )
        # Every accepted token goes through the identical finish logic as a
        # normally decoded one; n_pos advances per token so the context-full
        # check sees the same values as the plain path. Stop feeding at the
        # first token that ends the request (the slot is freed, so later
        # tokens have nowhere to go).
        self._feed_table(req, accepted)
        req.n_pos = entry.pos_base + 1
        outputs: list[StepOutput] = []
        for token in accepted:
            req.n_pos += 1
            out = self._advance_token(req, token)
            outputs.append(out)
            if out.finished:
                break
        # The per-token increments above leave n_pos one past the last accepted
        # position (each check reads position+1, as on the plain path); settle
        # it on the next write position, which the following step decodes at.
        req.n_pos = final_n_pos
        return outputs

    def _feed_table(self, req: RequestState, new_tokens: Sequence[int]) -> None:
        """Record the pairs ending in freshly accepted tokens.

        Only the window that can contain new pairs is fed -- the trailing
        order-1 history plus the new tokens. Re-feeding older pairs would cost
        O(history) per step, i.e. throughput decaying with context length.
        """
        table = self._spec_tables.get(req.request_id)
        if table is None or not new_tokens:
            return
        prefix = (
            (req.prompt + req.output_tokens)[-(table.order - 1) :]
            if table.order > 1
            else []
        )
        table.update_stream([*prefix, *new_tokens])

    def _retire(self, req: RequestState, reason: str, token: int = -1, piece: str = "") -> StepOutput:
        req.finished = True
        req.finish_reason = reason
        req.next_token = None
        self._stop_filters.pop(req.request_id, None)
        self._spec_tables.pop(req.request_id, None)
        # Off the hot path and into the read window, before anything that can raise: a
        # failed KV or sampler reset must not leave a finished request being scanned by
        # every subsequent decode (or, worse, keeping `has_work` true forever).
        self._states.pop(req.request_id, None)
        self._archive(req)
        # A fully-prefilled long prompt converts its slot into a pin instead
        # of freeing it (SPEC-prefix-cache.md); anything else frees as before.
        # Free the slot before the sampler, so an exception cannot leak the seq_id.
        if not self._maybe_pin(req):
            self.ctx.memory_seq_rm(req.seq_id, -1, -1)
            if req.seq_id not in self._free_seq_ids:
                self._free_seq_ids.append(req.seq_id)
        self.ctx.reset_seq_sampler(req.seq_id)
        return StepOutput(req.request_id, token, piece, True, reason)

    def _attempt_fork(
        self,
        request_id: int,
        seq_id: int,
        match: tuple[tuple[int, ...], int, int],
    ) -> bool:
        """Copy a pin then truncate one short of the shared run (True).

        Full-copy is always legal, even on split KV; the truncate re-decodes
        its last covered token through the normal chunked-prefill path so a
        logits row exists for sampling the continuation (re-decoding a
        resident cell would abort, hence the backoff of exactly one token).
        Best-effort: a failed truncate (hybrid suffix past the snapshot
        range) falls back to a full clear + plain prefill -- the no-cache
        path -- so False preserves correctness and only costs the copy.
        """
        key, pin_seq, shared = match
        try:
            self.ctx.memory_seq_cp(pin_seq, seq_id, 0, -1)
            rewound = self.ctx.memory_seq_rm(seq_id, shared - 1, -1)
        except Exception:  # noqa: BLE001 - best-effort fork; plain fallback below
            rewound = False
        if not rewound:
            self.ctx.memory_seq_rm(seq_id, -1, -1)
            return False
        req_state = self._states[request_id]
        req_state.n_prefilled = shared - 1
        req_state.n_pos = shared - 1
        self._pins[key] = self._pins.pop(key)  # refresh recency
        self.prefix_cache_hits += 1
        self.prefix_cache_tokens_saved += shared - 1
        return True

    def _find_pin(
        self, tokens: Sequence[int]
    ) -> tuple[tuple[int, ...], int, int] | None:
        """Longest pinned prefix shared with ``tokens``: (key, seq_id, K).

        Forks only when the backoff-adjusted savings (K - 1) clear
        min_tokens; otherwise the copy+truncate costs more machinery than the
        sub-chunk prefill it would save. Pure lookup -- recency refresh and
        counters happen at the fork site, not here. Inactive (disabled or no
        usable rewind) means no pins exist, so always None then.
        """
        if not self._prefix_cache_active or not self._pins:
            return None
        best: tuple[tuple[int, ...], int, int] | None = None
        for key, seq_id in self._pins.items():
            shared = 0
            for a, b in zip(key, tokens):
                if a != b:
                    break
                shared += 1
            if shared - 1 >= self.config.prefix_cache_min_tokens and (
                best is None or shared > best[2]
            ):
                best = (key, seq_id, shared)
        return best

    def _evict_pin(self, exclude: tuple[int, ...] | None) -> None:
        """Free the least-recently-used pin that is not ``exclude``."""
        for key in list(self._pins):
            if key != exclude:
                seq_id = self._pins.pop(key)
                self.ctx.memory_seq_rm(seq_id, -1, -1)
                if seq_id not in self._free_seq_ids:
                    self._free_seq_ids.append(seq_id)
                return

    def _maybe_pin(self, req: RequestState) -> bool:
        """Convert a retired slot into a pin when worthwhile (see SPEC).

        Only fully-prefilled prompts qualify: a mid-prefill cancel leaves
        fewer cells than the prompt claims, and forking from those would
        desync every later position. Exact-match dedupe refreshes recency
        without duplicating the slot; over capacity evicts the oldest.
        Inactive means pins could never fork -- don't waste slots on them.
        """
        if not self._prefix_cache_active:
            return False
        if req.n_prefilled < len(req.prompt):
            return False
        if len(req.prompt) - 1 < self.config.prefix_cache_min_tokens:
            return False
        key = tuple(req.prompt)
        if key in self._pins:
            self._pins[key] = self._pins.pop(key)
            return False
        self._pins[key] = req.seq_id
        while len(self._pins) > self.config.prefix_cache_pins:
            self._evict_pin(exclude=key)
        return True

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

    @property
    def spec_acceptance_rate(self) -> float | None:
        """Fraction of drafted tokens the target confirmed, or None before any
        draft has been verified."""
        if self.spec_drafted == 0:
            return None
        return self.spec_accepted / self.spec_drafted

    def drain(self, max_steps: int = 100_000) -> Iterable[StepOutput]:
        """Step until nothing is in flight, yielding every token produced."""
        for _ in range(max_steps):
            if not self.has_work:
                return
            yield from self.step()
