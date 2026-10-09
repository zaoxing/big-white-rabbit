"""Context benchmark: the largest prompt this machine can actually prefill.

Why measure instead of predict
==============================

`host.adapt_ctx` already answers a version of this question analytically --
weights + an OS reserve + `kv_bytes_per_token * ctx` against installed RAM --
and that estimate is what a plain `bwr serve` uses to pick `n_ctx`. It is a
budget, not an observation: it does not know the Metal heap's fragmentation,
what else on the machine holds memory right now, or that a hybrid model's KV
cost per token is not the uniform figure the formula assumes.

So the prediction seeds the search and a real prefill decides it. The number
this reports is one the machine demonstrably reached, which is the only kind
that is safe to write into a context-window setting.

What is measured, exactly
=========================

The TOTAL context (`n_ctx`) -- the same quantity `--ctx-size` sets and the
override file stores. Every probe engine is built with `n_seq_max = 1`, so
the whole context belongs to the one prompt under test and "can the engine
admit N tokens" and "can the machine hold a KV of N cells" are the same
question. Under the serving geometry a single request's share is
`n_ctx / n_seq_max` (`n_ctx_seq`, which /admin/api/models reports); the
applied value does not pretend otherwise, it is the window the machine held.

Both prefix caches are off on a probe engine too: every probe submits the
same filler text, and a cache hit would skip the very prefill being timed.

How a candidate is probed, and why it differs by backend
========================================================

MLX grows its KV cache as tokens arrive; `n_ctx` there is purely an admission
cap, so one engine loaded at the ceiling can be probed at every candidate
size by prefilling different amounts. llama.cpp allocates the whole KV cache
when the context is created, so for a Metal model the allocation under test
happens at LOAD time and each candidate needs its own load.

One bisection drives both. `_prefill` is the only thing that differs:

    mlx    load once at the ceiling, prefill N tokens
    metal  load at n_ctx = N, then prefill N tokens

The Metal path is why the client warns about runtime -- a load per probe on a
27B is minutes, not seconds.

A probe never touches the persisted override. It asks the pool for a load at
an explicit geometry (`ModelPool.reload(config=...)`), so the file beside the
weights only ever holds a value a completed run chose to apply -- a crash or
a kill mid-search cannot leave a trial size behind as if it were a setting.

While a run is going the pool is in maintenance: `acquire` answers busy, so
no request is served on an engine sized for a probe and no concurrent load
perturbs the memory being measured.

What "failure" means
====================

A probe fails when the engine refuses the prompt or the prefill raises. Both
are counted the same way, deliberately: an admission refusal and an
out-of-memory both mean "this machine will not serve a prompt that size",
which is the question being asked. The distinction is preserved in
`limit_reason` on the result and in the progress message, not in the
boundary.

`asyncio.CancelledError` is NEVER treated as a failed probe. Swallowing it
would turn a cancel into a measurement and write a fabricated boundary into
the model's settings.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..engine.pool import ModelPoolError
from .bench import _prompt_of_length

logger = logging.getLogger(__name__)

#: Targets the client offers, and the only values `start` accepts. A free
#: integer would let a caller ask for a size the UI cannot render or select
#: again; the whitelist keeps the two surfaces describing the same ladder.
TARGETS: tuple[int, ...] = (16384, 32768, 65536, 131072, 262144, 524288)

#: Applied values are floored to this. A context window is a setting a human
#: reads and a model's KV is sized from, and the last 900 tokens of a 41,100
#: measurement are noise -- the machine's free memory moved while we measured
#: it. Flooring also leaves headroom so the applied value is one the machine
#: reached with room to spare, not the exact edge it failed just above.
APPLY_GRANULARITY = 2048

#: The search stops refining once the bracket is this narrow. Below it, each
#: further probe costs a full prefill (or a full load) to move a number that
#: `APPLY_GRANULARITY` is about to round away.
SEARCH_GRANULARITY = 2048

#: Smallest context worth reporting. Under this, the honest answer is that
#: the model does not fit on this machine rather than a tiny window.
FLOOR_TOKENS = 2048


class ContextBenchError(RuntimeError):
    """A start that cannot be honoured: bad target, unknown model, or a run
    already going."""


@dataclass
class ContextBenchResult:
    """The measurement. Field names are the wire contract with the client.

    `ContextBenchResultDTO` declares model_id, target_tokens, measured_tokens,
    verified_tokens, applied_tokens, applied, capped_by, attempts and
    duration_s NON-optional, and Swift's Decodable fails the whole object on
    one missing key -- so every one of them is always present, even on a run
    that measured nothing.
    """

    model_id: str
    target_tokens: int
    native_context_length: int | None = None
    measured_tokens: int = 0
    verified_tokens: int = 0
    verified_prompt_tokens: int | None = None
    applied_tokens: int = 0
    applied: bool = False
    capped_by: str = "memory"
    # Why the last size above the boundary was refused, when memory capped
    # the answer: an admission message or the allocator's error. What lets a
    # reader tell "the engine said no" from "the machine ran out".
    limit_reason: str | None = None
    attempts: int = 0
    prefill_tps: float | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "target_tokens": self.target_tokens,
            "native_context_length": self.native_context_length,
            "measured_tokens": self.measured_tokens,
            "verified_tokens": self.verified_tokens,
            "verified_prompt_tokens": self.verified_prompt_tokens,
            "applied_tokens": self.applied_tokens,
            "applied": self.applied,
            "capped_by": self.capped_by,
            "limit_reason": self.limit_reason,
            "attempts": self.attempts,
            "prefill_tps": self.prefill_tps,
            "duration_s": self.duration_s,
        }


@dataclass
class ContextBenchRun:
    bench_id: str
    model_id: str
    target_tokens: int
    status: str = "running"
    phase: str = "preparing"
    progress: float = 0.0
    message: str = ""
    # Only ever a CONCLUSION: set when the run completes, never while it is
    # measuring and never after an error or a cancel. A half-filled result
    # has defaults that read as findings (`capped_by` "memory", measured 0),
    # and the clients copy whatever result they are handed.
    result: ContextBenchResult | None = None
    error: str | None = None
    started: float = field(default_factory=time.perf_counter)
    _cancel: asyncio.Event | None = None
    _task: asyncio.Task[None] | None = None
    # True once `_run` has taken its first step. A task cancelled BEFORE
    # that never executes a line of `_run` -- the CancelledError is thrown
    # into the unstarted coroutine -- so nothing would record the outcome.
    _started: bool = False

    def to_dict(self) -> dict[str, Any]:
        # phase/progress/message are non-optional on the client and are what
        # the progress card renders, so they are always strings/numbers --
        # never null, even before the first phase transition.
        return {
            "bench_id": self.bench_id,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "message": self.message,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
        }


class ContextBenchRunner:
    """Owns context-bench runs. One at a time, by design.

    Two concurrent runs would compete for the memory each is trying to
    measure, so each would report the other's pressure as its own ceiling.
    The same holds for ordinary requests, which is why a run puts the pool
    in maintenance for its duration.
    """

    def __init__(
        self,
        pool: Any,
        *,
        predictor: Callable[[str], int | None] | None = None,
    ) -> None:
        self._pool = pool
        self._runs: dict[str, ContextBenchRun] = {}
        self._active: str | None = None
        self._predictor = predictor or self._predict_ceiling

    # -- inspection --------------------------------------------------------

    def get(self, bench_id: str) -> dict[str, Any] | None:
        run = self._runs.get(bench_id)
        return run.to_dict() if run else None

    @property
    def active(self) -> str | None:
        return self._active

    def active_summary(self) -> dict[str, Any] | None:
        """The run in progress, in the shape a client re-attaches from, or
        None. What `/api/bench/context/active` reports."""
        run = self._runs.get(self._active) if self._active is not None else None
        if run is None or run.status != "running":
            return None
        return {
            "bench_id": run.bench_id,
            "model_id": run.model_id,
            "target_tokens": run.target_tokens,
            "status": run.status,
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self, model_id: str, target_tokens: int) -> dict[str, Any]:
        if target_tokens not in TARGETS:
            raise ContextBenchError(
                f"target_tokens must be one of {', '.join(str(t) for t in TARGETS)}"
            )
        if self._active is not None:
            running = self._runs.get(self._active)
            if running is not None and running.status == "running":
                raise ContextBenchError(
                    f"context benchmark {self._active} is still running; "
                    "cancel it first"
                )
        try:
            mid = self._pool.resolve(model_id)
        except ModelPoolError as exc:
            # A bad id is the caller's mistake, the same as a bad target: it
            # has to come back as one, not as a 500 with a traceback.
            raise ContextBenchError(str(exc)) from exc
        run = ContextBenchRun(
            bench_id=uuid.uuid4().hex[:12],
            model_id=mid,
            target_tokens=target_tokens,
        )
        run._cancel = asyncio.Event()
        self._runs[run.bench_id] = run
        self._active = run.bench_id
        run._task = asyncio.create_task(self._run(run))
        return {
            "bench_id": run.bench_id,
            "status": run.status,
            "target_tokens": run.target_tokens,
        }

    def cancel(self, bench_id: str) -> bool:
        """Stop a running benchmark.

        Both the flag and the task: the flag is what the search reads between
        probes, and cancelling the task is what interrupts a prefill that is
        under way. A LOAD under way cannot be interrupted (llama.cpp does not
        abort mid-allocation) and is shielded so it lands in the pool and is
        then unloaded by the run's cleanup; the status flips to "cancelled"
        at once, the slot frees when that load finishes.

        Only a task that has STARTED is cancelled. One that has not yet
        taken its first step would receive the CancelledError before any of
        `_run` executes and record nothing; the flag alone handles that
        case, since the first thing `_run` does after freeing the pool is
        read it.
        """
        run = self._runs.get(bench_id)
        if run is None or run.status != "running":
            return False
        if run._cancel is not None:
            run._cancel.set()
        if run._started and run._task is not None and not run._task.done():
            run._task.cancel()
        return True

    async def shutdown(self) -> None:
        """Cancel the active run and wait for its cleanup to finish.

        For lifespan teardown: a task left pending at loop close never runs
        its `finally`, so the pool would be left in maintenance with a probe
        engine resident. Waiting here is what guarantees the cleanup ran.
        """
        run = self._runs.get(self._active) if self._active is not None else None
        if run is None or run._task is None:
            return
        self.cancel(run.bench_id)
        # gather(return_exceptions=True): the task's own CancelledError is a
        # result to discard, not a cancellation of THIS coroutine.
        await asyncio.gather(run._task, return_exceptions=True)

    # -- worker ------------------------------------------------------------

    async def _run(self, run: ContextBenchRun) -> None:
        run._started = True
        try:
            self._pool.begin_maintenance(
                f"context benchmark {run.bench_id} is measuring {run.model_id}"
            )
        except Exception as exc:  # noqa: BLE001 - another maintainer holds the pool
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"
            run.message = run.error
            if self._active == run.bench_id:
                self._active = None
            return
        try:
            await self._measure(run)
        except asyncio.CancelledError:
            # Not re-raised: this task IS the unit of work, and its outcome
            # is "cancelled", recorded below. Re-raising would only make the
            # awaiter (shutdown, a test) see an exception for a run that
            # ended exactly as asked -- and the cleanup in `finally` needs to
            # await, which it can once the cancellation has been consumed.
            run.result = None
            run.status = "cancelled"
            run.message = "cancelled"
        except Exception as exc:  # noqa: BLE001 - a benchmark failure is a
            # result to report, not a reason to take the server down.
            run.result = None
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"
            run.message = run.error
        finally:
            try:
                # Whatever the outcome, the resident engine was built for a
                # probe (n_seq_max 1, caches off), not for serving. Drop it
                # so the next request loads at the configured geometry.
                await self._pool.unload_all(reason="context benchmark finished")
            except Exception:  # noqa: BLE001 - cleanup runs while an error is
                # already recorded; letting it raise here would replace the
                # real outcome with a cleanup failure.
                logger.warning("ctxbench: cleanup after %s failed", run.bench_id,
                               exc_info=True)
            # After the unload, so no request lands on a probe engine in the
            # gap between the gate lifting and the engine going.
            self._pool.end_maintenance()
            if self._active == run.bench_id:
                self._active = None

    async def _measure(self, run: ContextBenchRun) -> None:
        started = time.perf_counter()
        result = ContextBenchResult(
            model_id=run.model_id, target_tokens=run.target_tokens
        )

        # -- preparing: free the machine up, and find the ceiling ----------
        self._progress(run, "preparing", 2.0, "unloading resident models")
        await self._pool.unload_all(reason="context benchmark starting")
        if self._cancelled(run):
            return self._finish_cancelled(run)

        predicted = self._predictor(run.model_id)
        native = self._pool.native_ctx(run.model_id)
        if native is None:
            # A GGUF answers its trained length only once resident (only
            # llama.cpp parses its metadata), and the run has just unloaded
            # everything. Load at the starting point -- the first probe
            # would have anyway, so this costs nothing when it succeeds --
            # and ask again. A failed load is not an answer here; the search
            # discovers that on its own.
            ceiling, _capped = self._ceiling(run.target_tokens, None)
            hi_start = self._start_point(ceiling, predicted)
            self._progress(
                run, "preparing", 3.0,
                f"loading {run.model_id} to read its context length",
            )
            try:
                await self._engine_for(run.model_id, hi_start)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - see above
                pass
            if self._cancelled(run):
                return self._finish_cancelled(run)
            native = self._pool.native_ctx(run.model_id)
        result.native_context_length = native

        ceiling, capped_by = self._ceiling(run.target_tokens, native)
        hi_start = self._start_point(ceiling, predicted)

        if ceiling < FLOOR_TOKENS:
            result.capped_by = capped_by
            return self._complete(
                run, result, started,
                f"{run.model_id} tops out at {ceiling} tokens, below the "
                f"{FLOOR_TOKENS}-token floor",
            )

        # -- searching -----------------------------------------------------
        self._progress(
            run, "searching", 5.0,
            f"probing up to {ceiling:,} tokens"
            + (f" (estimate: {predicted:,})" if predicted else ""),
        )
        best, attempts, last_failure = await self._search(run, ceiling, hi_start)
        result.attempts = attempts
        if self._cancelled(run):
            return self._finish_cancelled(run)
        result.measured_tokens = best
        if best < ceiling:
            capped_by = "memory"
            result.limit_reason = last_failure
        result.capped_by = capped_by

        if best <= 0:
            return self._complete(
                run, result, started,
                f"no prompt of {FLOOR_TOKENS:,} tokens or more could be "
                f"prefilled for {run.model_id} on this machine",
            )

        # -- verifying -----------------------------------------------------
        #
        # The boundary came out of a search whose failed probes each left the
        # allocator in a different state. Re-running the winner on a FRESH
        # engine is what turns it from "succeeded once during a search" into
        # a number worth writing to a setting -- and it is where the reported
        # prefill rate comes from. Fresh means a reload even when the current
        # engine is big enough: the point is a clean start.
        self._progress(run, "verifying", 85.0, f"verifying {best:,} tokens")
        verified = await self._probe(run, best, measure=True, fresh=True)
        if self._cancelled(run):
            return self._finish_cancelled(run)
        result.attempts += 1
        if verified.ok:
            result.verified_tokens = best
            result.verified_prompt_tokens = verified.prompt_tokens
            result.prefill_tps = verified.prefill_tps
        else:
            # Verification failing is a real outcome, not an error: the
            # machine reached this size once and will not reach it reliably.
            # Reporting verified_tokens 0 against a non-zero boundary is how
            # the UI shows that, and nothing is applied.
            result.capped_by = "memory"
            result.limit_reason = verified.reason
            return self._complete(
                run, result, started,
                f"{best:,} tokens did not hold on a second run; nothing applied",
            )

        # -- applying ------------------------------------------------------
        applied_tokens = (best // APPLY_GRANULARITY) * APPLY_GRANULARITY
        result.applied_tokens = applied_tokens
        if applied_tokens >= FLOOR_TOKENS:
            self._progress(
                run, "applying", 95.0, f"applying {applied_tokens:,} tokens"
            )
            # The ONE write to the override file a run ever makes.
            self._pool.set_ctx_override(run.model_id, applied_tokens)
            result.applied = True

        self._complete(
            run, result, started,
            f"{result.applied_tokens:,} tokens applied to {run.model_id}"
            if result.applied
            else f"measured {best:,} tokens; nothing applied",
        )

    # -- search ------------------------------------------------------------

    async def _search(
        self, run: ContextBenchRun, ceiling: int, hi_start: int
    ) -> tuple[int, int, str | None]:
        """Largest prefill-able size in [FLOOR_TOKENS, ceiling], probes used,
        and why the last size above it was refused (None if none was).

        `hi_start` -- the prediction, or the ceiling when there is none -- is
        probed first: when the estimate holds, which is the common case, the
        answer is one or two probes away where a bisection from zero would
        have paid for six. Each probe is a real prefill, so that matters.

        The bracket is [lo, hi) where `lo` is known to work and `hi` is known
        to FAIL. `hi` starting at the ceiling would be a lie -- the ceiling is
        merely the highest size we care about, and assuming it fails is how
        the search comes back one grid step short of a machine that reaches
        its target. So a prediction that holds below the ceiling climbs to the
        ceiling and probes it for real before anything is bisected.
        """
        attempts = 0
        lo = 0                  # highest size known to work
        hi: int | None = None   # lowest size known to FAIL, if one is known
        last_failure: str | None = None

        probe = await self._probe(run, hi_start)
        attempts += 1
        if self._cancelled(run):
            return lo, attempts, last_failure
        if probe.ok:
            lo = hi_start
            if hi_start >= ceiling:
                return lo, attempts, last_failure
            # The estimate was conservative. Try the whole ceiling before
            # settling for anything under it.
            self._progress(
                run, "searching", 40.0,
                f"{hi_start:,} tokens held; trying the full {ceiling:,}",
            )
            probe = await self._probe(run, ceiling)
            attempts += 1
            if self._cancelled(run):
                return lo, attempts, last_failure
            if probe.ok:
                return ceiling, attempts, last_failure
            hi, last_failure = ceiling, probe.reason
        else:
            hi, last_failure = hi_start, probe.reason
            if hi_start <= FLOOR_TOKENS:
                # Not even the floor fits. Nothing below is worth probing.
                return 0, attempts, last_failure

        # `hi` is an int by here: every branch above either returned or set
        # it to a size a probe actually refused.
        while hi is not None and hi - lo > SEARCH_GRANULARITY:
            if self._cancelled(run):
                return lo, attempts, last_failure
            mid = lo + (hi - lo) // 2
            # Keep every probe on the grid the answer will be rounded to, so
            # the search never spends a full prefill resolving a difference
            # `APPLY_GRANULARITY` discards.
            mid = max(
                FLOOR_TOKENS, (mid // SEARCH_GRANULARITY) * SEARCH_GRANULARITY
            )
            if mid <= lo or mid >= hi:
                break
            span = max(1, ceiling - FLOOR_TOKENS)
            self._progress(
                run, "searching",
                5.0 + 80.0 * min(1.0, max(0.0, (lo - FLOOR_TOKENS) / span)),
                f"probing {mid:,} tokens (best so far {lo:,}; "
                f"{hi:,} refused: {_short(last_failure)})",
            )
            probe = await self._probe(run, mid)
            attempts += 1
            if probe.ok:
                lo = mid
            else:
                hi, last_failure = mid, probe.reason
        return lo, attempts, last_failure

    # -- probing -----------------------------------------------------------

    async def _probe(
        self,
        run: ContextBenchRun,
        n_tokens: int,
        *,
        measure: bool = False,
        fresh: bool = False,
    ) -> _Probe:
        """Can this machine prefill `n_tokens` for this model?

        Raises only `CancelledError`. Every other failure is the answer.
        """
        try:
            return await self._prefill(
                run.model_id, n_tokens, measure=measure, fresh=fresh
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a refusal or an OOM IS the
            # measurement; see the module docstring.
            return _Probe(ok=False, reason=f"{type(exc).__name__}: {exc}")

    async def _prefill(
        self, model_id: str, n_tokens: int, *, measure: bool, fresh: bool
    ) -> _Probe:
        from ..engine.config import RequestParams

        engine = await self._engine_for(model_id, n_tokens, fresh=fresh)
        raw = self._pool.engine_for(model_id)
        renderer = getattr(raw, "model", None) or raw

        prompt = _prompt_of_length(renderer, n_tokens)
        prompt_tokens = _count(renderer, prompt)
        # One token, greedily, with EOG off: this measures the prefill, and
        # generating more would add decode time to a prefill rate.
        params = RequestParams(temp=0.0, max_tokens=1, stop_at_eog=False)

        t0 = time.perf_counter()
        # The engine directly, not `submit_request`: that helper turns an
        # admission refusal into an HTTPException for a route, and here the
        # refusal's own message IS what `limit_reason` should carry.
        rid = await engine.submit(prompt, params)
        first: float | None = None
        try:
            async for _out in engine.stream(rid):
                if first is None:
                    first = time.perf_counter()
        finally:
            engine.release(rid)
        ttft = (first or time.perf_counter()) - t0
        tps = (prompt_tokens / ttft) if measure and ttft > 0 else None
        return _Probe(ok=True, prompt_tokens=prompt_tokens, prefill_tps=tps)

    async def _engine_for(
        self, model_id: str, n_tokens: int, *, fresh: bool = False
    ) -> Any:
        """An engine that can admit `n_tokens`, reloading at a probe geometry
        if the resident one cannot (or `fresh` asks for a clean one).

        The condition reads the LIVE engine's per-sequence capacity, not a
        size we last asked for. They differ exactly when a load failed: the
        engine is then gone, and the next probe must load at ITS size rather
        than retry the impossible one. Per-sequence (`n_ctx_seq`), because
        that is what admission checks; a probe engine has `n_seq_max = 1` so
        it equals `n_ctx` there, but a serving engine's would not.

        The reload is shielded: a cancel arriving mid-load cannot interrupt
        llama.cpp's allocation anyway, and letting the load land in the pool
        is what allows the run's cleanup to unload it instead of leaking it.
        """
        raw = self._pool.engine_for(model_id)
        live = getattr(getattr(raw, "ctx", None), "n_ctx_seq", 0) or 0
        if not fresh and live > n_tokens:
            engine = self._pool.resident(model_id)
            if engine is not None:
                return engine
        config = self._probe_config(n_tokens + 1)
        _mid, engine = await asyncio.shield(
            self._pool.reload(model_id, config=config)
        )
        return engine

    def _probe_config(self, n_ctx: int) -> Any:
        """The pool's config at `n_ctx`, shaped for ONE prompt and no caches
        (see the module docstring)."""
        return dataclasses.replace(
            self._pool.config,
            n_ctx=n_ctx,
            n_seq_max=1,
            prefix_cache=False,
            mlx_prefix_cache=False,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _ceiling(target: int, native: int | None) -> tuple[int, str]:
        if native and native < target:
            return native, "native"
        return target, "target"

    @staticmethod
    def _start_point(ceiling: int, predicted: int | None) -> int:
        # A prediction only narrows where the search STARTS. It never becomes
        # the answer: `capped_by` says memory only if a probe actually
        # failed, because the formula being pessimistic is not the same fact
        # as the machine refusing.
        if predicted and predicted < ceiling:
            return max(FLOOR_TOKENS, min(ceiling, predicted))
        return ceiling

    def _predict_ceiling(self, model_id: str) -> int | None:
        """`host`'s analytic ceiling for this model, or None if unknowable.

        Only ever used to pick where the search starts (see `_search`).
        """
        from .. import host

        try:
            model_b = next(
                (int(m.get("size_bytes") or 0) for m in self._pool.list()
                 if m.get("id") == model_id),
                0,
            )
            caps = host.probe()
            if not caps.mem_bytes or model_b <= 0:
                return None
            per_tok = host.kv_bytes_per_token(model_b)
            if per_tok <= 0:
                return None
            free = caps.mem_bytes - model_b - host.OS_RESERVE_BYTES
            return int(free // per_tok) if free > 0 else None
        except Exception:  # noqa: BLE001 - an unreadable host just means the
            # search starts at the ceiling instead of near the answer.
            return None

    def _cancelled(self, run: ContextBenchRun) -> bool:
        return run._cancel is not None and run._cancel.is_set()

    def _complete(
        self, run: ContextBenchRun, result: ContextBenchResult, started: float,
        message: str,
    ) -> None:
        """Publish `result` as the run's conclusion. The only place it is."""
        result.duration_s = time.perf_counter() - started
        run.result = result
        run.status = "completed"
        self._progress(run, "done", 100.0, message)

    def _finish_cancelled(self, run: ContextBenchRun) -> None:
        """No result at all, rather than a partial one.

        A cancelled run's half-filled result has defaults that read as
        findings: `capped_by` is "memory" because that is the field's default,
        and the Context Bench screen renders it as "Limited by: Available
        memory" on a run that never finished measuring anything. `result` is
        optional on the client for exactly this case, and the status and
        message already say what happened.
        """
        run.result = None
        run.status = "cancelled"
        self._progress(run, "cancelled", run.progress, "cancelled")

    def _progress(
        self, run: ContextBenchRun, phase: str, progress: float, message: str
    ) -> None:
        run.phase = phase
        run.progress = max(0.0, min(100.0, progress))
        run.message = message


@dataclass
class _Probe:
    ok: bool
    prompt_tokens: int = 0
    prefill_tps: float | None = None
    reason: str = ""


def _short(reason: str | None, limit: int = 80) -> str:
    text = (reason or "").strip().splitlines()[0] if reason else ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _count(renderer: Any, prompt: str) -> int:
    try:
        return len(renderer.tokenize(prompt, add_special=True, parse_special=True))
    except Exception:  # noqa: BLE001 - no tokenizer: the requested size stands
        return 0
