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

How a candidate is probed, and why it differs by backend
========================================================

MLX grows its KV cache as tokens arrive; `n_ctx` there is purely an admission
cap, so one engine loaded at the ceiling can be probed at every candidate
size by prefilling different amounts. llama.cpp allocates the whole KV cache
when the context is created, so for a Metal model the allocation under test
happens at LOAD time and each candidate needs its own load.

One bisection drives both. `_probe` is the only thing that differs:

    mlx    load once at the ceiling, prefill N tokens
    metal  load at n_ctx = N, then prefill N tokens

The Metal path is why the client warns about runtime -- a load per probe on a
27B is minutes, not seconds.

What "failure" means
====================

A probe fails when the engine refuses the prompt or the prefill raises. Both
are counted the same way, deliberately: an admission refusal and an
out-of-memory both mean "this machine will not serve a prompt that size",
which is the question being asked. The distinction is preserved in the
message the UI shows, not in the boundary.

`asyncio.CancelledError` is NEVER treated as a failed probe. Swallowing it
would turn a cancel into a measurement and write a fabricated boundary into
the model's settings.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

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

#: Repeated to build a prompt. One common word tokenises ~1:1, so the target
#: length is reached without searching for it.
_FILLER = "word "


class ContextBenchError(RuntimeError):
    """A start that cannot be honoured: bad target, or a run already going."""


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
    result: ContextBenchResult | None = None
    error: str | None = None
    started: float = field(default_factory=time.perf_counter)
    _cancel: asyncio.Event | None = None
    _task: asyncio.Task[None] | None = None

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
        mid = self._pool.resolve(model_id)
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
        run = self._runs.get(bench_id)
        if run is None or run.status != "running":
            return False
        if run._cancel is not None:
            run._cancel.set()
        return True

    # -- worker ------------------------------------------------------------

    async def _run(self, run: ContextBenchRun) -> None:
        # Probing REWRITES the model's context override, once per probe, and
        # the last value written is whatever size was being tested when the
        # run ended -- including a size that failed. Only a completed run that
        # reached the apply step gets to leave its value behind; every other
        # exit restores what was there before, so an abandoned benchmark does
        # not quietly reconfigure the model.
        previous = self._pool.ctx_override(run.model_id)
        try:
            await self._measure(run)
        except asyncio.CancelledError:
            run.status = "cancelled"
            run.message = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - a benchmark failure is a
            # result to report, not a reason to take the server down.
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"
            run.message = run.error
        finally:
            try:
                if not (run.result is not None and run.result.applied):
                    self._pool.set_ctx_override(run.model_id, previous)
                # Whatever the outcome, the resident engine was built for a
                # probe, not for serving. Drop it so the next request loads at
                # the value that is now configured rather than at the last
                # size tested.
                await self._unload_all()
            except Exception:  # noqa: BLE001 - cleanup runs while an error or
                # a task cancellation is already propagating; letting it raise
                # here would replace the real outcome with a cleanup failure.
                logger.warning("ctxbench: cleanup after %s failed", run.bench_id,
                               exc_info=True)
            if self._active == run.bench_id:
                self._active = None

    async def _measure(self, run: ContextBenchRun) -> None:
        started = time.perf_counter()
        result = ContextBenchResult(
            model_id=run.model_id, target_tokens=run.target_tokens
        )
        run.result = result

        # -- preparing: free the machine up, and find the ceiling ----------
        self._progress(run, "preparing", 2.0, "unloading resident models")
        await self._unload_all()
        if self._cancelled(run):
            return self._finish_cancelled(run, result, started)

        native = self._pool.native_ctx(run.model_id)
        result.native_context_length = native

        ceiling = run.target_tokens
        capped_by = "target"
        if native and native < ceiling:
            ceiling, capped_by = native, "native"
        predicted = self._predictor(run.model_id)
        if predicted and predicted < ceiling:
            # A prediction only narrows where the search STARTS. It never
            # becomes the answer: `capped_by` still says memory only if a
            # probe actually failed, because the formula being pessimistic
            # is not the same fact as the machine refusing.
            hi_start = max(FLOOR_TOKENS, min(ceiling, predicted))
        else:
            hi_start = ceiling

        if ceiling < FLOOR_TOKENS:
            result.capped_by = capped_by
            result.duration_s = time.perf_counter() - started
            run.status = "completed"
            self._progress(
                run, "done", 100.0,
                f"{run.model_id} tops out at {ceiling} tokens, below the "
                f"{FLOOR_TOKENS}-token floor",
            )
            return

        # -- searching -----------------------------------------------------
        self._progress(
            run, "searching", 5.0,
            f"probing up to {ceiling:,} tokens"
            + (f" (estimate: {predicted:,})" if predicted else ""),
        )
        best, attempts = await self._search(run, ceiling, hi_start)
        result.attempts = attempts
        if self._cancelled(run):
            return self._finish_cancelled(run, result, started)
        result.measured_tokens = best
        if best < ceiling:
            capped_by = "memory"
        result.capped_by = capped_by

        if best <= 0:
            run.status = "completed"
            result.duration_s = time.perf_counter() - started
            self._progress(
                run, "done", 100.0,
                f"no prompt of {FLOOR_TOKENS:,} tokens or more could be "
                f"prefilled for {run.model_id} on this machine",
            )
            return

        # -- verifying -----------------------------------------------------
        #
        # The boundary came out of a search whose failed probes each left the
        # allocator in a different state. Re-running the winner from a clean
        # start is what turns it from "succeeded once during a search" into a
        # number worth writing to a setting -- and it is where the reported
        # prefill rate comes from.
        self._progress(run, "verifying", 85.0, f"verifying {best:,} tokens")
        verified = await self._probe(run, best, measure=True)
        if self._cancelled(run):
            return self._finish_cancelled(run, result, started)
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
            run.status = "completed"
            result.duration_s = time.perf_counter() - started
            self._progress(
                run, "done", 100.0,
                f"{best:,} tokens did not hold on a second run; nothing applied",
            )
            return

        # -- applying ------------------------------------------------------
        applied_tokens = (best // APPLY_GRANULARITY) * APPLY_GRANULARITY
        if applied_tokens >= FLOOR_TOKENS:
            self._progress(
                run, "applying", 95.0, f"applying {applied_tokens:,} tokens"
            )
            self._pool.set_ctx_override(run.model_id, applied_tokens)
            result.applied_tokens = applied_tokens
            result.applied = True
        else:
            result.applied_tokens = applied_tokens

        result.duration_s = time.perf_counter() - started
        run.status = "completed"
        self._progress(
            run, "done", 100.0,
            f"{result.applied_tokens:,} tokens applied to {run.model_id}"
            if result.applied
            else f"measured {best:,} tokens; nothing applied",
        )

    # -- search ------------------------------------------------------------

    async def _search(
        self, run: ContextBenchRun, ceiling: int, hi_start: int
    ) -> tuple[int, int]:
        """Largest prefill-able size in [FLOOR_TOKENS, ceiling], and probes used.

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

        probe = await self._probe(run, hi_start)
        attempts += 1
        if self._cancelled(run):
            return lo, attempts
        if probe.ok:
            lo = hi_start
            if hi_start >= ceiling:
                return lo, attempts
            # The estimate was conservative. Try the whole ceiling before
            # settling for anything under it.
            self._progress(
                run, "searching", 40.0,
                f"{hi_start:,} tokens held; trying the full {ceiling:,}",
            )
            probe = await self._probe(run, ceiling)
            attempts += 1
            if self._cancelled(run):
                return lo, attempts
            if probe.ok:
                return ceiling, attempts
            hi = ceiling
        else:
            hi = hi_start
            if hi_start <= FLOOR_TOKENS:
                # Not even the floor fits. Nothing below is worth probing.
                return 0, attempts

        # `hi` is an int by here: every branch above either returned or set
        # it to a size a probe actually refused.
        while hi is not None and hi - lo > SEARCH_GRANULARITY:
            if self._cancelled(run):
                return lo, attempts
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
                f"probing {mid:,} tokens (best so far {lo:,})",
            )
            probe = await self._probe(run, mid)
            attempts += 1
            if probe.ok:
                lo = mid
            else:
                hi = mid
        return lo, attempts

    # -- probing -----------------------------------------------------------

    async def _probe(
        self, run: ContextBenchRun, n_tokens: int, *, measure: bool = False
    ) -> _Probe:
        """Can this machine prefill `n_tokens` for this model?

        Raises only `CancelledError`. Every other failure is the answer.
        """
        try:
            return await self._prefill(run.model_id, n_tokens, measure=measure)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a refusal or an OOM IS the
            # measurement; see the module docstring.
            return _Probe(ok=False, reason=f"{type(exc).__name__}: {exc}")

    async def _prefill(
        self, model_id: str, n_tokens: int, *, measure: bool
    ) -> _Probe:
        from ..engine.config import RequestParams
        from .common import submit_request

        # Raise the admission cap to the size under test and reload, so the
        # engine being probed is the one that would serve a prompt this long.
        #
        # The condition reads the LIVE engine's capacity, not the override we
        # last wrote. They differ exactly when a load failed: the override is
        # then set to a size this machine refused, and trusting it would make
        # every smaller probe reload at that same impossible size and fail for
        # a reason that has nothing to do with the size being probed.
        raw = self._pool.engine_for(model_id)
        capacity = getattr(getattr(raw, "config", None), "n_ctx", 0) or 0
        if capacity <= n_tokens:
            self._pool.set_ctx_override(model_id, n_tokens + 1)
            await self._pool.reload(model_id)
        mid, engine = await self._pool.acquire(model_id)
        raw = self._pool.engine_for(mid)
        renderer = getattr(raw, "model", None) or raw

        prompt = _prompt_of_length(renderer, n_tokens)
        prompt_tokens = _count(renderer, prompt)
        # One token, greedily, with EOG off: this measures the prefill, and
        # generating more would add decode time to a prefill rate.
        params = RequestParams(temp=0.0, max_tokens=1, stop_at_eog=False)

        t0 = time.perf_counter()
        rid = await submit_request(engine, prompt, params)
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

    # -- helpers -----------------------------------------------------------

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

    async def _unload_all(self) -> None:
        for mid in list(self._pool.loaded_ids):
            await self._pool.unload(mid)

    def _cancelled(self, run: ContextBenchRun) -> bool:
        return run._cancel is not None and run._cancel.is_set()

    def _finish_cancelled(
        self, run: ContextBenchRun, result: ContextBenchResult, started: float
    ) -> None:
        """Drop the partial result rather than publishing it.

        A cancelled run has a half-filled result whose defaults read as
        findings: `capped_by` is "memory" because that is the field's default,
        and the Context Bench screen renders it as "Limited by: Available
        memory" on a run that never finished measuring anything. Reporting no
        result at all is the truth -- `result` is optional on the client for
        exactly this case, and the status and message already say what
        happened.
        """
        result.duration_s = time.perf_counter() - started
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


def _count(renderer: Any, prompt: str) -> int:
    try:
        return len(renderer.tokenize(prompt, add_special=True, parse_special=True))
    except Exception:  # noqa: BLE001 - no tokenizer: the requested size stands
        return 0


def _prompt_of_length(renderer: Any, target_tokens: int) -> str:
    """A prompt of about `target_tokens` tokens.

    The same approximation `server/bench.py` uses, and for the same reason:
    tokenising in a loop to hit an exact count costs more than the precision
    buys. Here it is also why the result reports `verified_prompt_tokens`
    separately from `verified_tokens` -- the requested size and what the
    tokenizer actually produced are different numbers, and the client shows
    the one the prefill really completed.
    """
    text = _FILLER * max(1, target_tokens)
    try:
        n = len(renderer.tokenize(text, add_special=True, parse_special=True))
        if n > target_tokens and n > 0:
            keep = max(1, int(len(text) * target_tokens / n))
            text = text[:keep]
    except Exception:  # noqa: BLE001 - no tokenizer: the filler stands as-is
        pass
    return text
