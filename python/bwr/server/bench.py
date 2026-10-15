"""Throughput benchmarks run through the real serving path.

Why through the server and not straight at the engine
-----------------------------------------------------
`tune.py` drives a private engine for its own purposes. This one issues
ordinary requests against the SAME AsyncEngine that serves traffic, for two
reasons: the numbers then include the scheduling and detokenisation a real
request pays, and the alternative -- touching the raw engine while its worker
thread is running -- is not safe.

That is also why the runner is an asyncio task rather than a thread. It
awaits the engine exactly as a request handler does, so a batch of size N is
literally N concurrent requests, not a simulation of them.

What is measured, per (prompt length x batch size)
--------------------------------------------------
  ttft_ms        wall time to the first token, averaged across the batch
  tpot_ms        mean time per output token after the first
  processing_tps prompt tokens / time-to-first-token -- prefill rate
  gen_tps        generated tokens / decode time -- what "tokens per second"
                 usually means
  total_throughput  all tokens (prompt + generated) / end-to-end time
  peak_memory_bytes  MLX peak allocation, reset before each test

Prefill and decode are separated at the first token for the same reason
server/stats.py does it: one blended rate over a 4k prompt and a 20-token
answer describes neither phase.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

#: Filler repeated to build a prompt of a requested length. A single common
#: word tokenises ~1:1, so the target length is reached without a search.
_FILLER = "word "


@dataclass
class BenchResult:
    test_type: str
    pp: int | None = None
    tg: int | None = None
    batch_size: int | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    processing_tps: float | None = None
    gen_tps: float | None = None
    e2e_latency_s: float | None = None
    total_throughput: float | None = None
    peak_memory_bytes: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_type": self.test_type,
            "pp": self.pp,
            "tg": self.tg,
            "batch_size": self.batch_size,
            "ttft_ms": self.ttft_ms,
            "avg_ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "processing_tps": self.processing_tps,
            "pp_tps": self.processing_tps,
            "gen_tps": self.gen_tps,
            "tg_tps": self.gen_tps,
            "e2e_latency_s": self.e2e_latency_s,
            "total_throughput": self.total_throughput,
            "peak_memory_bytes": self.peak_memory_bytes,
            "error": self.error,
        }


@dataclass
class BenchRun:
    bench_id: str
    model_id: str
    status: str = "running"        # running|completed|cancelled|failed
    total_tests: int = 0
    results: list[BenchResult] = field(default_factory=list)
    error: str | None = None
    context_profile: str | None = None
    started_at: float = field(default_factory=time.time)
    _cancel: asyncio.Event | None = field(default=None, repr=False)
    # Held so the event loop's weak task set is not the only reference:
    # a task nobody holds can be collected mid-run.
    _task: asyncio.Task[None] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bench_id": self.bench_id,
            "status": self.status,
            "model_id": self.model_id,
            "context_profile": self.context_profile,
            "total_tests": self.total_tests,
            "completed_tests": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "error": self.error,
            # bwr has no submission endpoint to upload results to, and the
            # client renders this block only when it is present.
            "upload_state": None,
        }


class BenchError(RuntimeError):
    pass


def _peak_memory_reset() -> None:
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:  # noqa: BLE001 - Metal backend or no MLX: no peak to read
        pass


def _peak_memory() -> int | None:
    try:
        import mlx.core as mx

        return int(mx.get_peak_memory())
    except Exception:  # noqa: BLE001 - see above
        return None


class BenchRunner:
    """Owns benchmark runs. One in flight at a time, by design.

    Two concurrent benchmarks would contend for the same GPU and report
    numbers describing neither, so a second start is refused rather than
    quietly producing noise.
    """

    def __init__(
        self,
        resolve: Any,
        renderer_for: Any = None,
        *,
        request_started: Any = None,
        request_finished: Any = None,
    ) -> None:
        self._resolve = resolve
        # The pool's in-flight bracket, the same pair the request handlers
        # use. A benchmark row is a request as far as the pool is concerned:
        # without the bracket an eviction (a request for another model that
        # needs this one's budget) stopped the engine under the row, which
        # then "completed" with a rate computed from a few tokens.
        self._request_started = request_started or (lambda _mid: None)
        self._request_finished = request_finished or (lambda _mid: None)
        self._runs: dict[str, BenchRun] = {}
        self._active: str | None = None

    # -- inspection --------------------------------------------------------

    def get(self, bench_id: str) -> dict[str, Any] | None:
        run = self._runs.get(bench_id)
        return run.to_dict() if run else None

    @property
    def active(self) -> str | None:
        return self._active

    def active_summary(self) -> dict[str, Any] | None:
        """The run in progress, in the shape a client re-attaches from, or
        None. What `/api/bench/active` reports."""
        run = self._runs.get(self._active) if self._active is not None else None
        if run is None or run.status != "running":
            return None
        return {
            "bench_id": run.bench_id,
            "model_id": run.model_id,
            "status": run.status,
            "total_tests": run.total_tests,
            "completed_tests": len(run.results),
        }

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        model_id: str,
        *,
        prompt_lengths: list[int] | None = None,
        generation_length: int = 64,
        batch_sizes: list[int] | None = None,
        context_profile: str | None = None,
    ) -> dict[str, Any]:
        if self._active is not None:
            running = self._runs.get(self._active)
            if running is not None and running.status == "running":
                raise BenchError(
                    f"benchmark {self._active} is still running; cancel it first"
                )
        # `None` means "use the defaults"; an explicitly EMPTY list is a
        # caller mistake and must not silently become the default matrix.
        prompts = [p for p in (
            [128, 512] if prompt_lengths is None else prompt_lengths
        ) if p > 0]
        batches = [b for b in (
            [1] if batch_sizes is None else batch_sizes
        ) if b > 0]
        if not prompts or not batches:
            raise BenchError("prompt_lengths and batch_sizes must be non-empty")
        if generation_length <= 0:
            raise BenchError("generation_length must be positive")

        # Two kinds of test, because that is what both clients render. A
        # `single` row is one request at a prompt length; the page keys
        # those by (pp, tg). A `batch` row is N concurrent requests; the
        # page keys those by batch size ALONE, so every batch size runs
        # once, at the longest requested prompt, rather than once per
        # prompt length -- a second row for the same size would be dropped
        # on arrival and the run would look shorter than it was. A row
        # tagged anything else is dropped by both the web page and the Mac
        # app, which is how every bwr run used to render as empty.
        tests = [(pp, 1, "single") for pp in prompts]
        tests += [(max(prompts), bs, "batch") for bs in batches if bs > 1]

        run = BenchRun(
            bench_id=uuid.uuid4().hex[:12],
            model_id=model_id,
            total_tests=len(tests),
            context_profile=context_profile,
        )
        run._cancel = asyncio.Event()
        self._runs[run.bench_id] = run
        self._active = run.bench_id
        run._task = asyncio.create_task(self._run(run, tests, generation_length))
        return {
            "bench_id": run.bench_id,
            "status": run.status,
            "total_tests": run.total_tests,
        }

    def cancel(self, bench_id: str) -> bool:
        run = self._runs.get(bench_id)
        if run is None or run.status != "running":
            return False
        if run._cancel is not None:
            run._cancel.set()
        return True

    # -- worker ------------------------------------------------------------

    async def _run(
        self, run: BenchRun, tests: list[tuple[int, int, str]], gen: int
    ) -> None:
        try:
            for pp, bs, kind in tests:
                if run._cancel is not None and run._cancel.is_set():
                    run.status = "cancelled"
                    return
                run.results.append(
                    await self._one(run.model_id, pp, gen, bs, kind)
                )
            run.status = "completed"
        except asyncio.CancelledError:
            run.status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - a benchmark failure is a
            # result to report, not a reason to take the server down.
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
        finally:
            if self._active == run.bench_id:
                self._active = None

    async def _one(
        self, model_id: str, pp: int, gen: int, batch: int, kind: str
    ) -> BenchResult:
        result = BenchResult(test_type=kind, pp=pp, tg=gen, batch_size=batch)
        try:
            served, engine, renderer, _overlay = await self._resolve(model_id)
        except Exception as exc:  # noqa: BLE001 - an unknown model is a result
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        # Bracketed per row, right after resolve returns (no await in
        # between, so an eviction cannot slip into the gap) and released
        # once every request of the row has drained. NOT around the whole
        # run: a row for a second model would then wait on this one's
        # in-flight count from inside the bracket that holds it up.
        self._request_started(served)
        try:
            return await self._measure(result, engine, renderer, pp, gen, batch)
        finally:
            self._request_finished(served)

    async def _measure(
        self,
        result: BenchResult,
        engine: Any,
        renderer: Any,
        pp: int,
        gen: int,
        batch: int,
    ) -> BenchResult:
        from ..engine.config import RequestParams
        from .common import counts_as_token, submit_request

        prompt = _prompt_of_length(renderer, pp)
        # Greedy and EOG-free: the run must generate exactly `gen` tokens, or
        # the rates describe different amounts of work per row.
        params = RequestParams(temp=0.0, max_tokens=gen, stop_at_eog=False)

        _peak_memory_reset()
        t0 = time.perf_counter()

        async def one_row() -> tuple[float, int, float]:
            rid = await submit_request(engine, prompt, params)
            first: float | None = None
            n = 0
            try:
                async for out in engine.stream(rid):
                    if first is None:
                        first = time.perf_counter()
                    if counts_as_token(out):
                        n += 1
            finally:
                engine.release(rid)
            end = time.perf_counter()
            return (first or end), n, end

        # A row that fails does not end the batch on its own: its siblings
        # are still streaming on the engine, and the caller releases the
        # pool's in-flight bracket as soon as this returns. A bare gather
        # raised on the first failure with the rest live, so the bracket
        # went while the engine still held their requests. Stop the
        # siblings and wait for every one to settle (each releases its
        # own request in its finally) before the failure propagates.
        tasks = [asyncio.create_task(one_row()) for _ in range(batch)]
        try:
            rows = await asyncio.gather(*tasks)
        except BaseException:  # noqa: BLE001 - must catch cancellation too
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        t_end = time.perf_counter()

        firsts = [r[0] for r in rows]
        counts = [r[1] for r in rows]
        ends = [r[2] for r in rows]
        generated = sum(counts)
        ttft_s = statistics.mean(f - t0 for f in firsts)
        decode_s = statistics.mean(e - f for f, e in zip(firsts, ends))
        e2e = t_end - t0

        result.ttft_ms = ttft_s * 1000
        result.e2e_latency_s = e2e
        result.peak_memory_bytes = _peak_memory()
        # Prefill rate counts every row's prompt: a batch of 4 processed 4
        # prompts, even though they overlapped.
        result.processing_tps = (pp * batch / ttft_s) if ttft_s > 0 else None
        result.gen_tps = (generated / decode_s) if decode_s > 0 else None
        per_row = statistics.mean(counts) if counts else 0
        result.tpot_ms = (
            (decode_s / (per_row - 1)) * 1000 if per_row > 1 else None
        )
        result.total_throughput = ((pp * batch + generated) / e2e) if e2e > 0 else None
        return result


def _prompt_of_length(renderer: Any, target_tokens: int) -> str:
    """A prompt of about `target_tokens` tokens.

    Approximate on purpose. Tokenising in a loop to hit an exact count costs
    more than the precision buys: the measured rates use the REQUESTED length
    consistently, so a few tokens either way shifts every row equally and
    changes no comparison.
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
