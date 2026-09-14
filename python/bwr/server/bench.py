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

    def __init__(self, resolve: Any, renderer_for: Any = None) -> None:
        self._resolve = resolve
        self._runs: dict[str, BenchRun] = {}
        self._active: str | None = None

    # -- inspection --------------------------------------------------------

    def get(self, bench_id: str) -> dict[str, Any] | None:
        run = self._runs.get(bench_id)
        return run.to_dict() if run else None

    @property
    def active(self) -> str | None:
        return self._active

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

        run = BenchRun(
            bench_id=uuid.uuid4().hex[:12],
            model_id=model_id,
            total_tests=len(prompts) * len(batches),
            context_profile=context_profile,
        )
        run._cancel = asyncio.Event()
        self._runs[run.bench_id] = run
        self._active = run.bench_id
        asyncio.create_task(
            self._run(run, prompts, batches, generation_length)
        )
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
        self, run: BenchRun, prompts: list[int], batches: list[int], gen: int
    ) -> None:
        try:
            for pp in prompts:
                for bs in batches:
                    if run._cancel is not None and run._cancel.is_set():
                        run.status = "cancelled"
                        return
                    run.results.append(await self._one(run.model_id, pp, gen, bs))
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

    async def _one(self, model_id: str, pp: int, gen: int, batch: int) -> BenchResult:
        from .common import submit_request
        from ..engine.config import RequestParams

        result = BenchResult(test_type="throughput", pp=pp, tg=gen, batch_size=batch)
        try:
            served, engine, renderer, _overlay = await self._resolve(model_id)
        except Exception as exc:  # noqa: BLE001 - an unknown model is a result
            result.error = f"{type(exc).__name__}: {exc}"
            return result

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
                    if not (out.finished and out.piece == ""
                            and out.finish_reason == "eog"):
                        n += 1
            finally:
                engine.release(rid)
            end = time.perf_counter()
            return (first or end), n, end

        rows = await asyncio.gather(*[one_row() for _ in range(batch)])
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
