"""Throughput benchmark runner.

Driven against a stub engine, so what is under test is the measurement and
bookkeeping -- the test matrix, the prefill/decode split, cancellation, and
the refusal to run two benchmarks at once -- not the speed of any model.
"""

from __future__ import annotations

import asyncio

import pytest
from bwr.server.bench import BenchError, BenchRunner, _prompt_of_length


class _Out:
    def __init__(self, piece: str, finished: bool = False, reason: str | None = None):
        self.piece = piece
        self.finished = finished
        self.finish_reason = reason


class _StubEngine:
    """Emits `tokens` pieces per request, with a small delay for each."""

    def __init__(self, tokens: int = 4, delay: float = 0.001):
        self.tokens = tokens
        self.delay = delay
        self.submitted = 0
        self.released = 0
        self.concurrent = 0
        self.max_concurrent = 0

    async def submit(self, prompt, params):
        self.submitted += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        return self.submitted

    async def stream(self, rid):
        for i in range(self.tokens):
            await asyncio.sleep(self.delay)
            last = i == self.tokens - 1
            yield _Out("x", finished=last, reason="length" if last else None)

    def release(self, rid):
        self.released += 1
        self.concurrent -= 1


class _Renderer:
    def tokenize(self, text, add_special=True, parse_special=True):
        return text.split()


def _resolver(engine):
    async def resolve(model_id):
        return model_id, engine, _Renderer(), {}
    return resolve


def _run(coro):
    return asyncio.run(coro)


async def _drain(runner, bench_id, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        body = runner.get(bench_id)
        if body["status"] != "running":
            # Surface a runner failure here; otherwise it reaches the caller
            # as a puzzling zero count instead of the traceback it is.
            assert body["error"] is None, body["error"]
            return body
        await asyncio.sleep(0.005)
    raise AssertionError(f"benchmark did not finish: {runner.get(bench_id)}")


async def _drain_any(runner, bench_id, timeout=5.0):
    """`_drain` for a run EXPECTED to fail: no assertion on `error`."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        body = runner.get(bench_id)
        if body["status"] != "running":
            return body
        await asyncio.sleep(0.005)
    raise AssertionError(f"benchmark did not finish: {runner.get(bench_id)}")


class _SubmitRaises(_StubEngine):
    """An engine that resolves fine and then blows up on submit -- a failure
    past `_one`'s resolve guard, which is the runner's `failed` branch."""

    async def submit(self, prompt, params):
        raise RuntimeError("boom")


# -- the test matrix ---------------------------------------------------------


def test_singles_run_at_every_prompt_length_and_batches_once_at_the_longest():
    """The two clients render two tables. Singles are keyed by (pp, tg);
    batch rows by batch size ALONE, so a size that ran at two prompt
    lengths would show once and the run would look short. Rows tagged
    anything but `single`/`batch` are dropped by both clients -- a live run
    through the dashboard was the only thing that caught this."""
    async def main():
        engine = _StubEngine()
        runner = BenchRunner(_resolver(engine))
        started = runner.start("m", prompt_lengths=[8, 16], batch_sizes=[1, 2, 4],
                               generation_length=4)
        assert started["total_tests"] == 4
        body = await _drain(runner, started["bench_id"])
        assert body["status"] == "completed"
        rows = {(r["test_type"], r["pp"], r["batch_size"]) for r in body["results"]}
        assert rows == {
            ("single", 8, 1), ("single", 16, 1),
            ("batch", 16, 2), ("batch", 16, 4),
        }

    _run(main())


def test_a_run_with_only_batch_sizes_still_has_its_singles():
    """The Mac app's defaults are batch sizes [2, 4] with no 1: the single
    table must not come back empty for the default request."""
    async def main():
        runner = BenchRunner(_resolver(_StubEngine()))
        body = await _drain(runner, runner.start(
            "m", prompt_lengths=[8], batch_sizes=[2], generation_length=2
        )["bench_id"])
        assert [r["test_type"] for r in body["results"]] == ["single", "batch"]

    _run(main())


def test_a_batch_issues_that_many_concurrent_requests():
    """Batch size N must be N requests in flight at once, not N in sequence --
    otherwise the number describes latency, not batched throughput."""
    async def main():
        engine = _StubEngine(tokens=3, delay=0.005)
        runner = BenchRunner(_resolver(engine))
        started = runner.start("m", prompt_lengths=[8], batch_sizes=[4],
                               generation_length=3)
        await _drain(runner, started["bench_id"])
        assert engine.max_concurrent == 4

    _run(main())


def test_every_request_is_released():
    async def main():
        engine = _StubEngine()
        runner = BenchRunner(_resolver(engine))
        started = runner.start("m", prompt_lengths=[8], batch_sizes=[3],
                               generation_length=2)
        await _drain(runner, started["bench_id"])
        # One single plus a batch of three.
        assert engine.released == engine.submitted == 4

    _run(main())


# -- what the numbers mean ---------------------------------------------------


def test_results_carry_the_rates_the_client_renders():
    async def main():
        engine = _StubEngine(tokens=5, delay=0.002)
        runner = BenchRunner(_resolver(engine))
        started = runner.start("m", prompt_lengths=[16], batch_sizes=[1],
                               generation_length=5)
        body = await _drain(runner, started["bench_id"])
        r = body["results"][0]
        for key in ("ttft_ms", "tpot_ms", "processing_tps", "gen_tps",
                    "e2e_latency_s", "total_throughput"):
            assert r[key] is not None and r[key] > 0, key
        # BenchResultDTO reads some of these under a second name; both are
        # emitted so the screen does not need to know which server it is on.
        assert r["pp_tps"] == r["processing_tps"]
        assert r["tg_tps"] == r["gen_tps"]
        assert r["avg_ttft_ms"] == r["ttft_ms"]

    _run(main())


def test_prefill_rate_counts_every_row_in_the_batch():
    """A batch of 4 processed 4 prompts, even though they overlapped."""
    async def main():
        engine = _StubEngine(tokens=2, delay=0.001)
        runner = BenchRunner(_resolver(engine))
        one = await _drain(runner, runner.start(
            "m", prompt_lengths=[100], batch_sizes=[1], generation_length=2
        )["bench_id"])
        four = await _drain(runner, runner.start(
            "m", prompt_lengths=[100], batch_sizes=[4], generation_length=2
        )["bench_id"])
        batch = next(r for r in four["results"] if r["test_type"] == "batch")
        # Same per-row work, four times the prompt tokens: the reported
        # prefill rate must scale, not stay flat.
        assert batch["processing_tps"] > one["results"][0]["processing_tps"]

    _run(main())


def test_tpot_is_absent_for_a_single_token_answer():
    """Time per output token needs at least two tokens to be a gap."""
    async def main():
        engine = _StubEngine(tokens=1)
        runner = BenchRunner(_resolver(engine))
        body = await _drain(runner, runner.start(
            "m", prompt_lengths=[8], batch_sizes=[1], generation_length=1
        )["bench_id"])
        assert body["results"][0]["tpot_ms"] is None

    _run(main())


# -- lifecycle ---------------------------------------------------------------


def test_a_second_benchmark_is_refused_while_one_runs():
    """Two at once contend for the GPU and describe neither."""
    async def main():
        engine = _StubEngine(tokens=50, delay=0.01)
        runner = BenchRunner(_resolver(engine))
        first = runner.start("m", prompt_lengths=[8], batch_sizes=[1],
                             generation_length=50)
        with pytest.raises(BenchError):
            runner.start("m", prompt_lengths=[8], batch_sizes=[1])
        runner.cancel(first["bench_id"])
        await _drain(runner, first["bench_id"])

    _run(main())


def test_a_benchmark_can_be_cancelled_between_tests():
    async def main():
        engine = _StubEngine(tokens=3, delay=0.01)
        runner = BenchRunner(_resolver(engine))
        started = runner.start("m", prompt_lengths=[8, 16, 32, 64],
                               batch_sizes=[1], generation_length=3)
        await asyncio.sleep(0.02)
        assert runner.cancel(started["bench_id"]) is True
        body = await _drain(runner, started["bench_id"])
        assert body["status"] == "cancelled"
        assert len(body["results"]) < body["total_tests"]

    _run(main())


def test_cancelling_a_finished_benchmark_is_a_no_op():
    async def main():
        runner = BenchRunner(_resolver(_StubEngine()))
        started = runner.start("m", prompt_lengths=[8], batch_sizes=[1],
                               generation_length=2)
        await _drain(runner, started["bench_id"])
        assert runner.cancel(started["bench_id"]) is False

    _run(main())


def test_active_summary_reports_the_running_run_and_nothing_after():
    """What `/api/bench/active` hands a refreshed dashboard."""
    async def main():
        engine = _StubEngine(tokens=50, delay=0.01)
        runner = BenchRunner(_resolver(engine))
        assert runner.active_summary() is None
        started = runner.start("m", prompt_lengths=[8, 16], batch_sizes=[1],
                               generation_length=50)
        summary = runner.active_summary()
        assert summary["bench_id"] == started["bench_id"]
        assert summary["model_id"] == "m"
        assert summary["status"] == "running"
        assert (summary["completed_tests"], summary["total_tests"]) == (0, 2)
        runner.cancel(started["bench_id"])
        await _drain(runner, started["bench_id"])
        assert runner.active_summary() is None

    _run(main())


def test_a_run_finishing_frees_the_slot_for_the_next_one():
    async def main():
        runner = BenchRunner(_resolver(_StubEngine()))
        first = runner.start("m", prompt_lengths=[8], batch_sizes=[1],
                             generation_length=2)
        await _drain(runner, first["bench_id"])
        second = runner.start("m", prompt_lengths=[8], batch_sizes=[1],
                              generation_length=2)
        await _drain(runner, second["bench_id"])

    _run(main())


def test_a_failure_past_resolve_fails_the_run_with_the_exception():
    """An unknown model is a per-row result; anything the engine throws
    after that is the run's failure, named so the page can show it."""
    async def main():
        runner = BenchRunner(_resolver(_SubmitRaises()))
        started = runner.start("m", prompt_lengths=[8], batch_sizes=[1],
                               generation_length=2)
        body = await _drain_any(runner, started["bench_id"])
        assert body["status"] == "failed"
        # The runner submits through the serving path, which maps an
        # engine RuntimeError to the 503 a client would have seen; the
        # run's error names that wrapper and keeps the engine's message.
        assert body["error"] == "HTTPException: 503: boom"
        assert body["results"] == []

    _run(main())


def test_a_failed_run_releases_the_active_slot():
    """`finally: _active = None` -- without it one engine failure would
    refuse every benchmark until the server restarted."""
    async def main():
        runner = BenchRunner(_resolver(_SubmitRaises()))
        first = runner.start("m", prompt_lengths=[8], batch_sizes=[1],
                             generation_length=2)
        assert runner.active_summary()["bench_id"] == first["bench_id"]
        await _drain_any(runner, first["bench_id"])
        assert runner.active_summary() is None
        assert runner.active is None
        second = runner.start("m", prompt_lengths=[8], batch_sizes=[1],
                              generation_length=2)
        assert second["bench_id"] != first["bench_id"]
        await _drain_any(runner, second["bench_id"])

    _run(main())


def test_an_unknown_model_is_a_result_not_a_crash():
    async def main():
        async def resolve(model_id):
            raise ValueError(f"unknown model {model_id!r}")

        runner = BenchRunner(resolve)
        started = runner.start("nope", prompt_lengths=[8], batch_sizes=[1],
                               generation_length=2)
        body = await _drain(runner, started["bench_id"])
        assert body["status"] == "completed"
        assert "unknown model" in body["results"][0]["error"]

    _run(main())


@pytest.mark.parametrize("kwargs", [
    {"prompt_lengths": []},
    {"batch_sizes": []},
    {"generation_length": 0},
    {"generation_length": -1},
])
def test_a_degenerate_matrix_is_refused(kwargs):
    runner = BenchRunner(_resolver(_StubEngine()))
    with pytest.raises(BenchError):
        runner.start("m", **kwargs)


def test_unknown_bench_ids_read_as_absent():
    assert BenchRunner(_resolver(_StubEngine())).get("nope") is None


# -- prompt construction -----------------------------------------------------


def test_prompt_is_trimmed_towards_the_requested_length():
    r = _Renderer()
    text = _prompt_of_length(r, 20)
    assert abs(len(r.tokenize(text)) - 20) <= 5


def test_prompt_construction_survives_a_missing_tokenizer():
    class _NoTokenizer:
        def tokenize(self, *a, **k):
            raise AttributeError("no tokenizer here")

    assert _prompt_of_length(_NoTokenizer(), 10)
