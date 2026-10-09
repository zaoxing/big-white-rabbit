"""Context benchmark: the search, what bounds it, and what it writes.

The fake pool below is a whole machine in miniature: a prefill budget the
engine refuses to exceed, an admission cap that only changes on reload, and
engines that come and go. That is deliberate -- driving the REAL `_prefill`
through a fake engine exercises the reload/admission interplay, which is
where the interesting bugs live, rather than stubbing it out and testing the
bisection arithmetic against itself.

Admission is PER SEQUENCE, as llama.cpp's is: an engine built with
`n_seq_max` admits `n_ctx / n_seq_max` tokens to one prompt. A benchmark
that budgets against `n_ctx` while the engine admits by `n_ctx_seq` comes
out `n_seq_max` times too small, so the fake refuses exactly the way the
real thing does.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest
from bwr.engine.pool import ModelPoolBusy, ModelPoolError
from bwr.server.ctxbench import (
    APPLY_GRANULARITY,
    FLOOR_TOKENS,
    TARGETS,
    ContextBenchError,
    ContextBenchRunner,
)

# Matches bench.py's `_FILLER` ("word ") so a prompt of N tokens is 5N chars.
_CHARS_PER_TOKEN = 5


class _Tokenizer:
    def tokenize(self, text, add_special=False, parse_special=False):
        return [0] * (len(text) // _CHARS_PER_TOKEN)


@dataclass
class _Config:
    """The slice of EngineConfig a probe geometry touches."""

    n_ctx: int
    n_seq_max: int = 8
    prefix_cache: bool = True
    mlx_prefix_cache: bool = True


@dataclass
class _Ctx:
    """The live context geometry, as the real engine's `ctx` reports it."""

    n_ctx: int
    n_seq_max: int

    @property
    def n_ctx_seq(self) -> int:
        return self.n_ctx // self.n_seq_max


class _Output:
    finished = True
    piece = "x"
    finish_reason = "length"


class _FakeEngine(_Tokenizer):
    """An engine that admits up to `n_ctx_seq` and prefills up to
    `prefill_limit`.

    The two limits are separate on purpose: admission is a configured cap
    (ValueError, the same shape both engines raise) while the prefill limit
    is the machine running out (RuntimeError). The benchmark must treat both
    as "this size does not work", and the tests below check it does.
    """

    def __init__(
        self, config: _Config, prefill_limit: int, *, pool: _FakePool
    ) -> None:
        self.config = config
        self.ctx = _Ctx(n_ctx=config.n_ctx, n_seq_max=config.n_seq_max)
        self._prefill_limit = prefill_limit
        self._pool = pool
        self.submitted: list[int] = []

    async def submit(self, prompt, params):
        n = len(self.tokenize(prompt))
        self.submitted.append(n)
        self._pool.seen_in_maintenance.append(self._pool.maintenance is not None)
        if n >= self.ctx.n_ctx_seq:
            raise ValueError(
                f"prompt is {n} tokens, per-sequence capacity is {self.ctx.n_ctx_seq}"
            )
        if n > self._prefill_limit:
            raise RuntimeError("metal: failed to allocate")
        return 1

    async def stream(self, rid):
        if self._pool.block_streams is not None:
            self._pool.streaming.set()
            await self._pool.block_streams.wait()
        yield _Output()

    def release(self, rid) -> None:
        self._pool.released.append(rid)


class _FakePool:
    """Enough ModelPool to drive the runner, with a real override file."""

    def __init__(
        self,
        model_dir,
        *,
        prefill_limit: int,
        native: int | None = None,
        native_after_load: bool = False,
        size_bytes: int = 4 * 1024**3,
        load_limit: int | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.model_id = "m"
        # The SERVING geometry: eight sequences and both caches on, i.e.
        # the shape a probe must NOT inherit.
        self.config = _Config(n_ctx=4096)
        self._prefill_limit = prefill_limit
        self._native = native
        # A GGUF: the trained length is readable only once resident.
        self._native_after_load = native_after_load
        self._size_bytes = size_bytes
        # A ctx above this cannot be LOADED at all -- llama.cpp's behaviour,
        # where the KV cache is allocated when the context is created.
        self._load_limit = load_limit
        self._override: int | None = None
        self._engine: _FakeEngine | None = None
        self.loads: list[int] = []
        self.load_configs: list[_Config | None] = []
        self.overrides_written: list[int | None] = []
        self.maintenance: str | None = None
        self.maintenance_log: list[str] = []
        self.seen_in_maintenance: list[bool] = []
        self.acquires = 0
        self.released: list[int] = []
        # Set to an Event to make every prefill wait on it (a probe "in
        # flight" for the cancellation tests).
        self.block_streams: asyncio.Event | None = None
        # Set by the engine once a blocked prefill is actually waiting, so a
        # test cancels a probe that is genuinely in flight.
        self.streaming = asyncio.Event()
        # Called on every load with the runner's view of the run; lets a
        # test observe what a poll would see while the search runs.
        self.on_load = None

    # -- the slice of ModelPool the runner uses ---------------------------

    def resolve(self, model_id):
        if model_id not in (self.model_id, None):
            raise ModelPoolError(f"unknown model {model_id!r}; available: m")
        return self.model_id

    def list(self):
        return [{"id": self.model_id, "size_bytes": self._size_bytes}]

    @property
    def loaded_ids(self):
        return [self.model_id] if self._engine is not None else []

    def engine_for(self, model_id):
        return self._engine

    def ctx_override(self, model_id):
        return self._override

    def set_ctx_override(self, model_id, n_ctx):
        self._override = n_ctx
        self.overrides_written.append(n_ctx)

    def native_ctx(self, model_id):
        if self._native_after_load and self._engine is None:
            return None
        return self._native

    def begin_maintenance(self, reason):
        if self.maintenance is not None:
            raise ModelPoolBusy(f"model pool is busy: {self.maintenance}")
        self.maintenance = reason
        self.maintenance_log.append("begin")

    def end_maintenance(self):
        self.maintenance = None
        self.maintenance_log.append("end")

    def resident(self, model_id):
        return self._engine

    async def acquire(self, model_id):
        # The runner must never come through here: `acquire` is the gated
        # serving path, and a probe is not a request.
        self.acquires += 1
        if self.maintenance is not None:
            raise ModelPoolBusy(f"model pool is busy: {self.maintenance}")
        if self._engine is None:
            self._load(None)
        return self.model_id, self._engine

    async def reload(self, model_id, *, config=None):
        self._engine = None
        self._load(config)
        return self.model_id, self._engine

    async def unload(self, model_id):
        self._engine = None
        return True

    async def unload_all(self, *, reason="requested"):
        had = [self.model_id] if self._engine is not None else []
        self._engine = None
        return had

    def _load(self, config):
        if config is None:
            config = _Config(n_ctx=self._override or 4096)
        self.loads.append(config.n_ctx)
        self.load_configs.append(config)
        if self.on_load is not None:
            self.on_load()
        if self._load_limit is not None and config.n_ctx > self._load_limit:
            raise RuntimeError(f"cannot allocate KV for n_ctx={config.n_ctx}")
        self._engine = _FakeEngine(config, self._prefill_limit, pool=self)


async def _run_to_completion(runner, model_id="m", target=131072):
    handle = runner.start(model_id, target)
    run = runner._runs[handle["bench_id"]]
    await run._task
    return runner.get(handle["bench_id"])


# --- start validation -------------------------------------------------------


def test_start_refuses_a_target_outside_the_whitelist(tmp_path):
    runner = ContextBenchRunner(_FakePool(tmp_path, prefill_limit=10_000))
    with pytest.raises(ContextBenchError) as exc:
        runner.start("m", 100_000)
    # The message must name the ladder: the client picks from it, and a bare
    # "invalid" leaves a user guessing which sizes exist.
    assert "16384" in str(exc.value)


@pytest.mark.parametrize("target", TARGETS)
def test_every_advertised_target_is_accepted(tmp_path, target):
    """The whitelist and the client's picker are the same ladder."""

    async def go():
        runner = ContextBenchRunner(
            _FakePool(tmp_path, prefill_limit=10_000), predictor=lambda _m: None
        )
        body = runner.start("m", target)
        assert body["target_tokens"] == target
        await runner._runs[body["bench_id"]]._task

    asyncio.run(go())


def test_a_second_start_is_refused_while_one_is_running(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=10_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        first = runner.start("m", 16384)
        with pytest.raises(ContextBenchError):
            runner.start("m", 16384)
        await runner._runs[first["bench_id"]]._task
        # ...and accepted once it finishes.
        runner.start("m", 16384)
        await runner._runs[runner.active]._task

    asyncio.run(go())


# --- what bounds the answer -------------------------------------------------


def test_a_machine_that_reaches_the_target_is_capped_by_the_target(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=16384)
        result = body["result"]
        assert body["status"] == "completed"
        assert result["capped_by"] == "target"
        assert result["measured_tokens"] == 16384
        return result

    asyncio.run(go())


def test_the_model_s_own_context_length_caps_the_search(tmp_path):
    """A 32k model asked for 128k is limited by the model, not the machine."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000, native=32768)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=131072)
        result = body["result"]
        assert result["capped_by"] == "native"
        assert result["native_context_length"] == 32768
        assert result["measured_tokens"] == 32768
        # Nothing above the model's own window was ever probed.
        assert max(pool.loads) <= 32768 + 1

    asyncio.run(go())


def test_memory_caps_the_answer_below_the_target(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=20_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=131072)
        result = body["result"]
        assert result["capped_by"] == "memory"
        # The boundary sits at the grid point just under the real limit.
        assert 20_000 - APPLY_GRANULARITY <= result["measured_tokens"] <= 20_000

    asyncio.run(go())


def test_an_admission_refusal_counts_the_same_as_running_out_of_memory(tmp_path):
    """A model that cannot be LOADED above a size still yields a boundary.

    This is the Metal shape: the allocation under test happens at load time,
    so the probe fails before any prompt is submitted.
    """

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000, load_limit=24_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=131072)
        result = body["result"]
        assert body["status"] == "completed"
        assert result["capped_by"] == "memory"
        assert 24_000 - APPLY_GRANULARITY <= result["measured_tokens"] <= 24_000

    asyncio.run(go())


def test_a_failed_probe_does_not_strand_the_size_and_fail_the_rest(tmp_path):
    """Regression: the reload condition must read the LIVE engine, not the
    size we last asked for.

    A failed load leaves no engine behind. If a smaller probe trusted the
    size that just failed it would reload at the impossible size again and
    fail forever, reporting a boundary of zero on a machine that manages 24k.
    """

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000, load_limit=24_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=131072)
        assert body["result"]["measured_tokens"] > 0

    asyncio.run(go())


def test_probe_engines_are_one_sequence_with_the_caches_off(tmp_path):
    """The serving geometry is eight sequences with both prefix caches on.

    A probe that inherited it would (a) be refused at n_ctx/8 tokens, since
    admission is per sequence, and come out eight times too small, and (b)
    hit its own cache on the verify step, since every probe sends the same
    text, and time no prefill at all. So every probe load is one sequence,
    no caches -- and the search reaches the target it should.
    """

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        result = (await _run_to_completion(runner, target=32768))["result"]
        assert result["measured_tokens"] == 32768
        assert pool.load_configs, "the search loaded nothing"
        for cfg in pool.load_configs:
            assert cfg.n_seq_max == 1
            assert cfg.prefix_cache is False
            assert cfg.mlx_prefix_cache is False
        # ...and the serving config was left alone.
        assert pool.config.n_seq_max == 8
        assert pool.config.prefix_cache is True

    asyncio.run(go())


def test_probing_never_writes_the_override_file(tmp_path):
    """Only a completed run that applies writes the file, once.

    Routing each probe's size through the persisted override meant a crash
    or a kill mid-search left the model configured at a trial size -- quite
    possibly the one the machine had just refused -- and unservable on the
    next boot until someone found and deleted the file.
    """

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=20_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        result = (await _run_to_completion(runner, target=131072))["result"]
        assert result["applied"] is True
        assert pool.overrides_written == [result["applied_tokens"]]

    asyncio.run(go())


def test_the_pool_is_in_maintenance_for_the_run_and_not_after(tmp_path):
    """No request may be served on a probe engine, and no concurrent load
    may perturb the memory being measured: the pool refuses loads for the
    duration, and the runner never goes through the gated path itself."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        await _run_to_completion(runner, target=16384)
        assert pool.seen_in_maintenance and all(pool.seen_in_maintenance)
        assert pool.maintenance is None
        assert pool.maintenance_log == ["begin", "end"]
        assert pool.acquires == 0
        # The last engine standing was a probe engine; it does not serve.
        assert pool.engine_for("m") is None

    asyncio.run(go())


def test_a_gguf_learns_its_native_length_after_the_first_load(tmp_path):
    """Only llama.cpp parses a GGUF's metadata, so the trained length is
    unknown until the model is resident -- and the run has just unloaded
    everything. Asking before any load capped nothing, and a 8k model asked
    for 128k had 128k applied to it."""

    async def go():
        pool = _FakePool(
            tmp_path, prefill_limit=1_000_000, native=8192, native_after_load=True
        )
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        result = (await _run_to_completion(runner, target=131072))["result"]
        assert result["native_context_length"] == 8192
        assert result["capped_by"] == "native"
        assert result["measured_tokens"] == 8192
        assert result["applied_tokens"] == 8192

    asyncio.run(go())


def test_the_limit_reason_names_what_refused_the_size(tmp_path):
    """A reader can tell "the engine said no" from "the machine ran out"."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=20_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        result = (await _run_to_completion(runner, target=131072))["result"]
        assert result["capped_by"] == "memory"
        assert "RuntimeError" in result["limit_reason"]
        assert "allocate" in result["limit_reason"]

        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        result = (await _run_to_completion(runner, target=16384))["result"]
        assert result["capped_by"] == "target"
        assert result["limit_reason"] is None

    asyncio.run(go())


def test_verification_runs_on_a_fresh_engine(tmp_path):
    """The boundary came out of a search whose failed probes each left the
    allocator in a different state; the verify step reloads even when the
    resident engine is big enough, so "held on a second run" means it."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: 16384)
        result = (await _run_to_completion(runner, target=16384))["result"]
        assert result["attempts"] == 2
        # The probe's load, then the verify's -- not one load reused.
        assert pool.loads == [16385, 16385]

    asyncio.run(go())


def test_a_machine_too_small_for_the_floor_reports_nothing_applied(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=FLOOR_TOKENS - 1)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=16384)
        result = body["result"]
        assert body["status"] == "completed"
        assert result["measured_tokens"] == 0
        assert result["applied"] is False
        assert result["applied_tokens"] == 0

    asyncio.run(go())


# --- applying ---------------------------------------------------------------


def test_the_applied_value_is_floored_to_the_grid_and_written_to_the_pool(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=16384)
        result = body["result"]
        assert result["applied"] is True
        assert result["applied_tokens"] % APPLY_GRANULARITY == 0
        assert result["applied_tokens"] <= result["measured_tokens"]
        # The value the pool will load with from now on is the applied one --
        # not the probing override the search left behind.
        assert pool.ctx_override("m") == result["applied_tokens"]

    asyncio.run(go())


def test_an_abandoned_run_leaves_the_previous_override_untouched(tmp_path):
    """A run that does not reach the apply step writes nothing at all."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=FLOOR_TOKENS - 1)
        pool.set_ctx_override("m", 8192)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        await _run_to_completion(runner, target=16384)
        assert pool.ctx_override("m") == 8192
        assert pool.overrides_written == [8192]  # the test's own write only

    asyncio.run(go())


def test_the_verified_prefill_reports_a_rate_and_the_real_token_count(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        result = (await _run_to_completion(runner, target=16384))["result"]
        assert result["verified_tokens"] == result["measured_tokens"]
        assert result["verified_prompt_tokens"] > 0
        assert result["prefill_tps"] and result["prefill_tps"] > 0

    asyncio.run(go())


# --- cancellation -----------------------------------------------------------


def test_cancel_stops_the_run_and_applies_nothing(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        handle = runner.start("m", 524288)
        assert runner.cancel(handle["bench_id"]) is True
        await runner._runs[handle["bench_id"]]._task
        body = runner.get(handle["bench_id"])
        assert body["status"] == "cancelled"
        # No result at all, rather than a half-filled one whose DEFAULTS read
        # as findings -- `capped_by` would say "memory" on a run that never
        # finished measuring, and the screen renders that as a conclusion.
        assert body["result"] is None
        assert pool.ctx_override("m") is None

    asyncio.run(go())


def test_cancel_during_a_probe_interrupts_it(tmp_path):
    """Cancel used to set a flag the search read between probes, so a cancel
    during a minutes-long probe waited the probe out -- and `start` refused
    a new run the whole time."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        pool.block_streams = asyncio.Event()  # every prefill hangs
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        handle = runner.start("m", 16384)
        task = runner._runs[handle["bench_id"]]._task
        await asyncio.wait_for(pool.streaming.wait(), timeout=1.0)  # in flight
        assert runner.cancel(handle["bench_id"]) is True
        await asyncio.wait_for(task, timeout=1.0)
        body = runner.get(handle["bench_id"])
        assert body["status"] == "cancelled"
        assert body["result"] is None
        assert pool.released, "the interrupted request was not released"
        assert pool.maintenance is None
        assert pool.engine_for("m") is None
        # ...and a new run is accepted at once.
        runner.start("m", 16384)
        runner.cancel(runner.active)
        await asyncio.wait_for(runner._runs[runner.active]._task, timeout=1.0)

    asyncio.run(go())


def test_shutdown_cancels_the_active_run_and_waits_for_its_cleanup(tmp_path):
    """Lifespan teardown: a task left pending at loop close never runs its
    `finally`, leaving the pool in maintenance with a probe engine resident."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        pool.block_streams = asyncio.Event()
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        handle = runner.start("m", 16384)
        await asyncio.wait_for(pool.streaming.wait(), timeout=1.0)
        await asyncio.wait_for(runner.shutdown(), timeout=1.0)
        assert runner.get(handle["bench_id"])["status"] == "cancelled"
        assert pool.maintenance is None
        assert runner.active is None
        # Idempotent, and a no-op with nothing running.
        await runner.shutdown()

    asyncio.run(go())


def test_an_unknown_model_is_a_start_error_not_a_crash(tmp_path):
    """`resolve` raising ModelPoolError out of `start` reached the route as
    a 500; it is the caller's mistake, the same as a bad target."""
    runner = ContextBenchRunner(_FakePool(tmp_path, prefill_limit=10_000))
    with pytest.raises(ContextBenchError) as exc:
        runner.start("typo", 16384)
    assert "unknown model" in str(exc.value)


def test_no_result_is_published_while_the_run_is_measuring(tmp_path):
    """A half-filled result has defaults that read as findings, and the
    clients copy whatever result a poll hands them. Nothing until a
    conclusion."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=20_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        seen: list[object] = []
        handle = runner.start("m", 131072)
        pool.on_load = lambda: seen.append(runner.get(handle["bench_id"])["result"])
        await runner._runs[handle["bench_id"]]._task
        assert len(seen) > 1
        assert all(r is None for r in seen)
        assert runner.get(handle["bench_id"])["result"] is not None

    asyncio.run(go())


def test_an_error_publishes_no_result(tmp_path):
    """The same for a run that dies: `status` and `error` say what happened;
    a partial result would say "limited by memory, measured 0" instead."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)

        def boom(_model_id):
            raise RuntimeError("host probe exploded")

        runner = ContextBenchRunner(pool, predictor=boom)
        body = await _run_to_completion(runner, target=16384)
        assert body["status"] == "error"
        assert "host probe exploded" in body["error"]
        assert body["result"] is None
        assert pool.maintenance is None

    asyncio.run(go())


def test_a_busy_pool_is_an_error_run_not_a_crash(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        pool.begin_maintenance("someone else")
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=16384)
        assert body["status"] == "error"
        assert "busy" in body["error"]
        assert runner.active is None
        assert pool.maintenance == "someone else"  # not ours to clear

    asyncio.run(go())


def test_active_summary_reports_the_running_run(tmp_path):
    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        pool.block_streams = asyncio.Event()
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        assert runner.active_summary() is None
        handle = runner.start("m", 32768)
        summary = runner.active_summary()
        assert summary == {
            "bench_id": handle["bench_id"], "model_id": "m",
            "target_tokens": 32768, "status": "running",
        }
        pool.block_streams.set()
        await runner._runs[handle["bench_id"]]._task
        assert runner.active_summary() is None

    asyncio.run(go())


def test_cancelling_an_unknown_or_finished_run_is_not_an_error(tmp_path):
    async def go():
        runner = ContextBenchRunner(
            _FakePool(tmp_path, prefill_limit=10_000), predictor=lambda _m: None
        )
        assert runner.cancel("nope") is False
        handle = runner.start("m", 16384)
        await runner._runs[handle["bench_id"]]._task
        assert runner.cancel(handle["bench_id"]) is False

    asyncio.run(go())


# --- the wire contract ------------------------------------------------------


def test_the_status_payload_always_carries_the_client_s_required_fields(tmp_path):
    """ContextBenchStatusResponse declares bench_id/status/phase/progress/
    message non-optional, and Swift discards the WHOLE response on one
    missing key -- a null `phase` blanks the progress card, it does not dim
    one label."""

    async def go():
        runner = ContextBenchRunner(
            _FakePool(tmp_path, prefill_limit=1_000_000), predictor=lambda _m: None
        )
        handle = runner.start("m", 16384)
        # Before any phase transition...
        early = runner.get(handle["bench_id"])
        await runner._runs[handle["bench_id"]]._task
        late = runner.get(handle["bench_id"])
        for body in (early, late):
            for key in ("bench_id", "status", "phase", "progress", "message"):
                assert body[key] is not None, key
            assert isinstance(body["progress"], float)
            assert 0.0 <= body["progress"] <= 100.0

    asyncio.run(go())


def test_the_result_payload_always_carries_the_client_s_required_fields(tmp_path):
    async def go():
        runner = ContextBenchRunner(
            _FakePool(tmp_path, prefill_limit=FLOOR_TOKENS - 1),
            predictor=lambda _m: None,
        )
        # The emptiest possible run: nothing measured, nothing applied.
        result = (await _run_to_completion(runner, target=16384))["result"]
        for key in (
            "model_id", "target_tokens", "measured_tokens", "verified_tokens",
            "applied_tokens", "applied", "capped_by", "attempts", "duration_s",
        ):
            assert result[key] is not None, key
        assert result["capped_by"] in ("memory", "target", "native")
        assert "limit_reason" in result  # optional on the client, always present

    asyncio.run(go())


def test_a_start_body_is_json_serialisable_as_the_client_decodes_it(tmp_path):
    async def go():
        runner = ContextBenchRunner(
            _FakePool(tmp_path, prefill_limit=10_000), predictor=lambda _m: None
        )
        body = runner.start("m", 16384)
        assert set(body) == {"bench_id", "status", "target_tokens"}
        json.dumps(body)
        await runner._runs[body["bench_id"]]._task

    asyncio.run(go())


# --- the prediction only chooses where to start -----------------------------


def test_a_pessimistic_prediction_does_not_become_the_answer(tmp_path):
    """The formula seeds the search; a real prefill decides it.

    A prediction well under what the machine manages must still climb: a
    benchmark that returned the estimate would be the estimate, and the
    whole point is to measure instead.
    """

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: 8192)
        result = (await _run_to_completion(runner, target=16384))["result"]
        assert result["measured_tokens"] == 16384
        assert result["capped_by"] == "target"
        # The estimate, then the climb to the ceiling, then the verification.
        # A search that bisected between the estimate and an UNPROBED ceiling
        # would stop one grid step short and never notice.
        assert result["attempts"] == 3

    asyncio.run(go())


def test_an_accurate_prediction_converges_in_few_probes(tmp_path):
    """A good estimate should cost one probe plus the verify, not a bisection
    from zero -- each probe is a real prefill."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: 16384)
        result = (await _run_to_completion(runner, target=16384))["result"]
        assert result["measured_tokens"] == 16384
        assert result["attempts"] == 2  # the probe, then the verification

    asyncio.run(go())
