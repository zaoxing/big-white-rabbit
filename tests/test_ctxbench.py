"""Context benchmark: the search, what bounds it, and what it writes.

The fake pool below is a whole machine in miniature: a prefill budget the
engine refuses to exceed, an admission cap that only changes on reload, and
engines that come and go. That is deliberate -- driving the REAL `_prefill`
through a fake engine exercises the reload/admission interplay, which is
where the interesting bugs live, rather than stubbing it out and testing the
bisection arithmetic against itself.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from bwr.server.ctxbench import (
    APPLY_GRANULARITY,
    FLOOR_TOKENS,
    TARGETS,
    ContextBenchError,
    ContextBenchRunner,
)

# Matches `_FILLER` ("word ") so a prompt of N tokens is 5N characters.
_CHARS_PER_TOKEN = 5


class _Tokenizer:
    def tokenize(self, text, add_special=False, parse_special=False):
        return [0] * (len(text) // _CHARS_PER_TOKEN)


@dataclass
class _Config:
    n_ctx: int


class _Output:
    finished = True
    piece = "x"
    finish_reason = "length"


class _FakeEngine(_Tokenizer):
    """An engine that admits up to `n_ctx` and prefills up to `prefill_limit`.

    The two limits are separate on purpose: admission is a configured cap
    (ValueError, the same shape MLXEngine raises) while the prefill limit is
    the machine running out (RuntimeError). The benchmark must treat both as
    "this size does not work", and the tests below check it does.
    """

    def __init__(self, n_ctx: int, prefill_limit: int) -> None:
        self.config = _Config(n_ctx=n_ctx)
        self.ctx = _Config(n_ctx=n_ctx)
        self._prefill_limit = prefill_limit
        self.submitted: list[int] = []

    async def submit(self, prompt, params):
        n = len(self.tokenize(prompt))
        self.submitted.append(n)
        if n >= self.config.n_ctx:
            raise ValueError(f"prompt is {n} tokens, capacity is {self.config.n_ctx}")
        if n > self._prefill_limit:
            raise RuntimeError("metal: failed to allocate")
        return 1

    async def stream(self, rid):
        yield _Output()

    def release(self, rid) -> None:
        pass


class _FakePool:
    """Enough ModelPool to drive the runner, with a real override file."""

    def __init__(
        self,
        model_dir,
        *,
        prefill_limit: int,
        native: int | None = None,
        size_bytes: int = 4 * 1024**3,
        load_limit: int | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.model_id = "m"
        self._prefill_limit = prefill_limit
        self._native = native
        self._size_bytes = size_bytes
        # A ctx above this cannot be LOADED at all -- llama.cpp's behaviour,
        # where the KV cache is allocated when the context is created.
        self._load_limit = load_limit
        self._override: int | None = None
        self._engine: _FakeEngine | None = None
        self.loads: list[int] = []
        self.overrides_written: list[int | None] = []

    # -- the slice of ModelPool the runner uses ---------------------------

    def resolve(self, model_id):
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
        return self._native

    async def acquire(self, model_id):
        if self._engine is None:
            self._load()
        return self.model_id, self._engine

    async def reload(self, model_id):
        self._engine = None
        self._load()
        return self.model_id, self._engine

    async def unload(self, model_id):
        self._engine = None
        return True

    def _load(self):
        n_ctx = self._override or 4096
        self.loads.append(n_ctx)
        if self._load_limit is not None and n_ctx > self._load_limit:
            raise RuntimeError(f"cannot allocate KV for n_ctx={n_ctx}")
        self._engine = _FakeEngine(n_ctx, self._prefill_limit)


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


def test_a_failed_probe_does_not_strand_the_override_and_fail_the_rest(tmp_path):
    """Regression: the reload condition must read the LIVE engine, not the
    override we last wrote.

    A failed load leaves the override naming a size the machine refused. If a
    smaller probe trusted that value it would reload at the impossible size
    again and fail forever, reporting a boundary of zero on a machine that
    manages 24k.
    """

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=1_000_000, load_limit=24_000)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        body = await _run_to_completion(runner, target=131072)
        assert body["result"]["measured_tokens"] > 0

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


def test_an_abandoned_run_restores_the_previous_override(tmp_path):
    """Probing rewrites the override once per probe. A run that does not
    reach the apply step must not leave the last size tested behind."""

    async def go():
        pool = _FakePool(tmp_path, prefill_limit=FLOOR_TOKENS - 1)
        pool.set_ctx_override("m", 8192)
        runner = ContextBenchRunner(pool, predictor=lambda _m: None)
        await _run_to_completion(runner, target=16384)
        assert pool.ctx_override("m") == 8192

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
