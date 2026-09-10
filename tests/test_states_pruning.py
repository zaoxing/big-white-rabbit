"""A finished request must not be retained for the life of the process.

`MetalEngine._states` had nothing that ever removed an entry. Verified live on the
running server before this file existed: `_states` held 14 entries after 14 served
requests, and grew 7 -> 10 across a later battery. Two separate costs:

  * memory -- every finished request kept its full `prompt` token list and its
    `output_tokens` forever (a 4k-token prompt is ~150 KB of Python ints);
  * per-token CPU -- `step()` scans the state map three times per decode (the
    `is_prefilling` / `is_generating` comprehensions plus `has_work`), so the cost of
    ONE decode grew with the total number of requests ever served rather than with the
    number in flight. Throughput decayed linearly with uptime.

The fix splits the two jobs the map was doing. `_states` is now the HOT PATH and holds
in-flight requests only, so every per-decode scan is bounded by `n_seq_max`. Retired
requests move into a bounded ring (`n_retained_finished`, capped at
`retained_finished_limit`) that keeps the read accessors -- `state`, `tokens_of`,
`text_of` -- working for a request the caller can still legitimately ask about. That
retention window is the thing this file has to pin from both sides: unbounded retention
is the bug, and NO retention would be a regression, because callers read `tokens_of` and
`state(...).finish_reason` AFTER the request has finished (`test_phase1_batching.py`,
`test_slot_release_on_abnormal_exit.py`, `test_stop_sequences.py` all do).

The first test below is therefore the regression guard, not the leak test.
"""

from __future__ import annotations

import os

import pytest

import freetoken_mac as ftm

MODEL_PATH = os.environ.get("FTM_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_TEST_MODEL to a .gguf path to run these",
)

# Enough requests that a per-request leak is unmistakable: any bound this file asserts
# is far below it, so a map that still grows one entry per request cannot pass.
MANY = 40


@pytest.fixture(scope="module")
def model() -> ftm.Model:
    m = ftm.Model(MODEL_PATH, ftm.ModelParams())
    yield m
    # Release the weights before interpreter exit or ggml's Metal device destructor
    # aborts the process (exit 134). See docs/llamacpp-notes.md.
    m.close()


def _fresh_engine(model: ftm.Model, n_seq_max: int = 2) -> ftm.MetalEngine:
    return ftm.MetalEngine(
        model,
        ftm.EngineConfig(n_ctx=1024, n_batch=256, n_ubatch=256, n_seq_max=n_seq_max),
    )


def _greedy(max_tokens: int = 2) -> ftm.RequestParams:
    return ftm.RequestParams(temp=0.0, max_tokens=max_tokens)


# --- the regression guard: post-completion reads must keep working -------------------


def test_accessors_still_work_for_a_just_finished_request(model: ftm.Model) -> None:
    """Written first because it is the way a "fix" would break the product.

    Pruning on retire is the cheapest possible answer and it is wrong: the engine's read
    accessors are called after the request has ended, so a `tokens_of` that raises for a
    request that finished one moment ago is a worse defect than the leak.
    """
    engine = _fresh_engine(model)
    try:
        rid = engine.add_request("The capital of France is", _greedy(4))
        pieces = [o.piece for o in engine.drain()]

        state = engine.state(rid)
        assert state.finished is True
        assert state.finish_reason == "length"
        assert state.n_generated == 4
        assert state.request_id == rid

        tokens = engine.tokens_of(rid)
        assert len(tokens) == 4, "tokens_of lost a finished request's output"
        assert tokens == list(state.output_tokens)
        # text_of has to keep detokenising, i.e. the state it reads is still intact.
        assert engine.text_of(rid) == model.detokenize(tokens)
        assert engine.text_of(rid) == "".join(pieces)
    finally:
        engine.ctx.close()


def test_accessors_still_work_for_every_request_of_a_full_batch(model: ftm.Model) -> None:
    """The concurrent shape of the same guard: when several requests share the decodes,
    each one is read after it ends while its peers may still be running, so the window
    must cover a whole batch's worth of retirements, not just the last one."""
    n_seq_max = 4
    engine = _fresh_engine(model, n_seq_max=n_seq_max)
    try:
        prompts = ["Count: 1 2", "The capital of France is", "Red, green,", "One two"]
        ids = [engine.add_request(p, _greedy(3)) for p in prompts]
        list(engine.drain())
        assert len(ids) == n_seq_max
        for rid in ids:
            assert engine.state(rid).finish_reason == "length"
            assert len(engine.tokens_of(rid)) == 3
            assert engine.text_of(rid) == model.detokenize(engine.tokens_of(rid))
    finally:
        engine.ctx.close()


def test_the_retention_window_covers_at_least_a_full_batch(model: ftm.Model) -> None:
    """The window is a promise, not an accident: it must be at least as wide as the
    number of requests that can be in flight at once, or the guard above only passes by
    luck of the default."""
    for n_seq_max in (1, 2, 4):
        engine = _fresh_engine(model, n_seq_max=n_seq_max)
        try:
            assert engine.retained_finished_limit >= engine.ctx.n_seq_max
            assert engine.retained_finished_limit >= 1
        finally:
            engine.ctx.close()


# --- the leak itself ----------------------------------------------------------------


def test_serving_many_requests_does_not_leave_one_state_each(model: ftm.Model) -> None:
    """The statement of the bug. 40 served requests, and the engine must not be holding
    40 states -- neither on the hot path nor in the archive."""
    engine = _fresh_engine(model)
    try:
        limit = engine.retained_finished_limit
        assert limit < MANY, "the bound has to be below MANY or this proves nothing"

        for _ in range(MANY):
            rid = engine.add_request("Count: 1 2", _greedy(2))
            list(engine.drain())
            assert engine.state(rid).finish_reason == "length"

        assert engine.n_in_flight == 0, "nothing is running, so nothing may be in flight"
        assert len(engine._states) == 0, (
            f"the hot-path map kept {len(engine._states)} entries after {MANY} finished "
            "requests"
        )
        assert engine.n_retained_finished <= limit, (
            f"the archive grew to {engine.n_retained_finished}, past its {limit} bound"
        )
        assert engine.n_free_seq_slots == engine.ctx.n_seq_max
    finally:
        engine.ctx.close()


def test_a_pruned_request_raises_keyerror_and_says_so(model: ftm.Model) -> None:
    """The other end of the policy, stated so it cannot drift: once a request has fallen
    out of the window it is GONE, and asking about it raises KeyError -- the same failure
    an id that never existed gets, since a caller cannot act on either. Silently
    returning an empty token list would be indistinguishable from a request that
    generated nothing."""
    engine = _fresh_engine(model)
    try:
        first = engine.add_request("Count: 1 2", _greedy(2))
        list(engine.drain())
        assert engine.tokens_of(first) != [], "precondition: it produced output"

        for _ in range(engine.retained_finished_limit + 4):
            engine.add_request("Count: 1 2", _greedy(2))
            list(engine.drain())

        with pytest.raises(KeyError, match="no longer retained"):
            engine.state(first)
        with pytest.raises(KeyError, match="no longer retained"):
            engine.tokens_of(first)
        with pytest.raises(KeyError, match="no longer retained"):
            engine.text_of(first)
        # An id that was never admitted is a different mistake and says a different thing.
        with pytest.raises(KeyError, match="unknown"):
            engine.state(9_999_999)
    finally:
        engine.ctx.close()


def test_per_decode_scan_cost_is_independent_of_requests_served(model: ftm.Model) -> None:
    """The CPU half of the bug, asserted structurally rather than on the clock.

    `step()` scans the hot-path map twice and `has_work` once, so what has to be bounded
    is the SIZE OF THAT COLLECTION at every decode -- by concurrency, never by history.
    A wall-clock assertion would say the same thing far less reliably.
    """
    n_seq_max = 2
    engine = _fresh_engine(model, n_seq_max=n_seq_max)
    try:
        widths: list[int] = []
        served = 0
        for _ in range(MANY // n_seq_max):
            ids = [engine.add_request("Count: 1 2", _greedy(3)) for _ in range(n_seq_max)]
            served += len(ids)
            while engine.has_work:
                widths.append(len(engine._states))
                engine.step()
            widths.append(len(engine._states))
            for rid in ids:
                assert engine.state(rid).finished is True

        assert served == MANY
        assert max(widths) <= n_seq_max, (
            f"a decode scanned {max(widths)} states with only {n_seq_max} slots: the "
            "scan is sized by history, not by concurrency"
        )
        # And the tail is not merely equal to the head by accident -- it is the same.
        assert widths[-1] == 0
        assert max(widths[len(widths) // 2 :]) == max(widths[: len(widths) // 2])
    finally:
        engine.ctx.close()


# --- the stop-filter map has the same lifetime, including on the failure path --------


class _FailingSampleCtx:
    """The live context with `sample_seq` rigged to fail once.

    A proxy rather than a monkeypatch because `Context` is a C++ extension type whose
    attributes cannot be reassigned. Everything else -- decode, the geometry the step
    budget reads, the KV and sampler calls `_retire` makes -- delegates to the real
    context, so the failure is genuinely mid-`_advance` and nothing else is simulated.
    """

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self.n_sampled = 0

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def sample_seq(self, seq_id: int, row: int) -> int:
        self.n_sampled += 1
        raise RuntimeError("injected sample_seq failure")


def test_a_failed_advance_retains_no_stop_filter(model: ftm.Model) -> None:
    """An exception inside `_advance` used to land BEFORE `_retire`, leaving the
    request's `StopSequenceFilter` in `_stop_filters` (verified: `_states=1
    _stop_filters=1 finished=False`, cleared only by a later `cancel()`). The filter's
    lifetime is a strict subset of the state's, so the same policy has to cover it: the
    request is retired on the way out, which drops both maps and returns the slot.
    """
    engine = _fresh_engine(model)
    real_ctx = engine.ctx
    try:
        engine.ctx = _FailingSampleCtx(real_ctx)
        rid = engine.add_request(
            "Count: 1 2", ftm.RequestParams(temp=0.0, max_tokens=8, stop=("STOP",))
        )
        assert engine._stop_filters.get(rid) is not None, "precondition: a filter exists"
        free_before = engine.n_free_seq_slots

        with pytest.raises(RuntimeError, match="injected sample_seq failure"):
            engine.step()
        assert engine.ctx.n_sampled == 1, "the failure must be inside _advance"

        assert engine._stop_filters == {}, (
            f"a failed _advance kept {len(engine._stop_filters)} stop filter(s)"
        )
        assert len(engine._states) == 0, "a failed request stayed on the hot path"
        assert set(engine._stop_filters) <= set(engine._states), (
            "the invariant: a stop filter may never outlive its request's state"
        )
        # has_work must go false, or the worker thread re-runs the same failing step
        # forever -- the exact bug 097b84f fixed for the abandoned-stream case.
        assert engine.has_work is False
        assert engine.n_free_seq_slots == free_before + 1, "the seq_id was not returned"
        # Still readable, and honest about why it ended.
        assert engine.state(rid).finished is True
        assert engine.state(rid).finish_reason == "error"
    finally:
        engine.ctx = real_ctx
        real_ctx.close()


def test_a_normal_stop_sequence_request_retains_no_filter(model: ftm.Model) -> None:
    """The non-failure half: the map that holds the scanners is in-flight-only too."""
    engine = _fresh_engine(model)
    try:
        rid = engine.add_request(
            "Count: 1 2", ftm.RequestParams(temp=0.0, max_tokens=3, stop=("ZZZZ",))
        )
        assert rid in engine._stop_filters
        list(engine.drain())
        assert engine._stop_filters == {}
        assert len(engine.tokens_of(rid)) == 3
    finally:
        engine.ctx.close()


def test_stop_filter_treats_a_bare_string_as_one_sequence() -> None:
    """Latent, and one line to close: `tuple("abc")` is `('a','b','c')`, so a direct
    engine caller passing a bare string got a per-CHARACTER stop set and a turn that
    ended at the first 'a'. Unreachable from the wire (both routes call
    `normalize_stops` first), which is exactly why it needed pinning here."""
    from freetoken_mac.engine.config import StopSequenceFilter

    f = StopSequenceFilter("abc")
    assert f.enabled is True
    # "a" alone must not STOP it. It is briefly withheld -- a trailing "a" could still
    # become "abc" -- and released once the next piece rules that out, so the text comes
    # out whole and generation continues. Under the exploded reading the first push
    # returned ("x", True) and the turn was over.
    first, hit1 = f.push("xa")
    second, hit2 = f.push("bd")
    assert (hit1, hit2) == (False, False)
    assert first + second == "xabd"
    # The whole string is the delimiter.
    g = StopSequenceFilter("abc")
    emitted, hit = g.push("zzabc!")
    assert (emitted, hit) == ("zz", True)
    assert g.matched == "abc"
    # The empty string is still not a stop sequence (it sits at offset 0 of everything).
    assert StopSequenceFilter("").enabled is False


# --- both HTTP surfaces, after many requests ----------------------------------------


@pytest.fixture(scope="module")
def served(model: ftm.Model):
    tc = pytest.importorskip("fastapi.testclient")
    from freetoken_mac.server.app import build_app

    app = build_app(
        model, ftm.EngineConfig(n_ctx=2048, n_batch=256, n_ubatch=256, n_seq_max=4)
    )
    with tc.TestClient(app) as c:
        yield app, c
    assert app.state.engine.engine.ctx.closed, "lifespan must close the context"


def _payload(surface: str, *, stream: bool) -> tuple[str, dict]:
    body = {
        "model": "local",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "Count: 1 2 3"}],
        "stream": stream,
    }
    return ("/v1/chat/completions" if surface == "openai" else "/v1/messages"), body


def test_both_surfaces_serve_correctly_after_many_requests(served) -> None:
    """The shape the leak was actually observed in, and the wire behaviour that must
    survive the fix: request N of a long-lived server answers exactly like request 1,
    and the engine is not holding a state per request when they are all done."""
    app, client = served
    engine = app.state.engine.engine

    first: dict[str, str] = {}
    for i in range(MANY):
        surface = "openai" if i % 2 == 0 else "anthropic"
        url, body = _payload(surface, stream=(i % 4 == 3))
        r = client.post(url, json=body)
        assert r.status_code == 200, f"request {i} on {surface} failed: {r.text}"
        text = _text_of(surface, r)
        assert text != ""
        # Same seeded, greedy request every time: the answer may not drift as the
        # engine's bookkeeping is pruned underneath it.
        assert first.setdefault(surface, text) == text, (
            f"{surface} request {i} answered differently from request 1"
        )

    assert engine.n_in_flight == 0
    assert len(engine._states) == 0, (
        f"{MANY} served requests left {len(engine._states)} states on the hot path"
    )
    assert engine.n_retained_finished <= engine.retained_finished_limit
    assert engine.n_free_seq_slots == engine.ctx.n_seq_max
    assert client.get("/health").json()["free_seq_slots"] == engine.ctx.n_seq_max


def _text_of(surface: str, response) -> str:
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return _text_of_stream(surface, response.text)
    body = response.json()
    if surface == "openai":
        return body["choices"][0]["message"]["content"] or ""
    return "".join(b["text"] for b in body["content"] if b["type"] == "text")


def _text_of_stream(surface: str, raw: str) -> str:
    import json

    out = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            break
        event = json.loads(data)
        if surface == "openai":
            out.append(event["choices"][0]["delta"].get("content") or "")
        elif event.get("type") == "content_block_delta":
            out.append(event["delta"].get("text") or "")
    return "".join(out)
