"""Out-of-range sampling params must be refused, not leak a sequence slot.

`SamplerParams` is `uint32_t seed` / `int32_t top_k` in C++ (csrc/context.h), so a value
outside those ranges raises a TypeError out of pybind11 the moment
`RequestParams.to_sampler_params()` runs -- which is INSIDE `MetalEngine.add_request`,
after the seq_id has been popped off the free list and before the `RequestState` that
`_retire` needs in order to give it back. One such request used to burn one of the
`n_seq_max` concurrency slots permanently; `n_seq_max` of them wedged the server into
503-forever.

Both halves of that are asserted here:

  * behaviourally, `/health`'s `free_seq_slots` must come back unchanged and the NEXT
    request must still be served (a status-code assertion alone would not notice a leak);
  * at the engine level, `add_request` must be atomic for ANY exception in that window,
    not just for the two field bounds this file happens to know about.

`top_k = -1` is a positive control: it is a perfectly good int32 and llama.cpp's spelling
of "no top-k truncation", so it must keep returning 200.
"""

from __future__ import annotations

import os

import pytest

import bwr as bwr

MODEL_PATH = os.environ.get("BWR_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_TEST_MODEL to a .gguf path to run these",
)

# Two slots is enough to prove the DoS: with a leak, three bad requests exhaust the pool.
N_SEQ_MAX = 2


@pytest.fixture(scope="module")
def model() -> bwr.Model:
    m = bwr.Model(MODEL_PATH, bwr.ModelParams())
    yield m
    # Release the weights before interpreter exit or ggml's Metal device destructor
    # aborts the process (exit 134). See docs/llamacpp-notes.md.
    m.close()


@pytest.fixture(scope="module")
def client(model: bwr.Model):
    tc = pytest.importorskip("fastapi.testclient")
    from bwr.server.app import build_app

    app = build_app(
        model,
        bwr.EngineConfig(n_ctx=1024, n_batch=256, n_ubatch=256, n_seq_max=N_SEQ_MAX, engine="metal"),
    )
    with tc.TestClient(app) as c:
        yield c
    assert app.state.engine.engine.ctx.closed, "lifespan must close the context"


def _free_slots(client) -> int:
    r = client.get("/health")
    assert r.status_code == 200, r.text
    return r.json()["free_seq_slots"]


def _openai_body(**extra) -> dict:
    return {
        "model": "local",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4,
        **extra,
    }


def _anthropic_body(**extra) -> dict:
    return {
        "model": "local",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4,
        **extra,
    }


def _assert_both_surfaces_still_serve(client) -> None:
    """The real proof the DoS is closed: an ordinary request still gets served."""
    r = client.post("/v1/chat/completions", json=_openai_body(temperature=0))
    assert r.status_code == 200, f"/v1/chat/completions wedged: {r.status_code} {r.text}"
    r = client.post("/v1/messages", json=_anthropic_body(temperature=0))
    assert r.status_code == 200, f"/v1/messages wedged: {r.status_code} {r.text}"


# --- the protocol boundary: rejection, and no slot lost ------------------------------


@pytest.mark.parametrize("seed", [2**32, 2**64, -2, -(2**64)])
def test_out_of_range_seed_is_a_4xx_and_leaks_no_slot(client, seed: int) -> None:
    before = _free_slots(client)
    r = client.post("/v1/chat/completions", json=_openai_body(seed=seed))
    assert 400 <= r.status_code < 500, f"expected a 4xx, got {r.status_code}: {r.text}"
    assert "seed" in r.text, f"the error must name the field: {r.text}"
    assert _free_slots(client) == before, "a rejected request leaked a sequence slot"
    _assert_both_surfaces_still_serve(client)


@pytest.mark.parametrize("top_k", [2**31, 2**40, -(2**31) - 1])
def test_out_of_range_top_k_is_a_4xx_and_leaks_no_slot(client, top_k: int) -> None:
    before = _free_slots(client)
    r = client.post("/v1/chat/completions", json=_openai_body(top_k=top_k))
    assert 400 <= r.status_code < 500, f"expected a 4xx, got {r.status_code}: {r.text}"
    assert "top_k" in r.text, f"the error must name the field: {r.text}"
    assert _free_slots(client) == before, "a rejected request leaked a sequence slot"
    _assert_both_surfaces_still_serve(client)


@pytest.mark.parametrize("top_k", [2**31, 2**40, -(2**31) - 1])
def test_anthropic_out_of_range_top_k_is_a_4xx_and_leaks_no_slot(client, top_k: int) -> None:
    """/v1/messages shares the top_k vector, so it needs the same bound."""
    before = _free_slots(client)
    r = client.post("/v1/messages", json=_anthropic_body(top_k=top_k))
    assert 400 <= r.status_code < 500, f"expected a 4xx, got {r.status_code}: {r.text}"
    assert "top_k" in r.text, f"the error must name the field: {r.text}"
    assert _free_slots(client) == before, "a rejected request leaked a sequence slot"
    _assert_both_surfaces_still_serve(client)


def test_streaming_request_with_bad_params_leaks_no_slot(client) -> None:
    """The check must sit before admission on the streaming path too -- a 500 raised
    after the StreamingResponse was handed back would burn the slot the same way."""
    before = _free_slots(client)
    r = client.post("/v1/chat/completions", json=_openai_body(seed=2**32, stream=True))
    assert 400 <= r.status_code < 500, f"expected a 4xx, got {r.status_code}: {r.text}"
    assert _free_slots(client) == before
    _assert_both_surfaces_still_serve(client)


def test_repeated_bad_params_do_not_exhaust_the_pool(client) -> None:
    """More bad requests than there are slots, on both surfaces, then a good one."""
    before = _free_slots(client)
    assert before == N_SEQ_MAX
    for i in range(N_SEQ_MAX + 1):
        r = client.post("/v1/chat/completions", json=_openai_body(seed=2**32 + i))
        assert 400 <= r.status_code < 500, r.text
        r = client.post("/v1/messages", json=_anthropic_body(top_k=2**40 + i))
        assert 400 <= r.status_code < 500, r.text
    assert _free_slots(client) == before, "the sequence pool drained"
    _assert_both_surfaces_still_serve(client)


# --- positive controls: values that are in range must keep working -------------------


def test_top_k_minus_one_still_serves(client) -> None:
    """-1 is a valid int32 and llama.cpp's "disable top-k"; temperature > 0 so the
    sampler chain actually looks at it."""
    before = _free_slots(client)
    r = client.post("/v1/chat/completions", json=_openai_body(top_k=-1, temperature=0.7))
    assert r.status_code == 200, r.text
    r = client.post("/v1/messages", json=_anthropic_body(top_k=-1, temperature=0.7))
    assert r.status_code == 200, r.text
    assert _free_slots(client) == before


def test_seed_minus_one_means_random(client) -> None:
    """`seed: -1` is the llama.cpp/ollama "pick a seed for me" idiom, normalised at the
    boundary to LLAMA_DEFAULT_SEED rather than rejected."""
    before = _free_slots(client)
    r = client.post("/v1/chat/completions", json=_openai_body(seed=-1, temperature=0.7))
    assert r.status_code == 200, r.text
    assert _free_slots(client) == before


@pytest.mark.parametrize("seed", [0, 42, 2**32 - 1])
def test_in_range_seed_still_serves(client, seed: int) -> None:
    r = client.post("/v1/chat/completions", json=_openai_body(seed=seed, temperature=0.7))
    assert r.status_code == 200, r.text


# --- the root cause: add_request must be atomic --------------------------------------


class _Boom(Exception):
    pass


class _ExplodingParams(bwr.RequestParams):
    """Stands in for any future line added between popping the seq_id and registering
    the RequestState: the slot must come back whatever the exception is."""

    def to_sampler_params(self):  # type: ignore[no-untyped-def]
        raise _Boom("sampler construction failed")


@pytest.mark.parametrize(
    "params_factory, expected",
    [
        (lambda: bwr.RequestParams(seed=-1, max_tokens=4), TypeError),
        (lambda: bwr.RequestParams(top_k=2**40, max_tokens=4), TypeError),
        (lambda: _ExplodingParams(max_tokens=4), _Boom),
    ],
)
def test_add_request_is_atomic_on_failure(model: bwr.Model, params_factory, expected) -> None:
    eng = bwr.MetalEngine(
        model, bwr.EngineConfig(n_ctx=512, n_batch=256, n_ubatch=256, n_seq_max=N_SEQ_MAX)
    )
    try:
        before = eng.n_free_seq_slots
        # More failures than there are slots: a leak of one per call would raise
        # SeqIdExhausted instead of the expected error on the third iteration.
        for _ in range(before + 2):
            with pytest.raises(expected):
                eng.add_request("Hi", params_factory())
            assert eng.n_free_seq_slots == before, "add_request leaked a seq_id"
        # The pool is still usable afterwards.
        eng.add_request("Hi", bwr.RequestParams(max_tokens=4))
        assert eng.n_free_seq_slots == before - 1
    finally:
        eng.ctx.close()
