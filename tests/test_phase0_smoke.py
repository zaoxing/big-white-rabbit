"""Phase 0 bring-up tests: the binding loads a GGUF on Metal and generates from it.

Point BWR_TEST_MODEL at a .gguf to run these; they skip otherwise so the suite stays
runnable on a machine with no models checked out.
"""

from __future__ import annotations

import os

import pytest

import bwr as bwr

MODEL_PATH = os.environ.get("BWR_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_TEST_MODEL to a .gguf path to run bring-up tests",
)


@pytest.fixture(scope="module")
def model() -> bwr.Model:
    return bwr.Model(MODEL_PATH, bwr.ModelParams())


def test_metadata_is_populated(model: bwr.Model) -> None:
    assert model.n_layer > 0
    assert model.n_embd > 0
    assert model.n_vocab > 0
    assert model.n_ctx_train > 0
    assert model.n_params > 0
    assert model.desc
    # Every GGUF carries this key; its absence means we misread the metadata table.
    assert model.meta_val("general.architecture")


def test_metadata_only_load_skips_allocation() -> None:
    params = bwr.ModelParams()
    params.no_alloc = True
    meta_only = bwr.Model(MODEL_PATH, params)
    assert meta_only.n_layer > 0
    assert meta_only.size_bytes > 0


def test_tokenize_roundtrips(model: bwr.Model) -> None:
    text = "The quick brown fox"
    tokens = model.tokenize(text, add_special=False, parse_special=False)
    assert tokens
    assert model.detokenize(tokens) == text


def test_generation_produces_text(model: bwr.Model) -> None:
    ctx_params = bwr.ContextParams()
    ctx_params.n_ctx = 512
    ctx = bwr.Context(model, ctx_params, bwr.SamplerParams())

    pieces = list(bwr.generate(model, ctx, "The capital of France is", max_tokens=8))
    assert pieces, "generation yielded nothing"
    assert "".join(pieces).strip(), "generation yielded only whitespace"


def test_greedy_is_deterministic(model: bwr.Model) -> None:
    def run() -> str:
        ctx_params = bwr.ContextParams()
        ctx_params.n_ctx = 512
        ctx = bwr.Context(model, ctx_params, bwr.SamplerParams())  # temp=0 -> greedy
        return "".join(bwr.generate(model, ctx, "Count: 1 2 3", max_tokens=8))

    assert run() == run()


def test_effective_geometry_differs_from_request(model: bwr.Model) -> None:
    """llama.cpp rounds n_ctx UP (KV padding) and clamps n_batch DOWN to the requested
    n_ctx. Budgeting against the requested values instead of these silently walks into
    the n_batch assert below, so this pins the behaviour down."""
    ctx_params = bwr.ContextParams()
    ctx_params.n_ctx = 32
    ctx_params.n_batch = 512
    ctx = bwr.Context(model, ctx_params, bwr.SamplerParams())

    assert ctx.n_ctx >= 32
    assert ctx.n_batch <= ctx.n_ctx, "n_batch should be clamped to the context"


def test_oversized_batch_raises_instead_of_aborting(model: bwr.Model) -> None:
    """llama_decode asserts n_tokens <= n_batch, and a failed GGML_ASSERT calls abort().
    The binding must bounds-check first, or one bad request kills the whole process
    (fatal in the single-process server design)."""
    ctx_params = bwr.ContextParams()
    ctx_params.n_ctx = 32
    ctx = bwr.Context(model, ctx_params, bwr.SamplerParams())

    too_many = [1] * (ctx.n_batch + 1)
    with pytest.raises(ValueError, match="exceeds n_batch"):
        ctx.decode_seq0(too_many)


def test_prefill_longer_than_n_batch_is_chunked(model: bwr.Model) -> None:
    ctx_params = bwr.ContextParams()
    ctx_params.n_ctx = 1024
    ctx_params.n_batch = 64  # force multiple prefill chunks
    ctx = bwr.Context(model, ctx_params, bwr.SamplerParams())

    prompt = "word " * 100  # comfortably more tokens than n_batch=64
    assert len(model.tokenize(prompt)) > ctx.n_batch
    pieces = list(bwr.generate(model, ctx, prompt, max_tokens=4))
    assert pieces


def test_prompt_longer_than_context_is_rejected(model: bwr.Model) -> None:
    ctx_params = bwr.ContextParams()
    ctx_params.n_ctx = 256
    ctx = bwr.Context(model, ctx_params, bwr.SamplerParams())

    huge = "word " * (ctx.n_ctx + 100)
    with pytest.raises(ValueError, match="context is"):
        list(bwr.generate(model, ctx, huge, max_tokens=4))
