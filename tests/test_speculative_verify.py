"""Speculative verify path (SPEC-speculative-ngram.md, T3).

Needs FTM_TEST_MODEL like the other engine tests; skips without it.

The prompt is a long periodic digit run: its bigrams/trigrams seed the draft
table at admission, and a greedy continuation of a counting run repeats them,
so speculation actually engages (``spec_drafted > 0`` below is the tripwire
that fails if the prompt ever stops triggering drafts -- without it the
``<=`` comparisons would pass vacuously on the plain path).
"""

from __future__ import annotations

import os

import pytest

import freetoken_mac as ftm
from freetoken_mac.engine import EngineConfig, MetalEngine, RequestParams

MODEL_PATH = os.environ.get("FTM_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_TEST_MODEL to a .gguf path to run speculation tests",
)

# Periodic at the token level (single digits tokenise singly): the table seeded
# from this predicts the continuation vindicated, and greedy decoding of a
# counting run keeps emitting the same n-grams.
PROMPT = "Count: " + " ".join(str(i % 10) for i in range(48))
N_TOKENS = 16


def greedy(max_tokens: int = N_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False)


@pytest.fixture(scope="module")
def model() -> ftm.Model:
    return ftm.Model(MODEL_PATH, ftm.ModelParams())


def run(model: ftm.Model, speculative: bool) -> MetalEngine:
    config = EngineConfig(n_ctx=512, n_seq_max=1, speculative=speculative)
    engine = MetalEngine(model, config)
    engine.add_request(PROMPT, greedy())
    list(engine.drain())
    return engine


def test_spec_on_off_byte_identical(model: ftm.Model) -> None:
    """The invariant: speculation must be byte-identical to the plain path."""
    plain = run(model, False)
    spec = run(model, True)
    plain_id = plain._next_request_id - 1
    spec_id = spec._next_request_id - 1
    assert spec.tokens_of(spec_id) == plain.tokens_of(plain_id)
    assert spec.state(spec_id).finish_reason == plain.state(plain_id).finish_reason


def test_speculative_never_uses_more_decodes(model: ftm.Model) -> None:
    """Every verify step accepts >= 1 new token (a mismatch still yields the
    sampled token), so speculation cannot need more decodes for the same
    tokens. ``spec_drafted > 0`` proves drafts actually packed -- without it
    this comparison would pass on the plain path and prove nothing."""
    plain = run(model, False)
    spec = run(model, True)
    assert spec.spec_drafted > 0, "prompt did not trigger any drafts"
    assert spec.ctx.decode_calls <= plain.ctx.decode_calls
    rate = spec.spec_acceptance_rate
    assert rate is not None and 0.0 <= rate <= 1.0


def test_nongreedy_request_ignores_speculation(model: ftm.Model) -> None:
    """temp > 0 takes the plain path even with the flag on: no drafts packed,
    request still completes normally."""
    config = EngineConfig(n_ctx=512, n_seq_max=1, speculative=True)
    engine = MetalEngine(model, config)
    rid = engine.add_request(
        PROMPT, RequestParams(temp=1.0, max_tokens=8, stop_at_eog=False)
    )
    list(engine.drain())
    assert engine.spec_drafted == 0
    assert engine.spec_acceptance_rate is None
    assert len(engine.tokens_of(rid)) == 8
    assert engine.state(rid).finish_reason == "length"


def test_speculation_defaults_off(model: ftm.Model) -> None:
    """A default engine never drafts and reports no acceptance rate."""
    assert EngineConfig().speculative is False
    assert EngineConfig().spec_max_drafts == 4
    engine = run(model, False)
    assert engine.spec_drafted == 0
    assert engine.spec_accepted == 0
    assert engine.spec_acceptance_rate is None


def test_negative_spec_max_drafts_rejected(model: ftm.Model) -> None:
    """Fail fast at construction, before allocating a context."""
    with pytest.raises(ValueError, match="spec_max_drafts"):
        MetalEngine(model, EngineConfig(spec_max_drafts=-1))
