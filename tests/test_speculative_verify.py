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
    assert EngineConfig().engine == "mlx"
    engine = run(model, False)
    assert engine.spec_drafted == 0
    assert engine.spec_accepted == 0
    assert engine.spec_acceptance_rate is None


def test_negative_spec_max_drafts_rejected(model: ftm.Model) -> None:
    """Fail fast at construction, before allocating a context."""
    with pytest.raises(ValueError, match="spec_max_drafts"):
        MetalEngine(model, EngineConfig(spec_max_drafts=-1))


class _WrongTable:
    """A draft table whose every prediction is wrong on purpose.

    Built from a plain run's true trajectory: each observed context maps to
    (true_next + 1) % vocab, so no draft can ever match. Every verify step
    then takes the mismatch path -- rewind plus continue -- which is exactly
    the path a 100%-acceptance prompt never exercises (and where the
    inclusive-rewind bug hid).
    """

    order = 3

    def __init__(self, wrong: dict[tuple[int, ...], int]) -> None:
        self._wrong = wrong

    def update_stream(self, tokens) -> None:
        pass

    def predict(self, context, max_tokens: int) -> list[int]:
        token = self._wrong.get(tuple(context[-(self.order - 1) :]))
        return [token] * max_tokens if token is not None else []


def test_all_wrong_drafts_still_identical(
    model: ftm.Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Maximum adversity: drafts packed every step, zero accepted. Output must
    still equal the plain run, decode-for-decode (each step accepts exactly the
    one mismatched sample, i.e. the plain path's token)."""
    plain = run(model, False)
    plain_id = plain._next_request_id - 1
    expected = plain.tokens_of(plain_id)
    prompt_tokens = list(
        model.tokenize(PROMPT, add_special=True, parse_special=True)
    )
    hist = prompt_tokens + expected
    vocab = model.n_vocab
    wrong = {
        tuple(hist[max(0, i - 2) : i]): (tok + 1) % vocab
        for i, tok in enumerate(hist)
    }

    from freetoken_mac.engine import metal_engine as engine_module

    monkeypatch.setattr(
        engine_module, "NgramTable", lambda *args, **kwargs: _WrongTable(wrong)
    )
    config = EngineConfig(n_ctx=512, n_seq_max=1, speculative=True)
    engine = MetalEngine(model, config)
    rid = engine.add_request(PROMPT, greedy())
    list(engine.drain())

    assert engine.tokens_of(rid) == expected
    assert engine.state(rid).finish_reason == plain.state(plain_id).finish_reason
    assert engine.spec_drafted > 0, "scripted drafts did not pack"
    # Every step took the mismatch path (or near enough that at least one
    # rewind ran): fewer acceptances than drafts, and never more decodes.
    assert engine.spec_accepted < engine.spec_drafted
    assert engine.spec_acceptance_rate is not None
    assert engine.ctx.decode_calls <= plain.ctx.decode_calls


def test_speculative_engine_on_attention_needs_no_snapshots(
    model: ftm.Model,
) -> None:
    """The attention test model passes the capability gate either way, and its
    readback is 0: llama.cpp clamps n_rs_seq on architectures without
    recurrent state (snapshots only exist for recurrent memory), while partial
    removes on attention KV always succeed. Hybrid readback is covered by the
    27B validation runs (SPEC T5a evidence), not the fast suite."""
    spec = MetalEngine(model, EngineConfig(n_ctx=512, n_seq_max=1, speculative=True))
    assert spec.ctx.n_rs_seq == 0
    plain = MetalEngine(model, EngineConfig(n_ctx=512, n_seq_max=1))
    assert plain.ctx.n_rs_seq == 0
