"""Draft-model speculation (SPEC-draft-model.md, T7a/T7b).

Mechanics tests run on FTM_TEST_MODEL (any arch: acceptance never depends on
the pairing, only the rate does). Integration uses a second instance of the
SAME file as the draft -- vocab-identical by construction -- so the full
propose/verify/sync loop runs fast. The 27B<-4B pairing is economics only
(manual bench), not correctness.
"""

from __future__ import annotations

import os

import pytest

import freetoken_mac as ftm
from freetoken_mac.engine.draft import DraftEngine

MODEL_PATH = os.environ.get("FTM_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_TEST_MODEL to a .gguf path to run draft tests",
)

# Long enough to span several prefill chunks at n_batch=64 below.
PROMPT = "word " * 100


@pytest.fixture(scope="module")
def model() -> ftm.Model:
    return ftm.Model(MODEL_PATH, ftm.ModelParams())


def make_draft(model: ftm.Model) -> DraftEngine:
    return DraftEngine(model, n_ctx=512, n_seq_max=2, n_batch=64)


def prompt_tokens(model: ftm.Model) -> list[int]:
    return list(model.tokenize(PROMPT, add_special=True, parse_special=True))


def test_prepare_leaves_pending_continuation(model: ftm.Model) -> None:
    draft = make_draft(model)
    tokens = prompt_tokens(model)
    assert len(tokens) > 64, "prompt must span several chunks"
    draft.prepare(0, 0, tokens)
    first = draft.propose(0, 1)
    assert len(first) == 1 and isinstance(first[0], int)


def test_propose_is_greedily_deterministic(model: ftm.Model) -> None:
    """Same prompt, fresh slot each time -> identical drafts (greedy)."""
    draft = make_draft(model)
    tokens = prompt_tokens(model)
    draft.prepare(0, 0, tokens)
    first_run = draft.propose(0, 4)
    draft.release(0)
    draft.prepare(1, 0, tokens)
    assert draft.propose(1, 4) == first_run


def test_sync_advances_and_propose_continues(model: ftm.Model) -> None:
    draft = make_draft(model)
    tokens = prompt_tokens(model)
    draft.prepare(0, 0, tokens)
    base_pos = len(tokens)
    drafts = draft.propose(0, 4)
    assert len(drafts) == 4
    # Confirm the first two drafts plus a new tail token, as a verify would.
    confirmed, tail = base_pos + 2, [drafts[0], drafts[1], 9999 % model.n_vocab]
    draft.sync(0, confirmed, tail)
    assert draft._pos[0] == confirmed + len(tail)
    continued = draft.propose(0, 2)
    assert len(continued) == 2


def test_propose_zero_is_noop(model: ftm.Model) -> None:
    draft = make_draft(model)
    draft.prepare(0, 0, prompt_tokens(model))
    assert draft.propose(0, 0) == []
    assert len(draft.propose(0, 2)) == 2


def test_unknown_request_ids(model: ftm.Model) -> None:
    draft = make_draft(model)
    with pytest.raises(KeyError):
        draft.propose(99, 2)
    with pytest.raises(KeyError):
        draft.sync(99, 0, [1])
    draft.release(99)  # tolerant, like _retire


def test_empty_prompt_rejected(model: ftm.Model) -> None:
    draft = make_draft(model)
    with pytest.raises(ValueError, match="zero tokens"):
        draft.prepare(0, 0, [])


def test_draft_target_gate_rejects_hybrids() -> None:
    """Hybrid targets must fail loud: measured deterministic divergence on
    27B once foreign decodes interleave with multi-row target decodes
    (upstream #20075 class; our pin predates the fix)."""
    from freetoken_mac.engine.metal_engine import check_draft_target

    for arch in ("qwen35", "qwen35moe", "lfm2", "deepseek4"):
        with pytest.raises(ValueError, match="hybrid"):
            check_draft_target(arch)


def test_draft_target_gate_allows_attention_and_unknown() -> None:
    """Attention targets are exact by construction (cell-drop rewinds); the
    0.5B/8B integration below exercises them. Unknown archs fail open."""
    from freetoken_mac.engine.metal_engine import check_draft_target

    for arch in ("qwen2", "qwen3", "llama", None, "", "something-new"):
        check_draft_target(arch)


# NOTE: no end-to-end refusal test here on purpose. Refusal needs a hybrid
# target, and stubbing `meta_val` to "qwen35" on attention weights trips the
# T5a snapshot gate first (llama clamps n_rs_seq to 0 without recurrent
# state) -- an impossible state, not the real path. The gate LOGIC is pinned
# above; the real 27B refusal is bench evidence (SPEC-draft-model.md).


def _run(model: ftm.Model, draft_model: ftm.Model | None) -> ftm.MetalEngine:
    config = ftm.EngineConfig(n_ctx=512, n_seq_max=1)
    engine = ftm.MetalEngine(model, config, draft_model=draft_model)
    engine.add_request(
        "Count: " + " ".join(str(i % 10) for i in range(24)),
        ftm.RequestParams(temp=0.0, max_tokens=16, stop_at_eog=False),
    )
    list(engine.drain())
    return engine


def test_draft_on_off_byte_identical(model: ftm.Model) -> None:
    """The invariant, draft-sourced: a second same-file model drafts."""
    draft_model = ftm.Model(MODEL_PATH, ftm.ModelParams())
    plain = _run(model, None)
    spec = _run(model, draft_model)
    plain_id = plain._next_request_id - 1
    spec_id = spec._next_request_id - 1
    assert spec.tokens_of(spec_id) == plain.tokens_of(plain_id)
    assert spec.state(spec_id).finish_reason == plain.state(plain_id).finish_reason


def test_draft_never_uses_more_decodes(model: ftm.Model) -> None:
    """Same structural guarantee as the n-gram path: every verify accepts >=
    1 new token. `spec_drafted > 0` proves the draft loop actually ran."""
    draft_model = ftm.Model(MODEL_PATH, ftm.ModelParams())
    plain = _run(model, None)
    spec = _run(model, draft_model)
    assert spec.spec_drafted > 0, "draft loop did not pack any drafts"
    assert spec.ctx.decode_calls <= plain.ctx.decode_calls
    rate = spec.spec_acceptance_rate
    assert rate is not None and 0.0 <= rate <= 1.0


def test_draft_and_ngram_are_mutually_exclusive(model: ftm.Model) -> None:
    draft_model = ftm.Model(MODEL_PATH, ftm.ModelParams())
    with pytest.raises(ValueError, match="mutually exclusive"):
        ftm.MetalEngine(
            model,
            ftm.EngineConfig(n_ctx=512, n_seq_max=1, speculative=True),
            draft_model=draft_model,
        )


def test_negative_draft_max_drafts_rejected(model: ftm.Model) -> None:
    with pytest.raises(ValueError, match="draft_max_drafts"):
        ftm.MetalEngine(
            model, ftm.EngineConfig(draft_max_drafts=-1), draft_model=model
        )
