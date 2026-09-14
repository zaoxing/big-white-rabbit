"""MTP head discovery + contract parsing (no weights), and the cache
invariants that carry the design (weights required, skipped when absent).

See SPEC-mlx-mtp-draft.md. The two load-bearing facts under test:
proposal must leave the head cache exactly as it found it, and the prompt
prefill must actually populate it -- a fresh cache costs 43 points of
depth-1 acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bwr.engine import mtp

MODEL = Path("models/Qwen3.8-27B-MTPLX-Optimized-Speed")
needs_weights = pytest.mark.skipif(
    not (MODEL / mtp.SIDECAR).is_file(), reason=f"{MODEL}/{mtp.SIDECAR} not present"
)


# -- discovery / contract (no weights) ---------------------------------------


def test_available_false_on_plain_dir(tmp_path):
    assert mtp.available(tmp_path) is False
    assert mtp.sidecar_path(tmp_path) is None


def test_available_true_when_sidecar_present(tmp_path):
    (tmp_path / mtp.SIDECAR).write_bytes(b"")
    assert mtp.available(tmp_path) is True
    assert mtp.sidecar_path(tmp_path) == tmp_path / mtp.SIDECAR


def test_contract_defaults_when_runtime_absent(tmp_path):
    c = mtp.read_contract(tmp_path)
    assert c["concat_order"] == "embedding_hidden"
    assert c["mtp_quant_group_size"] == 64
    assert c["mtp_quant_mode"] == "affine"


def test_contract_reads_published_fields(tmp_path):
    (tmp_path / mtp.RUNTIME).write_text(
        json.dumps(
            {"mtp_contract": {"concat_order": "hidden_embedding",
                              "base_hidden_variant": "pre_norm"}}
        )
    )
    c = mtp.read_contract(tmp_path)
    assert c["concat_order"] == "hidden_embedding"
    assert c["base_hidden_variant"] == "pre_norm"
    # absent fields still fall back
    assert c["mtp_quant_group_size"] == 64


def test_contract_survives_corrupt_runtime_json(tmp_path):
    (tmp_path / mtp.RUNTIME).write_text("{not json")
    assert mtp.read_contract(tmp_path)["concat_order"] == "embedding_hidden"


def test_default_depth_clamped_to_published_max(tmp_path):
    (tmp_path / mtp.RUNTIME).write_text(
        json.dumps({"mtp_depth_default": 9, "mtp_depth_max": 3})
    )
    assert mtp.default_depth(tmp_path) == 3


def test_default_depth_fallback_without_runtime(tmp_path):
    assert mtp.default_depth(tmp_path, fallback=2) == 2


def test_full_attention_layer_idx_selects_self_attn_branch():
    class Args:
        full_attention_interval = 4

    idx = mtp._full_attention_layer_idx(Args())
    assert (idx + 1) % 4 == 0


def test_full_attention_layer_idx_defaults_when_field_missing():
    class Args:
        pass

    idx = mtp._full_attention_layer_idx(Args())
    assert (idx + 1) % 4 == 0


# -- live head (weights required) --------------------------------------------


@pytest.fixture(scope="module")
def head():
    from mlx_lm import load

    model, tok = load(str(MODEL))
    target = model.language_model
    return mtp.MTPDraft(target, MODEL, depth=3), target, tok


@needs_weights
def test_sidecar_loads_strictly(head):
    draft, _, _ = head
    # strict=True is the shape/layout check; 31 tensors in this checkpoint.
    assert draft.n_tensors == 31


@needs_weights
def test_rejects_unsupported_concat_order(tmp_path, head):
    _, target, _ = head
    (tmp_path / mtp.SIDECAR).write_bytes(b"")
    (tmp_path / mtp.RUNTIME).write_text(
        json.dumps({"mtp_contract": {"concat_order": "hidden_embedding"}})
    )
    with pytest.raises(ValueError, match="concat_order"):
        mtp.MTPDraft(target, tmp_path, depth=1)


@needs_weights
def test_prefill_populates_cache(head):
    import mlx.core as mx

    draft, target, tok = head
    ids = tok.encode("def merge(a, b):\n    out = []\n")
    cache = target.make_cache()
    _, hidden = draft.trunk_forward(mx.array(ids)[None], cache)
    mcache = draft.make_cache()
    assert mcache[0].offset == 0
    draft.prefill(mcache, hidden, ids)
    # one entry per (hidden[i], tokens[i+1]) pair
    assert mcache[0].offset == len(ids) - 1


@needs_weights
def test_propose_leaves_cache_unchanged(head):
    """The invariant: drafting is speculative, so the head cache must still
    mirror only accepted tokens when the caller verifies."""
    import mlx.core as mx

    draft, target, tok = head
    ids = tok.encode("def merge(a, b):\n    out = []\n")
    cache = target.make_cache()
    logits, hidden = draft.trunk_forward(mx.array(ids)[None], cache)
    mcache = draft.make_cache()
    draft.prefill(mcache, hidden, ids)
    before = mcache[0].offset

    nxt = int(mx.argmax(logits[0, -1]).item())
    drafts = draft.propose(mcache, hidden[:, -1:, :], nxt, depth=3)

    assert len(drafts) == 3
    assert all(isinstance(t, int) for t in drafts)
    assert mcache[0].offset == before, "propose must trim its speculative entries"


@needs_weights
def test_propose_depth_zero_is_noop(head):
    import mlx.core as mx

    draft, target, tok = head
    ids = tok.encode("hello world")
    cache = target.make_cache()
    logits, hidden = draft.trunk_forward(mx.array(ids)[None], cache)
    mcache = draft.make_cache()
    before = mcache[0].offset
    assert draft.propose(mcache, hidden[:, -1:, :], 5, depth=0) == []
    assert mcache[0].offset == before


@needs_weights
def test_accept_advances_cache_by_token_count(head):
    import mlx.core as mx

    draft, target, tok = head
    ids = tok.encode("def merge(a, b):\n")
    cache = target.make_cache()
    logits, hidden = draft.trunk_forward(mx.array(ids)[None], cache)
    mcache = draft.make_cache()
    draft.prefill(mcache, hidden, ids)
    before = mcache[0].offset
    draft.accept(mcache, hidden[:, -1:, :], [int(mx.argmax(logits[0, -1]).item())])
    assert mcache[0].offset == before + 1


@needs_weights
def test_trunk_forward_reproduces_unmodified_trunk(head):
    """The norm-swap must not perturb the trunk: logits from trunk_forward
    have to equal the ordinary forward, or every verify would be wrong."""
    import mlx.core as mx

    draft, target, tok = head
    ids = mx.array(tok.encode("The capital of France is"))[None]

    c1 = target.make_cache()
    plain = target(ids, cache=c1)
    mx.eval(plain)

    c2 = target.make_cache()
    logits, _ = draft.trunk_forward(ids, c2)
    mx.eval(logits)

    assert mx.allclose(plain, logits, atol=1e-3).item()


@needs_weights
def test_head_beats_chance_on_code_continuation(head):
    """Smoke gate on draft quality: the spec measures ~0.9 depth-1 acceptance
    on code continuation. Assert well clear of n-gram's ~0.3 so a wiring
    regression (e.g. the head silently ignoring the hidden state, which
    scores 0) fails loudly."""
    import mlx.core as mx

    draft, target, tok = head
    prompt = tok.encode(
        "Continue this Python function with production-quality code:\n\n"
        "def load_suite(path):\n    rows = []\n    with open(path) as fh:\n"
    )
    cache = target.make_cache()
    logits, hidden = draft.trunk_forward(mx.array(prompt)[None], cache)
    nxt = int(mx.argmax(logits[0, -1]).item())
    mcache = draft.make_cache()
    draft.prefill(mcache, hidden, prompt)

    hits = trials = 0
    for _ in range(24):
        proposed = draft.propose(mcache, hidden[:, -1:, :], nxt, depth=1)[0]
        logits, hidden = draft.trunk_forward(mx.array([nxt])[None], cache)
        true = int(mx.argmax(logits[0, -1]).item())
        draft.accept(mcache, hidden[:, -1:, :], [nxt])
        trials += 1
        hits += proposed == true
        nxt = true

    assert hits / trials > 0.5, f"depth-1 acceptance {hits}/{trials} too low"


# -- verify-path parity (weights required) -----------------------------------
#
# These pin the guarantee the whole verify path exists for: a draft source may
# only change SPEED, never output. The control is the manual loop with zero
# drafts, NOT `speculative=False` -- that takes the mlx-lm stream path, whose
# ordinary numeric drift shows up as a false parity failure.


def _greedy(tmp_model, **cfg_kw):
    from bwr.engine.config import EngineConfig, RequestParams
    from bwr.engine.mlx_engine import MLXEngine

    prompt = (
        "Continue this Python function with production-quality code and no prose:\n\n"
        "def load_suite(path):\n    rows = []\n"
    )
    eng = MLXEngine(str(tmp_model), EngineConfig(n_ctx=4096, **cfg_kw))
    rid = eng.add_request(prompt, RequestParams(max_tokens=32, temp=0.0))
    for _ in eng.drain():
        pass
    out = list(eng.tokens_of(rid))
    recomputes = eng.spec_recomputes
    eng.ctx.close()
    return out, recomputes


@needs_weights
def test_ngram_speculation_preserves_greedy_output():
    """Regression: _refill used to rebuild from `prompt + output_tokens` while
    the accepted tokens were still unfed, leaving the cache short by `matched`
    and diverging from AR at the first mismatch that followed an accept."""
    ar, _ = _greedy(MODEL, speculative=True, spec_max_drafts=0)
    spec, recomputes = _greedy(MODEL, speculative=True, spec_max_drafts=4)
    assert recomputes > 0, "no mismatch occurred; test would not exercise rewind"
    assert ar == spec


@needs_weights
def test_mtp_speculation_preserves_greedy_output():
    ar, _ = _greedy(MODEL, speculative=True, spec_max_drafts=0)
    drafted, recomputes = _greedy(MODEL, mlx_mtp=True, mlx_mtp_depth=3)
    assert recomputes > 0, "no mismatch occurred; test would not exercise rewind"
    assert ar == drafted


@needs_weights
def test_mtp_and_ngram_are_mutually_exclusive():
    from bwr.engine.config import EngineConfig
    from bwr.engine.mlx_engine import MLXEngine

    with pytest.raises(ValueError, match="mutually exclusive"):
        MLXEngine(str(MODEL), EngineConfig(mlx_mtp=True, speculative=True))


def test_mtp_without_sidecar_is_refused(tmp_path):
    from bwr.engine.config import EngineConfig
    from bwr.engine.mlx_engine import MLXEngine

    with pytest.raises(ValueError, match="sidecar"):
        MLXEngine(str(tmp_path), EngineConfig(mlx_mtp=True))
