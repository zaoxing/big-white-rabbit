"""MoE router recording (SPEC-residency.md, T10b).

Needs FTM_MOE_MODEL pointing at a GGUF with expert tensors; skips without
it. What is asserted: flag-off default, frame shape + softmax exactness
(rows sum to 1 -- proving the callback reads real distributions, not
garbage), determinism across runs, consume-on-read, identical outputs with
recording on vs off (the callback splits batches; batching must not perturb
numerics), and the single-sequence guard.
"""

from __future__ import annotations

import math
import os

import pytest

import freetoken_mac as ftm
from freetoken_mac.engine import EngineConfig, MetalEngine, RequestParams

MODEL_PATH = os.environ.get("FTM_MOE_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set FTM_MOE_MODEL to a MoE .gguf path to run residency tests",
)

PROMPT = "Count: 1 2 3"
N_TOKENS = 8


def recording_config(**overrides) -> EngineConfig:
    base = {"n_ctx": 512, "n_seq_max": 1, "record_experts": True}
    base.update(overrides)
    return EngineConfig(**base)


def plain_config(**overrides) -> EngineConfig:
    base = {"n_ctx": 512, "n_seq_max": 1}
    base.update(overrides)
    return EngineConfig(**base)


def greedy(max_tokens: int = N_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False)


@pytest.fixture(scope="module")
def model() -> ftm.Model:
    return ftm.Model(MODEL_PATH, ftm.ModelParams())


def run(engine: MetalEngine, prompt: str = PROMPT) -> int:
    rid = engine.add_request(prompt, greedy())
    list(engine.drain())
    return rid


def top1_per_layer(frames: list[dict]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for frame in frames:
        tokens = frame["tokens"]
        assert isinstance(tokens, list) and tokens
        out[int(frame["layer"])] = [max(range(len(row)), key=row.__getitem__) for row in tokens]
    return out


def test_recording_defaults_off(model: ftm.Model) -> None:
    assert EngineConfig().record_experts is False
    assert ftm.ContextParams().record_experts is False
    engine = MetalEngine(model, plain_config())
    rid = run(engine)
    assert engine.tokens_of(rid) is not None
    assert engine.expert_activations(rid) == []


def test_frames_shape_and_softmax(model: ftm.Model) -> None:
    engine = MetalEngine(model, recording_config())
    rid = run(engine)
    frames = engine.expert_activations(rid)
    assert frames, "no router frames recorded"
    # Frames accumulate across every decode since the drain (prefill plus one
    # per step), so layers repeat -- distinct coverage is the assertion.
    layers = sorted(set(f["layer"] for f in frames))
    assert layers == list(range(model.n_layer)), "every layer must report"
    widths = set()
    for frame in frames:
        for row in frame["tokens"]:
            assert all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in row)
            assert abs(sum(row) - 1.0) < 1e-2, "rows must be distributions"
            widths.add(len(row))
    assert len(widths) == 1, "one shared expert count"


def test_deterministic_routing(model: ftm.Model) -> None:
    first = MetalEngine(model, recording_config())
    rid_first = run(first, PROMPT)
    second = MetalEngine(model, recording_config())
    rid_second = run(second, PROMPT)
    assert top1_per_layer(first.expert_activations(rid_first)) == top1_per_layer(
        second.expert_activations(rid_second)
    )


def test_consume_semantics(model: ftm.Model) -> None:
    engine = MetalEngine(model, recording_config())
    rid = run(engine)
    assert engine.expert_activations(rid) != []
    assert engine.expert_activations(rid) == []


def test_recording_preserves_outputs(model: ftm.Model) -> None:
    """The callback splits every MoE layer's batch: prove the split changes
    nothing observable (identical tokens + reason vs a silent engine)."""
    rec = MetalEngine(model, recording_config())
    rid_rec = run(rec)
    plain = MetalEngine(model, plain_config())
    rid_plain = run(plain)
    assert rec.tokens_of(rid_rec) == plain.tokens_of(rid_plain)
    assert rec.state(rid_rec).finish_reason == plain.state(rid_plain).finish_reason


def test_multi_seq_refused(model: ftm.Model) -> None:
    with pytest.raises(ValueError, match="n_seq_max"):
        MetalEngine(model, recording_config(n_seq_max=2))


def test_unknown_request_rejected(model: ftm.Model) -> None:
    engine = MetalEngine(model, recording_config())
    with pytest.raises(KeyError):
        engine.expert_activations(999)
