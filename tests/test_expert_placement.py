"""Expert placement plumbing (SPEC-expert-placement.md, T9a/T9b).

Binding defaults and validation run without weights. The MoE A/B needs
BWR_MOE_MODEL pointing at a GGUF with expert tensors (e.g. Qwen3-30B-A3B)
and skips without it. The A/B's two claims: placement never changes
numerics (byte-identical), and CPU experts cost real time (the gap later
phases must beat -- a margin that fails loud if overrides ever match
nothing on a future arch).
"""

from __future__ import annotations

import os
import time

import pytest

import bwr as bwr
from bwr.engine import EngineConfig, MetalEngine, RequestParams

MOE_PATH = os.environ.get("BWR_MOE_MODEL")

PROMPT = "The capital of France is"
N_TOKENS = 8


def test_expert_weights_default_metal() -> None:
    assert bwr.ModelParams().expert_weights == "metal"


def test_expert_weights_invalid_rejected_before_load() -> None:
    mp = bwr.ModelParams()
    mp.expert_weights = "tpu"
    with pytest.raises(ValueError, match="expert_weights"):
        bwr.Model("/nonexistent/model.gguf", mp)


needs_moe = pytest.mark.skipif(
    not MOE_PATH or not os.path.exists(MOE_PATH),
    reason="set BWR_MOE_MODEL to a MoE .gguf path to run placement tests",
)


def _moe_params(expert_weights: str) -> bwr.ModelParams:
    mp = bwr.ModelParams()
    mp.expert_weights = expert_weights
    return mp


def _run(model: bwr.Model) -> tuple[list[int], str]:
    engine = MetalEngine(model, EngineConfig(n_ctx=512, n_seq_max=1))
    rid = engine.add_request(
        PROMPT, RequestParams(temp=0.0, max_tokens=N_TOKENS, stop_at_eog=False)
    )
    list(engine.drain())
    return engine.tokens_of(rid), engine.state(rid).finish_reason or ""


@needs_moe
def test_moe_expert_placement_identical() -> None:
    """All-Metal vs experts-CPU must agree token-for-token."""
    metal = bwr.Model(MOE_PATH, _moe_params("metal"))
    cpu = bwr.Model(MOE_PATH, _moe_params("cpu"))
    try:
        metal_toks, metal_reason = _run(metal)
        cpu_toks, cpu_reason = _run(cpu)
    finally:
        metal.close()
        cpu.close()
    assert cpu_toks == metal_toks
    assert cpu_reason == metal_reason == "length"


@needs_moe
def test_moe_cpu_experts_cost_time() -> None:
    """CPU experts must be MUCH slower: 16GB of experts on Accelerate vs
    Metal. If a future arch renames expert tensors, the override matches
    nothing, timings equalize, and this fails loud (by design)."""
    metal = bwr.Model(MOE_PATH, _moe_params("metal"))
    t0 = time.monotonic()
    _run(metal)
    metal_s = time.monotonic() - t0
    metal.close()
    cpu = bwr.Model(MOE_PATH, _moe_params("cpu"))
    t0 = time.monotonic()
    _run(cpu)
    cpu_s = time.monotonic() - t0
    cpu.close()
    print(f"\nplacement A/B: metal={metal_s:.1f}s cpu={cpu_s:.1f}s", flush=True)
    assert cpu_s > 2 * metal_s
