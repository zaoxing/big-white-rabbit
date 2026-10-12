"""EngineConfig -> ContextParams forwarding.

`to_context_params` is the only path by which EngineConfig reaches the
C++ context; a field dropped there is a knob that reads as set and does
nothing. Each forwarded field is pinned individually, so the test names
the one that went missing. `ContextParams()` needs no weights.
"""

from __future__ import annotations

import pytest
from bwr.engine.config import EngineConfig

# Every ContextParams attribute to_context_params sets, with a value that
# differs from the EngineConfig default so a dropped assignment shows.
_FORWARDED = {
    "n_ctx": 1234,
    "n_batch": 96,
    "n_ubatch": 48,
    "n_seq_max": 3,
    "n_threads": 5,
    "n_threads_batch": 7,
    "flash_attn": False,
    "kv_unified": True,
    "kv_type": "q8_0",
    "record_experts": True,
}


def test_kv_type_reaches_context_params():
    """The --kv-type lever: llama.cpp's type_k / type_v by ggml name."""
    assert EngineConfig(kv_type="q8_0").to_context_params().kv_type == "q8_0"


def test_kv_type_defaults_to_f16():
    assert EngineConfig().to_context_params().kv_type == "f16"


@pytest.mark.parametrize("name,value", sorted(_FORWARDED.items()))
def test_every_context_field_is_forwarded(name, value):
    cp = EngineConfig(**{name: value}).to_context_params()
    assert getattr(cp, name) == value


def test_all_forwarded_fields_at_once():
    """Together, so an assignment that copies the wrong source field (say
    n_batch into n_ubatch) is caught where one-at-a-time would not."""
    cp = EngineConfig(**_FORWARDED).to_context_params()
    assert {k: getattr(cp, k) for k in _FORWARDED} == _FORWARDED
