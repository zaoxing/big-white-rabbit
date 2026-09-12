"""KV snapshot primitive — in-memory + disk (T12a/b/c).

Needs BWR_MOE_MODEL or any model; uses n_seq_max=2 so a snapshot seq exists.
What is asserted: save while in-flight preserves KV, load restores and
generates identical continuation vs plain re-prefill, list/delete, and
T12c binary KV restores across engine restart without re-prefill.
"""

from __future__ import annotations

import os

import pytest

import bwr as bwr
from bwr.engine import EngineConfig, MetalEngine
from bwr.engine.kv_snapshot import KVSnapStore

MODEL_PATH = os.environ.get("BWR_MOE_MODEL") or os.environ.get("BWR_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_MOE_MODEL or BWR_TEST_MODEL to a .gguf path",
)

PROMPT = "Count: 1 2 3"
N_TOKENS = 8


def test_snapshot_save_load_identical(tmp_path) -> None:
    """Save prompt KV after prefill, restore into new request, identical output."""
    cfg = EngineConfig(n_ctx=512, n_seq_max=2, record_experts=False)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store = KVSnapStore(engine, kv_dir=tmp_path)

    # First request: prefill one step, snapshot, then drain to get reference output
    rid = engine.add_request(PROMPT, bwr.RequestParams(max_tokens=N_TOKENS, stop_at_eog=False, temp=0.0))
    engine.step()  # prefill
    store.save("snap", rid)
    ref_out = list(engine.drain())
    ref_tokens = engine.tokens_of(rid)

    # New request via snapshot restore (same prompt, same params)
    # Use a fresh engine to prove snapshot is self-contained (same process, new engine reuses same snapshot store? No, snapshot is per-engine in-memory, so we restore within same engine after original retired)
    # Instead, restore as a new rid in same engine after original retired (original's seq freed, but snapshot seq still holds KV)
    rid2 = store.load("snap")
    # rid2's seq already has prompt KV; drain should generate same output without re-prefill
    out2 = list(engine.drain())
    assert engine.tokens_of(rid2) == ref_tokens
    assert engine.state(rid2).finish_reason == engine.state(rid).finish_reason


def test_snapshot_list_delete(tmp_path) -> None:
    cfg = EngineConfig(n_ctx=512, n_seq_max=2)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store = KVSnapStore(engine, kv_dir=tmp_path)
    rid = engine.add_request(PROMPT, bwr.RequestParams(max_tokens=4, stop_at_eog=False, temp=0.0))
    engine.step()
    store.save("a", rid)
    assert "a" in store.list()
    store.delete("a")
    assert "a" not in store.list()
    store.save("b", rid)
    store.clear()
    assert store.list() == []


def test_snapshot_needs_spare_seq() -> None:
    cfg = EngineConfig(n_ctx=512, n_seq_max=1)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    with pytest.raises(ValueError, match="n_seq_max"):
        KVSnapStore(engine)


def test_snapshot_disk_roundtrip(tmp_path) -> None:
    """Disk persistence survives engine restart (T12b re-prefill or T12c binary)."""
    cfg = EngineConfig(n_ctx=512, n_seq_max=2)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store = KVSnapStore(engine, kv_dir=tmp_path)
    rid = engine.add_request(PROMPT, bwr.RequestParams(max_tokens=4, stop_at_eog=False, temp=0.0))
    engine.step()
    store.save("disk", rid)
    # New engine, same dir, no in-memory snapshot
    engine2 = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store2 = KVSnapStore(engine2, kv_dir=tmp_path)
    assert "disk" in store2.list()
    rid2 = store2.load("disk")
    list(engine2.drain())
    # Disk load via T12c binary or re-prefill — both deterministic
    assert len(engine2.tokens_of(rid2)) == 4
    assert engine2.state(rid2).finish_reason == "length"


def test_snapshot_binary_roundtrip_identical(tmp_path) -> None:
    """T12c: binary KV restores identical tokens across restart without re-prefill."""
    cfg = EngineConfig(n_ctx=512, n_seq_max=2, record_experts=False)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store = KVSnapStore(engine, kv_dir=tmp_path)

    # Reference: prefill + snapshot + drain
    rid = engine.add_request(PROMPT, bwr.RequestParams(max_tokens=N_TOKENS, stop_at_eog=False, temp=0.0))
    engine.step()
    store.save("snap_bin", rid)
    # bin should exist (T12c)
    assert (tmp_path / "snap_bin.bin").exists()
    assert (tmp_path / "snap_bin.bin").stat().st_size > 0
    ref_out = list(engine.drain())
    ref_tokens = engine.tokens_of(rid)

    # New engine, same dir — load via binary
    engine2 = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store2 = KVSnapStore(engine2, kv_dir=tmp_path)
    assert "snap_bin" in store2.list()
    rid2 = store2.load("snap_bin")
    list(engine2.drain())
    assert engine2.tokens_of(rid2) == ref_tokens
    assert engine2.state(rid2).finish_reason == engine.state(rid).finish_reason

    # Delete cleans both json and bin
    store2.delete("snap_bin")
    assert "snap_bin" not in store2.list()
    assert not (tmp_path / "snap_bin.bin").exists()
    assert not (tmp_path / "snap_bin.json").exists()
