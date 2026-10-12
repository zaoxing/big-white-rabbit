"""KV snapshot primitive — in-memory + disk (T12a/b/c).

Needs BWR_MOE_MODEL or any model; uses n_seq_max=2 so a snapshot seq exists.
What is asserted: save while in-flight preserves KV, load restores and
generates identical continuation vs plain re-prefill, list/delete, and
T12c binary KV restores across engine restart without re-prefill.
"""

from __future__ import annotations

import os

import bwr
import pytest
from bwr.engine import EngineConfig, MetalEngine
from bwr.engine.kv_snapshot import KVSnapStore

MODEL_PATH = os.environ.get("BWR_MOE_MODEL") or os.environ.get("BWR_TEST_MODEL")

# The weight-gated tests below carry this mark; the stub-level ones at the end
# run everywhere.
needs_model = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_MOE_MODEL or BWR_TEST_MODEL to a .gguf path",
)

PROMPT = "Count: 1 2 3"
N_TOKENS = 8


@needs_model
def test_snapshot_save_load_identical(tmp_path) -> None:
    """Save prompt KV after prefill, restore into new request, identical output."""
    cfg = EngineConfig(n_ctx=512, n_seq_max=2, record_experts=False)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store = KVSnapStore(engine, kv_dir=tmp_path)

    # First request: prefill one step, snapshot, then drain to get reference output
    rid = engine.add_request(PROMPT, bwr.RequestParams(max_tokens=N_TOKENS, stop_at_eog=False, temp=0.0))
    engine.step()  # prefill
    store.save("snap", rid)
    list(engine.drain())
    ref_tokens = engine.tokens_of(rid)

    # New request via snapshot restore (same prompt, same params)
    # Use a fresh engine to prove snapshot is self-contained (same process, new engine reuses same snapshot store? No, snapshot is per-engine in-memory, so we restore within same engine after original retired)
    # Instead, restore as a new rid in same engine after original retired (original's seq freed, but snapshot seq still holds KV)
    rid2 = store.load("snap")
    # rid2's seq already has prompt KV; drain should generate same output without re-prefill
    list(engine.drain())
    assert engine.tokens_of(rid2) == ref_tokens
    assert engine.state(rid2).finish_reason == engine.state(rid).finish_reason


@needs_model
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


@needs_model
def test_snapshot_needs_spare_seq() -> None:
    cfg = EngineConfig(n_ctx=512, n_seq_max=1)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    with pytest.raises(ValueError, match="n_seq_max"):
        KVSnapStore(engine)


@needs_model
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


@needs_model
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
    list(engine.drain())
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


@needs_model
def test_second_save_replaces_first_snapshot(tmp_path) -> None:
    """Two saves in a row, then load the second one.

    With kv_unified=True llama.cpp's same-stream seq_cp only adds the
    destination tag to cells, so unless `save` clears the snapshot seq
    first, the second snapshot stacks on top of the first and the restored
    seq has two cells per position -- decode then fails. The restored
    request must match a plain prefill of the same prompt."""
    cfg = EngineConfig(n_ctx=512, n_seq_max=2, record_experts=False, kv_unified=True)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store = KVSnapStore(engine, kv_dir=tmp_path)
    params = bwr.RequestParams(max_tokens=N_TOKENS, stop_at_eog=False, temp=0.0)

    long_prompt = " ".join(str(i) for i in range(60))
    rid_a = engine.add_request(long_prompt, params)
    engine.step()
    store.save("a", rid_a)
    list(engine.drain())

    rid_b = engine.add_request(PROMPT, params)
    engine.step()
    store.save("b", rid_b)
    list(engine.drain())
    ref_tokens = engine.tokens_of(rid_b)

    rid_c = store.load("b")
    list(engine.drain())
    assert engine.tokens_of(rid_c) == ref_tokens
    assert engine.state(rid_c).finish_reason == "length"


@needs_model
def test_load_of_an_earlier_save_restores_its_own_kv(tmp_path) -> None:
    """save a, save b, load a: the restored request must continue a's prompt.

    There is one snapshot seq, and the second save overwrites it with b's
    KV. If a's entry still claims its KV is live there, load("a") copies
    b's cells under a's token history and generates b's continuation."""
    cfg = EngineConfig(n_ctx=512, n_seq_max=2, record_experts=False)
    engine = MetalEngine(bwr.Model(MODEL_PATH, bwr.ModelParams()), cfg)
    store = KVSnapStore(engine, kv_dir=tmp_path)
    params = bwr.RequestParams(max_tokens=N_TOKENS, stop_at_eog=False, temp=0.0)

    prompt_a = "The capital of France is"
    rid_a = engine.add_request(prompt_a, params)
    engine.step()
    store.save("a", rid_a)
    list(engine.drain())
    ref_a = engine.tokens_of(rid_a)

    rid_b = engine.add_request(PROMPT, params)
    engine.step()
    store.save("b", rid_b)
    list(engine.drain())
    ref_b = engine.tokens_of(rid_b)
    assert ref_a != ref_b  # the two prompts must be distinguishable

    rid_c = store.load("a")
    list(engine.drain())
    assert engine.tokens_of(rid_c) == ref_a
    assert engine.state(rid_c).finish_reason == "length"


# --- stub-level: no weights needed -----------------------------------------


class _FakeCtx:
    """Just enough of the llama context for KVSnapStore's bookkeeping."""

    n_seq_max = 2

    def __init__(self, set_data_raises: bool = False) -> None:
        self.calls: list[tuple] = []
        self.set_data_raises = set_data_raises

    def memory_seq_rm(self, *a) -> None:
        self.calls.append(("rm", *a))

    def memory_seq_cp(self, *a) -> None:
        self.calls.append(("cp", *a))

    def state_seq_get_data(self, seq_id):
        return b"kv"

    def state_seq_set_data(self, seq_id, data):
        self.calls.append(("set_data", seq_id))
        if self.set_data_raises:
            raise RuntimeError("incompatible state blob")
        return len(data)

    def set_seq_sampler(self, seq_id, params) -> None:
        self.calls.append(("sampler", seq_id))


def _stub_engine(**ctx_kwargs) -> MetalEngine:
    eng = MetalEngine.__new__(MetalEngine)
    eng.ctx = _FakeCtx(**ctx_kwargs)
    eng._free_seq_ids = [0, 1]
    eng._states = {}
    eng._retired = {}
    eng._pins = {}
    eng._next_request_id = 0
    return eng


def _live_request(rid: int, seq_id: int, prompt: list[int], output: list[int]):
    """An in-flight request: prompt fully prefilled, `output` generated so far."""
    from bwr.engine.batching import RequestState

    st = RequestState(request_id=rid, seq_id=seq_id, prompt=list(prompt), params=bwr.RequestParams(max_tokens=50))
    st.n_prefilled = len(prompt)
    st.n_pos = len(prompt) + len(output)
    st.output_tokens = list(output)
    st.n_generated = len(output)
    st.next_token = output[-1]
    return st


def _finished_request(rid: int, seq_id: int):
    from bwr.engine.batching import RequestState

    st = RequestState(request_id=rid, seq_id=seq_id, prompt=[1, 2, 3], params=bwr.RequestParams(max_tokens=2))
    st.n_prefilled = 3
    st.n_pos = 5
    st.output_tokens = [7, 8]
    st.n_generated = 2
    st.next_token = None
    st.finished = True
    st.finish_reason = "length"
    return st


def test_save_refuses_a_finished_request(tmp_path) -> None:
    """_lookup answers for retired requests too, so without a guard save()
    captures next_token=None from a finished request. load() then installs a
    state that is neither prefilling, generating nor finished: has_work stays
    True forever and the worker spins on a request nobody can advance."""
    eng = _stub_engine()
    eng._retired[0] = _finished_request(0, 0)
    eng._next_request_id = 1
    store = KVSnapStore(eng, kv_dir=tmp_path)
    with pytest.raises(ValueError, match="finished"):
        store.save("done", 0)
    assert "done" not in store.list()
    assert not (tmp_path / "done.json").exists()


def test_load_refuses_a_dead_snapshot_file(tmp_path) -> None:
    """A snapshot file written before save() refused finished requests has
    next_token null with nothing left to prefill. load() must say so rather
    than install a state the engine can never advance, and must not leak
    the seq it would have used."""
    import json

    (tmp_path / "dead.json").write_text(
        json.dumps(
            {
                "prompt": [1, 2, 3],
                "output_tokens": [7, 8],
                "n_pos": 5,
                "n_prefilled": 3,
                "next_token": None,
                "params": {"max_tokens": 2},
            }
        )
    )
    (tmp_path / "dead.bin").write_bytes(b"kv")
    eng = _stub_engine()
    store = KVSnapStore(eng, kv_dir=tmp_path)
    with pytest.raises(ValueError, match="finished"):
        store.load("dead")
    assert eng._states == {}
    assert eng._free_seq_ids == [0]
    assert not eng.has_work


def test_second_save_demotes_the_first_to_its_bin_file(tmp_path) -> None:
    """There is one snapshot seq. After save("a"), save("b") the live KV in it
    is b's, so load("a") must not seq_cp from it: it has to come back through
    a.bin (state_seq_set_data) and carry a's token history, not b's."""
    eng = _stub_engine()
    eng.ctx.n_seq_max = 5  # seqs 0 and 2 in flight, 1 is the snapshot seq
    eng._free_seq_ids = [3, 4]
    store = KVSnapStore(eng, kv_dir=tmp_path)
    eng._states[0] = _live_request(0, 0, [1, 2, 3], [10])
    eng._states[1] = _live_request(1, 2, [7, 7, 7, 7], [20])
    store.save("a", 0)
    store.save("b", 1)
    eng.ctx.calls.clear()

    rid = store.load("a")
    cps = [c for c in eng.ctx.calls if c[0] == "cp"]
    assert cps == [], f"load('a') copied the snapshot seq, which holds b's KV: {cps}"
    assert ("set_data", 3) in eng.ctx.calls
    assert eng._states[rid].prompt == [1, 2, 3]
    assert eng._states[rid].seq_id == 3
    assert eng._free_seq_ids == [4]

    # The latest save is still the live one and restores by seq_cp.
    eng.ctx.calls.clear()
    rid_b = store.load("b")
    assert ("cp", store.snapshot_seq, 4, -1, -1) in eng.ctx.calls
    assert eng._states[rid_b].prompt == [7, 7, 7, 7]
    assert eng._free_seq_ids == []
