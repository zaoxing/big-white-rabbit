"""ModelPool: discovery, residency, LRU eviction, drain-before-unload.

Residency is tested with a stub engine factory -- the policy is the thing
under test, and loading real 18 GiB weights would make it untestable.
"""

from __future__ import annotations

import asyncio

import pytest

from bwr.engine.config import EngineConfig
from bwr.engine.pool import ModelEntry, ModelPool, ModelPoolError, discover

GIB = 1024**3


# -- discovery (real filesystem, no weights) ---------------------------------


def test_discover_finds_mlx_dir_and_gguf(tmp_path):
    (tmp_path / "mlx-model").mkdir()
    (tmp_path / "mlx-model" / "config.json").write_text("{}")
    (tmp_path / "metal-model.gguf").write_bytes(b"\x00" * 16)

    found = {e.model_id: e.kind for e in discover(tmp_path)}
    assert found == {"mlx-model": "mlx", "metal-model": "metal"}


def test_discover_follows_one_level_for_nested_variants(tmp_path):
    """Published repos often ship `<repo>/4-bit/config.json`; that layout has
    to work without the user reorganising files."""
    nested = tmp_path / "Some-27B" / "4-bit"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}")

    ids = [e.model_id for e in discover(tmp_path)]
    assert ids == ["Some-27B/4-bit"]


def test_discover_ignores_dotfiles_and_plain_dirs(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "config.json").write_text("{}")
    (tmp_path / "not-a-model").mkdir()
    assert discover(tmp_path) == []


def test_discover_missing_dir_is_empty_not_error(tmp_path):
    assert discover(tmp_path / "nope") == []


# -- residency (stub engines) ------------------------------------------------


class _StubRaw:
    def __init__(self, entry):
        self.entry = entry
        self.stopped = False

    class ctx:  # noqa: N801 - mirrors the real engine's attribute shape
        n_ctx = 4096
        n_ctx_seq = 4096
        n_seq_max = 8
        decode_calls = 0


class _StubAsync:
    """Stands in for AsyncEngine: only start/stop are exercised here."""

    def __init__(self, raw):
        self.raw = raw
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self, **_kw):
        self.stopped = True
        self.raw.stopped = True


@pytest.fixture
def pool_factory(tmp_path, monkeypatch):
    def make(models: dict[str, int], budget_gib: float):
        for name in models:
            d = tmp_path / name
            d.mkdir()
            (d / "config.json").write_text("{}")

        def fake_build(entry, _config):
            return _StubRaw(entry)

        p = ModelPool(
            tmp_path,
            EngineConfig(),
            budget_bytes=int(budget_gib * GIB),
            engine_factory=fake_build,
        )
        # sizes come from host.model_bytes on real files; override for the test
        for mid, gib in models.items():
            p._entries[mid].size_bytes = int(gib * GIB)
        monkeypatch.setattr("bwr.engine.pool.AsyncEngine", _StubAsync)
        return p

    return make


def test_acquire_loads_lazily(pool_factory):
    p = pool_factory({"a": 4}, budget_gib=32)
    assert p.loaded_ids == []
    mid, eng = asyncio.run(p.acquire("a"))
    assert mid == "a" and eng.started
    assert p.loaded_ids == ["a"]


def test_second_acquire_reuses_the_engine(pool_factory):
    p = pool_factory({"a": 4}, budget_gib=32)

    async def go():
        _, e1 = await p.acquire("a")
        _, e2 = await p.acquire("a")
        return e1, e2

    e1, e2 = asyncio.run(go())
    assert e1 is e2


def test_lru_evicts_to_make_room(pool_factory):
    """Budget fits two 10 GiB models, not three."""
    p = pool_factory({"a": 10, "b": 10, "c": 10}, budget_gib=25)

    async def go():
        await p.acquire("a")
        await p.acquire("b")
        await p.acquire("c")

    asyncio.run(go())
    assert "a" not in p.loaded_ids, "least-recently-used should have been evicted"
    assert set(p.loaded_ids) == {"b", "c"}


def test_eviction_is_by_last_use_not_load_order(pool_factory):
    """A model being actively served must be the last thing dropped."""
    p = pool_factory({"a": 10, "b": 10, "c": 10}, budget_gib=25)

    async def go():
        await p.acquire("a")
        await p.acquire("b")
        await p.acquire("a")  # a is now most-recently-used
        await p.acquire("c")

    asyncio.run(go())
    assert "b" not in p.loaded_ids
    assert set(p.loaded_ids) == {"a", "c"}


def test_model_larger_than_budget_is_refused_not_attempted(pool_factory):
    p = pool_factory({"huge": 80}, budget_gib=32)
    with pytest.raises(ModelPoolError, match="cannot be served"):
        asyncio.run(p.acquire("huge"))
    assert p.loaded_ids == []


def test_unload_stops_the_engine(pool_factory):
    p = pool_factory({"a": 4}, budget_gib=32)

    async def go():
        _, eng = await p.acquire("a")
        ok = await p.unload("a")
        return ok, eng

    ok, eng = asyncio.run(go())
    assert ok and eng.stopped
    assert p.loaded_ids == []


def test_unload_unknown_model_is_false_not_error(pool_factory):
    p = pool_factory({"a": 4}, budget_gib=32)
    assert asyncio.run(p.unload("nope")) is False


def test_unload_waits_for_in_flight_requests(pool_factory):
    """Eviction must not turn someone's live generation into a 500."""
    p = pool_factory({"a": 4}, budget_gib=32)
    order: list[str] = []

    async def go():
        await p.acquire("a")
        p.request_started("a")

        async def finisher():
            await asyncio.sleep(0.05)
            order.append("request-done")
            p.request_finished("a")

        task = asyncio.create_task(finisher())
        await p.unload("a")
        order.append("unloaded")
        await task

    asyncio.run(go())
    assert order == ["request-done", "unloaded"]


# -- request routing ---------------------------------------------------------


def test_single_model_answers_to_any_name(pool_factory):
    """Keeps clients that hardcode 'local' or an OpenAI model id working."""
    p = pool_factory({"only": 4}, budget_gib=32)
    assert p.resolve("gpt-3.5-turbo") == "only"
    assert p.resolve(None) == "only"


def test_multi_model_requires_an_exact_name(pool_factory):
    p = pool_factory({"a": 4, "b": 4}, budget_gib=32)
    assert p.resolve("b") == "b"
    with pytest.raises(ModelPoolError, match="unknown model"):
        p.resolve("nope")


def test_resolve_with_no_models_is_actionable(tmp_path):
    p = ModelPool(tmp_path, EngineConfig(), budget_bytes=GIB)
    with pytest.raises(ModelPoolError, match="no models found"):
        p.resolve("anything")


def test_engine_for_does_not_trigger_a_load(pool_factory):
    """Read-only callers (stats, /health) must never cause 18 GiB of I/O."""
    p = pool_factory({"a": 4}, budget_gib=32)
    assert p.engine_for("a") is None
    assert p.loaded_ids == []


def test_list_reports_loaded_state(pool_factory):
    p = pool_factory({"a": 4, "b": 4}, budget_gib=32)
    asyncio.run(p.acquire("a"))
    by_id = {m["id"]: m for m in p.list()}
    assert by_id["a"]["loaded"] is True
    assert by_id["b"]["loaded"] is False


def test_rescan_picks_up_new_models_without_dropping_loaded(pool_factory, tmp_path):
    p = pool_factory({"a": 4}, budget_gib=32)
    asyncio.run(p.acquire("a"))
    new = tmp_path / "later"
    new.mkdir()
    (new / "config.json").write_text("{}")
    p.rescan()
    assert {m["id"] for m in p.list()} == {"a", "later"}
    assert p.loaded_ids == ["a"]


# -- preload -----------------------------------------------------------------


def test_preload_first_resolves_to_a_concrete_id(pool_factory):
    """`--preload first` must pick the first LISTED model, not pass None.

    Regression: None was passed to acquire(), and resolve(None) only
    succeeds when the pool holds exactly one model -- so preloading failed
    with "unknown model None" on any real multi-model directory.
    """
    p = pool_factory({"a": 4, "b": 4}, budget_gib=32)
    listed = p.list()
    first = listed[0]["id"]
    mid, _engine = asyncio.run(p.acquire(first))
    assert mid == first
    assert p.loaded_ids == [first]

    with pytest.raises(ModelPoolError):
        p.resolve(None)      # the bug: None is not a valid multi-model target


# -- unknown memory budget ---------------------------------------------------


def test_an_unknown_budget_does_not_evict_everything(pool_factory):
    """`budget_bytes == 0` means the RAM size was unreadable, not "no room".

    The admission guard already reads it that way (it skips on falsy), but
    the eviction loop read the same 0 as a hard cap and unloaded EVERY
    resident model before each load. `sysctl` being off PATH -- which is the
    normal case for a process spawned by a GUI app, since it inherits
    launchd's PATH and not a login shell's -- was enough to silently turn the
    pool into a one-model cache that reloaded 17 GiB on every switch.
    """
    p = pool_factory({"a": 4, "b": 4}, budget_gib=32)
    p.budget_bytes = 0
    asyncio.run(p.acquire("a"))
    asyncio.run(p.acquire("b"))
    assert sorted(p.loaded_ids) == ["a", "b"]


def test_a_known_budget_still_evicts(pool_factory):
    """The companion to the above: a real budget must still make room."""
    p = pool_factory({"a": 4, "b": 4}, budget_gib=6)
    asyncio.run(p.acquire("a"))
    asyncio.run(p.acquire("b"))
    assert p.loaded_ids == ["b"]


# -- per-model context overrides ---------------------------------------------
#
# The context benchmark measures a window and writes it here; the pool is
# what makes it take effect. See server/ctxbench.py.


def _ctx_pool(tmp_path, monkeypatch, models=("a",)):
    """A pool whose stub factory records the n_ctx each load was given."""
    for name in models:
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        # Only if the test has not written its own: `native_ctx` reads this
        # file, so clobbering it would make that untestable.
        cfg = d / "config.json"
        if not cfg.exists():
            cfg.write_text("{}")
    seen: list[int] = []

    def fake_build(entry, config):
        seen.append(config.n_ctx)
        return _StubRaw(entry)

    monkeypatch.setattr("bwr.engine.pool.AsyncEngine", _StubAsync)
    p = ModelPool(
        tmp_path, EngineConfig(n_ctx=4096),
        budget_bytes=64 * GIB, engine_factory=fake_build,
    )
    return p, seen


def test_a_model_without_an_override_loads_at_the_shared_config(tmp_path, monkeypatch):
    p, seen = _ctx_pool(tmp_path, monkeypatch)
    asyncio.run(p.acquire("a"))
    assert seen == [4096]


def test_an_override_is_what_the_next_load_uses(tmp_path, monkeypatch):
    p, seen = _ctx_pool(tmp_path, monkeypatch)
    p.set_ctx_override("a", 40960)
    asyncio.run(p.acquire("a"))
    assert seen == [40960]


def test_an_override_does_not_disturb_a_resident_engine(tmp_path, monkeypatch):
    """KV geometry is fixed when the engine is built, so a live engine cannot
    adopt a new window. The pool must not pretend otherwise."""
    p, seen = _ctx_pool(tmp_path, monkeypatch)
    asyncio.run(p.acquire("a"))
    p.set_ctx_override("a", 40960)
    asyncio.run(p.acquire("a"))
    assert seen == [4096]  # still the engine loaded before the override


def test_reload_is_how_a_new_override_becomes_live(tmp_path, monkeypatch):
    p, seen = _ctx_pool(tmp_path, monkeypatch)
    asyncio.run(p.acquire("a"))
    p.set_ctx_override("a", 40960)
    asyncio.run(p.reload("a"))
    assert seen == [4096, 40960]
    assert p.loaded_ids == ["a"]


def test_reload_leaves_nothing_loaded_when_the_new_load_fails(tmp_path, monkeypatch):
    """A window the machine cannot allocate must not leave the old engine
    half-dropped and the pool claiming it is resident."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.json").write_text("{}")
    calls = {"n": 0}

    def fake_build(entry, config):
        calls["n"] += 1
        if config.n_ctx > 8192:
            raise RuntimeError("cannot allocate KV")
        return _StubRaw(entry)

    monkeypatch.setattr("bwr.engine.pool.AsyncEngine", _StubAsync)
    p = ModelPool(tmp_path, EngineConfig(n_ctx=4096), budget_bytes=64 * GIB,
                  engine_factory=fake_build)
    asyncio.run(p.acquire("a"))
    p.set_ctx_override("a", 999_999)
    with pytest.raises(RuntimeError):
        asyncio.run(p.reload("a"))
    assert p.loaded_ids == []
    assert p.engine_for("a") is None


def test_an_override_survives_a_restart(tmp_path, monkeypatch):
    """Measuring a context window costs minutes; losing it on the next boot
    would make the benchmark worth running only once per session."""
    p, _ = _ctx_pool(tmp_path, monkeypatch)
    p.set_ctx_override("a", 40960)
    fresh, seen = _ctx_pool(tmp_path, monkeypatch)
    assert fresh.ctx_override("a") == 40960
    asyncio.run(fresh.acquire("a"))
    assert seen == [40960]


def test_clearing_an_override_returns_the_model_to_the_shared_config(
    tmp_path, monkeypatch
):
    p, _ = _ctx_pool(tmp_path, monkeypatch)
    p.set_ctx_override("a", 40960)
    p.set_ctx_override("a", None)
    fresh, seen = _ctx_pool(tmp_path, monkeypatch)
    assert fresh.ctx_override("a") is None
    asyncio.run(fresh.acquire("a"))
    assert seen == [4096]


def test_a_corrupt_override_file_is_ignored_rather_than_fatal(tmp_path, monkeypatch):
    """A settings file nobody can parse must not stop the pool serving: the
    cost of ignoring it is one model at the default window."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.json").write_text("{}")
    (tmp_path / ".bwr-context.json").write_text("{not json")
    p, seen = _ctx_pool(tmp_path, monkeypatch)
    assert p.ctx_override("a") is None
    asyncio.run(p.acquire("a"))
    assert seen == [4096]


@pytest.mark.parametrize("bad", [0, -1, "8192", True, None])
def test_a_nonsense_override_value_is_dropped_not_loaded_with(
    tmp_path, monkeypatch, bad
):
    """`True` is in here on purpose: bool is an int in Python, and an
    override of `True` would load a model with n_ctx=1."""
    import json

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.json").write_text("{}")
    (tmp_path / ".bwr-context.json").write_text(json.dumps({"n_ctx": {"a": bad}}))
    p, _ = _ctx_pool(tmp_path, monkeypatch)
    assert p.ctx_override("a") is None


def test_setting_a_nonpositive_override_is_refused(tmp_path, monkeypatch):
    p, _ = _ctx_pool(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        p.set_ctx_override("a", 0)


def test_the_override_file_is_not_discovered_as_a_model(tmp_path, monkeypatch):
    p, _ = _ctx_pool(tmp_path, monkeypatch)
    p.set_ctx_override("a", 40960)
    assert [e.model_id for e in discover(tmp_path)] == ["a"]


# -- native context length ---------------------------------------------------


def test_native_ctx_reads_an_mlx_model_s_own_config(tmp_path, monkeypatch):
    import json

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.json").write_text(
        json.dumps({"max_position_embeddings": 262144})
    )
    p, _ = _ctx_pool(tmp_path, monkeypatch)
    assert p.native_ctx("a") == 262144


def test_native_ctx_is_unknown_rather_than_guessed(tmp_path, monkeypatch):
    """`config.json` without the key answers None. A default here would
    silently cap or uncap a benchmark's search."""
    p, _ = _ctx_pool(tmp_path, monkeypatch)  # writes an empty config.json
    assert p.native_ctx("a") is None
    assert p.native_ctx("no-such-model") is None
