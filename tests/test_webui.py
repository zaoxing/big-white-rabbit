"""/admin web UI adapter: endpoint shapes, and the rebrand staying clean.

The UI is derived from oMLX (Apache-2.0). Two things are worth pinning:
the endpoints the chat page polls must keep answering in the shape it reads,
and the rebrand must not regress -- a resynced asset that reintroduces the
licensor's marks would breach Apache-2.0 section 6.
"""

from __future__ import annotations

import json
import pathlib

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from bwr.server.webui import STATIC_DIR, TEMPLATES_DIR  # noqa: E402
from bwr.server.webui.routes import mount  # noqa: E402

PKG = pathlib.Path(TEMPLATES_DIR).parent


class _Ctx:
    n_ctx = 4096
    n_ctx_seq = 4096
    n_seq_max = 8
    decode_calls = 7


class _StubConfig:
    """The subset of EngineConfig the settings endpoint reports."""

    n_ctx = 4096
    n_batch = 512
    n_seq_max = 8
    speculative = False
    mlx_mtp = False
    mlx_prefix_cache = False


class _StubEngine:
    """Only what the adapter reads: counters and config, never work admission."""

    ctx = _Ctx()
    config = _StubConfig()
    n_free_seq_slots = 8
    n_in_flight = 0


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    mount(app, _StubEngine(), "test-model")
    return TestClient(app)


def test_chat_page_renders(client):
    r = client.get("/admin/")
    assert r.status_code == 200
    assert "Big White Rabbit" in r.text


def test_chat_alias_renders(client):
    assert client.get("/admin/chat").status_code == 200


def test_models_endpoint_reports_the_served_model(client):
    body = client.get("/admin/api/models").json()
    assert [m["id"] for m in body["models"]] == ["test-model"]


def test_stats_shape_matches_what_the_page_reads(client):
    """chat.html does `data.active_models?.models || []` then finds by id."""
    body = client.get("/admin/api/stats").json()
    models = body["active_models"]["models"]
    assert models[0]["id"] == "test-model"
    assert models[0]["decode_calls"] == 7


def test_global_settings_disables_auth(client):
    """bwr does not check an API key; the UI must not collect or send one."""
    assert client.get("/admin/api/global-settings").json()["auth_enabled"] is False


def test_models_status_shape(client):
    body = client.get("/v1/models/status").json()
    assert body["models"][0]["model_type"] == "llm"


@pytest.mark.parametrize(
    "path,key",
    [
        ("/admin/api/cluster/deployments", "deployments"),
        ("/v1/mcp/tools", "tools"),
    ],
)
def test_unsupported_surfaces_answer_empty_not_404(client, path, key):
    """200 + empty so the page degrades instead of erroring in the console."""
    r = client.get(path)
    assert r.status_code == 200
    assert r.json()[key] == []


def test_update_check_reports_no_channel(client):
    assert client.get("/admin/api/update-check").json()["update_available"] is False


def test_static_assets_served(client):
    assert client.get("/admin/static/favicon.svg").status_code == 200


# -- rebrand guards ----------------------------------------------------------


def test_served_page_carries_no_licensor_marks(client):
    """Apache-2.0 grants no trademark licence (section 6): the derivative
    must not ship under oMLX's name."""
    assert "omlx" not in client.get("/admin/").text.lower()


def test_logo_is_ours_not_theirs():
    svg = (pathlib.Path(STATIC_DIR) / "favicon.svg").read_text()
    assert "Big White Rabbit" in svg


def test_i18n_carries_no_licensor_service_instructions():
    """Renaming a string like `pip install "omlx[audio]"` would ship an
    instruction that does not work; those keys are dropped instead."""
    strings = json.loads((PKG / "i18n" / "en.json").read_text())
    offending = [k for k, v in strings.items() if isinstance(v, str) and "omlx" in v.lower()]
    assert not offending, f"licensor references left in i18n: {offending[:5]}"


def test_bundled_webfonts_are_not_vendored():
    """~11 MB of webfonts were deliberately dropped for the system stack;
    if a resync drags them back in, the package balloons silently."""
    assert not (pathlib.Path(STATIC_DIR) / "fonts").exists()


def test_attribution_is_present():
    """Apache-2.0 section 4: retain the licence and state the changes."""
    repo = PKG.parents[3]
    assert (repo / "vendor" / "LICENSE.omlx-Apache-2.0").is_file()
    doc = (PKG / "__init__.py").read_text()
    assert "Apache-2.0" in doc and "jundot/omlx" in doc


# -- pool-backed surfaces ----------------------------------------------------
#
# With a pool the model-management screens become real: list reflects actual
# residency, and load/unload act on it. Without one they must refuse clearly
# rather than pretend, because a single-model server has nothing to switch.


class _PoolStub:
    def __init__(self):
        self._models = {"a": False, "b": False}
        self.budget_bytes = 64 * 1024**3
        self.unloaded: list[str] = []
        self.model_dir = "/tmp/models"
        self.rescans = 0

    def rescan(self):
        self.rescans += 1
        self._models.setdefault("c", False)

    def list(self):
        return [
            {"id": k, "loaded": v, "kind": "mlx", "size_bytes": 4 * 1024**3,
             "path": f"/tmp/models/{k}"}
            for k, v in self._models.items()
        ]

    @property
    def loaded_ids(self):
        return [k for k, v in self._models.items() if v]

    @property
    def resident_bytes(self):
        return sum(4 * 1024**3 for v in self._models.values() if v)

    def engine_for(self, model_id):
        return _StubEngine() if self._models.get(model_id) else None

    async def acquire(self, model_id):
        if model_id not in self._models:
            raise RuntimeError(f"unknown model {model_id!r}")
        self._models[model_id] = True
        return model_id, object()

    async def unload(self, model_id):
        was = self._models.get(model_id, False)
        self._models[model_id] = False
        self.unloaded.append(model_id)
        return was


@pytest.fixture
def pool_client(tmp_path):
    from bwr.server.profiles import ProfileStore
    from bwr.server.stats import ServerStats

    app = FastAPI()
    pool = _PoolStub()
    stats = ServerStats()
    mount(app, _StubEngine(), "a", pool, stats=stats,
          profiles=ProfileStore(tmp_path))
    return TestClient(app), pool


def test_pool_models_list_reflects_residency(pool_client):
    client, pool = pool_client
    before = {m["id"]: m["loaded"] for m in client.get("/admin/api/models").json()["models"]}
    assert before == {"a": False, "b": False}
    client.post("/admin/api/models/b/load")
    after = {m["id"]: m["loaded"] for m in client.get("/admin/api/models").json()["models"]}
    assert after == {"a": False, "b": True}


def test_pool_load_and_unload_roundtrip(pool_client):
    client, pool = pool_client
    assert client.post("/admin/api/models/a/load").json()["loaded"] == ["a"]
    assert client.post("/admin/api/models/a/unload").json()["loaded"] == []
    assert pool.unloaded == ["a"]


def test_pool_load_unknown_model_is_400_with_reason(pool_client):
    client, _ = pool_client
    r = client.post("/admin/api/models/nope/load")
    assert r.status_code == 400
    assert "unknown model" in r.json()["detail"]


def test_pool_stats_report_memory_budget(pool_client):
    client, _ = pool_client
    client.post("/admin/api/models/a/load")
    body = client.get("/admin/api/stats").json()
    assert [m["id"] for m in body["active_models"]["models"]] == ["a"]
    assert body["memory"]["budget_bytes"] == 64 * 1024**3


def test_pool_stats_never_trigger_a_load(pool_client):
    """The dashboard polls this; it must not pull 18 GiB off disk."""
    client, pool = pool_client
    client.get("/admin/api/stats")
    assert pool.loaded_ids == []


def test_single_model_server_refuses_load_clearly(client):
    """No pool: say so, rather than silently succeeding."""
    r = client.post("/admin/api/models/anything/load")
    assert r.status_code == 409
    assert "single-model" in r.json()["detail"]


# -- dashboard surfaces ------------------------------------------------------


def test_device_info_reports_the_real_host(client):
    """bwr already probes the machine; the dashboard shows that, not a stub."""
    body = client.get("/admin/api/device-info").json()
    assert "chip" in body and "memory_bytes" in body
    assert body["memory_bytes"] > 0


def test_model_settings_are_marked_read_only(client):
    """EngineConfig is fixed at startup, so per-model editing would be a lie."""
    body = client.get("/admin/api/models/whatever/settings").json()
    assert body["read_only"] is True
    assert "n_ctx" in body["settings"]


@pytest.mark.parametrize(
    "path",
    ["hf/models", "bench/active", "grammar/parsers", "ms/recommended"],
)
def test_unsupported_families_answer_empty_not_404(client, path):
    """The vendored dashboard polls ~40 oMLX endpoints. 404-ing them all
    leaves the page throwing; these answer empty so panels render blank."""
    r = client.get(f"/admin/api/{path}")
    assert r.status_code == 200
    assert r.json()["supported"] is False


def test_unsupported_writes_are_501_not_a_fake_success(client):
    """Pretending a quantization job started is worse than saying it cannot.

    This used to POST /bench/start, which has a real handler now -- the
    assertion has to point at a family bwr genuinely has no equivalent for.
    """
    r = client.post("/admin/api/oq/start")
    assert r.status_code == 501
    assert r.json()["supported"] is False


def test_unknown_path_outside_those_families_still_404s(client):
    """The fallback is a scoped family list, not a blanket catch-all, so a
    typo in a real endpoint is still a visible bug."""
    assert client.get("/admin/api/modelz").status_code == 404


@pytest.mark.parametrize("path", ["models", "stats", "server-info", "global-settings"])
def test_real_endpoints_are_not_shadowed_by_the_fallback(client, path):
    assert client.get(f"/admin/api/{path}").status_code == 200


# -- found by exercising the live UI ------------------------------------------


def test_dashboard_preset_asset_is_vendored():
    """dashboard.js fetches /admin/static/bwr_preset.json. The first vendoring
    pass copied js/css/img and the brand SVGs but missed this one, so the
    preset panel 404'd against a live server."""
    preset = pathlib.Path(STATIC_DIR) / "bwr_preset.json"
    assert preset.is_file(), "bwr_preset.json was not vendored"
    body = json.loads(preset.read_text())
    assert body, "preset file is empty"
    assert "omlx" not in preset.read_text().lower()


@pytest.mark.parametrize(
    "path",
    ["logout", "reload", "server/restart", "ssd-cache/clear", "stats/clear",
     "upload/tasks", "web-search/test"],
)
def test_families_the_dashboard_polls_are_all_handled(client, path):
    """Walking every fetch() in the bundled JS against a live server turned up
    seven families with no handler at all, which 404'd instead of reporting
    themselves unsupported."""
    r = client.get(f"/admin/api/{path}")
    assert r.status_code == 200, f"/admin/api/{path} is unhandled"
    assert r.json()["supported"] is False


def test_stats_family_does_not_shadow_the_real_stats_endpoint(client):
    """'stats' is in the unsupported list so stats/clear is handled, but
    /admin/api/stats itself is real. Exact routes register first and win --
    if that ever changes, the dashboard loses its numbers."""
    body = client.get("/admin/api/stats").json()
    assert "active_models" in body
    assert body.get("supported") is not False


def test_audio_transcription_reports_unsupported(client):
    """The chat page offers mic input; bwr serves text models only. It lives
    under /v1, outside the /admin fallback, so it needs its own handler."""
    r = client.post("/v1/audio/transcriptions")
    assert r.status_code == 501
    assert r.json()["supported"] is False


def test_model_entries_carry_a_cluster_shape(client):
    """The dashboard dereferences `m.cluster.live` unguarded. bwr is
    single-node, so `live` must be present-and-null rather than absent --
    absent throws a TypeError in the template."""
    for m in client.get("/admin/api/models").json()["models"]:
        assert "cluster" in m, "cluster key missing -> template TypeError"
        assert m["cluster"]["live"] is None


def test_model_settings_carry_mtp_capability_flags(client):
    """_modal_model_settings.html calls
    `modelSettings.mtp_compatibility_reason.includes(...)` whenever
    mtp_compatible is falsy, so the reason must be a STRING, not absent."""
    body = client.get("/admin/api/models/x/settings").json()
    assert body["mtp_compatible"] is False
    assert isinstance(body["mtp_compatibility_reason"], str)


def test_usage_family_is_handled(client):
    """The dashboard polls /admin/api/usage?range=today on load.

    Once a stub; now a real handler backed by the request counters, so the
    assertion is that it answers a usage SHAPE rather than an empty one. Both
    branches exist: this `client` fixture mounts without an accumulator, which
    is what a caller that built the router directly gets.
    """
    r = client.get("/admin/api/usage?range=today&model=")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False, "no accumulator was mounted"
    assert body["totals"]["requests"] == 0


def test_vendored_js_initialises_the_mtp_fields():
    """Upstream's initial modelSettings omitted mtp_compatible /
    mtp_compatibility_reason, so the modal's x-show threw on first render,
    before any model was loaded. Patched in the vendored JS; a resync must
    not drop it."""
    js = (pathlib.Path(STATIC_DIR) / "js" / "dashboard.js").read_text()
    head = js[: js.index("model_alias: ''")]
    assert "mtp_compatibility_reason: ''" in head, (
        "initial modelSettings lost its mtp fields; the settings modal will "
        "throw on first render again"
    )


# -- cross-language contract with the macOS app -------------------------------


def test_server_info_carries_every_key_the_macos_dto_requires(client):
    """`apps/bwr-mac/Sources/Net/DTO/ServerInfoDTO.swift` declares host, port
    and aliases as NON-optional, so Swift's synthesised init(from:) throws
    keyNotFound on a payload without them -- the whole response is discarded,
    not just the missing field.

    bwr's server-info answered with name/version/uptime_s/model_count and none
    of those three, so `BWRClient.getServerInfo()` could never succeed against
    a real bwr server. It has no caller today, which is the only reason this
    was latent rather than a visible failure.

    If the Swift DTO gains a required field, this test is where the two
    languages are supposed to disagree loudly.
    """
    body = client.get("/admin/api/server-info").json()
    for key in ("host", "port", "aliases"):
        assert key in body, f"ServerInfoDTO requires {key!r}; decode would throw"
    assert isinstance(body["host"], str) and body["host"]
    assert isinstance(body["port"], int)
    assert isinstance(body["aliases"], list)
    assert all(isinstance(a, str) for a in body["aliases"])
    # The bwr-specific keys the dashboard reads must survive the addition.
    for key in ("name", "version", "uptime_s", "model_count", "backend"):
        assert key in body


def test_server_info_aliases_name_the_loopback(client):
    """The chips the app renders are connect URLs, so the aliases must be
    things a client can actually dial, and must not repeat."""
    aliases = client.get("/admin/api/server-info").json()["aliases"]
    assert "127.0.0.1" in aliases
    assert "localhost" in aliases
    assert len(aliases) == len(set(aliases)), "duplicate aliases render duplicate chips"


def test_api_status_is_served_at_the_root(client):
    """MenubarStatsPoller polls /api/status (NOT under /admin) every few
    seconds. It 404'd on every tick, so the menubar showed no activity."""
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "active_models" in r.json()


# -- the macOS app's decode contract -----------------------------------------
#
# Swift's Decodable fails the WHOLE value on one missing non-optional key, so
# a response that is merely incomplete does not degrade a screen -- it blanks
# it. The Swift side pins this against captured fixtures
# (apps/bwr-mac/.../BWRBackendContractTests.swift); these assert the same
# contract from the Python side, where a handler is actually edited.


def test_model_entries_carry_the_keys_modeldto_requires(pool_client):
    client, _ = pool_client
    for entry in client.get("/admin/api/models").json()["models"]:
        for key in ("id", "loaded", "is_loading", "estimated_size"):
            assert key in entry, f"ModelDTO requires {key}; the list decodes to nothing without it"


def test_stats_carry_the_keys_statsdto_requires(pool_client):
    client, _ = pool_client
    body = client.get("/admin/api/stats").json()
    for key in ("total_tokens_served", "total_cached_tokens", "cache_efficiency",
                "total_prompt_tokens", "total_completion_tokens", "total_requests",
                "avg_prefill_tps", "avg_generation_tps", "uptime_seconds",
                "active_models"):
        assert key in body, f"StatsDTO requires {key}"


def test_global_settings_carry_the_nested_server_block(pool_client):
    client, _ = pool_client
    body = client.get("/admin/api/global-settings").json()
    assert "server" in body, "GlobalSettingsDTO.server is non-optional"
    for key in ("host", "port", "log_level", "server_aliases"):
        assert key in body["server"], f"ServerSettings requires {key}"
    assert body["auth"]["api_key_set"] is False


def test_logs_are_one_string_not_a_list(pool_client):
    client, _ = pool_client
    body = client.get("/admin/api/logs").json()
    # LogsDTO declares `logs: String`. A list here is a type mismatch that
    # discards the response and blanks the Logs screen.
    assert isinstance(body["logs"], str)
    assert isinstance(body["total_lines"], int)
    assert isinstance(body["log_file"], str)
    assert isinstance(body["available_files"], list)


def test_usage_reports_available_and_totals(pool_client):
    client, _ = pool_client
    body = client.get("/admin/api/usage").json()
    assert body["available"] is True
    assert body["dropped_requests"] == 0
    for key in ("requests", "total_tokens", "prompt_tokens", "completion_tokens",
                "cached_tokens", "cache_efficiency"):
        assert key in body["totals"]
    assert isinstance(body["heatmap"], list)


# -- settings write path ------------------------------------------------------


def test_global_settings_patch_succeeds_instead_of_404(pool_client):
    """The Server screen sends this patch in the same do-block as its local
    port/base-path work. A throw here aborted changes the user had already
    confirmed, so the endpoint has to answer even for keys bwr cannot act on.
    """
    client, _ = pool_client
    r = client.post("/admin/api/global-settings", json={"auto_start_on_launch": True})
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert r.json()["runtime_applied"] == []


def test_global_settings_patch_applies_log_level(pool_client):
    import logging
    client, _ = pool_client
    before = logging.getLogger("bwr").level
    try:
        r = client.post("/admin/api/global-settings", json={"log_level": "warning"})
        assert r.json()["runtime_applied"] == ["log_level"]
        assert logging.getLogger("bwr").level == logging.WARNING
    finally:
        logging.getLogger("bwr").setLevel(before)


def test_global_settings_patch_rejects_an_unknown_log_level(pool_client):
    client, _ = pool_client
    r = client.post("/admin/api/global-settings", json={"log_level": "chatty"})
    assert r.status_code == 400
    assert r.json()["success"] is False


# -- reload -------------------------------------------------------------------


def test_reload_rescans_the_pool(pool_client):
    client, pool = pool_client
    r = client.post("/admin/api/reload")
    assert r.status_code == 200
    assert pool.rescans == 1
    assert r.json()["added"] == ["c"]


def test_reload_without_a_pool_refuses_clearly(client):
    r = client.post("/admin/api/reload")
    assert r.status_code == 409


# -- profiles and templates ---------------------------------------------------


def test_templates_list_includes_the_builtins(pool_client):
    client, _ = pool_client
    names = {t["name"] for t in client.get("/admin/api/profile-templates").json()["templates"]}
    assert {"precise", "balanced", "creative"} <= names


def test_template_crud_round_trip(pool_client):
    client, _ = pool_client
    r = client.post("/admin/api/profile-templates",
                    json={"name": "mine", "display_name": "Mine",
                          "settings": {"top_p": 0.5}})
    assert r.status_code == 200
    assert r.json()["template"]["settings"] == {"top_p": 0.5}

    r = client.put("/admin/api/profile-templates/mine", json={"display_name": "Ours"})
    assert r.json()["template"]["display_name"] == "Ours"

    assert client.delete("/admin/api/profile-templates/mine").json()["deleted"] is True


def test_editing_a_builtin_template_is_refused(pool_client):
    client, _ = pool_client
    assert client.put("/admin/api/profile-templates/precise",
                      json={"display_name": "x"}).status_code == 400
    assert client.delete("/admin/api/profile-templates/precise").status_code == 400


def test_profile_crud_round_trip(pool_client):
    client, _ = pool_client
    r = client.post("/admin/api/models/a/profiles",
                    json={"name": "fast", "settings": {"temperature": 0.1}})
    assert r.status_code == 200, r.text
    body = r.json()["profile"]
    assert body["settings"] == {"temperature": 0.1}
    assert body["model_id"] == "a:fast"
    assert body["has_engine_fields"] is False

    assert [p["name"] for p in
            client.get("/admin/api/models/a/profiles").json()["profiles"]] == ["fast"]

    r = client.put("/admin/api/models/a/profiles/fast",
                   json={"settings": {"temperature": 0.4}})
    assert r.json()["profile"]["settings"] == {"temperature": 0.4}

    assert client.delete("/admin/api/models/a/profiles/fast").json()["deleted"] is True


def test_a_profile_with_engine_fields_says_so(pool_client):
    """The client renders its reload warning from this flag rather than
    keeping its own copy of the field list."""
    client, _ = pool_client
    r = client.post("/admin/api/models/a/profiles",
                    json={"name": "big", "settings": {"n_ctx": 8192}})
    assert r.json()["profile"]["has_engine_fields"] is True


def test_apply_reports_what_takes_effect_now_versus_on_reload(pool_client):
    """bwr fixes engine settings at load, so "apply" cannot mutate a resident
    engine. Saying which half needs a reload beats reporting a success that
    changed nothing."""
    client, _ = pool_client
    client.post("/admin/api/models/a/profiles",
                json={"name": "mixed",
                      "settings": {"temperature": 0.2, "n_ctx": 8192}})
    body = client.post("/admin/api/models/a/profiles/mixed/apply").json()
    assert body["applied_now"] == {"temperature": 0.2}
    assert body["requires_reload"] is True


def test_applying_an_unknown_profile_is_404(pool_client):
    client, _ = pool_client
    assert client.post("/admin/api/models/a/profiles/nope/apply").status_code == 404


def test_a_bad_profile_name_is_400_not_500(pool_client):
    client, _ = pool_client
    r = client.post("/admin/api/models/a/profiles", json={"name": "has space"})
    assert r.status_code == 400


def test_profile_surfaces_are_empty_not_broken_without_a_store(client):
    """The single-model fixture mounts no store: the screens must render
    empty rather than error."""
    assert client.get("/admin/api/profile-templates").json()["templates"] == []
    assert client.get("/admin/api/models/x/profiles").json()["profiles"] == []


# -- downloads ---------------------------------------------------------------


@pytest.fixture
def download_client(tmp_path):
    """A client whose download manager never touches the network."""
    from bwr.server.downloads import DownloadManager

    class _Hub:
        def search(self, q, limit=30):
            return [{"repo_id": f"org/{q or 'x'}", "name": q or "x"}]

        def recommended(self, limit=20):
            return {"trending": [{"repo_id": "org/hot"}], "popular": []}

        def model_info(self, repo_id):
            return {"repo_id": repo_id, "size": 1} if repo_id == "org/known" else None

    app = FastAPI()
    pool = _PoolStub()
    mount(app, _StubEngine(), "a", pool,
          downloads=DownloadManager(tmp_path, pool=pool), hub=_Hub())
    return TestClient(app), tmp_path


def test_download_tasks_start_empty(download_client):
    client, _ = download_client
    assert client.get("/admin/api/hf/tasks").json()["tasks"] == []


def test_starting_a_download_returns_a_task_row(download_client, monkeypatch):
    client, _ = download_client
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: k["local_dir"])
    r = client.post("/admin/api/hf/download", json={"repo_id": "org/model"})
    assert r.status_code == 200
    task = r.json()["task"]
    assert task["repo_id"] == "org/model"
    for key in ("task_id", "status", "progress", "total_size", "downloaded_size",
                "error", "created_at", "started_at", "completed_at", "retry_count"):
        assert key in task, f"HFTaskDTO requires {key}"


def test_starting_a_download_without_a_repo_is_400(download_client):
    client, _ = download_client
    assert client.post("/admin/api/hf/download", json={}).status_code == 400


def test_cancelling_an_unknown_task_is_not_a_crash(download_client):
    client, _ = download_client
    assert client.post("/admin/api/hf/cancel/nope").json()["status"] == "not_running"


def test_an_unknown_task_reads_as_404(download_client):
    client, _ = download_client
    assert client.get("/admin/api/hf/task/nope").status_code == 404


def test_deleting_a_model_directory_rescans_the_pool(download_client):
    client, tmp_path = download_client
    (tmp_path / "gone").mkdir()
    r = client.delete("/admin/api/hf/models/gone")
    assert r.json() == {"deleted": True, "name": "gone"}
    assert not (tmp_path / "gone").exists()


def test_deleting_outside_the_model_directory_is_refused(download_client):
    """Two layers stop a traversal and either is enough.

    `..` never reaches the handler -- the URL is normalised first, so it
    misses the route and lands on the unsupported catch-all. An encoded one
    does reach it and hits the manager's own guard. What matters is that
    neither reports success, and that the directory outside survives.
    """
    client, tmp_path = download_client
    outside = tmp_path.parent / "keep-me"
    outside.mkdir(exist_ok=True)
    for path in ("/admin/api/hf/models/..",
                 "/admin/api/hf/models/%2E%2E%2Fkeep-me"):
        assert client.delete(path).status_code != 200, path
    assert outside.is_dir()


def test_hub_browse_surfaces_answer(download_client):
    client, _ = download_client
    assert client.get("/admin/api/hf/search?q=qwen").json()["models"]
    assert client.get("/admin/api/hf/recommended").json()["trending"]
    assert client.get("/admin/api/hf/model-info?repo_id=org/known").status_code == 200
    assert client.get("/admin/api/hf/model-info?repo_id=org/nope").status_code == 404


def test_download_surfaces_are_empty_not_broken_without_a_manager(client):
    assert client.get("/admin/api/hf/tasks").json()["tasks"] == []
    assert client.get("/admin/api/hf/search?q=x").json()["models"] == []
    assert client.post("/admin/api/hf/download", json={"repo_id": "a/b"}).status_code == 400


# -- benchmarks --------------------------------------------------------------


def test_bench_surfaces_refuse_clearly_without_a_runner(client):
    assert client.post("/admin/api/bench/start", json={}).status_code == 400
    assert client.get("/admin/api/bench/nope/results").status_code == 404
    assert client.post("/admin/api/bench/nope/cancel").json()["status"] == "not_running"
