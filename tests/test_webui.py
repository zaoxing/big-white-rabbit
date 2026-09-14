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

    def list(self):
        return [
            {"id": k, "loaded": v, "kind": "mlx", "size_bytes": 4 * 1024**3}
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
def pool_client():
    app = FastAPI()
    pool = _PoolStub()
    mount(app, _StubEngine(), "a", pool)
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
    ["logs", "hf/models", "bench/active", "grammar/parsers", "ms/recommended"],
)
def test_unsupported_families_answer_empty_not_404(client, path):
    """The vendored dashboard polls ~40 oMLX endpoints. 404-ing them all
    leaves the page throwing; these answer empty so panels render blank."""
    r = client.get(f"/admin/api/{path}")
    assert r.status_code == 200
    assert r.json()["supported"] is False


def test_unsupported_writes_are_501_not_a_fake_success(client):
    """Pretending a benchmark started is worse than saying it cannot."""
    r = client.post("/admin/api/bench/start")
    assert r.status_code == 501
    assert r.json()["supported"] is False


def test_unknown_path_outside_those_families_still_404s(client):
    """The fallback is a scoped family list, not a blanket catch-all, so a
    typo in a real endpoint is still a visible bug."""
    assert client.get("/admin/api/modelz").status_code == 404


@pytest.mark.parametrize("path", ["models", "stats", "server-info", "global-settings"])
def test_real_endpoints_are_not_shadowed_by_the_fallback(client, path):
    assert client.get(f"/admin/api/{path}").status_code == 200
