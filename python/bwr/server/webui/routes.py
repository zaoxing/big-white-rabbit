"""Adapter serving the /admin web UI over bwr's engine.

ORIGINAL CODE. The templates and static assets in this package come from oMLX
(Apache-2.0, see `__init__.py`), but nothing here is copied from
`omlx/admin/routes.py` -- this is a small shim that answers the handful of
endpoints the chat page actually calls, backed by bwr's own single-model
engine.

Endpoint status
===============

Live (real bwr data):
  GET  /admin/                      chat UI
  GET  /admin/chat                  chat UI
  GET  /admin/api/models            the one served model
  GET  /admin/api/stats             engine counters from /health's source
  GET  /admin/api/global-settings   minimal, auth disabled
  GET  /v1/models/status            model type for the selector

Deliberately empty / unsupported -- oMLX features bwr has no equivalent for.
They answer 200 with an empty payload rather than 404 so the page degrades
instead of erroring in the console:
  GET  /admin/api/cluster/deployments   bwr is single-node
  GET  /admin/api/update-check          no update channel
  GET  /v1/mcp/tools                    no MCP registry

bwr serves one model per process (see SPEC-mlx-engine.md), so the UI's
model-switching affordances have nothing to switch between; the list always
holds exactly the served model. That is a real architectural difference, not
a stub waiting to be filled.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import ChainableUndefined, Undefined
from starlette.requests import Request

from . import PACKAGE_DIR, STATIC_DIR, TEMPLATES_DIR

_STARTED = time.time()
_I18N_PATH = PACKAGE_DIR / "i18n" / "en.json"


def _load_strings() -> dict[str, str]:
    """The vendored English strings (rebranded). Flat dotted keys.

    English only: oMLX ships nine locales, but shipping translations whose
    brand strings we have rewritten in only one of them would show users a
    half-rebranded UI. One complete locale beats nine inconsistent ones.
    """
    try:
        return json.loads(_I18N_PATH.read_text())
    except (OSError, ValueError):
        return {}


_STRINGS = _load_strings()


def _version() -> str:
    try:
        from ... import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - version is cosmetic here
        return "0"


def _t(key: str, **kwargs: Any) -> str:
    """oMLX's template/JS translation helper.

    Falls back to the key itself, which is what the upstream UI does and
    keeps a missing string visible rather than blanking the element.
    """
    text = _STRINGS.get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def build_router(engine: Any, model_name: str, pool: Any = None) -> APIRouter:
    """Router for /admin.

    `engine` is the raw engine (not the AsyncEngine): the UI only reads
    counters, never admits work through it. `pool`, when present, makes the
    model-management surfaces real -- list/load/unload act on actual
    residency instead of describing a single fixed model.
    """
    router = APIRouter()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # The vendored templates were written against oMLX's much larger context
    # (auth, profiles, cluster, update channel). Rather than enumerate every
    # variable bwr has no analogue for -- and break again whenever upstream
    # adds one -- undefined names chain safely and serialise to JSON null.
    templates.env.undefined = ChainableUndefined
    templates.env.policies["json.dumps_kwargs"] = {
        "default": lambda o: None if isinstance(o, Undefined) else str(o),
        "sort_keys": True,
    }
    # Globals the vendored templates expect from oMLX's app.
    templates.env.globals.update(
        t=_t,
        current_lang="en",
        locale_json=json.dumps(_STRINGS, ensure_ascii=False),
        static=lambda path: f"/admin/static/{path.lstrip('/')}",
    )

    def _page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="chat.html",
            context={
                "request": request,
                "app_name": "Big White Rabbit",
                # bwr's server does not gate on an API key, so the UI must
                # not plant one in localStorage and then send it.
                "api_key": None,
            },
        )

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return _page(request)

    @router.get("/chat", response_class=HTMLResponse)
    async def chat(request: Request) -> HTMLResponse:
        return _page(request)

    # -- surfaces the macOS menubar app drives ----------------------------
    #
    # It opens /admin/dashboard and /admin/auto-login directly, and polls
    # server-info / activity. Auth is a no-op here: bwr checks no key, so
    # login must succeed rather than bounce the app to a form it cannot pass.

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={"request": request, "app_name": "Big White Rabbit", "api_key": None},
        )

    @router.get("/auto-login")
    async def auto_login() -> RedirectResponse:
        return RedirectResponse(url="/admin/", status_code=302)

    @router.post("/api/login")
    async def login() -> JSONResponse:
        return JSONResponse({"status": "ok", "auth_enabled": False})

    @router.get("/api/server-info")
    async def server_info() -> JSONResponse:
        entries = _entries()
        return JSONResponse(
            {
                "name": "Big White Rabbit",
                "version": _version(),
                "uptime_s": time.time() - _STARTED,
                "model_count": len(entries),
                "loaded": [e["id"] for e in entries if e.get("loaded")],
                "backend": "bwr",
            }
        )

    @router.get("/api/activity")
    async def activity() -> JSONResponse:
        """Per-request history. bwr keeps counters, not a request log, so
        this is empty by construction rather than unimplemented -- see
        engine.ctx.decode_calls in /api/stats for what it does track."""
        return JSONResponse({"activity": [], "requests": []})

    def _single_entry() -> dict[str, Any]:
        return {
            "id": model_name,
            "name": model_name,
            "model_type": "llm",
            "loaded": True,
            "status": "loaded",
            "n_ctx": engine.ctx.n_ctx,
            "n_ctx_seq": engine.ctx.n_ctx_seq,
        }

    def _pool_entry(m: dict[str, Any]) -> dict[str, Any]:
        raw = pool.engine_for(m["id"])
        out = {
            "id": m["id"],
            "name": m["id"],
            "model_type": "llm",
            "loaded": m["loaded"],
            "status": "loaded" if m["loaded"] else "available",
            "kind": m["kind"],
            "size_bytes": m["size_bytes"],
        }
        if raw is not None:
            out["n_ctx"] = raw.ctx.n_ctx
            out["n_ctx_seq"] = raw.ctx.n_ctx_seq
        return out

    def _entries() -> list[dict[str, Any]]:
        if pool is None:
            return [_single_entry()]
        return [_pool_entry(m) for m in pool.list()]

    def _model_entry() -> dict[str, Any]:
        return _entries()[0]

    @router.get("/api/models")
    async def api_models() -> JSONResponse:
        return JSONResponse({"models": _entries()})

    @router.post("/api/models/{model_id:path}/load")
    async def api_load(model_id: str) -> JSONResponse:
        """Make a model resident. Idempotent; may evict an LRU peer."""
        if pool is None:
            return JSONResponse(
                {"detail": "single-model server: nothing to load"}, status_code=409
            )
        try:
            mid, _engine = await pool.acquire(model_id)
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the UI
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"status": "ok", "model_id": mid, "loaded": pool.loaded_ids})

    @router.post("/api/models/{model_id:path}/unload")
    async def api_unload(model_id: str) -> JSONResponse:
        """Free a model. Waits for in-flight requests rather than killing them."""
        if pool is None:
            return JSONResponse(
                {"detail": "single-model server: nothing to unload"}, status_code=409
            )
        ok = await pool.unload(model_id)
        return JSONResponse(
            {"status": "ok" if ok else "not_loaded", "loaded": pool.loaded_ids}
        )

    @router.get("/api/stats")
    async def api_stats() -> JSONResponse:
        """Dashboard numbers. Reads only -- never triggers a load."""
        body: dict[str, Any] = {"uptime_s": time.time() - _STARTED}
        if pool is None:
            entry = _single_entry()
            entry.update(
                {
                    "free_seq_slots": engine.n_free_seq_slots,
                    "decode_calls": engine.ctx.decode_calls,
                    "in_flight": engine.n_in_flight,
                }
            )
            body["active_models"] = {"models": [entry]}
            return JSONResponse(body)

        active = []
        for m in pool.list():
            if not m["loaded"]:
                continue
            raw = pool.engine_for(m["id"])
            e = _pool_entry(m)
            if raw is not None:
                e.update(
                    {
                        "free_seq_slots": raw.n_free_seq_slots,
                        "decode_calls": raw.ctx.decode_calls,
                        "in_flight": raw.n_in_flight,
                    }
                )
            active.append(e)
        body["active_models"] = {"models": active}
        body["memory"] = {
            "resident_bytes": pool.resident_bytes,
            "budget_bytes": pool.budget_bytes,
            "loaded": pool.loaded_ids,
        }
        return JSONResponse(body)

    @router.get("/api/global-settings")
    async def api_global_settings() -> JSONResponse:
        # auth_enabled False keeps the UI from prompting for an API key it
        # would then send to a server that does not check one.
        return JSONResponse(
            {
                "auth_enabled": False,
                "app_name": "Big White Rabbit",
                "single_model": True,
            }
        )

    # -- single-node / no-update-channel surfaces --------------------------

    @router.get("/api/cluster/deployments")
    async def api_cluster_deployments() -> JSONResponse:
        return JSONResponse({"deployments": []})

    @router.get("/api/update-check")
    async def api_update_check() -> JSONResponse:
        return JSONResponse(
            {"update_available": False, "latest_version": None, "release_url": None}
        )

    # -- real bwr data -----------------------------------------------------

    @router.get("/api/device-info")
    async def device_info() -> JSONResponse:
        """The host probe bwr already owns (see bwr/host.py)."""
        from ... import host

        caps = host.probe()
        return JSONResponse(
            {
                "machine": caps.machine_model,
                "chip": caps.chip,
                "cpu_count": caps.cpu_count,
                "physical_cpus": caps.phys_cpus,
                "gpu_cores": caps.gpu_cores,
                "memory_bytes": caps.mem_bytes,
                "notes": caps.notes,
            }
        )

    @router.get("/api/models/{model_id:path}/settings")
    async def model_settings(model_id: str) -> JSONResponse:
        """Effective engine settings for one model.

        Read-only: bwr's knobs are process-wide (EngineConfig is fixed at
        startup), so there is nothing per-model to write back. The UI's
        editor is therefore not wired -- PUT falls through to the
        unsupported handler below rather than silently accepting edits that
        would never take effect.
        """
        cfg = getattr(engine, "config", None)
        if pool is not None:
            cfg = getattr(pool, "config", cfg)
        # `read_only` is part of the contract whether or not a config is
        # reachable: a response that sometimes omits it makes the UI decide
        # editability from a missing key.
        if cfg is None:
            return JSONResponse(
                {"model_id": model_id, "read_only": True, "settings": {}}
            )
        return JSONResponse(
            {
                "model_id": model_id,
                "read_only": True,
                "settings": {
                    "n_ctx": cfg.n_ctx,
                    "n_batch": cfg.n_batch,
                    "n_seq_max": cfg.n_seq_max,
                    "speculative": cfg.speculative,
                    "mlx_mtp": getattr(cfg, "mlx_mtp", False),
                    "mlx_prefix_cache": cfg.mlx_prefix_cache,
                },
            }
        )

    # -- oMLX features with no bwr equivalent ------------------------------
    #
    # The vendored dashboard polls ~40 endpoints for benchmark suites, ANE
    # tuning, HuggingFace/ModelScope downloaders, prompt profiles, grammar
    # parsers and a hot cache. None of those exist here, and 404-ing all of
    # them leaves the page throwing in the console.
    #
    # These are matched by an explicit family list rather than a blanket
    # catch-all, so a typo in a REAL endpoint still 404s instead of being
    # silently absorbed. GETs answer empty; writes answer 501 with the
    # reason, because pretending a benchmark started would be worse than
    # saying it cannot.

    _UNSUPPORTED = (
        "bench", "hf", "ms", "ane-tune", "profiles", "profile-fields",
        "profile-templates", "grammar", "hot-cache", "logs", "presets",
        "sub-keys", "oq", "cluster",
    )

    def _is_unsupported(path: str) -> bool:
        head = path.strip("/").split("/", 1)[0]
        return head in _UNSUPPORTED

    @router.get("/api/{path:path}")
    async def unsupported_get(path: str) -> JSONResponse:
        if not _is_unsupported(path):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        # Empty in every shape the dashboard destructures.
        return JSONResponse(
            {
                "supported": False,
                "reason": "not provided by the bwr backend",
                "items": [], "models": [], "tasks": [], "results": [],
                "logs": [], "parsers": [], "profiles": [], "queue": [],
                "active": None, "status": "unsupported",
            }
        )

    @router.api_route(
        "/api/{path:path}", methods=["POST", "PUT", "DELETE", "PATCH"]
    )
    async def unsupported_write(path: str) -> JSONResponse:
        if not _is_unsupported(path):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return JSONResponse(
            {
                "detail": f"/{path} is an oMLX feature with no bwr equivalent",
                "supported": False,
            },
            status_code=501,
        )

    return router


def mount(app: Any, engine: Any, model_name: str, pool: Any = None) -> None:
    """Attach the UI at /admin plus the two /v1 helpers the page polls."""
    app.include_router(build_router(engine, model_name, pool), prefix="/admin")
    app.mount(
        "/admin/static", StaticFiles(directory=str(STATIC_DIR)), name="bwr-webui-static"
    )

    @app.get("/v1/models/status")
    async def models_status() -> JSONResponse:
        if pool is not None:
            return JSONResponse(
                {
                    "models": [
                        {"id": m["id"], "model_type": "llm", "loaded": m["loaded"]}
                        for m in pool.list()
                    ]
                }
            )
        return JSONResponse(
            {"models": [{"id": model_name, "model_type": "llm", "loaded": True}]}
        )

    @app.get("/v1/mcp/tools")
    async def mcp_tools() -> JSONResponse:
        return JSONResponse({"tools": []})
