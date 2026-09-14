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
import logging
import time
from collections import deque
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import ChainableUndefined, Undefined
from starlette.requests import Request

from . import PACKAGE_DIR, STATIC_DIR, TEMPLATES_DIR
from ..ane import probe as ane_probe
from ..bench import BenchError
from ..ctxbench import ContextBenchError
from ..downloads import DownloadError
from ..profiles import ProfileError


class _Unavailable(RuntimeError):
    """A surface that needs state this server was not started with."""


async def _json_body(request: Request) -> dict[str, Any]:
    """Request body as a dict; `{}` for an absent or non-object body.

    Every profile write treats "no field" as "leave it alone", so a missing
    body is a no-op patch rather than a 400 -- the same shape the client
    sends when it omits its nil fields.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - absent or malformed body is not fatal
        return {}
    return body if isinstance(body, dict) else {}

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


def _aliases(host: str) -> list[str]:
    """Addresses a client on this machine can dial, for the macOS app's
    connect-URL chips. Order is stable and duplicates are dropped, because the
    app renders one chip per entry.

    The hostname probe is best-effort: a Mac with no DNS name still has a
    working loopback, and failing to resolve one must not blank the whole
    server-info response.
    """
    names = ["localhost", "127.0.0.1"]
    try:
        import socket

        local = socket.gethostname()
        if local:
            names.append(local)
            if not local.endswith(".local"):
                names.append(f"{local}.local")
    except OSError:  # noqa: BLE001 - no hostname is not an error worth raising
        pass
    if host:
        names.append(host)
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


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


class _LogBuffer(logging.Handler):
    """Last N formatted log lines, for GET /admin/api/logs.

    The Logs screen wants the server's log. bwr's own process does not own a
    log FILE -- when the macOS app runs it, the Swift parent captures the
    child's stdout/stderr into
    ~/Library/Application Support/BigWhiteRabbit/logs/server.log, and a bare
    `bwr serve` in a terminal writes to a terminal. Rather than guess at a
    path that may not exist, the server reports what it can prove: the
    records it emitted, kept in memory.

    `logs` is returned as ONE string, not a list. LogsDTO declares
    `logs: String`, and Swift's decoder fails the whole response on a type
    mismatch -- an array here blanked the entire Logs screen rather than
    showing it unstyled.
    """

    CAPACITY = 2000

    def __init__(self) -> None:
        super().__init__()
        self.records: deque[str] = deque(maxlen=self.CAPACITY)
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never raise into a caller
            pass


LOG_BUFFER = _LogBuffer()
# uvicorn.access is deliberately included: "which requests arrived" is most of
# what makes a server log worth reading.
_LOGGED = ("bwr", "uvicorn", "uvicorn.error", "uvicorn.access")


def install_log_buffer() -> None:
    """Attach the buffer once per process, idempotently."""
    for name in _LOGGED:
        lg = logging.getLogger(name)
        if LOG_BUFFER not in lg.handlers:
            lg.addHandler(LOG_BUFFER)


def build_router(
    engine: Any, model_name: str, pool: Any = None, stats: Any = None,
    profiles: Any = None, downloads: Any = None, hub: Any = None,
    bench: Any = None, ms_downloads: Any = None, ms_index: Any = None,
    ctxbench: Any = None,
) -> APIRouter:
    """Router for /admin.

    `engine` is the raw engine (not the AsyncEngine): the UI only reads
    counters, never admits work through it. `pool`, when present, makes the
    model-management surfaces real -- list/load/unload act on actual
    residency instead of describing a single fixed model. `stats` is the
    request accumulator owned by server/app.py; without it /api/stats can
    only report what it can see from the engine.
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
    async def server_info(request: Request) -> JSONResponse:
        entries = _entries()
        # host/port/aliases are for the macOS app: ServerInfoDTO declares all
        # three non-optional, so a payload without them makes Swift's decode
        # throw and discard the WHOLE response, not just the missing keys.
        # Derived from the request so they name the address the client
        # actually reached us on rather than whatever we think we bound.
        host = request.url.hostname or "127.0.0.1"
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        return JSONResponse(
            {
                "name": "Big White Rabbit",
                "version": _version(),
                "uptime_s": time.time() - _STARTED,
                "model_count": len(entries),
                "loaded": [e["id"] for e in entries if e.get("loaded")],
                "backend": "bwr",
                "host": host,
                "port": port,
                "aliases": _aliases(host),
            }
        )

    @router.get("/api/activity")
    async def activity() -> JSONResponse:
        """Per-request history. bwr keeps counters, not a request log, so
        this is empty by construction rather than unimplemented -- see
        engine.ctx.decode_calls in /api/stats for what it does track."""
        return JSONResponse({"activity": [], "requests": []})

    # The dashboard dereferences `m.cluster.live` unguarded. bwr is
    # single-node, so `live` is null -- present, so the expression
    # short-circuits instead of throwing on undefined.
    _NO_CLUSTER = {"live": None, "enabled": False}

    def _human_bytes(n: int) -> str:
        step = 1024.0
        value = float(n)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < step or unit == "TB":
                return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            value /= step
        return f"{value:.1f} TB"

    # ModelDTO declares id/loaded/is_loading/estimated_size non-optional, and
    # Swift's Decodable fails the WHOLE list on one missing key -- omitting
    # `is_loading` blanked the Models screen rather than dimming one badge.
    def _native_ctx_single() -> int | None:
        """The served model's trained context length, if the engine exposes it.

        MetalEngine carries a llama.cpp `Model` with `n_ctx_train`; MLXEngine
        does not surface the config value, so this answers None there rather
        than reporting the serving window as if it were the model's own.
        """
        model = getattr(engine, "model", None)
        value = getattr(model, "n_ctx_train", None)
        return int(value) if isinstance(value, int) and value > 0 else None

    def _single_entry() -> dict[str, Any]:
        return {
            "id": model_name,
            "name": model_name,
            "display_name": model_name,
            "model_type": "llm",
            "loaded": True,
            # bwr loads synchronously inside the request that needs the
            # model, so there is no observable in-between state to report.
            "is_loading": False,
            "status": "loaded",
            "cluster": dict(_NO_CLUSTER),
            "estimated_size": 0,
            "n_ctx": engine.ctx.n_ctx,
            "n_ctx_seq": engine.ctx.n_ctx_seq,
            # The model's OWN trained context length, which is not n_ctx (what
            # this server admits). The Context Bench screen filters its target
            # presets by it, so reporting the serving window here would hide
            # every target above it on a model that can reach them.
            "model_context_length": _native_ctx_single(),
        }

    def _pool_entry(m: dict[str, Any]) -> dict[str, Any]:
        raw = pool.engine_for(m["id"])
        size = int(m.get("size_bytes") or 0)
        out = {
            "id": m["id"],
            "name": m["id"],
            "display_name": m["id"],
            "model_path": m.get("path"),
            "model_type": "llm",
            "loaded": m["loaded"],
            "is_loading": False,
            "status": "loaded" if m["loaded"] else "available",
            "cluster": dict(_NO_CLUSTER),
            "kind": m["kind"],
            "size_bytes": size,
            # On-disk weight bytes. bwr mmaps, so this is the honest size of
            # the model -- not an RSS reading, which counts only the pages
            # the kernel happens to have faulted in.
            "estimated_size": size,
            "estimated_size_formatted": _human_bytes(size),
            "actual_size": size if m["loaded"] else None,
        }
        if raw is not None:
            out["n_ctx"] = raw.ctx.n_ctx
            out["n_ctx_seq"] = raw.ctx.n_ctx_seq
        # Optional on the client: absent means "unknown", which is the honest
        # answer for a GGUF nobody has loaded yet (only llama.cpp parses its
        # metadata). Unknown widens the target list rather than narrowing it,
        # and the benchmark caps at the real value anyway.
        native = pool.native_ctx(m["id"])
        if native:
            out["model_context_length"] = native
        override = pool.ctx_override(m["id"])
        if override:
            # A measured window that is configured but not yet live -- the
            # engine resident right now was built before it was applied.
            out["configured_n_ctx"] = override
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

    def _stats_totals() -> dict[str, Any]:
        """The request-accounting half of /api/stats.

        StatsDTO declares total_tokens_served, total_requests, the two tps
        averages and uptime_seconds NON-optional. A response missing any of
        them decodes to nothing at all in Swift, which is why this returns
        the full set (zeros before the first request) rather than only the
        keys that happen to be interesting.
        """
        if stats is not None:
            return dict(stats.snapshot())
        # No accumulator (a caller that built the router directly). Report
        # zeros in the right shape rather than omitting the keys.
        return {
            "total_requests": 0, "total_prompt_tokens": 0,
            "total_completion_tokens": 0, "total_tokens_served": 0,
            "total_cached_tokens": 0, "cache_efficiency": 0.0,
            "avg_prefill_tps": 0.0, "avg_generation_tps": 0.0,
            "uptime_seconds": time.time() - _STARTED, "persisted": False,
        }

    @router.post("/api/stats/clear")
    @router.post("/api/stats/clear-alltime")
    async def api_stats_clear() -> JSONResponse:
        """Reset the counters.

        One handler for both paths on purpose: bwr keeps no stats database,
        so "session" and "all time" are the same numbers (see server/stats.py)
        and pretending otherwise would make the two buttons look like they do
        different things.
        """
        if stats is not None:
            stats.clear()
        return JSONResponse({"status": "ok", "persisted": False})

    @router.get("/api/stats")
    async def api_stats(request: Request) -> JSONResponse:
        """Dashboard numbers. Reads only -- never triggers a load."""
        body: dict[str, Any] = {"uptime_s": time.time() - _STARTED}
        body.update(_stats_totals())
        body["host"] = request.url.hostname or "127.0.0.1"
        body["port"] = request.url.port or 1919
        # bwr does not gate on an API key; empty string is what the
        # Integrations command builders substitute for "no key needed".
        body["api_key"] = ""
        body["cli_prefix"] = "bwr"
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
    async def api_global_settings(request: Request) -> JSONResponse:
        # auth_enabled False keeps the UI from prompting for an API key it
        # would then send to a server that does not check one.
        #
        # The nested `server` block is required, not decoration:
        # GlobalSettingsDTO.server is non-optional and its own host/port/
        # log_level/server_aliases are too, so a flat payload decoded to
        # nothing and left the Server screen empty.
        host = request.url.hostname or "127.0.0.1"
        port = request.url.port or 1919
        model_dirs = [str(pool.model_dir)] if pool is not None else []
        return JSONResponse(
            {
                "auth_enabled": False,
                "app_name": "Big White Rabbit",
                "single_model": pool is None,
                "server": {
                    "host": host,
                    "port": port,
                    "log_level": logging.getLevelName(
                        logging.getLogger("bwr").getEffectiveLevel()
                    ).lower(),
                    "server_aliases": _aliases(host),
                    "sse_keepalive_mode": "chunk",
                    "auto_start_on_launch": None,
                    "max_audio_upload_size": None,
                },
                "model": {
                    "model_dirs": model_dirs,
                    "model_dir": model_dirs[0] if model_dirs else None,
                    "model_fallback": False,
                },
                "scheduler": {
                    "max_concurrent_requests": getattr(
                        getattr(engine, "ctx", None), "n_seq_max", 1
                    ),
                },
                # api_key_set is non-optional in AuthSettings; bwr checks no
                # key, so it is False rather than absent.
                "auth": {"auth_enabled": False, "api_key_set": False},
            }
        )

    # Settings bwr can apply at runtime. Everything else in the patch belongs
    # to the LAUNCH configuration -- host, port, model directory are fixed by
    # the argv the process was started with -- and is the macOS app's to
    # persist in settings.json and apply on the next start.
    #
    # This has to answer rather than 404: ServerScreenVM sends the patch and
    # its local storage/port work in the SAME do-block, so a throw here
    # aborted changes the user had already confirmed. The response says which
    # keys took effect so the caller can tell applied from stored.
    _RUNTIME_APPLIABLE = ("log_level",)

    @router.post("/api/global-settings")
    async def api_update_global_settings(request: Request) -> JSONResponse:
        try:
            patch = await request.json()
        except Exception:  # noqa: BLE001 - a malformed body is a 400, not a 500
            return JSONResponse(
                {"success": False, "message": "body is not JSON"}, status_code=400
            )
        if not isinstance(patch, dict):
            return JSONResponse(
                {"success": False, "message": "body must be an object"},
                status_code=400,
            )
        applied: list[str] = []
        level = patch.get("log_level")
        if isinstance(level, str) and level:
            resolved = logging.getLevelName(level.upper())
            if isinstance(resolved, int):
                logging.getLogger("bwr").setLevel(resolved)
                applied.append("log_level")
            else:
                return JSONResponse(
                    {"success": False, "message": f"unknown log level {level!r}"},
                    status_code=400,
                )
        stored = [k for k in patch if k not in _RUNTIME_APPLIABLE]
        return JSONResponse(
            {
                "success": True,
                "runtime_applied": applied,
                "message": (
                    "applied " + ", ".join(applied) if applied else
                    "no runtime-applicable keys in patch"
                ) + (
                    f"; {len(stored)} key(s) are launch configuration and take "
                    "effect when the server restarts" if stored else ""
                ),
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

    @router.get("/api/logs")
    async def api_logs(lines: int = 200, file: str = "") -> JSONResponse:
        """The server's own log records, newest last.

        `file` is accepted and ignored: bwr has exactly one stream, so there
        is nothing to switch between, and rejecting the parameter would make
        the Logs screen's file picker an error instead of a no-op.
        """
        buffered = list(LOG_BUFFER.records)
        take = max(1, min(int(lines or 200), _LogBuffer.CAPACITY))
        return JSONResponse(
            {
                # One string: LogsDTO declares `logs: String`, and an array
                # here fails the decode and blanks the screen.
                "logs": "\n".join(buffered[-take:]),
                "total_lines": len(buffered),
                "log_file": "<in-process buffer>",
                "available_files": [],
            }
        )

    # -- throughput benchmarks --------------------------------------------
    #
    # Runs go through the ordinary serving path, so the numbers include what
    # a real request pays. See server/bench.py.

    @router.post("/api/bench/start")
    async def api_bench_start(request: Request) -> JSONResponse:
        if bench is None:
            return JSONResponse(
                {"detail": "benchmarks need a model directory"}, status_code=400
            )
        body = await _json_body(request)
        try:
            return JSONResponse(bench.start(
                body.get("model_id") or body.get("modelId") or model_name,
                prompt_lengths=body.get("prompt_lengths") or body.get("promptLengths"),
                generation_length=int(
                    body.get("generation_length")
                    or body.get("generationLength") or 64
                ),
                batch_sizes=body.get("batch_sizes") or body.get("batchSizes"),
                context_profile=body.get("context_profile")
                or body.get("contextProfile"),
            ))
        except (BenchError, ValueError, TypeError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

    # -- context benchmark -------------------------------------------------
    #
    # Registered BEFORE `/api/bench/{bench_id}/...` on purpose: routes match
    # in registration order, and a literal `context` segment competing with a
    # `{bench_id}` placeholder is exactly the kind of ordering that works
    # until someone reorders the file. See server/ctxbench.py.

    @router.post("/api/bench/context/start")
    async def api_ctxbench_start(request: Request) -> JSONResponse:
        if ctxbench is None:
            return JSONResponse(
                {"detail": "the context benchmark needs a model directory"},
                status_code=400,
            )
        body = await _json_body(request)
        try:
            return JSONResponse(ctxbench.start(
                body.get("model_id") or body.get("modelId") or model_name,
                int(
                    body.get("target_tokens")
                    or body.get("targetTokens")
                    or 131072
                ),
            ))
        except (ContextBenchError, ValueError, TypeError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

    @router.get("/api/bench/context/{bench_id}/results")
    async def api_ctxbench_results(bench_id: str) -> JSONResponse:
        body = ctxbench.get(bench_id) if ctxbench is not None else None
        if body is None:
            return JSONResponse(
                {"detail": "unknown context benchmark"}, status_code=404
            )
        return JSONResponse(body)

    @router.post("/api/bench/context/{bench_id}/cancel")
    async def api_ctxbench_cancel(bench_id: str) -> JSONResponse:
        ok = ctxbench.cancel(bench_id) if ctxbench is not None else False
        return JSONResponse(
            {"status": "cancelling" if ok else "not_running", "bench_id": bench_id}
        )

    @router.get("/api/bench/{bench_id}/results")
    async def api_bench_results(bench_id: str) -> JSONResponse:
        body = bench.get(bench_id) if bench is not None else None
        if body is None:
            return JSONResponse({"detail": "unknown benchmark"}, status_code=404)
        return JSONResponse(body)

    @router.post("/api/bench/{bench_id}/cancel")
    async def api_bench_cancel(bench_id: str) -> JSONResponse:
        ok = bench.cancel(bench_id) if bench is not None else False
        return JSONResponse(
            {"status": "cancelling" if ok else "not_running", "bench_id": bench_id}
        )

    # -- Hugging Face downloads -------------------------------------------
    #
    # Downloads land in the serving model directory and the manager rescans
    # the pool when one finishes, so a completed download is servable without
    # a restart. See server/downloads.py.

    _EMPTY_TASKS = {"tasks": []}

    @router.get("/api/hf/tasks")
    async def api_hf_tasks() -> JSONResponse:
        if downloads is None:
            return JSONResponse(_EMPTY_TASKS)
        return JSONResponse({"tasks": downloads.list()})

    @router.post("/api/hf/download")
    async def api_hf_download(request: Request) -> JSONResponse:
        body = await _json_body(request)
        if downloads is None:
            return JSONResponse(
                {"success": False, "detail": "no model directory to download into"},
                status_code=400,
            )
        try:
            task = downloads.start(
                body.get("repo_id") or body.get("repoId") or "",
                token=body.get("hf_token") or body.get("hfToken") or None,
            )
        except DownloadError as exc:
            return JSONResponse({"success": False, "detail": str(exc)}, status_code=400)
        return JSONResponse({"success": True, "task": task})

    @router.post("/api/hf/cancel/{task_id}")
    async def api_hf_cancel(task_id: str) -> JSONResponse:
        if downloads is None:
            return JSONResponse({"status": "unknown"}, status_code=404)
        ok = downloads.cancel(task_id)
        return JSONResponse({"status": "cancelled" if ok else "not_running"})

    @router.post("/api/hf/retry/{task_id}")
    async def api_hf_retry(task_id: str) -> JSONResponse:
        if downloads is None:
            return JSONResponse({"success": False}, status_code=404)
        try:
            task = downloads.retry(task_id)
        except DownloadError as exc:
            return JSONResponse({"success": False, "detail": str(exc)}, status_code=400)
        return JSONResponse({"success": True, "task": task})

    @router.get("/api/hf/task/{task_id}")
    async def api_hf_task(task_id: str) -> JSONResponse:
        task = downloads.get(task_id) if downloads is not None else None
        if task is None:
            return JSONResponse({"detail": "unknown task"}, status_code=404)
        return JSONResponse({"task": task})

    @router.delete("/api/hf/task/{task_id}")
    async def api_hf_forget(task_id: str) -> JSONResponse:
        """Drop the task from the list. Downloaded files are left alone --
        removing a model is `DELETE /hf/models/{name}`, and conflating the
        two would make dismissing a finished row delete the weights."""
        if downloads is None:
            return JSONResponse({"deleted": False}, status_code=404)
        try:
            return JSONResponse({"deleted": downloads.forget(task_id)})
        except DownloadError as exc:
            return JSONResponse({"deleted": False, "detail": str(exc)}, status_code=400)

    @router.delete("/api/hf/models/{name:path}")
    async def api_hf_delete_model(name: str) -> JSONResponse:
        if downloads is None:
            return JSONResponse({"deleted": False}, status_code=404)
        try:
            deleted = downloads.delete_model(name)
        except DownloadError as exc:
            return JSONResponse({"deleted": False, "detail": str(exc)}, status_code=400)
        if deleted and pool is not None:
            pool.rescan()
        return JSONResponse({"deleted": deleted, "name": name})

    @router.get("/api/hf/search")
    async def api_hf_search(q: str = "", limit: int = 30) -> JSONResponse:
        if hub is None:
            return JSONResponse({"models": [], "total": 0})
        models = hub.search(q, limit=limit)
        return JSONResponse({"models": models, "total": len(models)})

    @router.get("/api/hf/recommended")
    async def api_hf_recommended(limit: int = 20) -> JSONResponse:
        if hub is None:
            return JSONResponse({"trending": [], "popular": []})
        return JSONResponse(hub.recommended(limit=limit))

    @router.get("/api/hf/model-info")
    async def api_hf_model_info(repo_id: str = "") -> JSONResponse:
        info = hub.model_info(repo_id) if (hub is not None and repo_id) else None
        if info is None:
            return JSONResponse({"detail": "unknown repo"}, status_code=404)
        return JSONResponse(info)

    # -- ANE tuning: a capability report, not a stub -----------------------
    #
    # bwr executes through MLX, which targets CPU and GPU only; reaching the
    # ANE needs a CoreML path this build does not have, so the candidate set
    # is empty. See server/ane.py -- every check is probed at call time, so
    # this answer corrects itself if that ever changes.

    @router.get("/api/bench/ane-tune/capability")
    async def api_ane_capability() -> JSONResponse:
        return JSONResponse(ane_probe())

    @router.post("/api/bench/ane-tune/start")
    async def api_ane_start() -> JSONResponse:
        report = ane_probe()
        if report["available"]:
            # Reachable only once a CoreML backend exists; refusing loudly
            # beats silently pretending a tuning run started.
            return JSONResponse(
                {"detail": "ANE tuning is not implemented yet", **report},
                status_code=501,
            )
        return JSONResponse({"detail": report["reason"], **report}, status_code=501)

    @router.get("/api/bench/ane-tune/{tuning_id}/results")
    async def api_ane_results(tuning_id: str) -> JSONResponse:
        """The capability report, so the screen can explain itself.

        Answering 200 with `status: unavailable` rather than 404: there is
        nothing wrong with the request, and a 404 would read as "that run
        expired" instead of "this cannot run here".
        """
        report = ane_probe()
        return JSONResponse({
            "tuning_id": tuning_id, "status": "unavailable",
            "candidates": [], "recommendation": None, **report,
        })

    @router.post("/api/bench/ane-tune/{tuning_id}/cancel")
    async def api_ane_cancel(tuning_id: str) -> JSONResponse:
        return JSONResponse({"status": "not_running", "tuning_id": tuning_id})

    # -- ModelScope -------------------------------------------------------
    #
    # Same task machinery as /hf, a different fetcher (server/downloads.py).
    # ModelScope has no trending ranking, so /ms/recommended returns popular
    # only rather than relabelling one list as the other.

    @router.get("/api/ms/status")
    async def api_ms_status() -> JSONResponse:
        return JSONResponse({
            "available": ms_downloads is not None,
            "endpoint": getattr(ms_index, "base", None),
        })

    @router.get("/api/ms/tasks")
    async def api_ms_tasks() -> JSONResponse:
        if ms_downloads is None:
            return JSONResponse({"tasks": []})
        return JSONResponse({"tasks": ms_downloads.list()})

    @router.post("/api/ms/download")
    async def api_ms_download(request: Request) -> JSONResponse:
        body = await _json_body(request)
        if ms_downloads is None:
            return JSONResponse(
                {"success": False, "detail": "no model directory to download into"},
                status_code=400,
            )
        try:
            task = ms_downloads.start(
                body.get("repo_id") or body.get("repoId") or "",
                token=body.get("ms_token") or body.get("msToken") or None,
            )
        except DownloadError as exc:
            return JSONResponse({"success": False, "detail": str(exc)}, status_code=400)
        return JSONResponse({"success": True, "task": task})

    @router.post("/api/ms/cancel/{task_id}")
    async def api_ms_cancel(task_id: str) -> JSONResponse:
        ok = ms_downloads.cancel(task_id) if ms_downloads is not None else False
        return JSONResponse({"status": "cancelled" if ok else "not_running"})

    @router.post("/api/ms/retry/{task_id}")
    async def api_ms_retry(task_id: str) -> JSONResponse:
        if ms_downloads is None:
            return JSONResponse({"success": False}, status_code=404)
        try:
            return JSONResponse({"success": True, "task": ms_downloads.retry(task_id)})
        except DownloadError as exc:
            return JSONResponse({"success": False, "detail": str(exc)}, status_code=400)

    @router.get("/api/ms/task/{task_id}")
    async def api_ms_task(task_id: str) -> JSONResponse:
        task = ms_downloads.get(task_id) if ms_downloads is not None else None
        if task is None:
            return JSONResponse({"detail": "unknown task"}, status_code=404)
        return JSONResponse({"task": task})

    @router.delete("/api/ms/task/{task_id}")
    async def api_ms_forget(task_id: str) -> JSONResponse:
        if ms_downloads is None:
            return JSONResponse({"deleted": False}, status_code=404)
        try:
            return JSONResponse({"deleted": ms_downloads.forget(task_id)})
        except DownloadError as exc:
            return JSONResponse({"deleted": False, "detail": str(exc)}, status_code=400)

    @router.get("/api/ms/search")
    async def api_ms_search(q: str = "", limit: int = 30) -> JSONResponse:
        if ms_index is None:
            return JSONResponse({"models": [], "total": 0})
        models = ms_index.search(q, limit=limit)
        return JSONResponse({"models": models, "total": len(models)})

    @router.get("/api/ms/recommended")
    async def api_ms_recommended(limit: int = 20) -> JSONResponse:
        if ms_index is None:
            return JSONResponse({"trending": [], "popular": []})
        return JSONResponse(ms_index.recommended(limit=limit))

    @router.get("/api/ms/model-info")
    async def api_ms_model_info(repo_id: str = "") -> JSONResponse:
        info = ms_index.model_info(repo_id) if (ms_index and repo_id) else None
        if info is None:
            return JSONResponse({"detail": "unknown repo"}, status_code=404)
        return JSONResponse(info)

    # -- profiles and templates -------------------------------------------
    #
    # Two collections: global templates (some built in and read-only) and
    # per-model profiles. See server/profiles.py for why the sampling/engine
    # split is enforced there rather than in the UI.

    def _need_profiles() -> Any:
        if profiles is None:
            raise _Unavailable("this server was started without a model directory")
        return profiles

    @router.get("/api/profile-templates")
    async def api_templates() -> JSONResponse:
        if profiles is None:
            return JSONResponse({"templates": []})
        return JSONResponse(
            {"templates": [t.to_dict() for t in profiles.list_templates()]}
        )

    @router.post("/api/profile-templates")
    async def api_create_template(request: Request) -> JSONResponse:
        body = await _json_body(request)
        try:
            t = _need_profiles().create_template(
                body.get("name", ""),
                display_name=body.get("display_name") or body.get("displayName") or "",
                description=body.get("description"),
                settings=body.get("settings") or {},
            )
        except (ProfileError, _Unavailable) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"template": t.to_dict()})

    @router.put("/api/profile-templates/{name}")
    async def api_update_template(name: str, request: Request) -> JSONResponse:
        body = await _json_body(request)
        try:
            t = _need_profiles().update_template(name, **body)
        except (ProfileError, _Unavailable) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"template": t.to_dict()})

    @router.delete("/api/profile-templates/{name}")
    async def api_delete_template(name: str) -> JSONResponse:
        try:
            deleted = _need_profiles().delete_template(name)
        except (ProfileError, _Unavailable) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"deleted": deleted, "name": name})

    @router.get("/api/models/{model_id:path}/profiles")
    async def api_profiles(model_id: str) -> JSONResponse:
        if profiles is None:
            return JSONResponse({"profiles": []})
        return JSONResponse(
            {"profiles": [p.to_dict(base_model=model_id)
                          for p in profiles.list_profiles(model_id)]}
        )

    @router.post("/api/models/{model_id:path}/profiles")
    async def api_create_profile(model_id: str, request: Request) -> JSONResponse:
        body = await _json_body(request)
        try:
            p = _need_profiles().create_profile(
                model_id,
                body.get("name", ""),
                display_name=body.get("display_name") or body.get("displayName") or "",
                description=body.get("description"),
                settings=body.get("settings") or {},
                source_template=body.get("source_template"),
                also_save_as_template=bool(body.get("also_save_as_template")),
                expose_as_model=bool(body.get("expose_as_model")),
            )
        except (ProfileError, _Unavailable) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"profile": p.to_dict(base_model=model_id)})

    @router.put("/api/models/{model_id:path}/profiles/{name}")
    async def api_update_profile(model_id: str, name: str, request: Request) -> JSONResponse:
        body = await _json_body(request)
        try:
            p = _need_profiles().update_profile(model_id, name, **body)
        except (ProfileError, _Unavailable) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"profile": p.to_dict(base_model=model_id)})

    @router.delete("/api/models/{model_id:path}/profiles/{name}")
    async def api_delete_profile(model_id: str, name: str) -> JSONResponse:
        try:
            deleted = _need_profiles().delete_profile(model_id, name)
        except (ProfileError, _Unavailable) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        return JSONResponse({"deleted": deleted, "name": name})

    @router.post("/api/models/{model_id:path}/profiles/{name}/apply")
    async def api_apply_profile(model_id: str, name: str) -> JSONResponse:
        """Report the settings this profile would put into effect.

        bwr's engine settings are fixed at load, so "apply" cannot mutate a
        resident engine. It answers what the profile holds and which half of
        it needs a reload, which is exactly what the caller has to know --
        rather than reporting a success that changed nothing.
        """
        if profiles is None:
            return JSONResponse({"detail": "no profile store"}, status_code=400)
        p = profiles.get_profile(model_id, name)
        if p is None:
            return JSONResponse(
                {"detail": f"unknown profile {name!r} for {model_id}"}, status_code=404
            )
        return JSONResponse({
            "model_id": model_id,
            "settings": dict(p.settings),
            "applied_now": p.sampling_settings,
            "requires_reload": p.has_engine_fields,
        })

    @router.get("/api/usage")
    async def api_usage(range: str = "today", model: str = "") -> JSONResponse:
        """Token/request totals, optionally for one model.

        `range` is accepted and ignored: the counters are not a time series
        (see server/stats.py), so every range is the same answer and a 400
        would only break a screen that has a range picker.
        """
        if stats is None:
            return JSONResponse({
                "enabled": False, "available": False, "dropped_requests": 0,
                "totals": {
                    "model_id": None, "requests": 0, "total_tokens": 0,
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "cached_tokens": 0, "generation_tps": None,
                    "cache_efficiency": 0.0,
                },
                "models": [], "heatmap": [],
            })
        body = stats.usage()
        if model:
            body["models"] = [m for m in body["models"] if m["model_id"] == model]
        return JSONResponse(body)

    @router.post("/api/reload")
    async def api_reload() -> JSONResponse:
        """Pick up models added to the model directory since startup.

        Real work, not a stub: ModelPool.rescan() re-runs discovery and keeps
        residency, so a model downloaded while the server was up becomes
        servable without a restart.
        """
        if pool is None:
            return JSONResponse(
                {"detail": "single-model server: nothing to rescan"}, status_code=409
            )
        before = {m["id"] for m in pool.list()}
        pool.rescan()
        after = [m["id"] for m in pool.list()]
        return JSONResponse(
            {
                "status": "ok",
                "models": after,
                "added": sorted(set(after) - before),
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
                # oMLX-shaped capability flags. All false: bwr supports none
                # of these. `mtp_compatibility_reason` must be a STRING, not
                # absent -- the template calls .includes() on it whenever
                # mtp_compatible is falsy, and undefined.includes throws.
                "mtp_compatible": False,
                "mtp_compatibility_reason": "not supported by the bwr backend",
                "mtp_enabled": False,
                "vlm_mtp_enabled": False,
                "dflash_enabled": False,
                "moe_expert_offload_enabled": False,
                "is_paroquant": False,
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

    # Families found by walking every fetch() in the bundled JS against a
    # live server -- the first pass missed seven of them, which 404'd.
    # "stats" is safe here even though /api/stats is real: exact routes are
    # registered first and win, so only unmatched subpaths (stats/clear)
    # reach this fallback.
    # Fallback for everything with no exact route. Heads that DO have a real
    # handler stay listed on purpose: routes are matched in registration
    # order and the catch-all is registered last, so `GET /api/stats` reaches
    # the real handler while `GET /api/stats/clear` -- which the dashboard
    # polls, and which is POST-only here -- still degrades to an empty body
    # instead of a 404 the page throws on. Removing a head because it "has a
    # handler now" silently breaks the method and sub-path variants it does
    # not have. See test_stats_family_does_not_shadow_the_real_stats_endpoint.
    _UNSUPPORTED = (
        "bench", "hf", "ms", "ane-tune", "profiles", "profile-fields",
        "profile-templates", "grammar", "hot-cache", "logs", "presets",
        "sub-keys", "oq", "cluster",
        "logout", "reload", "server", "ssd-cache", "stats", "upload",
        "web-search", "usage",
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


def mount(
    app: Any, engine: Any, model_name: str, pool: Any = None, *,
    stats: Any = None, profiles: Any = None, downloads: Any = None,
    hub: Any = None, bench: Any = None, ms_downloads: Any = None,
    ms_index: Any = None, ctxbench: Any = None,
) -> None:
    """Attach the UI at /admin plus the two /v1 helpers the page polls."""
    install_log_buffer()
    app.include_router(
        build_router(engine, model_name, pool, stats, profiles, downloads, hub,
                     bench, ms_downloads, ms_index, ctxbench),
        prefix="/admin"
    )
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

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        """Liveness + live activity for the macOS menubar poller.

        Root-level, not under /admin: MenubarStatsPoller polls `/api/status`
        every few seconds and decodes the same shape as /admin/api/stats.
        Without it the menubar logged a 404 on every tick and showed no
        activity. Its Stats fields are all optional, so the counters bwr does
        not keep are simply absent rather than faked.
        """
        body: dict[str, Any] = {"status": "ok"}
        if pool is None:
            entry = {
                "id": model_name,
                "loaded": True,
                "in_flight": engine.n_in_flight,
                "decode_calls": engine.ctx.decode_calls,
            }
            body["active_models"] = {"models": [entry]}
            return JSONResponse(body)
        active = []
        for m in pool.list():
            if not m["loaded"]:
                continue
            raw = pool.engine_for(m["id"])
            e = {"id": m["id"], "loaded": True}
            if raw is not None:
                e["in_flight"] = raw.n_in_flight
                e["decode_calls"] = raw.ctx.decode_calls
            active.append(e)
        body["active_models"] = {"models": active}
        body["loaded"] = pool.loaded_ids
        return JSONResponse(body)

    @app.get("/v1/mcp/tools")
    async def mcp_tools() -> JSONResponse:
        return JSONResponse({"tools": []})

    @app.post("/v1/audio/transcriptions")
    async def transcriptions() -> JSONResponse:
        """The chat page offers mic input; bwr serves text models only."""
        return JSONResponse(
            {"detail": "audio transcription is not provided by the bwr backend",
             "supported": False},
            status_code=501,
        )
