"""Hugging Face model downloads, as background tasks the UI can watch.

Downloads land DIRECTLY in the serving model directory
(`<model_dir>/<repo-name>`), not in the Hub cache. That is the whole point:
`ModelPool.discover()` walks that directory, so a finished download plus a
`POST /admin/api/reload` makes the model servable without a restart. Pulling
into the shared cache would leave the pool unable to see it.

Progress and cancellation
-------------------------
`snapshot_download` is a blocking call with no cancel handle, so both ride on
the one hook it does expose: `tqdm_class`. Each file gets a progress bar; the
subclass below folds every byte delta into the task and, on each update,
checks whether the task was cancelled and raises if so. That unwinds the
download from the inside, which is the only way to stop it mid-file.

The total is taken from the repo's file metadata up front rather than
accumulated from the bars, because bars appear one file at a time and a
percentage that rebases every file is worse than none.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Weights only. A repo often carries ONNX/GGUF/TF copies of the same model,
# and pulling all of them can triple a download the user did not ask for.
DEFAULT_IGNORE = (
    "*.onnx", "*.onnx_data", "*.gguf", "*.h5", "*.msgpack", "*.tflite",
    "*.pth", "*.bin.index.fp32.json", "original/*", "onnx/*", "coreml/*",
)


class DownloadError(RuntimeError):
    pass


class _Cancelled(RuntimeError):
    """Raised inside the progress hook to unwind a running download."""


@dataclass
class Task:
    task_id: str
    repo_id: str
    status: str = "pending"        # pending|downloading|completed|failed|cancelled
    total_size: int = 0
    downloaded_size: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0
    local_dir: str = ""
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def progress(self) -> float:
        if self.status == "completed":
            return 1.0
        if self.total_size <= 0:
            return 0.0
        return min(1.0, self.downloaded_size / self.total_size)

    def to_dict(self) -> dict[str, Any]:
        # Field names and types match HFTaskDTO exactly; every one of them is
        # non-optional there, so none may be omitted.
        return {
            "task_id": self.task_id,
            "repo_id": self.repo_id,
            "status": self.status,
            "progress": self.progress,
            "total_size": self.total_size,
            "downloaded_size": self.downloaded_size,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "local_dir": self.local_dir,
        }


def _safe_dirname(repo_id: str) -> str:
    """`org/Model-Name` -> `Model-Name`, with anything path-ish removed.

    The repo id comes from the network and becomes a directory under the
    user's model root, so it is not joined verbatim: a crafted id must not
    be able to write outside the model directory.
    """
    name = repo_id.rstrip("/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name).lstrip(".-")
    if not name:
        raise DownloadError(f"cannot derive a directory name from {repo_id!r}")
    return name


class DownloadManager:
    """Runs and tracks Hub downloads into the serving model directory."""

    #: Concurrent downloads. Two saturate a home connection; more mostly
    #: means every one of them finishes later.
    MAX_WORKERS = 2

    def __init__(
        self,
        model_dir: str | Path | None,
        *,
        endpoint: str | None = None,
        pool: Any = None,
        fetcher: Any = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.endpoint = endpoint or None
        self._pool = pool
        # The fetcher is what makes this class source-agnostic: Hugging Face
        # and ModelScope differ only in how bytes arrive, not in how a task
        # is tracked, cancelled, retried or guarded against path traversal.
        self._fetcher = fetcher or HFFetcher(endpoint=self.endpoint)
        self._tasks: dict[str, Task] = {}
        self._lock = threading.RLock()
        self._pool_exec = ThreadPoolExecutor(
            max_workers=self.MAX_WORKERS, thread_name_prefix="bwr-download"
        )

    # -- inspection --------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
            return [t.to_dict() for t in tasks]

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            t = self._tasks.get(task_id)
            return t.to_dict() if t else None

    # -- lifecycle ---------------------------------------------------------

    def start(self, repo_id: str, *, token: str | None = None) -> dict[str, Any]:
        if self.model_dir is None:
            raise DownloadError(
                "this server was started without a model directory, so there "
                "is nowhere to download to"
            )
        repo_id = (repo_id or "").strip()
        if not repo_id:
            raise DownloadError("repo_id is required")
        with self._lock:
            for t in self._tasks.values():
                if t.repo_id == repo_id and t.status in ("pending", "downloading"):
                    raise DownloadError(f"{repo_id} is already downloading")
            task = Task(
                task_id=uuid.uuid4().hex[:12],
                repo_id=repo_id,
                local_dir=str(self.model_dir / _safe_dirname(repo_id)),
            )
            self._tasks[task.task_id] = task
            # Snapshot INSIDE the lock and before submitting: the worker can
            # finish before this method returns, and a start() that reported
            # "completed" would make the caller skip the progress row it is
            # supposed to render.
            queued = task.to_dict()
        self._pool_exec.submit(self._run, task.task_id, token)
        return queued

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None or t.status not in ("pending", "downloading"):
                return False
            t._cancel.set()
            # Marked here rather than in the worker so a cancel of a task
            # still queued (never started) takes effect at all.
            t.status = "cancelled"
            t.completed_at = time.time()
            return True

    def retry(self, task_id: str, *, token: str | None = None) -> dict[str, Any]:
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                raise DownloadError(f"unknown task {task_id!r}")
            if t.status in ("pending", "downloading"):
                raise DownloadError("task is still running")
            t.status = "pending"
            t.error = ""
            t.downloaded_size = 0
            t.completed_at = 0.0
            t.retry_count += 1
            t._cancel = threading.Event()
        self._pool_exec.submit(self._run, task_id, token)
        return self.get(task_id) or {}

    def forget(self, task_id: str) -> bool:
        """Drop a finished task from the list. Never touches downloaded files."""
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return False
            if t.status in ("pending", "downloading"):
                raise DownloadError("cancel the task before removing it")
            del self._tasks[task_id]
            return True

    def delete_model(self, name: str) -> bool:
        """Remove a downloaded model directory from the model root."""
        import shutil

        if self.model_dir is None:
            raise DownloadError("no model directory")
        root = self.model_dir.resolve()
        target = (root / name).resolve()
        # `name` arrives from a URL path. Refuse anything that escapes the
        # model root, and refuse the root itself.
        if target == root or root not in target.parents:
            raise DownloadError(f"{name!r} is not inside the model directory")
        if not target.is_dir():
            return False
        shutil.rmtree(target)
        return True

    # -- worker ------------------------------------------------------------

    def _run(self, task_id: str, token: str | None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task._cancel.is_set():
                return
            task.status = "downloading"
            task.started_at = time.time()

        try:
            self._fetcher.fetch(task, token)
        except _Cancelled:
            with self._lock:
                task.status = "cancelled"
                task.completed_at = time.time()
            return
        except Exception as exc:  # noqa: BLE001 - any Hub or disk failure is
            # the user's to see verbatim: auth, quota, network, bad repo id.
            # A download thread must never take the server down with it.
            with self._lock:
                task.status = "failed"
                task.error = f"{type(exc).__name__}: {exc}"
                task.completed_at = time.time()
            return

        with self._lock:
            task.status = "completed"
            task.downloaded_size = task.total_size or task.downloaded_size
            task.completed_at = time.time()
        # Make it servable without a restart. Best-effort: a rescan failure
        # must not turn a finished download into a failed one.
        try:
            if self._pool is not None:
                self._pool.rescan()
        except Exception:  # noqa: BLE001 - reported by /admin/api/models being stale
            pass

class HFFetcher:
    """Hugging Face, via huggingface_hub.

    Progress and cancellation ride `tqdm_class` because `snapshot_download`
    exposes no other hook -- see the module docstring.
    """

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint or None

    def fetch(self, task: Task, token: str | None) -> None:
        from huggingface_hub import snapshot_download

        task.total_size = self._repo_size(task.repo_id, token)
        snapshot_download(
            repo_id=task.repo_id,
            local_dir=task.local_dir,
            token=token or None,
            endpoint=self.endpoint,
            ignore_patterns=list(DEFAULT_IGNORE),
            tqdm_class=_make_tqdm(task),
        )

    def _repo_size(self, repo_id: str, token: str | None) -> int:
        """Total bytes of the files this download will actually fetch."""
        try:
            from huggingface_hub import HfApi

            api = HfApi(endpoint=self.endpoint, token=token or None)
            info = api.model_info(repo_id, files_metadata=True)
            return sum(
                int(getattr(f, "size", 0) or 0)
                for f in (getattr(info, "siblings", None) or [])
                if not _ignored(getattr(f, "rfilename", ""))
            )
        except Exception as exc:  # noqa: BLE001 - the size is a nicety; a
            # download with an unknown total still runs, it just reports 0%
            # until it finishes rather than failing before it starts.
            logger.info("hub size probe for %s failed: %s", repo_id, exc)
            return 0


# -- ModelScope --------------------------------------------------------------
#
# Implemented against the public REST API with urllib rather than the
# `modelscope` SDK: the SDK is a large dependency (it pulls its own
# datasets/训练 stack) for three GET requests, and adding it would bloat the
# macOS bundle for a feature most users never touch.

MS_BASE = "https://modelscope.cn/api/v1"


def _ms_json(url: str, token: str | None = None, *, method: str = "GET",
             body: bytes | None = None, timeout: int = 30) -> Any:
    import urllib.request

    headers = {"User-Agent": "bwr", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _ms_safe_relpath(path: str) -> str:
    """A repo-relative file path that cannot escape the download directory.

    The path comes from a remote index, so it is validated rather than
    trusted: absolute paths and any `..` segment are refused outright rather
    than normalised, because a normalised traversal is still a file the
    caller did not ask for.
    """
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or path.startswith("/") or any(p == ".." for p in parts):
        raise DownloadError(f"unsafe path in repo listing: {path!r}")
    return "/".join(parts)


class MSFetcher:
    """ModelScope, over the REST API.

    Unlike the Hub path this streams each file itself, which means progress
    is exact and a cancel takes effect within one 1 MiB chunk instead of one
    file.
    """

    CHUNK = 1 << 20

    def __init__(self, base: str | None = None) -> None:
        self.base = (base or MS_BASE).rstrip("/")

    def files(self, repo_id: str, token: str | None = None) -> list[dict[str, Any]]:
        url = f"{self.base}/models/{repo_id}/repo/files?Revision=master"
        data = _ms_json(url, token) or {}
        out = []
        for f in ((data.get("Data") or {}).get("Files") or []):
            if f.get("Type") != "blob":
                continue
            path = f.get("Path") or f.get("Name") or ""
            if not path or _ignored(path):
                continue
            out.append({"path": path, "size": int(f.get("Size") or 0)})
        return out

    def fetch(self, task: Task, token: str | None) -> None:
        import urllib.request

        files = self.files(task.repo_id, token)
        if not files:
            raise DownloadError(f"{task.repo_id} lists no downloadable files")
        task.total_size = sum(f["size"] for f in files)
        root = Path(task.local_dir)
        headers = {"User-Agent": "bwr"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        for entry in files:
            if task._cancel.is_set():
                raise _Cancelled(task.task_id)
            rel = _ms_safe_relpath(entry["path"])
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = (
                f"{self.base}/models/{task.repo_id}/repo"
                f"?Revision=master&FilePath={urllib.parse.quote(rel)}"
            )
            req = urllib.request.Request(url, headers=headers)
            # Written to a temp name and renamed, so an interrupted file is
            # never left looking complete to the model loader.
            part = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as fh:
                while True:
                    if task._cancel.is_set():
                        part.unlink(missing_ok=True)
                        raise _Cancelled(task.task_id)
                    chunk = resp.read(self.CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    task.downloaded_size += len(chunk)
            part.replace(dest)


class ModelScopeIndex:
    """Browse surfaces for ModelScope. Degrades to empty, like HubIndex."""

    def __init__(self, base: str | None = None, token: str | None = None) -> None:
        self.base = (base or MS_BASE).rstrip("/")
        self.token = token or None

    def _row(self, m: dict[str, Any]) -> dict[str, Any]:
        repo = f"{m.get('Path', '')}/{m.get('Name', '')}".strip("/")
        return {
            "repo_id": repo,
            "name": m.get("Name"),
            "downloads": m.get("Downloads"),
            "likes": m.get("Stars"),
            "trending_score": None,
            "size": None,
            "size_formatted": None,
            "params": None,
            "params_formatted": None,
        }

    def _query(self, name: str, limit: int) -> list[dict[str, Any]]:
        body = json.dumps({
            "PageSize": limit, "PageNumber": 1, "SortBy": "Default",
            "Target": "", "SingleCriterion": [], "Name": name,
        }).encode()
        data = _ms_json(f"{self.base}/dolphin/models", self.token,
                        method="PUT", body=body) or {}
        models = ((data.get("Data") or {}).get("Model") or {}).get("Models") or []
        return [self._row(m) for m in models]

    def search(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        try:
            # MLX is appended because bwr cannot serve the rest, and
            # ModelScope has no library filter to say so structurally.
            return self._query(f"{query} mlx".strip(), limit)
        except Exception as exc:  # noqa: BLE001 - offline or blocked: empty,
            # but logged, so a wrong request shape is visible rather than
            # reading as "ModelScope returned nothing".
            logger.warning("modelscope search failed: %s: %s", type(exc).__name__, exc)
            return []

    def recommended(self, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
        try:
            popular = self._query("mlx", limit)
        except Exception as exc:  # noqa: BLE001 - see search()
            logger.warning("modelscope recommended failed: %s", exc)
            popular = []
        # The API exposes no trending ranking, so that list stays empty
        # rather than being filled with the popular one relabelled.
        return {"trending": [], "popular": popular}

    def model_info(self, repo_id: str) -> dict[str, Any] | None:
        try:
            data = _ms_json(f"{self.base}/models/{repo_id}", self.token) or {}
            info = data.get("Data") or {}
            files = MSFetcher(self.base).files(repo_id, self.token)
        except Exception as exc:  # noqa: BLE001 - unknown repo reads as absent
            logger.info("modelscope model_info(%s) failed: %s", repo_id, exc)
            return None
        total = sum(f["size"] for f in files)
        return {
            "repo_id": repo_id,
            "name": info.get("Name") or repo_id.split("/")[-1],
            "downloads": info.get("Downloads"),
            "likes": info.get("Stars"),
            "size": total,
            "size_formatted": _human(total),
            "files": [{"name": f["path"], "size": f["size"]} for f in files],
            "gated": False,
            "tags": list(info.get("Tags") or []),
        }


def _ignored(filename: str) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(filename, pat) for pat in DEFAULT_IGNORE)


def _make_tqdm(task: Task) -> type:
    """A tqdm subclass that reports into `task` and honours its cancel flag."""
    try:
        from tqdm.auto import tqdm as _base
    except ImportError:  # pragma: no cover - hub always brings tqdm
        from tqdm import tqdm as _base  # type: ignore[no-redef]

    class _TaskTqdm(_base):  # type: ignore[misc, valid-type]
        def update(self, n: int | None = 1) -> Any:
            if task._cancel.is_set():
                raise _Cancelled(task.task_id)
            if n:
                task.downloaded_size += int(n)
            return super().update(n)

    return _TaskTqdm


# -- Hub discovery -----------------------------------------------------------


def _model_row(m: Any) -> dict[str, Any]:
    size = getattr(m, "usedStorage", None) or getattr(m, "used_storage", None)
    params = None
    safetensors = getattr(m, "safetensors", None)
    if safetensors is not None:
        params = getattr(safetensors, "total", None)
    return {
        "repo_id": m.id,
        "name": m.id.split("/")[-1],
        "downloads": getattr(m, "downloads", None),
        "likes": getattr(m, "likes", None),
        "trending_score": getattr(m, "trending_score", None),
        "size": int(size) if size else None,
        "size_formatted": _human(size) if size else None,
        "params": int(params) if params else None,
        "params_formatted": _human_count(params) if params else None,
    }


def _human(n: int | None) -> str:
    value = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _human_count(n: int | None) -> str:
    value = float(n or 0)
    for unit in ("", "K", "M", "B", "T"):
        if value < 1000 or unit == "T":
            return f"{value:.0f}{unit}" if not unit else f"{value:.1f}{unit}"
        value /= 1000
    return f"{value:.1f}T"


class HubIndex:
    """Read-only Hub queries behind /hf/search, /recommended and /model-info.

    Every method degrades to an empty result rather than raising: these back
    a browse screen, and an offline machine should show an empty list, not an
    error dialog over a feature the user did not invoke.
    """

    #: Only MLX-format repos are worth listing -- bwr cannot serve the rest.
    #: Passed as `filter=`, not `library=`: huggingface_hub 1.x has no
    #: `library` parameter, and the TypeError it raised was swallowed by the
    #: degradation below, so every browse screen was silently empty.
    LIBRARY_FILTER = "mlx"

    def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
        self.endpoint = endpoint or None
        self.token = token or None

    def _api(self) -> Any:
        from huggingface_hub import HfApi

        return HfApi(endpoint=self.endpoint, token=self.token)

    def search(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        try:
            models = self._api().list_models(
                search=query or None, filter=self.LIBRARY_FILTER,
                sort="downloads", limit=limit,
            )
            return [_model_row(m) for m in models]
        except Exception as exc:  # noqa: BLE001 - offline or rate-limited:
            # an empty list, but LOGGED. Silently swallowing this is how a
            # wrong argument name read as "the Hub returned nothing".
            logger.warning("hub search failed: %s: %s", type(exc).__name__, exc)
            return []

    def recommended(self, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
        def _listed(sort: str) -> list[dict[str, Any]]:
            try:
                return [
                    _model_row(m) for m in self._api().list_models(
                        filter=self.LIBRARY_FILTER, sort=sort, limit=limit
                    )
                ]
            except Exception as exc:  # noqa: BLE001 - see search()
                logger.warning(
                    "hub recommended(%s) failed: %s: %s", sort, type(exc).__name__, exc
                )
                return []

        return {"trending": _listed("trendingScore"), "popular": _listed("downloads")}

    def model_info(self, repo_id: str) -> dict[str, Any] | None:
        try:
            info = self._api().model_info(repo_id, files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - unknown/private repo reads
            # as absent, but a transport failure is worth a line in the log.
            logger.info("hub model_info(%s) failed: %s", repo_id, exc)
            return None
        files = [
            {"name": f.rfilename, "size": int(getattr(f, "size", 0) or 0)}
            for f in (getattr(info, "siblings", None) or [])
        ]
        total = sum(f["size"] for f in files if not _ignored(f["name"]))
        return {
            "repo_id": info.id,
            "name": info.id.split("/")[-1],
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "size": total,
            "size_formatted": _human(total),
            "files": files,
            "gated": bool(getattr(info, "gated", False)),
            "tags": list(getattr(info, "tags", None) or []),
        }
