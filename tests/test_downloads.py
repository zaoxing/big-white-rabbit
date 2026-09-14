"""Hugging Face download tasks: lifecycle, progress, and path safety.

No network: `snapshot_download` and the Hub API are stubbed. What is under
test is the task bookkeeping the UI polls and the guards around writing into
the user's model directory -- not huggingface_hub itself.
"""

from __future__ import annotations

import threading
import time

import pytest

from bwr.server import downloads as dl
from bwr.server.downloads import DownloadError, DownloadManager, Task, _safe_dirname


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def manager(tmp_path):
    return DownloadManager(tmp_path)


# -- directory naming and path safety ----------------------------------------


@pytest.mark.parametrize("repo,expected", [
    ("org/Model-Name", "Model-Name"),
    ("Model", "Model"),
    ("org/sub/Deep", "Deep"),
    ("org/../../etc", "etc"),
    ("org/a b:c", "a-b-c"),
])
def test_repo_ids_become_safe_directory_names(repo, expected):
    """The repo id comes off the network and becomes a directory under the
    user's model root, so it is never joined verbatim."""
    assert _safe_dirname(repo) == expected


@pytest.mark.parametrize("repo", ["", "/", "..", "///"])
def test_undeivable_names_are_refused(repo):
    with pytest.raises(DownloadError):
        _safe_dirname(repo)


def test_delete_model_refuses_to_escape_the_model_directory(manager, tmp_path):
    outside = tmp_path.parent / "not-models"
    outside.mkdir(exist_ok=True)
    with pytest.raises(DownloadError):
        manager.delete_model("../not-models")
    with pytest.raises(DownloadError):
        manager.delete_model(".")
    assert outside.is_dir(), "the guard must not have deleted anything"


def test_delete_model_removes_a_real_directory(manager, tmp_path):
    (tmp_path / "victim").mkdir()
    assert manager.delete_model("victim") is True
    assert not (tmp_path / "victim").exists()
    assert manager.delete_model("victim") is False


# -- task lifecycle ----------------------------------------------------------


def test_start_requires_a_model_directory():
    with pytest.raises(DownloadError):
        DownloadManager(None).start("org/model")


def test_start_requires_a_repo_id(manager):
    with pytest.raises(DownloadError):
        manager.start("  ")


def test_a_download_runs_and_completes(manager, monkeypatch, tmp_path):
    started = threading.Event()

    def fake_snapshot(**kwargs):
        started.set()
        return kwargs["local_dir"]

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 1000)

    task = manager.start("org/model")
    assert task["status"] in ("pending", "downloading")
    assert _wait(lambda: manager.get(task["task_id"])["status"] == "completed")
    done = manager.get(task["task_id"])
    assert done["progress"] == 1.0
    assert done["total_size"] == 1000
    assert done["local_dir"] == str(tmp_path / "model")


def test_a_failed_download_reports_the_error_verbatim(manager, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)

    task = manager.start("org/model")
    assert _wait(lambda: manager.get(task["task_id"])["status"] == "failed")
    assert "401 unauthorized" in manager.get(task["task_id"])["error"]


def test_a_failure_does_not_take_down_the_manager(manager, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    t1 = manager.start("org/a")
    assert _wait(lambda: manager.get(t1["task_id"])["status"] == "failed")
    # Still usable afterwards.
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: k["local_dir"])
    t2 = manager.start("org/b")
    assert _wait(lambda: manager.get(t2["task_id"])["status"] == "completed")


def test_the_same_repo_cannot_be_queued_twice(manager, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **k: gate.wait(5) or k["local_dir"])
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    manager.start("org/model")
    with pytest.raises(DownloadError):
        manager.start("org/model")
    gate.set()


def test_cancelling_a_queued_task_takes_effect(manager, monkeypatch):
    """A task cancelled before its worker starts must still end cancelled --
    the flag alone would be checked too late."""
    gate = threading.Event()
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **k: gate.wait(5) or k["local_dir"])
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    task = manager.start("org/model")
    assert manager.cancel(task["task_id"]) is True
    assert manager.get(task["task_id"])["status"] == "cancelled"
    gate.set()


def test_cancelling_a_finished_task_is_a_no_op(manager, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: k["local_dir"])
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    task = manager.start("org/model")
    assert _wait(lambda: manager.get(task["task_id"])["status"] == "completed")
    assert manager.cancel(task["task_id"]) is False


def test_retry_resets_progress_and_counts(manager, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("nope")))
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    task = manager.start("org/model")
    assert _wait(lambda: manager.get(task["task_id"])["status"] == "failed")

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: k["local_dir"])
    manager.retry(task["task_id"])
    assert _wait(lambda: manager.get(task["task_id"])["status"] == "completed")
    after = manager.get(task["task_id"])
    assert after["retry_count"] == 1
    assert after["error"] == ""


def test_retrying_a_running_task_is_refused(manager, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **k: gate.wait(5) or k["local_dir"])
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    task = manager.start("org/model")
    assert _wait(lambda: manager.get(task["task_id"])["status"] == "downloading")
    with pytest.raises(DownloadError):
        manager.retry(task["task_id"])
    gate.set()


def test_forget_drops_the_row_but_never_the_files(manager, monkeypatch, tmp_path):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: k["local_dir"])
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    task = manager.start("org/model")
    assert _wait(lambda: manager.get(task["task_id"])["status"] == "completed")
    (tmp_path / "model").mkdir(exist_ok=True)

    assert manager.forget(task["task_id"]) is True
    assert manager.get(task["task_id"]) is None
    assert (tmp_path / "model").is_dir(), "dismissing a row must not delete weights"


def test_forgetting_a_running_task_is_refused(manager, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **k: gate.wait(5) or k["local_dir"])
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    task = manager.start("org/model")
    with pytest.raises(DownloadError):
        manager.forget(task["task_id"])
    gate.set()


def test_a_finished_download_rescans_the_pool(tmp_path, monkeypatch):
    """This is the whole point of downloading into the model directory: the
    model becomes servable without restarting the server."""
    class _Pool:
        rescans = 0

        def rescan(self):
            type(self).rescans += 1

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: k["local_dir"])
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    m = DownloadManager(tmp_path, pool=_Pool())
    task = m.start("org/model")
    assert _wait(lambda: m.get(task["task_id"])["status"] == "completed")
    assert _wait(lambda: _Pool.rescans == 1)


def test_a_rescan_failure_does_not_fail_the_download(tmp_path, monkeypatch):
    class _Pool:
        def rescan(self):
            raise OSError("model dir vanished")

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: k["local_dir"])
    monkeypatch.setattr(DownloadManager, "_repo_size", lambda self, r, t: 0)
    m = DownloadManager(tmp_path, pool=_Pool())
    task = m.start("org/model")
    assert _wait(lambda: m.get(task["task_id"])["status"] == "completed")


# -- progress ----------------------------------------------------------------


def test_progress_is_reported_from_the_tqdm_hook():
    task = Task(task_id="t", repo_id="org/model", total_size=100)
    bar = dl._make_tqdm(task)(total=100, disable=True)
    bar.update(25)
    assert task.downloaded_size == 25
    assert task.progress == 0.25


def test_the_tqdm_hook_aborts_a_cancelled_download():
    """snapshot_download has no cancel handle; raising from the progress hook
    is the only way to stop it mid-file."""
    task = Task(task_id="t", repo_id="org/model", total_size=100)
    bar = dl._make_tqdm(task)(total=100, disable=True)
    task._cancel.set()
    with pytest.raises(dl._Cancelled):
        bar.update(1)


def test_progress_is_zero_when_the_total_is_unknown():
    """A repo whose size could not be read still downloads; it just reports
    0% until it finishes rather than dividing by zero."""
    task = Task(task_id="t", repo_id="org/model", total_size=0)
    task.downloaded_size = 500
    assert task.progress == 0.0
    task.status = "completed"
    assert task.progress == 1.0


def test_task_dict_carries_every_field_hftaskdto_requires():
    task = Task(task_id="t", repo_id="org/model")
    body = task.to_dict()
    for key in ("task_id", "repo_id", "status", "progress", "total_size",
                "downloaded_size", "error", "created_at", "started_at",
                "completed_at", "retry_count"):
        assert key in body, f"HFTaskDTO requires {key}"


# -- Hub index degrades rather than raising ----------------------------------


def test_hub_queries_return_empty_when_the_hub_is_unreachable(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.HfApi", _Boom)
    hub = dl.HubIndex()
    assert hub.search("qwen") == []
    assert hub.recommended() == {"trending": [], "popular": []}
    assert hub.model_info("org/model") is None
