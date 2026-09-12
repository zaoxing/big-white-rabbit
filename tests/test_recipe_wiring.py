"""Recipe -> serve() wiring: every recipe key must reach the engine.

Regression guard for T14, where serve() silently dropped speculative /
spec_max_drafts / n_threads / n_ubatch (recipe spec=True never took
effect), and for the review findings on f01a0c3 (flash_attn no-op,
explicit flags clobbered, draft+spec conflict).
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def rootdir(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    return REPO_ROOT


def _serve_kwargs(monkeypatch, argv):
    """Run _cmd_serve with serve() mocked; return (rc, kwargs)."""
    import bwr.server.launch as launch

    captured = {}

    def mock_serve(model_path, **kw):
        captured["model"] = model_path
        captured.update(kw)
        return 0

    monkeypatch.setattr(launch, "serve", mock_serve)
    from bwr.cli import _cmd_serve

    rc = _cmd_serve(argv)
    return rc, captured


def test_30b_recipe_forwards_tuned_knobs(monkeypatch, rootdir):
    rc, kw = _serve_kwargs(monkeypatch, ["--recipe", "30b"])
    assert rc == 0
    assert kw["engine"] == "metal"
    assert kw["n_ctx"] == 8192
    assert kw["n_batch"] == 2048
    assert kw["n_ubatch"] == 512
    assert kw["n_seq_max"] == 2
    assert kw["n_threads"] == 0
    assert kw["n_threads_batch"] == 0
    assert kw["kv_unified"] is True
    assert kw["speculative"] is True
    assert kw["spec_max_drafts"] == 4
    assert kw["prefix_cache"] is True
    assert kw["flash_attn"] is True


def test_27b_recipe_forwards_mlx_defaults(monkeypatch, rootdir):
    rc, kw = _serve_kwargs(monkeypatch, ["--recipe", "27b"])
    assert rc == 0
    assert kw["engine"] == "mlx"
    assert kw["n_ctx"] == 8192
    assert kw["speculative"] is False
    assert kw["prefix_cache"] is False
    assert kw["mlx_prefix_cache"] is True
    assert kw["mlx_prefix_cache_size"] == 2


def test_explicit_flag_beats_recipe(monkeypatch, rootdir):
    rc, kw = _serve_kwargs(monkeypatch, ["--recipe", "30b", "--n-seq-max", "8"])
    assert rc == 0
    assert kw["n_seq_max"] == 8


def test_no_flash_attn_flag_reaches_engine(monkeypatch, rootdir):
    rc, kw = _serve_kwargs(monkeypatch, ["--recipe", "30b", "--no-flash-attn"])
    assert rc == 0
    assert kw["flash_attn"] is False


def test_draft_model_disables_spec_with_warning(monkeypatch, rootdir, capsys):
    rc, kw = _serve_kwargs(
        monkeypatch, ["--recipe", "30b", "--draft-model", "models/tiny.gguf"]
    )
    assert rc == 0
    assert kw["draft_model_path"] == "models/tiny.gguf"
    assert kw["speculative"] is False
    assert "speculation" in capsys.readouterr().err.lower()


def test_unknown_recipe_key_rejected(monkeypatch, rootdir, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": "m", "n_ctx": 1024, "bogus_knob": 1}))
    rc, _ = _serve_kwargs(monkeypatch, ["--recipe", str(bad)])
    assert rc == 2


def test_missing_model_without_recipe_fails(monkeypatch, rootdir):
    from bwr.cli import _cmd_serve

    assert _cmd_serve([]) == 2
