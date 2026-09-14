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


# -- MTP-head speculation (SPEC-mlx-mtp-draft.md) ----------------------------
#
# Same guard as the rest of this file: a knob that never reaches the engine is
# worse than one that fails loudly. mlx_mtp is off by default because it is
# output-identical but SLOWER on mlx-lm 0.31.3 (the trunk forward is linear in
# rows), so these pin reachability, not a recommendation to enable it.


def test_mlx_mtp_defaults_off(monkeypatch, rootdir, tmp_path):
    rc, kw = _serve_kwargs(
        monkeypatch, ["-m", str(tmp_path), "--engine", "mlx"]
    )
    assert rc == 0
    assert kw["mlx_mtp"] is False
    assert kw["mlx_mtp_depth"] == 0


def test_mlx_mtp_flag_reaches_serve(monkeypatch, rootdir, tmp_path):
    rc, kw = _serve_kwargs(
        monkeypatch,
        ["-m", str(tmp_path), "--engine", "mlx", "--mlx-mtp", "--mlx-mtp-depth", "2"],
    )
    assert rc == 0
    assert kw["mlx_mtp"] is True
    assert kw["mlx_mtp_depth"] == 2


def test_mlx_mtp_recipe_keys_are_accepted(monkeypatch, rootdir, tmp_path):
    """A recipe carrying mlx_mtp must not be rejected as an unknown key."""
    recipe = tmp_path / "mtp.json"
    recipe.write_text(
        json.dumps(
            {
                "model": str(tmp_path),
                "engine": "mlx",
                "n_ctx": 4096,
                "mlx_mtp": True,
                "mlx_mtp_depth": 3,
            }
        )
    )
    rc, kw = _serve_kwargs(monkeypatch, ["--recipe", str(recipe)])
    assert rc == 0, "recipe with mlx_mtp was rejected"
    assert kw["mlx_mtp"] is True
    assert kw["mlx_mtp_depth"] == 3


def test_serve_accepts_every_kwarg_the_cli_passes(monkeypatch, rootdir, tmp_path):
    """The mocked-serve tests above cannot see serve()'s real signature.

    Regression: --mlx-mtp reached _cmd_serve and was forwarded to serve(),
    which did not accept it -- every mocked test passed while the real server
    died with TypeError on startup. Compare against the true signature.
    """
    import inspect

    import bwr.server.launch as launch

    real_params = set(inspect.signature(launch.serve).parameters)
    _rc, kw = _serve_kwargs(
        monkeypatch,
        ["-m", str(tmp_path), "--engine", "mlx", "--mlx-mtp", "--mlx-mtp-depth", "2"],
    )
    passed = set(kw) - {"model"}  # model is positional
    missing = sorted(passed - real_params)
    assert not missing, f"serve() would TypeError on: {missing}"


# -- the same number in three places -------------------------------------------


def _serve_help() -> str:
    """`bwr serve --help` as the user sees it, whitespace-normalised.

    Taken from the rendered output rather than the parser object because
    each subcommand builds its ArgumentParser inside its own function --
    there is no module-level parser to introspect. argparse wraps on
    whitespace, so a token like `15.7` is never split; collapsing runs of
    whitespace is enough to make the text searchable.
    """
    import contextlib
    import io

    from bwr.cli import _cmd_serve

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        _cmd_serve(["--help"])
    return " ".join(buf.getvalue().split())


@pytest.mark.parametrize("shorthand", ["27b", "30b"])
def test_recipe_help_quotes_the_recipes_own_bench(rootdir, shorthand):
    """The throughput figure lives in three places -- the recipe JSON's
    `bench.tok_s`, the README table, and this help string -- and it has now
    drifted twice. T18 caught README vs recipe (13.2 vs 10.9) and fixed the
    README; the CLI kept saying 13.2 for another five turns because nothing
    compared them.

    The JSON is the source of truth: it is written by whoever ran the bench.
    """
    bench = json.loads(
        (rootdir / "models" / "recipes" / f"{shorthand}.json").read_text()
    )["bench"]
    measured = bench["tok_s"]
    assert f"{measured}" in _serve_help(), (
        f"--recipe help does not quote {shorthand}.json's measured "
        f"{measured} tok/s; one of the two is stale"
    )


def test_recipe_help_does_not_quote_the_retired_27b_figure():
    """13.2 was the censored Qwen3.8-27B-MLX-4bit, which is not on disk any
    more and was never comparable to the build the recipe now points at."""
    assert "13.2" not in _serve_help()


def test_serve_accepts_every_flag_the_macos_app_passes(monkeypatch, rootdir, tmp_path):
    """The menubar app spawns `bwr serve` with a fixed argv (ServerProcess
    .makeArguments). A flag it passes that argparse rejects exits 2 and the
    app can never start a server -- that is exactly how --base-path shipped
    broken. Parse the app's argv here so the two cannot drift apart.
    """
    argv = ["--model-dir", str(tmp_path), "--host", "127.0.0.1",
            "--port", "1919", "--preload", "first"]
    rc, kw = _serve_kwargs(monkeypatch, argv)
    assert rc == 0, f"bwr serve rejected the app's argv: {argv}"
    assert kw["preload"] == "first"
    assert kw["model_dir"] == str(tmp_path)


# -- bwr's own log output -----------------------------------------------------


def test_bwr_loggers_reach_a_handler_under_serve():
    """bwr's INFO records must actually be emitted.

    uvicorn configures only its own logger namespace and leaves the root
    logger without a handler, so `logging.lastResort` dropped everything bwr
    logged below WARNING. The packaged app's server.log therefore never said
    whether `--preload` warmed a model -- the one place a user would look.
    """
    import logging

    from bwr.server.launch import _configure_bwr_logging

    bwr_logger = logging.getLogger("bwr")
    saved = (list(bwr_logger.handlers), bwr_logger.level, bwr_logger.propagate)
    try:
        bwr_logger.handlers.clear()
        _configure_bwr_logging("info")
        assert bwr_logger.handlers, "bwr records would go to lastResort"
        assert bwr_logger.isEnabledFor(logging.INFO)
        # Scoped to bwr: turning our own logging on must not also switch on
        # INFO chatter from transformers / httpx / mlx.
        assert not logging.getLogger().handlers or bwr_logger.propagate is False
    finally:
        bwr_logger.handlers[:] = saved[0]
        bwr_logger.setLevel(saved[1])
        bwr_logger.propagate = saved[2]
