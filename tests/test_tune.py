"""bwr tune logic: grid parsing, winner gate, rendering. No weights needed."""

from __future__ import annotations

import pytest

from bwr.tune import depth_name, parse_depths, pick_winner, render


def test_parse_depths_default_grid():
    assert parse_depths("off,2,4") == [None, 2, 4]


def test_parse_depths_inserts_baseline():
    assert parse_depths("2,4")[0] is None


def test_parse_depths_zero_is_off():
    assert parse_depths("0,4") == [None, 4]


def test_parse_depths_rejects_garbage():
    with pytest.raises(ValueError):
        parse_depths("off,ludicrous")
    with pytest.raises(ValueError):
        parse_depths("off,-2")
    with pytest.raises(ValueError):
        parse_depths("")


def test_pick_winner_needs_noise_margin():
    base = {"depth": None, "tok_s": 100.0, "accept": None}
    # +4% is noise, not a win.
    assert pick_winner([base, {"depth": 4, "tok_s": 104.0, "accept": 0.7}]) is None
    # +7.5% wins.
    win = pick_winner([base, {"depth": 4, "tok_s": 107.5, "accept": 0.7}])
    assert win is not None and win["depth"] == 4


def test_pick_winner_baseline_fastest():
    rows = [
        {"depth": None, "tok_s": 100.0, "accept": None},
        {"depth": 2, "tok_s": 90.0, "accept": 0.3},
    ]
    assert pick_winner(rows) is None


def test_render_verdict_lines():
    rows = [
        {"depth": None, "tok_s": 100.0, "accept": None},
        {"depth": 4, "tok_s": 110.0, "accept": 0.7},
    ]
    assert "keep baseline" in render(rows, None)
    out = render(rows, rows[1])
    assert "--spec-max-drafts 4" in out
    assert depth_name(None) == "off (AR)"


def test_cmd_tune_mocked_end_to_end(monkeypatch, capsys):
    """_cmd_tune with model + bench mocked: exercises wiring, no weights."""
    import bwr
    import bwr.tune as tune

    monkeypatch.setattr(bwr, "Model", lambda *a, **k: object())
    monkeypatch.setattr(
        bwr, "ModelParams", lambda *a, **k: type("MP", (), {})()
    )
    calls = []

    def fake_bench(model, **kw):
        calls.append(kw["depth"])
        d = kw["depth"]
        return {"depth": d, "tok_s": 100.0 if d is None else 112.0, "accept": None}

    monkeypatch.setattr(tune, "bench_one", fake_bench)
    # _cmd_tune does `from . import Model` / `from .tune import ...` at call
    # time, so patching the source modules suffices.
    from bwr.cli import _cmd_tune

    rc = _cmd_tune(["-m", "x.gguf", "--depths", "off,4", "--reps", "1"])
    assert rc == 0
    assert calls == [None, 4]
    out = capsys.readouterr().out
    assert "--spec-max-drafts 4" in out


def test_cmd_tune_rejects_non_gguf(capsys):
    from bwr.cli import _cmd_tune

    assert _cmd_tune(["-m", "models/Qwen3.8-27B-MLX-4bit"]) == 2
    assert ".gguf" in capsys.readouterr().err


def test_cmd_tune_mlx_needs_dir(capsys):
    from bwr.cli import _cmd_tune

    assert _cmd_tune(["-m", "x.gguf", "--engine", "mlx"]) == 2
    assert "weights dir" in capsys.readouterr().err


def test_cmd_tune_mlx_mocked_end_to_end(monkeypatch, capsys, tmp_path):
    """MLX path wires the weights dir through with no GGUF involved."""
    import bwr.engine as engine_mod
    import bwr.tune as tune

    assert tune is not None

    class FakeCtx:
        def close(self):
            pass

    class FakeMLX:
        def __init__(self, path, cfg):
            self.path = path
            self.cfg = cfg
            self.ctx = FakeCtx()
            self.spec_acceptance_rate = 0.5

        def add_request(self, prompt, params):
            return 0

        def drain(self):
            return iter([object()])

        def tokens_of(self, rid):
            return [1] * 8

    monkeypatch.setattr(engine_mod, "MLXEngine", FakeMLX)
    from bwr.cli import _cmd_tune

    rc = _cmd_tune(["-m", str(tmp_path), "--engine", "mlx",
                    "--depths", "off,2", "--reps", "1", "--max-tokens", "8"])
    assert rc == 0
    assert "tok/s" in capsys.readouterr().out
