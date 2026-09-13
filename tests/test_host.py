"""Weight-free tests for host probing + ctx fit adaptation (no hardware needed)."""

import pytest

from bwr.host import (
    CTX_LADDER,
    HostCaps,
    HostFitError,
    adapt_ctx,
    describe,
    kv_bytes_per_token,
    probe,
)

GiB = 1024**3


def _fake_probe():
    values = {
        "hw.model": "MacBookPro18,2\n",
        "machdep.cpu.brand_string": "Apple M1 Max\n",
        "hw.ncpu": "10\n",
        "hw.physicalcpu": "10\n",
        "hw.memsize": f"{64 * GiB}\n",
    }
    sp = "Chipset Model: Apple M1 Max\n  Total Number of Cores: 32\n"
    return probe(_sysctl_runner=lambda a: values[a[-1]],
                 _sp_runner=lambda a: sp)


def test_probe_parses_all_fields():
    caps = _fake_probe()
    assert caps.machine_model == "MacBookPro18,2"
    assert caps.chip == "Apple M1 Max"
    assert caps.cpu_count == 10
    assert caps.phys_cpus == 10
    assert caps.mem_bytes == 64 * GiB
    assert caps.gpu_cores == 32
    assert caps.notes == []


def test_probe_never_raises():
    caps = probe(_sysctl_runner=lambda a: (_ for _ in ()).throw(OSError("no sysctl")),
                 _sp_runner=lambda a: (_ for _ in ()).throw(OSError("no sp")))
    assert caps.mem_bytes == 0
    assert caps.gpu_cores is None
    assert caps.notes  # explains adaptation is disabled


def test_adapt_no_clamp_when_fits():
    caps = HostCaps(chip="Apple M1 Max", mem_bytes=64 * GiB)
    res = adapt_ctx(8192, 15 * GiB, caps)
    assert res.n_ctx == 8192
    assert res.notes == []


def test_adapt_clamps_down_ladder():
    caps = HostCaps(chip="Apple M2", mem_bytes=8 * GiB)
    res = adapt_ctx(8192, int(2.8 * GiB), caps)
    assert res.n_ctx == 4096
    assert len(res.notes) == 1 and "4096" in res.notes[0]


def test_adapt_model_too_big_errors():
    caps = HostCaps(chip="Apple M1", mem_bytes=16 * GiB)
    with pytest.raises(HostFitError, match="fallback"):
        adapt_ctx(8192, 15 * GiB, caps)


def test_adapt_explicit_honoured_when_fits():
    caps = HostCaps(chip="Apple M1 Max", mem_bytes=64 * GiB)
    res = adapt_ctx(2048, 15 * GiB, caps, explicit=True)
    assert res.n_ctx == 2048


def test_adapt_explicit_still_fails_fast_on_oom():
    caps = HostCaps(chip="Apple M1", mem_bytes=16 * GiB)
    with pytest.raises(HostFitError, match="--ctx-size"):
        adapt_ctx(8192, int(10 * GiB), caps, explicit=True)


def test_adapt_skipped_without_caps_or_size():
    assert adapt_ctx(8192, 0, HostCaps()).n_ctx == 8192
    assert adapt_ctx(8192, 15 * GiB, HostCaps()).n_ctx == 8192


def test_kv_tiers_documented():
    assert kv_bytes_per_token(21 * GiB) > kv_bytes_per_token(9 * GiB)
    assert kv_bytes_per_token(9 * GiB) > kv_bytes_per_token(GiB)
    assert CTX_LADDER[0] == 8192 and CTX_LADDER[-1] == 512


def test_describe_renders(capsys):
    print(describe(_fake_probe()))
    out = capsys.readouterr().out
    assert "MacBookPro18,2" in out and "32" in out and "64GiB" in out


def test_cmd_host_no_model(capsys):
    from bwr.cli import _cmd_host

    assert _cmd_host([]) == 0
    assert "memory" in capsys.readouterr().out


def test_cmd_host_fit_small_model(tmp_path, capsys):
    from bwr.cli import _cmd_host

    tiny = tmp_path / "m.gguf"
    tiny.write_bytes(b"\0" * 1024)
    assert _cmd_host(["-m", str(tiny), "-c", "1024"]) == 0
    assert "fit" in capsys.readouterr().out


def test_ctx_explicit_forms():
    from bwr.cli import _ctx_explicit

    assert _ctx_explicit(["-m", "x", "-c", "2048"])
    assert _ctx_explicit(["--ctx-size=2048"])
    assert _ctx_explicit(["--ctx-size", "2048"])
    assert not _ctx_explicit(["-m", "x"])
    assert not _ctx_explicit(["--n-batch", "512"])


def test_no_adapt_skips_probe(monkeypatch):
    import bwr.cli as cli

    def boom(*a, **k):
        raise AssertionError("probe must not run with --no-adapt")

    monkeypatch.setattr("bwr.host.probe", boom)
    import argparse

    ns = argparse.Namespace(no_adapt=True, model="x.gguf", ctx_size=8192)
    assert cli._adapt_ctx_or_die(ns, False, "tune") is None
    assert ns.ctx_size == 8192
