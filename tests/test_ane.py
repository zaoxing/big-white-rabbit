"""ANE capability probe.

The value here is that the answer is PROBED, not hardcoded: if MLX ever
exposes an ANE device, or a CoreML backend is added, the report has to
change on its own rather than waiting for someone to notice a stale string.
"""

from __future__ import annotations

import sys
import types

from bwr.server import ane


def test_ane_is_unavailable_on_this_build():
    report = ane.probe()
    assert report["available"] is False
    assert report["candidates"] == []
    assert report["reason"]


def test_the_report_names_every_check_that_failed():
    names = {c["name"] for c in ane.probe()["checks"]}
    assert names == {"mlx_ane_device", "coremltools", "bwr_coreml_backend"}


def test_mlx_having_no_ane_device_is_the_first_reason():
    """MLX exposes cpu and gpu only, so there is nothing to place work on."""
    check = next(c for c in ane.probe()["checks"] if c["name"] == "mlx_ane_device")
    assert check["ok"] is False
    assert "cpu" in check["detail"] and "gpu" in check["detail"]


def test_the_hardware_is_not_blamed():
    """Every Apple Silicon Mac has an ANE. The blocker is the software path,
    and the report must not suggest otherwise."""
    report = ane.probe()
    assert report["chip"] is None or report["chip"].startswith("Apple")


def test_the_answer_is_probed_rather_than_hardcoded(monkeypatch):
    """The point of the module: satisfy every check and it reports available.

    A hardcoded "unavailable" would pass the tests above and silently stay
    wrong the day a CoreML path lands.
    """
    monkeypatch.setattr(ane, "_mlx_devices", lambda: ["cpu", "gpu", "ane"])
    monkeypatch.setitem(sys.modules, "coremltools", types.ModuleType("coremltools"))
    monkeypatch.setitem(
        sys.modules, "bwr.server.ane_backend", types.ModuleType("ane_backend")
    )
    report = ane.probe()
    assert report["available"] is True
    assert report["reason"] is None
