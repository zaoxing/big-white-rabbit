"""Whether this build can run anything on the Apple Neural Engine.

Short answer today: no, and not because of a missing switch.

bwr executes through MLX, and MLX exposes exactly two device types -- `cpu`
and `gpu`. There is no ANE device to target. The only supported route to the
ANE on Apple Silicon is CoreML, and bwr has no CoreML execution path: no
model conversion, no split prefill, no second runtime. So "ANE tuning" --
searching over which submodules run on the ANE versus the GPU -- has an
empty candidate set. That is an architectural fact, not a configuration one.

This module exists so the admin surface can say that with evidence rather
than returning a bare 501, and so the answer stays honest on its own. Every
check below is PROBED at call time: if MLX ever grows an ANE device, or a
CoreML backend is added to bwr, the report changes without anyone editing a
hardcoded "unavailable" string.
"""

from __future__ import annotations

import subprocess
from typing import Any

# Absolute path: a GUI-spawned process does not inherit a login shell's PATH
# (the same reason host.py pins sysctl).
_SYSCTL = "/usr/sbin/sysctl"


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def _mlx_devices() -> list[str]:
    try:
        import mlx.core as mx

        return [a for a in dir(mx.DeviceType) if not a.startswith("_")]
    except Exception:  # noqa: BLE001 - no MLX is itself an answer
        return []


def _apple_silicon_chip() -> str | None:
    """The chip name, or None off Apple Silicon.

    Reported for completeness only. Every Apple Silicon Mac has an ANE, so
    naming the chip pre-empts the obvious objection -- the blocker is the
    software path, not the hardware.
    """
    try:
        res = subprocess.run(
            [_SYSCTL, "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        chip = res.stdout.strip()
        return chip if chip.startswith("Apple") else None
    except Exception:  # noqa: BLE001 - best effort; absence is not an error
        return None


def probe() -> dict[str, Any]:
    """Structured capability report for the ANE surfaces."""
    devices = _mlx_devices()
    has_ane_device = any("ane" in d.lower() or "neural" in d.lower() for d in devices)

    try:
        import coremltools  # noqa: F401

        coreml = True
        coreml_detail = "coremltools is importable"
    except Exception:  # noqa: BLE001 - not installed is the common case
        coreml = False
        coreml_detail = "coremltools is not installed"

    try:
        from . import ane_backend  # type: ignore[attr-defined]  # noqa: F401

        backend = True
        backend_detail = "a CoreML execution path is present"
    except Exception:  # noqa: BLE001 - there is no such module today
        backend = False
        backend_detail = (
            "bwr has no CoreML execution path: no conversion, no split "
            "prefill, no second runtime"
        )

    checks = [
        _check(
            "mlx_ane_device", has_ane_device,
            f"MLX device types: {', '.join(devices) or 'unavailable'}"
            + ("" if has_ane_device else " -- no ANE device to target"),
        ),
        _check("coremltools", coreml, coreml_detail),
        _check("bwr_coreml_backend", backend, backend_detail),
    ]
    available = all(c["ok"] for c in checks)
    return {
        "available": available,
        "reason": None if available else (
            "bwr runs through MLX, which targets only CPU and GPU. Reaching "
            "the ANE needs a CoreML execution path, which this build does "
            "not have -- so there are no placements to tune."
        ),
        "checks": checks,
        # Kept as an empty list rather than omitted: the client iterates it,
        # and "no candidates" is the accurate answer.
        "candidates": [],
        # The hardware is not the blocker; naming the chip says so.
        "chip": _apple_silicon_chip(),
    }
