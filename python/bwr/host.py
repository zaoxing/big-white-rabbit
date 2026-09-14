"""Host hardware probing + memory-fit parameter adaptation.

Recipes are tuned on a 64GB M1 Max. On a smaller (or bigger) Mac the same
``n_ctx`` can OOM or leave headroom on the table, so every command that
allocates a context adapts ``--ctx-size`` to the machine it runs on:

- :func:`probe` reads the Mac model, chip, CPU count, GPU core count and
  unified-memory size via ``sysctl``/``system_profiler``. It never raises:
  anything unreadable comes back ``None``/``0`` and adaptation is skipped.
- :func:`adapt_ctx` is the pure policy: given a requested ``n_ctx``, the
  model footprint in bytes and caps, it clamps ``n_ctx`` down the ladder
  ``8192 → 4096 → 2048 → 1024 → 512`` until weights + a 4GiB OS reserve +
  estimated KV fit in RAM. An explicitly passed ``--ctx-size`` is honoured
  (explicit wins, like recipes) but still fails fast when it cannot fit.
- KV per token is a documented heuristic by model-size tier, deliberately
  conservative (~2-4x the fp16 math): under-clamping wastes context, but
  over-clamping OOMs the process, and only one of those is recoverable.

``bwr host`` prints the detected caps and the fit verdict for a model.
``--no-adapt`` on serve/generate/tune disables the adjustment.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field


# Down the ladder until it fits; 512 is the floor (below that llama.cpp's
# own n_batch clamping makes the context useless anyway).
CTX_LADDER = (8192, 4096, 2048, 1024, 512)

# Bytes the OS + framework + compute buffers need beside the weights.
# Metal compute buffers run ~0.3-2GB on the measured models; 4GiB covers
# them with margin without starving mid-size machines.
OS_RESERVE_BYTES = 4 * 1024**3

# Model weights must leave this fraction of RAM free to even attempt a load
# (mmap needs the address space, Metal needs resident pages).
MODEL_FIT_FRACTION = 0.75


@dataclass
class HostCaps:
    """What the machine is. ``None``/``0`` = could not be detected."""

    machine_model: str | None = None  # e.g. MacBookPro18,2 (hw.model)
    chip: str | None = None           # e.g. Apple M1 Max
    cpu_count: int = 0                # logical CPUs (hw.ncpu)
    phys_cpus: int = 0                # physical CPUs (hw.physicalcpu)
    mem_bytes: int = 0                # unified memory (hw.memsize)
    gpu_cores: int | None = None      # Apple GPU cores, system_profiler
    notes: list[str] = field(default_factory=list)


def _sysctl(runner, key: str) -> str | None:
    try:
        out = runner(["sysctl", "-n", key])
    except Exception:  # noqa: BLE001 - probe is best-effort; any failure (missing
        # sysctl, unreadable key, non-Mac) means "unknown", and the caller
        # already degrades on None. Never let a probe abort a serve.
        return None
    text = out.strip() if isinstance(out, str) else None
    return text or None


def probe(
    _sysctl_runner=None,
    _sp_runner=None,
) -> HostCaps:
    """Detect host caps. Runners are injectable for weight-free tests."""
    caps = HostCaps()

    def sysctl(args: list[str]) -> str:
        if _sysctl_runner is not None:
            return _sysctl_runner(args)
        return subprocess.run(
            args, capture_output=True, text=True, timeout=10
        ).stdout

    def sysprof(args: list[str]) -> str:
        if _sp_runner is not None:
            return _sp_runner(args)
        return subprocess.run(
            args, capture_output=True, text=True, timeout=30
        ).stdout

    caps.machine_model = _sysctl(sysctl, "hw.model")
    # machdep.cpu.brand_string is "Apple M1 Max" on Apple Silicon.
    caps.chip = _sysctl(sysctl, "machdep.cpu.brand_string")
    for attr, key in (("cpu_count", "hw.ncpu"), ("phys_cpus", "hw.physicalcpu"),
                      ("mem_bytes", "hw.memsize")):
        raw = _sysctl(sysctl, key)
        try:
            setattr(caps, attr, int(raw) if raw is not None else 0)
        except ValueError:
            setattr(caps, attr, 0)

    # GPU cores: SPDisplaysDataType reports "Total Number of Cores: N" for
    # the Apple GPU. Best-effort: absent on non-Mac or minimal installs.
    try:
        sp = sysprof(["system_profiler", "SPDisplaysDataType"])
        m = re.search(r"Total Number of Cores:\s*(\d+)", sp)
        caps.gpu_cores = int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001 - same best-effort contract: system_profiler
        # is absent on minimal installs and slow/flaky under load. Unknown GPU
        # core count is reported as None, not an error.
        caps.gpu_cores = None

    if not caps.mem_bytes:
        caps.notes.append("RAM size unreadable; fit adaptation disabled")
    return caps


def model_bytes(path: str) -> int:
    """Resident footprint estimate: file size for GGUF, tree size for MLX dirs."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total


def kv_bytes_per_token(model_b: int) -> int:
    """Conservative KV estimate per context token, by weights tier.

    27B-class fp16 math is ~0.25MB/tok (64L × 8 KV heads × 128 dim × 2×2B);
    tiers below double that margin for smaller machines where the OS reserve
    dominates anyway. Deliberately coarse: this only picks a ladder rung.
    """
    if model_b > 20 * 1024**3:
        return 512 * 1024
    if model_b > 8 * 1024**3:
        return 320 * 1024
    return 160 * 1024


@dataclass
class AdaptResult:
    n_ctx: int
    notes: list[str] = field(default_factory=list)


class HostFitError(RuntimeError):
    """The model cannot fit this machine's RAM at any usable context."""


def adapt_ctx(
    want_ctx: int,
    model_b: int,
    caps: HostCaps,
    explicit: bool = False,
) -> AdaptResult:
    """Clamp ``want_ctx`` until weights + reserve + KV fit in RAM.

    ``explicit`` = the user passed ``--ctx-size`` themselves: honour it
    when it fits, fail fast when it cannot. Returns the (possibly
    unchanged) ctx plus human-readable notes for stderr logging.
    """
    if not caps.mem_bytes or model_b <= 0:
        return AdaptResult(want_ctx)
    mem = caps.mem_bytes
    if model_b > mem * MODEL_FIT_FRACTION:
        raise HostFitError(
            f"model is {model_b / 1024**3:.1f}GiB on a {mem / 1024**3:.0f}GiB Mac; "
            f"weights alone exceed {MODEL_FIT_FRACTION:.0%} of RAM "
            f"({caps.chip or 'unknown chip'}). Use a smaller quant, a smaller "
            f"model, or the recipe fallback (e.g. --engine metal with a GGUF)."
        )
    per_tok = kv_bytes_per_token(model_b)
    base = model_b + OS_RESERVE_BYTES

    def fits(ctx: int) -> bool:
        return base + ctx * per_tok <= mem

    if fits(want_ctx):
        return AdaptResult(want_ctx)
    if explicit:
        raise HostFitError(
            f"--ctx-size {want_ctx} needs ~{(base + want_ctx * per_tok) / 1024**3:.1f}GiB "
            f"(weights + reserve + KV) on a {mem / 1024**3:.0f}GiB Mac. "
            f"Lower --ctx-size or pass --no-adapt to take responsibility."
        )
    for rung in CTX_LADDER:
        if rung < want_ctx and fits(rung):
            return AdaptResult(rung, notes=[
                f"host adapt ({caps.chip or 'unknown Mac'}, "
                f"{mem / 1024**3:.0f}GiB RAM): n_ctx {want_ctx}->{rung} to fit "
                f"weights + KV in memory (--no-adapt to override)",
            ])
    raise HostFitError(
        f"no usable n_ctx fits a {model_b / 1024**3:.1f}GiB model on a "
        f"{mem / 1024**3:.0f}GiB Mac (floor is {CTX_LADDER[-1]})."
    )


def describe(caps: HostCaps) -> str:
    """One-screen `bwr host` report."""
    def gb(b: int) -> str:
        return f"{b / 1024**3:.0f}GiB" if b else "?"

    lines = [
        f"machine       : {caps.machine_model or '?'}",
        f"chip          : {caps.chip or '?'}",
        f"cpus          : {caps.phys_cpus or '?'} physical / {caps.cpu_count or '?'} logical",
        f"gpu_cores     : {caps.gpu_cores if caps.gpu_cores is not None else '?'}",
        f"memory        : {gb(caps.mem_bytes)} unified",
    ]
    lines.extend(f"note          : {n}" for n in caps.notes)
    return "\n".join(lines)
