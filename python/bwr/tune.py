"""Per-machine decode tuning (``bwr tune``).

Measures greedy decode throughput for a small candidate grid on the user's
own hardware and reports the winner. Report-only: prints a table plus the
flags to pass to ``bwr serve``. A candidate must beat the autoregressive
baseline by more than ``NOISE`` (5%, the run-to-run tok/s variance on Apple
Silicon) or the verdict is "keep baseline".

v1 tunes n-gram depth, the knob with a measured spread (drafts=4 +7.5%
on repetitive Metal text, drafts=8 collapses). Both backends: Metal tunes a
GGUF file, MLX tunes a weights dir (MLXEngine loads it per candidate).
"""

from __future__ import annotations

import statistics
import time

# Deltas inside this band are run-to-run noise on Apple Silicon, not wins
# (loop-constraints.md: Environment realities).
NOISE = 0.05

# n-gram speculation only pays on repetitive text (acceptance needs repeats),
# so the default probe is deliberately loopy: one repeated word (same shape
# as the prefix-cache suite's stem, whose pieces decode cleanly). Realistic
# steady state: one engine per candidate, reps run sequentially so the table
# warms as in serve.
DEFAULT_PROMPT = ("word " * 150).strip()

DEFAULT_DEPTHS = ("off", "2", "4")


def parse_depths(raw: str) -> list[int | None]:
    """``"off,2,4"`` -> ``[None, 2, 4]``. Raises ValueError on bad entries."""
    out: list[int | None] = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if tok in ("off", "ar", "none"):
            out.append(None)
        else:
            try:
                n = int(tok)
            except ValueError:
                raise ValueError(f"bad depth {tok!r}: want 'off' or an int") from None
            if n < 0:
                raise ValueError(f"bad depth {tok!r}: must be >= 0")
            out.append(n if n > 0 else None)
    if not out:
        raise ValueError("empty depth grid")
    # Baseline first: plain autoregressive decoding is the comparison point.
    if None not in out:
        out.insert(0, None)
    return out


def depth_name(depth: int | None) -> str:
    return "off (AR)" if depth is None else f"drafts={depth}"


def bench_one(
    model,
    *,
    backend: str = "metal",
    prompt: str,
    max_tokens: int,
    reps: int,
    n_ctx: int,
    n_batch: int,
    depth: int | None,
) -> dict:
    """Median greedy decode tok/s for one depth. One engine per candidate.

    ``model`` is a loaded Metal ``Model`` for backend="metal", an MLX weights
    dir for backend="mlx" (MLXEngine loads it lazily itself).
    """
    from .engine import EngineConfig, RequestParams

    cfg = EngineConfig(
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_seq_max=1,
        speculative=depth is not None,
        spec_max_drafts=depth or 0,
    )
    if backend == "mlx":
        from .engine import MLXEngine

        eng = MLXEngine(model, cfg)
    else:
        from .engine import MetalEngine

        eng = MetalEngine(model, cfg)
    rates: list[float] = []
    for _ in range(reps):
        t0 = time.monotonic()
        rid = eng.add_request(
            prompt, RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False)
        )
        list(eng.drain())
        dt = time.monotonic() - t0
        n = len(eng.tokens_of(rid))
        rates.append(n / dt if dt > 0 else 0.0)
    if backend == "mlx":
        eng.ctx.close()
    return {
        "depth": depth,
        "tok_s": statistics.median(rates),
        "accept": eng.spec_acceptance_rate,
    }


def pick_winner(rows: list[dict]) -> dict | None:
    """Best row beating the baseline (rows[0]) by more than NOISE, else None."""
    base = rows[0]["tok_s"]
    best = max(rows, key=lambda r: r["tok_s"])
    if base <= 0:
        return None
    if best is rows[0]:
        return None
    return best if (best["tok_s"] - base) / base > NOISE else None


def render(rows: list[dict], winner: dict | None) -> str:
    lines = [f"{'depth':<12}{'tok/s':>10}{'accept':>10}"]
    for r in rows:
        acc = "-" if r["accept"] is None else f"{r['accept']:.0%}"
        mark = "  <--" if winner is r else ""
        lines.append(f"{depth_name(r['depth']):<12}{r['tok_s']:>10.1f}{acc:>10}{mark}")
    if winner is None:
        lines.append("verdict: keep baseline (nothing beat AR past the 5% noise band)")
    elif winner["depth"] is None:
        lines.append("verdict: keep baseline")
    else:
        lines.append(
            f"verdict: --speculative --spec-max-drafts {winner['depth']} "
            f"({winner['tok_s'] / rows[0]['tok_s'] - 1:+.1%} vs AR)"
        )
    return "\n".join(lines)
