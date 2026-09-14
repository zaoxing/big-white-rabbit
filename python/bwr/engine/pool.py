"""Multi-model pool: discover, lazily load, LRU-evict.

Why this exists
===============

The server was one engine over one model for its whole life (see
`server/app.py`'s module docstring). That is the right shape for a dedicated
serving process, but it leaves every model-management surface in the web UI
with nothing to act on, and it means switching models is a process restart.

This adds the missing piece without changing that core bet: still one
process, still one decode thread per engine, still a direct call rather than
an IPC hop. The pool just owns more than one engine and decides which are
resident.

Residency policy
================

Loading a 27B is ~15-18 GiB of weights, so "keep them all warm" is not
available on a 64 GiB Mac. The pool holds a byte budget and evicts
least-recently-used engines until the incoming model fits:

- the budget comes from `host.probe()` (real unified memory), minus an OS
  reserve, unless the caller passes one
- eviction is strictly LRU by last *acquire*, not by load time, so a model
  being actively served is the last thing dropped
- an engine is never evicted while it has work in flight; the pool waits for
  it to drain rather than killing live requests

A model larger than the whole budget is refused at admission with an
actionable error, rather than loaded and then OOM-ing the process.

What the budget is NOT
======================

Measured on this machine: three 27B-class models "loaded" concurrently put
`resident_bytes` at 57.8 GiB while the process RSS was **2.2 GiB**. MLX mmaps
weights, so loading maps a file rather than filling RAM, and the OS pages
blocks in on demand.

So this accounting is a POLICY guardrail sized from file bytes, not a
measurement of memory in use. Two consequences worth knowing before tuning
it:

- The budget bounds how much weight data we are willing to have mapped, and
  how many engines exist (each does hold real KV cache and Metal buffers).
  It does not predict RSS.
- The true cost of over-admitting is page-cache thrash, not OOM. Alternating
  between two 27Bs that both "fit" re-pages ~15-18 GiB per switch and drifts
  throughput by ~25% -- measured while benchmarking quants earlier in this
  repo. Eviction therefore earns its keep even when RSS looks fine.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import host
from .async_engine import AsyncEngine
from .config import EngineConfig

logger = logging.getLogger(__name__)

# Weights are not the whole cost: KV, activations and the Metal heap ride
# along. Reserve headroom so a model that "just fits" on paper does not push
# the process into swap.
_OS_RESERVE_BYTES = 4 * 1024**3
_LOAD_OVERHEAD = 1.15

_MLX_MARKER = "config.json"
_GGUF_SUFFIX = ".gguf"


@dataclass
class ModelEntry:
    """A model the pool can serve, loaded or not."""

    model_id: str
    path: Path
    kind: str  # "mlx" | "metal"
    size_bytes: int
    loaded: bool = False
    last_used: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "path": str(self.path),
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "loaded": self.loaded,
        }


@dataclass
class _Loaded:
    engine: AsyncEngine
    raw: Any
    entry: ModelEntry
    in_flight: int = 0
    drained: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.drained.set()


def discover(model_dir: str | Path) -> list[ModelEntry]:
    """Find servable models directly under `model_dir`.

    An MLX model is a directory holding `config.json`; a Metal model is a
    `.gguf` file. Nested MLX layouts (a `4-bit/` subdirectory, as some
    published repos use) are followed one level so the published shape works
    without the user reorganising their files.
    """
    root = Path(model_dir)
    out: list[ModelEntry] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_file() and child.suffix == _GGUF_SUFFIX:
            out.append(
                ModelEntry(child.stem, child, "metal", host.model_bytes(str(child)))
            )
            continue
        if not child.is_dir():
            continue
        if (child / _MLX_MARKER).is_file():
            out.append(
                ModelEntry(child.name, child, "mlx", host.model_bytes(str(child)))
            )
            continue
        # one level down: models/<name>/<variant>/config.json
        for grand in sorted(child.iterdir()):
            if grand.is_dir() and (grand / _MLX_MARKER).is_file():
                out.append(
                    ModelEntry(
                        f"{child.name}/{grand.name}",
                        grand,
                        "mlx",
                        host.model_bytes(str(grand)),
                    )
                )
    return out


class ModelPoolError(RuntimeError):
    """Admission refused: unknown model, or one that cannot ever fit."""


class ModelPool:
    """Owns the engines. One asyncio lock serialises load/evict."""

    def __init__(
        self,
        model_dir: str | Path,
        config: EngineConfig,
        *,
        budget_bytes: int | None = None,
        engine_factory: Callable[[ModelEntry, EngineConfig], Any] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config
        self._entries: dict[str, ModelEntry] = {
            e.model_id: e for e in discover(model_dir)
        }
        self._loaded: OrderedDict[str, _Loaded] = OrderedDict()
        self._lock = asyncio.Lock()
        self._factory = engine_factory or _build_engine
        self._clock = 0.0
        if budget_bytes is not None:
            self.budget_bytes = budget_bytes
        else:
            caps = host.probe()
            self.budget_bytes = max(0, caps.mem_bytes - _OS_RESERVE_BYTES)

    # -- inspection --------------------------------------------------------

    def rescan(self) -> None:
        """Pick up models added on disk since startup, keeping loaded state."""
        for entry in discover(self.model_dir):
            if entry.model_id in self._entries:
                continue
            self._entries[entry.model_id] = entry

    def list(self) -> list[dict[str, Any]]:
        for mid, entry in self._entries.items():
            entry.loaded = mid in self._loaded
        return [e.to_dict() for e in self._entries.values()]

    @property
    def loaded_ids(self) -> list[str]:
        return list(self._loaded)

    @property
    def resident_bytes(self) -> int:
        return sum(int(l.entry.size_bytes * _LOAD_OVERHEAD) for l in self._loaded.values())

    def engine_for(self, model_id: str) -> Any | None:
        """The raw engine if resident, else None. Read-only callers (stats,
        /health) use this so they never trigger a load."""
        rec = self._loaded.get(model_id)
        return rec.raw if rec is not None else None

    def resolve(self, model_id: str | None) -> str:
        """Map a request's `model` onto a known id.

        A single known model answers to any name, which keeps clients that
        hardcode "local" or "gpt-3.5-turbo" working -- the same latitude the
        single-model server had. With several models the name must match.
        """
        if model_id and model_id in self._entries:
            return model_id
        if len(self._entries) == 1:
            return next(iter(self._entries))
        if not self._entries:
            raise ModelPoolError(f"no models found under {self.model_dir}")
        raise ModelPoolError(
            f"unknown model {model_id!r}; available: {', '.join(sorted(self._entries))}"
        )

    # -- residency ---------------------------------------------------------

    async def acquire(self, model_id: str | None) -> tuple[str, AsyncEngine]:
        """Resolve, load if needed, and mark most-recently-used."""
        mid = self.resolve(model_id)
        entry = self._entries[mid]
        async with self._lock:
            rec = self._loaded.get(mid)
            if rec is None:
                await self._make_room(entry)
                rec = await self._load(entry)
            self._loaded.move_to_end(mid)
            self._clock += 1.0
            entry.last_used = self._clock
            return mid, rec.engine

    def request_started(self, model_id: str) -> None:
        rec = self._loaded.get(model_id)
        if rec is not None:
            rec.in_flight += 1
            rec.drained.clear()

    def request_finished(self, model_id: str) -> None:
        rec = self._loaded.get(model_id)
        if rec is None:
            return
        rec.in_flight = max(0, rec.in_flight - 1)
        if rec.in_flight == 0:
            rec.drained.set()

    async def _make_room(self, incoming: ModelEntry) -> None:
        need = int(incoming.size_bytes * _LOAD_OVERHEAD)
        if self.budget_bytes and need > self.budget_bytes:
            raise ModelPoolError(
                f"{incoming.model_id} needs ~{need // 1024**3} GiB but the pool budget "
                f"is {self.budget_bytes // 1024**3} GiB; it cannot be served on this host"
            )
        while self._loaded and self.resident_bytes + need > self.budget_bytes:
            victim_id = next(iter(self._loaded))  # LRU end
            await self._unload_locked(victim_id, reason="evicted for " + incoming.model_id)

    async def _load(self, entry: ModelEntry) -> _Loaded:
        logger.info("pool: loading %s (%s)", entry.model_id, entry.kind)
        raw = await asyncio.to_thread(self._factory, entry, self.config)
        engine = AsyncEngine(raw)
        await engine.start()
        rec = _Loaded(engine=engine, raw=raw, entry=entry)
        self._loaded[entry.model_id] = rec
        entry.loaded = True
        return rec

    async def unload(self, model_id: str) -> bool:
        async with self._lock:
            return await self._unload_locked(model_id, reason="requested")

    async def _unload_locked(self, model_id: str, *, reason: str) -> bool:
        rec = self._loaded.get(model_id)
        if rec is None:
            return False
        if rec.in_flight:
            # Never kill live requests: wait for the stream to finish. The
            # alternative (dropping them) turns an eviction into a 500 for
            # whoever happened to be mid-generation.
            logger.info(
                "pool: %s has %d in flight, draining before unload",
                model_id, rec.in_flight,
            )
            await rec.drained.wait()
        logger.info("pool: unloading %s (%s)", model_id, reason)
        await rec.engine.stop()
        self._loaded.pop(model_id, None)
        rec.entry.loaded = False
        return True

    async def stop_all(self) -> None:
        async with self._lock:
            for mid in list(self._loaded):
                await self._unload_locked(mid, reason="shutdown")


def _build_engine(entry: ModelEntry, config: EngineConfig) -> Any:
    """Construct the backend engine for one entry. Runs off the event loop."""
    if entry.kind == "mlx":
        from .mlx_engine import MLXEngine

        cfg = _with_engine(config, "mlx")
        return MLXEngine(str(entry.path), cfg)
    from .._bwr_metal import Model
    from .metal_engine import MetalEngine

    cfg = _with_engine(config, "metal")
    model = Model(str(entry.path))
    return MetalEngine(model, cfg)


def _with_engine(config: EngineConfig, kind: str) -> EngineConfig:
    """A copy of `config` pinned to one backend.

    The pool can hold MLX and Metal models at once, so a single config's
    `engine` field cannot describe all of them; each load gets its own.
    """
    import dataclasses

    return dataclasses.replace(config, engine=kind)
