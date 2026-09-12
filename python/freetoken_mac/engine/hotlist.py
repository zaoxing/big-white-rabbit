"""Expert hotlist — per-layer LRU over routed experts (SPEC-ssd-hotlist.md, T11a).

No I/O, no buffer mutation, no engine changes yet: tracks what WOULD be
resident given a bounded cache, validates K vs hit rate on live traffic
before touching memory. Driven by `expert_activations()` frames after each
decode (full prob rows), extracting top-k routed experts per token/layer.

Single-process, no distribution, no eviction of non-routed weights (they
are always resident by invariant, same as ds4).
"""

from __future__ import annotations

from collections import OrderedDict


class ExpertHotlist:
    """Per-layer LRU cache over expert ids.

    `k_per_layer` is the resident budget per layer (e.g. 32). `top_k` is
    how many experts per token/layer are considered routed (Qwen3 MoE:
    8). Both must be >= 1.
    """

    def __init__(self, k_per_layer: int = 32, top_k: int = 8) -> None:
        if k_per_layer < 1:
            raise ValueError(f"k_per_layer must be >= 1; got {k_per_layer}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {top_k}")
        self.k_per_layer = k_per_layer
        self.top_k = top_k
        # layer -> OrderedDict[expert_id, None] in LRU order (LRU first, MRU last)
        self._caches: dict[int, OrderedDict[int, None]] = {}
        self.hits = 0
        self.misses = 0

    def _cache_for(self, layer: int) -> OrderedDict[int, None]:
        try:
            return self._caches[layer]
        except KeyError:
            od: OrderedDict[int, None] = OrderedDict()
            self._caches[layer] = od
            return od

    @staticmethod
    def _top_k_indices(row: list[float], k: int) -> list[int]:
        # Partial sort is fine for k=8 over 128; overhead negligible vs decode.
        indexed = sorted(range(len(row)), key=row.__getitem__, reverse=True)  # type: ignore[arg-type]
        return indexed[:k]

    def update(self, frames: list[dict]) -> dict[str, int]:
        """Update LRU from `expert_activations()` frames.

        Each frame: {"layer": int, "tokens": [[float, ...], ...]}.
        Returns {"hits": int, "misses": int} for this call.
        """
        hits = 0
        misses = 0
        for frame in frames:
            layer = int(frame["layer"])
            cache = self._cache_for(layer)
            for row in frame["tokens"]:
                top = self._top_k_indices(row, self.top_k)
                for eid in top:
                    if eid in cache:
                        hits += 1
                        cache.move_to_end(eid)
                    else:
                        misses += 1
                        cache[eid] = None
                        if len(cache) > self.k_per_layer:
                            cache.popitem(last=False)
        self.hits += hits
        self.misses += misses
        return {"hits": hits, "misses": misses}

    def hit_rate(self) -> float | None:
        """Overall hit rate, or None before any update."""
        total = self.hits + self.misses
        return (self.hits / total) if total else None

    def resident(self, layer: int) -> list[int]:
        """Current resident expert ids for a layer, LRU->MRU order."""
        cache = self._caches.get(layer)
        return list(cache.keys()) if cache else []

    def clear(self) -> None:
        """Reset all state."""
        self._caches.clear()
        self.hits = 0
        self.misses = 0
