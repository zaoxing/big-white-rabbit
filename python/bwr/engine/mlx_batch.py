"""Continuous batching for the MLX backend.

What it buys, measured before building it
=========================================

On 27B/M1 Max the trunk forward costs 63.4 ms at 1 row and 450.0 ms at 8, so
eight sequences in one step produce eight tokens for 7.1x the time of one:
**~1.13x aggregate throughput**. MLX's quantized kernel does not use the
simdgroup matrix units that make fp16 matmul flat across rows (see
SPEC-mlx-mtp-draft.md), so batching cannot pay what it pays on a backend with
a proper batched GEMM.

The reason to have it anyway is latency, not throughput: without batching the
eighth concurrent request waits for seven whole generations to finish. With
it, all eight advance together from their first step. That is the difference
between a server that feels stuck and one that feels slow.

Design
======

mlx-lm's cache classes already carry the primitives, so this coordinates them
rather than reimplementing them:

- `merge([per_seq_caches])` stacks single-sequence caches into a batched one,
  setting `left_padding` so sequences of different prompt lengths align. This
  is the JOIN.
- `filter(indices)` drops rows in place. This is the LEAVE.

A joining request is prefilled ALONE and then merged. Prefills are ragged by
nature and batching them would need padding plus a mask on the prefill path;
merging a finished prefill keeps that complexity out of the hot loop, at the
cost of one solo forward per admission.

Parity is the bar: two prompts decoded together produce exactly what each
produces alone (verified on qwen3_5 before this module was written, and
pinned by tests). A batch whose output depends on who else is in flight would
be worse than no batching.
"""

from __future__ import annotations

from typing import Any, Sequence


class MLXBatch:
    """One batched cache plus the row order that maps to request ids."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.caches: list | None = None
        self.rows: list[int] = []          # request_id per batch row

    # -- inspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def empty(self) -> bool:
        return not self.rows

    def index_of(self, request_id: int) -> int | None:
        try:
            return self.rows.index(request_id)
        except ValueError:
            return None

    # -- membership --------------------------------------------------------

    def join(self, request_id: int, cache: list) -> None:
        """Add one prefilled single-sequence cache as a new row.

        `cache` must be a freshly prefilled cache for exactly this request;
        it is consumed (merged) and must not be reused by the caller.
        """
        if request_id in self.rows:
            raise ValueError(f"request {request_id} is already batched")
        if self.caches is None:
            # First member: still merge, so a batch of one has the same
            # shapes (and left_padding) as a batch of many. Special-casing
            # B==1 would make the one-member path structurally different
            # from every other and hide bugs until a second request showed up.
            self.caches = self._merge([cache])
            self.rows = [request_id]
            return
        merged = []
        for layer, existing in enumerate(self.caches):
            single = type(cache[layer]).merge([cache[layer]])  # -> batch of 1
            merged.append(_concat_batched(existing, single))
        self.caches = merged
        self.rows.append(request_id)

    def leave(self, request_ids: Sequence[int]) -> None:
        """Drop rows for finished/cancelled requests."""
        drop = {r for r in request_ids if r in self.rows}
        if not drop:
            return
        keep = [i for i, rid in enumerate(self.rows) if rid not in drop]
        self.rows = [rid for rid in self.rows if rid not in drop]
        if not keep:
            self.caches = None
            return
        import mlx.core as mx

        idx = mx.array(keep)
        for c in self.caches or []:
            c.filter(idx)

    def _merge(self, caches: list[list]) -> list:
        out = []
        for layer in range(len(caches[0])):
            per_layer = [c[layer] for c in caches]
            out.append(type(per_layer[0]).merge(per_layer))
        return out

    # -- decode ------------------------------------------------------------

    def step(self, tokens: Sequence[int]) -> Any:
        """One batched decode. `tokens` is one token per row, in row order.

        Returns logits shaped (B, 1, vocab); the caller samples per row so
        each request keeps its own sampling parameters.
        """
        import mlx.core as mx

        if self.caches is None or not self.rows:
            raise RuntimeError("step() on an empty batch")
        if len(tokens) != len(self.rows):
            raise ValueError(
                f"got {len(tokens)} tokens for {len(self.rows)} rows; "
                "the caller must supply one token per row, in row order"
            )
        ids = mx.array([[int(t)] for t in tokens])
        logits = self._model(ids, cache=self.caches)
        mx.eval(logits)
        return logits


def _concat_batched(a: Any, b: Any) -> Any:
    """Join two ALREADY-batched caches along the batch axis.

    mlx-lm's own `merge` only accepts single-sequence caches -- it reads
    `c.offset` as an int, which is an array once a cache is batched. So a
    mid-flight join (batch of N + the newcomer) has to be done here.

    Two shapes to handle:

    - `ArraysCache` holds fixed-size recurrent state with no sequence axis,
      so rows concatenate directly.
    - `BatchKVCache` holds (B, H, T, D) with per-row `left_padding`. Rows of
      different length are right-aligned, so joining re-pads every row to the
      new maximum rather than assuming the incumbent is longest -- a joiner
      with a longer prompt than anything in flight is entirely normal.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    if isinstance(a, ArraysCache):
        out = type(a)(len(a.cache))
        out.cache = [
            None if x is None else mx.concatenate([x, y], axis=0)
            for x, y in zip(a.cache, b.cache)
        ]
        if a.left_padding is not None and b.left_padding is not None:
            out.left_padding = mx.concatenate([a.left_padding, b.left_padding])
        return out

    ka, va, oa, pa = a.state
    kb, vb, ob, pb = b.state
    # For a BATCHED cache `offset` is ALREADY the per-row content length
    # (the constructor seeds it to -left_padding and merge adds T back), so
    # content length is offset, NOT offset - left_padding. Subtracting the
    # padding a second time is the easy mistake and it silently corrupts
    # only the SHORTER rows -- the longest row has zero padding and keeps
    # looking correct, which is exactly how it hides.
    lens = [int(x) for x in oa.tolist()] + [int(x) for x in ob.tolist()]
    max_len = max(lens) if lens else 0
    B = ka.shape[0] + kb.shape[0]
    H, Dk = ka.shape[1], ka.shape[3]
    Dv = va.shape[3]
    keys = mx.zeros((B, H, max_len, Dk), dtype=ka.dtype)
    values = mx.zeros((B, H, max_len, Dv), dtype=va.dtype)

    row = 0
    for src_k, src_v, src_o, src_p in ((ka, va, oa, pa), (kb, vb, ob, pb)):
        for i in range(src_k.shape[0]):
            n, pad = int(src_o[i]), int(src_p[i])
            if n > 0:
                # source content lives at [pad, pad + n); destination is
                # right-aligned in the new width.
                dst = max_len - n
                keys[row : row + 1, :, dst:max_len] = src_k[i : i + 1, :, pad : pad + n]
                values[row : row + 1, :, dst:max_len] = src_v[i : i + 1, :, pad : pad + n]
            row += 1

    # Finish exactly as mlx-lm's own merge does. The constructor seeds
    # `offset = -left_padding`, and `offset += T` then yields each row's
    # CONTENT length -- not T. Setting offset = T directly (the obvious
    # guess) over-reports every padded row and corrupts the shorter
    # sequences while the longest one still looks correct.
    out = type(a)([max_len - n for n in lens])
    out.keys = keys
    out.values = values
    out.offset += keys.shape[2]
    out._idx = keys.shape[2]
    return out
