"""KV snapshot primitive — in-memory (T12a).

Uses existing `memory_seq_cp` / `memory_seq_rm` to snapshot a seq's KV
to a reserved snapshot seq, and restore by copying back. No new C++,
single-seq only (like record_experts), `n_seq_max >= 2` required so a
snapshot seq exists. `T12b` will add disk persistence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Snap:
    prompt: list[int]
    output_tokens: list[int]
    n_pos: int
    n_prefilled: int
    seq_id: int  # snapshot seq holding the KV copy
    params: object | None = None
    next_token: int | None = None


class KVSnapStore:
    """In-memory KV snapshots for one MetalEngine.

    `snapshot_seq` is the reserved seq id (default 1, so engine needs
    `n_seq_max >= 2`). `save` copies `seq 0`'s KV there and stashes
    prompt/output/n_pos; `load` copies back and returns the stashed
    state for the caller to rehydrate a RequestState.
    """

    def __init__(self, engine, snapshot_seq: int = 1) -> None:
        self.engine = engine
        self.snapshot_seq = snapshot_seq
        if snapshot_seq >= self.engine.ctx.n_seq_max:
            raise ValueError(
                f"snapshot_seq {snapshot_seq} >= n_seq_max {self.engine.ctx.n_seq_max}; "
                f"need n_seq_max >= 2 for KV snapshots"
            )
        self._snaps: dict[str, _Snap] = {}
        # Reserve snapshot seq: remove from free list so normal admissions
        # never allocate it, and clear any stale KV there.
        if snapshot_seq in self.engine._free_seq_ids:
            self.engine._free_seq_ids.remove(snapshot_seq)
        self.engine.ctx.memory_seq_rm(snapshot_seq, -1, -1)

    def save(self, name: str, request_id: int) -> None:
        """Snapshot `request_id`'s KV and prompt state under `name`."""
        req = self.engine._lookup(request_id)
        if req.seq_id == self.snapshot_seq:
            raise ValueError("cannot snapshot the snapshot seq itself")
        # Copy full KV (the snapshot seq is otherwise unused)
        self.engine.ctx.memory_seq_cp(req.seq_id, self.snapshot_seq, -1, -1)
        self._snaps[name] = _Snap(
            prompt=list(req.prompt),
            output_tokens=list(req.output_tokens),
            n_pos=req.n_pos,
            n_prefilled=req.n_prefilled,
            seq_id=self.snapshot_seq,
            params=req.params,
            next_token=req.next_token,
        )

    def load(self, name: str, new_request_id: int | None = None) -> int:
        """Restore snapshot `name` into a fresh request id.

        Copies snapshot seq's KV back to a free seq, rehydrates a
        RequestState, and returns the new request_id. Caller must have a
        free slot; raises SeqIdExhausted otherwise (same as add_request).
        """
        snap = self._snaps.get(name)
        if snap is None:
            raise KeyError(f"unknown snapshot {name!r}")
        # Need a free seq — reuse engine's free list (snapshot seq itself is
        # not in the free list, so this is a normal admission slot).
        if not self.engine._free_seq_ids:
            # Try to evict a pin first, like add_request does
            self.engine._evict_pin(exclude=None)
        if not self.engine._free_seq_ids:
            from .metal_engine import SeqIdExhausted

            raise SeqIdExhausted("no free seq for snapshot restore")
        seq_id = self.engine._free_seq_ids.pop(0)
        # Copy snapshot KV to the new seq
        self.engine.ctx.memory_seq_cp(self.snapshot_seq, seq_id, -1, -1)
        # Rehydrate RequestState (reuse Engine's bookkeeping but bypass
        # tokenization / prompt-length checks — snapshot is already validated)
        from .batching import RequestState

        rid = new_request_id if new_request_id is not None else self.engine._next_request_id
        if new_request_id is None:
            self.engine._next_request_id += 1
        else:
            self.engine._next_request_id = max(self.engine._next_request_id, rid + 1)
        state = RequestState(
            request_id=rid,
            seq_id=seq_id,
            prompt=list(snap.prompt),
            params=snap.params,  # type: ignore[arg-type]
        )
        state.output_tokens = list(snap.output_tokens)
        state.n_pos = snap.n_pos
        state.n_prefilled = snap.n_prefilled
        state.n_generated = len(snap.output_tokens)
        state.next_token = snap.next_token
        # n_pos already includes prompt + output, so next decode continues
        self.engine._states[rid] = state
        # Re-install sampler
        self.engine.ctx.set_seq_sampler(seq_id, snap.params.to_sampler_params())  # type: ignore[union-attr]
        return rid

    def list(self) -> list[str]:
        return list(self._snaps.keys())

    def delete(self, name: str) -> None:
        self._snaps.pop(name, None)

    def clear(self) -> None:
        self._snaps.clear()
