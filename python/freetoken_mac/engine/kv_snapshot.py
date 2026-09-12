"""KV snapshot primitive — in-memory + disk (T12a/b/c).

In-memory uses `memory_seq_cp` / `memory_seq_rm` to snapshot a seq's KV
to a reserved snapshot seq, and restore by copying back. No new C++,
single-seq only (like record_experts), `n_seq_max >= 2` required.
Disk persistence (T12b) saves the stashed prompt/output/n_pos to
`~/.freetoken-metal/kv/` as JSON; on load after restart the KV is rebuilt
via re-prefill (slow but correct). T12c adds true KV serialization via
`llama_state_seq_get_data` / `set_data` to `kv_dir/<name>.bin` for
`O(ms)` restore without re-prefill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Snap:
    prompt: list[int]
    output_tokens: list[int]
    n_pos: int
    n_prefilled: int
    seq_id: int  # snapshot seq holding the KV copy
    params: object | None = None
    next_token: int | None = None
    live: bool = True  # True if KV is live in snapshot_seq (same engine save)


class KVSnapStore:
    """KV snapshots for one MetalEngine — in-memory + disk (T12a/b/c).

    `snapshot_seq` is the reserved seq id (default 1, so engine needs
    `n_seq_max >= 2`). `save` copies `seq 0`'s KV there and stashes
    prompt/output/n_pos; `load` copies back and returns the stashed
    state for the caller to rehydrate a RequestState. Disk files live in
    `~/.freetoken-metal/kv/` as `<name>.json` + `<name>.bin` (T12c binary)
    and survive restarts — `bin` restores via `state_seq_set_data` in
    `O(ms)` without re-prefill, otherwise falls back to re-prefill.
    """

    def __init__(
        self, engine, snapshot_seq: int = 1, kv_dir: str | Path | None = None
    ) -> None:
        self.engine = engine
        self.snapshot_seq = snapshot_seq
        if snapshot_seq >= self.engine.ctx.n_seq_max:
            raise ValueError(
                f"snapshot_seq {snapshot_seq} >= n_seq_max {self.engine.ctx.n_seq_max}; "
                f"need n_seq_max >= 2 for KV snapshots"
            )
        self.kv_dir = Path(kv_dir) if kv_dir else Path.home() / ".freetoken-metal" / "kv"
        self.kv_dir.mkdir(parents=True, exist_ok=True)
        self._snaps: dict[str, _Snap] = {}
        # Reserve snapshot seq: remove from free list so normal admissions
        # never allocate it, and clear any stale KV there.
        if snapshot_seq in self.engine._free_seq_ids:
            self.engine._free_seq_ids.remove(snapshot_seq)
        self.engine.ctx.memory_seq_rm(snapshot_seq, -1, -1)
        # Hydrate in-memory map from disk files already present (survives restart)
        for p in self.kv_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                params = None
                if data.get("params"):
                    try:
                        from .config import RequestParams

                        pd = data["params"]
                        params = RequestParams(
                            temp=pd.get("temp", 0.0),
                            top_k=pd.get("top_k", 40),
                            top_p=pd.get("top_p", 0.95),
                            seed=pd.get("seed", 0xFFFFFFFF),
                            max_tokens=pd.get("max_tokens", 128),
                            stop_at_eog=pd.get("stop_at_eog", True),
                            stop=tuple(pd.get("stop", ())),
                        )
                    except Exception:  # noqa: BLE001 - best-effort disk hydration, ignore malformed params
                        params = None
                self._snaps[p.stem] = _Snap(
                    prompt=data["prompt"],
                    output_tokens=data["output_tokens"],
                    n_pos=data["n_pos"],
                    n_prefilled=data["n_prefilled"],
                    seq_id=self.snapshot_seq,
                    params=params,
                    next_token=data.get("next_token"),
                    live=False,  # disk-hydrated: no live KV in this engine
                )
            except Exception:  # noqa: BLE001 - best-effort disk hydration, ignore malformed snapshot file
                continue

    def save(self, name: str, request_id: int) -> None:
        """Snapshot `request_id`'s KV and prompt state under `name`."""
        req = self.engine._lookup(request_id)
        if req.seq_id == self.snapshot_seq:
            raise ValueError("cannot snapshot the snapshot seq itself")
        # Copy full KV (the snapshot seq is otherwise unused)
        self.engine.ctx.memory_seq_cp(req.seq_id, self.snapshot_seq, -1, -1)
        snap = _Snap(
            prompt=list(req.prompt),
            output_tokens=list(req.output_tokens),
            n_pos=req.n_pos,
            n_prefilled=req.n_prefilled,
            seq_id=self.snapshot_seq,
            params=req.params,
            next_token=req.next_token,
            live=True,
        )
        self._snaps[name] = snap
        # Persist to disk (T12b JSON + T12c binary) — best-effort, no throw on I/O
        try:
            params_dict = None
            if snap.params is not None:
                try:
                    params_dict = {
                        "temp": snap.params.temp,
                        "top_k": snap.params.top_k,
                        "top_p": snap.params.top_p,
                        "seed": snap.params.seed,
                        "max_tokens": snap.params.max_tokens,
                        "stop_at_eog": snap.params.stop_at_eog,
                        "stop": list(snap.params.stop),
                    }
                except Exception:  # noqa: BLE001 - best-effort param serialization, ignore missing attrs
                    params_dict = None
            (self.kv_dir / f"{name}.json").write_text(
                json.dumps(
                    {
                        "prompt": snap.prompt,
                        "output_tokens": snap.output_tokens,
                        "n_pos": snap.n_pos,
                        "n_prefilled": snap.n_prefilled,
                        "next_token": snap.next_token,
                        "params": params_dict,
                    }
                )
            )
        except Exception:  # noqa: BLE001 - best-effort disk persistence, no throw on I/O
            pass
        # T12c: true KV binary — best-effort, ignore failures (fallback to re-prefill)
        try:
            # Prefer the live request's seq (has full KV up to n_pos); snapshot_seq is identical after cp
            data = self.engine.ctx.state_seq_get_data(req.seq_id)  # type: ignore[attr-defined]
            if data:
                (self.kv_dir / f"{name}.bin").write_bytes(bytes(data))
        except Exception:  # noqa: BLE001 - best-effort KV serialization, fallback to re-prefill
            pass

    def load(self, name: str, new_request_id: int | None = None) -> int:
        """Restore snapshot `name` into a fresh request id.

        Copies snapshot seq's KV back to a free seq, rehydrates a
        RequestState, and returns the new request_id. Caller must have a
        free slot; raises SeqIdExhausted otherwise (same as add_request).
        If the snapshot was loaded from disk after restart (no live KV),
        falls back to re-prefill via `add_request` (slow but correct).
        """
        snap = self._snaps.get(name)
        if snap is None:
            # Try disk (survives restart)
            p = self.kv_dir / f"{name}.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    params = None
                    if data.get("params"):
                        try:
                            from .config import RequestParams

                            pd = data["params"]
                            params = RequestParams(
                                temp=pd.get("temp", 0.0),
                                top_k=pd.get("top_k", 40),
                                top_p=pd.get("top_p", 0.95),
                                seed=pd.get("seed", 0xFFFFFFFF),
                                max_tokens=pd.get("max_tokens", 128),
                                stop_at_eog=pd.get("stop_at_eog", True),
                                stop=tuple(pd.get("stop", ())),
                            )
                        except Exception:  # noqa: BLE001 - best-effort disk hydration, ignore malformed params
                            params = None
                    snap = _Snap(
                        prompt=data["prompt"],
                        output_tokens=data["output_tokens"],
                        n_pos=data["n_pos"],
                        n_prefilled=data["n_prefilled"],
                        seq_id=self.snapshot_seq,
                        params=params,
                        next_token=data.get("next_token"),
                        live=False,
                    )
                    self._snaps[name] = snap
                except Exception as exc:  # noqa: BLE001 - disk read failure, re-raised as KeyError
                    raise KeyError(f"unknown snapshot {name!r}") from exc
            else:
                raise KeyError(f"unknown snapshot {name!r}")
        # If snapshot has no live KV (disk-hydrated after restart, or params None),
        # try T12c fast path: restore binary KV via state_seq_set_data if .bin exists.
        # Falls back to re-prefill (slow but correct) when bin missing or incompatible.
        if not getattr(snap, "live", False) or snap.params is None:
            bin_path = self.kv_dir / f"{name}.bin"
            if bin_path.exists():
                try:
                    # Need a free seq for the restored KV
                    if not self.engine._free_seq_ids:
                        self.engine._evict_pin(exclude=None)
                    if self.engine._free_seq_ids:
                        seq_id = self.engine._free_seq_ids.pop(0)
                        data = bin_path.read_bytes()
                        # state_seq_set_data returns bytes consumed (0 on failure)
                        consumed = self.engine.ctx.state_seq_set_data(seq_id, data)  # type: ignore[attr-defined]
                        if consumed and consumed > 0:
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
                            self.engine._states[rid] = state
                            self.engine.ctx.set_seq_sampler(seq_id, snap.params.to_sampler_params())  # type: ignore[union-attr]
                            return rid
                        # set_data failed — return seq to free list for fallback
                        self.engine._free_seq_ids.insert(0, seq_id)
                except Exception:  # noqa: BLE001 - best-effort binary restore, fallback to re-prefill
                    pass
            # Fallback: re-prefill via normal admission (slow but correct)
            rid = self.engine.add_request(
                snap.prompt,  # type: ignore[arg-type] — already tokenized
                snap.params,  # type: ignore[arg-type] — preserves max_tokens etc.
                add_special=False,
            )
            return rid
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
        # Merge in-memory and on-disk (on-disk may have entries not yet hydrated)
        on_disk = {p.stem for p in self.kv_dir.glob("*.json")}
        # Also include bin-only snapshots (e.g. after manual delete of json)
        on_disk |= {p.stem for p in self.kv_dir.glob("*.bin")}
        return sorted(set(self._snaps.keys()) | on_disk)

    def delete(self, name: str) -> None:
        self._snaps.pop(name, None)
        try:
            (self.kv_dir / f"{name}.json").unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - best-effort delete, ignore filesystem errors
            pass
        try:
            (self.kv_dir / f"{name}.bin").unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - best-effort delete, ignore filesystem errors
            pass

    def clear(self) -> None:
        self._snaps.clear()
        for p in self.kv_dir.glob("*.json"):
            try:
                p.unlink()
            except Exception:  # noqa: BLE001 - best-effort clear, ignore filesystem errors
                continue
        for p in self.kv_dir.glob("*.bin"):
            try:
                p.unlink()
            except Exception:  # noqa: BLE001 - best-effort clear, ignore filesystem errors
                continue
