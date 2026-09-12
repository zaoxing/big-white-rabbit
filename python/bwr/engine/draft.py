"""Draft-model speculation (SPEC-draft-model.md).

A small model proposes, the target verifies with the SAME machinery as n-gram
speculation (`_verify_rows` + rewind). Verification makes output identical
regardless of draft quality -- a bad draft only costs time, never correctness.
(Vocab mismatch included: foreign drafts just reject. Only the acceptance
rate, never the output, depends on the pairing.)

The draft context mirrors the target token stream at IDENTICAL positions: the
prompt, then every target-accepted token. It never advances speculatively
past confirmed tokens except inside `propose`, and it never partially rewinds
except in `sync` (which is why it carries rollback snapshots like the target
when the draft model is a hybrid).

Per-step call order (all on the worker thread, the single owner of both
contexts): `propose` (draft decodes now) -> target packs + decodes ->
`_verify` -> `sync` (exactly the newly-confirmed tail). `propose` leaves
pending logits behind; `sync` leaves pending logits behind; `prepare` leaves
pending logits behind. That pending-logits invariant is what makes each call
valid -- never call `propose` twice without an intervening `sync`, and never
`sync` a request that just retired (release it instead).
"""

from __future__ import annotations

from typing import Sequence

from .._bwr_metal import Batch, Context, ContextParams, Model, SamplerParams


class DraftEngine:
    """A small model's propose/sync/release loop behind target verification.

    Greedy chains only: drafts feed greedy verification, and a greedy chain is
    stateless enough that unsampled prompt decodes in `prepare` need no replay.
    (A sampling draft would need its chain replayed over the prompt -- out of
    scope for Phase 1, same as non-greedy targets.)
    """

    def __init__(
        self,
        model: Model,
        *,
        n_ctx: int,
        n_seq_max: int,
        n_batch: int = 512,
        n_snapshots: int = 5,
    ) -> None:
        if n_snapshots < 0:
            raise ValueError(f"n_snapshots must be >= 0; got {n_snapshots}")
        self.model = model
        cp = ContextParams()
        cp.n_ctx = n_ctx
        cp.n_batch = n_batch
        cp.n_seq_max = n_seq_max
        cp.n_rs_seq = n_snapshots
        self.ctx = Context(model, cp, SamplerParams())
        self._batch = Batch(self.ctx.n_batch, 1)
        self._seq: dict[int, int] = {}
        self._pos: dict[int, int] = {}
        self._pending_row: dict[int, int] = {}

    def _lookup(self, request_id: int) -> int:
        try:
            return self._seq[request_id]
        except KeyError:
            raise KeyError(f"unknown draft request {request_id}") from None

    def prepare(
        self, request_id: int, seq_id: int, prompt: Sequence[int]
    ) -> None:
        """Prefill the prompt onto a cleared slot; leaves pending logits."""
        tokens = list(prompt)
        if not tokens:
            raise ValueError("draft prompt tokenized to zero tokens")
        # A retired predecessor may have left KV behind on this slot.
        self.ctx.memory_seq_rm(seq_id, -1, -1)
        self.ctx.set_seq_sampler(seq_id, SamplerParams())
        pos = 0
        pending = -1
        total = len(tokens)
        for i in range(0, total, self.ctx.n_batch):
            chunk = tokens[i : i + self.ctx.n_batch]
            self._batch.clear()
            for j, tok in enumerate(chunk):
                want = i + j == total - 1
                pending = self._batch.add(tok, pos + j, seq_id, want)
            self.ctx.decode(self._batch)
            pos += len(chunk)
        self._seq[request_id] = seq_id
        self._pos[request_id] = pos
        self._pending_row[request_id] = pending

    def propose(self, request_id: int, max_drafts: int) -> list[int]:
        """Autoregress up to ``max_drafts`` continuations; leaves pending logits."""
        seq_id = self._lookup(request_id)
        pos = self._pos[request_id]
        row = self._pending_row[request_id]
        out: list[int] = []
        for _ in range(max(0, max_drafts)):
            tok = self.ctx.sample_seq(seq_id, row)
            out.append(tok)
            self._batch.clear()
            row = self._batch.add(tok, pos, seq_id, True)
            self.ctx.decode(self._batch)
            pos += 1
        self._pos[request_id] = pos
        self._pending_row[request_id] = row
        return out

    def sync(
        self, request_id: int, confirmed_pos: int, tail: Sequence[int]
    ) -> None:
        """Confirm ``tail`` at ``confirmed_pos`` (decoding it) after dropping
        the rejected drafts at/after that cell. Always called with the newly
        confirmed tail only -- matched drafts already sit at the right cells,
        so re-decoding them would desync positions (llama requires strictly
        consecutive writes). Leaves pending logits."""
        seq_id = self._lookup(request_id)
        tail_tokens = list(tail)
        if not tail_tokens:
            return
        if not self.ctx.memory_seq_rm(seq_id, confirmed_pos, -1):
            raise RuntimeError(
                f"draft rewind failed for request {request_id}: partial KV "
                f"removal of [{confirmed_pos}, inf) unsupported (draft model "
                f"needs rollback snapshots?)"
            )
        self._batch.clear()
        row = -1
        for k, tok in enumerate(tail_tokens):
            row = self._batch.add(
                tok, confirmed_pos + k, seq_id, k == len(tail_tokens) - 1
            )
        self.ctx.decode(self._batch)
        self._pos[request_id] = confirmed_pos + len(tail_tokens)
        self._pending_row[request_id] = row

    def release(self, request_id: int) -> None:
        """Drop draft state; tolerant of unknown ids (mirrors `_retire`)."""
        seq_id = self._seq.pop(request_id, None)
        self._pos.pop(request_id, None)
        self._pending_row.pop(request_id, None)
        if seq_id is None:
            return
        self.ctx.memory_seq_rm(seq_id, -1, -1)
        self.ctx.reset_seq_sampler(seq_id)

    def close(self) -> None:
        """Release the draft context now (same Metal-teardown abort class as
        the main context -- see Context::close). Idempotent."""
        self.ctx.close()
