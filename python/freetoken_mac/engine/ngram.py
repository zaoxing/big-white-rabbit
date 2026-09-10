"""Greedy n-gram draft table for speculative decoding (SPEC-speculative-ngram.md).

Phase 1 speculation drafts WITHOUT a draft model: it replays token runs the
request has already produced. ``NgramTable`` is the whole of that memory -- a
bounded map from (order-1)-token contexts to the token that most recently
followed them, plus the walk that turns one lookup into a multi-token draft.

Deliberately dependency-free (no Model, no Context) so the table and its
eviction/draft rules are unit-testable without loading weights. The engine owns
one table per request: sharing across requests would let one request's jargon
steer another's drafts, and a per-request table dies with the request, which is
the only eviction policy that cannot leak.

Capacity is bounded by entry count (``max_entries``), not by bytes: keys are
short int tuples and the bound that matters is lookup-table growth over a long
session, which entry count caps directly.
"""

from __future__ import annotations

from typing import Sequence


class NgramTable:
    """Most-recent-continuation map over (order-1)-token contexts.

    ``order=3`` means: given the last two tokens, predict the next one. A
    lookup that misses returns nothing; a draft walk stops at the first miss,
    so a draft is always a run this request actually emitted before -- never a
    guess. That is what makes greedy verification exact: accepting a draft
    token the target also sampled is indistinguishable from decoding it.
    """

    def __init__(self, order: int = 3, max_entries: int = 8192) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1; got {order}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1; got {max_entries}")
        self.order = order
        self.max_entries = max_entries
        # Insertion-ordered: the oldest entry is next(iter(...)), which is what
        # _evict drops. A re-observed context is deleted and re-inserted so a
        # live pattern is never the eviction victim of its own repeats.
        self._table: dict[tuple[int, ...], int] = {}

    def __len__(self) -> int:
        return len(self._table)

    def clear(self) -> None:
        """Forget everything (request teardown is the usual caller)."""
        self._table.clear()

    def _key(self, context: Sequence[int]) -> tuple[int, ...]:
        # order=1 has an empty context: every token updates the one entry, so
        # predict() always returns the most recently seen token.
        return tuple(context[-(self.order - 1) :]) if self.order > 1 else ()

    def add(self, context: Sequence[int], token: int) -> None:
        """Record that ``token`` followed ``context`` (most recent wins)."""
        key = self._key(context)
        if key in self._table:
            del self._table[key]
        self._table[key] = token
        while len(self._table) > self.max_entries:
            del self._table[next(iter(self._table))]

    def update_stream(self, tokens: Sequence[int]) -> None:
        """Feed a whole run (prompt at admission, accepted tokens after)."""
        for i, token in enumerate(tokens):
            self.add(tokens[max(0, i - self.order + 1) : i], token)

    def predict(self, context: Sequence[int], max_tokens: int) -> list[int]:
        """Walk the table from ``context`` for up to ``max_tokens`` drafts.

        Each draft extends the lookup context, so a 3-token draft is a trigram
        chain the request emitted before, not three independent guesses. Stops
        at the first miss or when ``max_tokens`` is reached; never raises on an
        unknown context -- no drafts is a normal answer, not an error.
        """
        out: list[int] = []
        ctx = self._key(context)
        for _ in range(max(0, max_tokens)):
            nxt = self._table.get(ctx)
            if nxt is None:
                break
            out.append(nxt)
            ctx = self._key((*ctx, nxt))
        return out
