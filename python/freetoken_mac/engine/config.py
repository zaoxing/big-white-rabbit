"""Engine and per-request configuration, plus the shared stop-sequence rule.

``EngineConfig`` shapes the ``llama_context``; ``RequestParams`` carries one
request's sampling settings and stop conditions. Sampling is per request because
each sequence owns its own sampler chain in the C++ context.

``StopSequenceFilter`` and its helpers live here rather than under ``server/``
because BOTH layers need the identical rule and the dependency only runs one way.
The engine has to detect a stop sequence to stop *decoding* (that is the whole
point: an agent loop should not pay for tokens past its delimiter), and the
server has to detect the same one at the same byte to truncate what it puts on
the wire and to name the match. Two implementations of "where does this text
first contain one of these strings" would be free to disagree, and the failure
mode of disagreeing is a response whose text does not match the reason given for
ending it. ``server/`` may import from ``engine/``; the reverse would be a cycle,
so the shared rule belongs on this side of the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .._freetoken_metal import ContextParams, SamplerParams


@dataclass
class EngineConfig:
    """Requested context geometry. llama.cpp rounds n_ctx UP and clamps n_batch DOWN
    to n_ctx, so these are a request, not a promise; budget against ``ctx.n_batch``."""

    n_ctx: int = 4096
    n_batch: int = 512
    n_ubatch: int = 512
    # Hard ceiling on concurrency: the seq_id pool is exactly this wide, and llama.cpp
    # rejects any batch referencing seq_id >= n_seq_max.
    n_seq_max: int = 8
    n_threads: int = 0
    n_threads_batch: int = 0
    flash_attn: bool = True
    # One shared KV buffer instead of one stream per sequence. Off = llama.cpp's own
    # default; on is what a partial-range ctx.memory_seq_cp (prefix fork) needs.
    kv_unified: bool = False
    # Greedy n-gram speculative decoding (SPEC-speculative-ngram.md). Off by
    # default: drafts cost batch rows, which only pays when the acceptance
    # rate is high. Greedy requests only; non-greedy always takes the plain
    # path (verification without logits cannot beat greedy).
    speculative: bool = False
    spec_max_drafts: int = 4
    # Draft-model speculation (SPEC-draft-model.md): max drafts per step from
    # the draft model. The draft Model object itself is passed to MetalEngine
    # (not a path here: the caller owns loading). Mutually exclusive with
    # `speculative`; MetalEngine raises if both are set.
    draft_max_drafts: int = 4

    def to_context_params(self) -> ContextParams:
        cp = ContextParams()
        cp.n_ctx = self.n_ctx
        cp.n_batch = self.n_batch
        cp.n_ubatch = self.n_ubatch
        cp.n_seq_max = self.n_seq_max
        cp.n_threads = self.n_threads
        cp.n_threads_batch = self.n_threads_batch
        cp.flash_attn = self.flash_attn
        cp.kv_unified = self.kv_unified
        return cp


@dataclass
class RequestParams:
    """One request's sampling params and stop conditions (temp <= 0 -> greedy)."""

    temp: float = 0.0
    top_k: int = 40
    top_p: float = 0.95
    seed: int = 0xFFFFFFFF  # LLAMA_DEFAULT_SEED
    max_tokens: int = 128
    stop_at_eog: bool = True
    # Text sequences that end generation (OpenAI's `stop`, Anthropic's
    # `stop_sequences`). Empty -- the default -- means the request behaves exactly as it
    # did before stop sequences existed: nothing is scanned and nothing is withheld.
    # A tuple, not a list, because RequestParams is shared across the engine thread and
    # the event loop and this field is read on every sampled token.
    stop: tuple[str, ...] = ()

    def to_sampler_params(self) -> SamplerParams:
        sp = SamplerParams()
        sp.temp = self.temp
        sp.top_k = self.top_k
        sp.top_p = self.top_p
        sp.seed = self.seed
        return sp


# --- stop sequences -----------------------------------------------------------------


@dataclass(frozen=True)
class StopMatch:
    """Which stop sequence matched, and where in the text it sits.

    ``start`` is the truncation point: the client asked to stop AT the delimiter, so the
    delimiter itself and everything after it are not part of the turn.
    """

    text: str
    start: int
    end: int


def normalize_stops(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """A client's ``stop`` / ``stop_sequences`` as a tuple of usable sequences.

    OpenAI accepts a bare string as well as a list of them and means the same thing by
    both, so both spellings normalise here instead of at each call site.

    An empty string is DROPPED rather than honoured. It occurs at offset 0 of every
    text, so honouring it would truncate every response to nothing -- there is no
    reading of "stop at the empty string" that a client can have meant, and OpenAI
    rejects it outright. Duplicates collapse (a repeat cannot change where generation
    ends) and order is otherwise preserved, so the reported match is stable.
    """
    if value is None:
        return ()
    items: Sequence[object] = (value,) if isinstance(value, str) else tuple(value)
    out: list[str] = []
    for item in items:
        if isinstance(item, str) and item and item not in out:
            out.append(item)
    return tuple(out)


def partial_stop_len(buf: str, stops: Iterable[str]) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of some stop.

    That suffix is exactly the text that cannot be released yet: it may still turn into
    a stop sequence on the next token, and text already handed to a client cannot be
    taken back. Zero when nothing is pending, which is the common case.
    """
    best = 0
    for token in stops:
        limit = min(len(buf), len(token) - 1)
        for k in range(limit, best, -1):
            if buf.endswith(token[:k]):
                best = k
                break
    return best


def first_stop_match(text: str, stops: Iterable[str]) -> StopMatch | None:
    """The stop sequence that COMPLETES earliest in ``text``, or None for no match.

    Earliest by end offset, not by start offset, because generation ends at the token
    that completes a sequence: with ``stop=["ab", "abc"]`` against "xabc" the answer is
    "ab", since at the character that finished it "abc" was still unwritten and the
    model would never have been asked for the "c". A tie on the end offset is broken by
    the earliest start, i.e. by the longer match, so the truncation keeps the least text
    that either reading allows.
    """
    best: StopMatch | None = None
    for token in stops:
        i = text.find(token)
        if i < 0:
            continue
        candidate = StopMatch(token, i, i + len(token))
        if best is None or (candidate.end, candidate.start) < (best.end, best.start):
            best = candidate
    return best


class StopSequenceFilter:
    """Incremental stop-sequence detector over a stream of generated text pieces.

    Matching is on the ACCUMULATED text, never on one piece: the tokeniser does not
    align to a stop sequence. "4" arrives inside a piece like "\\n4", and a delimiter
    like "\\nObservation:" spans several pieces, so a per-piece test would miss exactly
    the case an agent loop depends on.

    ``push`` returns ``(emittable_text, stopped)``. *Emittable* means the text can no
    longer become part of a stop sequence; a trailing run that is still a proper prefix
    of one is WITHHELD, and released by ``flush`` if generation ends without completing
    the match. That is the same tactic ``ToolCallStreamParser`` uses for a partial
    ``<tool_call>`` tag, and for the same reason: a byte already sent to a client cannot
    be recalled, so the decision has to be deferred rather than corrected.

    Once a sequence matches, ``push`` emits nothing further -- everything the model
    produced after the delimiter belongs to no one -- and ``matched`` names the sequence
    (which is what Anthropic's ``stop_sequence`` response field carries).

    With no stop sequences this is a pass-through that withholds nothing and buffers
    nothing, so a request that did not ask for one is byte-identical to a server without
    the feature.
    """

    def __init__(self, stops: Sequence[str] = ()) -> None:
        # Through `normalize_stops`, not `tuple(stops)`: a bare string is ONE sequence,
        # and `tuple("abc")` is `('a', 'b', 'c')` -- a per-character stop set that ends
        # the turn at the first "a". The routes normalise before they get here (and the
        # function is idempotent), so this only closes the door on a direct engine
        # caller, which is exactly the caller with no other line of defence.
        self._stops = normalize_stops(stops)
        self._buffer = ""
        self._match: StopMatch | None = None

    @property
    def enabled(self) -> bool:
        """False when there are no stop sequences, i.e. when this is the pass-through."""
        return bool(self._stops)

    @property
    def stopped(self) -> bool:
        return self._match is not None

    @property
    def match(self) -> StopMatch | None:
        return self._match

    @property
    def matched(self) -> str | None:
        """The stop sequence that ended generation, or None if none has."""
        return self._match.text if self._match is not None else None

    def push(self, new_text: str) -> tuple[str, bool]:
        """Feed the next piece; return the text safe to emit and whether it stopped."""
        if not self._stops:
            return new_text, False
        if self._match is not None:
            # Already past the delimiter. Callers stop pulling on the first True, but a
            # canned or already-buffered stream can still deliver more.
            return "", False
        self._buffer += new_text
        match = first_stop_match(self._buffer, self._stops)
        if match is not None:
            head = self._buffer[: match.start]
            self._buffer = ""
            self._match = match
            return head, True
        hold = partial_stop_len(self._buffer, self._stops)
        safe = self._buffer[: len(self._buffer) - hold] if hold else self._buffer
        self._buffer = self._buffer[len(safe) :]
        return safe, False

    def flush(self) -> str:
        """End of generation: release text withheld for a match that never completed."""
        if self._match is not None:
            self._buffer = ""
            return ""
        out, self._buffer = self._buffer, ""
        return out
