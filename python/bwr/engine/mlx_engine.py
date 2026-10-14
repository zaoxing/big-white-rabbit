"""MLXEngine: mlx-lm behind the MetalEngine interface (SPEC-mlx-engine.md).

Every request owns a cache and runs the manual loop (`_start_spec` then
`_verify` per step); the worker thread drives them one step at a time, and
each step queues the next forward before forcing the current token (see
`_lookahead`). Sampling is per request: greedy by default, mlx-lm's sampler
for temp > 0. Drafting is greedy-only and comes from either the n-gram table
(``NgramTable``, reused verbatim) or the model's MTP head; the engine-wide
`spec_gate` decides first whether a step drafts at all, by timing drafting
and plain steps on this engine, and a per-request rolling acceptance then
disables drafting below 0.5 after 8 drafted. A mismatch rewinds the cache
through a snapshot taken before the verify forward (`_snapshot_cache`) and
re-feeds only the accepted prefix, so output is identical to plain decode.

mlx-lm is a core dependency (it ships with big-white-rabbit). Nothing in this
module imports it at top level anyway: the import happens in ``__init__``
so a broken install fails with a clear error naming the backend instead of
at ``import bwr`` time.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any

from .batching import RequestState
from .config import EngineConfig, RequestParams, StopSequenceFilter
from .detok import TokenDetokenizer
from .metal_engine import MIN_RETAINED_FINISHED, StepOutput
from .ngram import NgramTable

logger = logging.getLogger(__name__)


def _require_mlx():
    """Import mlx-lm lazily, with a clear error naming the backend."""
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise ImportError(
            "the mlx backend needs mlx/mlx-lm installed "
            "(pip install big-white-rabbit)"
        ) from exc
    return load, make_sampler


def _mx():
    """mlx.core, lazily (same no-top-level-import rule as above)."""
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ImportError(
            "the mlx backend needs mlx/mlx-lm installed "
            "(pip install big-white-rabbit)"
        ) from exc
    return mx


# Prefill chunking mirrors mlx-lm's own default: bound transient activation
# memory no matter how long the prompt is.
_PREFILL_CHUNK = 512
# Auto-fallback: stop drafting for a request past this many drafted tokens
# or this many recomputes when its rolling acceptance is below the rate.
# Bounds hostile-text cost to a few full re-prefills; repetitive text never
# trips it. The recompute clause catches sparse-draft prose that never packs
# enough drafts to reach the count clause.
_SPEC_FALLBACK_MIN_DRAFTS = 8
_SPEC_FALLBACK_MIN_RECOMPUTES = 3
_SPEC_FALLBACK_MIN_RATE = 0.5


class _MLXContext:
    """Teardown handle satisfying AsyncEngine.stop's ``engine.ctx.close()``,
    plus the read-only geometry `/health` reports.

    Releasing weights on Apple Silicon is load-bearing (same abort class as
    Context::close documents): drop every reference and clear the Metal-side
    caches instead of waiting for the collector.

    Geometry semantics differ from split-KV Metal by design: every request
    gets the whole window (no pre-partition), so ``n_ctx_seq == n_ctx`` and
    ``n_seq_max`` is the configured concurrency shape, not a slot count.
    ``decode_calls`` counts evaluated tokens (one forward pass each), the
    closest analog to llama.cpp's counter.
    """

    def __init__(self, engine: MLXEngine) -> None:
        self._engine = engine
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        eng = self._engine
        eng._states.clear()
        eng._retired.clear()
        eng._caches.clear()
        eng._spec_tables.clear()
        eng._spec_ok.clear()
        eng._spec_stats.clear()
        eng._pending.clear()
        eng._batch_pending.clear()
        eng._cache_tokens.clear()
        eng._mtp_caches.clear()
        eng._mtp_hidden.clear()
        eng._mtp = None
        eng._batch = None
        eng._prefix.clear()
        eng._model = None
        eng._tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def n_ctx(self) -> int:
        return self._engine.config.n_ctx

    @property
    def n_ctx_seq(self) -> int:
        return self._engine.config.n_ctx

    @property
    def n_seq_max(self) -> int:
        return self._engine.config.n_seq_max

    @property
    def decode_calls(self) -> int:
        return self._engine._decode_calls


class MLXEngine:
    """One owned cache per request, stepped in lockstep on the manual loop.

    Greedy by default like everywhere else here; temp/top_p/top_k map onto
    mlx-lm's sampler. NOTE on temp > 0: mlx-lm samplers take no seed, so
    sampled runs are unseeded (MetalEngine seeds its chains). Greedy runs are
    exactly deterministic.
    """

    def __init__(
        self,
        model_path: str,
        config: EngineConfig | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        # Knob validation BEFORE load(): a bad value fails in milliseconds
        # with an error naming it, not after pulling in the weights (or,
        # for mlx_kv_bits, at the first request's cache build).
        if self.config.spec_max_drafts < 0:
            raise ValueError(
                f"spec_max_drafts must be >= 0; got {self.config.spec_max_drafts}"
            )
        if self.config.prefix_cache_min_tokens < 1:
            raise ValueError(
                f"prefix_cache_min_tokens must be >= 1; got "
                f"{self.config.prefix_cache_min_tokens}"
            )
        if self.config.mlx_kv_bits not in (None, 0, 4, 8):
            raise ValueError(
                f"mlx_kv_bits must be 0, 4 or 8; got {self.config.mlx_kv_bits}"
            )
        # Effective KV width. Normally the configured value; batching forces
        # f16 (see below), so the engine tracks its own rather than mutating
        # the caller's EngineConfig, which may be shared across engines.
        self._kv_bits = self.config.mlx_kv_bits
        if self.config.mlx_batch:
            if self.config.mlx_kv_bits not in (None, 0):
                # mlx-lm ships BatchKVCache and BatchRotatingKVCache but no
                # batched QuantizedKVCache, and QuantizedKVCache has no
                # merge(), so a batched join would AttributeError on the
                # first quantized layer. Quantized KV buys KV HEADROOM, not
                # speed (~4% slower decode, measured), so dropping to f16
                # costs memory rather than throughput -- worth a warning, not
                # a refusal.
                logger.warning(
                    "mlx_batch: ignoring mlx_kv_bits=%s and using f16 KV; "
                    "mlx-lm has no batched QuantizedKVCache. This costs KV "
                    "memory (2-4x more per sequence), not decode speed.",
                    self.config.mlx_kv_bits,
                )
                self._kv_bits = None
            clashes = [
                n for n in ("speculative", "mlx_mtp")
                if getattr(self.config, n, False)
            ]
            if clashes:
                raise ValueError(
                    "mlx_batch is mutually exclusive with "
                    + ", ".join(sorted(clashes))
                    + " (those own a per-request cache; batching owns one "
                    "shared batched cache)"
                )
        # MTP-head speculation (SPEC-mlx-mtp-draft.md). Mutually exclusive
        # with n-gram: both feed the one verify path, so allowing both would
        # silently pick a winner. Also before load(), so the error names the
        # real problem rather than a missing config.json.
        self._mtp = None
        self._mtp_caches: dict[int, list] = {}
        self._mtp_hidden: dict[int, object] = {}
        self._batch = None
        if self.config.mlx_mtp:
            from . import mtp as _mtp_mod

            if self.config.speculative:
                raise ValueError(
                    "mlx_mtp and speculative are mutually exclusive "
                    "(both source drafts for the same verify path)"
                )
            if not _mtp_mod.available(model_path):
                raise ValueError(
                    f"mlx_mtp needs an {_mtp_mod.SIDECAR} sidecar next to the "
                    f"weights; {model_path} has none"
                )
        load, _ = _require_mlx()
        self._model, self._tokenizer = load(model_path)
        self.model_path = model_path
        if self.config.mlx_batch:
            from .mlx_batch import MLXBatch

            self._batch = MLXBatch(self._model)
        if self.config.mlx_mtp:
            from . import mtp as _mtp_mod

            depth = self.config.mlx_mtp_depth or _mtp_mod.default_depth(model_path)
            target = getattr(self._model, "language_model", self._model)
            self._mtp = _mtp_mod.MTPDraft(target, model_path, depth)
        self.ctx = _MLXContext(self)
        self._stop_filters: dict[int, StopSequenceFilter] = {}
        self._detoks: dict[int, TokenDetokenizer] = {}
        self._states: dict[int, RequestState] = {}
        self._retired: dict[int, RequestState] = {}
        self._decode_calls = 0
        # Speculative state, populated only when config.speculative is on (so
        # a default engine allocates nothing extra per request): per-request
        # draft tables, owned caches, fallback flags, and
        # [drafted, accepted, recomputes] per request.
        self._spec_tables: dict[int, NgramTable] = {}
        self._caches: dict[int, list] = {}
        self._spec_ok: dict[int, bool] = {}
        self._spec_stats: dict[int, list[int]] = {}
        # Lookahead per request (see _lookahead): when present, the owned
        # cache already holds next_token, and this is (argmax, logits,
        # hidden) for the position after it, queued together but not
        # necessarily done. `hidden` is None without an MTP head.
        self._pending: dict[int, tuple[object, object, object]] = {}
        # The batch path's equivalent: per row, the lazy token sampled from
        # the forward queued at the end of the previous step.
        self._batch_pending: dict[int, object] = {}
        self.spec_drafted = 0
        self.spec_accepted = 0
        self.spec_recomputes = 0
        self.spec_fallbacks = 0
        # Engine-wide: drafts only while a drafting step measures more
        # tokens per second than a plain one on THIS engine. See spec_gate.
        from .spec_gate import SpecThroughputGate

        self.spec_gate = SpecThroughputGate()
        # Prefix prompt cache: token sequence -> (cache snapshot, base
        # token, MTP head cache or None, MTP hidden row or None, last
        # logits row). OrderedDict as bounded LRU (move_to_end on hit,
        # popitem oldest on overflow). The cache covers exactly len(key)
        # tokens and base is the argmax of the last logits row, so a greedy
        # exact hit replays deterministically; a sampled hit draws its own
        # first token from that row. A prompt that EXTENDS a key is a
        # partial hit: the snapshot is installed and only the rest is
        # prefilled. Entries are stored at prefill (the prompt) and at
        # retire (prompt + answer, what the next chat turn extends).
        self._prefix: OrderedDict[
            tuple[int, ...], tuple[list, int | None, Any, Any, Any]
        ] = OrderedDict()
        self.prefix_hits = 0
        self.prefix_partial_hits = 0
        self.prefix_hit_tokens = 0
        self.prefix_misses = 0
        # Tokens each owned cache actually holds (prompt + every token fed
        # since, including a lookahead's). The key an end-of-request prefix
        # entry is stored under: what the next turn of a chat extends.
        self._cache_tokens: dict[int, list[int]] = {}
        self._next_request_id = 0

    # --- helpers ------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self.ctx.closed:
            raise RuntimeError(
                "this MLXEngine has been closed; create a new one to keep serving"
            )

    def _make_cache(self) -> list:
        """Owned cache with optional quantized KV (mlx_kv_bits).

        Builds the model default (ArraysCache for linear layers, KVCache
        for full attention on qwen3_5) then swaps exact-type KVCache entries
        for QuantizedKVCache. Order is preserved positionally, which the
        model relies on (cache[i] belongs to layer i).

        Through mlx-lm's own builder, not `model.make_cache()` directly:
        only hybrid models define that method, and a plain-attention one
        (qwen3_moe, the 30B-A3B) relies on the builder's per-layer KVCache
        default. The stream path always went through the builder; the
        manual loop did not, which surfaced the day greedy requests moved
        onto it.
        """
        from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache

        cache = make_prompt_cache(self._model)
        bits = getattr(self, "_kv_bits", self.config.mlx_kv_bits)
        if bits is None:
            return cache
        if bits == 0:
            return cache  # manual-loop f16 control (no swap)
        if bits not in (4, 8):
            # Unreachable through __init__ (validated there); kept for an
            # engine built without it.
            raise ValueError(f"mlx_kv_bits must be 0, 4 or 8; got {bits}")
        return [
            QuantizedKVCache(group_size=64, bits=bits)
            if type(c) is KVCache
            else c
            for c in cache
        ]

    def _lookup(self, request_id: int) -> RequestState:
        req = self._states.get(request_id)
        if req is None:
            req = self._retired.get(request_id)
        if req is None:
            if 0 <= request_id < self._next_request_id:
                raise KeyError(
                    f"request {request_id} finished and is no longer retained "
                    f"(only the last {self.retained_finished_limit} finished requests are)"
                )
            raise KeyError(f"unknown request {request_id}")
        return req

    def _retire(
        self, req: RequestState, reason: str, token: int = -1, piece: str = ""
    ) -> StepOutput:
        req.finished = True
        req.finish_reason = reason
        req.next_token = None
        if reason != "error" and self.config.mlx_prefix_cache:
            self._prefix_insert_final(req)
        self._cache_tokens.pop(req.request_id, None)
        self._stop_filters.pop(req.request_id, None)
        self._detoks.pop(req.request_id, None)
        self._spec_tables.pop(req.request_id, None)
        self._caches.pop(req.request_id, None)
        self._pending.pop(req.request_id, None)
        self._batch_pending.pop(req.request_id, None)
        self._spec_ok.pop(req.request_id, None)
        self._spec_stats.pop(req.request_id, None)
        self._mtp_caches.pop(req.request_id, None)
        self._mtp_hidden.pop(req.request_id, None)
        if self._batch is not None:
            self._batch.leave([req.request_id])
        self._states.pop(req.request_id, None)
        self._retired[req.request_id] = req
        while len(self._retired) > self.retained_finished_limit:
            del self._retired[next(iter(self._retired))]
        return StepOutput(req.request_id, token, piece, True, reason)

    def _eog_ids(self) -> set[int]:
        # Prefer the wrapper's plural set (what stream_generate itself stops
        # on): Qwen-family tokenizers end turns with <|endoftext|> while
        # eos_token_id names <|im_end|>. A singular-only set leaks the
        # other's piece text into output and mislabels the finish reason.
        plural = getattr(self._tokenizer, "eos_token_ids", None)
        if plural:
            return set(plural)
        eos = getattr(self._tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return set(eos)

    def tokenize(
        self, text: str, add_special: bool = True, parse_special: bool = True
    ) -> list[int]:
        """Tokenize like ``Model.tokenize`` (server helpers call this shape).

        ``parse_special`` is accepted for signature parity and ignored: the HF
        tokenizer handles special tokens itself.
        """
        self._ensure_open()
        return list(
            self._tokenizer.encode(text, add_special_tokens=add_special)
        )

    def apply_chat_template(
        self, pairs: Sequence[tuple[str, str]], add_assistant: bool = True
    ) -> str:
        """Render chat pairs via the HF chat template (server helper shape).

        Thinking is disabled server-wide (``enable_thinking=False``): thinking
        traces would otherwise leak into output text and corrupt the tool-call
        protocol the routes parse out of it. Per-request opt-in is a future
        API item, not a silent default. Raises ValueError when the tokenizer
        carries no usable template, so ``render_pairs`` maps it to a 400 like
        the llama path.
        """
        self._ensure_open()
        messages = [{"role": role, "content": text} for role, text in pairs]
        try:
            return str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_assistant,
                    enable_thinking=False,
                )
            )
        except Exception as exc:  # noqa: BLE001 - request-level failure, mapped to 400
            raise ValueError(
                f"this model's chat template cannot be applied ({exc})"
            ) from exc

    # --- admission ----------------------------------------------------------

    def add_request(
        self,
        prompt: str | Sequence[int],
        params: RequestParams | None = None,
        *,
        add_special: bool = True,
    ) -> int:
        """Admit a request and return its id. Same refusal shape as MetalEngine
        (non-positive max_tokens, empty prompt, over-long prompt)."""
        self._ensure_open()
        rp = params or RequestParams()
        if rp.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be >= 1; got {rp.max_tokens} (no generation "
                f"satisfies a non-positive cap)"
            )
        if isinstance(prompt, str):
            tokens = list(
                self._tokenizer.encode(prompt, add_special_tokens=add_special)
            )
        else:
            tokens = list(prompt)
        if not tokens:
            raise ValueError("prompt tokenized to zero tokens")
        if len(tokens) >= self.config.n_ctx:
            raise ValueError(
                f"prompt is {len(tokens)} tokens, capacity is "
                f"{self.config.n_ctx}"
            )
        request_id = self._next_request_id
        self._next_request_id += 1
        self._states[request_id] = RequestState(
            request_id=request_id, seq_id=0, prompt=tokens, params=rp
        )
        if rp.stop:
            self._stop_filters[request_id] = StopSequenceFilter(rp.stop)
        # Every request runs the manual loop (see step) and carries the
        # verify path's per-request counters. The n-gram table
        # is seeded with the prompt (same rule as MetalEngine) only when
        # speculation is on; the MTP head sources its own drafts.
        self._spec_ok[request_id] = True
        self._spec_stats[request_id] = [0, 0, 0]
        if self.config.speculative and self._mtp is None and rp.temp <= 0:
            table = NgramTable()
            table.update_stream(tokens)
            self._spec_tables[request_id] = table
        return request_id

    def cancel(self, request_id: int) -> bool:
        """Drop a request mid-flight. False if already finished."""
        req = self._lookup(request_id)
        if req.finished:
            return False
        self._retire(req, "cancelled")
        return True

    # --- stepping -----------------------------------------------------------

    def _sampler_for(self, params: RequestParams):
        _, make_sampler = _require_mlx()
        return make_sampler(
            temp=params.temp, top_p=params.top_p, top_k=params.top_k
        )

    def step(self) -> list[StepOutput]:
        """Advance every in-flight request by one step (one token, or one
        verify round for speculating requests)."""
        self._ensure_open()
        if self._batch is not None:
            return self._step_batch()
        outputs: list[StepOutput] = []
        for req in list(self._states.values()):
            if req.finished:
                continue
            if req.request_id not in self._caches:
                outputs.extend(self._start_spec(req))
                if req.finished:
                    continue
            outputs.extend(self._verify(req))
        return outputs

    # Every request runs the manual loop (`_start_spec` + `_verify`). It used
    # to be the price of speculation, quantized KV or the prefix cache, with
    # mlx-lm's stream_generate as the fast default; with its lookahead (see
    # _lookahead) the loop outruns the generator by ~5% on a 27B, greedy or
    # sampled, so the generator path is gone. Sampling is per request and
    # lazy (`_sample_row_lazy`), drafting is greedy-only.

    def _prefix_lookup(self, req: RequestState):
        """Shared prefix probe: `(cache_copy, base, entry, matched)` or None.

        `matched` is how many leading prompt tokens `cache_copy` covers.
        On an exact hit that is the whole prompt and `base` is the first
        token to emit (see _first_token); on a partial hit -- the prompt
        extends a stored key -- `base` is None and the caller prefills
        `prompt[matched:]` itself. A strict prefix of a stored key never
        hits: the recurrent layers cannot be rewound to an earlier
        position.

        Split out so the batching path can reuse it. A hit yields a
        single-sequence cache -- which is precisely what `MLXBatch.join`
        wants, so the prefix cache and continuous batching compose instead
        of excluding each other.
        """
        key = tuple(req.prompt)
        entry = self._prefix.get(key)
        if entry is not None:
            self._prefix.move_to_end(key)
            self.prefix_hits += 1
            self.prefix_hit_tokens += len(req.prompt)
            return deepcopy(entry[0]), self._first_token(req, entry), entry, len(key)
        # No exact entry: the longest stored sequence this prompt EXTENDS.
        # A chat turn is the previous prompt + the answer + the new message,
        # and the previous request stored exactly that cache when it
        # retired; only the new message needs prefilling. Only extension,
        # never truncation: the recurrent layers' state cannot be rewound
        # to an earlier position.
        best = None
        for stored in self._prefix:
            n = len(stored)
            if n < len(key) and key[:n] == stored and (best is None or n > len(best)):
                best = stored
        if best is None:
            self.prefix_misses += 1
            return None
        entry = self._prefix[best]
        self._prefix.move_to_end(best)
        self.prefix_hits += 1
        self.prefix_partial_hits += 1
        self.prefix_hit_tokens += len(best)
        return deepcopy(entry[0]), None, entry, len(best)

    def _first_token(self, req: RequestState, entry: tuple) -> int:
        """A hit's first token: the stored argmax for a greedy request, a
        fresh draw from the stored last-position logits for a sampled one.
        Replaying the argmax into a sampled request would hand every repeat
        of a prompt the same first token, which is not what temp > 0 asked
        for; the KV is sampling-independent and is what the hit reuses."""
        if req.params.temp <= 0:
            return entry[1]
        return self._sample_row(req, entry[4])

    def _prefix_hit(self, req: RequestState) -> list[StepOutput] | None:
        """Prefix hit: install the snapshot and emit the first token.

        Returns None on miss (caller prefills normally). An exact hit costs
        no forward pass -- TTFT is deepcopy time -- and replays the stored
        base token (or, sampled, a draw from the stored logits row). A
        partial hit prefills only `prompt[matched:]` on top of the
        snapshot and stores the whole prompt for the next extension.
        """
        found = self._prefix_lookup(req)
        if found is None:
            return None
        cache, base, entry, matched = found
        rid = req.request_id
        self._caches[rid] = cache
        self._cache_tokens[rid] = list(req.prompt[:matched])
        if self._mtp is not None:
            # The head cache is part of the snapshot: restoring only the trunk
            # would leave the head cold and silently halve draft quality
            # (0.906 -> 0.479 at depth 1). hidden rows are immutable mx
            # arrays, so only the cache needs copying.
            self._mtp_caches[rid] = deepcopy(entry[2])
            self._mtp_hidden[rid] = entry[3]
        if base is None:
            # Partial hit: prefill only what this prompt adds, then store
            # the whole prompt so the next extension starts from here.
            logits = self._prefill_owned(rid, req.prompt, start=matched)
            base = self._sample_row(req, logits[0, -1])
            self._prefix_insert(req, self._caches[rid], base, logits[0, -1])
        req.next_token = base
        return [self._feed(req, base, self._piece(req, base))]

    def _prefix_insert(
        self, req: RequestState, cache: list, base: int, last_logits: Any
    ) -> None:
        """Snapshot a freshly prefilled prompt for future exact repeats.
        `last_logits` is the final position's logits row, kept so a sampled
        repeat can draw its own first token (see _first_token)."""
        if len(req.prompt) < self.config.prefix_cache_min_tokens:
            return
        self._prefix_store(tuple(req.prompt), req.request_id, cache, base, last_logits)

    def _prefix_store(
        self, key: tuple[int, ...], rid: int, cache: list, base: int | None,
        last_logits: Any, hidden: Any = None,
    ) -> None:
        if key in self._prefix:
            self._prefix.move_to_end(key)
            return
        # A shorter entry this one extends is NOT redundant: an exact repeat
        # of that prompt still needs it, because the longer cache cannot be
        # cut back (recurrent state). LRU eviction by size decides.
        mcache = self._mtp_caches.get(rid)
        self._prefix[key] = (
            deepcopy(cache),
            base,
            deepcopy(mcache) if mcache is not None else None,
            self._mtp_hidden.get(rid) if hidden is None else hidden,
            last_logits,
        )
        while len(self._prefix) > max(1, self.config.mlx_prefix_cache_size):
            self._prefix.popitem(last=False)

    def _prefix_insert_final(self, req: RequestState) -> None:
        """At retire: store the cache as it stands, under the tokens it
        holds (prompt + answer + the end token, if one was fed). That is the
        sequence the next turn of a chat extends, so the next prompt is a
        partial hit that prefills only the new message. Needs the lookahead
        that fed the last token: its logits are the entry's last row."""
        rid = req.request_id
        cache = self._caches.get(rid)
        tokens = self._cache_tokens.get(rid)
        pending = self._pending.get(rid)
        if cache is None or tokens is None or pending is None:
            return
        if len(tokens) < self.config.prefix_cache_min_tokens:
            return
        nxt, logits, hidden = pending
        key = tuple(int(t.item()) if not isinstance(t, int) else t for t in tokens)
        if self._mtp is not None:
            # The head pairs a token only when the step after it runs; the
            # last fed token never got its step, so pair it now, and the
            # entry's hidden row is the lookahead's (the last position).
            self._mtp.accept(self._mtp_caches[rid], self._mtp_hidden[rid], [key[-1]])
        self._prefix_store(key, rid, cache, int(nxt.item()), logits[0, -1], hidden)

    def _snapshot_cache(self, cache: list) -> list:
        """A rewind point for one verify round, at effectively zero cost.

        mlx arrays are immutable and `ArraysCache.__setitem__` REPLACES the
        entry rather than writing through it, so copying the list of
        references captures the recurrent state without copying the 154 MB
        behind it. `KVCache`/`QuantizedKVCache` append into a preallocated
        buffer, so their rewind is just the offset -- anything past it is
        overwritten on the next write.

        This is what makes the hybrid trunk rewindable after all: the module
        docstring's "no hybrid rewind" held for a full re-prefill, not for a
        snapshot/restore. Rejection costs O(matched) re-fed tokens instead of
        O(prompt + output).
        """
        from mlx_lm.models.cache import ArraysCache

        return [
            ("arrays", list(c.cache)) if isinstance(c, ArraysCache)
            else ("offset", c.offset)
            for c in cache
        ]

    def _restore_cache(self, cache: list, snap: list) -> None:
        """Undo one verify round's writes. Exact: same arrays, same offsets."""

        for c, (kind, val) in zip(cache, snap):
            if kind == "arrays":
                c.cache = list(val)
            else:
                c.offset = val

    def _prefill_owned(
        self, rid: int, tokens: Sequence[int], *, start: int = 0
    ) -> object:
        """Prefill `tokens` into an owned cache; return the last logits.

        `start > 0` continues the cache already installed for `rid` (a
        partial prefix hit) from that position; the MTP head, if any, is
        continued the same way, pairing its stored last hidden row with
        `tokens[start]` first.

        The one prefill path, for a cold prompt and for a partial prefix
        hit alike, so the trunk cache and the MTP head cache can never drift
        apart -- whatever fills one fills the other, chunked identically.

        Position convention the head depends on: after this returns, the head
        cache holds pairs `(hidden[p], tokens[p+1])` for `p` in
        `0..len(tokens)-2`, and `_mtp_hidden[rid]` is the hidden row at
        `len(tokens)-1` -- exactly the row that pairs with the token the
        caller is about to make `next_token`. `propose` then appends that
        pair as position `len(tokens)-1` and trims it back.
        """
        mx = _mx()
        mtp = self._mtp
        if start:
            cache = self._caches[rid]
            mcache = self._mtp_caches.get(rid)
            if mtp is not None:
                mtp.accept(mcache, self._mtp_hidden[rid], [tokens[start]])
        else:
            cache = self._make_cache()
            mcache = mtp.make_cache() if mtp is not None else None
        logits = hidden = None
        for i in range(start, len(tokens), _PREFILL_CHUNK):
            chunk = list(tokens[i : i + _PREFILL_CHUNK])
            if mtp is None:
                logits = self._model(mx.array(chunk)[None], cache=cache)
                continue
            logits, hidden = mtp.trunk_forward(mx.array(chunk)[None], cache)
            # Pairs for this chunk's positions; the final position of the
            # whole sequence has no successor yet, so it is left for propose.
            nxt = list(tokens[i + 1 : i + len(chunk) + 1])
            if nxt:
                mtp.accept(mcache, hidden[:, : len(nxt), :], nxt)
        mx.eval(logits)
        self._caches[rid] = cache
        self._cache_tokens[rid] = list(tokens)
        if mtp is not None:
            self._mtp_caches[rid] = mcache
            self._mtp_hidden[rid] = hidden[:, -1:, :]
        return logits

    def _start_spec(self, req: RequestState) -> list[StepOutput]:
        """Prefill the prompt into an owned cache and emit the continuation.

        Mirrors MetalEngine's prefill-completing row: the sampled base token
        is OUTPUT now (via _feed, so EOG/stop/cap apply) and stored as
        next_token for the following verify step to pack. Dropping it here
        would shift every later verify by one (the index-0 divergence).
        With mlx_prefix_cache, an exact hit skips prefill via _prefix_hit
        and a miss snapshots the fresh cache for future repeats.
        """
        try:
            if self.config.mlx_prefix_cache:
                hit = self._prefix_hit(req)
                if hit is not None:
                    self._lookahead(req, req.next_token)
                    return hit
            logits = self._prefill_owned(req.request_id, req.prompt)
            cache = self._caches[req.request_id]
            base = self._sample_row(req, logits[0, -1])
            req.next_token = base
            if self.config.mlx_prefix_cache:
                # Snapshot covers exactly the prompt: before the lookahead.
                self._prefix_insert(req, cache, base, logits[0, -1])
            self._lookahead(req, base)
            return [self._feed(req, base, self._piece(req, base))]
        except BaseException:  # noqa: BLE001 - re-raised; must retire slot on any failure
            try:
                self._retire(req, "error")
            except BaseException:  # noqa: BLE001 - must never mask the real failure
                pass
            raise

    def _lookahead(self, req: RequestState, token: int) -> None:
        """Queue the forward for `token` now, so the GPU computes it while
        Python decodes the piece, runs stop filters and builds outputs.

        This is what mlx-lm's generate_step does (async_eval of the next
        step before yielding the current token) and the manual loop did not,
        which left it ~10% behind the stream path on a 27B. The cache then
        holds `prompt + output` rather than `prompt + output[:-1]`; the next
        verify step takes the row from `_pending` instead of recomputing it
        and appends only its draft rows -- the base row was always going to
        be forwarded, so nothing is wasted when drafts follow. With an MTP
        head the trunk's hidden row rides along, and the head cache's lazy
        state is queued in the same batch so no later step waits on it.
        """
        if req.finished:
            return
        mx = _mx()
        # `token` may be a lazy scalar: the argmax queued with the PREVIOUS
        # lookahead. Its own argmax is queued here with its logits, in one
        # batch, so that forcing the token later waits only on that batch
        # and never on the forward queued after it. Queue the argmax
        # separately and MLX commits it with the next forward, and forcing
        # it then waits for both: the loop serialises and the overlap is
        # gone (measured: 78 ms/step against the stream path's 69).
        tok = token if isinstance(token, mx.array) else mx.array(token)
        rid = req.request_id
        fed = self._cache_tokens.get(rid)
        if fed is not None:
            # Lazy tokens are appended as arrays and resolved when the key
            # is built (see _prefix_insert_final); forcing here would undo
            # the pipelining this lookahead exists for.
            fed.append(tok)
        if self._mtp is None:
            logits = self._model(tok.reshape(1, 1), cache=self._caches[rid])
            nxt = self._sample_row_lazy(req, logits[0, -1])
            mx.async_eval(nxt, logits)
            self._pending[rid] = (nxt, logits, None)
            return
        logits, hidden = self._mtp.trunk_forward(tok.reshape(1, 1), self._caches[rid])
        nxt = self._sample_row_lazy(req, logits[0, -1])
        head_state = [a for a in self._mtp_caches[rid][0].state if a is not None]
        mx.async_eval(nxt, logits, hidden, *head_state)
        self._pending[rid] = (nxt, logits, hidden)

    def _verify(self, req: RequestState) -> list[StepOutput]:
        """Verify one request's drafts; same retire-on-failure contract as
        ``_advance`` on MetalEngine. Returns every accepted token's output."""
        try:
            return self._verify_rows(req)
        except BaseException:  # noqa: BLE001 - re-raised; must retire slot on any failure
            try:
                self._retire(req, "error")
            except BaseException:  # noqa: BLE001 - must never mask the real failure
                pass
            raise

    def _verify_rows(self, req: RequestState) -> list[StepOutput]:
        mx = _mx()
        table = self._spec_tables.get(req.request_id)
        remaining = req.params.max_tokens - req.n_generated
        # Drafting only when speculation is on; mlx_kv_bits alone rides the
        # manual loop with zero drafts (single-row forward, same as plain
        # decode but through the owned, possibly quantized, cache).
        # The gate is asked once per step, whatever the draft source: an
        # MTP head's rows cost the forward exactly what n-gram rows do.
        greedy = req.params.temp <= 0
        gate_open = greedy and self.spec_gate.wants_drafts()
        max_drafts = (
            self.config.spec_max_drafts
            if (
                self.config.speculative
                and gate_open
                and self._spec_ok.get(req.request_id, True)
            )
            else 0
        )
        allow = max(0, min(max_drafts, remaining - 1))
        base = req.next_token
        assert base is not None
        rid = req.request_id
        mtp = self._mtp
        t_start = time.perf_counter()
        if mtp is not None:
            # The head drafts from the trunk hidden row that pairs with
            # `base`; propose() trims its own speculative entries, so the
            # head cache still mirrors only confirmed tokens on return.
            allow = max(0, min(mtp.depth, remaining - 1)) if gate_open else 0
            h_prev = self._mtp_hidden[rid]
            drafts = mtp.propose(self._mtp_caches[rid], h_prev, base, allow)
        elif table is None or allow == 0:
            # kv-bits-only manual loop: no drafting, single-row forward.
            drafts = []
        else:
            history = req.prompt + req.output_tokens
            context = history[-(table.order - 1) :] if table.order > 1 else []
            drafts = table.predict(context, allow)
        rows = [base, *drafts]
        # With a lookahead pending, the cache already holds `base` and its
        # logits are queued: forward only the draft rows and put the two
        # together as if one (1 + drafts)-row forward had run.
        pending_tok, pending, pending_hidden = self._pending.pop(
            rid, (None, None, None)
        )
        # Rewind point for the mismatch path (free: see _snapshot_cache).
        # Only worth taking when there is something to reject. Taken after
        # the lookahead's write, which is confirmed either way.
        snap = self._snapshot_cache(self._caches[rid]) if drafts else None
        # `_cache_tokens` must follow every write below: it is the key the
        # end-of-request prefix entry is stored under, and a key shorter
        # than the cache hands the next turn a partial hit on positions it
        # never asked for.
        fed = self._cache_tokens.get(rid)
        n_fed = len(fed) if fed is not None else 0
        msnap = (
            self._mtp_caches[rid][0].offset
            if (mtp is not None and drafts)
            else None
        )
        lookahead_queued = False
        if mtp is not None and not drafts and pending is not None:
            # Plain MTP step, pipelined like the n-gram one below. The head
            # pairs (h_prev, base) lazily; its state is queued with the
            # lookahead, so nothing here waits on the forward in flight.
            logits, hidden, tok_arr = pending, pending_hidden, pending_tok
            mtp.accept(self._mtp_caches[rid], h_prev, [base], sync=False)
            self._mtp_hidden[rid] = hidden[:, -1:, :]
            self._lookahead(req, tok_arr)
            lookahead_queued = True
        elif mtp is not None and pending is not None:
            logits_d, hidden_d = mtp.trunk_forward(
                mx.array(drafts)[None], self._caches[rid]
            )
            logits = mx.concatenate([pending, logits_d], axis=1)
            hidden = mx.concatenate([pending_hidden, hidden_d], axis=1)
            mx.eval(logits)
        elif mtp is not None:
            logits, hidden = mtp.trunk_forward(
                mx.array(rows)[None], self._caches[rid]
            )
            mx.eval(logits)
        elif not drafts:
            # Plain step, mlx-lm's generate_step shape: the token stays a
            # lazy array (queued with the previous lookahead) while the NEXT
            # forward is built from it and queued, and only then is it
            # forced. Forcing first would leave the GPU idle for the graph
            # build of 64 layers every step.
            if pending is None:
                logits = self._model(mx.array(rows)[None], cache=self._caches[rid])
                tok_arr = self._sample_row_lazy(req, logits[0, -1])
            else:
                logits, tok_arr = pending, pending_tok
            self._lookahead(req, tok_arr)
            lookahead_queued = True
        elif pending is None:
            logits = self._model(
                mx.array(rows)[None], cache=self._caches[rid]
            )
            mx.eval(logits)
        else:
            logits = mx.concatenate([
                pending,
                self._model(mx.array(drafts)[None], cache=self._caches[rid]),
            ], axis=1)
            mx.eval(logits)
        if fed is not None and not lookahead_queued:
            # The forward above fed the draft rows (or every row, without
            # a lookahead's base already in place).
            fed.extend(drafts if pending is not None else rows)
        self.spec_drafted += len(drafts)
        stats = self._spec_stats[req.request_id]
        stats[0] += len(drafts)
        accepted: list[int] = []
        matched = 0
        if lookahead_queued:
            accepted.append(int(tok_arr.item()))
        for k in range(0 if not lookahead_queued else len(rows), len(rows)):
            token = int(mx.argmax(logits[0, k]).item())
            accepted.append(token)
            if k < len(drafts):
                if token != drafts[k]:
                    break
                matched += 1
            # k == len(drafts) is the bonus row past the last draft:
            # conditioned on a fully matched prefix, so always valid output.
        self.spec_accepted += matched
        stats[1] += matched
        if len(accepted) < len(rows):
            # Mismatch: the cache advanced past the accepted prefix, so
            # rewind it to the snapshot. Full matches skip this: the cache
            # already holds exactly the accepted stream.
            self.spec_recomputes += 1
            stats[2] += 1
            # Rejection: rewind to the pre-forward state and re-feed only the
            # accepted prefix (matched + 1 tokens), rather than re-prefilling
            # prompt + output. accepted[:-1] are confirmed but not yet fed;
            # accepted[-1] becomes next_token via the _feed loop below and
            # must stay unconsumed, so it is NOT re-fed here.
            self._restore_cache(self._caches[rid], snap)
            keep = rows[: len(accepted)]
            n = len(keep) - 1
            if pending is not None:
                # The restored cache already holds `base`.
                keep = keep[1:]
            if fed is not None:
                del fed[n_fed:]
                fed.extend(keep)
            if mtp is not None:
                self._mtp_caches[rid][0].offset = msnap
                if keep:
                    logits, hidden = mtp.trunk_forward(
                        mx.array(keep)[None], self._caches[rid]
                    )
                    mx.eval(logits)
                    if pending is not None:
                        hidden = mx.concatenate([pending_hidden, hidden], axis=1)
                else:
                    # Only the base row survives, and its hidden row was the
                    # lookahead's.
                    hidden = pending_hidden
                h_seq = h_prev
                if n:
                    h_seq = mx.concatenate([h_prev, hidden[:, :n, :]], axis=1)
                mtp.accept(self._mtp_caches[rid], h_seq, [base, *accepted[:n]])
                self._mtp_hidden[rid] = hidden[:, -1:, :]
            elif keep:
                logits = self._model(
                    mx.array(keep)[None], cache=self._caches[rid]
                )
                mx.eval(logits)
        elif mtp is not None and not lookahead_queued:
            # Full match: the trunk cache already holds the accepted stream,
            # so advance the head by the same span. Entries run from the
            # position of `base` through the last accepted token's
            # predecessor -- hence h_prev prepended and accepted[:-1] paired
            # with the rows' own hidden states.
            n = len(drafts)
            h_seq = h_prev
            if n:
                h_seq = mx.concatenate([h_prev, hidden[:, :n, :]], axis=1)
            mtp.accept(self._mtp_caches[rid], h_seq, [base, *accepted[:n]])
            self._mtp_hidden[rid] = hidden[:, -1:, :]
        # Measured here, after any mismatch re-feed and before the token
        # bookkeeping: the forwards are the cost the gate is weighing.
        self.spec_gate.record(
            bool(drafts), len(accepted), time.perf_counter() - t_start
        )
        # Queue the next row before the bookkeeping below, so it overlaps
        # (the plain step queued it before forcing its token, above).
        if not lookahead_queued:
            self._lookahead(req, accepted[-1])
        self._feed_table(req, accepted)
        drafted, accepted_n, recomputed = stats
        if (
            drafted > 0
            and (
                drafted >= _SPEC_FALLBACK_MIN_DRAFTS
                or recomputed >= _SPEC_FALLBACK_MIN_RECOMPUTES
            )
            and accepted_n / drafted < _SPEC_FALLBACK_MIN_RATE
        ):
            if self._spec_ok.get(req.request_id, True):
                self._spec_ok[req.request_id] = False
                self.spec_fallbacks += 1
        outputs: list[StepOutput] = []
        for token in accepted:
            outputs.append(self._feed(req, token, self._piece(req, token)))
            if outputs[-1].finished:
                break
        return outputs

    def _feed_table(self, req: RequestState, new_tokens: Sequence[int]) -> None:
        """Record the pairs ending in freshly accepted tokens (same bounded-
        window rule as MetalEngine: only the trailing order-1 history plus
        the new tokens, never O(history) per step)."""
        table = self._spec_tables.get(req.request_id)
        if table is None or not new_tokens:
            return
        prefix = (
            (req.prompt + req.output_tokens)[-(table.order - 1) :]
            if table.order > 1
            else []
        )
        table.update_stream([*prefix, *new_tokens])


    def _step_batch(self) -> list[StepOutput]:
        """One continuous-batching step: admit joiners, then decode all rows.

        Admission prefills the joiner ALONE and merges the result, so ragged
        prompt lengths never reach the decode loop (mlx-lm's merge sets
        left_padding for us). The prefill's own argmax is that request's
        first token, emitted here exactly as the single-stream path does --
        dropping it would shift every later token by one.
        """
        mx = _mx()
        outputs: list[StepOutput] = []
        batch = self._batch
        assert batch is not None

        for req in list(self._states.values()):
            if req.finished or batch.index_of(req.request_id) is not None:
                continue
            try:
                cache = token = None
                if self.config.mlx_prefix_cache:
                    # A hit skips prefill entirely: the snapshot already
                    # covers the whole prompt, so admission costs a deepcopy
                    # instead of a forward pass. Its stored base token is the
                    # prefill's argmax, so replay stays deterministic.
                    found = self._prefix_lookup(req)
                    if found is not None:
                        cache, token, _entry, matched = found
                        if token is None:
                            # Partial hit: extend the snapshot with what this
                            # prompt adds, on its own before the join.
                            logits = None
                            for i in range(matched, len(req.prompt), _PREFILL_CHUNK):
                                chunk = req.prompt[i : i + _PREFILL_CHUNK]
                                logits = self._model(mx.array(chunk)[None], cache=cache)
                            mx.eval(logits)
                            token = self._sample_row(req, logits[0, -1])
                            self._prefix_insert(req, cache, token, logits[0, -1])
                if cache is None:
                    cache = self._make_cache()
                    logits = None
                    for i in range(0, len(req.prompt), _PREFILL_CHUNK):
                        chunk = req.prompt[i : i + _PREFILL_CHUNK]
                        logits = self._model(mx.array(chunk)[None], cache=cache)
                    mx.eval(logits)
                    token = self._sample_row(req, logits[0, -1])
                    if self.config.mlx_prefix_cache:
                        # Snapshot BEFORE the join: once merged the cache is
                        # batched, and a batched snapshot could not be
                        # restored into a different batch later.
                        self._prefix_insert(req, cache, token, logits[0, -1])
                batch.join(req.request_id, cache)
                outputs.append(
                    self._feed(req, token, self._piece(req, token))
                )
            except BaseException:  # noqa: BLE001 - re-raised; must not leave a half-joined row
                try:
                    self._retire(req, "error")
                except BaseException:  # noqa: BLE001 - never mask the real failure
                    pass
                raise

        if batch.empty:
            self._batch_pending.clear()
            return outputs

        # Rows retired between steps (cancel) leave before anything is fed.
        batch.leave([
            rid for rid in batch.rows
            if (self._states.get(rid) is None or self._states[rid].finished)
        ])
        if batch.empty:
            self._batch_pending.clear()
            return outputs

        # Decode every live row together, mlx-lm's generate_step shape: the
        # forward for what each row feeds next is built from LAZY tokens --
        # the previous step's sampled rows, or a joiner's prefill token --
        # and queued, with its own per-row sampling, as one batch; only
        # then are this step's tokens forced. Forcing first would idle the
        # GPU through the graph build and the Python below, which is what
        # left this loop a tenth behind the stream path (see _lookahead).
        rows = list(batch.rows)
        feed = [
            self._batch_pending.get(rid, self._states[rid].next_token)
            for rid in rows
        ]
        logits = batch.step_lazy(feed)
        sampled = {
            rid: self._sample_row_lazy(self._states[rid], logits[i, -1])
            for i, rid in enumerate(rows)
        }
        mx.async_eval(*sampled.values())
        emit = self._batch_pending
        self._batch_pending = sampled
        for rid in rows:
            tok = emit.get(rid)
            if tok is None:
                continue  # a joiner: its first token was emitted at admission
            req = self._states[rid]
            token = int(tok.item())
            outputs.append(
                self._feed(req, token, self._piece(req, token))
            )
        # A row that finished on the token just forced had its next forward
        # queued above for nothing; harmless, the row leaves now.
        done = [rid for rid in rows if self._states.get(rid) is None or self._states[rid].finished]
        if done:
            batch.leave(done)
            for rid in done:
                self._batch_pending.pop(rid, None)
        return outputs

    def _sample_row_lazy(self, req: RequestState, row):
        """`_sample_row` without the force: a lazy scalar the caller may
        queue with the forward that produced `row`."""
        mx = _mx()
        if req.params.temp and req.params.temp > 0:
            return self._sampler_for(req.params)(row[None]).reshape(())
        return mx.argmax(row)

    def _sample_row(self, req: RequestState, row) -> int:
        """Sample one row's logits with THAT request's parameters.

        Per-row rather than one sampler for the batch: two requests in the
        same step can ask for different temperatures, and a shared sampler
        would silently give one of them the other's settings.
        """
        mx = _mx()
        if req.params.temp and req.params.temp > 0:
            sampler = self._sampler_for(req.params)
            return int(sampler(row[None]).item())
        return int(mx.argmax(row).item())

    def _piece(self, req: RequestState, token: int) -> str:
        """The text this token adds, through the request's incremental
        detokenizer: a character spelled over several byte-fallback tokens
        is emitted whole by the one that completes it, where a per-token
        decode gave U+FFFD for each (the routes stream pieces, not text_of).
        Built here rather than at admission so the stub engines and the
        hand-fed tests need no table of their own."""
        detok = self._detoks.get(req.request_id)
        if detok is None:
            detok = self._detoks[req.request_id] = TokenDetokenizer(
                self._tokenizer.decode
            )
        return detok.push(token)

    def _flush_piece(self, req: RequestState) -> str:
        """Whatever `_piece` still holds, for the retire that ends the
        request: joined pieces must stay equal to text_of, which decodes a
        trailing partial character to a replacement character."""
        detok = self._detoks.get(req.request_id)
        return detok.flush() if detok is not None else ""

    def _feed(self, req: RequestState, token: int, piece: str) -> StepOutput:
        """Account for one generated token: EOG, then stop, then cap, then
        context."""
        self._decode_calls += 1
        if req.params.stop_at_eog and token in self._eog_ids():
            # No piece: the end-of-turn marker is a control token, not text.
            # MetalEngine retires the same way (token, empty piece), and the
            # stream path never yields it at all -- passing `piece` here made
            # the manual-loop paths (prefix cache, batching) emit a literal
            # "<|im_end|>" into the response body. Only a held partial
            # character (already in text_of as its replacement) rides along.
            return self._retire(req, "eog", token, self._flush_piece(req))
        req.output_tokens.append(token)
        req.n_generated += 1
        req.next_token = token
        # The finishing token travels with its retire, as in MetalEngine: the
        # routes build the response text from StepOutput.piece and run their
        # own StopSequenceFilter over it, so a bare retire dropped the capped
        # token from the text (usage still counted it) and withheld the
        # delimiter from the route's filter, leaking the partial-match prefix.
        stop_filter = self._stop_filters.get(req.request_id)
        if stop_filter is not None:
            _emittable, hit = stop_filter.push(piece)
            if hit:
                return self._retire(req, "stop_sequence", token, piece + self._flush_piece(req))
        if req.n_generated >= req.params.max_tokens:
            return self._retire(req, "length", token, piece + self._flush_piece(req))
        # Here, not in the verify loop alone: the batch step and the prefix
        # hit account their tokens through this function too, and with the
        # check elsewhere a batched request ran past n_ctx and called it
        # "length". Counted after the append, as Metal's n_pos check is, so
        # the retiring token is in tokens_of like the piece is in the text.
        if len(req.prompt) + req.n_generated >= self.config.n_ctx:
            return self._retire(req, "context", token, piece + self._flush_piece(req))
        return StepOutput(req.request_id, token, piece, False, None)

    # --- inspection ---------------------------------------------------------

    @property
    def has_work(self) -> bool:
        return any(not r.finished for r in self._states.values())

    @property
    def n_in_flight(self) -> int:
        return len(self._states)

    @property
    def n_free_seq_slots(self) -> int:
        return max(0, self.config.n_seq_max - len(self._states))

    @property
    def n_retained_finished(self) -> int:
        """Finished requests still readable through `state` / `tokens_of` / `text_of`."""
        return len(self._retired)

    @property
    def retained_finished_limit(self) -> int:
        """Cap on `n_retained_finished`: MetalEngine's rule, the window must
        cover a full batch's worth of retirements (every request in flight
        together is read after IT ends while its peers still run), and never
        shrink below the floor for a narrow configuration."""
        return max(self.config.n_seq_max, MIN_RETAINED_FINISHED)

    @property
    def spec_acceptance_rate(self) -> float | None:
        """Fraction of drafted tokens the target confirmed, or None before any."""
        if self.spec_drafted == 0:
            return None
        return self.spec_accepted / self.spec_drafted

    def state(self, request_id: int) -> RequestState:
        return self._lookup(request_id)

    def tokens_of(self, request_id: int) -> list[int]:
        return list(self._lookup(request_id).output_tokens)

    def text_of(self, request_id: int) -> str:
        return self._tokenizer.decode(self._lookup(request_id).output_tokens)

    def drain(self, max_steps: int = 100_000) -> Iterable[StepOutput]:
        """Step until nothing is in flight, yielding every token produced."""
        for _ in range(max_steps):
            if not self.has_work:
                return
            yield from self.step()
