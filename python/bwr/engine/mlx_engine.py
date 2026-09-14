"""MLXEngine: mlx-lm behind the MetalEngine interface (SPEC-mlx-engine.md).

Each request owns an independent mlx generator (stream path) or an owned
cache (speculative path); the worker thread drives them one step at a time.
N-gram speculation (SPEC-mlx-engine.md M12) reuses ``NgramTable`` verbatim --
only the verify loop is backend-specific, because mlx-lm 0.31.3 has no hybrid
rewind: on any mismatch the cache is rebuilt exactly by re-feeding (never
lossy), and a per-request rolling acceptance disables drafting below 0.5
after 8 drafted. Greedy only, like Metal.

mlx-lm is a core dependency (it ships with big-white-rabbit). Nothing in this
module imports it at top level anyway: the import happens in ``__init__``
so a broken install fails with a clear error naming the backend instead of
at ``import bwr`` time.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Iterable, Iterator, Sequence

from .batching import RequestState
from .config import EngineConfig, RequestParams, StopSequenceFilter
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

    def __init__(self, engine: "MLXEngine") -> None:
        self._engine = engine
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        eng = self._engine
        eng._states.clear()
        eng._retired.clear()
        eng._streams.clear()
        eng._caches.clear()
        eng._spec_tables.clear()
        eng._spec_ok.clear()
        eng._spec_stats.clear()
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
    """One mlx-lm generator per request, stepped in lockstep.

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
        # silently pick a winner. Validated BEFORE load() so a misconfigured
        # run fails in milliseconds instead of after pulling in ~19 GB of
        # weights (and with an error naming the real problem, not a missing
        # config.json).
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
        self._streams: dict[int, Iterator] = {}
        self._stop_filters: dict[int, StopSequenceFilter] = {}
        self._states: dict[int, RequestState] = {}
        self._retired: dict[int, RequestState] = {}
        self._retired_limit = max(8, MIN_RETAINED_FINISHED)
        self._decode_calls = 0
        # Speculative state, populated only when config.speculative is on (so
        # a default engine allocates nothing extra per request): per-request
        # draft tables, owned caches, fallback flags, and
        # [drafted, accepted, recomputes] per request.
        self._spec_tables: dict[int, NgramTable] = {}
        self._caches: dict[int, list] = {}
        self._spec_ok: dict[int, bool] = {}
        self._spec_stats: dict[int, list[int]] = {}
        self.spec_drafted = 0
        self.spec_accepted = 0
        self.spec_recomputes = 0
        self.spec_fallbacks = 0
        # Exact-prefix prompt cache: prompt tokens -> (cache snapshot, base
        # token). OrderedDict as bounded LRU (move_to_end on hit, popitem
        # oldest on overflow). Entries are prompt-boundary snapshots: the
        # cache covers exactly len(key) tokens and base is the argmax of the
        # prefill's last logits row, so a hit replays deterministically.
        self._prefix: OrderedDict[tuple[int, ...], tuple[list, int]] = OrderedDict()
        self.prefix_hits = 0
        self.prefix_misses = 0
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
        """
        from mlx_lm.models.cache import QuantizedKVCache
        from mlx_lm.models.cache import KVCache

        cache = self._model.make_cache()
        bits = getattr(self, "_kv_bits", self.config.mlx_kv_bits)
        if bits is None:
            return cache
        if bits == 0:
            return cache  # manual-loop f16 control (no swap)
        if bits not in (4, 8):
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
                    f"(only the last {self._retired_limit} finished requests are)"
                )
            raise KeyError(f"unknown request {request_id}")
        return req

    def _retire(
        self, req: RequestState, reason: str, token: int = -1, piece: str = ""
    ) -> StepOutput:
        req.finished = True
        req.finish_reason = reason
        req.next_token = None
        self._stop_filters.pop(req.request_id, None)
        self._streams.pop(req.request_id, None)
        self._spec_tables.pop(req.request_id, None)
        self._caches.pop(req.request_id, None)
        self._spec_ok.pop(req.request_id, None)
        self._spec_stats.pop(req.request_id, None)
        self._mtp_caches.pop(req.request_id, None)
        self._mtp_hidden.pop(req.request_id, None)
        if self._batch is not None:
            self._batch.leave([req.request_id])
        self._states.pop(req.request_id, None)
        self._retired[req.request_id] = req
        while len(self._retired) > self._retired_limit:
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
        if (
            self.config.speculative
            or self.config.mlx_kv_bits is not None
            or self.config.mlx_prefix_cache
            or self._mtp is not None
        ):
            # Seed with the prompt (same rule as MetalEngine); the cache
            # itself is built lazily on the first spec step. The MTP head
            # sources its own drafts, so it allocates no table -- only the
            # per-request counters the verify path shares.
            if self._mtp is None:
                table = NgramTable()
                table.update_stream(tokens)
                self._spec_tables[request_id] = table
            self._spec_ok[request_id] = True
            self._spec_stats[request_id] = [0, 0, 0]
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

    def _start(self, req: RequestState) -> None:
        """Create the mlx generator on first step (needs the live request)."""
        from mlx_lm import stream_generate

        self._streams[req.request_id] = stream_generate(
            self._model,
            self._tokenizer,
            prompt=req.prompt,
            max_tokens=req.params.max_tokens,
            sampler=self._sampler_for(req.params),
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
            if self._use_manual(req):
                if req.request_id not in self._caches:
                    outputs.extend(self._start_spec(req))
                    if req.finished:
                        continue
                outputs.extend(self._verify(req))
            else:
                outputs.extend(self._step_stream(req))
        return outputs

    def _use_manual(self, req: RequestState) -> bool:
        """Manual-loop ownership is sticky: once a request owns a cache it
        stays manual. Falling back to the stream path would restart
        generation from the prompt (a fresh generator knows nothing of
        emitted tokens), so fallback only ever disables DRAFTING, never the
        loop. Entry requires the flag plus greedy sampling, or a quantized
        KV setting (the stream path builds its own f16 cache internally,
        so mlx_kv_bits can only take effect through the owned cache)."""
        if req.request_id in self._caches:
            return True
        if self.config.mlx_kv_bits is not None:
            return True
        if self.config.mlx_prefix_cache:
            return True
        if self._mtp is not None and req.params.temp <= 0:
            return True
        return bool(self.config.speculative and req.params.temp <= 0)

    def _prefix_lookup(self, req: RequestState):
        """Shared exact-prefix probe: `(cache_copy, base, entry)` or None.

        Split out so the batching path can reuse it. A hit yields a
        single-sequence cache covering exactly the prompt -- which is
        precisely what `MLXBatch.join` wants, so the prefix cache and
        continuous batching compose instead of excluding each other.
        """
        key = tuple(req.prompt)
        entry = self._prefix.get(key)
        if entry is None:
            self.prefix_misses += 1
            return None
        self._prefix.move_to_end(key)
        self.prefix_hits += 1
        return deepcopy(entry[0]), entry[1], entry

    def _prefix_hit(self, req: RequestState) -> list[StepOutput] | None:
        """Exact-prefix hit: install the snapshot and replay its base token.

        Returns None on miss (caller prefills normally). The snapshot covers
        exactly len(prompt) positions and base is the prefill's argmax, so
        replay is deterministic — no forward pass, TTFT is deepcopy time.
        """
        found = self._prefix_lookup(req)
        if found is None:
            return None
        cache, base, entry = found
        self._caches[req.request_id] = cache
        if self._mtp is not None:
            # The head cache is part of the snapshot: restoring only the trunk
            # would leave the head cold and silently halve draft quality
            # (0.906 -> 0.479 at depth 1). hidden rows are immutable mx
            # arrays, so only the cache needs copying.
            self._mtp_caches[req.request_id] = deepcopy(entry[2])
            self._mtp_hidden[req.request_id] = entry[3]
        req.next_token = base
        return [self._feed(req, base, self._tokenizer.decode([base]))]

    def _prefix_insert(self, req: RequestState, cache: list, base: int) -> None:
        """Snapshot a freshly prefilled prompt for future exact repeats."""
        if len(req.prompt) < self.config.prefix_cache_min_tokens:
            return
        key = tuple(req.prompt)
        if key in self._prefix:
            return
        mcache = self._mtp_caches.get(req.request_id)
        self._prefix[key] = (
            deepcopy(cache),
            base,
            deepcopy(mcache) if mcache is not None else None,
            self._mtp_hidden.get(req.request_id),
        )
        while len(self._prefix) > max(1, self.config.mlx_prefix_cache_size):
            self._prefix.popitem(last=False)

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
        from mlx_lm.models.cache import ArraysCache

        for c, (kind, val) in zip(cache, snap):
            if kind == "arrays":
                c.cache = list(val)
            else:
                c.offset = val

    def _prefill_owned(self, rid: int, tokens: Sequence[int]) -> object:
        """Prefill `tokens` into a fresh owned cache; return the last logits.

        Shared by `_start_spec` and `_refill` so the trunk cache and the MTP
        head cache can never drift apart -- every rebuild of one rebuilds the
        other, chunked identically.

        Position convention the head depends on: after this returns, the head
        cache holds pairs `(hidden[p], tokens[p+1])` for `p` in
        `0..len(tokens)-2`, and `_mtp_hidden[rid]` is the hidden row at
        `len(tokens)-1` -- exactly the row that pairs with the token the
        caller is about to make `next_token`. `propose` then appends that
        pair as position `len(tokens)-1` and trims it back.
        """
        mx = _mx()
        cache = self._make_cache()
        mtp = self._mtp
        mcache = mtp.make_cache() if mtp is not None else None
        logits = hidden = None
        for i in range(0, len(tokens), _PREFILL_CHUNK):
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
                    return hit
            mx = _mx()
            logits = self._prefill_owned(req.request_id, req.prompt)
            cache = self._caches[req.request_id]
            base = int(mx.argmax(logits[0, -1]).item())
            req.next_token = base
            out = [self._feed(req, base, self._tokenizer.decode([base]))]
            if self.config.mlx_prefix_cache:
                self._prefix_insert(req, cache, base)
            return out
        except BaseException:  # noqa: BLE001 - re-raised; must retire slot on any failure
            try:
                self._retire(req, "error")
            except BaseException:  # noqa: BLE001 - must never mask the real failure
                pass
            raise

    def _refill(self, req: RequestState, extra: Sequence[int] = ()) -> None:
        """Rebuild the cache exactly: re-feed prompt + accepted tokens.

        The mismatch path: the evaluated batch advanced the cache past the
        accepted prefix and ArraysCache cannot rewind, so replay from scratch.
        Exact by construction -- same tokens, same order, same positions.

        `extra` is the tokens this verify round accepted but has not fed yet.
        The caller runs `_feed` AFTER this, so `req.output_tokens` still lags
        by exactly those; without them the rebuilt cache is short by `matched`
        and every later step decodes against a truncated history (greedy
        speculative output then diverges from greedy AR -- the invariant the
        whole verify path exists to preserve). The engine-wide invariant is
        `cache == prompt + output_tokens[:-1]`, `next_token == output[-1]`;
        passing `accepted[:-1]` here is what keeps it true once `_feed` runs.
        """
        mx = _mx()
        tokens = req.prompt + req.output_tokens + list(extra)
        logits = self._prefill_owned(req.request_id, tokens)
        req.next_token = int(mx.argmax(logits[0, -1]).item())

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
        max_drafts = (
            self.config.spec_max_drafts
            if (self.config.speculative and self._spec_ok.get(req.request_id, True))
            else 0
        )
        allow = max(0, min(max_drafts, remaining - 1))
        base = req.next_token
        assert base is not None
        rid = req.request_id
        mtp = self._mtp
        if mtp is not None:
            # The head drafts from the trunk hidden row that pairs with
            # `base`; propose() trims its own speculative entries, so the
            # head cache still mirrors only confirmed tokens on return.
            allow = max(0, min(mtp.depth, remaining - 1))
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
        # Rewind point for the mismatch path (free: see _snapshot_cache).
        # Only worth taking when there is something to reject.
        snap = self._snapshot_cache(self._caches[rid]) if drafts else None
        msnap = (
            self._mtp_caches[rid][0].offset
            if (mtp is not None and drafts)
            else None
        )
        if mtp is not None:
            logits, hidden = mtp.trunk_forward(
                mx.array(rows)[None], self._caches[rid]
            )
        else:
            logits = self._model(
                mx.array(rows)[None], cache=self._caches[rid]
            )
        mx.eval(logits)
        self.spec_drafted += len(drafts)
        stats = self._spec_stats[req.request_id]
        stats[0] += len(drafts)
        accepted: list[int] = []
        matched = 0
        for k in range(len(rows)):
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
            # Mismatch: the cache advanced past the accepted prefix and
            # cannot rewind -- rebuild it exactly. Full matches skip this:
            # the cache already holds exactly the accepted stream.
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
            if mtp is not None:
                self._mtp_caches[rid][0].offset = msnap
                logits, hidden = mtp.trunk_forward(
                    mx.array(keep)[None], self._caches[rid]
                )
                mx.eval(logits)
                h_seq = h_prev
                if n:
                    h_seq = mx.concatenate([h_prev, hidden[:, :n, :]], axis=1)
                mtp.accept(self._mtp_caches[rid], h_seq, [base, *accepted[:n]])
                self._mtp_hidden[rid] = hidden[:, -1:, :]
            else:
                logits = self._model(
                    mx.array(keep)[None], cache=self._caches[rid]
                )
                mx.eval(logits)
        elif mtp is not None:
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
            if len(req.prompt) + req.n_generated >= self.config.n_ctx:
                piece = self._tokenizer.decode([token])
                outputs.append(self._retire(req, "context", token, piece))
                break
            outputs.append(self._feed(req, token, self._tokenizer.decode([token])))
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
                        cache, token, _entry = found
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
                        self._prefix_insert(req, cache, token)
                batch.join(req.request_id, cache)
                outputs.append(
                    self._feed(req, token, self._tokenizer.decode([token]))
                )
            except BaseException:  # noqa: BLE001 - re-raised; must not leave a half-joined row
                try:
                    self._retire(req, "error")
                except BaseException:  # noqa: BLE001 - never mask the real failure
                    pass
                raise

        if batch.empty:
            return outputs

        # Decode every live row together. next_token is what each row feeds.
        rows = list(batch.rows)
        tokens = []
        for rid in rows:
            req = self._states.get(rid)
            if req is None or req.next_token is None:
                # Retired between join and step: drop the row rather than
                # feeding a stale token into the shared forward.
                batch.leave([rid])
                continue
            tokens.append(req.next_token)
        rows = list(batch.rows)
        if not rows:
            return outputs

        logits = batch.step(tokens)
        for i, rid in enumerate(rows):
            req = self._states.get(rid)
            if req is None or req.finished:
                continue
            token = self._sample_row(req, logits[i, -1])
            outputs.append(
                self._feed(req, token, self._tokenizer.decode([token]))
            )
        return outputs

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

    def _step_stream(self, req: RequestState) -> list[StepOutput]:
        """Advance one request down the plain generator path."""
        stream = self._streams.get(req.request_id)
        if stream is None:
            self._start(req)
            stream = self._streams[req.request_id]
        try:
            chunk = next(stream)
        except StopIteration:
            # The generator ends early only at EOS (it swallows the token:
            # stream_generate breaks before yielding it), otherwise it runs
            # to max_tokens and our cap rule retires first. So an incomplete
            # count here IS a natural end, not an error.
            if req.n_generated >= req.params.max_tokens:
                return [self._retire(req, "length")]
            return [self._retire(req, "eog")]
        outputs = [self._feed(req, chunk.token, chunk.text)]
        if chunk.finish_reason in ("length", "stop") and not req.finished:
            # Belt-and-braces: our rules below retire first in every
            # reachable case (max_tokens == mlx's cap, EOS in our EOG
            # set); this only fires if mlx stops early for another
            # reason, and hanging would be worse than a wrong reason.
            outputs.append(self._retire(req, "length"))
        return outputs

    def _feed(self, req: RequestState, token: int, piece: str) -> StepOutput:
        """Account for one generated token: EOG, then stop, then cap."""
        self._decode_calls += 1
        if req.params.stop_at_eog and token in self._eog_ids():
            return self._retire(req, "eog", token, piece)
        req.output_tokens.append(token)
        req.n_generated += 1
        req.next_token = token
        stop_filter = self._stop_filters.get(req.request_id)
        if stop_filter is not None:
            _emittable, hit = stop_filter.push(piece)
            if hit:
                return self._retire(req, "stop_sequence")
        if req.n_generated >= req.params.max_tokens:
            return self._retire(req, "length")
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
