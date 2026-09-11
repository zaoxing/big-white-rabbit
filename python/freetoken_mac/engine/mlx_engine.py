"""MLXEngine: mlx-lm behind the MetalEngine interface (SPEC-mlx-engine.md).

Phase 1 is single-stream plain decode only: no batching across requests (each
request owns an independent mlx generator; the worker thread drives them one
token per step), no speculation, no prefix cache. The point is interface
parity -- AsyncEngine, the routes, and the tests below treat this exactly
like MetalEngine -- so the serving surface survives a backend swap.

mlx-lm is a core dependency (it ships with freetoken-mac). Nothing in this
module imports it at top level anyway: the import happens in ``__init__``
so a broken install fails with a clear error naming the backend instead of
at ``import freetoken_mac`` time.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from .batching import RequestState
from .config import EngineConfig, RequestParams, StopSequenceFilter
from .metal_engine import MIN_RETAINED_FINISHED, StepOutput


def _require_mlx():
    """Import mlx-lm lazily, with a clear error naming the backend."""
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise ImportError(
            "the mlx backend needs mlx/mlx-lm installed "
            "(pip install freetoken-mac)"
        ) from exc
    return load, make_sampler


class _MLXContext:
    """Teardown handle satisfying AsyncEngine.stop's ``engine.ctx.close()``.

    Releasing weights on Apple Silicon is load-bearing (same abort class as
    Context::close documents): drop every reference and clear the Metal-side
    caches instead of waiting for the collector.
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
        load, _ = _require_mlx()
        self._model, self._tokenizer = load(model_path)
        self.model_path = model_path
        self.ctx = _MLXContext(self)
        self._streams: dict[int, Iterator] = {}
        self._stop_filters: dict[int, StopSequenceFilter] = {}
        self._states: dict[int, RequestState] = {}
        self._retired: dict[int, RequestState] = {}
        self._retired_limit = max(8, MIN_RETAINED_FINISHED)
        self._next_request_id = 0

    # --- helpers ------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self.ctx.closed:
            raise RuntimeError(
                "this MLXEngine has been closed; create a new one to keep serving"
            )

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
        self._states.pop(req.request_id, None)
        self._retired[req.request_id] = req
        while len(self._retired) > self._retired_limit:
            del self._retired[next(iter(self._retired))]
        return StepOutput(req.request_id, token, piece, True, reason)

    def _eog_ids(self) -> set[int]:
        eos = getattr(self._tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return set(eos)

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
        """Advance every in-flight request by one token."""
        self._ensure_open()
        outputs: list[StepOutput] = []
        for req in list(self._states.values()):
            if req.finished:
                continue
            stream = self._streams.get(req.request_id)
            if stream is None:
                self._start(req)
                stream = self._streams[req.request_id]
            try:
                chunk = next(stream)
            except StopIteration:
                # Generator spent without our rules firing: the request made
                # its cap exactly (mlx stops after max_tokens) or something
                # odd happened. Length iff we produced the full cap.
                if req.n_generated >= req.params.max_tokens:
                    outputs.append(self._retire(req, "length"))
                else:
                    outputs.append(self._retire(req, "error"))
                continue
            outputs.append(self._feed(req, chunk.token, chunk.text))
            if chunk.finish_reason in ("length", "stop") and not req.finished:
                # Belt-and-braces: our rules below retire first in every
                # reachable case (max_tokens == mlx's cap, EOS in our EOG
                # set); this only fires if mlx stops early for another
                # reason, and hanging would be worse than a wrong reason.
                outputs.append(self._retire(req, "length"))
        return outputs

    def _feed(self, req: RequestState, token: int, piece: str) -> StepOutput:
        """Account for one generated token: EOG, then stop, then cap."""
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
