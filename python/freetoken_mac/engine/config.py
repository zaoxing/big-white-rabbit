"""Engine and per-request configuration.

``EngineConfig`` shapes the ``llama_context``; ``RequestParams`` carries one
request's sampling settings and stop conditions. Sampling is per request because
each sequence owns its own sampler chain in the C++ context.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def to_sampler_params(self) -> SamplerParams:
        sp = SamplerParams()
        sp.temp = self.temp
        sp.top_k = self.top_k
        sp.top_p = self.top_p
        sp.seed = self.seed
        return sp
