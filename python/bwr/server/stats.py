"""Server-wide request counters behind `/admin/api/stats`.

The macOS app's `StatsDTO` declares `total_tokens_served`, `total_requests`,
`avg_prefill_tps`, `avg_generation_tps` and friends as NON-optional. Swift
`Decodable` fails the whole value on one missing key, so a stats response
without them does not render a partial Status screen -- it renders none of
it. These counters exist to satisfy that contract with real numbers.

Scope and honesty
-----------------
Everything here lives in the serving process and starts at zero when it does.
bwr keeps no stats database, so "all time" cannot mean more than "since this
server started"; `snapshot()` says so in `persisted: False` rather than
implying a history it does not have. That is also why `clear()` and the
all-time clear behave identically -- there is only one set of numbers to
reset.

Timing splits prefill from decode at the FIRST token: the wait for token one
covers prompt processing, everything after it is generation. A prompt served
from the prefix cache does no prefill work at all, so it contributes tokens
to `total_cached_tokens` and no time to the prefill average -- averaging it
in would drag the reported prefill rate toward infinity on cache hits.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServerStats:
    started_at: float = field(default_factory=time.time)

    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0

    # Separate token/second pairs per phase: a global tokens/elapsed ratio
    # would mix a 4k-token prefill with a 20-token generation and report
    # neither.
    _prefill_tokens: int = 0
    _prefill_seconds: float = 0.0
    _decode_tokens: int = 0
    _decode_seconds: float = 0.0

    # Per-model totals, so the Usage screen can break the numbers down the
    # way it is built to. Same fields as the global counters; kept here
    # rather than derived later because a request knows which model served
    # it and nothing downstream does.
    _by_model: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        prefill_seconds: float | None = None,
        decode_seconds: float | None = None,
        model: str | None = None,
    ) -> None:
        """Fold one finished request in. Never raises: a bad count must not
        fail the response that produced it."""
        with self._lock:
            self.total_requests += 1
            self.total_prompt_tokens += max(0, prompt_tokens)
            self.total_completion_tokens += max(0, completion_tokens)
            self.total_cached_tokens += max(0, cached_tokens)
            # A cached prompt did no prefill; see the module docstring.
            if prefill_seconds and prefill_seconds > 0 and cached_tokens <= 0:
                self._prefill_tokens += max(0, prompt_tokens)
                self._prefill_seconds += prefill_seconds
            if decode_seconds and decode_seconds > 0:
                self._decode_tokens += max(0, completion_tokens)
                self._decode_seconds += decode_seconds
            if model:
                m = self._by_model.setdefault(
                    model,
                    {"requests": 0, "prompt": 0, "completion": 0,
                     "cached": 0, "decode_tokens": 0, "decode_seconds": 0.0},
                )
                m["requests"] += 1
                m["prompt"] += max(0, prompt_tokens)
                m["completion"] += max(0, completion_tokens)
                m["cached"] += max(0, cached_tokens)
                if decode_seconds and decode_seconds > 0:
                    m["decode_tokens"] += max(0, completion_tokens)
                    m["decode_seconds"] += decode_seconds

    def clear(self) -> None:
        with self._lock:
            self.total_requests = 0
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cached_tokens = 0
            self._prefill_tokens = 0
            self._prefill_seconds = 0.0
            self._decode_tokens = 0
            self._decode_seconds = 0.0
            self._by_model.clear()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            prompt = self.total_prompt_tokens
            cached = self.total_cached_tokens
            return {
                "total_requests": self.total_requests,
                "total_prompt_tokens": prompt,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens_served": prompt + self.total_completion_tokens,
                "total_cached_tokens": cached,
                # Ratio of prompt tokens that never hit the model. Zero
                # prompt tokens means zero efficiency, not a division error.
                "cache_efficiency": (cached / prompt) if prompt else 0.0,
                "avg_prefill_tps": (
                    self._prefill_tokens / self._prefill_seconds
                    if self._prefill_seconds > 0 else 0.0
                ),
                "avg_generation_tps": (
                    self._decode_tokens / self._decode_seconds
                    if self._decode_seconds > 0 else 0.0
                ),
                "uptime_seconds": time.time() - self.started_at,
                # No stats database: these numbers begin at process start.
                "persisted": False,
            }

    def usage(self) -> dict[str, Any]:
        """`/admin/api/usage` in the shape UsageHistoryDTO decodes.

        `available` is True because the totals are real. `heatmap` is empty
        because they are not a TIME SERIES -- bwr records no per-day history,
        and inventing buckets would be worse than an empty chart beside
        correct totals.
        """
        def totals(requests: int, prompt: int, completion: int, cached: int,
                   decode_tokens: float, decode_seconds: float,
                   model_id: str | None) -> dict[str, Any]:
            return {
                "model_id": model_id,
                "requests": int(requests),
                "total_tokens": int(prompt + completion),
                "prompt_tokens": int(prompt),
                "completion_tokens": int(completion),
                "cached_tokens": int(cached),
                "generation_tps": (
                    decode_tokens / decode_seconds if decode_seconds > 0 else None
                ),
                "cache_efficiency": (cached / prompt) if prompt else 0.0,
            }

        with self._lock:
            return {
                "enabled": True,
                "available": True,
                # bwr admits every request it accepts; there is no queue it
                # can overflow, so nothing is ever dropped for capacity.
                "dropped_requests": 0,
                "totals": totals(
                    self.total_requests, self.total_prompt_tokens,
                    self.total_completion_tokens, self.total_cached_tokens,
                    self._decode_tokens, self._decode_seconds, None,
                ),
                "models": [
                    totals(m["requests"], m["prompt"], m["completion"],
                           m["cached"], m["decode_tokens"], m["decode_seconds"],
                           mid)
                    for mid, m in sorted(self._by_model.items())
                ],
                "heatmap": [],
            }
