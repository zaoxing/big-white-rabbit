"""MTP-head speculation for the MLX backend (SPEC-mlx-mtp-draft.md).

The model's own multi-token-prediction head proposes; the target verifies with
the SAME machinery as n-gram speculation (`_verify_rows` + refill).
Verification makes output identical regardless of draft quality -- a bad draft
only costs time, never correctness. Only the acceptance rate depends on the
head.

Why this exists: n-gram drafting accepts ~0.3 on general prose and so *costs*
2x on 27B MLX. The MTP head measures ~0.91 at depth 1 on code continuation,
against the same verify path.

Two invariants carry the whole design, and both were established by
measurement rather than from the runtime metadata (see the spec):

1. The head's KV cache mirrors the ACCEPTED token stream at identical
   positions -- prefilled over the prompt, then advanced by accepted tokens
   only. A fresh per-round cache drops depth-1 acceptance from 0.906 to
   0.479.
2. Proposal appends `depth` speculative entries to that cache; they MUST be
   trimmed back before the next round, or the cache drifts by `depth` per
   round. `KVCache.trim` is O(1) (an offset decrement), which is why
   rejection is cheap on the head even though the hybrid trunk has to refill.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

SIDECAR = "mtp.safetensors"
RUNTIME = "mtplx_runtime.json"
_PREFIX = "mtp."


def sidecar_path(model_path: str | Path) -> Path | None:
    """The head weights next to a weights dir, or None when absent."""
    p = Path(model_path) / SIDECAR
    return p if p.is_file() else None


def available(model_path: str | Path) -> bool:
    """Whether `model_path` carries an MTP head we can drive."""
    return sidecar_path(model_path) is not None


def read_contract(model_path: str | Path) -> dict[str, Any]:
    """The published MTP contract, with measured defaults for absent fields.

    `concat_order` is load-bearing (the wrong order scores 0/96);
    `base_hidden_variant` is not (an RMSNorm downstream washes out the
    difference) but is honoured anyway.
    """
    rt = Path(model_path) / RUNTIME
    contract: dict[str, Any] = {}
    if rt.is_file():
        try:
            contract = dict(json.loads(rt.read_text()).get("mtp_contract") or {})
        except (OSError, ValueError):
            contract = {}
    contract.setdefault("concat_order", "embedding_hidden")
    contract.setdefault("base_hidden_variant", "post_norm")
    contract.setdefault("mtp_quant_group_size", 64)
    contract.setdefault("mtp_quant_mode", "affine")
    return contract


def default_depth(model_path: str | Path, fallback: int = 3) -> int:
    """The checkpoint's own recommended draft depth, clamped to its max."""
    rt = Path(model_path) / RUNTIME
    if not rt.is_file():
        return fallback
    try:
        meta = json.loads(rt.read_text())
    except (OSError, ValueError):
        return fallback
    depth = int(meta.get("mtp_depth_default") or fallback)
    ceiling = int(meta.get("mtp_depth_max") or depth)
    return max(1, min(depth, ceiling))


def _full_attention_layer_idx(args: Any) -> int:
    """A layer_idx that qwen3_5.DecoderLayer builds as FULL attention.

    `(idx + 1) % interval == 0` picks the self-attention branch. A linear-
    attention layer would build a different module tree and fail the strict
    load below -- which is the guard, not a coincidence.
    """
    interval = int(getattr(args, "full_attention_interval", 4) or 4)
    return interval - 1


class MTPDraft:
    """The MTP head: load, prefill, propose. One instance per engine.

    Caches are per-request and owned by the caller (mirroring the engine's
    `_caches`), because the head's cache has to be rebuilt in lockstep with
    the trunk's on a refill.
    """

    def __init__(self, target: Any, model_path: str | Path, depth: int) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        self._mx = mx
        self.model_path = Path(model_path)
        self.contract = read_contract(model_path)
        self.depth = max(1, int(depth))
        self.hidden_variant = str(self.contract["base_hidden_variant"])
        self.concat_order = str(self.contract["concat_order"])
        if self.concat_order != "embedding_hidden":
            # Measured: the reversed order scores 0/96. Refuse rather than
            # silently draft garbage that verification will reject anyway.
            raise ValueError(
                f"unsupported MTP concat_order {self.concat_order!r} "
                "(only 'embedding_hidden' is implemented)"
            )
        self._target = target
        self._trunk = target.model
        self._module = self._build(target.args, nn)
        self._load(nn)

    # -- construction -----------------------------------------------------

    def _build(self, args: Any, nn: Any) -> Any:
        from mlx_lm.models.qwen3_5 import DecoderLayer

        hidden, eps = args.hidden_size, args.rms_norm_eps
        idx = _full_attention_layer_idx(args)

        class _MTP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pre_fc_norm_embedding = nn.RMSNorm(hidden, eps=eps)
                self.pre_fc_norm_hidden = nn.RMSNorm(hidden, eps=eps)
                self.fc = nn.Linear(hidden * 2, hidden, bias=False)
                self.layers = [DecoderLayer(args=args, layer_idx=idx)]
                self.norm = nn.RMSNorm(hidden, eps=eps)

        return _MTP()

    def _load(self, nn: Any) -> None:
        """Quantize to the sidecar's layout, then load it strictly.

        strict=True IS the shape check: the sidecar ships pre-quantized
        (.scales/.biases on fc/self_attn/mlp), so a mismatched group_size,
        bit width, or layer kind surfaces here rather than as silently wrong
        drafts.
        """
        path = sidecar_path(self.model_path)
        if path is None:
            raise FileNotFoundError(f"no {SIDECAR} in {self.model_path}")
        nn.quantize(
            self._module,
            group_size=int(self.contract["mtp_quant_group_size"]),
            bits=4,
            mode=str(self.contract["mtp_quant_mode"]),
        )
        raw = self._mx.load(str(path))
        weights = [
            (k[len(_PREFIX) :], v) for k, v in raw.items() if k.startswith(_PREFIX)
        ]
        if not weights:
            raise ValueError(f"{path} carries no {_PREFIX}* tensors")
        self._module.load_weights(weights, strict=True)
        self._mx.eval(self._module.parameters())
        self.n_tensors = len(weights)

    # -- trunk hidden states ---------------------------------------------

    def trunk_forward(self, ids: Any, cache: list) -> tuple[Any, Any]:
        """`(logits, hidden)` from one trunk pass.

        `Qwen3_5TextModel` applies its final norm before returning, so swap
        that norm for identity to expose the residual stream, then re-apply
        the real norm. Re-applying reproduces the unmodified trunk output
        exactly, so `logits` here is not an approximation of the normal path
        -- it IS the normal path.
        """
        trunk = self._trunk
        real_norm = trunk.norm
        try:
            trunk.norm = lambda x: x
            pre = trunk(ids, cache=cache)
        finally:
            trunk.norm = real_norm
        post = real_norm(pre)
        logits = self._target.lm_head(post)
        hidden = post if self.hidden_variant == "post_norm" else pre
        return logits, hidden

    # -- head cache -------------------------------------------------------

    def make_cache(self) -> list:
        from mlx_lm.models.cache import KVCache

        return [KVCache()]

    def _forward(
        self, hidden: Any, next_ids: Any, cache: list, want_logits: bool = True
    ) -> tuple[Any, Any]:
        """One head step. `want_logits=False` skips the output projection.

        That projection is over the FULL vocab (248320 x 5120 here), so it
        costs about as much as the trunk's own lm_head. The cache-advance
        paths (`prefill`, `accept`) never look at the logits, and paying for
        them there dominated the head's cost -- prefill would have run it
        once per prompt token.
        """
        from mlx_lm.models.base import create_attention_mask

        mx, m = self._mx, self._module
        e = m.pre_fc_norm_embedding(self._trunk.embed_tokens(next_ids))
        h = m.pre_fc_norm_hidden(hidden)
        mixed = m.fc(mx.concatenate([e, h], axis=-1))
        mask = create_attention_mask(mixed, cache)
        out = m.layers[0](mixed, mask=mask, cache=cache[0])
        logits = self._target.lm_head(m.norm(out)) if want_logits else None
        return logits, out

    def prefill(self, cache: list, hidden: Any, tokens: Sequence[int]) -> None:
        """Seed the head cache over the prompt: `(hidden[i], tokens[i+1])`.

        Worth 43 points of depth-1 acceptance (0.479 -> 0.906) versus
        starting the head cold, so this is not an optimisation -- skipping it
        is a correctness-grade regression in draft quality.
        """
        if len(tokens) < 2:
            return
        mx = self._mx
        self._forward(
            hidden[:, :-1, :],
            mx.array(list(tokens[1:]))[None],
            cache,
            want_logits=False,
        )
        mx.eval(cache[0].state)

    def propose(
        self, cache: list, hidden_last: Any, next_token: int, depth: int | None = None
    ) -> list[int]:
        """Draft up to `depth` tokens, leaving the cache exactly as found.

        The recursion feeds the head's own layer output back as the next
        hidden state. Each step appends one speculative entry to the cache;
        all of them are trimmed before returning, so the cache still mirrors
        only accepted tokens when the caller verifies.
        """
        n = self.depth if depth is None else int(depth)
        if n <= 0:
            return []
        mx = self._mx
        drafts: list[int] = []
        h, token = hidden_last, int(next_token)
        appended = 0
        try:
            for _ in range(n):
                logits, h = self._forward(h, mx.array([token])[None], cache)
                token = int(mx.argmax(logits[0, -1]).item())
                appended += 1
                drafts.append(token)
        finally:
            if appended:
                cache[0].trim(appended)
        return drafts

    def accept(self, cache: list, hidden: Any, tokens: Sequence[int]) -> None:
        """Advance the head cache by tokens the target actually accepted."""
        if not tokens:
            return
        mx = self._mx
        self._forward(hidden, mx.array(list(tokens))[None], cache, want_logits=False)
        mx.eval(cache[0].state)
