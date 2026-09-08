"""Single-sequence greedy/sampled generation loop.

This is Phase 0 scaffolding whose only job is to prove the binding works end to end.
Phase 1 replaces it with ``engine.MetalEngine``, which drives one ``llama_batch`` per
step across many sequences instead of one token at a time down a single sequence.
"""

from __future__ import annotations

from typing import Iterator

from ._freetoken_metal import Context, Model


def generate(
    model: Model,
    ctx: Context,
    prompt: str,
    *,
    max_tokens: int = 128,
    stop_at_eog: bool = True,
) -> Iterator[str]:
    """Yield generated text pieces for ``prompt``, one decoded token at a time."""
    tokens = model.tokenize(prompt, add_special=True, parse_special=True)
    if not tokens:
        raise ValueError("prompt tokenized to zero tokens")

    # Budget against the EFFECTIVE geometry: llama.cpp rounds n_ctx up and clamps
    # n_batch down to the requested n_ctx, so neither matches what was asked for.
    n_ctx = ctx.n_ctx
    n_batch = ctx.n_batch
    if len(tokens) >= n_ctx:
        raise ValueError(f"prompt is {len(tokens)} tokens, context is {n_ctx}")

    # Prefill in n_batch-sized chunks; a single decode call may not exceed n_batch.
    for i in range(0, len(tokens), n_batch):
        ctx.decode_seq0(tokens[i : i + n_batch])

    n_used = len(tokens)
    for _ in range(max_tokens):
        tok = ctx.sample_last()
        if stop_at_eog and model.is_eog(tok):
            return
        ctx.accept(tok)
        yield model.token_to_piece(tok)

        n_used += 1
        if n_used >= n_ctx:
            return
        # Decode the sampled token to produce the next step's logits.
        ctx.decode_seq0([tok])
