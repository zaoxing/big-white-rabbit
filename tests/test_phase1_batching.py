"""Phase 1 tests: N requests interleave through ONE llama_decode per step.

Point BWR_TEST_MODEL at a .gguf to run these; they skip otherwise so the suite stays
runnable on a machine with no models checked out.
"""

from __future__ import annotations

import math
import os

import pytest

import bwr as bwr
from bwr.engine import EngineConfig, MetalEngine, RequestParams

MODEL_PATH = os.environ.get("BWR_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_TEST_MODEL to a .gguf path to run bring-up tests",
)

# Three visibly different prompts. Cross-talk between sequences shows up as one
# sequence's continuation drifting toward another's topic, so the prompts are chosen
# to have nothing in common.
PROMPTS = [
    "The capital of France is",
    "Count: 1 2 3",
    "def add(a, b):",
]

# Fixed length, no EOG stop: every sequence must produce exactly N tokens, so the
# comparisons below are over full-length sequences rather than a shared early exit.
N_TOKENS = 8


def greedy(max_tokens: int = N_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False)


@pytest.fixture(scope="module")
def model() -> bwr.Model:
    return bwr.Model(MODEL_PATH, bwr.ModelParams())


def solo_tokens(model: bwr.Model, prompt: str, max_tokens: int = N_TOKENS) -> list[int]:
    """Greedy continuation of one prompt, alone in a FRESH context (n_seq_max=1)."""
    engine = MetalEngine(model, EngineConfig(n_ctx=512, n_seq_max=1))
    request_id = engine.add_request(prompt, greedy(max_tokens))
    list(engine.drain())
    return engine.tokens_of(request_id)


def run_batched(
    model: bwr.Model, prompts: list[str], max_tokens: int = N_TOKENS
) -> tuple[MetalEngine, list[int]]:
    engine = MetalEngine(model, EngineConfig(n_ctx=512, n_seq_max=4))
    ids = [engine.add_request(p, greedy(max_tokens)) for p in prompts]
    return engine, ids


def test_batched_matches_solo(model: bwr.Model) -> None:
    """The no-cross-talk proof: three sequences sharing every decode call must each
    produce exactly what that prompt produces alone in its own context. A shared
    llama_batch mixes positions and seq_ids in one attention pass, so a seq_id or
    position bug shows up here as one sequence contaminated by its neighbours."""
    solo = [solo_tokens(model, p) for p in PROMPTS]
    for tokens in solo:
        assert len(tokens) == N_TOKENS, "solo run did not produce a full-length sample"
    # The prompts must actually differ, or "equals solo" would be vacuous.
    assert len({tuple(t) for t in solo}) == len(PROMPTS)

    engine, ids = run_batched(model, PROMPTS)
    # All three are in flight together: they share the prefill decode and every
    # subsequent one.
    assert sum(1 for i in ids if not engine.state(i).finished) == 3
    list(engine.drain())

    for prompt, request_id, expected in zip(PROMPTS, ids, solo):
        got = engine.tokens_of(request_id)
        assert got == expected, (
            f"batched output for {prompt!r} diverged from its solo output: "
            f"{got} != {expected}"
        )


def test_one_decode_call_per_step(model: bwr.Model) -> None:
    """One step == one llama_decode, however many sequences are in flight. Asserted on
    the C++-side counter, never on timing."""
    engine, ids = run_batched(model, PROMPTS, max_tokens=6)
    assert len(ids) == 3

    steps = 0
    while engine.has_work:
        before = engine.ctx.decode_calls
        outputs = engine.step()
        assert engine.ctx.decode_calls == before + 1, (
            "a step must issue exactly one llama_decode, "
            f"got {engine.ctx.decode_calls - before}"
        )
        if steps == 0:
            # Three prompts prefilled and three tokens sampled out of that one decode.
            assert len(outputs) == 3
        steps += 1

    # 6 tokens per sequence, all three advancing in lockstep -> 6 decodes total, not 18.
    assert steps == 6
    assert engine.ctx.decode_calls == 6
    for request_id in ids:
        assert len(engine.tokens_of(request_id)) == 6


def test_cancel_does_not_disturb_others(model: bwr.Model) -> None:
    """Cancelling one sequence reclaims its slot without perturbing its peers: their
    tokens must match an otherwise identical run in which nobody was cancelled."""

    def run(cancel_after: int | None = None) -> list[list[int]]:
        engine, ids = run_batched(model, PROMPTS, max_tokens=N_TOKENS)
        steps = 0
        while engine.has_work:
            engine.step()
            steps += 1
            if cancel_after is not None and steps == cancel_after:
                assert engine.cancel(ids[0]) is True
        return [engine.tokens_of(i) for i in ids]

    baseline = run()
    cancelled = run(cancel_after=3)

    assert all(len(t) == N_TOKENS for t in baseline)
    # Peers are neither truncated nor corrupted.
    for idx in (1, 2):
        assert len(cancelled[idx]) == N_TOKENS, "a peer was truncated by the cancel"
        assert cancelled[idx] == baseline[idx], "a peer's output changed after a cancel"
    # The cancelled sequence stops where it was cancelled, keeping what it had.
    assert cancelled[0] == baseline[0][:3]


def test_seq_id_exhaustion_raises(model: bwr.Model) -> None:
    """Over-admission must raise, not abort: llama.cpp rejects a batch naming a
    seq_id >= n_seq_max, and the surrounding failure modes are GGML_ASSERT/abort()."""
    engine = MetalEngine(model, EngineConfig(n_ctx=256, n_seq_max=2))
    assert engine.ctx.n_seq_max == 2

    first = engine.add_request("The capital of France is", greedy(4))
    engine.add_request("Count: 1 2 3", greedy(4))
    assert engine.n_free_seq_slots == 0

    with pytest.raises(RuntimeError, match="sequence slots"):
        engine.add_request("def add(a, b):", greedy(4))

    # Same guard one level down: a hand-built batch naming an out-of-range seq_id is a
    # Python exception, not a dead process.
    rogue = bwr.Batch(4, 1)
    rogue.add(1, 0, 5, True)
    with pytest.raises(ValueError, match="n_seq_max"):
        engine.ctx.decode(rogue)

    # The engine survived both and slots recycle after a cancel.
    assert engine.step()
    assert engine.cancel(first) is True
    assert engine.n_free_seq_slots == 1
    third = engine.add_request("def add(a, b):", greedy(4))
    assert engine.step()
    assert engine.state(third).n_pos > 0


def test_engine_chunks_prefill_to_effective_n_batch(model: bwr.Model) -> None:
    """A prompt longer than the EFFECTIVE n_batch is admitted and prefilled in
    n_batch-sized chunks -- one decode per chunk, still one decode per step."""
    engine = MetalEngine(model, EngineConfig(n_ctx=1024, n_batch=64, n_ubatch=64, n_seq_max=2))
    n_batch = engine.ctx.n_batch
    assert n_batch <= engine.ctx.n_ctx, "n_batch must be read back post-clamp"

    prompt = "word " * 100
    n_prompt = len(model.tokenize(prompt))
    assert n_prompt > n_batch, "prompt must exceed n_batch for this test to mean anything"
    assert n_prompt < engine.ctx.n_ctx

    request_id = engine.add_request(prompt, greedy(3))
    expected_prefill_steps = math.ceil(n_prompt / n_batch)
    assert expected_prefill_steps >= 2

    steps = 0
    first_token_step = None
    while engine.has_work:
        before = engine.ctx.decode_calls
        outputs = engine.step()
        steps += 1
        assert engine.ctx.decode_calls == before + 1
        if outputs and first_token_step is None:
            first_token_step = steps

    # The first token can only appear once the last prompt chunk has been decoded.
    assert first_token_step == expected_prefill_steps
    assert len(engine.tokens_of(request_id)) == 3
    assert engine.text_of(request_id)


def test_batch_append_past_capacity_raises(model: bwr.Model) -> None:
    """Batch bounds-checks its own appends: llama_batch_init hands back raw arrays and
    nothing in the C API stops a write past n_tokens."""
    batch = bwr.Batch(2, 1)
    batch.add(1, 0, 0, False)
    batch.add(2, 1, 0, True)
    assert batch.n_tokens == 2
    with pytest.raises(ValueError, match="batch is full"):
        batch.add(3, 2, 0, True)
    batch.clear()
    assert batch.n_tokens == 0
    assert batch.add(3, 0, 0, True) == 0


def raw_context(
    model: bwr.Model,
    *,
    n_ctx: int = 256,
    n_batch: int = 64,
    n_seq_max: int = 2,
    kv_unified: bool = False,
) -> bwr.Context:
    """A bare Context: the KV-manipulation tests below drive seq_cp/seq_rm directly,
    below MetalEngine, because that is the layer whose guards they are checking."""
    cp = bwr.ContextParams()
    cp.n_ctx = n_ctx
    cp.n_batch = n_batch
    cp.n_ubatch = n_batch
    cp.n_seq_max = n_seq_max
    cp.kv_unified = kv_unified
    return bwr.Context(model, cp)


def prefill(ctx: bwr.Context, tokens: list[int], seq_id: int, start: int = 0) -> int:
    """Decode tokens[start:] into `seq_id` at their own positions and sample the next
    token from the (only) flagged row."""
    batch = bwr.Batch(ctx.n_batch, 1)
    last = len(tokens) - 1
    for pos in range(start, len(tokens)):
        batch.add(tokens[pos], pos, seq_id, pos == last)
    ctx.decode(batch)
    return ctx.sample_seq(seq_id, last - start)


def test_partial_seq_cp_without_kv_unified_raises(model: bwr.Model) -> None:
    """With per-sequence KV streams (llama.cpp's default), a cross-sequence copy of a
    PARTIAL position range hits GGML_ASSERT(is_full && "seq_cp() is only supported for
    full KV buffers") -> abort(). The binding must refuse it as an exception, and the
    process must still be usable afterwards."""
    tokens = list(model.tokenize(PROMPTS[0]))
    ctx = raw_context(model, n_seq_max=2)
    assert ctx.kv_unified is False
    ctx.set_seq_sampler(0, greedy().to_sampler_params())
    expected = prefill(ctx, tokens, seq_id=0)

    for p0, p1 in ((0, 3), (-1, 3), (2, -1)):
        with pytest.raises(ValueError, match="kv_unified"):
            ctx.memory_seq_cp(0, 1, p0, p1)

    # A partial range onto the SAME sequence never reaches the assert, so it stays legal.
    ctx.memory_seq_cp(0, 0, 0, 3)
    # Still alive and still holding seq 0's KV: the refused copies changed nothing.
    assert prefill(ctx, tokens + [expected], seq_id=0, start=len(tokens)) > 0


def test_partial_seq_cp_with_kv_unified_forks_a_prefix(model: bwr.Model) -> None:
    """The Phase 4 primitive: with a unified KV buffer a partial copy is legal and must
    actually transplant the prefix -- seq 1 gets [0, split) from seq 0, decodes only the
    remaining prompt tokens, and lands on the same greedy token as the full prefill."""
    tokens = list(model.tokenize(PROMPTS[0]))
    split = len(tokens) - 2
    assert 0 < split < len(tokens)

    ctx = raw_context(model, n_ctx=512, n_seq_max=3, kv_unified=True)
    assert ctx.kv_unified is True
    for seq_id in (0, 1, 2):
        ctx.set_seq_sampler(seq_id, greedy().to_sampler_params())

    expected = prefill(ctx, tokens, seq_id=0)
    ctx.memory_seq_cp(0, 1, 0, split)
    assert prefill(ctx, tokens, seq_id=1, start=split) == expected

    # Control: seq 2 got no copy. The identical suffix-only decode is accepted but,
    # attending over none of the prefix, must land somewhere else -- which is what makes
    # the equality above evidence that the partial copy really transplanted the prefix
    # rather than the suffix alone being enough to predict the token.
    assert prefill(ctx, tokens, seq_id=2, start=split) != expected


def test_full_range_seq_cp_forks_a_sequence(model: bwr.Model) -> None:
    """The full-range copy is the one cross-stream case llama.cpp implements, and it must
    keep working: seq 1 becomes a clone of seq 0 and continues identically."""
    tokens = list(model.tokenize(PROMPTS[0]))
    ctx = raw_context(model, n_seq_max=2)
    ctx.set_seq_sampler(0, greedy().to_sampler_params())
    ctx.set_seq_sampler(1, greedy().to_sampler_params())

    first = prefill(ctx, tokens, seq_id=0)
    ctx.memory_seq_cp(0, 1, -1, -1)

    # One batch, both sequences fed the same token at the same position. A clone must
    # produce the same continuation; had the copy been a no-op, seq 1's row would start
    # at a position with no KV under it and llama.cpp would reject the batch.
    batch = bwr.Batch(ctx.n_batch, 1)
    row0 = batch.add(first, len(tokens), 0, True)
    row1 = batch.add(first, len(tokens), 1, True)
    ctx.decode(batch)
    assert ctx.sample_seq(1, row1) == ctx.sample_seq(0, row0)


def test_memory_seq_rm_rejects_out_of_range_seq_id(model: bwr.Model) -> None:
    """seq_rm's assert exempts exactly -1 ("all sequences"), so -1 must still work while
    every other out-of-range id raises instead of aborting."""
    tokens = list(model.tokenize(PROMPTS[0]))
    ctx = raw_context(model, n_seq_max=2)
    ctx.set_seq_sampler(0, greedy().to_sampler_params())
    expected = prefill(ctx, tokens, seq_id=0)

    for bad in (9999, -2):
        with pytest.raises(ValueError, match="out of range"):
            ctx.memory_seq_rm(bad, -1, -1)
    # seq_keep has no wildcard: -1 asserts there, so it must be rejected.
    with pytest.raises(ValueError, match="out of range"):
        ctx.memory_seq_keep(-1)

    ctx.memory_seq_rm(-1, -1, -1)  # documented wildcard: clears every sequence
    # The wildcard really emptied seq 0 -- re-prefilling from position 0 would be a
    # decreasing-position error otherwise -- and reproduces the same greedy token.
    assert prefill(ctx, tokens, seq_id=0) == expected


def test_context_narrower_than_n_seq_max_raises(model: bwr.Model) -> None:
    """llama.cpp reserves one output row per sequence out of the EFFECTIVE n_batch and
    GGML_ASSERTs when they do not fit -- inside llama_init_from_model, before any handle
    exists -- so the geometry has to be refused up front."""
    with pytest.raises(ValueError, match="n_seq_max"):
        raw_context(model, n_ctx=2048, n_batch=8, n_seq_max=16)
    # n_batch is clamped DOWN to the requested n_ctx, so a wide n_batch does not save a
    # narrow context.
    with pytest.raises(ValueError, match="n_seq_max"):
        raw_context(model, n_ctx=4, n_batch=512, n_seq_max=8)
    with pytest.raises(ValueError, match="n_seq_max"):
        MetalEngine(model, EngineConfig(n_ctx=4096, n_batch=4, n_ubatch=4, n_seq_max=8))

    # The boundary case (one row per sequence, exactly) is legal and must still build.
    ctx = raw_context(model, n_ctx=1024, n_batch=8, n_seq_max=8)
    assert ctx.n_batch >= ctx.n_seq_max


def test_sampling_an_unflagged_row_raises(model: bwr.Model) -> None:
    """llama_sampler_sample GGML_ASSERTs on a row that produced no logits, so the
    binding refuses the row instead of letting the process die."""
    prompt = "The capital of France is"
    n_prompt = len(model.tokenize(prompt))
    assert n_prompt > 1

    engine = MetalEngine(model, EngineConfig(n_ctx=256, n_seq_max=1))
    engine.add_request(prompt, greedy(2))
    engine.step()

    # Only the last prompt row is flagged for output; the rest are pure KV fill.
    assert engine.ctx.row_has_logits(n_prompt - 1) is True
    assert engine.ctx.row_has_logits(0) is False
    with pytest.raises(ValueError, match="no logits"):
        engine.ctx.sample_seq(0, 0)
    with pytest.raises(ValueError, match="no logits"):
        engine.ctx.sample_seq(0, 999)


def test_sample_last_on_fresh_context_raises(model: bwr.Model) -> None:
    """sample_last() samples index -1 -- "the last OUTPUT row" -- which only resolves if
    the last decoded batch produced logits at all. On a context that has decoded nothing
    there is no such row, and llama_sampler_sample() would hit
    GGML_ASSERT(logits != nullptr) -> abort(), taking the whole process down. It must
    raise instead, and the context must stay usable afterwards."""
    ctx = raw_context(model, n_seq_max=1)
    assert ctx.decode_calls == 0
    assert ctx.any_row_has_logits is False
    with pytest.raises(ValueError, match="no logits"):
        ctx.sample_last()

    # Survived the refusal: a normal decode still works in this same process.
    ctx.decode_seq0(list(model.tokenize(PROMPTS[0])))
    assert isinstance(ctx.sample_last(), int)


def test_sample_last_after_unflagged_decode_raises(model: bwr.Model) -> None:
    """The state Phase 1 introduced: a real decode whose batch flagged NO row for output.
    Phase 0 could not reach it (llama_batch_get_one's null .logits always means "last
    token only", so decode_raw records a set flag), but decode(const Batch &) can, and
    it is exactly the engine's mid-chunked-prefill state. row_has_logits() already
    reports the truth; sample_last() must consult it rather than abort."""
    tokens = list(model.tokenize(PROMPTS[0]))
    assert len(tokens) > 1

    ctx = raw_context(model, n_seq_max=1)
    batch = bwr.Batch(ctx.n_batch, 1)
    for pos, token in enumerate(tokens):
        batch.add(token, pos, 0, False)
    ctx.decode(batch)
    assert ctx.decode_calls == 1, "the decode must really have happened"
    assert ctx.row_has_logits(len(tokens) - 1) is False
    assert ctx.any_row_has_logits is False
    with pytest.raises(ValueError, match="no logits"):
        ctx.sample_last()

    # An empty batch is a no-op decode, so the no-logits state persists and still raises.
    batch.clear()
    ctx.decode(batch)
    with pytest.raises(ValueError, match="no logits"):
        ctx.sample_last()

    # The context survived both refusals: flagging a row makes sample_last work again.
    batch.add(tokens[-1], len(tokens), 0, True)
    ctx.decode(batch)
    assert ctx.any_row_has_logits is True
    assert isinstance(ctx.sample_last(), int)


def test_engine_sample_last_mid_prefill_raises(model: bwr.Model) -> None:
    """The realistic reachable case: MetalEngine's _fill_batch flags only the FINAL
    prompt token, so every intermediate chunk of a chunked prefill leaves the context
    with zero logits rows. `Context` is re-exported and `MetalEngine.ctx` is public, so
    `engine.ctx.sample_last()` there is a one-liner that used to abort the process."""
    engine = MetalEngine(model, EngineConfig(n_ctx=1024, n_batch=64, n_ubatch=64, n_seq_max=2))
    n_prompt = len(model.tokenize("word " * 100))
    assert n_prompt > engine.ctx.n_batch, "prompt must span >1 chunk to reach this state"

    request_id = engine.add_request("word " * 100, greedy(2))
    outputs = engine.step()
    # Mid-prefill: the chunk decoded, but it asked for no logits, so no token came out.
    assert outputs == []
    assert engine.ctx.decode_calls == 1
    assert engine.ctx.any_row_has_logits is False
    with pytest.raises(ValueError, match="no logits"):
        engine.ctx.sample_last()

    # The engine is undisturbed and still finishes the request in this same process.
    list(engine.drain())
    assert len(engine.tokens_of(request_id)) == 2


def test_sample_last_after_decode_seq0_still_works(model: bwr.Model) -> None:
    """The Phase 0 path must be untouched by the guard: llama_batch_get_one leaves
    .logits null, which llama.cpp reads as "last token only", so a normal decode_seq0
    always leaves exactly one output row and sample_last() must return a token."""
    tokens = list(model.tokenize(PROMPTS[0]))
    ctx = raw_context(model, n_ctx=512, n_batch=256, n_seq_max=1)

    ctx.decode_seq0(tokens)
    assert ctx.any_row_has_logits is True
    first = ctx.sample_last()
    assert isinstance(first, int)
    ctx.accept(first)

    # And the incremental single-token continuation generate.py relies on.
    ctx.decode_seq0([first])
    second = ctx.sample_last()
    assert isinstance(second, int)


# --- per-sequence capacity (n_ctx_seq) vs context total (n_ctx) ----------------------
#
# With kv_unified=false llama.cpp divides the context between sequences
# (n_ctx_seq = n_ctx / n_seq_max) and then inflates the REPORTED n_ctx back to
# n_ctx_seq * n_seq_max. Budgeting a single request against n_ctx therefore overstates
# its room by a factor of n_seq_max, admitting prompts that only fail later mid-decode.


def test_n_ctx_seq_is_the_per_sequence_capacity(model: bwr.Model) -> None:
    """n_ctx_seq must differ from n_ctx exactly when the KV buffer is split, and match
    it when unified -- otherwise the accessor is not measuring what we think."""
    split = raw_context(model, n_ctx=512, n_batch=256, n_seq_max=2)
    assert split.kv_unified is False
    assert split.n_ctx_seq * split.n_seq_max == split.n_ctx
    assert split.n_ctx_seq < split.n_ctx, "split KV must give a sequence less than the total"

    unified = raw_context(model, n_ctx=512, n_batch=256, n_seq_max=2, kv_unified=True)
    assert unified.kv_unified is True
    assert unified.n_ctx_seq == unified.n_ctx, "a unified buffer is shared, not divided"


def test_engine_rejects_prompt_over_per_sequence_capacity(model: bwr.Model) -> None:
    """A prompt that fits ctx.n_ctx but NOT ctx.n_ctx_seq must be refused at admission.
    Before this fix it was admitted and died later inside decode with a KV-slot error."""
    engine = bwr.MetalEngine(
        model, bwr.EngineConfig(n_ctx=512, n_batch=512, n_ubatch=512, n_seq_max=2)
    )
    n_seq = engine.ctx.n_ctx_seq
    assert n_seq < engine.ctx.n_ctx, "geometry precondition: KV must be split"

    # Sits in the gap: longer than one sequence's room, shorter than the reported total.
    prompt = [1] * (n_seq + 10)
    assert len(prompt) < engine.ctx.n_ctx, "must be a prompt n_ctx would have allowed"

    with pytest.raises(ValueError, match="per-sequence capacity"):
        engine.add_request(prompt, greedy())

    # A prompt that genuinely fits is still admitted, and still generates.
    ok = engine.add_request([1] * (n_seq // 2), greedy(max_tokens=2))
    list(engine.drain())
    assert len(engine.tokens_of(ok)) == 2


# --- failed decode must not leave a claimable logits mask ---------------------------


def test_failed_decode_clears_the_logits_mask(model: bwr.Model) -> None:
    """A decode that fails must leave NO claimable logits.

    Two unsound behaviours are pinned here at once. Recording the mask before checking
    llama.cpp's return code left (a) rc=1 with a flagged row while n_outputs was stale,
    where only get_logits_ith's NDEBUG nullptr stood between the caller and an abort,
    and (b) a failed decode after a SUCCESSFUL one silently returning the previous
    decode's token -- a wrong answer reported as a good one.
    """
    tokens = list(model.tokenize(PROMPTS[0]))
    ctx = raw_context(model, n_ctx=256, n_batch=64, n_seq_max=2)
    n_seq = ctx.n_ctx_seq

    # (b) first: a good flagged decode, so logits genuinely exist.
    # The sampler must exist before prefill(), which samples as its last act.
    ctx.set_seq_sampler(0, bwr.SamplerParams())
    prefill(ctx, tokens, seq_id=0)
    assert ctx.any_row_has_logits is True
    good = ctx.sample_last()
    assert isinstance(good, int)

    # Now overrun sequence 0's own capacity so llama_decode fails (rc=1).
    batch = bwr.Batch(64, 1)
    pos = len(tokens)
    while pos < n_seq + 32:
        batch.clear()
        room = min(64, n_seq + 32 - pos)
        for k in range(room):
            batch.add(token=tokens[k % len(tokens)], pos=pos + k, seq_id=0,
                      logits=(k == room - 1))
        try:
            ctx.decode(batch)
        except (RuntimeError, ValueError):
            break
        pos += room
    else:
        pytest.fail("expected a decode failure once the sequence outgrew n_ctx_seq")

    # The mask is cleared, so nothing can claim the failed batch's (or the stale
    # previous) logits -- sampling raises instead of returning `good` again.
    assert ctx.any_row_has_logits is False
    with pytest.raises(ValueError, match="no logits"):
        ctx.sample_last()
    with pytest.raises(ValueError, match="no logits"):
        ctx.sample_seq(0, 0)
