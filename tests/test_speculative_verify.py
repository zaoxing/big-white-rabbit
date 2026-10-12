"""Speculative verify path (SPEC-speculative-ngram.md, T3).

Needs BWR_TEST_MODEL like the other engine tests; skips without it (the
``needs_model`` mark -- the stubbed context-boundary test at the end runs
without weights).

The prompt is a long periodic digit run: its bigrams/trigrams seed the draft
table at admission, and a greedy continuation of a counting run repeats them,
so speculation actually engages (``spec_drafted > 0`` below is the tripwire
that fails if the prompt ever stops triggering drafts -- without it the
``<=`` comparisons would pass vacuously on the plain path).
"""

from __future__ import annotations

import os

import bwr
import pytest
from bwr.engine import EngineConfig, MetalEngine, RequestParams
from bwr.engine.batching import RequestState
from bwr.engine.config import StopSequenceFilter

MODEL_PATH = os.environ.get("BWR_TEST_MODEL")

needs_model = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set BWR_TEST_MODEL to a .gguf path to run speculation tests",
)

# Periodic at the token level (single digits tokenise singly): the table seeded
# from this predicts the continuation vindicated, and greedy decoding of a
# counting run keeps emitting the same n-grams.
PROMPT = "Count: " + " ".join(str(i % 10) for i in range(48))
N_TOKENS = 16


def greedy(max_tokens: int = N_TOKENS) -> RequestParams:
    return RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=False)


@pytest.fixture(scope="module")
def model() -> bwr.Model:
    return bwr.Model(MODEL_PATH, bwr.ModelParams())


def run(model: bwr.Model, speculative: bool) -> MetalEngine:
    config = EngineConfig(n_ctx=512, n_seq_max=1, speculative=speculative)
    engine = MetalEngine(model, config)
    engine.add_request(PROMPT, greedy())
    list(engine.drain())
    return engine


@needs_model
def test_spec_on_off_byte_identical(model: bwr.Model) -> None:
    """The invariant: speculation must be byte-identical to the plain path."""
    plain = run(model, False)
    spec = run(model, True)
    plain_id = plain._next_request_id - 1
    spec_id = spec._next_request_id - 1
    assert spec.tokens_of(spec_id) == plain.tokens_of(plain_id)
    assert spec.state(spec_id).finish_reason == plain.state(plain_id).finish_reason


@needs_model
def test_speculative_never_uses_more_decodes(model: bwr.Model) -> None:
    """Every verify step accepts >= 1 new token (a mismatch still yields the
    sampled token), so speculation cannot need more decodes for the same
    tokens. ``spec_drafted > 0`` proves drafts actually packed -- without it
    this comparison would pass on the plain path and prove nothing."""
    plain = run(model, False)
    spec = run(model, True)
    assert spec.spec_drafted > 0, "prompt did not trigger any drafts"
    assert spec.ctx.decode_calls <= plain.ctx.decode_calls
    rate = spec.spec_acceptance_rate
    assert rate is not None and 0.0 <= rate <= 1.0


@needs_model
def test_nongreedy_request_ignores_speculation(model: bwr.Model) -> None:
    """temp > 0 takes the plain path even with the flag on: no drafts packed,
    request still completes normally."""
    config = EngineConfig(n_ctx=512, n_seq_max=1, speculative=True)
    engine = MetalEngine(model, config)
    rid = engine.add_request(
        PROMPT, RequestParams(temp=1.0, max_tokens=8, stop_at_eog=False)
    )
    list(engine.drain())
    assert engine.spec_drafted == 0
    assert engine.spec_acceptance_rate is None
    assert len(engine.tokens_of(rid)) == 8
    assert engine.state(rid).finish_reason == "length"


@needs_model
def test_speculation_defaults_off(model: bwr.Model) -> None:
    """A default engine never drafts and reports no acceptance rate."""
    assert EngineConfig().speculative is False
    assert EngineConfig().spec_max_drafts == 4
    assert EngineConfig().engine == "mlx"
    engine = run(model, False)
    assert engine.spec_drafted == 0
    assert engine.spec_accepted == 0
    assert engine.spec_acceptance_rate is None


@needs_model
def test_negative_spec_max_drafts_rejected(model: bwr.Model) -> None:
    """Fail fast at construction, before allocating a context."""
    with pytest.raises(ValueError, match="spec_max_drafts"):
        MetalEngine(model, EngineConfig(spec_max_drafts=-1))


class _WrongTable:
    """A draft table whose every prediction is wrong on purpose.

    Built from a plain run's true trajectory: each observed context maps to
    (true_next + 1) % vocab, so no draft can ever match. Every verify step
    then takes the mismatch path -- rewind plus continue -- which is exactly
    the path a 100%-acceptance prompt never exercises (and where the
    inclusive-rewind bug hid).
    """

    order = 3

    def __init__(self, wrong: dict[tuple[int, ...], int]) -> None:
        self._wrong = wrong

    def update_stream(self, tokens) -> None:
        pass

    def predict(self, context, max_tokens: int) -> list[int]:
        token = self._wrong.get(tuple(context[-(self.order - 1) :]))
        return [token] * max_tokens if token is not None else []


@needs_model
def test_all_wrong_drafts_still_identical(
    model: bwr.Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Maximum adversity: drafts packed every step, zero accepted. Output must
    still equal the plain run, decode-for-decode (each step accepts exactly the
    one mismatched sample, i.e. the plain path's token)."""
    plain = run(model, False)
    plain_id = plain._next_request_id - 1
    expected = plain.tokens_of(plain_id)
    prompt_tokens = list(
        model.tokenize(PROMPT, add_special=True, parse_special=True)
    )
    hist = prompt_tokens + expected
    vocab = model.n_vocab
    wrong = {
        tuple(hist[max(0, i - 2) : i]): (tok + 1) % vocab
        for i, tok in enumerate(hist)
    }

    from bwr.engine import metal_engine as engine_module

    monkeypatch.setattr(
        engine_module, "NgramTable", lambda *args, **kwargs: _WrongTable(wrong)
    )
    config = EngineConfig(n_ctx=512, n_seq_max=1, speculative=True)
    engine = MetalEngine(model, config)
    rid = engine.add_request(PROMPT, greedy())
    list(engine.drain())

    assert engine.tokens_of(rid) == expected
    assert engine.state(rid).finish_reason == plain.state(plain_id).finish_reason
    assert engine.spec_drafted > 0, "scripted drafts did not pack"
    # Every step took the mismatch path (or near enough that at least one
    # rewind ran): fewer acceptances than drafts, and never more decodes.
    assert engine.spec_accepted < engine.spec_drafted
    assert engine.spec_acceptance_rate is not None
    assert engine.ctx.decode_calls <= plain.ctx.decode_calls


@needs_model
def test_speculative_engine_on_attention_needs_no_snapshots(
    model: bwr.Model,
) -> None:
    """The attention test model passes the capability gate either way, and its
    readback is 0: llama.cpp clamps n_rs_seq on architectures without
    recurrent state (snapshots only exist for recurrent memory), while partial
    removes on attention KV always succeed. Hybrid readback is covered by the
    27B validation runs (SPEC T5a evidence), not the fast suite."""
    spec = MetalEngine(model, EngineConfig(n_ctx=512, n_seq_max=1, speculative=True))
    assert spec.ctx.n_rs_seq == 0
    plain = MetalEngine(model, EngineConfig(n_ctx=512, n_seq_max=1))
    assert plain.ctx.n_rs_seq == 0


@needs_model
def test_spec_matches_plain_at_context_boundary(model: bwr.Model) -> None:
    """Run the counting prompt into the per-sequence context limit on both
    paths (n_ctx pads to 256; max_tokens never fires). The speculative run
    must retire with the same tokens and the same reason -- not one token
    early because a verify step checked each accepted token against the
    position AFTER it rather than its own."""
    params = RequestParams(temp=0.0, max_tokens=4000, stop_at_eog=False)
    plain = MetalEngine(model, EngineConfig(n_ctx=256, n_seq_max=1))
    plain_id = plain.add_request(PROMPT, params)
    list(plain.drain())
    spec = MetalEngine(model, EngineConfig(n_ctx=256, n_seq_max=1, speculative=True))
    spec_id = spec.add_request(PROMPT, params)
    list(spec.drain())
    assert plain.state(plain_id).finish_reason == "context"
    assert spec.spec_accepted > 0, "no draft matched: boundary never speculated"
    assert spec.tokens_of(spec_id) == plain.tokens_of(plain_id)
    assert spec.state(spec_id).finish_reason == "context"


class _ScriptedCtx:
    """Just enough context for `_verify_rows`: a fixed per-row sample."""

    n_ctx_seq = 10

    def __init__(self, samples: dict[int, int]) -> None:
        self._samples = samples

    def sample_seq(self, seq_id: int, row: int) -> int:
        return self._samples[row]

    def memory_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        return True

    def reset_seq_sampler(self, seq_id: int) -> None:
        pass


class _PlainModel:
    def is_eog(self, token: int) -> bool:
        return False

    def token_to_piece(self, token: int) -> str:
        return str(token)


def _stub_engine(ctx: _ScriptedCtx) -> MetalEngine:
    eng = MetalEngine.__new__(MetalEngine)
    eng.ctx = ctx
    eng.model = _PlainModel()
    eng.draft = None
    eng._stop_filters = {}
    eng._spec_tables = {}
    eng._states = {}
    eng._retired = {}
    eng._retired_limit = 8
    eng._prefix_cache_active = False
    eng._free_seq_ids = []
    eng.spec_drafted = 0
    eng.spec_accepted = 0
    return eng


def _generating_request(rid: int, n_pos: int) -> RequestState:
    req = RequestState(
        request_id=rid, seq_id=0, prompt=[1, 2, 3],
        params=RequestParams(temp=0.0, max_tokens=4000, stop_at_eog=False),
    )
    req.n_prefilled = 3
    req.n_pos = n_pos
    req.next_token = 5
    return req


def test_verify_checks_each_token_at_its_own_position() -> None:
    """n_ctx_seq=10, base packed at 7, two matched drafts: the accepted tokens
    fill cells 8, 9, 10 and the plain path (which checks each sampled token
    against the position it will be packed at) emits all three, retiring on
    the third with "context". The verify path must emit exactly the same."""
    from bwr.engine.metal_engine import _VerifyEntry

    base, d0, d1 = 7, 11, 12
    bonus = 13
    # Row 0 continues the base token -> d0; row 1 continues d0 -> d1; row 2
    # continues d1 -> the bonus token.
    ctx = _ScriptedCtx({0: d0, 1: d1, 2: bonus})
    plain = _stub_engine(ctx)
    plain_req = _generating_request(0, base)
    plain._states[0] = plain_req
    plain_outputs = []
    for token in (d0, d1, bonus):
        # One plain step: pack at n_pos, commit n_pos + 1, then account.
        plain_req.n_pos += 1
        plain_outputs.append(plain._advance_token(plain_req, token))
        if plain_outputs[-1].finished:
            break
    assert [o.token for o in plain_outputs] == [d0, d1, bonus]
    assert plain_outputs[-1].finish_reason == "context"

    spec = _stub_engine(ctx)
    spec_req = _generating_request(1, base)
    spec._states[1] = spec_req
    # The step committed n_pos past every packed row (base + 2 drafts).
    spec_req.n_pos = base + 3
    entry = _VerifyEntry(spec_req, base_row=0, draft_rows=[1, 2], drafts=[d0, d1], pos_base=base)
    outputs, accepted, matched = spec._verify_rows(entry)
    assert (accepted, matched) == ([d0, d1, bonus], 2)
    assert [(o.token, o.finished, o.finish_reason) for o in outputs] == [
        (o.token, o.finished, o.finish_reason) for o in plain_outputs
    ]


class _StepCtx(_ScriptedCtx):
    """`_ScriptedCtx` plus what `step` itself touches: the batch size and a
    decode that just records the rows it was handed."""

    n_batch = 8
    n_ctx_seq = 64

    def __init__(self, samples: dict[int, int]) -> None:
        super().__init__(samples)
        self.decoded: list[list[tuple[int, int, int, bool]]] = []

    def decode(self, batch: _FakeBatch) -> None:
        self.decoded.append(list(batch.rows))


class _FakeBatch:
    """Row-recording stand-in for `Batch`: `add` returns the row index."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, int, int, bool]] = []

    def clear(self) -> None:
        self.rows = []

    def add(self, token: int, pos: int, seq_id: int, want: bool) -> int:
        self.rows.append((token, pos, seq_id, want))
        return len(self.rows) - 1


class _FakeDraft:
    """A draft model that proposes a fixed list and records every call."""

    def __init__(self, proposals: list[int]) -> None:
        self.proposals = proposals
        self.calls: list[tuple] = []

    def propose(self, request_id: int, n: int) -> list[int]:
        self.calls.append(("propose", request_id, n))
        return list(self.proposals)[:n]

    def sync(self, request_id: int, pos: int, tail: list[int]) -> None:
        self.calls.append(("sync", request_id, pos, list(tail)))

    def release(self, request_id: int) -> None:
        self.calls.append(("release", request_id))


def _step_engine(ctx: _StepCtx, draft: _FakeDraft) -> MetalEngine:
    """`_stub_engine` plus the attributes the real `step` reads."""
    from bwr.engine.batching import FCFSPolicy

    eng = _stub_engine(ctx)
    eng.draft = draft
    eng.config = EngineConfig(draft_max_drafts=2)
    eng.policy = FCFSPolicy()
    eng._batch = _FakeBatch()
    eng._hotlist = None
    eng._pins = {}
    return eng


def test_plain_step_syncs_the_sampled_token_into_the_draft() -> None:
    """A step that packed no drafts samples a token the draft never saw. It
    is the base token the target packs at `req.n_pos` next step, so `step`
    must hand it to the draft at that same cell before the next propose;
    otherwise the draft runs one cell behind and every proposal compares
    against the wrong target row."""
    ctx = _StepCtx({0: 42})
    draft = _FakeDraft(proposals=[])
    eng = _step_engine(ctx, draft)
    req = _generating_request(7, n_pos=3)
    eng._states[7] = req
    outputs = eng.step()
    assert [o.token for o in outputs] == [42]
    assert req.n_pos == 4
    assert draft.calls == [("propose", 7, 2), ("sync", 7, 4, [42])], draft.calls


def test_verify_step_syncs_only_the_newly_confirmed_tail() -> None:
    """Two matched drafts, row 2 is the bonus: the draft already holds the
    matched tokens at pos_base+1 and pos_base+2, so the sync carries only
    the bonus token at pos_base+1+matched -- not the whole accepted run."""
    pos_base = 3
    ctx = _StepCtx({0: 11, 1: 12, 2: 13})
    draft = _FakeDraft(proposals=[11, 12])
    eng = _step_engine(ctx, draft)
    req = _generating_request(7, n_pos=pos_base)
    eng._states[7] = req
    outputs = eng.step()
    assert [o.token for o in outputs] == [11, 12, 13]
    assert draft.calls == [
        ("propose", 7, 2),
        ("sync", 7, pos_base + 1 + 2, [13]),
    ], draft.calls


class _ByteModel:
    """A byte-fallback vocabulary: tokens below 256 are that single byte, so
    a multi-byte character is spelled over several tokens; 300 is EOG."""

    EOG = 300

    def is_eog(self, token: int) -> bool:
        return token == self.EOG

    def token_to_piece(self, token: int) -> str:
        # Never the right answer when `token_to_piece_bytes` exists: a lone
        # lead byte repaired on its own is U+FFFD, which is the bug under test.
        return "?"

    def token_to_piece_bytes(self, token: int) -> bytes:
        return bytes([token]) if token < 256 else b""

    def detokenize(self, tokens: list[int]) -> str:
        return b"".join(self.token_to_piece_bytes(t) for t in tokens).decode(errors="replace")


# "é" is C3 A9: a lead byte and one continuation byte.
_LEAD, _CONT = 0xC3, 0xA9


def _byte_engine(max_tokens: int = 4000, stop: list[str] | None = None) -> tuple[MetalEngine, RequestState]:
    eng = _stub_engine(_ScriptedCtx({}))
    eng.model = _ByteModel()
    req = _generating_request(0, 3)
    req.params = RequestParams(temp=0.0, max_tokens=max_tokens, stop_at_eog=True, stop=stop or [])
    eng._states[0] = req
    if stop:
        eng._stop_filters[0] = StopSequenceFilter(stop)
    return eng, req


def test_lead_byte_is_held_until_its_continuation_arrives() -> None:
    """The lead byte alone adds no text; the continuation emits the whole
    character. Before the incremental detokenizer each token was repaired on
    its own and the stream carried U+FFFD for the lead byte."""
    eng, req = _byte_engine()
    first = eng._advance_token(req, _LEAD)
    second = eng._advance_token(req, _CONT)
    assert (first.piece, first.finished) == ("", False)
    assert (second.piece, second.finished) == ("é", False)
    assert eng.text_of(0) == "é"


def test_length_retire_carries_the_held_partial_character() -> None:
    """A cap that lands on a lead byte: the retire piece carries the flushed
    replacement character, so joined pieces still equal text_of."""
    eng, req = _byte_engine(max_tokens=2)
    outs = [eng._advance_token(req, 0x48), eng._advance_token(req, _LEAD)]
    assert [o.finished for o in outs] == [False, True]
    assert outs[-1].finish_reason == "length"
    assert outs[-1].piece == "�"
    assert "".join(o.piece for o in outs) == eng.text_of(0) == "H�"


def test_eog_retire_carries_the_held_partial_character_and_no_eog_token() -> None:
    """EOG right after a lead byte: the eog output carries the flushed
    replacement (the control token itself adds no text) and the eog token is
    not counted in tokens_of."""
    eng, req = _byte_engine()
    outs = [eng._advance_token(req, t) for t in (0x48, _LEAD, _ByteModel.EOG)]
    assert outs[-1].finished and outs[-1].finish_reason == "eog"
    assert outs[-1].piece == "�"
    assert eng.tokens_of(0) == [0x48, _LEAD]
    assert "".join(o.piece for o in outs) == eng.text_of(0) == "H�"


def test_stop_sequence_completed_by_a_multibyte_character_retires() -> None:
    """The stop filter sees whole characters: "é" spelled over two tokens
    completes the delimiter on the continuation byte, and the request retires
    with "stop_sequence" rather than decoding on past it."""
    eng, req = _byte_engine(stop=["é"])
    first = eng._advance_token(req, _LEAD)
    second = eng._advance_token(req, _CONT)
    assert not first.finished
    assert second.finished and second.finish_reason == "stop_sequence"
    assert second.piece == "é"
    assert eng.text_of(0) == "é"
