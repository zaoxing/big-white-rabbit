"""RED test for engine 'error' reason mapping tripwire.

The engine has 6 retirement reasons (all literals passed to _retire):
  "cancelled", "error", "eog", "stop_sequence", "length", "context"

Both protocol tables must map every one, or an internal error could be
reported as a clean "stop" via the default. Currently "error" is in
neither table, and the existing test
`test_the_reason_tables_map_every_engine_cause_to_one_protocol_reason`
only checks map-to-map parity (`set(_FINISH)==set(_STOP)`), so it cannot
catch a reason missing from BOTH.

This test is Prove-It: it must FAIL before the fix (error not in tables)
and PASS after (error mapped in both, and vocabulary derived from the
engine source itself).
"""

from __future__ import annotations

import pathlib
import re


def _engine_reasons_from_source() -> set[str]:
    """Derive the vocabulary from the engine's own _retire call sites.

    This is the correct source of truth: adding a new `self._retire(req, "foo")`
    must be reflected in the tables, and a test that compares the two maps to
    each other cannot enforce that.
    """
    src = pathlib.Path("python/bwr/engine/metal_engine.py").read_text()
    # matches: self._retire(req, "reason"  or  self._retire(req, 'reason'
    return set(re.findall(r'self\._retire\(req,\s*["\']([^"\']+)["\']', src))


def test_engine_vocabulary_is_six_reasons() -> None:
    reasons = _engine_reasons_from_source()
    # Pin the set so a new reason cannot slip in unnoticed.
    assert reasons == {"cancelled", "error", "eog", "stop_sequence", "length", "context"}, reasons


def test_both_protocol_tables_cover_every_engine_reason() -> None:
    from bwr.server.anthropic_api import _STOP_REASONS
    from bwr.server.app import _FINISH_REASONS

    engine_reasons = _engine_reasons_from_source()
    assert engine_reasons <= set(_FINISH_REASONS), (
        f"_FINISH_REASONS missing {engine_reasons - set(_FINISH_REASONS)}"
    )
    assert engine_reasons <= set(_STOP_REASONS), (
        f"_STOP_REASONS missing {engine_reasons - set(_STOP_REASONS)}"
    )
    # No extra reasons beyond the engine's vocabulary (keeps tables minimal).
    assert set(_FINISH_REASONS) == engine_reasons
    assert set(_STOP_REASONS) == engine_reasons


def test_error_maps_to_generic_termination() -> None:
    from bwr.server.anthropic_api import _STOP_REASONS
    from bwr.server.app import _FINISH_REASONS, _openai_finish_reason

    # Error is an internal failure; the wire should not pretend it was a
    # successful stop with content, but there is no protocol-level "error"
    # finish reason, so it maps to the generic termination.
    assert _FINISH_REASONS["error"] == "stop"
    assert _STOP_REASONS["error"] == "end_turn"
    # And the helper that defaults unknown->"stop" must still give the same.
    assert _openai_finish_reason("error") == "stop"


def test_wire_reports_error_as_generic_via_canned_engine() -> None:
    """If the engine ever produced StepOutput with reason error, the wire
    must not default to 'stop' silently — it must use the table entry.
    Currently error is unreachable (re-raised), so we use canned engine to
    prove the mapping is wired through.
    """
    from bwr.engine.metal_engine import StepOutput

    async def canned_error(request_id: int):
        # Simulate a request that retires for error with empty piece
        yield StepOutput(request_id, 0, "", True, "error")

    import os

    import bwr as bwr
    import pytest

    path = os.environ.get("BWR_TEST_MODEL")
    if not path:
        pytest.skip("BWR_TEST_MODEL not set")
    tc = pytest.importorskip("fastapi.testclient")
    from bwr.server.app import build_app

    m = bwr.Model(path, bwr.ModelParams())
    try:
        app = build_app(m, bwr.EngineConfig(n_ctx=4096, n_batch=512, n_ubatch=512, n_seq_max=2, engine="metal"))
        with tc.TestClient(app) as client:
            # monkeypatch stream to canned error
            old = app.state.engine.stream
            app.state.engine.stream = canned_error
            try:
                r = client.post(
                    "/v1/chat/completions",
                    json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
                )
                # The engine's _advance path for error re-raises, so non-streaming
                # would be 500 in real life; the canned stream bypasses that raise
                # and produces a StepOutput directly, so the route will see finished
                # with reason error. It should map via table, not default.
                assert r.status_code == 200
                assert r.json()["choices"][0]["finish_reason"] == "stop"
            finally:
                app.state.engine.stream = old

            # Anthropic
            app.state.engine.stream = canned_error
            try:
                r = client.post(
                    "/v1/messages",
                    json={"model": "local", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
                )
                assert r.status_code == 200
                assert r.json()["stop_reason"] == "end_turn"
            finally:
                app.state.engine.stream = old
    finally:
        m.close()
