"""Shared mapping from engine retirement reasons to protocol stop reasons.

The engine retires for one of six reasons (see `engine/metal_engine.py:_retire`
call sites): ``eog``, ``length``, ``context``, ``cancelled``,
``stop_sequence``, ``error``. Both protocol surfaces must map every one,
and the two tables must agree on which causes exist -- a new engine reason
added to one table and not the other is a response whose text and reason
disagree.

This module is the single source: `app.py` and `anthropic_api.py` derive their
protocol-specific dicts from it, so a future engine change has to be made in
one place, not two. The shape is keyed by engine reason, as the backlog
prescribes.
"""

from __future__ import annotations

# engine reason -> (openai_finish_reason, anthropic_stop_reason)
_REASONS: dict[str, tuple[str, str]] = {
    "eog": ("stop", "end_turn"),
    "length": ("length", "max_tokens"),
    "context": ("length", "max_tokens"),
    "cancelled": ("stop", "end_turn"),
    "stop_sequence": ("stop", "stop_sequence"),
    "error": ("stop", "end_turn"),
}

# Derived views for the two surfaces -- kept as plain dicts so existing
# imports (`from reason import _FINISH_REASONS`) keep working, and so the
# dict-identity tests in `test_error_reason_mapping` remain valid.
FINISH_REASONS: dict[str, str] = {k: v[0] for k, v in _REASONS.items()}
STOP_REASONS: dict[str, str] = {k: v[1] for k, v in _REASONS.items()}

# Re-export under the old names for import compatibility:
# `app._FINISH_REASONS` and `anthropic_api._STOP_REASONS` were the original
# spellings. Keeping aliases avoids a sweeping rename in one go.
_REASON_TABLE = _REASONS
