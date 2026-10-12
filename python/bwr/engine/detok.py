"""Per-request incremental detokenisers.

A character can span several tokens: byte-fallback vocabularies (SmolLM,
Qwen, Llama 3's GPT-2 style tables) spell "日" as up to three single-byte
tokens, and an emoji with a skin-tone modifier as seven. Decoding each
token on its own therefore yields U+FFFD for every token but the last,
and since every route builds its response from StepOutput.piece, the
client saw "���" where `text_of` (a decode over all the tokens) said "日".

Both classes hold back whatever cannot be decoded yet and emit it with the
token that completes it; `flush` returns what is still held, decoded with
replacement, for the retire that ends the request. On text that decodes
cleanly token by token (all of ASCII) nothing is held, so the pieces are
exactly what the single-token decode produced before.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

REPLACEMENT = "�"

# Tokens held while their joint decode still ends in U+FFFD. A code point is
# at most four bytes and a token at least one, so a genuine partial character
# resolves within four tokens; a run of stray continuation bytes never would,
# and this bound keeps it from being held for the rest of the request.
_MAX_HELD_TOKENS = 8


def _utf8_len(lead: int) -> int:
    """Length of the sequence a lead byte announces, 0 for a byte that cannot lead."""
    if lead < 0x80:
        return 1
    if lead < 0xC0:
        return 0
    if lead < 0xE0:
        return 2
    if lead < 0xF0:
        return 3
    if lead < 0xF8:
        return 4
    return 0


def valid_utf8(data: bytes) -> str:
    """`csrc/model.cpp`'s ensure_valid_utf8, byte for byte: every byte that
    does not start a complete well-formed sequence becomes one U+FFFD. The
    Metal engine's pieces were produced by that routine, so the buffered
    decode must agree with it (and with `detokenize`) on every input, not
    only the well-formed ones Python's errors="replace" treats the same."""
    out: list[str] = []
    i = 0
    n = len(data)
    while i < n:
        length = _utf8_len(data[i])
        if length == 1:
            out.append(chr(data[i]))
            i += 1
            continue
        if (
            length
            and i + length <= n
            and all((data[i + k] & 0xC0) == 0x80 for k in range(1, length))
        ):
            out.append(data[i : i + length].decode("utf-8", errors="replace"))
            i += length
        else:
            out.append(REPLACEMENT)
            i += 1
    return "".join(out)


def _incomplete_tail(data: bytes) -> int:
    """Number of trailing bytes that are the well-formed start of a sequence
    the next piece may complete (0 when the buffer ends on a boundary)."""
    n = len(data)
    # The tail is at most three bytes: a lead plus two continuations.
    for back in range(1, min(3, n) + 1):
        lead = data[n - back]
        if (lead & 0xC0) == 0x80:
            continue  # a continuation byte; look further back for its lead
        length = _utf8_len(lead)
        if length > back and all(
            (data[n - back + k] & 0xC0) == 0x80 for k in range(1, back)
        ):
            return back
        return 0
    return 0


class ByteDetokenizer:
    """Incremental decode over raw token bytes (MetalEngine: llama.cpp's
    pieces are bytes, not text)."""

    def __init__(self) -> None:
        self._held = b""

    def push(self, piece: bytes) -> str:
        data = self._held + piece
        tail = _incomplete_tail(data)
        self._held = data[len(data) - tail :] if tail else b""
        return valid_utf8(data[: len(data) - tail]) if tail else valid_utf8(data)

    def flush(self) -> str:
        held, self._held = self._held, b""
        return valid_utf8(held)


class TokenDetokenizer:
    """Incremental decode over token ids for a tokenizer that only exposes
    `decode(ids) -> str` (MLXEngine's Hugging Face tokenizer). A token whose
    decode ends in U+FFFD is held and decoded together with the tokens that
    follow, so the piece for "日" arrives whole on its third byte-token."""

    def __init__(self, decode: Callable[[Sequence[int]], str]) -> None:
        self._decode = decode
        self._held: list[int] = []

    def push(self, token: int) -> str:
        self._held.append(token)
        text = self._decode(self._held)
        if text.endswith(REPLACEMENT) and len(self._held) < _MAX_HELD_TOKENS:
            return ""
        self._held = []
        return text

    def flush(self) -> str:
        if not self._held:
            return ""
        held, self._held = self._held, []
        return self._decode(held)
