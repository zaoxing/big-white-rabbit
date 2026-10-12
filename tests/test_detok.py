"""The incremental detokenisers: a character split across tokens is emitted
whole with the token that completes it, never as per-token U+FFFD."""

from __future__ import annotations

import pytest
from bwr.engine.detok import ByteDetokenizer, TokenDetokenizer, valid_utf8

CRAB = "🦀".encode()  # f0 9f a6 80
JA = "日本語".encode()


# --- ensure_valid_utf8, ported -----------------------------------------------


@pytest.mark.parametrize(
    ("data", "text"),
    [
        (b"hello", "hello"),
        (JA, "日本語"),
        (CRAB, "🦀"),
        (b"\x80", "�"),
        (b"\xe6\x97", "��"),  # one per byte, as the C++ does
        (b"\xe6\x97a", "��a"),
        (b"\xf8\x80", "��"),
        (b"", ""),
    ],
)
def test_valid_utf8_matches_the_cpp_routine(data: bytes, text: str) -> None:
    assert valid_utf8(data) == text


# --- bytes in, text out ---------------------------------------------------------


def test_ascii_pieces_pass_through_unchanged() -> None:
    d = ByteDetokenizer()
    assert [d.push(p) for p in (b"Hello", b",", b" world", b"\n")] == [
        "Hello", ",", " world", "\n"
    ]
    assert d.flush() == ""


def test_a_whole_multibyte_piece_passes_through() -> None:
    d = ByteDetokenizer()
    assert d.push(JA) == "日本語"
    assert d.push(CRAB) == "🦀"


def test_a_code_point_split_across_pieces_is_emitted_when_complete() -> None:
    d = ByteDetokenizer()
    assert d.push(CRAB[:1]) == ""
    assert d.push(CRAB[1:2]) == ""
    assert d.push(CRAB[2:3]) == ""
    assert d.push(CRAB[3:]) == "🦀"
    assert d.flush() == ""


def test_a_piece_ending_mid_character_holds_only_the_tail() -> None:
    d = ByteDetokenizer()
    assert d.push(b" " + JA[:4]) == " 日"
    assert d.push(JA[4:]) == "本語"


def test_a_piece_spanning_two_characters() -> None:
    d = ByteDetokenizer()
    assert d.push(JA[:2]) == ""
    assert d.push(JA[2:5]) == "日"
    assert d.push(JA[5:]) == "本語"


def test_flush_replaces_an_incomplete_tail() -> None:
    d = ByteDetokenizer()
    assert d.push(JA[:2]) == ""
    assert d.flush() == "��"
    assert d.flush() == ""


def test_a_stray_continuation_byte_is_replaced_not_held() -> None:
    d = ByteDetokenizer()
    assert d.push(b"\x80") == "�"
    assert d.push(b"a") == "a"


def test_an_invalid_lead_then_a_valid_character() -> None:
    d = ByteDetokenizer()
    assert d.push(b"\xe6\x97") == ""
    # 'a' cannot continue the sequence: the held bytes are replaced, 'a' passes.
    assert d.push(b"a") == "��a"


# --- token ids in, text out -----------------------------------------------------


class _ByteVocab:
    """Token t decodes to the byte t (0-255); ids past that are words."""

    def decode(self, ids) -> str:
        return valid_utf8(
            b"".join(bytes([t]) if t < 256 else _WORDS[t].encode() for t in ids)
        )


_WORDS = {256: "Hello", 257: " world", 258: "\n"}


def _ids(data: bytes) -> list[int]:
    return list(data)


def test_token_ascii_pieces_pass_through_unchanged() -> None:
    d = TokenDetokenizer(_ByteVocab().decode)
    assert [d.push(t) for t in (256, 257, 258)] == ["Hello", " world", "\n"]
    assert d.flush() == ""


def test_token_split_code_point_is_emitted_when_complete() -> None:
    d = TokenDetokenizer(_ByteVocab().decode)
    out = [d.push(t) for t in _ids(CRAB + b" " + JA)]
    assert "".join(out) == "🦀 日本語"
    assert out == ["", "", "", "🦀", " ", "", "", "日", "", "", "本", "", "", "語"]
    assert d.flush() == ""


def test_token_flush_replaces_an_incomplete_tail() -> None:
    d = TokenDetokenizer(_ByteVocab().decode)
    assert [d.push(t) for t in _ids(JA[:2])] == ["", ""]
    assert d.flush() == "��"
    assert d.flush() == ""


def test_token_garbage_is_not_held_forever() -> None:
    d = TokenDetokenizer(_ByteVocab().decode)
    out = [d.push(0x80) for _ in range(20)]
    assert "".join(out) == "�" * 16
    assert d.flush() == "�" * 4
