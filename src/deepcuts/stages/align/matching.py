"""Map aligner output tokens back onto our lyric words (robust to dropped/merged tokens)."""

from __future__ import annotations

import difflib


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def match_words(expected: list[str], got: list[str]) -> list[int | None]:
    """For each expected word, the index of the matching output token (or None)."""
    if len(expected) == len(got):
        return list(range(len(expected)))
    sm = difflib.SequenceMatcher(
        a=[_norm(x) for x in expected], b=[_norm(x) for x in got], autojunk=False
    )
    mapping: list[int | None] = [None] * len(expected)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    return mapping
