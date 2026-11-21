"""Plain-text lyric parsing: section headers, singer tags, stanzas, comments.

Conventions (all optional):
    # comment lines are ignored
    [Chorus], [Verse 2: Someone]   -> section headers, dropped (they start a new stanza)
    @A / @B / @AB at line start     -> duet part for that line
    blank line                      -> stanza break
"""

from __future__ import annotations

import re

from deepcuts.models import Line, Word

SECTION_RE = re.compile(r"^\s*\[[^\]]*\]\s*$")
SINGER_RE = re.compile(r"^@(A\+B|AB|A|B)\s+", re.IGNORECASE)
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def split_singer(text: str) -> tuple[str | None, str]:
    m = SINGER_RE.match(text)
    if not m:
        return None, text
    return m.group(1).upper().replace("+", ""), text[m.end() :]


def tokenize(text: str) -> list[Word]:
    return [Word(text=tok) for tok in text.split()]


def is_alignable(word: str) -> bool:
    """Words with no letters ("—", "&", "99") can't be aligned by a character CTC model."""
    return bool(LETTER_RE.search(word))


def parse_plain(text: str) -> list[Line]:
    lines: list[Line] = []
    stanza = 0
    pending_break = False
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("#"):
            continue
        if not s or SECTION_RE.match(s):
            pending_break = bool(lines)
            continue
        if pending_break:
            stanza += 1
            pending_break = False
        singer, body = split_singer(s)
        words = tokenize(body)
        if words:
            lines.append(Line(words=words, singer=singer, stanza=stanza))
    return lines


def lines_to_text(lines: list[Line]) -> str:
    out: list[str] = []
    prev_stanza = None
    for line in lines:
        if prev_stanza is not None and line.stanza != prev_stanza:
            out.append("")
        prefix = f"@{line.singer} " if line.singer else ""
        out.append(prefix + line.text)
        prev_stanza = line.stanza
    return "\n".join(out) + "\n"
