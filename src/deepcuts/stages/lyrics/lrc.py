"""LRC parsing and writing, including the "enhanced"/A2 word-timing extension.

[00:12.34]Line text                      standard line
[00:12.34][01:40.00]Chorus line          repeated line, expanded to two lines
[00:12.34]<00:12.34> Word <00:12.80> two <00:13.20>   enhanced (word-level)
[01:02.00]                               empty line = end of previous line / break
[ar:Artist] [ti:Title] [length: 3:21]    metadata, ignored
"""

from __future__ import annotations

import re

from deepcuts.models import Line, SongMeta, Word
from deepcuts.text import split_singer, tokenize

TAG_RE = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
WORD_TAG_RE = re.compile(r"<(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?>")
META_RE = re.compile(r"^\s*\[[A-Za-z#]+:[^\]]*\]\s*$")
HAS_WORDCHAR = re.compile(r"\w", re.UNICODE)


def _secs(minutes: str, seconds: str, frac: str | None) -> float:
    value = int(minutes) * 60 + int(seconds)
    if frac:
        value += int(frac) / (10 ** len(frac))
    return value


def is_lrc(text: str) -> bool:
    return any(TAG_RE.match(line.strip()) for line in text.splitlines())


def _parse_body(body: str) -> list[Word]:
    tags = list(WORD_TAG_RE.finditer(body))
    if not tags:
        return tokenize(body)
    words = tokenize(body[: tags[0].start()])
    segment: list[Word] = []
    seg_start = 0.0
    for k, m in enumerate(tags):
        t = _secs(*m.groups())
        if segment:  # close the previous tagged segment, spreading its words evenly
            step = (t - seg_start) / len(segment)
            for idx, w in enumerate(segment):
                w.start, w.end = seg_start + idx * step, seg_start + (idx + 1) * step
        seg_end = tags[k + 1].start() if k + 1 < len(tags) else len(body)
        segment = [Word(text=tok) for tok in body[m.end() : seg_end].split()]
        seg_start = t
        if segment:
            segment[0].start = t
        words.extend(segment)
    return words


def parse_lrc(text: str) -> list[Line]:
    entries: list[tuple[float, int, str]] = []
    for order, raw in enumerate(text.splitlines()):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if META_RE.match(s) and not TAG_RE.match(s):
            continue
        times: list[float] = []
        pos = 0
        while m := TAG_RE.match(s, pos):
            times.append(_secs(*m.groups()))
            pos = m.end()
        if not times:
            continue
        body = s[pos:].strip()
        entries.extend((t, order, body) for t in times)
    entries.sort(key=lambda e: (e[0], e[1]))

    lines: list[Line] = []
    stanza = 0
    pending_break = False
    for t, _, body in entries:
        if not HAS_WORDCHAR.search(body):  # "", "♪", "..." -> instrumental break marker
            if lines:
                if lines[-1].anchor_end is None:
                    lines[-1].anchor_end = t
                pending_break = True
            continue
        if pending_break:
            stanza += 1
            pending_break = False
        singer, body = split_singer(body)
        words = _parse_body(body)
        if words:
            lines.append(Line(words=words, singer=singer, anchor=t, stanza=stanza))
    return lines


def fmt_time(t: float) -> str:
    cs = max(0, round(t * 100))
    minutes, rem = divmod(cs, 6000)
    return f"{minutes:02d}:{rem // 100:02d}.{rem % 100:02d}"


def dump_lrc(lines: list[Line], meta: SongMeta | None = None, enhanced: bool = True) -> str:
    out: list[str] = []
    if meta:
        out += [f"[ar:{meta.artist}]", f"[ti:{meta.title}]"]
        if meta.album:
            out.append(f"[al:{meta.album}]")
        out += [f"[length:{fmt_time(meta.duration)[:5]}]", "[re:deepcuts]", ""]
    for line in lines:
        if line.start is None:
            continue
        prefix = f"@{line.singer} " if line.singer else ""
        if enhanced and all(w.timed for w in line.words):
            body = " ".join(f"<{fmt_time(w.start)}> {w.text}" for w in line.words)  # type: ignore[arg-type]
            body += f" <{fmt_time(line.end)}>"  # type: ignore[arg-type]
        else:
            body = line.text
        out.append(f"[{fmt_time(line.start)}]{prefix}{body}")
    return "\n".join(out) + "\n"
