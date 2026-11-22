"""Turn raw aligner output into timings that *feel* right on screen.

Order matters: fill gaps -> clamp runaway words -> make monotonic -> close micro-gaps.
All functions are pure (they work on deep copies) so they're easy to test and reuse.
"""

from __future__ import annotations

from deepcuts.config import AlignConfig
from deepcuts.models import Line, Overrides, Word

PER_WORD = 0.6  # seconds per word when we have to invent timing


def _copy(lines: list[Line]) -> list[Line]:
    return [line.model_copy(deep=True) for line in lines]


def _flat(lines: list[Line]) -> list[Word]:
    return [w for line in lines for w in line.words]


def _left_bound(lines: list[Line], li: int) -> float | None:
    if lines[li].anchor is not None:
        return lines[li].anchor
    for prev in reversed(lines[:li]):
        if prev.end is not None:
            return prev.end
    return None


def _right_bound(lines: list[Line], li: int) -> float | None:
    for nxt in lines[li + 1 :]:
        if nxt.start is not None:
            return nxt.start
        if nxt.anchor is not None:
            return nxt.anchor
    return None


def fill_missing(lines: list[Line]) -> list[Line]:
    """Interpolate untimed words (numbers, symbols, alignment failures), weighted by length."""
    lines = _copy(lines)
    for li, line in enumerate(lines):
        words = line.words
        n = len(words)
        i = 0
        while i < n:
            if words[i].timed:
                i += 1
                continue
            j = i
            while j < n and not words[j].timed:
                j += 1
            count = j - i
            left = words[i - 1].end if i > 0 else _left_bound(lines, li)
            right = words[j].start if j < n else _right_bound(lines, li)
            if left is None and right is None:
                left = 0.0
            if left is None:
                left = max(0.0, right - PER_WORD * count)  # type: ignore[operator]
            if right is None or right <= left:
                right = left + PER_WORD * count
            if i == 0 and line.anchor is None:
                left = max(left, right - PER_WORD * count)  # don't stretch back over a break
            if j == n:
                right = min(right, left + PER_WORD * count)  # ...or forward into one
            weights = [max(1, len(w.text)) for w in words[i:j]]
            total = sum(weights)
            t = left
            for w, wt in zip(words[i:j], weights, strict=True):
                d = (right - left) * wt / total
                w.start, w.end, w.confidence = t, t + d, 0.0
                t += d
            i = j
    return lines


def clamp_long_words(lines: list[Line], max_dur: float) -> list[Line]:
    """A line's first word tends to swallow leading silence; later words swallow trailing."""
    lines = _copy(lines)
    for line in lines:
        for k, w in enumerate(line.words):
            if w.timed and w.end - w.start > max_dur:  # type: ignore[operator]
                if k == 0:
                    w.start = w.end - max_dur  # type: ignore[operator]
                else:
                    w.end = w.start + max_dur  # type: ignore[operator]
    return lines


def enforce_monotonic(lines: list[Line], min_dur: float) -> list[Line]:
    lines = _copy(lines)
    flat = [w for w in _flat(lines) if w.timed]
    prev_start = float("-inf")
    for w in flat:
        w.start = max(w.start, prev_start + 0.01)  # type: ignore[type-var]
        w.end = max(w.end, w.start + min_dur)  # type: ignore[type-var, operator]
        prev_start = w.start
    for a, b in zip(flat, flat[1:], strict=False):
        if a.end > b.start:  # type: ignore[operator]
            a.end = max(a.start + 0.01, b.start)  # type: ignore[operator, type-var]
    return lines


def close_gaps(lines: list[Line], gap: float) -> list[Line]:
    lines = _copy(lines)
    for line in lines:
        for a, b in zip(line.words, line.words[1:], strict=False):
            if a.timed and b.timed and 0 < b.start - a.end <= gap:  # type: ignore[operator]
                a.end = b.start
    return lines


def _round(lines: list[Line]) -> list[Line]:
    for w in _flat(lines):
        if w.start is not None:
            w.start = round(w.start, 3)
        if w.end is not None:
            w.end = round(w.end, 3)
        if w.confidence is not None:
            w.confidence = round(w.confidence, 3)
    return lines


def finalize(lines: list[Line], cfg: AlignConfig) -> list[Line]:
    lines = fill_missing(lines)
    lines = clamp_long_words(lines, cfg.max_word_dur)
    lines = enforce_monotonic(lines, cfg.min_word_dur)
    lines = close_gaps(lines, cfg.gap_close)
    return _round(lines)


def apply_overrides(lines: list[Line], ov: Overrides) -> list[Line]:
    lines = _copy(lines)
    for i, line in enumerate(lines):
        shift = (ov.global_offset_ms + ov.line_shift_ms.get(i, 0)) / 1000
        if not shift:
            continue
        for w in line.words:
            if w.start is not None:
                w.start = max(0.0, w.start + shift)
            if w.end is not None:
                w.end = max(0.0, w.end + shift)
    return lines


def low_confidence(
    lines: list[Line], threshold: float, word_threshold: float = 0.0
) -> list[tuple[int, float]]:
    """(index, mean confidence) of lines worth a listen: low mean, or any very weak word."""
    out = []
    for i, line in enumerate(lines):
        confs = [w.confidence for w in line.words if w.confidence is not None]
        mean = line.confidence or 0.0
        if mean < threshold or (confs and min(confs) < word_threshold):
            out.append((i, mean))
    return out
