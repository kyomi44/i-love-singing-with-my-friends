"""Using synced (LRC) line times as priors without trusting them blindly.

Community LRC files are often timed against a different master/edit than your file, so the
`auto` strategy first aligns the whole song globally, measures the median offset between the
aligned line starts and the LRC anchors, shifts the anchors by it, and only then refines each
line inside a tight window around its (corrected) anchor.
"""

from __future__ import annotations

import statistics

from deepcuts.models import Line


def line_windows(
    starts: list[float],
    ends: list[float | None],
    duration: float,
    pad_before: float,
    pad_after: float,
    max_len: float = 25.0,
) -> list[tuple[float, float]]:
    """Search window per line: [anchor - pad, next anchor (or explicit end) + pad]."""
    out: list[tuple[float, float]] = []
    n = len(starts)
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < n else min(duration, s + max_len)
        end = ends[i] if ends[i] is not None else nxt
        end = min(max(end, s + 0.5), s + max_len)
        ws = max(0.0, s - pad_before)
        we = min(duration, end + pad_after)
        out.append((ws, max(we, min(duration, ws + 0.5))))
    return out


def estimate_offset(lines: list[Line], min_conf: float = 0.5, min_lines: int = 3) -> float:
    """Median (aligned start - LRC anchor) over confidently aligned lines."""
    pairs = [
        (line.start - line.anchor, line.confidence or 0.0)
        for line in lines
        if line.anchor is not None and line.start is not None
    ]
    good = [d for d, c in pairs if c >= min_conf]
    pool = good if len(good) >= min_lines else [d for d, _ in pairs]
    return float(statistics.median(pool)) if pool else 0.0


def pick_best(
    global_lines: list[Line], anchored_lines: list[Line], margin: float = 0.05
) -> list[Line]:
    """Per line, keep the anchored result unless the global one is clearly more confident."""
    out: list[Line] = []
    for g, a in zip(global_lines, anchored_lines, strict=True):
        a_ok = all(w.timed for w in a.words)
        g_ok = all(w.timed for w in g.words)
        ca, cg = a.confidence or 0.0, g.confidence or 0.0
        chosen = a if (a_ok and (ca >= cg - margin or not g_ok)) else (g if g_ok else a)
        chosen = chosen.model_copy(deep=True)
        chosen.anchor, chosen.anchor_end = a.anchor, a.anchor_end  # keep corrected anchors
        out.append(chosen)
    return out
