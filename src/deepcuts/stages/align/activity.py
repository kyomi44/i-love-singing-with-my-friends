"""Where is anybody singing? Frame-level vocal activity from the separated vocal stem.

Found on a real song: a line's last words, sung as long soft vowels, have weak CTC emissions.
Left alone, the aligner either padded them across an 8 s instrumental gap (6 s per word, scored
~97%) or moved them onto the next line's first syllable. Activity fixes that in three places:

1. emissions in silent frames only allow blank/<star> (ctc.py), so no word can sit in silence;
2. a line's window stops at the next line's onset when silence separates them;
3. word spans are trimmed to the singing they contain, then held through sustained notes.

Pure Python on per-frame levels so it is testable without torch; ctc.py computes the levels.
"""

from __future__ import annotations

import math

from deepcuts.models import Line


def active_frames(
    levels_db: list[float], rel_db: float = 35.0, floor_db: float = -60.0, hangover: int = 10
) -> list[bool]:
    """Frames louder than (stem's loud level - rel_db), widened by `hangover` frames each side.

    The loud level is the 95th-percentile frame level, so the threshold follows the stem's own
    loudness (quiet singers, hot masters). Never below `floor_db`, so separation bleed stays out.
    """
    if not levels_db:
        return []
    ranked = sorted(levels_db)
    loud = ranked[min(len(ranked) - 1, math.floor(0.95 * len(ranked)))]
    threshold = max(floor_db, loud - rel_db)
    raw = [db > threshold for db in levels_db]
    if hangover <= 0:
        return raw
    out = [False] * len(raw)
    last_active = -(10**9)
    for i, a in enumerate(raw):  # forward pass: hold `hangover` frames after activity
        if a:
            last_active = i
        out[i] = i - last_active <= hangover
    next_active = 10**9
    for i in range(len(raw) - 1, -1, -1):  # backward pass: open `hangover` frames before it
        if raw[i]:
            next_active = i
        out[i] = out[i] or next_active - i <= hangover
    return out


def clip_at_next_onset(
    windows: list[tuple[float, float]],
    anchors: list[float],
    active: list[bool],
    frame_s: float,
    min_gap: float = 0.5,
    search: float = 1.5,
) -> list[tuple[float, float]]:
    """End each line's window at the next line's onset, if a silence of >= min_gap precedes it.

    The onset is the start of an active run (after >= min_gap of silence) within `search` s of
    the next line's anchor. Without a clear pause (legato singing) the window is left alone.
    """
    gap = max(1, round(min_gap / frame_s))
    out = list(windows)
    for i in range(len(windows) - 1):
        ws, we = windows[i]
        nxt = anchors[i + 1]
        lo = max(int(ws / frame_s) + gap, int((nxt - search) / frame_s))
        hi = min(len(active) - 1, int(we / frame_s))
        onsets = [
            f for f in range(max(lo, gap), hi + 1) if active[f] and not any(active[f - gap : f])
        ]
        best = min(onsets, key=lambda f: abs(f * frame_s - nxt), default=None)
        if best is not None and best * frame_s > ws:
            out[i] = (ws, min(we, best * frame_s))
    return out


def fit_to_activity(lines: list[Line], active: list[bool], frame_s: float) -> None:
    """In place: trim each word to the singing inside its span, then hold it through the note.

    Trim: CTC pads a word's span halfway into the blanks around it, which can reach far into
    silence. Hold: a sustained vowel is a short token burst then blanks, so the end is extended
    while the stem stays active, never past the next word's start. Words whose span contains no
    singing at all are left untouched (they'll show up as low confidence).
    """
    words = [w for line in lines for w in line.words if w.start is not None and w.end is not None]
    n = len(active)
    for w in words:  # trim
        a, b = max(0, int(w.start / frame_s)), min(n, int(round(w.end / frame_s)))
        idx = [f for f in range(a, b) if active[f]]
        if idx:
            w.start = max(w.start, idx[0] * frame_s)
            w.end = min(w.end, (idx[-1] + 1) * frame_s)
    for i, w in enumerate(words):  # hold
        cap = words[i + 1].start if i + 1 < len(words) else n * frame_s
        f = int(round(w.end / frame_s))
        while f < n and active[f] and (f + 1) * frame_s <= cap:
            f += 1
        w.end = max(w.end, min(f * frame_s, cap))
