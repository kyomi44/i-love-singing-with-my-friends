"""Two-row karaoke layout: when each line appears, in which row, and when it leaves.

Mirrored in web/player.html so the editor preview matches the rendered video exactly.

Rules:
- a line appears `lead_in` seconds before its first word, but never before the previous line in
  the same row has finished;
- after a break (previous line already gone) the screen starts fresh in the top row;
- a line stays `hold` seconds after its last word, or until its row is needed again.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Schedule:
    rows: list[int]
    appear: list[float]
    disappear: list[float]
    fresh: list[bool]


def schedule(
    spans: list[tuple[float, float]], lead_in: float, hold: float, gap: float = 0.05
) -> Schedule:
    rows: list[int] = []
    appear: list[float] = []
    fresh: list[bool] = []
    last: dict[int, int | None] = {0: None, 1: None}
    for i, (s, _e) in enumerate(spans):
        desired = s - lead_in
        is_fresh = i == 0 or spans[i - 1][1] + hold <= desired
        row = 0 if is_fresh else 1 - rows[i - 1]
        a = desired
        j = last[row]
        if j is not None:
            a = max(a, spans[j][1] + gap)
        appear.append(max(0.0, min(a, s)))
        rows.append(row)
        fresh.append(is_fresh)
        last[row] = i
    disappear: list[float] = []
    for i, (_s, e) in enumerate(spans):
        d = e + hold
        nxt = next((j for j in range(i + 1, len(spans)) if rows[j] == rows[i]), None)
        if nxt is not None:
            d = min(d, appear[nxt])
        disappear.append(max(d, e))
    return Schedule(rows, appear, disappear, fresh)


def fit_font_size(
    text: str, base: int, width: int, margin: float = 0.92, char_w: float = 0.56
) -> int:
    """Shrink long lines instead of wrapping them (wrapping would collide with the other row)."""
    est = len(text) * base * char_w
    if est <= width * margin:
        return base
    return max(28, int(base * width * margin / est))
