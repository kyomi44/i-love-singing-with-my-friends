r"""ASS (Advanced SubStation Alpha) karaoke subtitles.

Each word gets a ``{\kf<centiseconds>}`` tag, which libass renders as a smooth left-to-right
colour wipe from SecondaryColour (unsung) to PrimaryColour (sung). Durations are computed from
absolute rounded times, so rounding error never accumulates across a line.
"""

from __future__ import annotations

from deepcuts.config import RenderConfig
from deepcuts.models import Line, SongMeta, Word
from deepcuts.stages.render.layout import fit_font_size, schedule


def ts(t: float) -> str:
    cs = max(0, round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def cs(t: float) -> int:
    return max(0, round(t * 100))


def _rgb(hex_color: str) -> tuple[str, str, str]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #RRGGBB, got {hex_color!r}")
    return h[0:2].upper(), h[2:4].upper(), h[4:6].upper()


def style_color(hex_color: str, alpha: int = 0) -> str:
    r, g, b = _rgb(hex_color)
    return f"&H{alpha:02X}{b}{g}{r}"


def override_color(hex_color: str) -> str:
    r, g, b = _rgb(hex_color)
    return f"&H{b}{g}{r}&"


def escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def karaoke_text(words: list[Word], appear: float, lead: float = 0.0) -> str:
    parts: list[str] = []
    cursor = cs(appear)
    italic = False
    for k, w in enumerate(words):
        ws = cs(w.start - lead)  # type: ignore[operator]
        we = cs(w.end - lead)  # type: ignore[operator]
        gap = ws - cursor
        if gap > 0:
            parts.append(f"{{\\k{gap}}}")
            cursor += gap
        dur = max(1, we - cursor)
        cursor += dur
        tags = f"\\kf{dur}"
        if w.text.startswith("(") and not italic:
            tags += "\\i1"
            italic = True
        text = escape(w.text) + (" " if k < len(words) - 1 else "")
        close = ""
        if italic and w.text.rstrip(",.!?;:'\"").endswith(")"):
            close = "{\\i0}"
            italic = False
        parts.append(f"{{{tags}}}{text}{close}")
    return "".join(parts)


def _dialogue(
    start: float, end: float, style: str, text: str, name: str = "", layer: int = 0
) -> str:
    return f"Dialogue: {layer},{ts(start)},{ts(end)},{style},{name},0,0,0,,{text}"


MIN_INTRO = 2.5
"""Seconds of intro needed before the cover is worth showing."""
COVER_HEIGHT = 0.6
COVER_CENTER_Y = 0.42
"""Cover size and position as fractions of the frame height (shared with video.py)."""


def _timeline(lines: list[Line], cfg: RenderConfig):
    lead = cfg.highlight_lead_ms / 1000.0
    timed = [line for line in lines if line.start is not None and line.end is not None]
    spans = [(line.start - lead, line.end - lead) for line in timed]  # type: ignore[operator]
    return timed, spans, schedule(spans, cfg.lead_in, cfg.hold)


def intro_end(lines: list[Line], cfg: RenderConfig) -> float | None:
    """When the cover intro has fully faded out, or None if the intro is too short to show.

    It ends before the first line appears, and before the first line's 3-2-1 countdown (which
    sits where the cover is), with a 0.2 s breath in between.
    """
    _, spans, sched = _timeline(lines, cfg)
    if not spans:
        return None
    s0 = spans[0][0]
    end = sched.appear[0]
    if s0 >= 4.0:  # build_ass gives the first line a countdown from s0 - 3
        end = min(end, s0 - 3.0)
    end -= 0.2
    return end if end >= MIN_INTRO else None


def countdown_dots(i: int, spans, sched, cfg: RenderConfig) -> list[tuple[float, int]]:
    """(start, dot count) for the 3-2-1 countdown before line i: after a long gap, or a late
    first line. Each step lasts 1 s; steps that would overlap the previous line are dropped."""
    s = spans[i][0]
    if not sched.fresh[i]:
        return []
    prev_end = spans[i - 1][1] if i else 0.0
    if not (s - prev_end >= cfg.countdown_gap or (i == 0 and s >= 4.0)):
        return []
    return [(s - k, k) for k in (3, 2, 1) if s - k >= prev_end + 0.3]


def display_schedule(lines: list[Line], cfg: RenderConfig) -> list[dict | None]:
    """Per line (None if untimed): row, appear, disappear, countdown and the highlight lead, as
    the video lays it out. Written into timing.json so other players needn't re-derive it."""
    timed, spans, sched = _timeline(lines, cfg)
    index = {id(line): i for i, line in enumerate(timed)}
    out: list[dict | None] = []
    for line in lines:
        i = index.get(id(line))
        if i is None:
            out.append(None)
            continue
        out.append(
            {
                "row": sched.rows[i],
                "appear": round(sched.appear[i], 3),
                "disappear": round(sched.disappear[i], 3),
                "countdown": [[round(t, 3), k] for t, k in countdown_dots(i, spans, sched, cfg)],
            }
        )
    return out


def build_ass(
    lines: list[Line], meta: SongMeta, cfg: RenderConfig, intro: float | None = None
) -> str:
    """`intro`: end time of the cover intro (see intro_end), if the video opens on the cover."""
    W, H, fs = cfg.width, cfg.height, cfg.font_size
    lead = cfg.highlight_lead_ms / 1000.0
    timed, spans, sched = _timeline(lines, cfg)
    row_y = [round(H / 2 - fs * 0.85), round(H / 2 + fs * 0.85)]
    bold = -1 if cfg.bold else 0

    header = f"""[Script Info]
; generated by deepcuts
Title: {escape(meta.artist)} - {escape(meta.title)}
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,{cfg.font},{fs},{style_color(cfg.sung_color)},{style_color(cfg.unsung_color)},{style_color(cfg.outline_color)},&H80000000,{bold},0,0,0,100,100,0,0,1,{cfg.outline},2,5,40,40,40,1
Style: Title,{cfg.font},{round(fs * 1.25)},{style_color(cfg.unsung_color)},{style_color(cfg.unsung_color)},{style_color(cfg.outline_color)},&H80000000,-1,0,0,0,100,100,0,0,1,{cfg.outline},2,5,40,40,40,1
Style: Countdown,{cfg.font},{round(fs * 0.8)},{style_color(cfg.sung_color)},{style_color(cfg.sung_color)},{style_color(cfg.outline_color)},&H80000000,-1,0,0,0,100,100,0,0,1,{cfg.outline},0,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    if cfg.title_card and spans and intro is not None:
        # under the cover, fading out with it
        fade_ms = round(min(cfg.cover_fade, intro / 3) * 1000)
        cover_bottom = H * (COVER_CENTER_Y + COVER_HEIGHT / 2)
        pos_t = f"\\pos({W // 2},{round(cover_bottom + fs * 1.1)})"
        pos_a = f"\\pos({W // 2},{round(cover_bottom + fs * 2.15)})"
        fad = f"\\fad(600,{fade_ms})"
        events.append(_dialogue(0.0, intro, "Title", f"{{\\an5{pos_t}{fad}}}{escape(meta.title)}"))
        events.append(
            _dialogue(
                0.0,
                intro,
                "Title",
                f"{{\\an5{pos_a}\\fs{round(fs * 0.7)}{fad}}}{escape(meta.artist)}",
            )
        )
    elif cfg.title_card and spans:
        end = min(spans[0][0] - cfg.lead_in - 0.3, 6.5)
        if end > 1.8:
            pos_t, pos_a = (
                f"\\pos({W // 2},{round(H * 0.43)})",
                f"\\pos({W // 2},{round(H * 0.56)})",
            )
            events.append(
                _dialogue(0.3, end, "Title", f"{{\\an5{pos_t}\\fad(400,400)}}{escape(meta.title)}")
            )
            events.append(
                _dialogue(
                    0.3,
                    end,
                    "Title",
                    f"{{\\an5{pos_a}\\fs{round(fs * 0.7)}\\fad(400,400)}}{escape(meta.artist)}",
                )
            )

    for i, line in enumerate(timed):
        for t0, k in countdown_dots(i, spans, sched, cfg):
            dots = " ".join(["●"] * k)
            pos = f"\\an5\\pos({W // 2},{round(row_y[0] - fs * 1.25)})"
            events.append(_dialogue(t0, t0 + 1.0, "Countdown", f"{{{pos}}}{dots}"))
        size = fit_font_size(line.text, fs, W)
        tags = f"\\an5\\pos({W // 2},{row_y[sched.rows[i]]})"
        if size != fs:
            tags += f"\\fs{size}"
        if line.singer and line.singer in cfg.singer_colors:
            tags += f"\\1c{override_color(cfg.singer_colors[line.singer])}"
        body = karaoke_text(line.words, sched.appear[i], lead)
        events.append(
            _dialogue(
                sched.appear[i],
                sched.disappear[i],
                "Lyrics",
                f"{{{tags}}}{body}",
                line.singer or "",
                1,
            )
        )
    return header + "\n".join(events) + "\n"
