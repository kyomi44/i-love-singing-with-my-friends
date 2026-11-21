import re
import shutil
import subprocess

import pytest

from deepcuts.config import RenderConfig
from deepcuts.models import Line, SongMeta, Word
from deepcuts.stages.render.ass import (
    build_ass,
    escape,
    karaoke_text,
    override_color,
    style_color,
    ts,
)
from deepcuts.stages.render.layout import fit_font_size, schedule
from deepcuts.stages.render.video import build_ffmpeg_cmd, render_video

META = SongMeta(artist="Some Band", title="A Deep Cut", duration=12.0, source="test")


def song_lines():
    return [
        Line(
            words=[Word(text="hello", start=5.0, end=5.5), Word(text="there", start=5.7, end=6.3)]
        ),
        Line(
            words=[Word(text="(ooh", start=6.6, end=7.0), Word(text="yeah)", start=7.0, end=7.8)],
            singer="B",
        ),
        Line(
            words=[
                Word(text="{weird}", start=8.0, end=8.6),
                Word(text="back\\slash", start=8.6, end=9.2),
            ]
        ),
    ]


def test_time_and_colour_formatting():
    assert ts(0) == "0:00:00.00" and ts(3723.456) == "1:02:03.46"
    assert style_color("#4FC3F7") == "&H00F7C34F"
    assert override_color("#ff0000") == "&H0000FF&"
    assert escape("{a}\\b") == "(a)/b"


def test_karaoke_text_durations_sum_exactly():
    words = song_lines()[0].words
    text = karaoke_text(words, appear=3.8)
    ks = [int(n) for n in re.findall(r"\\kf?(\d+)", text)]
    assert ks == [120, 50, 20, 60]  # lead-in gap, hello, gap, there
    assert sum(ks) == round((6.3 - 3.8) * 100)


def test_karaoke_text_italicises_parentheticals():
    text = karaoke_text(song_lines()[1].words, appear=6.0)
    assert "\\i1}(ooh" in text and "yeah){\\i0}" in text


def test_schedule_rows_never_overlap():
    spans = [(1, 2), (2.2, 3), (3.1, 4), (4.2, 5), (20, 21), (21.2, 22)]
    s = schedule(spans, lead_in=1.2, hold=0.4)
    assert s.rows == [0, 1, 0, 1, 0, 1]
    assert s.fresh == [True, False, False, False, True, False]
    for i in range(len(spans)):
        assert s.appear[i] <= spans[i][0] and s.disappear[i] >= spans[i][1]
        for j in range(i + 1, len(spans)):
            if s.rows[j] == s.rows[i]:
                assert s.disappear[i] <= s.appear[j] + 1e-9
                break


def test_fit_font_size_shrinks_long_lines():
    assert fit_font_size("short line", 76, 1920) == 76
    assert fit_font_size("x" * 80, 76, 1920) < 76


def test_build_ass_structure():
    cfg = RenderConfig()
    ass = build_ass(song_lines(), META, cfg)
    assert "[Script Info]" in ass and "PlayResX: 1920" in ass
    dialogues = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert sum(",Lyrics," in d for d in dialogues) == 3
    assert any(",Title," in d and "A Deep Cut" in d for d in dialogues)
    assert any(",Countdown," in d for d in dialogues)  # first line starts at 5s
    assert "\\1c&HB18FF4&" in ass  # singer B (#F48FB1) colour override, BGR order
    assert "{weird}" not in ass and "(weird)" in ass


def test_ffmpeg_cmd_guide_vocals_and_image_background(tmp_path):
    img = tmp_path / "cover.jpg"
    img.write_bytes(b"x")
    cfg = RenderConfig(background=str(img), guide_vocals=0.15)
    cmd = build_ffmpeg_cmd(
        ass_name="lyrics.ass", track=tmp_path / "k.wav", out_path=tmp_path / "o.mp4",
        duration=10, cfg=cfg, vocals=tmp_path / "v.wav",
    )  # fmt: skip
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "drawbox" in graph and "ass=lyrics.ass" in graph and "volume=0.15" in graph
    assert cmd[cmd.index("-map", cmd.index("-map") + 1) + 1] == "[a]"


def _has_libass() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
    ).stdout
    return " ass " in out


@pytest.mark.skipif(not _has_libass(), reason="ffmpeg with libass not available")
def test_render_video_smoke(tmp_path):
    track = tmp_path / "karaoke.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-ac", "2", str(track)],
        check=True,
    )  # fmt: skip
    cfg = RenderConfig(width=640, height=360, font_size=32, fps=15)
    ass = tmp_path / "lyrics.ass"
    ass.write_text(build_ass(song_lines(), META, cfg))
    out = render_video(
        ass_path=ass, track=track, out_path=tmp_path / "out.mp4", duration=12, cfg=cfg
    )
    assert out.exists() and out.stat().st_size > 10_000
