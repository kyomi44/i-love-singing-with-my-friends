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
    assert (
        "colorchannelmixer=rr=0.55" in graph
        and "ass=lyrics.ass" in graph
        and "volume=0.15" in graph
    )
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


def _lines_starting_at(t):
    return [Line(words=[Word(text="hello", start=t, end=t + 0.5, confidence=0.9)])]


def test_intro_end_stops_before_the_first_countdown():
    from deepcuts.stages.render.ass import intro_end

    cfg = RenderConfig()  # lead_in 1.2: the first line appears 1.2 s early
    # first line at 12.88 s (Salem Sisters): countdown from 9.88, so the cover is gone by 9.68
    assert intro_end(_lines_starting_at(12.88), cfg) == pytest.approx(9.68)
    # first line at 3.5 s: no countdown; the line appears at 2.3, too soon for an intro
    assert intro_end(_lines_starting_at(3.5), cfg) is None
    assert intro_end([], cfg) is None


def test_build_ass_puts_title_under_the_cover_during_the_intro():
    cfg = RenderConfig()
    plain = build_ass(_lines_starting_at(12.88), META, cfg)
    with_cover = build_ass(_lines_starting_at(12.88), META, cfg, intro=9.68)
    title = [ln for ln in with_cover.splitlines() if ln.startswith("Dialogue") and ",Title," in ln]
    assert len(title) == 2
    assert all(",0:00:09.68," in ln for ln in title)  # gone exactly when the cover is
    ys = [int(re.search(r"\\pos\(\d+,(\d+)\)", ln).group(1)) for ln in title]
    assert all(y > cfg.height * 0.72 for y in ys)  # below the cover's bottom edge, not mid-screen
    assert "\\pos(960,464)" in plain  # the no-cover title card is unchanged


def test_ffmpeg_cmd_cover_intro_and_cover_background(tmp_path):
    cover = tmp_path / "cover.png"
    cover.write_bytes(b"x")
    cfg = RenderConfig(background="cover")
    cmd = build_ffmpeg_cmd(
        ass_name="lyrics.ass", track=tmp_path / "k.wav", out_path=tmp_path / "o.mp4",
        duration=10, cfg=cfg, cover=cover, intro=9.68,
    )  # fmt: skip
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "fade=t=out:st=8.680:d=1.000:alpha=1" in graph
    assert "overlay=eof_action=pass" in graph and graph.endswith("[under]ass=lyrics.ass[v]")
    assert cmd.count(str(cover.resolve())) == 2  # blurred background + the intro
    assert "9.680" in cmd  # the intro input is only decoded for its own length
    # "cover" background without any art falls back to the plain colour
    cmd = build_ffmpeg_cmd(
        ass_name="lyrics.ass", track=tmp_path / "k.wav", out_path=tmp_path / "o.mp4",
        duration=10, cfg=cfg,
    )  # fmt: skip
    assert "color=c=0x101014" in " ".join(cmd)


@pytest.mark.skipif(not _has_libass(), reason="ffmpeg with libass not available")
def test_cover_extracted_from_audio_and_rendered_as_intro(tmp_path):
    from deepcuts.stages.media import ensure_cover, import_media
    from deepcuts.workspace import SongDir

    song = SongDir(tmp_path / "Band - Song").create()
    art = tmp_path / "art.png"
    run = lambda *a: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)  # noqa: E731
    run("-f", "lavfi", "-i", "color=c=0x3366aa:s=300x300", "-frames:v", "1", str(art))
    run("-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-i", str(art),
        "-map", "0", "-map", "1", "-c:v", "png", "-disposition:v", "attached_pic",
        "-ac", "2", str(song.root / "source.flac"))  # fmt: skip
    cover = ensure_cover(song)
    assert cover == song.root / "cover.png" and cover.exists()
    assert ensure_cover(song) == cover  # extracted once, then reused
    webp = tmp_path / "big.webp"
    run("-f", "lavfi", "-i", "color=c=0xaa3366:s=400x400", "-frames:v", "1", str(webp))
    assert (
        import_media(song, "cover", webp) == song.root / "cover.png"
    )  # converted, replaces the old one

    track = tmp_path / "karaoke.wav"
    run("-i", str(song.root / "source.flac"), "-vn", str(track))
    cfg = RenderConfig(width=640, height=360, font_size=32, fps=15)
    ass = tmp_path / "lyrics.ass"
    ass.write_text(build_ass(_lines_starting_at(8.0), META, cfg, intro=4.8))
    out = render_video(
        ass_path=ass, track=track, out_path=tmp_path / "out.mp4", duration=12, cfg=cfg,
        cover=song.root / "cover.png", intro=4.8,
    )  # fmt: skip
    assert out.exists() and out.stat().st_size > 10_000


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _clip(path, colour="0x228844", seconds=2, size="320x180"):
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"color=c={colour}:s={size}:d={seconds}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )
    return path


def test_media_args_split_settings_from_files(tmp_path):
    from deepcuts.cli import _media_args

    assert _media_args(None, None) == (None, {})
    assert _media_args("#000000", None) == ("#000000", {"background": None})
    assert _media_args("cover", None) == ("cover", {"background": None})
    clip = tmp_path / "loop.mp4"
    assert _media_args(str(clip), clip) == (None, {"background": clip, "cover": clip})


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_import_media_images_and_videos_share_a_slot(tmp_path):
    from deepcuts.errors import DeepcutsError
    from deepcuts.stages.media import background_file, import_media, is_video
    from deepcuts.workspace import SongDir

    song = SongDir(tmp_path / "Band - Song").create()
    img = tmp_path / "pic.jpeg"
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(img))
    assert import_media(song, "background", img) == song.root / "background.jpg"
    clip = _clip(tmp_path / "loop.MOV")
    dst = import_media(song, "background", clip)
    assert dst == song.root / "background.mov" and is_video(dst)
    assert background_file(song) == dst and not (song.root / "background.jpg").exists()
    with pytest.raises(DeepcutsError, match="image .* or a video"):
        import_media(song, "background", _txt(tmp_path))


def _txt(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("x")
    return p


def test_ffmpeg_cmd_loops_videos_in_either_slot(tmp_path):
    bg, cover = tmp_path / "bg.mp4", tmp_path / "cover.webm"
    cmd = build_ffmpeg_cmd(
        ass_name="lyrics.ass", track=tmp_path / "k.wav", out_path=tmp_path / "o.mp4",
        duration=10, cfg=RenderConfig(), cover=cover, intro=9.68, background=bg,
    )  # fmt: skip
    i_bg, i_cover = cmd.index(str(bg.resolve())), cmd.index(str(cover.resolve()))
    assert cmd[i_bg - 3 : i_bg] == ["-stream_loop", "-1", "-i"]
    assert cmd[i_cover - 5 : i_cover] == ["-stream_loop", "-1", "-t", "9.680", "-i"]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.startswith("[0:v]fps=30,scale=") and "[2:v]fps=30,split=2" in graph
    assert "-loop" not in cmd  # no still-image inputs at all


@pytest.mark.skipif(not _has_libass(), reason="ffmpeg with libass not available")
@pytest.mark.parametrize("cover_kind", ["image", "video"])
def test_render_with_video_background_and_image_or_video_intro(tmp_path, cover_kind):
    track = tmp_path / "karaoke.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=330:duration=10", "-ac", "2", str(track))
    bg = _clip(tmp_path / "bg.mp4", "0x224488", seconds=3)  # shorter than the song: must loop
    if cover_kind == "video":
        cover = _clip(tmp_path / "cover.mp4", "0xaa6622", seconds=1, size="200x200")
    else:
        cover = tmp_path / "cover.png"
        _ffmpeg("-f", "lavfi", "-i", "color=c=0xaa6622:s=200x200", "-frames:v", "1", str(cover))
    cfg = RenderConfig(width=320, height=180, font_size=18, fps=10)
    ass = tmp_path / "lyrics.ass"
    ass.write_text(build_ass(_lines_starting_at(7.0), META, cfg, intro=3.8))
    out = render_video(
        ass_path=ass, track=track, out_path=tmp_path / "out.mp4", duration=10, cfg=cfg,
        cover=cover, intro=3.8, background=bg,
    )  # fmt: skip
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert abs(float(info) - 10) < 0.3  # looped background + intro, full song length
    # mid-song frame is the (dimmed) blue background video, not black or the orange cover
    frame = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", "8", "-i", str(out), "-frames:v", "1",
         "-vf", "crop=8:8:0:0,scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    r, g, b = frame[:3]
    # 0x224488 dimmed by 0.45 is about (18, 37, 75): still clearly blue, not washed out to grey
    assert b > 60 and b > 2 * r


def test_display_schedule_matches_the_video_layout():
    from deepcuts.stages.render.ass import display_schedule

    cfg = RenderConfig()
    lines = song_lines() + [Line(words=[Word(text="untimed")])]
    disp = display_schedule(lines, cfg)
    assert disp[-1] is None  # untimed lines have no slot on screen
    sched = schedule([(ln.start, ln.end) for ln in lines[:-1]], cfg.lead_in, cfg.hold)
    assert [d["row"] for d in disp[:-1]] == sched.rows
    assert [d["appear"] for d in disp[:-1]] == [round(a, 3) for a in sched.appear]
    # first line at 5.0 s gets the full 3-2-1, in the same places build_ass draws it
    assert disp[0]["countdown"] == [[2.0, 3], [3.0, 2], [4.0, 1]]
    ass = build_ass(lines, META, cfg)
    assert ass.count(",Countdown,") == 3 and "0:00:02.00,0:00:03.00,Countdown" in ass
