"""End-to-end pipeline run with fake ML backends (real ffmpeg, real caching, real render)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import deepcuts.stages.align as align_mod
import deepcuts.stages.separate as separate_mod
from deepcuts.cli import app
from deepcuts.config import Config
from deepcuts.pipeline import Pipeline

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

LRC = """[ar:Fake Band]
[00:02.00]First line of the song
[00:05.00]Second line here
[00:08.00]
[00:15.00]@B After the break
"""


class FakeBackend:
    name = "local"

    def split(self, model, src, out_dir, names):
        out_dir.mkdir(parents=True, exist_ok=True)
        names = names or {"Vocals": "x_(Vocals)", "Instrumental": "x_(Instrumental)"}
        outs = []
        for label in names.values():
            dst = out_dir / f"{label}.wav"
            shutil.copy2(src, dst)
            outs.append(dst)
        return outs


class FakeAligner:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def align(self, audio, lines, language, strategy):
        self.calls += 1
        out = [line.model_copy(deep=True) for line in lines]
        for line in out:
            t = line.anchor + 0.3  # pretend the LRC is 300 ms early
            for w in line.words:
                if w.text.isalpha():
                    w.start, w.end, w.confidence = t, t + 0.35, 0.9
                t += 0.4
        return out, 0.3


@pytest.fixture
def env(tmp_path, monkeypatch):
    src = tmp_path / "input.flac"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=20",
         "-ac", "2", "-metadata", "artist=Fake Band", "-metadata", "title=Deep Cut", str(src)],
        check=True,
    )  # fmt: skip
    lrc = tmp_path / "song.lrc"
    lrc.write_text(LRC)
    aligner = FakeAligner()
    monkeypatch.setattr(separate_mod, "get_backend", lambda cfg: FakeBackend())
    monkeypatch.setattr(align_mod, "get_aligner", lambda cfg, device: aligner)
    cfg = Config.model_validate(
        {"library": str(tmp_path / "lib"), "render": {"width": 640, "height": 360, "fps": 10}}
    )
    return cfg, src, lrc, aligner


def test_make_then_cached_rerun_then_edit_lyrics(env):
    cfg, src, lrc, aligner = env
    res = Pipeline(cfg).make(str(src), lyrics_file=lrc)
    song = res.song
    assert song.name == "Fake Band - Deep Cut"
    assert res.outputs["video"].stat().st_size > 5_000
    stems = json.loads(song.stems_path.read_text())
    assert stems["karaoke"] == "stems/karaoke.wav" and (song.root / stems["lead"]).exists()
    aln = song.load_alignment()
    assert aln.lrc_offset == 0.3 and aln.strategy == "auto"
    assert aln.lines[2].singer == "B" and aln.lines[1].anchor_end == 8.0
    assert all(w.timed for line in aln.lines for w in line.words)
    ass = res.outputs["ass"].read_text()
    assert ass.count(",Lyrics,") == 3

    # second run: everything cached, aligner not called again
    Pipeline(cfg).make(str(src))
    assert aligner.calls == 1

    # editing the lyrics file re-runs alignment (and only that + render)
    song.lyrics_lrc.write_text(LRC.replace("Second line here", "Second line changed"))
    Pipeline(cfg).make(str(src))
    assert aligner.calls == 2
    assert "changed" in song.load_alignment().lines[1].text


def test_cli_nudge_report_ls(env):
    cfg, src, lrc, _ = env
    Pipeline(cfg).make(str(src), lyrics_file=lrc)
    conf = Path(cfg.library_path).parent / "deepcuts.toml"
    conf.write_text(f'library = "{cfg.library_path}"\n[render]\nvideo = false\n')
    runner = CliRunner()
    r = runner.invoke(app, ["nudge", "deep cut", "--ms", "-120", "-c", str(conf)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["nudge", "deep cut", "--ms", "40", "--line", "2", "-c", str(conf)])
    assert r.exit_code == 0, r.output
    ov = json.loads((cfg.library_path / "Fake Band - Deep Cut" / "overrides.json").read_text())
    assert ov == {"global_offset_ms": -120, "line_shift_ms": {"1": 40}}
    timing = json.loads(
        (cfg.library_path / "Fake Band - Deep Cut" / "out" / "timing.json").read_text()
    )
    assert abs(timing["lines"][0]["words"][0]["start"] - (2.3 - 0.12)) < 1e-6
    for cmd in (["report", "deep"], ["ls"]):
        r = runner.invoke(app, [*cmd, "-c", str(conf)])
        assert r.exit_code == 0, r.output
    assert "Fake Band - Deep Cut" in r.output
