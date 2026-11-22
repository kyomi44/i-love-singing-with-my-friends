"""compute = "modal" client logic, with a fake worker standing in for the Modal container."""

import shutil
import subprocess

import pytest

import deepcuts.remote.client as client
from deepcuts.config import Config
from deepcuts.models import Line, Lyrics, SongMeta, StemSet, Word
from deepcuts.stages.align import run_alignment
from deepcuts.stages.separate import run_separation
from deepcuts.workspace import SongDir

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


class _Method:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        return self.fn(*args)


class FakeWorker:
    """Echoes the uploaded mix back as every stem, like a (useless) separator."""

    def __init__(self):
        self.separate = _Method(self._separate)
        self.align = _Method(self._align)

    @staticmethod
    def _separate(audio_flac, cfg):
        assert audio_flac[:4] == b"fLaC"
        Config.model_validate(cfg)  # the config survives the JSON round trip
        return {
            "stems": {n: audio_flac for n in ("vocals", "instrumental", "lead", "backing")},
            "models": ["fake.ckpt", "fake-karaoke.ckpt"],
        }

    @staticmethod
    def _align(stem_flac, lines, language, strategy, align_cfg):
        assert stem_flac[:4] == b"fLaC"
        out = [Line.model_validate(x) for x in lines]
        for i, line in enumerate(out):
            for j, w in enumerate(line.words):
                w.start, w.end, w.confidence = i + j * 0.3, i + j * 0.3 + 0.25, 0.8
        return [x.model_dump(mode="json") for x in out], 0.5


@pytest.fixture
def song(tmp_path):
    sd = SongDir(tmp_path / "Band - Song").create()
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(sd.audio)],
        check=True,
    )  # fmt: skip
    return sd


@pytest.fixture
def fake(monkeypatch):
    w = FakeWorker()
    monkeypatch.setattr(client, "worker", lambda cfg: w)
    return w


def test_separation_dispatches_to_modal_and_lands_stems_locally(song, fake):
    cfg = Config.model_validate({"compute": "modal"})
    stems = run_separation(song, cfg)
    assert len(fake.separate.calls) == 1
    assert stems.backend == "modal"
    assert stems.models == ["fake.ckpt", "fake-karaoke.ckpt"]
    for p in (stems.vocals, stems.instrumental, stems.lead, stems.backing, stems.karaoke):
        assert p.exists() and p.parent == song.stems_dir and p.suffix == ".wav"
    assert not list(song.stems_dir.glob("*.flac"))  # transfer files cleaned up


def test_backing_remove_makes_karaoke_from_instrumental_only(song, fake):
    cfg = Config.model_validate({"compute": "modal", "separation": {"backing": "remove"}})
    stems = run_separation(song, cfg)
    assert stems.karaoke.read_bytes() == stems.instrumental.read_bytes()


def test_mvsep_backend_is_not_routed_through_modal(song, fake, monkeypatch):
    import deepcuts.stages.separate as sep

    monkeypatch.setattr(
        sep, "get_backend", lambda cfg: (_ for _ in ()).throw(RuntimeError("mvsep"))
    )
    cfg = Config.model_validate({"compute": "modal", "separation": {"backend": "mvsep"}})
    with pytest.raises(RuntimeError, match="mvsep"):
        run_separation(song, cfg)
    assert fake.separate.calls == []


def test_alignment_runs_remotely_and_is_postprocessed_locally(song, fake):
    cfg = Config.model_validate({"compute": "modal"})
    lines = [
        Line(words=[Word(text="hello"), Word(text="there")], anchor=1.0),
        Line(words=[Word(text="general"), Word(text="kenobi")], anchor=2.0),
    ]
    lyrics = Lyrics(lines=lines, source="test")
    meta = SongMeta(artist="Band", title="Song", duration=3.0, source="x")
    stems = StemSet(vocals=song.audio, instrumental=song.audio, karaoke=song.audio)
    aln = run_alignment(meta, lyrics, stems, cfg)
    assert len(fake.align.calls) == 1
    _, sent_lines, language, strategy, _ = fake.align.calls[0]
    assert language == "eng" and strategy == "auto" and sent_lines[0]["anchor"] == 1.0
    assert aln.lrc_offset == 0.5 and aln.aligner == "ctc"
    assert all(w.timed for line in aln.lines for w in line.words)


def test_missing_extra_survives_pickling():
    import pickle

    from deepcuts.errors import MissingExtra

    exc = pickle.loads(pickle.dumps(MissingExtra("align", "ctc-forced-aligner")))
    assert isinstance(exc, MissingExtra) and "uv sync --extra align" in str(exc)
