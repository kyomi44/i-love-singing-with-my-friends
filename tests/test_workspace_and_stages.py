from pathlib import Path

from deepcuts.config import Config, with_overrides
from deepcuts.models import Line, StemSet, Word
from deepcuts.stages.diarize import assign_singers
from deepcuts.stages.ingest import parse_filename
from deepcuts.stages.separate import classify_pair, stem_label
from deepcuts.workspace import SongDir, fingerprint, resolve_song, slugify


def test_slugify_strips_path_chars():
    assert slugify("AC/DC", 'What: "Now"?') == "AC_DC - What_ _Now__"


def test_fingerprint_changes_when_lyrics_edited(tmp_path):
    lyr = tmp_path / "lyrics.txt"
    lyr.write_text("one")
    a = fingerprint([lyr], {"x": 1})
    lyr.write_text("two")
    assert fingerprint([lyr], {"x": 1}) != a
    assert fingerprint([lyr], {"x": 2}) != fingerprint([lyr], {"x": 1})
    assert fingerprint([None, tmp_path / "missing.wav"], {}) == fingerprint(
        [None, tmp_path / "missing.wav"], {}
    )


def test_manifest_and_stems_roundtrip(tmp_path):
    song = SongDir(tmp_path / "Band - Song").create()
    song.mark("align", "abc")
    assert song.is_fresh("align", "abc") and not song.is_fresh("align", "xyz")
    stems = StemSet(
        vocals=song.stems_dir / "vocals.wav",
        instrumental=song.stems_dir / "instrumental.wav",
        karaoke=song.stems_dir / "karaoke.wav",
    )
    song.save_stems(stems)
    assert '"stems/vocals.wav"' in song.stems_path.read_text()
    assert song.load_stems().vocals == song.stems_dir / "vocals.wav"


def test_lyrics_file_replacement(tmp_path):
    song = SongDir(tmp_path / "s").create()
    song.write_lyrics("[00:01.00]hi", "lrc", "lrclib:1")
    song.write_lyrics("hi", "txt", "file:x")
    assert song.lyrics_file() == song.lyrics_txt and not song.lyrics_lrc.exists()
    assert song.lyrics_source() == "file:x"


def test_resolve_song_by_substring(tmp_path):
    for name in ("Band - First Song", "Band - Second Song"):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text("{}")
    assert resolve_song("second", tmp_path).name == "Band - Second Song"
    try:
        resolve_song("band", tmp_path)
    except FileNotFoundError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("expected ambiguity error")


def test_parse_filename():
    assert parse_filename("03 - Broadcast - Pendulum") == ("Broadcast", "Pendulum")
    assert parse_filename("Stereolab - French Disko") == ("Stereolab", "French Disko")
    assert parse_filename("untitled demo") == (None, "untitled demo")


def test_stem_classification_from_audio_separator_names():
    outs = [Path("vocals_(Instrumental)_model.wav"), Path("vocals_(Vocals)_model.wav")]
    assert stem_label(outs[0]) == "instrumental"
    lead, other = classify_pair(outs, ("lead", "vocal"), ("back", "instrum", "other", "karaoke"))
    assert lead.name.startswith("vocals_(Vocals)") and other == outs[0]
    custom = [Path("backing.wav"), Path("lead.wav")]
    lead, other = classify_pair(custom, ("lead", "vocal"), ("back",))
    assert (lead.name, other.name) == ("lead.wav", "backing.wav")
    melband = [Path("audio_(Other)_vocals_mel_band_roformer.wav"), Path("audio_(Vocals)_m.wav")]
    vox, inst = classify_pair(melband, ("vocal",), ("instrum", "other"), ("instrum", "other"))
    assert "(Vocals)" in vox.name and "(Other)" in inst.name
    mvsep = [Path("00_(Back vocals).flac"), Path("01_(Lead vocals).flac")]
    lead, other = classify_pair(mvsep, ("lead", "vocal"), ("back",), ("back",))
    assert "Lead" in lead.name and "Back" in other.name


def test_assign_singers_and_manual_tags_win():
    def line(s, e, singer=None):
        return Line(words=[Word(text="x", start=s, end=e)], singer=singer)

    lines = [line(0, 4), line(5, 9), line(10, 14), line(15, 19, singer="B")]
    turns = [
        (0, 4.5, "SPK_1"),
        (4.8, 9, "SPK_0"),
        (10, 14, "SPK_1"),
        (10.5, 14, "SPK_0"),
        (15, 19, "SPK_1"),
    ]
    out = assign_singers(lines, turns)
    assert [x.singer for x in out] == ["A", "B", "AB", "B"]


def test_config_overrides():
    cfg = with_overrides(Config(), **{"align.aligner": "whisperx", "render.guide_vocals": None})
    assert cfg.align.aligner == "whisperx" and cfg.render.guide_vocals == 0.0
