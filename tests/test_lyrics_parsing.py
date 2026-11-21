from deepcuts.models import Line, SongMeta, Word
from deepcuts.stages.lyrics.lrc import dump_lrc, fmt_time, is_lrc, parse_lrc
from deepcuts.stages.lyrics.lrclib import pick_candidate
from deepcuts.text import is_alignable, lines_to_text, parse_plain


def test_plain_sections_stanzas_singers_comments():
    text = """# a comment
[Verse 1]
@A I walked the line
Through the night

[Chorus: Both]
@AB And we sang
@B (ooh, ooh)
"""
    lines = parse_plain(text)
    assert [line.text for line in lines] == [
        "I walked the line",
        "Through the night",
        "And we sang",
        "(ooh, ooh)",
    ]
    assert [line.singer for line in lines] == ["A", None, "AB", "B"]
    assert [line.stanza for line in lines] == [0, 0, 1, 1]


def test_plain_roundtrip():
    lines = parse_plain("@A one two\nthree\n\nfour")
    assert parse_plain(lines_to_text(lines)) == lines


def test_lrc_basic_repeat_and_meta():
    text = """[ar:Someone]
[ti:Song]
[length: 03:21]
[00:12.34]First line
[00:15.00][01:30.50]Chorus line
[00:20.5]Short fraction
"""
    lines = parse_lrc(text)
    assert [(round(line.anchor, 2), line.text) for line in lines] == [
        (12.34, "First line"),
        (15.0, "Chorus line"),
        (20.5, "Short fraction"),
        (90.5, "Chorus line"),
    ]
    assert is_lrc(text)
    assert not is_lrc("just words\nmore words")


def test_lrc_break_markers_set_anchor_end_and_stanza():
    lines = parse_lrc("[00:01.00]one\n[00:04.00]\n[00:10.00]two\n[00:12.00]♪\n[00:20.00]three")
    assert lines[0].anchor_end == 4.0
    assert lines[1].anchor_end == 12.0
    assert [line.stanza for line in lines] == [0, 1, 2]


def test_lrc_enhanced_word_tags():
    lines = parse_lrc("[00:10.00]<00:10.00> Hello <00:10.50> big world <00:11.40>")
    words = lines[0].words
    assert [w.text for w in words] == ["Hello", "big", "world"]
    assert (words[0].start, words[0].end) == (10.0, 10.5)
    assert words[1].start == 10.5 and abs(words[2].start - 10.95) < 1e-9  # shared tag, split evenly
    assert words[2].end == 11.4
    tail = parse_lrc("[00:01.00]<00:01.00> no closing tag")[0].words
    assert tail[0].start == 1.0 and tail[0].end is None


def test_fmt_time_never_rolls_to_60():
    assert fmt_time(59.996) == "01:00.00"
    assert fmt_time(61.234) == "01:01.23"


def test_dump_lrc_roundtrip():
    meta = SongMeta(artist="A", title="T", duration=200, source="x")
    line = Line(
        words=[Word(text="hi", start=1.0, end=1.5), Word(text="there", start=1.5, end=2.25)],
        singer="B",
    )
    text = dump_lrc([line], meta)
    back = parse_lrc(text)[0]
    assert back.singer == "B"
    assert [(w.text, w.start) for w in back.words] == [("hi", 1.0), ("there", 1.5)]
    assert back.words[-1].end == 2.25


def test_is_alignable():
    assert is_alignable("don't") and is_alignable("ça") and is_alignable("(ooh")
    assert not is_alignable("—") and not is_alignable("99") and not is_alignable("&")


def test_pick_candidate_prefers_synced_within_tolerance():
    cands = [
        {"id": 1, "duration": 200, "plainLyrics": "x"},
        {"id": 2, "duration": 250, "syncedLyrics": "[00:01.00]x"},
        {"id": 3, "duration": 201.5, "syncedLyrics": "[00:01.00]x"},
    ]
    assert pick_candidate(cands, 200, 3.0)["id"] == 3
    assert pick_candidate(cands, None, 3.0)["id"] in (2, 3)
    assert pick_candidate([], 200, 3.0) is None
