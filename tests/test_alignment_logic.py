import pytest

from deepcuts.config import AlignConfig
from deepcuts.models import Line, Overrides, Word
from deepcuts.stages.align.anchors import estimate_offset, line_windows, pick_best
from deepcuts.stages.align.ctc import confidence
from deepcuts.stages.align.matching import match_words
from deepcuts.stages.align.postprocess import (
    apply_overrides,
    clamp_long_words,
    close_gaps,
    enforce_monotonic,
    fill_missing,
    finalize,
    low_confidence,
)


def W(text, start=None, end=None, conf=None):
    return Word(text=text, start=start, end=end, confidence=conf)


def test_fill_missing_interpolates_between_neighbours_weighted_by_length():
    line = Line(words=[W("a", 1.0, 1.2, 0.9), W("xx"), W("xxxx"), W("b", 2.0, 2.2, 0.9)])
    out = fill_missing([line])[0].words
    assert out[1].start == 1.2 and abs(out[1].end - (1.2 + 0.8 / 3)) < 1e-9
    assert abs(out[2].end - 2.0) < 1e-9
    assert out[1].confidence == 0.0


def test_fill_missing_trailing_run_does_not_stretch_into_break():
    lines = [
        Line(words=[W("a", 1.0, 1.5, 0.9), W("99")]),
        Line(words=[W("b", 30.0, 30.5, 0.9)]),
    ]
    out = fill_missing(lines)
    assert out[0].words[1].end <= 1.5 + 0.6 + 1e-9


def test_fill_missing_uses_anchor_for_fully_failed_line():
    lines = [Line(words=[W("x"), W("y")], anchor=10.0), Line(words=[W("z", 12.0, 12.4, 1)])]
    out = fill_missing(lines)
    assert out[0].words[0].start == 10.0 and out[0].words[1].end == 11.2


def test_clamp_long_words_trims_leading_silence_on_first_word():
    lines = [Line(words=[W("a", 0.0, 9.0), W("b", 9.0, 20.0)])]
    out = clamp_long_words(lines, 6.0)[0].words
    assert (out[0].start, out[0].end) == (3.0, 9.0)
    assert (out[1].start, out[1].end) == (9.0, 15.0)


def test_enforce_monotonic_and_min_duration():
    lines = [Line(words=[W("a", 1.0, 1.01), W("b", 0.9, 1.2), W("c", 1.1, 1.3)])]
    words = enforce_monotonic(lines, 0.08)[0].words
    starts = [w.start for w in words]
    assert starts == sorted(starts)
    for a, b in zip(words, words[1:], strict=False):
        assert a.end <= b.start + 1e-9
    assert all(w.end > w.start for w in words)


def test_close_gaps_only_small_ones():
    lines = [Line(words=[W("a", 0, 1), W("b", 1.2, 2), W("c", 3, 4)])]
    words = close_gaps(lines, 0.35)[0].words
    assert words[0].end == 1.2 and words[1].end == 2


def test_finalize_is_pure():
    lines = [Line(words=[W("a", 1, 2, 0.5), W("—")])]
    finalize(lines, AlignConfig())
    assert lines[0].words[1].start is None


def test_overrides_global_and_per_line():
    lines = [Line(words=[W("a", 1, 2)]), Line(words=[W("b", 3, 4)])]
    out = apply_overrides(lines, Overrides(global_offset_ms=-100, line_shift_ms={1: 250}))
    assert (out[0].words[0].start, out[1].words[0].start) == (0.9, 3.15)
    assert lines[0].words[0].start == 1


def test_overrides_parse_string_keys_from_json():
    ov = Overrides.model_validate_json('{"global_offset_ms": 5, "line_shift_ms": {"3": -40}}')
    assert ov.line_shift_ms == {3: -40}


def test_low_confidence():
    lines = [Line(words=[W("a", 0, 1, 0.9)]), Line(words=[W("b", 1, 2, 0.1)])]
    assert low_confidence(lines, 0.35) == [(1, 0.1)]


def test_match_words_handles_dropped_tokens():
    assert match_words(["a", "b", "c"], ["a", "b", "c"]) == [0, 1, 2]
    assert match_words(["Hello,", "big", "world"], ["hello", "world"]) == [0, None, 1]


def test_line_windows_padding_and_ends():
    wins = line_windows([10.0, 14.0, 30.0], [None, 18.0, None], 40.0, 0.5, 1.0)
    assert wins[0] == (9.5, 15.0)
    assert wins[1] == (13.5, 19.0)
    assert wins[2][0] == 29.5 and wins[2][1] <= 40.0


def test_estimate_offset_uses_confident_lines_median():
    lines = [
        Line(words=[W("a", 11.3, 12, 0.9)], anchor=10.0),
        Line(words=[W("b", 21.3, 22, 0.8)], anchor=20.0),
        Line(words=[W("c", 31.2, 32, 0.7)], anchor=30.0),
        Line(words=[W("d", 80.0, 81, 0.1)], anchor=40.0),  # junk, low confidence
    ]
    assert abs(estimate_offset(lines) - 1.3) < 1e-9


def test_pick_best_prefers_anchored_unless_clearly_worse():
    g = [Line(words=[W("a", 1, 2, 0.9)], anchor=0.0), Line(words=[W("b", 3, 4, 0.5)], anchor=2.0)]
    a = [
        Line(words=[W("a", 1.1, 2, 0.3)], anchor=1.0),
        Line(words=[W("b", 3.1, 4, 0.48)], anchor=3.0),
    ]
    out = pick_best(g, a)
    assert out[0].words[0].start == 1 and out[1].words[0].start == 3.1
    assert [line.anchor for line in out] == [1.0, 3.0]


def test_confidence_from_mean_logprob():
    assert confidence(None) is None
    assert abs(confidence(0.0) - 1.0) < 1e-9
    assert abs(confidence(-1.0) - 0.3679) < 1e-3
    assert 0 < confidence(-3.0) < 0.1


def test_low_confidence_flags_a_single_very_weak_word():
    # an overlapping second voice: decent mean, one word the aligner basically guessed
    crammed = Line(words=[W("happily", 0, 1, 0.6), W("said)", 1, 1.01, 0.05), W("x", 1.1, 2, 0.7)])
    clean = Line(words=[W("a", 0, 1, 0.9), W("b", 1, 2, 0.3)])
    assert low_confidence([clean, crammed], 0.35) == []
    [(i, mean)] = low_confidence([clean, crammed], 0.35, 0.2)
    assert i == 1 and abs(mean - 0.45) < 1e-9


def test_active_frames_threshold_follows_stem_loudness():
    from deepcuts.stages.align.activity import active_frames

    # 20 loud frames (-15 dB), 30 frames of digital silence, 20 loud frames again
    levels = [-15.0] * 20 + [-120.0] * 30 + [-15.0] * 20
    act = active_frames(levels, rel_db=35, hangover=0)
    assert act == [True] * 20 + [False] * 30 + [True] * 20
    # a quieter mix: the same shape 20 dB down is still detected the same way
    assert active_frames([v - 20 for v in levels], rel_db=35, hangover=0) == act
    # bleed far below the threshold never counts as singing
    assert not any(active_frames([-80.0] * 50, rel_db=35, hangover=0))


def test_active_frames_hangover_widens_both_sides():
    from deepcuts.stages.align.activity import active_frames

    levels = [-120.0] * 10 + [-10.0] * 5 + [-120.0] * 10 + [-10.0] * 50
    act = active_frames(levels, rel_db=35, hangover=2)
    assert act[:8] == [False] * 8 and act[8:10] == [True, True]  # opens 2 frames early
    assert act[15:17] == [True, True] and act[17:23] == [False] * 6  # holds 2 frames late


def _activity(spans, n=500, fs=0.02):
    return [any(a <= f * fs < b for a, b in spans) for f in range(n)]


def test_fit_to_activity_trims_padding_into_silence_and_holds_notes():
    from deepcuts.stages.align.activity import fit_to_activity

    fs = 0.02
    active = _activity([(3.0, 4.2), (8.0, 9.0)])  # singing, 3.8 s of silence, singing
    lines = [
        # what edge-padded CTC produced on the real song: words stretched across the gap
        Line(words=[W("bites", 3.0, 3.3, 0.7), W("at", 3.3, 5.9, 0.98), W("me", 5.9, 7.9, 0.97)]),
        # tight global-pass spans: a held note shows up as a 40 ms burst
        Line(words=[W("quail", 8.0, 8.04, 0.8)]),
    ]
    fit_to_activity(lines, active, fs)
    bites, at, me = lines[0].words
    assert bites.end == pytest.approx(3.3)  # already up against the next word
    assert at.end == pytest.approx(4.2)  # trimmed back to where the singing stops
    assert (me.start, me.end) == (5.9, 7.9)  # no singing in its span at all: left for review
    assert lines[1].words[0].end == pytest.approx(9.0)  # held through the note


def test_fit_to_activity_never_moves_words_on_a_silent_stem():
    from deepcuts.stages.align.activity import fit_to_activity

    lines = [Line(words=[W("ooh", 1.0, 2.0, 0.9)])]
    fit_to_activity(lines, [False] * 200, 0.02)
    assert (lines[0].words[0].start, lines[0].words[0].end) == (1.0, 2.0)


def test_clip_at_next_onset_stops_window_before_next_line():
    from deepcuts.stages.align.activity import clip_at_next_onset

    fs = 0.02
    # line A sung 59.9-64.2, silence, line B starts 72.64 (its anchor is 72.95)
    active = _activity([(59.9, 64.2), (72.64, 80.0)], n=4500, fs=fs)
    windows = [(59.5, 73.95), (72.35, 81.0)]
    out = clip_at_next_onset(windows, [60.1, 72.95], active, fs)
    assert out[0][0] == 59.5 and out[0][1] == pytest.approx(72.64, abs=fs)
    assert out[1] == windows[1]  # last line: nothing after it


def test_clip_at_next_onset_leaves_legato_lines_alone():
    from deepcuts.stages.align.activity import clip_at_next_onset

    active = _activity([(10.0, 30.0)], n=2000)  # continuous singing across the line change
    windows = [(9.4, 21.0), (19.4, 31.0)]
    assert clip_at_next_onset(windows, [10.0, 20.0], active, 0.02) == windows
