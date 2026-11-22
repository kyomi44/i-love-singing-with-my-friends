//! What's on screen at time t: pure functions over the song's timing, unit-tested below.

use crate::song::{Line, Word};

/// One run of text with the same look.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub text: String,
    pub sung: bool,
    pub italic: bool,
}

/// Split a line into sung / unsung pieces at time `t`, wiping through the current word letter
/// by letter (like the video's `\kf`). Text in parentheses is italic, as in the video.
pub fn wipe(words: &[Word], t: f64) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    let mut push = |text: &str, sung: bool, italic: bool| {
        if text.is_empty() {
            return;
        }
        match out.last_mut() {
            Some(p) if p.sung == sung && p.italic == italic => p.text.push_str(text),
            _ => out.push(Piece {
                text: text.to_string(),
                sung,
                italic,
            }),
        }
    };
    let mut in_paren = false;
    for (i, w) in words.iter().enumerate() {
        let italic = in_paren || w.text.starts_with('(');
        let chars: Vec<char> = w.text.chars().collect();
        let n = match (w.start, w.end) {
            (Some(_), Some(e)) if t >= e => chars.len(),
            (Some(s), Some(e)) if t > s && e > s => {
                ((chars.len() as f64) * (t - s) / (e - s)).floor() as usize
            }
            (Some(_), Some(_)) => 0,
            _ => 0, // untimed word: never lights up
        };
        let sung: String = chars[..n].iter().collect();
        let rest: String = chars[n..].iter().collect();
        push(&sung, true, italic);
        push(&rest, false, italic);
        if w.text.contains('(') {
            in_paren = true;
        }
        if w.text.contains(')') {
            in_paren = false;
        }
        if i + 1 < words.len() {
            // the gap lights up once the next word has started
            let next_started = words[i + 1].start.is_some_and(|s| t >= s);
            push(" ", next_started, in_paren);
        }
    }
    out
}

/// Lines on screen at `t`, as (row, line index), in row order.
pub fn visible(lines: &[Line], t: f64) -> Vec<(u8, usize)> {
    let mut v: Vec<(u8, usize)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            let s = l.display.as_ref()?;
            (s.appear <= t && t < s.disappear).then_some((s.row, i))
        })
        .collect();
    v.sort();
    v
}

/// Countdown dots showing at `t` (each step lasts one second), if any.
pub fn countdown(lines: &[Line], t: f64) -> Option<u8> {
    lines
        .iter()
        .filter_map(|l| l.display.as_ref())
        .flat_map(|s| s.countdown.iter())
        .find(|(t0, _)| *t0 <= t && t < t0 + 1.0)
        .map(|(_, k)| *k)
}

fn line_start(l: &Line) -> Option<f64> {
    l.words.iter().find_map(|w| w.start)
}

/// Where `[` / `]` jump: 1.5 s before the previous / next line's first word.
pub fn line_jump(lines: &[Line], t: f64, forward: bool) -> Option<f64> {
    let starts: Vec<f64> = lines.iter().filter_map(line_start).collect();
    let target = if forward {
        starts.iter().copied().find(|s| *s - 1.5 > t + 0.05)
    } else {
        // the line we're in (or just before) restarts it; tap again for the previous one
        starts.iter().copied().rev().find(|s| *s - 1.5 < t - 0.5)
    };
    target.map(|s| (s - 1.5).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::song::Slot;

    fn w(text: &str, s: f64, e: f64) -> Word {
        Word {
            text: text.into(),
            start: Some(s),
            end: Some(e),
        }
    }

    fn texts(p: &[Piece]) -> Vec<(&str, bool)> {
        p.iter().map(|x| (x.text.as_str(), x.sung)).collect()
    }

    #[test]
    fn wipe_lights_whole_words_and_part_of_the_current_one() {
        let words = vec![
            w("Bites", 60.0, 60.6),
            w("at", 60.6, 62.1),
            w("me", 62.1, 64.0),
        ];
        assert_eq!(texts(&wipe(&words, 59.0)), vec![("Bites at me", false)]);
        // halfway through "at": 1 of 2 letters
        assert_eq!(
            texts(&wipe(&words, 61.35)),
            vec![("Bites a", true), ("t me", false)]
        );
        assert_eq!(texts(&wipe(&words, 64.0)), vec![("Bites at me", true)]);
    }

    #[test]
    fn wipe_italicises_parentheticals_like_the_video() {
        let words = vec![
            w("Happily", 0.0, 1.0),
            w("(there's", 1.0, 2.0),
            w("no)", 2.0, 3.0),
        ];
        let p = wipe(&words, 0.0);
        assert_eq!(
            p[0],
            Piece {
                text: "Happily ".into(),
                sung: false,
                italic: false
            }
        );
        assert_eq!(
            p[1],
            Piece {
                text: "(there's no)".into(),
                sung: false,
                italic: true
            }
        );
    }

    #[test]
    fn untimed_words_never_light_up() {
        let words = vec![
            w("a", 0.0, 1.0),
            Word {
                text: "b".into(),
                start: None,
                end: None,
            },
        ];
        assert_eq!(texts(&wipe(&words, 5.0)), vec![("a", true), (" b", false)]);
    }

    fn line(start: f64, row: u8, appear: f64, disappear: f64, countdown: Vec<(f64, u8)>) -> Line {
        Line {
            words: vec![w("x", start, start + 1.0)],
            singer: None,
            display: Some(Slot {
                row,
                appear,
                disappear,
                countdown,
            }),
        }
    }

    #[test]
    fn visible_lines_and_countdown_follow_the_schedule() {
        let lines = vec![
            line(
                12.88,
                0,
                11.68,
                21.88,
                vec![(9.88, 3), (10.88, 2), (11.88, 1)],
            ),
            line(21.48, 1, 20.28, 24.0, vec![]),
        ];
        assert!(visible(&lines, 11.0).is_empty());
        assert_eq!(visible(&lines, 21.0), vec![(0, 0), (1, 1)]);
        assert_eq!(countdown(&lines, 10.0), Some(3));
        assert_eq!(countdown(&lines, 12.5), Some(1));
        assert_eq!(countdown(&lines, 13.0), None);
    }

    #[test]
    fn line_jumps_go_to_just_before_a_line() {
        let lines = vec![
            line(12.88, 0, 11.7, 21.9, vec![]),
            line(21.48, 1, 20.3, 24.0, vec![]),
        ];
        assert_eq!(line_jump(&lines, 0.0, true), Some(12.88 - 1.5));
        assert_eq!(line_jump(&lines, 15.0, true), Some(21.48 - 1.5));
        assert_eq!(line_jump(&lines, 15.0, false), Some(12.88 - 1.5)); // restart current line
        assert_eq!(line_jump(&lines, 30.0, true), None);
    }
}
