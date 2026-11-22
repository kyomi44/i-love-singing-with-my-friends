//! Big lyrics: an 8x8 bitmap font drawn with block characters, so the words can be much larger
//! than the terminal's own font (which a program can't change).
//!
//!   size 1  plain text
//!   size 2  quadrant blocks (2x2 pixels a cell)  -> 4 x 4 cells a letter, condensed
//!   size 3  half blocks     (1x2 pixels a cell)  -> 8 x 4 cells a letter
//!   size 4  full blocks     (1x1 pixel a cell)   -> 8 x 8 cells a letter

use font8x8::{UnicodeFonts, BASIC_FONTS, LATIN_FONTS};
use ratatui::{buffer::Buffer, style::Style};

pub const MAX_SIZE: u8 = 4;

/// Cells per letter (width, height) at a size >= 2.
pub fn dims(size: u8) -> (u16, u16) {
    match size {
        2 => (4, 4),
        3 => (8, 4),
        _ => (8, 8),
    }
}

/// Typographic characters lyrics sites love, folded to what the font has.
fn fold(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{2032}' => '\'',
        '\u{201C}' | '\u{201D}' => '"',
        '\u{2013}' | '\u{2014}' => '-',
        '\u{2026}' => '.',
        _ => c,
    }
}

/// Rows of the letter's 8x8 bitmap; bit 0 of each row is the leftmost pixel.
pub fn glyph(c: char) -> [u8; 8] {
    let c = fold(c);
    BASIC_FONTS
        .get(c)
        .or_else(|| LATIN_FONTS.get(c))
        .or_else(|| BASIC_FONTS.get('?'))
        .unwrap_or([0; 8])
}

fn px(g: &[u8; 8], x: u16, y: u16) -> bool {
    x < 8 && y < 8 && g[y as usize] & (1 << x) != 0
}

// Quadrant blocks indexed by (top-left, top-right, bottom-left, bottom-right) bits.
const QUADS: [&str; 16] = [
    " ", "▘", "▝", "▀", "▖", "▌", "▞", "▛", "▗", "▚", "▐", "▜", "▄", "▙", "▟", "█",
];

/// Draw one letter with its top-left cell at (x, y). `style_at` gets the horizontal position
/// (0..1) of each cell's centre within the letter, which is how the wipe sweeps across it.
pub fn draw_glyph(
    buf: &mut Buffer,
    x: u16,
    y: u16,
    size: u8,
    c: char,
    style_at: &dyn Fn(f64) -> Style,
) {
    let g = glyph(c);
    let (w, h) = dims(size);
    for cy in 0..h {
        for cx in 0..w {
            let sym = match size {
                2 => {
                    let (px0, py0) = (cx * 2, cy * 2);
                    let bits = px(&g, px0, py0) as usize
                        | (px(&g, px0 + 1, py0) as usize) << 1
                        | (px(&g, px0, py0 + 1) as usize) << 2
                        | (px(&g, px0 + 1, py0 + 1) as usize) << 3;
                    QUADS[bits]
                }
                3 => match (px(&g, cx, cy * 2), px(&g, cx, cy * 2 + 1)) {
                    (true, true) => "█",
                    (true, false) => "▀",
                    (false, true) => "▄",
                    (false, false) => " ",
                },
                _ => {
                    if px(&g, cx, cy) {
                        "█"
                    } else {
                        " "
                    }
                }
            };
            if sym == " " {
                continue;
            }
            if let Some(cell) = buf.cell_mut((x + cx, y + cy)) {
                cell.set_symbol(sym)
                    .set_style(style_at((cx as f64 + 0.5) / w as f64));
            }
        }
    }
}

/// Break `chars` into lines of at most `max` characters, at spaces where possible.
/// Returns (start, end) ranges with the breaking spaces left out.
pub fn wrap(chars: &[char], max: usize) -> Vec<(usize, usize)> {
    let max = max.max(1);
    let mut out = Vec::new();
    let mut start = 0;
    let n = chars.len();
    while start < n {
        while start < n && chars[start] == ' ' {
            start += 1;
        }
        if start >= n {
            break;
        }
        if n - start <= max {
            out.push((start, n));
            break;
        }
        let limit = start + max;
        match (start + 1..=limit).rev().find(|&i| chars[i] == ' ') {
            Some(sp) => {
                out.push((start, sp));
                start = sp + 1;
            }
            None => {
                out.push((start, limit)); // one word longer than the line: hard break
                start = limit;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn wrap_breaks_at_spaces_and_hard_breaks_long_words() {
        let c = chars("Salad days aren't balanced");
        let lines: Vec<String> = wrap(&c, 12)
            .into_iter()
            .map(|(a, b)| c[a..b].iter().collect())
            .collect();
        assert_eq!(lines, vec!["Salad days", "aren't", "balanced"]);
        assert_eq!(wrap(&chars("abcdefgh"), 3), vec![(0, 3), (3, 6), (6, 8)]);
        assert_eq!(wrap(&chars("fits"), 10), vec![(0, 4)]);
    }

    #[test]
    fn curly_apostrophes_render_like_straight_ones() {
        assert_eq!(glyph('\u{2019}'), glyph('\''));
        assert_ne!(glyph('A'), [0; 8]);
        assert_eq!(glyph('é'), LATIN_FONTS.get('é').unwrap());
    }

    #[test]
    fn letters_are_drawn_with_the_wipe_splitting_columns() {
        use ratatui::layout::Rect;
        use ratatui::style::Color;
        let mut buf = Buffer::empty(Rect::new(0, 0, 8, 8));
        let sung = Style::default().fg(Color::Cyan);
        let unsung = Style::default().fg(Color::White);
        draw_glyph(&mut buf, 0, 0, 4, 'H', &|x| {
            if x < 0.5 {
                sung
            } else {
                unsung
            }
        });
        // 'H' has a full-height left stem; its cells are on the sung side of the wipe
        let stem: Vec<_> = (0..7).map(|y| buf[(0, y)].symbol().to_string()).collect();
        assert!(stem.iter().all(|s| s == "█"));
        assert_eq!(buf[(0, 0)].fg, Color::Cyan);
        let right = (4..8).find(|&x| buf[(x, 3)].symbol() == "█").unwrap();
        assert_eq!(buf[(right, 3)].fg, Color::White);
    }
}
