//! Cover art in the terminal: each cell is "▀" with the top pixel as the foreground colour and
//! the bottom pixel as the background, so one cell = two roughly square pixels.

use std::path::Path;

use image::{imageops::FilterType, RgbImage};
use ratatui::{buffer::Buffer, layout::Rect, style::Color};

pub struct Cover {
    image: RgbImage,
    cache: Option<(u16, u16, RgbImage)>,
}

impl Cover {
    pub fn load(path: &Path) -> Option<Cover> {
        let image = image::open(path).ok()?.to_rgb8();
        Some(Cover { image, cache: None })
    }

    /// Size in cells that fits in `max_w` x `max_h` keeping the aspect ratio.
    pub fn fit(&self, max_w: u16, max_h: u16) -> (u16, u16) {
        let (iw, ih) = (self.image.width() as f64, self.image.height() as f64);
        // width in cells = pixel width; height in cells = pixel height / 2
        let w_for_h = (max_h as f64 * 2.0 * iw / ih).floor() as u16;
        if w_for_h <= max_w {
            (w_for_h.max(1), max_h.max(1))
        } else {
            let h = (max_w as f64 * ih / iw / 2.0).floor() as u16;
            (max_w.max(1), h.max(1))
        }
    }

    pub fn draw(&mut self, area: Rect, buf: &mut Buffer, truecolor: bool) {
        let (w, h) = (area.width, area.height);
        if w == 0 || h == 0 {
            return;
        }
        let stale = !matches!(&self.cache, Some((cw, ch, _)) if *cw == w && *ch == h);
        if stale {
            let px =
                image::imageops::resize(&self.image, w as u32, h as u32 * 2, FilterType::Triangle);
            self.cache = Some((w, h, px));
        }
        let px = &self.cache.as_ref().unwrap().2;
        for y in 0..h {
            for x in 0..w {
                let top = px.get_pixel(x as u32, y as u32 * 2).0;
                let bottom = px.get_pixel(x as u32, y as u32 * 2 + 1).0;
                if let Some(cell) = buf.cell_mut((area.x + x, area.y + y)) {
                    cell.set_symbol("▀")
                        .set_fg(color(top, truecolor))
                        .set_bg(color(bottom, truecolor));
                }
            }
        }
    }
}

/// 24-bit colour where the terminal supports it, else the nearest xterm-256 colour
/// (macOS Terminal.app, for one, has no truecolor).
pub fn color([r, g, b]: [u8; 3], truecolor: bool) -> Color {
    if truecolor {
        return Color::Rgb(r, g, b);
    }
    let cube = |v: u8| -> u8 {
        if v < 48 {
            0
        } else if v < 115 {
            1
        } else {
            (v - 35) / 40
        }
    };
    let (cr, cg, cb) = (cube(r), cube(g), cube(b));
    let level = |c: u8| if c == 0 { 0 } else { 55 + 40 * c as i32 };
    let cube_idx = 16 + 36 * cr + 6 * cg + cb;
    let cube_err = (level(cr) - r as i32).pow(2)
        + (level(cg) - g as i32).pow(2)
        + (level(cb) - b as i32).pow(2);
    let avg = (r as i32 + g as i32 + b as i32) / 3;
    let gi = if avg > 238 {
        23
    } else {
        ((avg - 3).max(0) / 10) as u8
    };
    let gl = 8 + 10 * gi as i32;
    let gray_err = (gl - r as i32).pow(2) + (gl - g as i32).pow(2) + (gl - b as i32).pow(2);
    Color::Indexed(if gray_err < cube_err {
        232 + gi
    } else {
        cube_idx
    })
}

pub fn terminal_has_truecolor() -> bool {
    std::env::var("COLORTERM").is_ok_and(|v| v.contains("truecolor") || v.contains("24bit"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xterm256_maps_primaries_and_greys_sensibly() {
        assert_eq!(color([0, 0, 0], false), Color::Indexed(16));
        assert_eq!(color([255, 0, 0], false), Color::Indexed(196));
        assert_eq!(color([128, 128, 128], false), Color::Indexed(244));
        assert_eq!(color([1, 2, 3], true), Color::Rgb(1, 2, 3));
    }
}
