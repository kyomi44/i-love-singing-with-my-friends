//! Drawing: the song picker and the karaoke screen.

use ratatui::{
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line as TLine, Span},
    widgets::{Gauge, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::cover::{color, Cover};
use crate::lyrics::{countdown, visible, wipe};
use crate::song::Song;

pub struct Look {
    pub truecolor: bool,
}

impl Look {
    fn rgb(&self, hex: u32) -> Color {
        color(
            [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8],
            self.truecolor,
        )
    }
    /// Same colours as the video (render.sung_color / singer_colors).
    fn sung(&self, singer: Option<&str>) -> Color {
        match singer {
            Some("B") => self.rgb(0xF48FB1),
            Some("AB") => self.rgb(0xCE93D8),
            _ => self.rgb(0x4FC3F7),
        }
    }
    fn dim(&self) -> Color {
        self.rgb(0x8A8A99)
    }
}

pub struct Status {
    pub paused: bool,
    pub guide: Option<bool>, // None = no vocal stem
    pub sync_ms: i64,
    pub duration: f64,
    pub ended: bool,
    pub flash: Option<String>,
}

fn mmss(t: f64) -> String {
    let s = t.max(0.0) as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

pub fn draw_player(
    f: &mut Frame,
    song: &Song,
    cover: Option<&mut Cover>,
    look: &Look,
    t: f64,
    st: &Status,
) {
    let area = f.area();
    let [top, mid, bar, keys] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(5),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(area);

    let meta = &song.timing.meta;
    let mut right = Vec::new();
    if st.guide == Some(true) {
        right.push(Span::styled(
            " ♪ guide vocals ",
            Style::default().fg(look.sung(None)),
        ));
    }
    if st.sync_ms != 0 {
        right.push(Span::styled(
            format!(" sync {:+} ms ", st.sync_ms),
            Style::default().fg(look.dim()),
        ));
    }
    if st.paused && !st.ended {
        right.push(Span::styled(
            " ❚❚ paused ",
            Style::default().add_modifier(Modifier::BOLD),
        ));
    }
    let title = TLine::from(vec![
        Span::styled(
            format!(" {} ", meta.title),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("— {}", meta.artist),
            Style::default().fg(look.dim()),
        ),
    ]);
    f.render_widget(Paragraph::new(title), top);
    f.render_widget(
        Paragraph::new(TLine::from(right)).alignment(Alignment::Right),
        top,
    );

    if st.ended {
        draw_center(
            f,
            mid,
            vec![
                TLine::styled(
                    "♪  fin  ♪",
                    Style::default()
                        .fg(look.sung(None))
                        .add_modifier(Modifier::BOLD),
                ),
                TLine::raw(""),
                TLine::styled(
                    "r  sing it again     q  quit",
                    Style::default().fg(look.dim()),
                ),
            ],
        );
    } else if t < song.intro_end() {
        draw_intro(f, mid, song, cover, look);
    } else {
        draw_lyrics(f, mid, song, look, t);
    }

    let ratio = (t / st.duration.max(0.001)).clamp(0.0, 1.0);
    let label = st
        .flash
        .clone()
        .unwrap_or_else(|| format!("{} / {}", mmss(t), mmss(st.duration)));
    f.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(look.sung(None)).bg(look.rgb(0x22222A)))
            .ratio(ratio)
            .label(Span::styled(
                label,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
        bar,
    );
    let mut hint =
        String::from(" space pause  ←/→ 5s  [/] line  r restart  +/- volume  ,/. sync  q quit");
    if st.guide.is_some() {
        hint.push_str("  v guide vocals");
    }
    f.render_widget(
        Paragraph::new(hint).style(Style::default().fg(look.dim())),
        keys,
    );
}

fn draw_center(f: &mut Frame, area: Rect, lines: Vec<TLine>) {
    let h = lines.len() as u16;
    let y = area.y + area.height.saturating_sub(h) / 2;
    let r = Rect {
        x: area.x,
        y,
        width: area.width,
        height: h.min(area.height),
    };
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), r);
}

fn draw_intro(f: &mut Frame, area: Rect, song: &Song, cover: Option<&mut Cover>, look: &Look) {
    let meta = &song.timing.meta;
    let text = vec![
        TLine::styled(
            meta.title.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        TLine::styled(meta.artist.clone(), Style::default().fg(look.dim())),
    ];
    let Some(cover) = cover else {
        return draw_center(f, area, text);
    };
    let max_h = area.height.saturating_sub(4);
    let (w, h) = cover.fit(area.width.saturating_sub(4), (max_h as f64 * 0.95) as u16);
    let total = h + 3;
    let y0 = area.y + area.height.saturating_sub(total) / 2;
    let art = Rect {
        x: area.x + area.width.saturating_sub(w) / 2,
        y: y0,
        width: w,
        height: h,
    };
    cover.draw(art, f.buffer_mut(), look.truecolor);
    let below = Rect {
        x: area.x,
        y: y0 + h + 1,
        width: area.width,
        height: 2,
    };
    f.render_widget(Paragraph::new(text).alignment(Alignment::Center), below);
}

fn draw_lyrics(f: &mut Frame, area: Rect, song: &Song, look: &Look, t: f64) {
    let lines = &song.timing.lines;
    let t_words = t + song.lead();
    // countdown, row 0 (up to 2 terminal lines), gap, row 1
    let block_h = 1 + 2 + 1 + 2;
    let y0 = area.y + area.height.saturating_sub(block_h) / 2;
    let rect = |dy: u16, h: u16| Rect {
        x: area.x + 2,
        y: y0 + dy,
        width: area.width.saturating_sub(4),
        height: h,
    };

    if let Some(k) = countdown(lines, t) {
        let dots = vec!["●"; k as usize].join(" ");
        f.render_widget(
            Paragraph::new(Span::styled(dots, Style::default().fg(look.sung(None))))
                .alignment(Alignment::Center),
            rect(0, 1),
        );
    }
    for (row, i) in visible(lines, t) {
        let line = &lines[i];
        let sung = look.sung(line.singer.as_deref());
        let spans: Vec<Span> = wipe(&line.words, t_words)
            .into_iter()
            .map(|p| {
                let mut s = if p.sung {
                    Style::default().fg(sung).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                if p.italic {
                    s = s.add_modifier(Modifier::ITALIC);
                }
                Span::styled(p.text, s)
            })
            .collect();
        let dy = if row == 0 { 1 } else { 4 };
        f.render_widget(
            Paragraph::new(TLine::from(spans))
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true }),
            rect(dy, 2),
        );
    }
}

pub fn draw_picker(
    f: &mut Frame,
    names: &[String],
    state: &mut ListState,
    library: &str,
    look: &Look,
) {
    let area = f.area();
    let [head, list, keys] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .areas(area);
    f.render_widget(
        Paragraph::new(vec![
            TLine::styled(
                " ♪ deepcuts — pick a song",
                Style::default()
                    .fg(look.sung(None))
                    .add_modifier(Modifier::BOLD),
            ),
            TLine::styled(format!("   {library}"), Style::default().fg(look.dim())),
        ]),
        head,
    );
    if names.is_empty() {
        f.render_widget(
            Paragraph::new("   No rendered songs here yet. Make one with: deepcuts make song.flac")
                .style(Style::default().fg(look.dim())),
            list,
        );
    } else {
        let items: Vec<ListItem> = names
            .iter()
            .map(|n| ListItem::new(format!("  {n}")))
            .collect();
        let list_w = List::new(items)
            .highlight_style(
                Style::default()
                    .fg(look.sung(None))
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(" ▶");
        f.render_stateful_widget(list_w, list, state);
    }
    f.render_widget(
        Paragraph::new(" ↑/↓ choose   enter sing   q quit").style(Style::default().fg(look.dim())),
        keys,
    );
}
