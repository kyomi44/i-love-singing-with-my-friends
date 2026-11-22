//! deepcuts-sing: karaoke in the terminal for songs rendered by deepcuts.
//!
//!     deepcuts-sing                 # pick a song from your deepcuts library
//!     deepcuts-sing salem           # any unique part of a song folder's name, or a path

mod audio;
mod cover;
mod lyrics;
mod song;
mod ui;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::widgets::ListState;
use ratatui::DefaultTerminal;

use crate::audio::Player;
use crate::cover::{terminal_has_truecolor, Cover};
use crate::song::{library_dir, rendered_songs, resolve, Song};
use crate::ui::{Look, Status};

#[derive(Parser)]
#[command(version, about = "Sing deepcuts karaoke in the terminal.")]
struct Args {
    /// Song folder, or any unique part of its name. Omit to pick from the library.
    song: Option<String>,
    /// Song library (default: `library` in ~/.config/deepcuts/config.toml, else ~/deepcuts).
    #[arg(long)]
    library: Option<PathBuf>,
    /// Start at this many seconds in.
    #[arg(long, default_value_t = 0.0)]
    start: f64,
    /// Show lyrics this many ms later, for output latency (Bluetooth headphones: ~150-250).
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    sync_ms: i64,
    /// No audio device: the lyrics follow the wall clock (for testing, or singing to a record).
    #[arg(long)]
    silent: bool,
    /// Start with the guide vocal on.
    #[arg(long)]
    guide: bool,
    /// Starting volume in percent (0-150).
    #[arg(long, default_value_t = 100)]
    volume: u16,
}

enum Next {
    Quit,
    BackToPicker,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let library = library_dir(args.library.as_deref());
    let look = Look {
        truecolor: terminal_has_truecolor(),
    };

    // resolve a named song before touching the terminal, so errors print normally
    let direct = match &args.song {
        Some(q) => Some(Song::load(&resolve(q, &library)?)?),
        None => None,
    };
    let mut terminal = ratatui::init();
    let result = (|| -> Result<()> {
        if let Some(song) = direct {
            play(&mut terminal, song, &args, &look)?;
            return Ok(());
        }
        let mut state = ListState::default().with_selected(Some(0));
        loop {
            let Some(dir) = pick(&mut terminal, &library, &mut state, &look)? else {
                return Ok(());
            };
            let song = Song::load(&dir)?;
            if let Next::Quit = play(&mut terminal, song, &args, &look)? {
                return Ok(());
            }
        }
    })();
    ratatui::restore();
    result
}

fn pick(
    terminal: &mut DefaultTerminal,
    library: &std::path::Path,
    state: &mut ListState,
    look: &Look,
) -> Result<Option<PathBuf>> {
    let songs = rendered_songs(library);
    let names: Vec<String> = songs
        .iter()
        .map(|p| {
            p.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    let lib = library.display().to_string();
    loop {
        terminal.draw(|f| ui::draw_picker(f, &names, state, &lib, look))?;
        if let Event::Key(k) = event::read()? {
            if k.kind != KeyEventKind::Press {
                continue;
            }
            match k.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(None)
                }
                KeyCode::Down | KeyCode::Char('j') => state.select_next(),
                KeyCode::Up | KeyCode::Char('k') => state.select_previous(),
                KeyCode::Enter if !songs.is_empty() => {
                    let i = state.selected().unwrap_or(0).min(songs.len() - 1);
                    return Ok(Some(songs[i].clone()));
                }
                _ => {}
            }
        }
    }
}

fn play(terminal: &mut DefaultTerminal, song: Song, args: &Args, look: &Look) -> Result<Next> {
    let mut player = Player::open(&song.karaoke(), song.vocals().as_deref(), args.silent)?;
    let mut cover = song.cover().and_then(|p| Cover::load(&p));
    let mut sync_ms = args.sync_ms;
    let mut flash: Option<(String, Instant)> = None;
    player.nudge_volume(args.volume.min(150) as f32 / 100.0 - 1.0);
    if args.guide && player.has_vocals() {
        player.toggle_guide();
    }
    player.seek(args.start);
    player.set_paused(false);

    loop {
        let duration = player.duration().min(song.timing.meta.duration.max(1.0));
        let t_audio = player.time();
        let ended = t_audio >= duration - 0.05;
        if ended && !player.paused() {
            player.set_paused(true);
        }
        let t = t_audio - sync_ms as f64 / 1000.0;
        let status = Status {
            paused: player.paused(),
            guide: player.has_vocals().then(|| player.guide()),
            sync_ms,
            duration,
            ended,
            flash: flash
                .as_ref()
                .filter(|(_, at)| at.elapsed() < Duration::from_millis(1200))
                .map(|(m, _)| m.clone()),
        };
        terminal.draw(|f| ui::draw_player(f, &song, cover.as_mut(), look, t, &status))?;

        if !event::poll(Duration::from_millis(16))? {
            continue;
        }
        let Event::Key(k) = event::read()? else {
            continue;
        };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        let mut say = |m: String| flash = Some((m, Instant::now()));
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                return Ok(if args.song.is_some() {
                    Next::Quit
                } else {
                    Next::BackToPicker
                });
            }
            KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                return Ok(Next::Quit)
            }
            KeyCode::Char(' ') if !ended => {
                let p = !player.paused();
                player.set_paused(p);
            }
            KeyCode::Char('r') => {
                player.seek(0.0);
                player.set_paused(false);
            }
            KeyCode::Left => player.seek(t_audio - 5.0),
            KeyCode::Right => player.seek(t_audio + 5.0),
            KeyCode::Char('[') | KeyCode::Char(']') => {
                let fwd = k.code == KeyCode::Char(']');
                if let Some(to) = lyrics::line_jump(&song.timing.lines, t, fwd) {
                    player.seek(to + sync_ms as f64 / 1000.0);
                    if ended {
                        player.set_paused(false);
                    }
                }
            }
            KeyCode::Char('v') if player.has_vocals() => {
                player.toggle_guide();
                say(format!(
                    "guide vocals {}",
                    if player.guide() { "on" } else { "off" }
                ));
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                player.nudge_volume(0.1);
                say(format!("volume {:.0}%", player.volume() * 100.0));
            }
            KeyCode::Char('-') => {
                player.nudge_volume(-0.1);
                say(format!("volume {:.0}%", player.volume() * 100.0));
            }
            KeyCode::Char(',') => {
                sync_ms -= 20;
                say(format!("lyrics {:+} ms", sync_ms));
            }
            KeyCode::Char('.') => {
                sync_ms += 20;
                say(format!("lyrics {:+} ms", sync_ms));
            }
            _ => {}
        }
    }
}
