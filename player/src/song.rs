//! A rendered deepcuts song folder: `out/timing.json` plus the stems and cover it points at.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub artist: String,
    pub title: String,
    pub duration: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Word {
    pub text: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

/// How the video lays the line out (deepcuts `render/ass.py::display_schedule`).
#[derive(Debug, Clone, Deserialize)]
pub struct Slot {
    pub row: u8,
    pub appear: f64,
    pub disappear: f64,
    #[serde(default)]
    pub countdown: Vec<(f64, u8)>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Line {
    pub words: Vec<Word>,
    #[serde(default)]
    pub singer: Option<String>,
    #[serde(default)]
    pub display: Option<Slot>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Display {
    #[serde(default)]
    pub highlight_lead_ms: i64,
    #[serde(default)]
    pub intro_end: Option<f64>,
    #[serde(default)]
    pub cover: Option<String>,
    #[serde(default)]
    pub karaoke: Option<String>,
    #[serde(default)]
    pub vocals: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Timing {
    pub meta: Meta,
    #[serde(default)]
    pub display: Option<Display>,
    pub lines: Vec<Line>,
}

#[derive(Debug, Clone)]
pub struct Song {
    pub root: PathBuf,
    pub timing: Timing,
}

impl Song {
    pub fn load(root: &Path) -> Result<Song> {
        let path = root.join("out").join("timing.json");
        let text = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "{} not found: render the song with deepcuts first",
                path.display()
            )
        })?;
        let timing: Timing = serde_json::from_str(&text)
            .with_context(|| format!("couldn't read {}", path.display()))?;
        if timing.lines.iter().all(|l| l.display.is_none()) {
            bail!(
                "{} is from an older deepcuts: re-render it (deepcuts render \"{}\" --no-video)",
                path.display(),
                root.file_name().unwrap_or_default().to_string_lossy()
            );
        }
        Ok(Song {
            root: root.to_path_buf(),
            timing,
        })
    }

    fn display(&self) -> Display {
        self.timing.display.clone().unwrap_or_default()
    }

    pub fn karaoke(&self) -> PathBuf {
        self.root.join(
            self.display()
                .karaoke
                .unwrap_or_else(|| "stems/karaoke.wav".into()),
        )
    }

    pub fn vocals(&self) -> Option<PathBuf> {
        let p = self.root.join(
            self.display()
                .vocals
                .unwrap_or_else(|| "stems/vocals.wav".into()),
        );
        p.exists().then_some(p)
    }

    pub fn cover(&self) -> Option<PathBuf> {
        let name = self.display().cover?;
        let p = self.root.join(name);
        // the terminal draws stills only; a video cover just shows the title card
        let still = matches!(
            p.extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("png" | "jpg" | "jpeg")
        );
        (still && p.exists()).then_some(p)
    }

    /// Seconds to add to the clock before comparing with word times (the video's wipe lead).
    pub fn lead(&self) -> f64 {
        self.display().highlight_lead_ms as f64 / 1000.0
    }

    /// When the cover intro ends: deepcuts' own value, else just before the first lyrics.
    pub fn intro_end(&self) -> f64 {
        if let Some(t) = self.display().intro_end {
            return t;
        }
        let first = self
            .timing
            .lines
            .iter()
            .filter_map(|l| l.display.as_ref())
            .next();
        match first {
            Some(s) => s.countdown.first().map(|c| c.0).unwrap_or(s.appear) - 0.2,
            None => 0.0,
        }
    }
}

/// The song library: `library` from ~/.config/deepcuts/config.toml, else ~/deepcuts.
pub fn library_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return expand(&p.to_string_lossy());
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let cfg = PathBuf::from(&home).join(".config/deepcuts/config.toml");
    if let Ok(text) = std::fs::read_to_string(cfg) {
        if let Ok(value) = text.parse::<toml::Table>() {
            if let Some(lib) = value.get("library").and_then(|v| v.as_str()) {
                return expand(lib);
            }
        }
    }
    PathBuf::from(home).join("deepcuts")
}

fn expand(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(p),
    }
}

/// Song folders in the library that have been rendered (have out/timing.json), sorted.
pub fn rendered_songs(library: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(library)
        .map(|it| {
            it.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.join("out").join("timing.json").exists())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// A folder path, or any unique case-insensitive part of a song folder's name.
pub fn resolve(query: &str, library: &Path) -> Result<PathBuf> {
    let direct = expand(query);
    if direct.join("out").join("timing.json").exists() {
        return Ok(direct);
    }
    let q = query.to_lowercase();
    let hits: Vec<PathBuf> = rendered_songs(library)
        .into_iter()
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().to_lowercase().contains(&q))
        })
        .collect();
    match hits.len() {
        1 => Ok(hits.into_iter().next().unwrap()),
        0 => bail!(
            "no rendered song matching {query:?} in {}",
            library.display()
        ),
        _ => bail!(
            "{query:?} matches {} songs: {}",
            hits.len(),
            hits.iter()
                .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
