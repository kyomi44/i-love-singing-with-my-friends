//! Playback with a sample-accurate clock.
//!
//! The karaoke and vocal stems are decoded into memory and mixed by our own rodio `Source`. Its
//! frame counter *is* the song clock the lyrics follow, so they can't drift from the audio:
//! pausing outputs silence without advancing it, seeking just moves it, and the guide vocal is
//! a gain applied in the same loop. (Two independent players would drift and seek apart.)

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

pub struct Pcm {
    pub rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>, // interleaved
}

impl Pcm {
    pub fn frames(&self) -> u64 {
        (self.samples.len() / self.channels as usize) as u64
    }
}

pub fn read_wav(path: &Path) -> Result<Pcm> {
    let mut r =
        hound::WavReader::open(path).with_context(|| format!("can't open {}", path.display()))?;
    let spec = r.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    Ok(Pcm {
        rate: spec.sample_rate,
        channels: spec.channels,
        samples,
    })
}

/// Make `v` line up with `k` frame for frame (same channel count), or None if it can't.
fn match_layout(v: Pcm, k: &Pcm) -> Option<Vec<f32>> {
    if v.rate != k.rate {
        return None;
    }
    let (vc, kc) = (v.channels as usize, k.channels as usize);
    let frames = k.frames() as usize;
    let mut out = vec![0.0f32; frames * kc];
    for f in 0..frames.min(v.frames() as usize) {
        for c in 0..kc {
            out[f * kc + c] = v.samples[f * vc + c.min(vc - 1)];
        }
    }
    Some(out)
}

struct Shared {
    frame: AtomicU64,
    paused: AtomicBool,
    guide: AtomicU32,  // f32 bits
    volume: AtomicU32, // f32 bits
}

fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

struct Mix {
    karaoke: Arc<Vec<f32>>,
    vocals: Option<Arc<Vec<f32>>>,
    channels: u16,
    rate: u32,
    shared: Arc<Shared>,
    ch: u16,
}

impl Iterator for Mix {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        let s = &self.shared;
        if s.paused.load(Ordering::Relaxed) {
            return Some(0.0);
        }
        let frame = s.frame.load(Ordering::Relaxed) as usize;
        let i = frame * self.channels as usize + self.ch as usize;
        let mut x = self.karaoke.get(i).copied().unwrap_or(0.0);
        if let Some(v) = &self.vocals {
            x += load_f32(&s.guide) * v.get(i).copied().unwrap_or(0.0);
        }
        self.ch += 1;
        if self.ch == self.channels {
            self.ch = 0;
            // only advance if nobody seeked meanwhile (compare-and-swap)
            let _ = s.frame.compare_exchange(
                frame as u64,
                frame as u64 + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
        Some(x * load_f32(&s.volume))
    }
}

impl rodio::Source for Mix {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        self.channels
    }
    fn sample_rate(&self) -> u32 {
        self.rate
    }
    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

enum Output {
    Speakers {
        _stream: rodio::OutputStream,
        _sink: rodio::Sink,
    },
    /// --silent: no audio device; the clock follows the wall clock instead.
    Silent { since: Option<Instant>, base: u64 },
}

pub struct Player {
    shared: Arc<Shared>,
    rate: u32,
    frames: u64,
    has_vocals: bool,
    out: Output,
}

pub const GUIDE_GAIN: f32 = 0.3;

impl Player {
    pub fn open(karaoke: &Path, vocals: Option<&Path>, silent: bool) -> Result<Player> {
        let k = read_wav(karaoke)?;
        if k.channels == 0 || k.samples.is_empty() {
            bail!("{} has no audio", karaoke.display());
        }
        let v = match vocals {
            Some(p) => read_wav(p).ok().and_then(|v| match_layout(v, &k)),
            None => None,
        };
        let shared = Arc::new(Shared {
            frame: AtomicU64::new(0),
            paused: AtomicBool::new(true),
            guide: AtomicU32::new(0f32.to_bits()),
            volume: AtomicU32::new(1f32.to_bits()),
        });
        let (rate, frames, has_vocals) = (k.rate, k.frames(), v.is_some());
        let out = if silent {
            Output::Silent {
                since: None,
                base: 0,
            }
        } else {
            let (stream, handle) = rodio::OutputStream::try_default()
                .context("no audio output device (try --silent)")?;
            let sink = rodio::Sink::try_new(&handle)?;
            sink.append(Mix {
                karaoke: Arc::new(k.samples),
                vocals: v.map(Arc::new),
                channels: k.channels,
                rate,
                shared: shared.clone(),
                ch: 0,
            });
            Output::Speakers {
                _stream: stream,
                _sink: sink,
            }
        };
        Ok(Player {
            shared,
            rate,
            frames,
            has_vocals,
            out,
        })
    }

    pub fn duration(&self) -> f64 {
        self.frames as f64 / self.rate as f64
    }

    pub fn has_vocals(&self) -> bool {
        self.has_vocals
    }

    fn frame(&self) -> u64 {
        match &self.out {
            Output::Speakers { .. } => self.shared.frame.load(Ordering::Relaxed),
            Output::Silent { since, base } => {
                base + since.map_or(0, |t| (t.elapsed().as_secs_f64() * self.rate as f64) as u64)
            }
        }
    }

    /// Song time in seconds (clamped to the end).
    pub fn time(&self) -> f64 {
        self.frame().min(self.frames) as f64 / self.rate as f64
    }

    pub fn paused(&self) -> bool {
        self.shared.paused.load(Ordering::Relaxed)
    }

    pub fn set_paused(&mut self, paused: bool) {
        if let Output::Silent { since, base } = &mut self.out {
            if paused {
                if let Some(t) = since.take() {
                    *base += (t.elapsed().as_secs_f64() * self.rate as f64) as u64;
                }
            } else if since.is_none() {
                *since = Some(Instant::now());
            }
        }
        self.shared.paused.store(paused, Ordering::Relaxed);
    }

    pub fn seek(&mut self, t: f64) {
        let f = ((t.max(0.0) * self.rate as f64) as u64).min(self.frames);
        self.shared.frame.store(f, Ordering::Relaxed);
        if let Output::Silent { since, base } = &mut self.out {
            *base = f;
            if since.is_some() {
                *since = Some(Instant::now());
            }
        }
    }

    pub fn guide(&self) -> bool {
        load_f32(&self.shared.guide) > 0.0
    }

    pub fn toggle_guide(&self) {
        let g = if self.guide() { 0.0 } else { GUIDE_GAIN };
        self.shared.guide.store(f32::to_bits(g), Ordering::Relaxed);
    }

    pub fn volume(&self) -> f32 {
        load_f32(&self.shared.volume)
    }

    pub fn nudge_volume(&self, delta: f32) {
        let v = (self.volume() + delta).clamp(0.0, 1.5);
        self.shared.volume.store(v.to_bits(), Ordering::Relaxed);
    }
}
