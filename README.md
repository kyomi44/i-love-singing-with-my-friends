# deepcuts 🎤

**Karaoke for deep cuts.** Point it at any track you own and get a karaoke video with the vocals removed and the words highlighted *on the beat*, without waiting for Sing King to cover it.

```
deepcuts make "~/Music/Broadcast - Pendulum.flac"
```

```
✓ ingest    Broadcast — Pendulum (232s)
✓ lyrics    41 lines from lrclib:123456 (synced)
✓ separate  vocals_mel_band_roformer.ckpt, mel_band_roformer_karaoke_becruily.ckpt
✓ align     ctc/auto, LRC was +1.32s off your audio (corrected)
✓ render    Broadcast - Pendulum (karaoke).mp4
```

## Why the timing is better than YouTube karaoke

Most bad karaoke timing comes from someone tapping along by hand, or from line-level LRC files timed against a different master than the one playing. deepcuts does it differently:

1. **Separate first.** A RoFormer model isolates the vocals, so the aligner hears the singer instead of the band.
2. **Align, don't transcribe.** The lyrics are already known, so the only question is *when* each word is sung. Forced alignment (MMS CTC, about 1,100 languages) answers that. It never invents lyrics the way speech-to-text does on singing.
3. **Treat synced lyrics as hints, not truth.** The `auto` strategy aligns the whole song first, measures how far off the LRC file is from *your* audio, corrects for it, and then refines each line inside a tight window. The result from either pass is kept per line, whichever the aligner is more confident in.
4. **Tidy up for the eye.** Monotonic timings, no flicker between words, no first word that soaks up 8 seconds of intro.
5. **Flag what it isn't sure about.** Low-confidence lines are listed after every run. You fix them by ear in a browser editor and re-render in seconds, with no ML re-run.

> "Diarization" (who is singing when) is a different problem from timing. deepcuts uses it only for **duets**, to colour lines by singer. Timing comes from forced alignment.

## Pipeline

```
 source.flac ─► ingest ─► lyrics ─► separate ─────────────► align ─► (diarize) ─► render
   or URL          │         │         │                      │          │           │
                audio.wav   LRCLIB   vocals + instrumental   word      singer      .mp4  .ass
                meta.json   + more   lead + backing          timings   parts       .lrc  .json
                            or ASR   karaoke = inst + backing
```

Every stage caches its output in the song folder and re-runs only when its inputs or settings change. Edit `lyrics.txt` and `deepcuts align` reruns. Tweak colours and only `render` reruns.

## Quickstart

```bash
# 1. system dependency: ffmpeg built with libass (the default builds from brew and apt include it)
brew install ffmpeg            # or: sudo apt install ffmpeg

# 2. install (Python 3.11–3.13)
git clone <this repo> deepcuts && cd deepcuts
uv sync --extra separate --extra align
#   NVIDIA GPU: also  uv pip install "audio-separator[gpu]"   (separation is ~20x faster)

# 3. check everything
uv run deepcuts doctor

# 4. go
uv run deepcuts make "~/Music/Stereolab - French Disko.flac"
uv run deepcuts make "https://artist.bandcamp.com/track/..." -a "Artist" -t "Title"   # needs --extra download
```

Songs live in `~/deepcuts/<Artist - Title>/` (configurable). Play the `.mp4` on the TV however you already play YouTube: Chromecast, Plex, a USB stick, or a browser tab.

## No GPU? Use Modal

Separation and alignment can run on a serverless GPU at [Modal](https://modal.com). Everything else (lyrics, rendering, the editor) stays on your machine, and you don't install torch locally at all:

```bash
uv sync --extra modal
uv run modal token new                 # once: log in in the browser
uv run deepcuts make song.flac --compute modal
```

Or set `compute = "modal"` in your config to make it the default (`--compute local` overrides it per command). The first run builds the container image (a few minutes) and downloads the models into a Modal Volume. After that a song costs one container start plus about a minute of GPU time. GPUs are billed per second (L4 about $0.80/h; if no L4 is free it falls back to an A10G or T4, see `[modal] gpu`), so a song costs a few cents. Nothing runs, and nothing is billed, between songs.

The container runs the same stage code from this repo's `uv.lock`, so local and remote results match.

## Fixing timing by ear

```bash
deepcuts report "french disko"     # which lines the aligner was unsure about
deepcuts serve  "french disko"     # opens the timing editor in your browser
deepcuts nudge  "french disko" --ms -120            # or shift the whole song
deepcuts nudge  "french disko" --ms 80 --line 12    # ...or a single line
```

The editor plays the karaoke track with the same two-row layout as the video. Low-confidence lines get a red dot.

| key | action |
|---|---|
| `space` / `←` `→` | play-pause / seek 5 s |
| `↑` `↓` / `⏎` | pick a line / replay it from just before |
| `[` `]` | whole-song offset ±10 ms (`⇧` = ±50) |
| `,` `.` | selected line ±20 ms (`⇧` = ±100) |
| `T` | **tap-sync**: press as the line starts, and it snaps there |
| `V` | toggle guide vocals (hear the singer quietly) |
| `S` / `R` | save to `overrides.json` / save and re-render the video |

Corrections live in `overrides.json`, separate from the machine alignment, so re-aligning never throws away your fixes. If you prefer the wipe to start a hair early, set `render.highlight_lead_ms = 60`.

## Lyrics

Lookup order: `--lyrics FILE` → an existing `lyrics.lrc`/`lyrics.txt` in the song folder → [LRCLIB](https://lrclib.net) (free, synced) → [syncedlyrics](https://github.com/moehmeni/syncedlyrics) providers (Musixmatch, NetEase…, `--extra lyrics`) → optional `--asr` transcription of the vocal stem ([WhisperX](#whisperx)), written out for you to proof-read.

Lyrics are a plain file you can edit:

```text
[Verse 1]                 ← section headers are dropped (and start a new screen)
@A I walked down to the harbour
@B and I followed you there
@AB we both sang this one  ← duet parts are coloured per singer
(ooh, ooh)                ← parentheticals render in italics

                          ← blank line = stanza break
```

## Separation

| setting | default | notes |
|---|---|---|
| `separation.vocal_model` | `vocals_mel_band_roformer.ckpt` | best vocal SDR in audio-separator's bundled benchmark; `audio-separator --list_models` for alternatives |
| `separation.karaoke_models` | becruily → gabox → aufr33/viperx Mel-RoFormer karaoke | splits lead from backing vocals; tried in order |
| `separation.backing` | `keep` | harmonies stay in the track, like real karaoke. `remove` gives a fully dry instrumental |
| `separation.backend` | `local` | `mvsep` sends jobs to [MVSEP](https://mvsep.com), which hosts the current leaderboard models (needs `MVSEP_API_TOKEN`; `deepcuts mvsep-algorithms roformer` lists model ids) |

## Duets

Tag lines `@A` / `@B` / `@AB` in the lyrics (most reliable), or try `deepcuts make song.flac --diarize` (`--extra diarize` plus a Hugging Face token with the [community-1](https://hf.co/pyannote/speaker-diarization-community-1) terms accepted). Diarization models are trained on speech. They do fine on clearly different voices and badly on stacked harmonies, and manual tags always win.

## WhisperX

WhisperX powers the `--asr` fallback and the alternative `--aligner whisperx`. It isn't an extra because it pins an older PyTorch, which would hold the separation models back for everyone. If you want it, install it into the same env and accept the downgrade, or keep a second env. Stems are cached on disk, so a second env never re-separates:

```bash
uv pip install whisperx                    # same env (downgrades torch)
uv run --with whisperx deepcuts make song.flac --asr    # or a throwaway overlay env
```

## Outputs (`<song>/out/`)

- `<song> (karaoke).mp4`: 1080p H.264 over a colour, image or looping video (see below), optional `--guide-vocals 0.15`
- **Intro and background media**: each can be an image *or* a video, in any combination.
  - The intro (`--cover FILE`) opens the video with the title underneath, then fades into the lyrics just before the first line's countdown. Without one, the art embedded in the audio file is used. Turn it off with `render.cover_intro = false`.
  - The background (`--background FILE`) sits behind the lyrics for the whole song. Videos loop, their own audio is ignored, and everything is dimmed by `render.background_dim`. `--background cover` uses the cover art, blurred, and `--background '#hex'` a plain colour.
  - Files are saved in the song folder as `cover.*` / `background.*` (you can also drop them there yourself), so re-renders, including from the editor, keep them.
- `lyrics.ass`: the karaoke subtitles (`\kf` wipes), playable over the stems in mpv/VLC
- `lyrics.lrc`: enhanced LRC with word timestamps
- `timing.json`: everything, for building your own player

## Configuration

Copy `deepcuts.example.toml` to `./deepcuts.toml` or `~/.config/deepcuts/config.toml`. Most options also have CLI flags (`deepcuts make --help`).

## Project layout

```
src/deepcuts/
  cli.py              typer CLI (make, lyrics, separate, align, diarize, render, serve, nudge, report, ls, doctor)
  pipeline.py         stage orchestration + caching
  config.py models.py workspace.py audio.py text.py
  stages/
    ingest.py         file/URL -> audio.wav + meta.json
    lyrics/           LRCLIB client, LRC parser/writer, ASR fallback
    separate/         audio-separator (local) and MVSEP (remote) backends
    align/            CTC (default) + WhisperX aligners, anchor/offset logic, post-processing
    diarize.py        pyannote turns -> singer per line
    render/           layout, ASS karaoke, ffmpeg video
  serve.py web/       browser timing editor
tests/                pure-logic tests + an ffmpeg render smoke test
docs/ARCHITECTURE.md  stage contracts and design notes
CLAUDE.md             context for Claude Code sessions (decisions, gotchas, next steps)
```

```bash
uv sync --extra dev && uv run pytest && uv run ruff check
```

## Prior art and credits

- **[karaoke-gen](https://github.com/nomadkaraoke/karaoke-gen)** by Nomad Karaoke is a mature end-to-end pipeline with a similar goal, and well worth trying. deepcuts takes a different, lyrics-first approach (forced alignment of known lyrics, LRC offset correction, a by-ear editor) and aims to stay small and hackable.
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator): separation models and runtime
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner): MMS forced alignment
- [LRCLIB](https://lrclib.net): community synced lyrics
- [pyannote.audio](https://github.com/pyannote/pyannote-audio), [WhisperX](https://github.com/m-bain/whisperX), [MVSEP](https://mvsep.com)

## Licences and fair play

The code is MIT. Model weights carry their own licences. Notably, the default MMS aligner checkpoint is **CC BY-NC 4.0**: fine for singing with friends, not for a commercial product. Use audio you own, and keep the karaoke versions for private use.

## Roadmap

- [ ] Party mode: queue + full-screen player for the TV, phones as remotes (the editor's renderer is most of it)
- [ ] Key/pitch shift per singer (rubberband) without re-separating
- [ ] Syllable-level wipes for melismas (split words across sung notes using pitch onsets)
- [ ] ASR cross-check that flags lines where the fetched lyrics don't match what's sung
- [ ] Batch mode over a folder or playlist; CDG+MP3 export for hardware machines
- [ ] Japanese/Chinese lyrics: the aligner switches to per-character output there, and the mapping back to lyric "words" isn't written yet (lines fall back to interpolated timing)

## Status

Early but working. The full pipeline (local separation backend and CTC alignment, run on Modal) has been used on a real song end to end, and the first real run shaped the alignment logic: vocal-activity fitting and stricter low-confidence flagging. A 3-minute track takes about 2 minutes from file to video. Diarization and the MVSEP backend haven't been run on real audio yet. `deepcuts -v make …` shows debug logs, including timings from the Modal container.

Known limit: lines where a second voice sings *over* the lead, written as parentheticals (`Happily (there's no use fearing…)`), are aligned as if sung one after the other, so their wipe is wrong. They're flagged in `deepcuts report`, so you can fix them in the editor. Doing it properly means aligning the parenthetical against the backing stem (the karaoke model already puts that voice there) and rendering it as a second row.
