# Architecture

## Principles

- **The song folder is the database.** Everything a stage produces is a plain file in
  `~/deepcuts/<Artist - Title>/`, so you can inspect, edit or delete any of it by hand.
- **Human edits are first-class inputs.** `lyrics.lrc|txt`, `meta.json` and `overrides.json` are
  meant to be edited. Their contents feed the stage fingerprints, so an edit re-runs exactly
  the stages downstream of it.
- **Heavy dependencies are optional and lazy.** Torch, audio-separator, pyannote and whisperx are
  imported inside the stage that needs them. The core (parsing, layout, ASS, editor) installs in
  seconds and is fully unit-tested.
- **Machine output and human corrections never mix.** `alignment.json` is pure aligner output plus
  deterministic post-processing. `overrides.json` is layered on top at render time.

## Stage contracts

| stage | inputs | outputs | cache key |
|---|---|---|---|
| ingest | file / URL | `source.*`, `audio.wav` (44.1k stereo s16), `meta.json` | exists unless `-f ingest` |
| lyrics | meta | `lyrics.lrc` or `lyrics.txt`, `lyrics.source.json` | exists unless `-f lyrics` |
| separate | `audio.wav` | `stems/{vocals,instrumental,lead,backing,karaoke}.wav`, `stems.json` | audio + `[separation]` |
| align | lyrics file, vocal stem | `alignment.json` | lyrics content + stem + `[align]` + language |
| diarize | vocal stem | `diarization.json` (turns) | stem + `[diarize]` |
| render | alignment, overrides, turns, karaoke stem | `out/lyrics.{ass,lrc}`, `out/timing.json`, `out/*.mp4` | all inputs + `[render]` |

## Alignment in detail

```
vocal stem ──► MMS emissions (once, 20 ms frames)
                     │
     ┌───────────────┴────────────────┐
 global pass                     anchored pass (synced lyrics only)
 all words, <star> between       per line: window = [anchor+offset-0.6s, next anchor+offset+1.0s]
 every word                      <star> at the edges absorbs neighbouring lines' audio
     │                                │
     └──► offset = median(aligned line start - LRC anchor) over confident lines
                                      │
                   per line: keep anchored unless global is clearly more confident
                                      │
            post-process: interpolate untimed words (digits, symbols, failures)
                          → clamp runaway words → monotonic → close micro-gaps
```

Confidence per word is `exp(mean log-prob over its frames)`. It's a relative signal: use it to
rank lines worth checking, not as a calibrated probability.

## Layout

`stages/render/layout.py::schedule` decides row, appear and disappear time for each line. The
same function is ported to `web/player.html`, so the editor preview matches the video frame for
frame. Change one, change both (there's a test for the Python side's invariants).

## Adding things

- **A new aligner:** implement `align(audio, lines, language, strategy) -> (lines, offset)` and
  register it in `stages/align/__init__.py::get_aligner`.
- **A new separation backend:** implement `split(model, src, out_dir, names) -> [paths]`. Output
  names just need to contain the stem label (e.g. `(Vocals)`); `classify_pair` sorts them out.
- **A new output format:** add it to `stages/render/__init__.py::run_render` and `output_paths`.
