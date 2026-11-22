# CLAUDE.md

deepcuts turns any track into a karaoke video: RoFormer stem separation, then forced alignment of
*known* lyrics against the vocal stem, then an ASS karaoke render via ffmpeg. The owner and their
friends are music nerds who sing obscure songs that no karaoke channel covers; timing accuracy is
the whole point. The README has user docs, docs/ARCHITECTURE.md has design and stage contracts.

## Commands

```bash
uv sync --extra dev                       # core + tests (no ML); enough for most work
uv sync --extra separate --extra align    # + real separation and alignment (needs a GPU to be quick)
uv sync --extra dev --extra modal         # or: run separation + alignment on Modal (no torch locally)
uv run pytest                             # ffmpeg+libass and node are used when present
uv run ruff check && uv run ruff format --check
uv run deepcuts doctor                    # environment check
cd player && cargo test && cargo clippy -- -D warnings   # the Rust terminal player
```

## Where things are

- `pipeline.py` orchestrates stages with fingerprint caching (`workspace.py::fingerprint`). Human-edited files
  (lyrics, meta.json, overrides.json) are part of the fingerprints, so editing them re-runs downstream stages.
- `stages/align/ctc.py` is the default aligner (ctc-forced-aligner + MMS). Emissions are computed once; the
  `auto` strategy does a global pass, estimates the LRC→audio offset (`anchors.py`), then refines per line.
- `stages/align/postprocess.py` is pure functions; `apply_overrides` is applied at render time, never baked
  into alignment.json.
- `stages/align/activity.py`: vocal activity from the stem. ctc.py uses it to mask emissions in silence, clip
  each anchored window at the next line's onset, and fit word spans (trim into singing, hold through notes).
- `stages/media.py`: per-song `cover.*` (intro) and `background.*` (behind the lyrics), each an image or a video,
  set by `--cover` / `--background FILE` or dropped in the folder. The cover falls back to art embedded in the source
  audio. `render/ass.py::intro_end` decides when the intro ends, before the first countdown, and both the ASS title
  card and the ffmpeg fade in `render/video.py` use it. Videos loop with `-stream_loop -1`, images with `-loop 1`.
- `remote/`: `compute = "modal"`. `modal_app.py` is the GPU container (image from this repo's uv.lock + the local
  package), `client.py` starts one ephemeral app per process and swaps in for the separation stage and the aligner.
- `player/` (Rust, `deepcuts-sing`): terminal karaoke player. It reads `out/timing.json`, whose per-line `display`
  (row/appear/disappear/countdown from `render/ass.py::display_schedule`) means layout.py is *not* re-implemented
  in Rust. `audio.rs` mixes the stems in its own rodio Source, whose frame counter is the lyrics clock.
  `lyrics.rs` is pure and unit-tested. `cd player && cargo clippy -- -D warnings && cargo test`.
- `stages/render/layout.py::schedule` is mirrored in JS in `web/player.html` between `// <layout>` markers.
  `tests/test_editor_parity.py` runs both and compares. Change both together.

## Gotchas already learned (don't re-learn them)

- ctc-forced-aligner 0.3 reports a word's `score` as the **mean** per-frame log-prob, so confidence is just
  `exp(score)`. An earlier version of this code divided by frame count again, which pinned everything near 100%.
- `preprocess_text(..., star_frequency="segment"|"edges")`; for `jpn`/`chi` it switches to per-character
  output, which `match_words` can't map back yet (open roadmap item).
- audio-separator 0.48: `vocals_mel_band_roformer.ckpt` (default) outputs stems "Vocals" + "Other", not
  "Instrumental". `classify_pair` reads the `(Label)` in output filenames; custom names map Other too.
  Its MDXC defaults are `batch_size=None, overlap=None` (= the model's own value).
- WhisperX is deliberately **not** an extra: it pins torch 2.8, which held audio-separator at 0.45 in the
  universal lock. Without it the lock resolves to torch 2.15 / audio-separator 0.48. uv `conflicts` didn't help
  (uv doesn't fork when a shared solution exists).
- ffmpeg's `ass=` filter chokes on `:` and quotes in paths, so `render_video` runs ffmpeg with `cwd` = out dir.
- Typer list options default to `[]`, so B006 is ignored for cli.py only. E501 is ignored (formatter owns length).
- audio-separator 0.48 imports `onnxruntime` (only in its `[cpu]`/`[gpu]` extras) and `audioread` (undeclared),
  so the `separate` extra lists both. torch 2.15 has no macOS 13 wheel: local ML on older Macs is out, use Modal.
- The becruily karaoke model's own overlap is 8: 153 s for a 3-min song on an L4. `separation.overlap = 2` → 27 s,
  stems within ~30 dB SDR. The vocal model was already at that speed.
- ctc-forced-aligner `get_spans` pads each word halfway into neighbouring blanks. In "edges" mode (stars only at the
  line edges) a soft held word got padded across an 8 s instrumental gap and scored ~0.97, which beat the honest
  global result in `pick_best`. Switching the windows to "segment" stars made it worse (words jumped onto the next
  line's onset). The fix that worked is activity.py: mask + window clip + fit. Don't trust confidence on long spans.
- Modal: the GPU type is checked when the app starts, so it comes from env vars set before importing modal_app
  (`with_options()` is too late). Any GPU needs a payment method on the account. Exceptions cross the wire by
  pickle, so DeepcutsError subclasses with extra __init__ args need `__reduce__` (see MissingExtra).
- `-v` only sets the `deepcuts` logger to DEBUG: root DEBUG made Modal's grpc/hpack layer log auth tokens.
- Dimming is `colorchannelmixer` (an RGB multiply). `drawbox=color=black@a` looked fine on images but washed
  YUV video frames out to grey (blue 0x224488 came out as (30, 37, 39) instead of about (18, 37, 75)).
- Stage fingerprints cover inputs and config, not code. After changing stage code use `-f align` etc.

## Status and next steps

- Done and tested: lyrics (LRCLIB, LRC/enhanced LRC, plain text with `@A/@B` duet tags), layout, ASS, video,
  caching, CLI, browser timing editor (verified in headless Chrome), and a full pipeline run with fake ML
  backends (`tests/test_pipeline_e2e.py`).
- Run on real audio (Nov 2026, Black Country, New Road "Salem Sisters", via Modal): `separate/local.py`,
  `align/ctc.py`. A 3-min song takes ~2 min (separation ~80 s round trip, align ~20 s, render ~11 s).
  Flagging = line mean < 0.35 or any word < 0.2: on that song it caught exactly the lines with overlapping voices.
- **Not yet run on real audio:** `stages/diarize.py`, `stages/separate/mvsep.py` (MVSEP's `files` response shape
  is undocumented; `_download` is defensive).
- Next: overlapping second voices written as parentheticals. The karaoke model puts that voice on the backing stem,
  so align the parenthetical words against `backing.wav` in the same window and render them as a second row
  (layout.py + the JS mirror + ASS). Then more songs to calibrate the thresholds; CJK; party/TV mode; key shift.

## Conventions

- Heavy deps (torch, audio_separator, ctc_forced_aligner, pyannote, whisperx, yt_dlp) are imported lazily
  inside the stage that needs them and raise `MissingExtra` with an install hint.
- User-facing failures raise `DeepcutsError` subclasses; `cli.run()` prints them without a traceback.
- Keep stage logic pure where possible and test it without models (see the fakes in test_pipeline_e2e.py).
