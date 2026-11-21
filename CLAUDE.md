# CLAUDE.md

deepcuts turns any track into a karaoke video: RoFormer stem separation, then forced alignment of
*known* lyrics against the vocal stem, then an ASS karaoke render via ffmpeg. The owner and their
friends are music nerds who sing obscure songs that no karaoke channel covers; timing accuracy is
the whole point. The README has user docs, docs/ARCHITECTURE.md has design and stage contracts.

## Commands

```bash
uv sync --extra dev                       # core + tests (no ML); enough for most work
uv sync --extra separate --extra align    # + real separation and alignment (needs a GPU to be quick)
uv run pytest                             # 44 tests; ffmpeg+libass and node are used when present
uv run ruff check && uv run ruff format --check
uv run deepcuts doctor                    # environment check
```

## Where things are

- `pipeline.py` orchestrates stages with fingerprint caching (`workspace.py::fingerprint`). Human-edited files
  (lyrics, meta.json, overrides.json) are part of the fingerprints, so editing them re-runs downstream stages.
- `stages/align/ctc.py` is the default aligner (ctc-forced-aligner + MMS). Emissions are computed once; the
  `auto` strategy does a global pass, estimates the LRC→audio offset (`anchors.py`), then refines per line.
- `stages/align/postprocess.py` is pure functions; `apply_overrides` is applied at render time, never baked
  into alignment.json.
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

## Status and next steps

- Done and tested: lyrics (LRCLIB, LRC/enhanced LRC, plain text with `@A/@B` duet tags), layout, ASS, video,
  caching, CLI, browser timing editor (verified in headless Chrome), and a full pipeline run with fake ML
  backends (`tests/test_pipeline_e2e.py`).
- **Not yet run on real audio:** `stages/separate/local.py`, `stages/align/ctc.py`, `stages/diarize.py`,
  `stages/separate/mvsep.py` (MVSEP's `files` response shape is undocumented; `_download` is defensive).
  First job: run `deepcuts -v make <song>` on a real track on a GPU box and fix whatever breaks.
- Then: tune `align.low_confidence` against real songs; CJK character mapping; party/TV mode; key shift.

## Conventions

- Heavy deps (torch, audio_separator, ctc_forced_aligner, pyannote, whisperx, yt_dlp) are imported lazily
  inside the stage that needs them and raise `MissingExtra` with an install hint.
- User-facing failures raise `DeepcutsError` subclasses; `cli.run()` prints them without a traceback.
- Keep stage logic pure where possible and test it without models (see the fakes in test_pipeline_e2e.py).
