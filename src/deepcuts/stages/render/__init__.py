"""Render stage: alignment (+ your overrides, + singer parts) -> .ass / .lrc / .json / .mp4."""

from __future__ import annotations

import json
from pathlib import Path

from deepcuts.config import Config
from deepcuts.models import Line, StemSet
from deepcuts.stages.align.postprocess import apply_overrides
from deepcuts.stages.diarize import assign_singers, load_turns
from deepcuts.stages.lyrics.lrc import dump_lrc
from deepcuts.stages.media import background_file, cover_file
from deepcuts.stages.render.ass import build_ass, display_schedule, intro_end
from deepcuts.stages.render.video import render_video
from deepcuts.workspace import SongDir


def singer_lines(song: SongDir, cfg: Config) -> list[Line]:
    """Alignment + diarized singer parts, *without* overrides (the editor applies those live)."""
    lines = song.load_alignment().lines
    if cfg.diarize.enabled and song.diarization_path.exists():
        lines = assign_singers(lines, load_turns(song.diarization_path), cfg.diarize.both_ratio)
    return lines


def prepared_lines(song: SongDir, cfg: Config) -> list[Line]:
    return apply_overrides(singer_lines(song, cfg), song.load_overrides())


def output_paths(song: SongDir) -> dict[str, Path]:
    out = song.out_dir
    return {
        "ass": out / "lyrics.ass",
        "lrc": out / "lyrics.lrc",
        "json": out / "timing.json",
        "video": out / f"{song.name} (karaoke).mp4",
    }


def run_render(song: SongDir, stems: StemSet, cfg: Config) -> dict[str, Path]:
    meta = song.load_meta()
    lines = prepared_lines(song, cfg)
    song.out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(song)
    cover = cover_file(song)
    intro = intro_end(lines, cfg.render) if cover and cfg.render.cover_intro else None
    # the ASS title sits under the cover only in the video; played over stems it's mid-screen
    ass_intro = intro if cfg.render.video else None
    paths["ass"].write_text(build_ass(lines, meta, cfg.render, ass_intro), encoding="utf-8")
    paths["lrc"].write_text(dump_lrc(lines, meta), encoding="utf-8")
    rc = cfg.render
    display = display_schedule(lines, rc)
    timing = {
        "meta": meta.model_dump(),
        # How the video lays lines out, so other players (e.g. player/, the terminal one) can
        # match it without re-deriving layout.py. Word times are raw: apply highlight_lead_ms.
        "display": {
            "highlight_lead_ms": rc.highlight_lead_ms,
            "intro_end": None if intro is None else round(intro, 3),
            "cover": cover.name if cover else None,
            "karaoke": stems.karaoke.relative_to(song.root).as_posix(),
            "vocals": stems.vocals.relative_to(song.root).as_posix(),
        },
        "lines": [
            {**line.model_dump(exclude_none=True), **({"display": d} if d else {})}
            for line, d in zip(lines, display, strict=True)
        ],
    }
    paths["json"].write_text(json.dumps(timing, indent=1, ensure_ascii=False), encoding="utf-8")
    result = {k: v for k, v in paths.items() if k != "video"}
    if cfg.render.video:
        render_video(
            ass_path=paths["ass"],
            track=stems.karaoke,
            out_path=paths["video"],
            duration=meta.duration,
            cfg=cfg.render,
            vocals=stems.vocals,
            cover=cover,
            intro=intro,
            background=background_file(song),
        )
        result["video"] = paths["video"]
    return result
