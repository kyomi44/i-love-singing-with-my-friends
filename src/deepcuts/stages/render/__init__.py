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
from deepcuts.stages.render.ass import build_ass, intro_end
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
    intro = (
        intro_end(lines, cfg.render)
        if cover and cfg.render.cover_intro and cfg.render.video
        else None
    )
    paths["ass"].write_text(build_ass(lines, meta, cfg.render, intro), encoding="utf-8")
    paths["lrc"].write_text(dump_lrc(lines, meta), encoding="utf-8")
    paths["json"].write_text(
        json.dumps(
            {
                "meta": meta.model_dump(),
                "lines": [line.model_dump(exclude_none=True) for line in lines],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
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
