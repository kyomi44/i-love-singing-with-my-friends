"""Alignment stage: lyrics + vocal stem -> word-level timings."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from deepcuts.config import AlignConfig, Config
from deepcuts.models import Alignment, Line, Lyrics, SongMeta, StemSet
from deepcuts.stages.align.postprocess import finalize


class Aligner(Protocol):
    name: str

    def align(
        self, audio: Path, lines: list[Line], language: str, strategy: str
    ) -> tuple[list[Line], float | None]: ...


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def get_aligner(cfg: AlignConfig, device: str) -> Aligner:
    if cfg.aligner == "whisperx":
        from deepcuts.stages.align.whisperx_aligner import WhisperXAligner

        return WhisperXAligner(cfg, device)
    from deepcuts.stages.align.ctc import CtcAligner

    return CtcAligner(cfg, device)


def alignment_stem(stems: StemSet, cfg: AlignConfig) -> Path:
    return stems.lead if cfg.stem == "lead" and stems.lead else stems.vocals


def run_alignment(meta: SongMeta, lyrics: Lyrics, stems: StemSet, cfg: Config) -> Alignment:
    stem = alignment_stem(stems, cfg.align)
    if cfg.compute == "modal":
        from deepcuts.remote.client import RemoteAligner

        aligner: Aligner = RemoteAligner(cfg)
    else:
        aligner = get_aligner(cfg.align, resolve_device(cfg.align.device))
    strategy = cfg.align.strategy
    if strategy != "global" and not lyrics.synced:
        strategy = "global"
    lines, offset = aligner.align(stem, lyrics.lines, meta.language, strategy)
    return Alignment(
        lines=finalize(lines, cfg.align),
        aligner=aligner.name,
        stem=stem.name,
        strategy=strategy,
        lrc_offset=None if offset is None else round(offset, 3),
    )
