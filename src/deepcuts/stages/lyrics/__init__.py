"""Lyrics stage: find lyrics (synced if possible) and keep them as an editable file in the song dir.

The file (lyrics.lrc or lyrics.txt) is the source of truth: edit it and re-run `deepcuts align`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from deepcuts.config import Config
from deepcuts.errors import LyricsNotFound
from deepcuts.models import Lyrics, SongMeta
from deepcuts.stages.lyrics.lrc import is_lrc, parse_lrc
from deepcuts.stages.lyrics.lrclib import LrclibClient
from deepcuts.text import parse_plain

log = logging.getLogger(__name__)


def load_lyrics_file(path: Path, source: str | None = None) -> Lyrics:
    text = path.read_text(encoding="utf-8")
    lines = parse_lrc(text) if path.suffix.lower() == ".lrc" or is_lrc(text) else parse_plain(text)
    if not lines:
        raise LyricsNotFound(f"{path} contains no lyric lines")
    return Lyrics(lines=lines, source=source or f"file:{path.name}", path=path)


def fetch_lyrics(meta: SongMeta, cfg: Config) -> tuple[str, str, str]:
    """Return (content, kind, source) where kind is "lrc" or "txt"."""
    for src in cfg.lyrics.sources:
        try:
            if src == "lrclib":
                found = _from_lrclib(meta, cfg)
            elif src == "syncedlyrics":
                found = _from_syncedlyrics(meta, cfg)
            else:
                continue
        except LyricsNotFound:
            raise
        except Exception as exc:  # network hiccups shouldn't kill the run; try the next source
            log.warning("lyrics source %s failed: %s", src, exc)
            continue
        if found:
            return found
    raise LyricsNotFound(
        f"No lyrics found for {meta.artist} - {meta.title}. Pass --lyrics FILE (.lrc or .txt), "
        "or enable [lyrics] asr_fallback to transcribe the vocal stem."
    )


def _from_lrclib(meta: SongMeta, cfg: Config) -> tuple[str, str, str] | None:
    rec = LrclibClient().best_match(
        meta.artist, meta.title, meta.album, meta.duration, cfg.lyrics.duration_tolerance
    )
    if not rec:
        return None
    if rec.get("instrumental"):
        raise LyricsNotFound(f"LRCLIB marks {meta.artist} - {meta.title} as instrumental")
    source = f"lrclib:{rec.get('id')}"
    if rec.get("syncedLyrics"):
        return rec["syncedLyrics"], "lrc", source
    if rec.get("plainLyrics"):
        return rec["plainLyrics"], "txt", source
    return None


def _from_syncedlyrics(meta: SongMeta, cfg: Config) -> tuple[str, str, str] | None:
    try:
        import syncedlyrics
    except ImportError:
        log.debug("syncedlyrics not installed; skipping")
        return None
    text = syncedlyrics.search(
        f"{meta.artist} {meta.title}", providers=cfg.lyrics.syncedlyrics_providers
    )
    if not text:
        return None
    return text, ("lrc" if is_lrc(text) else "txt"), "syncedlyrics"
