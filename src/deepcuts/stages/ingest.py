"""Ingest: local file or URL -> song dir with source copy, normalised audio.wav and meta.json."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from deepcuts import audio
from deepcuts.config import Config
from deepcuts.errors import DeepcutsError, MissingExtra
from deepcuts.models import SongMeta
from deepcuts.workspace import SongDir, slugify

URL_RE = re.compile(r"^https?://", re.IGNORECASE)
TRACKNO_RE = re.compile(r"^\s*\d{1,3}\s*[-._)]?\s+")


def parse_filename(stem: str) -> tuple[str | None, str]:
    """'03 - Artist - Title' / 'Artist - Title' / 'Title' -> (artist, title)."""
    stem = TRACKNO_RE.sub("", stem).strip()
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip() or None, title.strip()
    return None, stem


def ingest(
    source: str,
    cfg: Config,
    artist: str | None = None,
    title: str | None = None,
    album: str | None = None,
    force: bool = False,
) -> SongDir:
    if URL_RE.match(source):
        return _ingest_url(source, cfg, artist, title, album, force)

    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise DeepcutsError(f"file not found: {src}")
    t = audio.tags(src)
    guess_artist, guess_title = parse_filename(src.stem)
    artist = artist or t.get("artist") or t.get("album_artist") or guess_artist or "Unknown Artist"
    title = title or t.get("title") or guess_title
    album = album or t.get("album")

    song = SongDir(cfg.library_path / slugify(artist, title)).create()
    if song.meta_path.exists() and song.audio.exists() and not force:
        return song
    dst = song.root / f"source{src.suffix.lower()}"
    shutil.copy2(src, dst)
    return _finish(song, dst, cfg, artist, title, album, str(src))


def _ingest_url(
    url: str, cfg: Config, artist: str | None, title: str | None, album: str | None, force: bool
) -> SongDir:
    try:
        import yt_dlp
    except ImportError as exc:
        raise MissingExtra("download", "yt-dlp") from exc

    with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    raw_title = info.get("track") or info.get("title") or "Untitled"
    guess_artist, guess_title = (
        parse_filename(raw_title) if not info.get("track") else (None, raw_title)
    )
    artist = (
        artist or info.get("artist") or guess_artist or info.get("uploader") or "Unknown Artist"
    )
    title = title or guess_title
    album = album or info.get("album")

    song = SongDir(cfg.library_path / slugify(artist, title)).create()
    if song.meta_path.exists() and song.audio.exists() and not force:
        return song
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(song.root / "source.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    src = song.source_file()
    if src is None:
        raise DeepcutsError(f"download produced no file for {url}")
    return _finish(song, src, cfg, artist, title, album, url)


def _finish(
    song: SongDir, src: Path, cfg: Config, artist: str, title: str, album: str | None, origin: str
) -> SongDir:
    audio.to_wav(src, song.audio)
    meta = SongMeta(
        artist=artist,
        title=title,
        album=album,
        duration=audio.duration(song.audio),
        source=origin,
        language=cfg.language,
    )
    song.save_meta(meta)
    return song
