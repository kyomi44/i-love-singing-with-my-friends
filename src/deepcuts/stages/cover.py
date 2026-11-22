"""Cover art for the video intro: ``cover.png`` / ``cover.jpg`` in the song folder.

Like the lyrics file, the cover in the song folder is the source of truth. It comes from
``--cover FILE`` (any format ffmpeg reads, normalised to PNG unless it's already a JPEG), or,
if there is none, from art embedded in the source audio (most FLAC/MP3 rips carry some).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from deepcuts import audio
from deepcuts.errors import DeepcutsError
from deepcuts.workspace import SongDir

COVER_NAMES = ("cover.png", "cover.jpg")


def cover_file(song: SongDir) -> Path | None:
    return next((song.root / n for n in COVER_NAMES if (song.root / n).exists()), None)


def import_cover(song: SongDir, src: Path) -> Path:
    """Copy (JPEG) or convert (anything else) an image to the song's cover file."""
    src = src.expanduser().resolve()
    if not src.exists():
        raise DeepcutsError(f"cover image not found: {src}")
    for name in COVER_NAMES:
        (song.root / name).unlink(missing_ok=True)
    if src.suffix.lower() in (".jpg", ".jpeg"):
        dst = song.root / "cover.jpg"
        shutil.copy2(src, dst)
        return dst
    dst = song.root / "cover.png"
    audio.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-frames:v", "1", str(dst)]
    )  # fmt: skip
    return dst


def ensure_cover(song: SongDir) -> Path | None:
    """The song's cover, extracting embedded art from the source audio the first time."""
    existing = cover_file(song)
    if existing is not None:
        return existing
    src = song.source_file()
    if src is None:
        return None
    pics = [
        s
        for s in audio.probe(src).get("streams", [])
        if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic")
    ]
    if not pics:
        return None
    dst = song.root / "cover.png"
    audio.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-map", f"0:{pics[0]['index']}", "-frames:v", "1", str(dst)]
    )  # fmt: skip
    return dst
