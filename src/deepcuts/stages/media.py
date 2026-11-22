"""Per-song media for the video. Each slot holds an image *or* a video, in the song folder:

    cover.*        the intro: shown with the title, then faded into the lyrics
    background.*   behind the lyrics for the whole song (overrides render.background)

Like the lyrics file, what's in the song folder is the source of truth: `--cover FILE` and
`--background FILE` copy files in, and you can also just drop files there. Images are kept as
JPEG or converted to PNG; videos are copied as they are and loop to fill the time. If there's no
cover, the art embedded in the source audio is extracted (most FLAC/MP3 rips carry some).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from deepcuts import audio
from deepcuts.errors import DeepcutsError
from deepcuts.workspace import SongDir

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".gif")
STORED_EXTS = (".png", ".jpg", *VIDEO_EXTS)  # what a slot file can be named, in lookup order


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def media_file(song: SongDir, slot: str) -> Path | None:
    return next(
        (song.root / f"{slot}{e}" for e in STORED_EXTS if (song.root / f"{slot}{e}").exists()),
        None,
    )


def clear_media(song: SongDir, slot: str) -> bool:
    """Remove the slot's file(s); True if there was anything to remove."""
    found = False
    for ext in STORED_EXTS:
        p = song.root / f"{slot}{ext}"
        if p.exists():
            p.unlink()
            found = True
    return found


def import_media(song: SongDir, slot: str, src: Path) -> Path:
    """Copy an image or video into the slot, replacing whatever was there."""
    src = src.expanduser().resolve()
    if not src.exists():
        raise DeepcutsError(f"{slot} file not found: {src}")
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        raise DeepcutsError(
            f"{src.name}: use an image ({', '.join(IMAGE_EXTS)}) or a video ({', '.join(VIDEO_EXTS)})"
        )
    clear_media(song, slot)
    if ext in VIDEO_EXTS or ext in (".jpg", ".jpeg"):
        dst = song.root / f"{slot}{'.jpg' if ext == '.jpeg' else ext}"
        shutil.copy2(src, dst)
        return dst
    dst = song.root / f"{slot}.png"
    audio.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-frames:v", "1", str(dst)]
    )  # fmt: skip
    return dst


def cover_file(song: SongDir) -> Path | None:
    return media_file(song, "cover")


def background_file(song: SongDir) -> Path | None:
    return media_file(song, "background")


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
