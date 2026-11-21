"""Per-song working directory + stage cache.

    ~/deepcuts/Artist - Title/
        source.flac            original file (copied)        audio.wav      normalised 44.1k stereo
        meta.json              artist/title/language (edit freely)
        lyrics.lrc|lyrics.txt  editable lyrics (source of truth for alignment)
        stems/                 vocals, instrumental, lead, backing, karaoke (.wav)
        alignment.json         word timings straight from the aligner (+ post-processing)
        overrides.json         your hand corrections (editor / `deepcuts nudge`)
        diarization.json       singer turns, when diarization is enabled
        out/                   lyrics.ass, lyrics.lrc, timing.json, "<name> (karaoke).mp4"
        manifest.json          fingerprints so unchanged stages are skipped

A stage re-runs only when its inputs (file contents / sizes) or its config section change.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from deepcuts.models import Alignment, Overrides, SongMeta, StemSet

INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
TEXT_SUFFIXES = {".lrc", ".txt", ".json", ".ass", ".toml"}


def slugify(artist: str, title: str) -> str:
    name = INVALID_CHARS.sub("_", f"{artist} - {title}").strip(" .")
    return name[:150] or "untitled"


def file_fingerprint(path: Path | None) -> str:
    if path is None or not path.exists():
        return "missing"
    if path.suffix.lower() in TEXT_SUFFIXES:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    st = path.stat()
    return f"{st.st_size}-{st.st_mtime_ns}"


def fingerprint(inputs: list[Path | None], params: dict) -> str:
    h = hashlib.sha256()
    for p in inputs:
        h.update((p.name if p else "-").encode())
        h.update(file_fingerprint(p).encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


class SongDir:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ---- paths -------------------------------------------------------------------------
    meta_path = property(lambda self: self.root / "meta.json")
    audio = property(lambda self: self.root / "audio.wav")
    lyrics_lrc = property(lambda self: self.root / "lyrics.lrc")
    lyrics_txt = property(lambda self: self.root / "lyrics.txt")
    lyrics_info = property(lambda self: self.root / "lyrics.source.json")
    stems_dir = property(lambda self: self.root / "stems")
    stems_path = property(lambda self: self.root / "stems" / "stems.json")
    alignment_path = property(lambda self: self.root / "alignment.json")
    overrides_path = property(lambda self: self.root / "overrides.json")
    diarization_path = property(lambda self: self.root / "diarization.json")
    out_dir = property(lambda self: self.root / "out")
    manifest_path = property(lambda self: self.root / "manifest.json")

    @property
    def name(self) -> str:
        return self.root.name

    def create(self) -> SongDir:
        for d in (self.root, self.stems_dir, self.out_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def source_file(self) -> Path | None:
        return next(iter(sorted(self.root.glob("source.*"))), None)

    # ---- lyrics ------------------------------------------------------------------------
    def lyrics_file(self) -> Path | None:
        for p in (self.lyrics_lrc, self.lyrics_txt):
            if p.exists():
                return p
        return None

    def write_lyrics(self, content: str, kind: str, source: str) -> Path:
        for p in (self.lyrics_lrc, self.lyrics_txt):
            p.unlink(missing_ok=True)
        target = self.lyrics_lrc if kind == "lrc" else self.lyrics_txt
        target.write_text(content.strip() + "\n", encoding="utf-8")
        self.lyrics_info.write_text(json.dumps({"source": source}, indent=2))
        return target

    def lyrics_source(self) -> str | None:
        if self.lyrics_info.exists():
            return json.loads(self.lyrics_info.read_text()).get("source")
        return None

    # ---- json models -------------------------------------------------------------------
    def load_meta(self) -> SongMeta:
        return SongMeta.model_validate_json(self.meta_path.read_text())

    def save_meta(self, meta: SongMeta) -> None:
        self.meta_path.write_text(meta.model_dump_json(indent=2))

    def load_alignment(self) -> Alignment:
        return Alignment.model_validate_json(self.alignment_path.read_text())

    def save_alignment(self, alignment: Alignment) -> None:
        self.alignment_path.write_text(alignment.model_dump_json(indent=2, exclude_none=True))

    def load_overrides(self) -> Overrides:
        if self.overrides_path.exists():
            return Overrides.model_validate_json(self.overrides_path.read_text())
        return Overrides()

    def save_overrides(self, overrides: Overrides) -> None:
        self.overrides_path.write_text(overrides.model_dump_json(indent=2))

    def save_stems(self, stems: StemSet) -> None:
        data = stems.model_dump(mode="json")
        for key in ("vocals", "instrumental", "karaoke", "lead", "backing"):
            if data.get(key):
                data[key] = str(Path(data[key]).resolve().relative_to(self.root.resolve()))
        self.stems_path.write_text(json.dumps(data, indent=2))

    def load_stems(self) -> StemSet:
        data = json.loads(self.stems_path.read_text())
        for key in ("vocals", "instrumental", "karaoke", "lead", "backing"):
            if data.get(key):
                data[key] = str(self.root / data[key])
        return StemSet.model_validate(data)

    # ---- stage cache -------------------------------------------------------------------
    def manifest(self) -> dict[str, str]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def is_fresh(self, stage: str, fp: str) -> bool:
        return self.manifest().get(stage) == fp

    def mark(self, stage: str, fp: str) -> None:
        m = self.manifest()
        m[stage] = fp
        self.manifest_path.write_text(json.dumps(m, indent=2))


def resolve_song(ref: str | Path, library: Path) -> SongDir:
    """Accept a song dir path, a folder name inside the library, or a unique substring of one."""
    p = Path(ref).expanduser()
    if (p / "meta.json").exists():
        return SongDir(p)
    if (library / str(ref) / "meta.json").exists():
        return SongDir(library / str(ref))
    needle = str(ref).lower()
    hits = [
        d for d in library.glob("*/") if needle in d.name.lower() and (d / "meta.json").exists()
    ]
    if len(hits) == 1:
        return SongDir(hits[0])
    if not hits:
        raise FileNotFoundError(f"no song matching {ref!r} in {library}")
    names = ", ".join(sorted(h.name for h in hits)[:8])
    raise FileNotFoundError(f"{ref!r} is ambiguous: {names}")
