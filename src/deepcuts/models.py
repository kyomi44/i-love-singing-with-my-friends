"""Data model shared by every pipeline stage.

All times are seconds on the timeline of the song's ingested ``audio.wav``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    """0..1 from the aligner. 0.0 means the time was interpolated; None means never aligned."""

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None


class Line(BaseModel):
    words: list[Word]
    singer: str | None = None
    """Duet part: "A", "B" or "AB". From ``@A``/``@B`` tags in the lyrics, or diarization."""
    anchor: float | None = None
    """Line start from synced (LRC) lyrics; a prior for alignment, not ground truth."""
    anchor_end: float | None = None
    """End marker from an empty timestamped LRC line (instrumental break), if any."""
    stanza: int = 0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def start(self) -> float | None:
        return next((w.start for w in self.words if w.start is not None), None)

    @property
    def end(self) -> float | None:
        return next((w.end for w in reversed(self.words) if w.end is not None), None)

    @property
    def confidence(self) -> float | None:
        vals = [w.confidence for w in self.words if w.confidence is not None]
        return sum(vals) / len(vals) if vals else None


class Lyrics(BaseModel):
    lines: list[Line]
    source: str
    path: Path | None = None

    @property
    def synced(self) -> bool:
        return bool(self.lines) and all(line.anchor is not None for line in self.lines)


class SongMeta(BaseModel):
    artist: str
    title: str
    album: str | None = None
    duration: float
    source: str
    language: str = "eng"
    """ISO 639-3 code (what the MMS aligner wants). Edit meta.json to change per song."""


class StemSet(BaseModel):
    vocals: Path
    instrumental: Path
    karaoke: Path
    """The track you actually sing over (instrumental, optionally + backing vocals)."""
    lead: Path | None = None
    backing: Path | None = None
    backend: str = "local"
    models: list[str] = Field(default_factory=list)


class Alignment(BaseModel):
    lines: list[Line]
    aligner: str
    stem: str
    strategy: str
    lrc_offset: float | None = None
    """Estimated (audio - LRC) offset in seconds, when synced lyrics were used as anchors."""


class Overrides(BaseModel):
    """Hand corrections layered on top of the alignment; written by the editor / `nudge`."""

    global_offset_ms: int = 0
    line_shift_ms: dict[int, int] = Field(default_factory=dict)
