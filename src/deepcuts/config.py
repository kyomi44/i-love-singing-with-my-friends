"""Configuration. Defaults live here; override in ./deepcuts.toml or ~/.config/deepcuts/config.toml.

See deepcuts.example.toml for a commented copy of every option.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


class SeparationConfig(BaseModel):
    backend: Literal["local", "mvsep"] = "local"
    vocal_model: str = "vocals_mel_band_roformer.ckpt"
    """Step 1: mix -> vocals + instrumental. Top vocal SDR in audio-separator's own benchmark;
    `audio-separator --list_models` shows alternatives (e.g. model_bs_roformer_ep_317_sdr_12.9755.ckpt)."""
    karaoke_models: list[str] = Field(
        default_factory=lambda: [
            "mel_band_roformer_karaoke_becruily.ckpt",
            "mel_band_roformer_karaoke_gabox.ckpt",
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        ]
    )
    """Step 2: vocals -> lead + backing. Tried in order until one loads."""
    backing: Literal["keep", "remove"] = "keep"
    """keep = harmonies/backing vocals stay in the karaoke track (how real karaoke tracks sound)."""
    model_dir: Path = Path("~/.cache/deepcuts/models")
    overlap: int | None = None
    """RoFormer overlap (2 = faster, 8 = best seams). None = audio-separator default."""


class MvsepConfig(BaseModel):
    """Remote separation on mvsep.com (top of the leaderboards, needs an API token)."""

    api_token_env: str = "MVSEP_API_TOKEN"
    base_url: str = "https://mvsep.com/api"
    vocals_sep_type: int = 40  # BS Roformer (vocals, instrumental)
    vocals_add_opt1: str = "81"  # model version; run `deepcuts mvsep-algorithms roformer` for newer
    karaoke_sep_type: int = 49  # MVSep Karaoke (lead/back vocals)
    karaoke_add_opt1: str = "6"
    output_format: int = 2  # 2 = FLAC 16-bit (smaller uploads/downloads than WAV)
    poll_seconds: float = 5.0
    timeout_minutes: float = 30.0


class LyricsConfig(BaseModel):
    sources: list[Literal["lrclib", "syncedlyrics"]] = Field(
        default_factory=lambda: ["lrclib", "syncedlyrics"]
    )
    syncedlyrics_providers: list[str] = Field(
        default_factory=lambda: ["Lrclib", "Musixmatch", "NetEase", "Megalobiz"]
    )
    duration_tolerance: float = 3.0
    """Seconds of slack when matching LRCLIB entries to your file's duration."""
    asr_fallback: bool = False
    """Transcribe the vocal stem with WhisperX when no lyrics are found anywhere."""
    asr_model: str = "large-v3"


class AlignConfig(BaseModel):
    aligner: Literal["ctc", "whisperx"] = "ctc"
    stem: Literal["vocals", "lead"] = "vocals"
    """Which stem to align against. 'vocals' includes backing parts that lyrics usually list."""
    strategy: Literal["auto", "global", "anchored"] = "auto"
    """auto: global pass -> estimate LRC offset -> per-line anchored refinement -> keep the best."""
    device: str = "auto"
    batch_size: int = 8
    ctc_model: str | None = None
    """HF model id for ctc-forced-aligner. None = MahmoudAshraf/mms-300m-1130-forced-aligner."""
    window_pad_before: float = 0.6
    window_pad_after: float = 1.0
    min_word_dur: float = 0.08
    max_word_dur: float = 6.0
    gap_close: float = 0.35
    """Gaps shorter than this between words in a line are closed so the wipe never stutters."""
    low_confidence: float = 0.35


class DiarizeConfig(BaseModel):
    enabled: bool = False
    model: str = "pyannote/speaker-diarization-community-1"
    hf_token_env: str = "HF_TOKEN"
    num_speakers: int | None = 2
    both_ratio: float = 0.35
    """A line becomes "AB" when the 2nd singer covers at least this fraction of it."""


class RenderConfig(BaseModel):
    video: bool = True
    width: int = 1920
    height: int = 1080
    fps: int = 30
    font: str = "Arial"
    fonts_dir: Path | None = None
    font_size: int = 76
    bold: bool = True
    outline: float = 4.0
    sung_color: str = "#4FC3F7"
    unsung_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    singer_colors: dict[str, str] = Field(
        default_factory=lambda: {"A": "#4FC3F7", "B": "#F48FB1", "AB": "#CE93D8"}
    )
    background: str = "#101014"
    """A hex colour or a path to an image (album art works great)."""
    background_dim: float = 0.45
    lead_in: float = 1.2
    """Seconds a line is shown before its first word."""
    hold: float = 0.4
    countdown_gap: float = 6.0
    """Show a 3-2-1 countdown before lines that follow a gap at least this long."""
    title_card: bool = True
    highlight_lead_ms: int = 0
    """Start every wipe this many ms early. Some singers like ~50-100 ms of warning."""
    guide_vocals: float = 0.0
    """Mix the vocal stem back in at this gain (0.1-0.2 = quiet guide for unfamiliar songs)."""
    crf: int = 20
    preset: str = "veryfast"
    audio_bitrate: str = "320k"


class Config(BaseModel):
    library: Path = Path("~/deepcuts")
    language: str = "eng"
    separation: SeparationConfig = Field(default_factory=SeparationConfig)
    mvsep: MvsepConfig = Field(default_factory=MvsepConfig)
    lyrics: LyricsConfig = Field(default_factory=LyricsConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    diarize: DiarizeConfig = Field(default_factory=DiarizeConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    @property
    def library_path(self) -> Path:
        return expand(self.library)


CONFIG_LOCATIONS = (Path("deepcuts.toml"), Path("~/.config/deepcuts/config.toml"))


def load_config(path: Path | None = None) -> Config:
    if path is not None:
        p = expand(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        return Config.model_validate(tomllib.loads(p.read_text()))
    for candidate in CONFIG_LOCATIONS:
        p = expand(candidate)
        if p.exists():
            return Config.model_validate(tomllib.loads(p.read_text()))
    return Config()


def with_overrides(cfg: Config, **dotted: object) -> Config:
    """Return a copy with e.g. ``{"align.aligner": "whisperx"}`` applied; None values are skipped."""
    data = cfg.model_dump()
    for key, value in dotted.items():
        if value is None:
            continue
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node[part]
        node[leaf] = value
    return Config.model_validate(data)
