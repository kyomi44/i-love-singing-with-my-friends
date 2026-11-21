"""Local separation with python-audio-separator (RoFormer / MDX-Net / Demucs checkpoints)."""

from __future__ import annotations

import logging
from pathlib import Path

from deepcuts.config import SeparationConfig, expand
from deepcuts.errors import MissingExtra

# audio-separator's MDXC defaults (0.48: None = use the model's own setting); we only set `overlap`.
DEFAULT_MDXC = {
    "segment_size": 256,
    "override_model_segment_size": False,
    "batch_size": None,
    "overlap": None,
    "pitch_shift": 0,
}


class LocalBackend:
    name = "local"

    def __init__(self, cfg: SeparationConfig) -> None:
        try:
            from audio_separator.separator import Separator
        except ImportError as exc:
            raise MissingExtra("separate", "audio-separator") from exc
        self._Separator = Separator
        self.cfg = cfg

    def split(self, model: str, src: Path, out_dir: Path, names: dict[str, str]) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = expand(self.cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {
            "output_dir": str(out_dir),
            "output_format": "WAV",
            "model_file_dir": str(model_dir),
            "log_level": logging.WARNING,
        }
        if self.cfg.overlap is not None:
            kwargs["mdxc_params"] = {**DEFAULT_MDXC, "overlap": self.cfg.overlap}
        sep = self._Separator(**kwargs)
        sep.load_model(model_filename=model)
        outputs = sep.separate(str(src), names) if names else sep.separate(str(src))
        paths = [Path(p) if Path(p).is_absolute() else out_dir / p for p in outputs]
        return [p for p in paths if p.exists()]
