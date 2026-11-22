"""The GPU half of deepcuts (stem separation + CTC alignment) as a Modal app.

The container runs the *same* stage code as a local install: its Python environment comes from
this repo's uv.lock (extras `separate` + `align`), and the local `deepcuts` package is mounted on
every run, so local and remote can't drift. Model weights live on a Modal Volume and are
downloaded once.

You don't deploy this by hand: `compute = "modal"` starts it as an ephemeral app for the length of
one deepcuts command (see client.py). The first run builds the image (a few minutes, once).
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path

import modal

MODELS = "/models"
ROOT = Path(__file__).resolve().parents[3]  # repo root: has pyproject.toml + uv.lock

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "build-essential")
    .uv_sync(str(ROOT), extras=["separate", "align"])
    .env({"HF_HOME": f"{MODELS}/hf", "TORCH_HOME": f"{MODELS}/torch"})
    .add_local_python_source("deepcuts")
)
models = modal.Volume.from_name("deepcuts-models", create_if_missing=True)
app = modal.App("deepcuts")

STEMS = ("vocals", "instrumental", "lead", "backing")


@app.cls(
    image=image,
    # Read when the app starts (client.py sets these from [modal] before importing this module):
    # Modal checks the GPU type at app start, so with_options() afterwards would be too late.
    gpu=os.environ.get("DEEPCUTS_MODAL_GPU", "L4,A10G,T4").split(","),
    volumes={MODELS: models},
    scaledown_window=int(os.environ.get("DEEPCUTS_MODAL_SCALEDOWN", "120")),
    timeout=int(os.environ.get("DEEPCUTS_MODAL_TIMEOUT", "1200")),
)
class Worker:
    @modal.enter()
    def setup(self) -> None:
        self._aligners: dict[tuple, object] = {}

    @modal.method()
    def separate(self, audio_flac: bytes, cfg: dict) -> dict:
        """Mix (FLAC bytes) -> {"stems": {name: FLAC bytes}, "models": [...]}."""
        from deepcuts import audio
        from deepcuts.config import Config
        from deepcuts.remote.client import flac_bytes
        from deepcuts.stages.separate import run_separation
        from deepcuts.workspace import SongDir

        c = Config.model_validate(cfg)
        c.compute = "local"
        c.separation.backend = "local"
        c.separation.model_dir = Path(MODELS) / "audio-separator"
        t0 = time.monotonic()
        with _captured_logs() as notes, tempfile.TemporaryDirectory() as td:
            song = SongDir(Path(td) / "song").create()
            src = song.root / "source.flac"
            src.write_bytes(audio_flac)
            audio.to_wav(src, song.audio)
            stems = run_separation(song, c)
            models.commit()  # keep any newly downloaded checkpoints
            t1 = time.monotonic()
            out = {
                name: flac_bytes(getattr(stems, name))
                for name in STEMS
                if getattr(stems, name) is not None
            }
            notes.append(
                f"on {_gpu_name()}: separation {t1 - t0:.1f}s, encode {time.monotonic() - t1:.1f}s"
            )
        _free_gpu()
        return {"stems": out, "models": stems.models, "log": notes}

    @modal.method()
    def align(
        self, stem_flac: bytes, lines: list[dict], language: str, strategy: str, align_cfg: dict
    ) -> tuple[list[dict], float | None]:
        from deepcuts import audio
        from deepcuts.config import AlignConfig
        from deepcuts.models import Line
        from deepcuts.stages.align import get_aligner

        cfg = AlignConfig.model_validate(align_cfg)
        key = (cfg.aligner, cfg.ctc_model)
        aligner = self._aligners.get(key)
        if aligner is None:
            aligner = self._aligners[key] = get_aligner(cfg, "cuda")
            models.commit()
        aligner.cfg = cfg  # windows / batch size may differ between calls
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "stem.flac"
            src.write_bytes(stem_flac)
            wav = audio.to_wav(src, Path(td) / "stem.wav")
            out, offset = aligner.align(
                wav, [Line.model_validate(x) for x in lines], language, strategy
            )
        return [line.model_dump(mode="json") for line in out], offset


@contextlib.contextmanager
def _captured_logs():
    """Collect deepcuts INFO logs from inside the container so the client can show them."""
    notes: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            notes.append(record.getMessage())

    logger = logging.getLogger("deepcuts")
    handler, old_level = _Collect(level=logging.INFO), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield notes
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown GPU"


def _free_gpu() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
