"""Client side of `compute = "modal"`.

One ephemeral Modal app is started per deepcuts process (on first use) and stopped at exit, so
`deepcuts make` pays the container start once: the align call lands on the container that just
did the separation (it stays warm for `modal.scaledown_window` seconds).

Audio crosses the wire as 16-bit FLAC in both directions (lossless at audio.wav's depth).
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import time
from pathlib import Path

from deepcuts import audio
from deepcuts.config import Config
from deepcuts.errors import DeepcutsError, MissingExtra
from deepcuts.models import Line, StemSet
from deepcuts.workspace import SongDir

log = logging.getLogger(__name__)

_stack: contextlib.ExitStack | None = None
_worker = None


def worker(cfg: Config):
    """The remote Worker, starting the ephemeral app on first use (once per process)."""
    global _stack, _worker
    if _worker is not None:
        return _worker
    try:
        import modal
        from modal.exception import AuthError, InvalidError
    except ImportError as exc:
        raise MissingExtra("modal", "modal") from exc
    gpu = cfg.modal.gpu
    os.environ["DEEPCUTS_MODAL_GPU"] = ",".join(gpu) if isinstance(gpu, list) else gpu
    os.environ["DEEPCUTS_MODAL_SCALEDOWN"] = str(cfg.modal.scaledown_window)
    os.environ["DEEPCUTS_MODAL_TIMEOUT"] = str(cfg.modal.timeout)
    from deepcuts.remote.modal_app import Worker, app

    stack = contextlib.ExitStack()
    if log.isEnabledFor(logging.DEBUG):
        stack.enter_context(modal.enable_output())  # image build + container logs
    try:
        stack.enter_context(app.run())
    except AuthError as exc:
        stack.close()
        raise DeepcutsError(
            "Modal isn't logged in. Run `uv run modal token new` once, then try again."
        ) from exc
    except InvalidError as exc:
        stack.close()
        hint = ""
        if "payment method" in str(exc):
            hint = (
                "\nAdd a payment method at https://modal.com/settings, then run the same "
                "command again (or use --compute local)."
            )
        raise DeepcutsError(f"Modal refused to start deepcuts' GPU worker: {exc}{hint}") from exc
    atexit.register(stack.close)
    _stack, _worker = stack, Worker()
    return _worker


def flac_bytes(path: Path) -> bytes:
    dst = path.with_suffix(".xfer.flac")
    audio.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-c:a", "flac", "-sample_fmt", "s16", str(dst)]
    )  # fmt: skip
    try:
        return dst.read_bytes()
    finally:
        dst.unlink(missing_ok=True)


def write_wav(data: bytes, dst: Path) -> Path:
    tmp = dst.with_suffix(".xfer.flac")
    tmp.write_bytes(data)
    try:
        return audio.to_wav(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def separate(song: SongDir, cfg: Config, remote=None) -> StemSet:
    """Run the separation stage on Modal; stems land in song/stems like a local run."""
    from deepcuts.stages.separate import make_karaoke

    remote = remote or worker(cfg)
    t0 = time.monotonic()
    upload = flac_bytes(song.audio)
    res = remote.separate.remote(upload, cfg.model_dump(mode="json"))
    log.debug(
        "modal separate: %.1fs round trip (%.1f MB up)", time.monotonic() - t0, len(upload) / 1e6
    )
    for note in res.get("log", []):
        log.debug("modal: %s", note)
    stems_dir = song.stems_dir
    stems_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: write_wav(data, stems_dir / f"{name}.wav") for name, data in res["stems"].items()
    }
    if "vocals" not in paths or "instrumental" not in paths:
        raise DeepcutsError(f"remote separation returned {sorted(paths)}")
    karaoke = make_karaoke(stems_dir, paths["instrumental"], paths.get("backing"), cfg)
    return StemSet(
        vocals=paths["vocals"], instrumental=paths["instrumental"], karaoke=karaoke,
        lead=paths.get("lead"), backing=paths.get("backing"),
        backend="modal", models=list(res["models"]),
    )  # fmt: skip


class RemoteAligner:
    """Aligner protocol, executed on the Modal worker (CTC by default)."""

    def __init__(self, cfg: Config, remote=None) -> None:
        self.cfg = cfg
        self.name = cfg.align.aligner
        self._remote = remote

    def align(
        self, audio_path: Path, lines: list[Line], language: str, strategy: str
    ) -> tuple[list[Line], float | None]:
        remote = self._remote or worker(self.cfg)
        out, offset = remote.align.remote(
            flac_bytes(audio_path),
            [line.model_dump(mode="json") for line in lines],
            language,
            strategy,
            self.cfg.align.model_dump(mode="json"),
        )
        return [Line.model_validate(x) for x in out], offset
