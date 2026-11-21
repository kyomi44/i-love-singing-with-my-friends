"""Thin ffmpeg/ffprobe wrappers. ffmpeg (with libass) is the only hard system dependency."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from deepcuts.errors import FFmpegError


def require(binary: str = "ffmpeg") -> str:
    path = shutil.which(binary)
    if path is None:
        raise FFmpegError(
            f"{binary} not found on PATH. Install ffmpeg (brew install ffmpeg / apt install ffmpeg)."
        )
    return path


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    require(cmd[0])
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"{cmd[0]} exited with {proc.returncode}:\n{tail}")
    return proc


def probe(path: Path) -> dict:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(out.stdout)


def duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


def tags(path: Path) -> dict[str, str]:
    """Container + stream tags, lower-cased keys (Ogg/Opus keep tags on the stream)."""
    info = probe(path)
    merged: dict[str, str] = {}
    for stream in info.get("streams", []):
        merged.update({k.lower(): v for k, v in (stream.get("tags") or {}).items()})
    merged.update({k.lower(): v for k, v in (info.get("format", {}).get("tags") or {}).items()})
    return merged


def to_wav(src: Path, dst: Path, sample_rate: int = 44100, channels: int = 2) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn",
         "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    )  # fmt: skip
    return dst


def mix(inputs: list[Path], dst: Path, gains: list[float] | None = None) -> Path:
    """Sum stems without normalisation (32-bit float WAV so nothing clips)."""
    gains = gains or [1.0] * len(inputs)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    chains = [f"[{i}:a]volume={g}[a{i}]" for i, g in enumerate(gains)]
    labels = "".join(f"[a{i}]" for i in range(len(inputs)))
    graph = (
        ";".join(chains) + f";{labels}amix=inputs={len(inputs)}:normalize=0:duration=longest[out]"
    )
    cmd += ["-filter_complex", graph, "-map", "[out]", "-c:a", "pcm_f32le", str(dst)]
    run(cmd)
    return dst


def has_filter(name: str) -> bool:
    try:
        out = run(["ffmpeg", "-hide_banner", "-filters"]).stdout
    except FFmpegError:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())
