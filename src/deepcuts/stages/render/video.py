"""Burn the ASS karaoke onto a background and mux with the karaoke track (H.264 + AAC MP4)."""

from __future__ import annotations

from pathlib import Path

from deepcuts import audio
from deepcuts.config import RenderConfig, expand


def build_ffmpeg_cmd(
    *,
    ass_name: str,
    track: Path,
    out_path: Path,
    duration: float,
    cfg: RenderConfig,
    vocals: Path | None = None,
) -> list[str]:
    W, H, fps = cfg.width, cfg.height, cfg.fps
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    vf: list[str] = []
    if cfg.background.startswith("#"):
        cmd += ["-f", "lavfi", "-i", f"color=c=0x{cfg.background.lstrip('#')}:s={W}x{H}:r={fps}"]
    else:
        image = expand(cfg.background).resolve()
        if not image.exists():
            raise FileNotFoundError(f"background image not found: {image}")
        cmd += ["-loop", "1", "-framerate", str(fps), "-i", str(image)]
        vf += [f"scale={W}:{H}:force_original_aspect_ratio=increase", f"crop={W}:{H}", "setsar=1"]
        if cfg.background_dim > 0:
            vf.append(f"drawbox=color=black@{cfg.background_dim}:t=fill")
    ass = f"ass={ass_name}"
    if cfg.fonts_dir:
        ass += f":fontsdir={expand(cfg.fonts_dir).resolve()}"
    vf.append(ass)
    cmd += ["-i", str(track.resolve())]
    graph = [f"[0:v]{','.join(vf)}[v]"]
    amap = "1:a"
    if vocals is not None and cfg.guide_vocals > 0:
        cmd += ["-i", str(vocals.resolve())]
        graph.append(
            f"[2:a]volume={cfg.guide_vocals}[g];[1:a][g]amix=inputs=2:normalize=0:duration=first[a]"
        )
        amap = "[a]"
    cmd += [
        "-filter_complex", ";".join(graph), "-map", "[v]", "-map", amap,
        "-t", f"{duration:.3f}", "-r", str(fps),
        "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", cfg.audio_bitrate, "-movflags", "+faststart",
        str(out_path.resolve()),
    ]  # fmt: skip
    return cmd


def render_video(
    *,
    ass_path: Path,
    track: Path,
    out_path: Path,
    duration: float,
    cfg: RenderConfig,
    vocals: Path | None = None,
) -> Path:
    # ffmpeg filter args choke on ':' and quotes in paths, so run inside the ASS file's folder.
    cmd = build_ffmpeg_cmd(
        ass_name=ass_path.name,
        track=track,
        out_path=out_path,
        duration=duration,
        cfg=cfg,
        vocals=vocals,
    )
    audio.run(cmd, cwd=ass_path.parent)
    return out_path
