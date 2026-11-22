"""Burn the ASS karaoke onto a background and mux with the karaoke track (H.264 + AAC MP4)."""

from __future__ import annotations

from pathlib import Path

from deepcuts import audio
from deepcuts.config import RenderConfig, expand
from deepcuts.stages.render.ass import COVER_CENTER_Y, COVER_HEIGHT


def _blurred_fill(W: int, H: int) -> str:
    """Scale+crop to fill the frame, then a cheap heavy blur (blur small, scale back up)."""
    return (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
        f"scale={W // 8}:{H // 8},gblur=sigma=6,scale={W}:{H},setsar=1"
    )


def build_ffmpeg_cmd(
    *,
    ass_name: str,
    track: Path,
    out_path: Path,
    duration: float,
    cfg: RenderConfig,
    vocals: Path | None = None,
    cover: Path | None = None,
    intro: float | None = None,
) -> list[str]:
    """`cover` + `intro` (seconds): open on the cover, fading out so it's gone at `intro`."""
    W, H, fps = cfg.width, cfg.height, cfg.fps
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    vf: list[str] = []
    background = cfg.background
    if background == "cover" and cover is None:
        background = "#101014"  # no art for this song: plain default
    if background.startswith("#"):
        cmd += ["-f", "lavfi", "-i", f"color=c=0x{background.lstrip('#')}:s={W}x{H}:r={fps}"]
    else:
        image = (cover if background == "cover" else expand(background)).resolve()  # type: ignore[union-attr]
        if not image.exists():
            raise FileNotFoundError(f"background image not found: {image}")
        cmd += ["-loop", "1", "-framerate", str(fps), "-i", str(image)]
        if background == "cover":
            vf.append(_blurred_fill(W, H))
        else:
            vf += [
                f"scale={W}:{H}:force_original_aspect_ratio=increase",
                f"crop={W}:{H}",
                "setsar=1",
            ]
        if cfg.background_dim > 0:
            vf.append(f"drawbox=color=black@{cfg.background_dim}:t=fill")
    cmd += ["-i", str(track.resolve())]
    graph = [f"[0:v]{','.join(vf) or 'null'}[base]"]
    amap = "1:a"
    if vocals is not None and cfg.guide_vocals > 0:
        cmd += ["-i", str(vocals.resolve())]
        graph.append(
            f"[2:a]volume={cfg.guide_vocals}[g];[1:a][g]amix=inputs=2:normalize=0:duration=first[a]"
        )
        amap = "[a]"
    below = "[base]"
    if cover is not None and intro is not None:
        idx = 3 if amap == "[a]" else 2
        cmd += [
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-t",
            f"{intro:.3f}",
            "-i",
            str(cover.resolve()),
        ]
        fade = min(cfg.cover_fade, intro / 3)
        ch = round(H * COVER_HEIGHT / 2) * 2
        cy = round(H * COVER_CENTER_Y)
        graph.append(
            f"[{idx}:v]split=2[cb][cf];"
            f"[cb]{_blurred_fill(W, H)},drawbox=color=black@0.35:t=fill[cbg];"
            f"[cf]scale=-2:{ch},setsar=1[cfg];"
            f"[cbg][cfg]overlay=x=(main_w-overlay_w)/2:y={cy}-overlay_h/2,"
            f"format=yuva420p,fade=t=out:st={intro - fade:.3f}:d={fade:.3f}:alpha=1[intro];"
            f"[base][intro]overlay=eof_action=pass[under]"
        )
        below = "[under]"
    ass = f"ass={ass_name}"
    if cfg.fonts_dir:
        ass += f":fontsdir={expand(cfg.fonts_dir).resolve()}"
    graph.append(f"{below}{ass}[v]")
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
    cover: Path | None = None,
    intro: float | None = None,
) -> Path:
    # ffmpeg filter args choke on ':' and quotes in paths, so run inside the ASS file's folder.
    cmd = build_ffmpeg_cmd(
        ass_name=ass_path.name,
        track=track,
        out_path=out_path,
        duration=duration,
        cfg=cfg,
        vocals=vocals,
        cover=cover,
        intro=intro,
    )
    audio.run(cmd, cwd=ass_path.parent)
    return out_path
