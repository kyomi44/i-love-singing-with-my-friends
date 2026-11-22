"""Burn the ASS karaoke onto a background and mux with the karaoke track (H.264 + AAC MP4)."""

from __future__ import annotations

from pathlib import Path

from deepcuts import audio
from deepcuts.config import RenderConfig, expand
from deepcuts.stages.media import is_video
from deepcuts.stages.render.ass import COVER_CENTER_Y, COVER_HEIGHT


def _fill(W: int, H: int) -> str:
    return f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"


def _blurred_fill(W: int, H: int) -> str:
    """Scale+crop to fill the frame, then a cheap heavy blur (blur small, scale back up)."""
    return f"{_fill(W, H)},scale={W // 8}:{H // 8},gblur=sigma=6,scale={W}:{H},setsar=1"


def _dim(amount: float) -> str:
    """Darken by multiplying RGB. (drawbox=black@a looked right on images, but on YUV video
    frames it blends chroma wrongly and washed a blue clip out to grey.)"""
    k = round(1.0 - amount, 3)
    return f"colorchannelmixer=rr={k}:gg={k}:bb={k}"


def _media_input(path: Path, fps: int, length: float | None = None) -> list[str]:
    """ffmpeg input args: images loop as stills, videos loop end to end (their audio is ignored)."""
    t = ["-t", f"{length:.3f}"] if length is not None else []
    if is_video(path):
        return ["-stream_loop", "-1", *t, "-i", str(path.resolve())]
    return ["-loop", "1", "-framerate", str(fps), *t, "-i", str(path.resolve())]


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
    background: Path | None = None,
) -> list[str]:
    """Video = background, then the cover intro (if `cover` + `intro`), then the lyrics.

    `background` (the song folder's background image/video) wins over render.background, which
    is a "#hex" colour, "cover" (the cover, blurred), or a path. `intro` is when the cover has
    fully faded out. Images and videos work in either slot.
    """
    W, H, fps = cfg.width, cfg.height, cfg.fps
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    src, blur = background, False
    if src is None and cfg.background == "cover":
        src, blur = cover, True  # None if the song has no art: plain colour below
    elif src is None and not cfg.background.startswith("#"):
        src = expand(cfg.background).resolve()
        if not src.exists():
            raise FileNotFoundError(f"background not found: {src}")
    if src is None:
        colour = cfg.background if cfg.background.startswith("#") else "#101014"
        cmd += ["-f", "lavfi", "-i", f"color=c=0x{colour.lstrip('#')}:s={W}x{H}:r={fps}"]
        vf = ["null"]
    else:
        cmd += _media_input(src, fps)
        vf = [f"fps={fps}", _blurred_fill(W, H) if blur else _fill(W, H)]
        if cfg.background_dim > 0:
            vf.append(_dim(cfg.background_dim))
    cmd += ["-i", str(track.resolve())]
    graph = [f"[0:v]{','.join(vf)}[base]"]
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
        cmd += _media_input(cover, fps, intro)
        fade = min(cfg.cover_fade, intro / 3)
        ch = round(H * COVER_HEIGHT / 2) * 2
        cy = round(H * COVER_CENTER_Y)
        graph.append(
            f"[{idx}:v]fps={fps},split=2[cb][cf];"
            f"[cb]{_blurred_fill(W, H)},{_dim(0.35)}[cbg];"
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
    background: Path | None = None,
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
        background=background,
    )
    audio.run(cmd, cwd=ass_path.parent)
    return out_path
