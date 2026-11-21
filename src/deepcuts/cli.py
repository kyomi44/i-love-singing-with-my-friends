"""deepcuts command line.

deepcuts make song.flac                     # everything, end to end
deepcuts make "https://...bandcamp..." -a "Artist" -t "Title"
deepcuts serve "title"                      # fix timing by ear in the browser
deepcuts nudge "title" --ms -120            # or just shift everything
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from deepcuts import __version__
from deepcuts.config import Config, load_config, with_overrides
from deepcuts.errors import DeepcutsError
from deepcuts.pipeline import STAGES, Pipeline
from deepcuts.workspace import SongDir, resolve_song

app = typer.Typer(
    name="deepcuts",
    help="Karaoke for deep cuts: any track -> separated stems + word-accurate lyrics -> video.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
out = Console()
err = Console(stderr=True)

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to a deepcuts.toml.")]
ForceOpt = Annotated[
    list[str],
    typer.Option(
        "--force", "-f", help=f"Re-run stage(s) even if cached: {', '.join(STAGES)}, all."
    ),
]
SongArg = Annotated[str, typer.Argument(help="Song folder, or any unique part of its name.")]


def _cfg(config: Path | None, **dotted: object) -> Config:
    return with_overrides(load_config(config), **dotted)


def _song(ref: str, cfg: Config) -> SongDir:
    try:
        return resolve_song(ref, cfg.library_path)
    except FileNotFoundError as exc:
        raise DeepcutsError(str(exc)) from exc


def _lyrics_or_die(pipe: Pipeline, song: SongDir):
    lyrics = pipe.lyrics(song)
    if lyrics is None:
        raise DeepcutsError(f"no lyrics yet: add lyrics.txt or lyrics.lrc to {song.root}")
    return lyrics


def _print_flagged(song: SongDir, cfg: Config) -> None:
    from deepcuts.stages.align.postprocess import low_confidence

    aln = song.load_alignment()
    flagged = low_confidence(aln.lines, cfg.align.low_confidence)
    if not flagged:
        out.print("[green]All lines aligned confidently.[/]")
        return
    table = Table(
        title=f"{len(flagged)} line(s) worth a listen", show_edge=False, header_style="bold"
    )
    table.add_column("#", justify="right")
    table.add_column("time", justify="right")
    table.add_column("conf", justify="right")
    table.add_column("line")
    for i, conf in flagged[:25]:
        line = aln.lines[i]
        table.add_row(str(i + 1), f"{line.start or 0:.2f}", f"{conf:.0%}", line.text[:70])
    out.print(table)
    out.print(f'Fix by ear: [bold]deepcuts serve "{song.name}"[/]')


@app.callback(invoke_without_command=True)
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING, format="%(name)s: %(message)s"
    )
    if version:
        out.print(f"deepcuts {__version__}")
        raise typer.Exit()


@app.command()
def make(
    source: Annotated[str, typer.Argument(help="Audio file (flac/mp3/wav/...) or URL.")],
    artist: Annotated[str | None, typer.Option("--artist", "-a")] = None,
    title: Annotated[str | None, typer.Option("--title", "-t")] = None,
    album: Annotated[str | None, typer.Option("--album")] = None,
    lyrics: Annotated[
        Path | None, typer.Option("--lyrics", "-l", help=".lrc or .txt to use.")
    ] = None,
    language: Annotated[str | None, typer.Option(help="ISO 639-3, e.g. eng, fra, jpn.")] = None,
    backing: Annotated[str | None, typer.Option(help="keep | remove backing vocals.")] = None,
    separator: Annotated[str | None, typer.Option(help="local | mvsep")] = None,
    aligner: Annotated[str | None, typer.Option(help="ctc | whisperx")] = None,
    stem: Annotated[str | None, typer.Option(help="Align against: vocals | lead")] = None,
    diarize: Annotated[
        bool | None, typer.Option("--diarize/--no-diarize", help="Detect duet parts.")
    ] = None,
    asr: Annotated[
        bool, typer.Option("--asr", help="Transcribe if no lyrics exist anywhere.")
    ] = False,
    guide_vocals: Annotated[
        float | None, typer.Option(help="Mix vocals back in at this gain.")
    ] = None,
    background: Annotated[str | None, typer.Option(help="#hex colour or image path.")] = None,
    no_video: Annotated[
        bool, typer.Option("--no-video", help="Only write .ass/.lrc/.json.")
    ] = False,
    force: ForceOpt = [],
    config: ConfigOpt = None,
) -> None:
    """Run the whole pipeline: ingest -> lyrics -> separate -> align -> render."""
    cfg = _cfg(
        config,
        language=language,
        **{
            "separation.backing": backing,
            "separation.backend": separator,
            "align.aligner": aligner,
            "align.stem": stem,
            "diarize.enabled": diarize,
            "lyrics.asr_fallback": True if asr else None,
            "render.guide_vocals": guide_vocals,
            "render.background": background,
            "render.video": False if no_video else None,
        },
    )
    result = Pipeline(cfg, force, err).make(source, artist, title, album, lyrics)
    out.print()
    for kind, path in result.outputs.items():
        out.print(f"  [dim]{kind:>5}[/]  {path}")
    out.print()
    _print_flagged(result.song, cfg)


@app.command("lyrics")
def lyrics_cmd(
    song: SongArg,
    file: Annotated[Path | None, typer.Option("--file", help="Use this .lrc/.txt instead.")] = None,
    refetch: Annotated[
        bool, typer.Option("--refetch", help="Look lyrics up again online.")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Fetch or replace a song's lyrics (then edit lyrics.lrc/.txt by hand if needed)."""
    cfg = _cfg(config)
    sd = _song(song, cfg)
    lyr = Pipeline(cfg, ["lyrics"] if refetch else [], err).lyrics(sd, file)
    if lyr is None:
        raise DeepcutsError("no lyrics found; pass --file")
    out.print(lyr.path)


@app.command()
def separate(song: SongArg, force: ForceOpt = [], config: ConfigOpt = None) -> None:
    """(Re)run stem separation."""
    cfg = _cfg(config)
    Pipeline(cfg, force, err).separate(_song(song, cfg))


@app.command()
def align(
    song: SongArg,
    aligner: Annotated[str | None, typer.Option(help="ctc | whisperx")] = None,
    stem: Annotated[str | None, typer.Option(help="vocals | lead")] = None,
    strategy: Annotated[str | None, typer.Option(help="auto | global | anchored")] = None,
    force: ForceOpt = [],
    config: ConfigOpt = None,
) -> None:
    """(Re)align lyrics. Re-runs automatically when you edit the lyrics file."""
    cfg = _cfg(config, **{"align.aligner": aligner, "align.stem": stem, "align.strategy": strategy})
    sd = _song(song, cfg)
    pipe = Pipeline(cfg, force, err)
    pipe.align(sd, _lyrics_or_die(pipe, sd), pipe.separate(sd))
    _print_flagged(sd, cfg)


@app.command("diarize")
def diarize_cmd(song: SongArg, force: ForceOpt = [], config: ConfigOpt = None) -> None:
    """Detect which singer sings each line (duets). Enables diarization for later renders too."""
    cfg = _cfg(config, **{"diarize.enabled": True})
    sd = _song(song, cfg)
    pipe = Pipeline(cfg, force, err)
    pipe.diarize(sd, pipe.separate(sd))


@app.command()
def render(
    song: SongArg,
    guide_vocals: Annotated[
        float | None, typer.Option(help="Mix vocals back in at this gain.")
    ] = None,
    background: Annotated[str | None, typer.Option(help="#hex colour or image path.")] = None,
    no_video: Annotated[bool, typer.Option("--no-video")] = False,
    diarize: Annotated[bool | None, typer.Option("--diarize/--no-diarize")] = None,
    force: ForceOpt = [],
    config: ConfigOpt = None,
) -> None:
    """Re-render outputs from the current alignment + overrides (no ML needed)."""
    cfg = _cfg(
        config,
        **{
            "render.guide_vocals": guide_vocals,
            "render.background": background,
            "render.video": False if no_video else None,
            "diarize.enabled": diarize,
        },
    )
    outputs = Pipeline(cfg, force, err).render(_song(song, cfg))
    for kind, path in outputs.items():
        out.print(f"  [dim]{kind:>5}[/]  {path}")


@app.command()
def nudge(
    song: SongArg,
    ms: Annotated[int, typer.Option("--ms", help="Milliseconds to add (negative = earlier).")] = 0,
    line: Annotated[
        int | None, typer.Option("--line", help="1-based line number; omit for all.")
    ] = None,
    reset: Annotated[bool, typer.Option("--reset", help="Drop all overrides.")] = False,
    rerender: Annotated[bool, typer.Option("--render/--no-render")] = True,
    config: ConfigOpt = None,
) -> None:
    """Shift timing without re-aligning: whole song, or one line."""
    from deepcuts.models import Overrides

    cfg = _cfg(config)
    sd = _song(song, cfg)
    ov = Overrides() if reset else sd.load_overrides()
    if line is None:
        ov.global_offset_ms += ms
    else:
        idx = line - 1
        ov.line_shift_ms[idx] = ov.line_shift_ms.get(idx, 0) + ms
        if not ov.line_shift_ms[idx]:
            del ov.line_shift_ms[idx]
    sd.save_overrides(ov)
    out.print(f"global {ov.global_offset_ms:+d} ms, {len(ov.line_shift_ms)} line override(s)")
    if rerender:
        Pipeline(cfg, (), err).render(sd)


@app.command()
def report(song: SongArg, config: ConfigOpt = None) -> None:
    """List lines the aligner was unsure about."""
    cfg = _cfg(config)
    sd = _song(song, cfg)
    aln = sd.load_alignment()
    out.print(f"[bold]{sd.name}[/]  aligner={aln.aligner} strategy={aln.strategy} stem={aln.stem}")
    if aln.lrc_offset is not None:
        out.print(f"LRC anchors were {aln.lrc_offset:+.2f}s off this audio (auto-corrected)")
    _print_flagged(sd, cfg)


@app.command()
def serve(
    song: SongArg,
    port: Annotated[int, typer.Option(help="Port for the local editor.")] = 8765,
    host: Annotated[
        str, typer.Option(help="Use 0.0.0.0 to open it from a phone on your Wi-Fi.")
    ] = "127.0.0.1",
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
    config: ConfigOpt = None,
) -> None:
    """Open the in-browser timing editor for a song."""
    from deepcuts.serve import serve as make_server

    cfg = _cfg(config)
    sd = _song(song, cfg)
    httpd = make_server(sd, cfg, host, port)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    out.print(f"Timing editor for [bold]{sd.name}[/] at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


@app.command("ls")
def list_songs(config: ConfigOpt = None) -> None:
    """List songs in your library."""
    cfg = _cfg(config)
    lib = cfg.library_path
    table = Table(show_edge=False, header_style="bold")
    for col in ("song", "lyrics", "stems", "aligned", "video"):
        table.add_column(col)
    tick = lambda ok: "[green]✓[/]" if ok else "[dim]–[/]"  # noqa: E731
    for d in sorted(p for p in lib.glob("*/") if (p / "meta.json").exists()):
        sd = SongDir(d)
        video = any(sd.out_dir.glob("*.mp4"))
        table.add_row(
            sd.name,
            tick(sd.lyrics_file()),
            tick(sd.stems_path.exists()),
            tick(sd.alignment_path.exists()),
            tick(video),
        )
    out.print(table if table.row_count else f"[dim]library is empty: {lib}[/]")


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check system dependencies, optional extras and hardware."""
    from deepcuts import audio

    cfg = _cfg(config)
    rows: list[tuple[str, bool, str]] = []
    rows.append(("python", sys.version_info >= (3, 11), sys.version.split()[0]))
    for binary in ("ffmpeg", "ffprobe"):
        try:
            rows.append((binary, True, audio.require(binary)))
        except DeepcutsError as exc:
            rows.append((binary, False, str(exc)))
    rows.append(("ffmpeg libass", audio.has_filter("ass"), "needed to burn karaoke subtitles"))
    extras = {
        "audio_separator": "separate",
        "ctc_forced_aligner": "align",
        "whisperx": "see README",
        "pyannote.audio": "diarize",
        "yt_dlp": "download",
        "syncedlyrics": "lyrics",
    }
    for module, extra in extras.items():
        found = importlib.util.find_spec(module.split(".")[0]) is not None
        how = "see README > WhisperX" if extra == "see README" else f"uv sync --extra {extra}"
        rows.append((module, found, "installed" if found else f"optional: {how}"))
    if importlib.util.find_spec("torch"):
        from deepcuts.stages.align import resolve_device

        dev = resolve_device("auto")
        rows.append(
            ("torch device", True, dev + ("" if dev != "cpu" else " (separation will be slow)"))
        )
    for name, env in (
        ("HF token", cfg.diarize.hf_token_env),
        ("MVSEP token", cfg.mvsep.api_token_env),
    ):
        rows.append(
            (
                name,
                bool(os.environ.get(env)),
                f"${env}" + ("" if os.environ.get(env) else " not set (optional)"),
            )
        )
    rows.append(("library", True, str(cfg.library_path)))
    table = Table(show_edge=False, show_header=False)
    for name, ok, detail in rows:
        table.add_row("[green]✓[/]" if ok else "[yellow]·[/]", name, f"[dim]{escape(detail)}[/]")
    out.print(table)


@app.command("mvsep-algorithms")
def mvsep_algorithms(
    query: Annotated[
        str, typer.Argument(help="Filter by name, e.g. 'karaoke' or 'roformer'.")
    ] = "",
) -> None:
    """List MVSEP sep_type ids and model options (for the mvsep config section)."""
    from deepcuts.stages.separate.mvsep import list_algorithms

    for algo in list_algorithms():
        name = str(algo.get("name", ""))
        if query.lower() not in name.lower():
            continue
        out.print(f"[bold]{algo.get('render_id')}[/]  {name}")
        for fld in algo.get("algorithm_fields") or []:
            opts = fld.get("options")
            if isinstance(opts, dict):
                preview = ", ".join(f"{k}={v}" for k, v in list(opts.items())[:12])
                out.print(f"     [dim]{fld.get('name')}: {preview}[/]")


def run() -> None:
    try:
        app()
    except DeepcutsError as exc:
        err.print(f"[red]✗[/] {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    run()
