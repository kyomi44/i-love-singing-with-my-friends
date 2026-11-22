"""Orchestrates the stages with caching: unchanged stages are skipped on re-runs.

    ingest -> lyrics -> separate -> (asr fallback) -> align -> (diarize) -> render

Lyrics are looked up *before* the expensive separation so a missing-lyrics failure is instant.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from deepcuts.audio import require
from deepcuts.config import Config
from deepcuts.errors import LyricsNotFound
from deepcuts.models import Alignment, Lyrics, StemSet
from deepcuts.stages import ingest as ingest_stage
from deepcuts.stages.align import alignment_stem, resolve_device, run_alignment
from deepcuts.stages.align.postprocess import low_confidence
from deepcuts.stages.lyrics import fetch_lyrics, load_lyrics_file
from deepcuts.workspace import SongDir, fingerprint

STAGES = ("ingest", "lyrics", "separate", "align", "diarize", "render")


@dataclass
class MakeResult:
    song: SongDir
    outputs: dict[str, Path]
    alignment: Alignment
    flagged: list[tuple[int, float]] = field(default_factory=list)


class Pipeline:
    def __init__(self, cfg: Config, force: Iterable[str] = (), console: Console | None = None):
        self.cfg = cfg
        self.force = {f.strip().lower() for f in force}
        self.console = console or Console(stderr=True)

    def forced(self, stage: str) -> bool:
        return "all" in self.force or stage in self.force

    def _step(self, msg: str):
        return self.console.status(f"[bold]{msg}…")

    def _done(self, stage: str, detail: str = "", cached: bool = False) -> None:
        mark = "[dim]·[/] " if cached else "[green]✓[/] "
        note = " [dim](cached)[/]" if cached else ""
        self.console.print(f"{mark}{stage:<9} {detail}{note}")

    # ---- stages ------------------------------------------------------------------------
    def ingest(
        self,
        source: str,
        artist: str | None = None,
        title: str | None = None,
        album: str | None = None,
    ) -> SongDir:
        with self._step("Ingesting"):
            song = ingest_stage.ingest(
                source, self.cfg, artist, title, album, force=self.forced("ingest")
            )
        meta = song.load_meta()
        self._done("ingest", f"{meta.artist} — {meta.title} ({meta.duration:.0f}s)")
        return song

    def lyrics(self, song: SongDir, lyrics_file: Path | None = None) -> Lyrics | None:
        if lyrics_file is not None:
            kind = "lrc" if lyrics_file.suffix.lower() == ".lrc" else "txt"
            song.write_lyrics(
                lyrics_file.read_text(encoding="utf-8"), kind, f"file:{lyrics_file.name}"
            )
        existing = song.lyrics_file()
        if existing and not self.forced("lyrics"):
            lyr = load_lyrics_file(existing, song.lyrics_source())
            self._done(
                "lyrics", f"{len(lyr.lines)} lines from {lyr.source}", cached=lyrics_file is None
            )
            return lyr
        with self._step("Finding lyrics"):
            try:
                content, kind, source = fetch_lyrics(song.load_meta(), self.cfg)
            except LyricsNotFound as exc:
                self.console.print(f"[yellow]![/] lyrics    {exc}")
                return None
        path = song.write_lyrics(content, kind, source)
        lyr = load_lyrics_file(path, source)
        self._done(
            "lyrics", f"{len(lyr.lines)} lines from {source}{' (synced)' if lyr.synced else ''}"
        )
        return lyr

    def separate(self, song: SongDir) -> StemSet:
        params = {
            "sep": self.cfg.separation.model_dump(mode="json"),
            "align_stem": self.cfg.align.stem,
        }
        if self.cfg.separation.backend == "mvsep":
            params["mvsep"] = self.cfg.mvsep.model_dump(mode="json")
        fp = fingerprint([song.audio], params)
        if (
            not self.forced("separate")
            and song.is_fresh("separate", fp)
            and song.stems_path.exists()
        ):
            stems = song.load_stems()
            self._done("separate", ", ".join(stems.models), cached=True)
            return stems
        from deepcuts.stages.separate import run_separation

        with self._step("Separating stems (this is the slow part)"):
            stems = run_separation(song, self.cfg)
        song.save_stems(stems)
        song.mark("separate", fp)
        self._done("separate", ", ".join(stems.models))
        return stems

    def asr(self, song: SongDir, stems: StemSet) -> Lyrics:
        from deepcuts.stages.lyrics.asr import transcribe

        meta = song.load_meta()
        with self._step("Transcribing vocals (no lyrics found anywhere)"):
            text = transcribe(
                stems.vocals,
                meta.language,
                resolve_device(self.cfg.align.device),
                self.cfg.lyrics.asr_model,
            )
        path = song.write_lyrics(text, "txt", "asr:whisperx")
        lyr = load_lyrics_file(path, "asr:whisperx")
        self._done("lyrics", f"{len(lyr.lines)} lines transcribed — please proof-read {path.name}")
        return lyr

    def align(self, song: SongDir, lyrics: Lyrics, stems: StemSet) -> Alignment:
        meta = song.load_meta()
        stem = alignment_stem(stems, self.cfg.align)
        params = {"align": self.cfg.align.model_dump(mode="json"), "language": meta.language}
        fp = fingerprint([lyrics.path, stem], params)
        if not self.forced("align") and song.is_fresh("align", fp) and song.alignment_path.exists():
            aln = song.load_alignment()
            self._done("align", f"{aln.aligner}/{aln.strategy}", cached=True)
            return aln
        with self._step(f"Aligning lyrics to {stem.name} ({self.cfg.align.aligner})"):
            aln = run_alignment(meta, lyrics, stems, self.cfg)
        song.save_alignment(aln)
        song.mark("align", fp)
        detail = f"{aln.aligner}/{aln.strategy}"
        if aln.lrc_offset:
            detail += f", LRC was {aln.lrc_offset:+.2f}s off your audio (corrected)"
        self._done("align", detail)
        return aln

    def diarize(self, song: SongDir, stems: StemSet) -> None:
        if not self.cfg.diarize.enabled:
            return
        from deepcuts.stages.diarize import run_diarization, save_turns

        fp = fingerprint([stems.vocals], self.cfg.diarize.model_dump(mode="json"))
        if (
            not self.forced("diarize")
            and song.is_fresh("diarize", fp)
            and song.diarization_path.exists()
        ):
            self._done("diarize", "", cached=True)
            return
        with self._step("Diarizing singers"):
            turns = run_diarization(
                stems.vocals, self.cfg.diarize, resolve_device(self.cfg.align.device)
            )
        save_turns(song.diarization_path, turns)
        song.mark("diarize", fp)
        self._done("diarize", f"{len({t[2] for t in turns})} voices, {len(turns)} turns")

    def render(self, song: SongDir, stems: StemSet | None = None) -> dict[str, Path]:
        from deepcuts.stages.cover import ensure_cover
        from deepcuts.stages.render import output_paths, run_render

        stems = stems or song.load_stems()
        rc = self.cfg.render
        cover = (
            ensure_cover(song)
            if rc.video and (rc.cover_intro or rc.background == "cover")
            else None
        )
        inputs = [
            song.alignment_path,
            song.overrides_path,
            song.diarization_path,
            stems.karaoke,
            cover,
        ]
        if self.cfg.render.guide_vocals > 0:
            inputs.append(stems.vocals)
        params = {
            "render": self.cfg.render.model_dump(mode="json"),
            "diarize": self.cfg.diarize.enabled,
        }
        fp = fingerprint(inputs, params)
        expected = output_paths(song)
        if not self.cfg.render.video:
            expected.pop("video")
        if (
            not self.forced("render")
            and song.is_fresh("render", fp)
            and all(p.exists() for p in expected.values())
        ):
            self._done("render", str(expected.get("video", expected["ass"]).name), cached=True)
            return expected
        if self.cfg.render.video:
            require("ffmpeg")
        with self._step(
            "Rendering karaoke video" if self.cfg.render.video else "Writing subtitles"
        ):
            outputs = run_render(song, stems, self.cfg)
        song.mark("render", fp)
        self._done("render", str(outputs.get("video", outputs["ass"]).name))
        return outputs

    # ---- everything --------------------------------------------------------------------
    def make(
        self,
        source: str,
        artist: str | None = None,
        title: str | None = None,
        album: str | None = None,
        lyrics_file: Path | None = None,
        cover_file: Path | None = None,
    ) -> MakeResult:
        song = self.ingest(source, artist, title, album)
        if cover_file is not None:
            from deepcuts.stages.cover import import_cover

            import_cover(song, cover_file)
        lyrics = self.lyrics(song, lyrics_file)
        if lyrics is None and not self.cfg.lyrics.asr_fallback:
            raise LyricsNotFound(
                f"no lyrics for {song.name}. Drop a lyrics.txt / lyrics.lrc into {song.root} "
                "(or pass --lyrics FILE, or use --asr) and run the same command again."
            )
        stems = self.separate(song)
        if lyrics is None:
            lyrics = self.asr(song, stems)
        aln = self.align(song, lyrics, stems)
        self.diarize(song, stems)
        outputs = self.render(song, stems)
        flagged = low_confidence(
            aln.lines, self.cfg.align.low_confidence, self.cfg.align.low_word_confidence
        )
        return MakeResult(song=song, outputs=outputs, alignment=aln, flagged=flagged)
