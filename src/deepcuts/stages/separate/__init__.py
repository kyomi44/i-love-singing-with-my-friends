"""Separation stage.

Two passes, so we get every stem karaoke needs:

    mix ──(vocal model)──► vocals + instrumental
    vocals ──(karaoke model)──► lead + backing

    karaoke track = instrumental + backing   (backing="keep", default: harmonies stay in)
                  = instrumental              (backing="remove")

The lead/backing split also gives the aligner a cleaner signal when align.stem = "lead".
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Protocol

from deepcuts import audio
from deepcuts.config import Config
from deepcuts.errors import SeparationError
from deepcuts.models import StemSet
from deepcuts.workspace import SongDir

LABEL_RE = re.compile(r"\(([^()]+)\)")


class Backend(Protocol):
    name: str

    def split(self, model: str, src: Path, out_dir: Path, names: dict[str, str]) -> list[Path]:
        """Run one 2-stem separation; return the output files (any names, any audio format)."""


def stem_label(path: Path) -> str:
    """'song_(Vocals)_model' -> 'vocals'; 'lead' -> 'lead'."""
    labels = LABEL_RE.findall(path.stem)
    return (labels[-1] if labels else path.stem).strip().lower()


def pick(paths: list[Path], include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> Path | None:
    for p in paths:
        label = stem_label(p)
        if any(k in label for k in include) and not any(k in label for k in exclude):
            return p
    return None


def classify_pair(
    paths: list[Path],
    primary: tuple[str, ...],
    exclude: tuple[str, ...],
    secondary: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    """Split a separation result into (primary, other). Extra outputs are ignored."""
    if len(paths) < 2:
        raise SeparationError(f"expected 2 stems, got {[p.name for p in paths]}")
    first = pick(paths, primary, exclude)
    if first is None:
        raise SeparationError(f"couldn't tell which output is which: {[p.name for p in paths]}")
    rest = [p for p in paths if p != first]
    other = (pick(rest, secondary) if secondary else None) or rest[0]
    return first, other


def _canonical(src: Path, dst: Path) -> Path:
    if src.suffix.lower() == ".wav":
        shutil.move(str(src), dst)
    else:
        audio.to_wav(src, dst)
        src.unlink(missing_ok=True)
    return dst


def get_backend(cfg: Config) -> Backend:
    if cfg.separation.backend == "mvsep":
        from deepcuts.stages.separate.mvsep import MvsepBackend

        return MvsepBackend(cfg.mvsep)
    from deepcuts.stages.separate.local import LocalBackend

    return LocalBackend(cfg.separation)


def run_separation(song: SongDir, cfg: Config) -> StemSet:
    sc = cfg.separation
    backend = get_backend(cfg)
    stems = song.stems_dir
    raw = stems / "_raw"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)

    vocal_model = sc.vocal_model if backend.name == "local" else "mvsep:vocals"
    outs = backend.split(
        vocal_model,
        song.audio,
        raw / "vocals",
        {"Vocals": "vocals", "Instrumental": "instrumental", "Other": "instrumental"},
    )
    v, inst = classify_pair(
        outs,
        ("vocal",),
        ("instrum", "no_vocal", "no vocal", "karaoke", "other"),
        ("instrum", "no_vocal", "other"),
    )
    vocals = _canonical(v, stems / "vocals.wav")
    instrumental = _canonical(inst, stems / "instrumental.wav")
    models = [vocal_model]

    lead = backing = None
    need_split = sc.backing == "keep" or cfg.align.stem == "lead"
    if need_split:
        outs, used = _split_lead_backing(backend, sc.karaoke_models, vocals, raw / "karaoke")
        ld, bk = classify_pair(
            outs, ("lead", "vocal"), ("back", "instrum", "other", "karaoke"), ("back", "instrum")
        )
        lead = _canonical(ld, stems / "lead.wav")
        backing = _canonical(bk, stems / "backing.wav")
        models.append(used)

    if sc.backing == "keep" and backing is not None:
        karaoke = audio.mix([instrumental, backing], stems / "karaoke.wav")
    else:
        karaoke = stems / "karaoke.wav"
        shutil.copy2(instrumental, karaoke)

    shutil.rmtree(raw, ignore_errors=True)
    return StemSet(
        vocals=vocals, instrumental=instrumental, karaoke=karaoke,
        lead=lead, backing=backing, backend=backend.name, models=models,
    )  # fmt: skip


def _split_lead_backing(
    backend: Backend, models: list[str], vocals: Path, out_dir: Path
) -> tuple[list[Path], str]:
    if backend.name != "local":
        return backend.split("mvsep:karaoke", vocals, out_dir, {}), "mvsep:karaoke"
    errors: list[str] = []
    for model in models:
        try:
            return backend.split(
                model,
                vocals,
                out_dir,
                {"Vocals": "lead", "Instrumental": "backing", "Other": "backing"},
            ), model
        except Exception as exc:  # unknown/renamed model, download failure... try the next one
            errors.append(f"{model}: {exc}")
            shutil.rmtree(out_dir, ignore_errors=True)
    raise SeparationError("no karaoke (lead/backing) model worked:\n  " + "\n  ".join(errors))
