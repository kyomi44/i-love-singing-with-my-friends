"""Who sings which line, for duets and call-and-response (experimental).

pyannote's community-1 pipeline is trained on *speech*; on separated singing it works reasonably
for clearly different voices (e.g. male/female duets) and poorly for similar voices or stacked
harmonies. Manual `@A` / `@B` / `@AB` tags in the lyrics file always win over diarization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from deepcuts.config import DiarizeConfig
from deepcuts.errors import DeepcutsError, MissingExtra
from deepcuts.models import Line

Turn = tuple[float, float, str]


def run_diarization(audio: Path, cfg: DiarizeConfig, device: str) -> list[Turn]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise MissingExtra("diarize", "pyannote.audio") from exc
    token = os.environ.get(cfg.hf_token_env)
    if not token:
        raise DeepcutsError(
            f"set {cfg.hf_token_env} to a Hugging Face token and accept the model terms at "
            f"https://hf.co/{cfg.model}"
        )
    pipeline = Pipeline.from_pretrained(cfg.model, token=token)
    if device != "cpu":
        import torch

        pipeline.to(torch.device(device))
    kwargs = {"num_speakers": cfg.num_speakers} if cfg.num_speakers else {}
    output = pipeline(str(audio), **kwargs)
    ann = getattr(output, "exclusive_speaker_diarization", None) or getattr(
        output, "speaker_diarization", output
    )
    if hasattr(ann, "itertracks"):
        return [(t.start, t.end, str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]
    return [(t.start, t.end, str(spk)) for t, spk in ann]


def save_turns(path: Path, turns: list[Turn]) -> None:
    path.write_text(
        json.dumps([{"start": s, "end": e, "speaker": k} for s, e, k in turns], indent=1)
    )


def load_turns(path: Path) -> list[Turn]:
    return [(d["start"], d["end"], d["speaker"]) for d in json.loads(path.read_text())]


def assign_singers(lines: list[Line], turns: list[Turn], both_ratio: float = 0.35) -> list[Line]:
    order: list[str] = []
    for _, _, spk in sorted(turns):
        if spk not in order:
            order.append(spk)
    names = {spk: "AB"[i] for i, spk in enumerate(order[:2])}
    out: list[Line] = []
    for line in lines:
        line = line.model_copy(deep=True)
        if line.singer is None and line.start is not None and line.end is not None:
            dur = max(1e-3, line.end - line.start)
            overlap: dict[str, float] = {}
            for s, e, spk in turns:
                o = min(e, line.end) - max(s, line.start)
                if o > 0 and spk in names:
                    overlap[names[spk]] = overlap.get(names[spk], 0.0) + o
            ranked = sorted(overlap.items(), key=lambda kv: -kv[1])
            if len(ranked) > 1 and ranked[1][1] / dur >= both_ratio:
                line.singer = "AB"
            elif ranked:
                line.singer = ranked[0][0]
        out.append(line)
    return out
