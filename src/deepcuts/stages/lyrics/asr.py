"""Last resort: transcribe the vocal stem when no lyrics exist anywhere (very deep cuts).

Whisper on sung vocals is decent but not perfect, so the output is written as lyrics.txt with a
header asking you to proof-read it. Alignment then runs on the corrected text as usual.
"""

from __future__ import annotations

from pathlib import Path

from deepcuts.errors import WHISPERX_HINT, MissingExtra
from deepcuts.lang import to_iso1

HEADER = (
    "# Auto-transcribed by deepcuts (WhisperX). Fix mistakes, add [Section] headers or @A/@B\n"
    "# duet tags if you like, then re-run:  deepcuts align <song> && deepcuts render <song>\n"
)


def transcribe(vocals: Path, language: str, device: str, model_name: str = "large-v3") -> str:
    try:
        import whisperx
    except ImportError as exc:
        raise MissingExtra("whisperx", "whisperx", WHISPERX_HINT) from exc
    if device == "mps":  # faster-whisper runs on CPU or CUDA only
        device = "cpu"
    model = whisperx.load_model(
        model_name,
        device,
        compute_type="float16" if device == "cuda" else "int8",
        language=to_iso1(language),
    )
    audio = whisperx.load_audio(str(vocals))
    result = model.transcribe(audio, batch_size=8)
    lines = [
        seg["text"].strip() for seg in result.get("segments", []) if seg.get("text", "").strip()
    ]
    return HEADER + "\n".join(lines) + "\n"
