"""Alternative aligner: WhisperX's wav2vec2 alignment, seeded with LRC line windows.

Needs synced lyrics (segments must have start/end). Handy to A/B against the CTC aligner:
    deepcuts align <song> --aligner whisperx -f align
"""

from __future__ import annotations

from pathlib import Path

from deepcuts.config import AlignConfig
from deepcuts.errors import WHISPERX_HINT, AlignmentError, MissingExtra
from deepcuts.lang import to_iso1
from deepcuts.models import Line
from deepcuts.stages.align.anchors import line_windows
from deepcuts.stages.align.matching import match_words

SAMPLE_RATE = 16000


class WhisperXAligner:
    name = "whisperx"

    def __init__(self, cfg: AlignConfig, device: str) -> None:
        try:
            import whisperx
        except ImportError as exc:
            raise MissingExtra("whisperx", "whisperx", WHISPERX_HINT) from exc
        self.wx = whisperx
        self.cfg = cfg
        self.device = "cpu" if device == "mps" else device

    def align(
        self, audio: Path, lines: list[Line], language: str, strategy: str
    ) -> tuple[list[Line], float | None]:
        if not all(line.anchor is not None for line in lines):
            raise AlignmentError(
                "the whisperx aligner needs synced (LRC) lyrics; use --aligner ctc for plain text"
            )
        out = [line.model_copy(deep=True) for line in lines]
        model, metadata = self.wx.load_align_model(
            language_code=to_iso1(language), device=self.device
        )
        samples = self.wx.load_audio(str(audio))
        duration = len(samples) / SAMPLE_RATE
        windows = line_windows(
            [line.anchor for line in out],  # type: ignore[misc]
            [line.anchor_end for line in out],
            duration,
            self.cfg.window_pad_before,
            self.cfg.window_pad_after,
        )
        segments = [
            {"text": line.text, "start": ws, "end": we}
            for line, (ws, we) in zip(out, windows, strict=True)
        ]
        result = self.wx.align(
            segments, model, metadata, samples, self.device, return_char_alignments=False
        )
        got = result.get("word_segments") or [
            w for seg in result.get("segments", []) for w in seg.get("words", [])
        ]
        flat = [w for line in out for w in line.words]
        mapping = match_words([w.text for w in flat], [str(g.get("word", "")) for g in got])
        for w, gi in zip(flat, mapping, strict=True):
            if gi is None or "start" not in got[gi] or "end" not in got[gi]:
                continue
            w.start, w.end = float(got[gi]["start"]), float(got[gi]["end"])
            w.confidence = float(got[gi].get("score", 0.5))
        return out, None
