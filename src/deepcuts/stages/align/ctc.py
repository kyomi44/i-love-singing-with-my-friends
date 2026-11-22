"""Default aligner: CTC forced alignment of the *known* lyrics against the vocal stem.

Uses ctc-forced-aligner (MMS-300M fine-tuned for forced alignment, ~1100 languages) with its own
memory-efficient alignment kernel, so it doesn't depend on torchaudio's deprecated forced_align.

Why this instead of transcribe-then-match: we already know the words. Forced alignment only has to
answer "when", which is a far easier problem than "what", and it never invents lyrics.

Emissions are computed once for the whole track; global and per-line passes just slice them.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from deepcuts.config import AlignConfig
from deepcuts.errors import MissingExtra
from deepcuts.models import Line, Word
from deepcuts.stages.align.activity import active_frames, clip_at_next_onset, fit_to_activity
from deepcuts.stages.align.anchors import estimate_offset, line_windows, pick_best
from deepcuts.stages.align.matching import match_words
from deepcuts.text import is_alignable

log = logging.getLogger(__name__)


def confidence(score: float | None) -> float | None:
    """ctc-forced-aligner reports each word's mean per-frame log-prob; map it to 0..1."""
    if score is None:
        return None
    score = float(score)
    return max(0.0, min(1.0, math.exp(score))) if score <= 0 else min(1.0, score)


class CtcAligner:
    name = "ctc"

    def __init__(self, cfg: AlignConfig, device: str) -> None:
        try:
            import ctc_forced_aligner as cfa
            import torch
        except ImportError as exc:
            raise MissingExtra("align", "ctc-forced-aligner", cause=str(exc)) from exc
        self._cfa = cfa
        self._torch = torch
        self.cfg = cfg
        kwargs: dict = {"dtype": torch.float16 if device == "cuda" else torch.float32}
        if cfg.ctc_model:
            kwargs["model_path"] = cfg.ctc_model
        self.model, self.tokenizer = cfa.load_alignment_model(device, **kwargs)
        self.stride_ms = 20.0

    # ---- public ------------------------------------------------------------------------
    def align(
        self, audio: Path, lines: list[Line], language: str, strategy: str
    ) -> tuple[list[Line], float | None]:
        cfa = self._cfa
        wav = cfa.load_audio(str(audio), self.model.dtype, self.model.device)
        with self._torch.inference_mode():
            emissions, stride = cfa.generate_emissions(
                self.model, wav, batch_size=self.cfg.batch_size
            )
        self.stride_ms = float(stride)
        duration = emissions.shape[0] * self.stride_ms / 1000.0
        # Vocal activity keeps words where somebody is actually singing (see activity.py):
        # masked emissions, windows that stop at the next line's onset, spans fitted afterwards.
        self._active = self._activity(wav, int(emissions.shape[0]))
        if self._active is not None:
            emissions = self._mask_silence(emissions, self._active)
        lines, offset = self._align_all(emissions, lines, language, strategy, duration)
        if self._active is not None:
            fit_to_activity(lines, self._active, self.stride_ms / 1000.0)
        return lines, offset

    def _align_all(
        self, emissions, lines: list[Line], language: str, strategy: str, duration: float
    ) -> tuple[list[Line], float | None]:
        has_anchors = all(line.anchor is not None for line in lines)

        if strategy == "anchored" and has_anchors:
            return self._anchored(emissions, lines, language, 0.0, duration), 0.0

        glob = [line.model_copy(deep=True) for line in lines]
        self._align_words(
            emissions, [w for line in glob for w in line.words], 0.0, duration, language, "segment"
        )
        if strategy == "global" or not has_anchors:
            return glob, None

        offset = estimate_offset(glob)
        log.info("LRC anchors are %+.2fs off this audio; correcting before refinement", offset)
        anchored = self._anchored(emissions, lines, language, offset, duration)
        return pick_best(glob, anchored), offset

    # ---- internals ---------------------------------------------------------------------
    def _anchored(
        self, emissions, lines: list[Line], language: str, offset: float, duration: float
    ) -> list[Line]:
        out = [line.model_copy(deep=True) for line in lines]
        for line in out:
            line.anchor = line.anchor + offset  # type: ignore[operator]
            if line.anchor_end is not None:
                line.anchor_end += offset
        anchors = [line.anchor for line in out]
        windows = line_windows(
            anchors,  # type: ignore[arg-type]
            [line.anchor_end for line in out],
            duration,
            self.cfg.window_pad_before,
            self.cfg.window_pad_after,
        )
        if self._active is not None:
            windows = clip_at_next_onset(
                windows,
                anchors,
                self._active,
                self.stride_ms / 1000.0,  # type: ignore[arg-type]
            )
        for line, (ws, we) in zip(out, windows, strict=True):
            if not self._align_words(emissions, line.words, ws, we, language, "edges"):
                log.debug("line window alignment failed, will interpolate: %r", line.text)
        return out

    def _activity(self, wav, frames: int) -> list[bool] | None:
        """Per-emission-frame vocal activity of the stem (see activity.py), or None if disabled."""
        if self.cfg.silence_rel_db is None:
            return None
        torch = self._torch
        hop = round(16000 * self.stride_ms / 1000.0)  # MMS runs at 16 kHz
        x = wav.flatten().float()
        x = torch.nn.functional.pad(x, (0, max(0, frames * hop - x.numel())))[: frames * hop]
        levels = (10 * torch.log10(x.view(frames, hop).pow(2).mean(dim=1) + 1e-10)).tolist()
        hangover = round(self.cfg.silence_hangover * 1000.0 / self.stride_ms)
        active = active_frames(levels, self.cfg.silence_rel_db, hangover=hangover)
        log.debug("vocal activity: %d of %d frames active", sum(active), frames)
        if not any(active):
            return None
        return active

    def _mask_silence(self, emissions, active: list[bool]):
        """Only blank/<star> may occupy frames where nobody is singing."""
        torch = self._torch
        silent = torch.tensor([not a for a in active], device=emissions.device)
        vocab = self.tokenizer.get_vocab()
        blank = vocab.get("<blank>", self.tokenizer.pad_token_id)
        cols = torch.ones(emissions.shape[1], dtype=torch.bool, device=emissions.device)
        cols[blank] = False
        cols[-1] = False  # the <star> column ctc-forced-aligner appends
        out = emissions.masked_fill(silent[:, None] & cols[None, :], -1e4)
        out[silent, blank] = 0.0
        return out

    def _preprocess(self, text: str, language: str, star: str):
        try:
            return self._cfa.preprocess_text(
                text, romanize=True, language=language, star_frequency=star
            )
        except TypeError:  # older releases have no star_frequency
            return self._cfa.preprocess_text(text, romanize=True, language=language)

    def _align_words(
        self, emissions, words: list[Word], t0: float, t1: float, language: str, star: str
    ) -> bool:
        cfa = self._cfa
        targets = [w for w in words if is_alignable(w.text)]
        if not targets:
            return True
        fps = 1000.0 / self.stride_ms
        a = max(0, int(t0 * fps))
        b = min(int(emissions.shape[0]), int(math.ceil(t1 * fps)) + 1)
        if b - a < 2:
            return False
        try:
            tokens, text_starred = self._preprocess(
                " ".join(w.text for w in targets), language, star
            )
            segments, scores, blank = cfa.get_alignments(emissions[a:b], tokens, self.tokenizer)
            spans = cfa.get_spans(tokens, segments, blank)
            results = cfa.postprocess_results(text_starred, spans, self.stride_ms, scores)
        except Exception as exc:  # too few frames for the tokens, empty romanization, ...
            log.debug("alignment chunk failed: %s", exc)
            return False
        offset = a / fps
        mapping = match_words([w.text for w in targets], [str(r.get("text", "")) for r in results])
        for w, ri in zip(targets, mapping, strict=True):
            if ri is None:
                continue
            r = results[ri]
            w.start = offset + float(r["start"])
            w.end = offset + float(r["end"])
            w.confidence = confidence(r.get("score"))
        return True
