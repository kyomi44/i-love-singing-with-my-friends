"""Exceptions with messages meant to be shown to the user as-is."""

from __future__ import annotations


class DeepcutsError(Exception):
    """Base class: anything the CLI should print without a traceback."""


class MissingExtra(DeepcutsError):
    def __init__(self, extra: str, package: str, hint: str | None = None) -> None:
        how = hint or f"uv sync --extra {extra}     (or: pip install 'deepcuts[{extra}]')"
        super().__init__(f"{package} is not installed. Install it with:\n    {how}")


WHISPERX_HINT = "uv pip install whisperx   (it pins an older PyTorch; see README > WhisperX)"


class LyricsNotFound(DeepcutsError):
    pass


class SeparationError(DeepcutsError):
    pass


class AlignmentError(DeepcutsError):
    pass


class FFmpegError(DeepcutsError):
    pass
