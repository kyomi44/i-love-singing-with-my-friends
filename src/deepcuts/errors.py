"""Exceptions with messages meant to be shown to the user as-is."""

from __future__ import annotations


class DeepcutsError(Exception):
    """Base class: anything the CLI should print without a traceback."""


class MissingExtra(DeepcutsError):
    def __init__(
        self, extra: str, package: str, hint: str | None = None, cause: str | None = None
    ) -> None:
        self.extra, self.package, self.hint, self.cause = extra, package, hint, cause
        how = hint or f"uv sync --extra {extra}     (or: pip install 'deepcuts[{extra}]')"
        msg = f"{package} is not installed. Install it with:\n    {how}"
        if cause:  # the package may be there but one of *its* imports failed
            msg += f"\n    (import error: {cause})"
        super().__init__(msg)

    def __reduce__(self):  # picklable, so remote (Modal) failures reach the CLI intact
        return (type(self), (self.extra, self.package, self.hint, self.cause))


WHISPERX_HINT = "uv pip install whisperx   (it pins an older PyTorch; see README > WhisperX)"


class LyricsNotFound(DeepcutsError):
    pass


class SeparationError(DeepcutsError):
    pass


class AlignmentError(DeepcutsError):
    pass


class FFmpegError(DeepcutsError):
    pass
