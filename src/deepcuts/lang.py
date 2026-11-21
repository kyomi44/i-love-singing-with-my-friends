"""Language code helpers. MMS (ctc-forced-aligner) wants ISO 639-3; Whisper wants ISO 639-1."""

from __future__ import annotations

ISO3_TO_ISO1 = {
    "ara": "ar", "cat": "ca", "ces": "cs", "cmn": "zh", "dan": "da", "deu": "de", "ell": "el",
    "eng": "en", "fin": "fi", "fra": "fr", "heb": "he", "hin": "hi", "hun": "hu", "ita": "it",
    "jpn": "ja", "kor": "ko", "nld": "nl", "nob": "no", "nor": "no", "pol": "pl", "por": "pt",
    "ron": "ro", "rus": "ru", "spa": "es", "swe": "sv", "tur": "tr", "ukr": "uk", "vie": "vi",
    "zho": "zh", "isl": "is", "gle": "ga", "cym": "cy", "eus": "eu", "glg": "gl",
}  # fmt: skip


def to_iso1(code: str) -> str:
    return ISO3_TO_ISO1.get(code.lower(), code.lower())
