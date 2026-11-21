"""Remote separation via the MVSEP API (https://mvsep.com/full_api).

MVSEP hosts the current leaderboard models (BS-RoFormer variants, ensembles, karaoke models)
that are not always released as public checkpoints. Needs MVSEP_API_TOKEN in the environment.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

from deepcuts.config import MvsepConfig
from deepcuts.errors import SeparationError

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a")


class MvsepBackend:
    name = "mvsep"

    def __init__(self, cfg: MvsepConfig) -> None:
        token = os.environ.get(cfg.api_token_env)
        if not token:
            raise SeparationError(
                f"set {cfg.api_token_env} (your token from mvsep.com) to use the mvsep backend"
            )
        self.cfg = cfg
        self.token = token
        self.http = httpx.Client(base_url=cfg.base_url, timeout=120)

    def split(self, model: str, src: Path, out_dir: Path, names: dict[str, str]) -> list[Path]:
        if model == "mvsep:karaoke":
            sep_type, opt1 = self.cfg.karaoke_sep_type, self.cfg.karaoke_add_opt1
        else:
            sep_type, opt1 = self.cfg.vocals_sep_type, self.cfg.vocals_add_opt1
        job = self._create(src, sep_type, opt1)
        data = self._wait(job)
        return self._download(data.get("files") or [], out_dir)

    def _create(self, src: Path, sep_type: int, add_opt1: str) -> str:
        with src.open("rb") as fh:
            r = self.http.post(
                "/separation/create",
                data={
                    "api_token": self.token,
                    "sep_type": str(sep_type),
                    "add_opt1": add_opt1,
                    "output_format": str(self.cfg.output_format),
                },
                files={"audiofile": (src.name, fh)},
            )
        body = r.json()
        if r.status_code != 200 or not body.get("success"):
            raise SeparationError(f"mvsep create failed: {body.get('data', body)}")
        return body["data"]["hash"]

    def _wait(self, job: str) -> dict:
        deadline = time.monotonic() + self.cfg.timeout_minutes * 60
        while time.monotonic() < deadline:
            r = self.http.get("/separation/get", params={"hash": job})
            body = r.json()
            status = body.get("status")
            if status == "done":
                return body.get("data", {})
            if status in ("failed", "not_found"):
                raise SeparationError(
                    f"mvsep job {job} {status}: {body.get('data', {}).get('message')}"
                )
            time.sleep(self.cfg.poll_seconds)
        raise SeparationError(f"mvsep job {job} timed out")

    def _download(self, files: list[dict], out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, entry in enumerate(files):
            url = entry.get("url") or entry.get("link")
            if not url:
                continue
            label = str(entry.get("type") or entry.get("name") or f"stem{i}").strip()
            ext = next((e for e in AUDIO_EXTS if url.split("?")[0].lower().endswith(e)), ".flac")
            dst = out_dir / f"{i:02d}_({label}){ext}"
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as resp:
                resp.raise_for_status()
                with dst.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
            paths.append(dst)
        if not paths:
            raise SeparationError(f"mvsep returned no downloadable files: {files!r}")
        return paths


def list_algorithms(base_url: str = "https://mvsep.com/api") -> list[dict]:
    r = httpx.get(f"{base_url}/app/algorithms", timeout=30)
    r.raise_for_status()
    return r.json()
