"""`deepcuts serve`: a local timing editor in the browser.

Plays the karaoke track (optionally with the vocal stem as a guide) with the same two-row layout
as the video, and lets you fix timing by ear: global offset, per-line shifts, tap-to-sync.
Corrections are saved to overrides.json; `R` in the editor re-renders the video.
"""

from __future__ import annotations

import json
import shutil
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from deepcuts.config import Config
from deepcuts.models import Overrides
from deepcuts.stages.render import singer_lines
from deepcuts.workspace import SongDir


class EditorApp:
    def __init__(self, song: SongDir, cfg: Config) -> None:
        self.song = song
        self.cfg = cfg
        self.stems = song.load_stems()
        self.audio = {"track": self.stems.karaoke, "vocals": self.stems.vocals}
        self.render_lock = threading.Lock()

    def payload(self) -> dict:
        rc = self.cfg.render
        return {
            "meta": self.song.load_meta().model_dump(),
            "lines": [line.model_dump() for line in singer_lines(self.song, self.cfg)],
            "overrides": self.song.load_overrides().model_dump(),
            "audio": {k: f"/audio/{k}" for k, v in self.audio.items() if v and v.exists()},
            "render": {
                "lead_in": rc.lead_in,
                "hold": rc.hold,
                "sung_color": rc.sung_color,
                "unsung_color": rc.unsung_color,
                "singer_colors": rc.singer_colors,
                "highlight_lead_ms": rc.highlight_lead_ms,
                "countdown_gap": rc.countdown_gap,
            },
            "low_confidence": self.cfg.align.low_confidence,
        }

    def rerender(self) -> str:
        from deepcuts.pipeline import Pipeline

        with self.render_lock:
            outputs = Pipeline(self.cfg).render(self.song, self.stems)
        return str(outputs.get("video", outputs["ass"]))

    def handler(self) -> type[BaseHTTPRequestHandler]:
        app = self
        page = resources.files("deepcuts.web").joinpath("player.html").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the terminal quiet
                pass

            def _send(self, status: int, body: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj: object, status: int = 200) -> None:
                self._send(status, json.dumps(obj).encode(), "application/json")

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    return self._send(200, page, "text/html; charset=utf-8")
                if self.path == "/api/song":
                    return self._json(app.payload())
                if self.path.startswith("/audio/"):
                    path = app.audio.get(self.path.rsplit("/", 1)[-1])
                    if path and path.exists():
                        self.send_response(200)
                        self.send_header("Content-Type", "audio/wav")
                        self.send_header("Content-Length", str(path.stat().st_size))
                        self.end_headers()
                        with path.open("rb") as fh:
                            shutil.copyfileobj(fh, self.wfile)
                        return None
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b"{}"
                try:
                    if self.path == "/api/overrides":
                        app.song.save_overrides(Overrides.model_validate_json(body))
                        return self._json({"ok": True})
                    if self.path == "/api/render":
                        return self._json({"ok": True, "output": app.rerender()})
                except Exception as exc:  # report to the page instead of dropping the socket
                    return self._json({"ok": False, "error": str(exc)}, 500)
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        return Handler


def serve(
    song: SongDir, cfg: Config, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), EditorApp(song, cfg).handler())
