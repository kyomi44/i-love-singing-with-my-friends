"""Minimal client for LRCLIB (https://lrclib.net): free, keyless, community synced lyrics."""

from __future__ import annotations

import httpx

from deepcuts import __version__

USER_AGENT = f"deepcuts/{__version__} (https://github.com/your-name/deepcuts)"


class LrclibClient:
    def __init__(self, base_url: str = "https://lrclib.net/api", timeout: float = 15.0) -> None:
        self._http = httpx.Client(
            base_url=base_url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )

    def get(
        self, artist: str, title: str, album: str | None = None, duration: float | None = None
    ) -> dict | None:
        params: dict[str, str | int] = {"artist_name": artist, "track_name": title}
        if album:
            params["album_name"] = album
        if duration:
            params["duration"] = round(duration)
        r = self._http.get("/get", params=params)
        if r.status_code in (400, 404):
            return None
        r.raise_for_status()
        return r.json()

    def search(self, artist: str, title: str) -> list[dict]:
        r = self._http.get("/search", params={"artist_name": artist, "track_name": title})
        r.raise_for_status()
        return r.json()

    def best_match(
        self,
        artist: str,
        title: str,
        album: str | None = None,
        duration: float | None = None,
        tolerance: float = 3.0,
    ) -> dict | None:
        """Exact signature match first; otherwise the closest-duration search hit, synced preferred."""
        hit = self.get(artist, title, album, duration)
        if hit and (hit.get("syncedLyrics") or hit.get("plainLyrics") or hit.get("instrumental")):
            return hit
        candidates = [
            c for c in self.search(artist, title) if c.get("syncedLyrics") or c.get("plainLyrics")
        ]
        return pick_candidate(candidates, duration, tolerance)


def pick_candidate(candidates: list[dict], duration: float | None, tolerance: float) -> dict | None:
    if not candidates:
        return None

    def off(c: dict) -> float:
        if duration is None or c.get("duration") is None:
            return 0.0
        return abs(float(c["duration"]) - duration)

    in_range = [c for c in candidates if off(c) <= tolerance] if duration else candidates
    pool = in_range or candidates
    # synced beats plain; then closest duration
    return sorted(pool, key=lambda c: (not c.get("syncedLyrics"), off(c)))[0]
