# backend/app/services/spotify_client.py
from __future__ import annotations

import re
import time
from typing import Any

import requests

from app.core.config import settings

TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"
_TRACK_RE = re.compile(r"(track/|spotify:track:)([A-Za-z0-9]{22})")


class SpotifyClient:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        self.client_id = client_id or settings.SPOTIFY_CLIENT_ID
        self.client_secret = client_secret or settings.SPOTIFY_CLIENT_SECRET
        if not self.client_id or not self.client_secret:
            raise RuntimeError("Missing SPOTIFY_CLIENT_ID/SECRET in environment")
        self._token: str | None = None
        self._exp = 0.0
        self._session = requests.Session()

    def _get_token(self) -> str:
        now = time.time()
        if self._token and now < self._exp - 30:
            return self._token
        assert self.client_id is not None and self.client_secret is not None
        r = self._session.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        self._exp = now + int(data.get("expires_in", 3600))
        assert self._token is not None  # Help MyPy understand this won't be None
        return self._token

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        tok = self._get_token()
        for _ in range(3):
            r = self._session.get(
                f"{API_BASE}{path}",
                headers={"Authorization": f"Bearer {tok}"},
                params=params or {},
                timeout=15,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()  # last try
        return {}

    @staticmethod
    def parse_track_id(value: str) -> str:
        m = _TRACK_RE.search(value)
        if m:
            return m.group(2)
        if len(value) == 22:
            return value
        raise ValueError("Invalid track identifier")

    def get_track(self, track_id_or_url: str) -> dict[str, Any]:
        tid = self.parse_track_id(track_id_or_url)
        return self._get(f"/tracks/{tid}")

    def get_audio_features(self, track_id_or_url: str) -> dict[str, Any]:
        tid = self.parse_track_id(track_id_or_url)
        return self._get(f"/audio-features/{tid}")

    def get_artist(self, artist_id_or_url: str) -> dict[str, Any]:
        aid = artist_id_or_url.split(":")[-1].split("/")[-1]
        return self._get(f"/artists/{aid}")

    def search_tracks(self, q: str, limit: int = 5) -> dict[str, Any]:
        return self._get("/search", params={"q": q, "type": "track", "limit": limit})
