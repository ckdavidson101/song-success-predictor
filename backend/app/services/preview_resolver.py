# backend/app/services/preview_resolver.py
from typing import Optional, Tuple

import requests


class PreviewResolver:
    DEEZER_API = "https://api.deezer.com"

    @staticmethod
    def from_spotify(track: dict) -> Tuple[str, Optional[str]]:
        """Return preview_url if Spotify provides one."""
        url = track.get("preview_url")
        if url:
            return "spotify", url
        return "spotify", None

    @classmethod
    def from_deezer(cls, track: dict, artist: dict) -> Tuple[str, Optional[str]]:
        """Try Deezer API with ISRC, then fallback to artist+title."""
        isrc = track.get("external_ids", {}).get("isrc")
        if isrc:
            r = requests.get(f"{cls.DEEZER_API}/track/isrc:{isrc}", timeout=10)
            if r.ok:
                data = r.json()
                if "preview" in data and data["preview"]:
                    return "deezer_isrc", data["preview"]

        # fallback: search by title+artist
        q = f'artist:"{artist["name"]}" track:"{track["name"]}"'
        r = requests.get(f"{cls.DEEZER_API}/search", params={"q": q}, timeout=10)
        if r.ok:
            data = r.json()
            if data.get("data"):
                candidate = data["data"][0]
                if candidate.get("preview"):
                    return "deezer_search", candidate["preview"]

        return "deezer", None

    @classmethod
    def resolve(cls, track: dict, artist: dict) -> dict:
        """Return a dict with best preview source."""
        src, url = cls.from_spotify(track)
        if url:
            return {"source": src, "preview_url": url}

        src, url = cls.from_deezer(track, artist)
        return {"source": src, "preview_url": url}
