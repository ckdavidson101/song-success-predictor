# backend/app/services/feature_builder.py
import math
from datetime import date
from typing import Any, Dict


def build_features(track: Dict[str, Any], artist: Dict[str, Any]) -> Dict[str, Any]:
    feats = {
        "track_popularity": track.get("popularity"),
        "duration_ms": track.get("duration_ms"),
        "explicit": int(bool(track.get("explicit"))),
        "release_date": track.get("album", {}).get("release_date"),
        "days_since_release": _days_since(track.get("album", {}).get("release_date")),
        "release_type_single": int(track.get("album", {}).get("album_type") == "single"),
        "artist_popularity": artist.get("popularity"),
        "artist_followers": artist.get("followers", {}).get("total"),
        "artist_followers_log": _log1p(artist.get("followers", {}).get("total")),
        "artist_genres": artist.get("genres", []),
    }
    return feats


def _log1p(x):
    return math.log1p(x or 0)


def _days_since(release_date: str | None) -> int:
    if not release_date:
        return 0
    parts = release_date.split("-")
    y = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 1
    d = int(parts[2]) if len(parts) > 2 else 1
    return (date.today() - date(y, m, d)).days
