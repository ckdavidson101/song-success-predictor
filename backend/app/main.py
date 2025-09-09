from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.services.feature_builder import build_features
from app.services.spotify_client import SpotifyClient

app = FastAPI(title="Song Success API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sp = SpotifyClient()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(q: str):
    return _sp.search_tracks(q)


@app.post("/predict/spotify_track")
def predict_spotify_track(uri: str = Query(..., description="Spotify track URL/URI/ID")):
    try:
        t = _sp.get_track(uri)
        a = _sp.get_artist(t["artists"][0]["id"])
        feats = build_features(t, a)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch/assemble features: {e}"
        ) from e
    # MODEL PLACEHOLDER: return context features + stubbed score
    return {
        "track_uri": uri,
        "track_name": t.get("name"),
        "artist_name": t["artists"][0]["name"],
        "context_features": feats,
        "predicted_popularity": 63.2,
    }


@app.post("/translate/upload")
async def translate_upload(file: Annotated[UploadFile, File()]):
    # TODO: run DSP translator on uploaded audio → pseudo-Spotify features
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    # stub return
    return {"features": {"tempo": 124.0, "valence": 0.42}}


@app.post("/predict/upload")
async def predict_upload(file: Annotated[UploadFile, File()]):
    # TODO: translate → predict
    return {"predicted_popularity": 60.1}


@app.post("/simulate")
def simulate(
    uri: str, playlists_delta: int = 0, followers_delta: int = 0, trends_delta: float = 0.0
):
    # TODO: apply tweaks and rescore
    return {"track_uri": uri, "delta": +3.4}
