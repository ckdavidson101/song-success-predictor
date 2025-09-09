from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(title="Song Success API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/spotify_track")
def predict_spotify_track(uri: str = Query(..., description="spotify:track:... or https://open.spotify.com/track/...")):
    # TODO: fetch features from store/spotify, run model
    return {"track_uri": uri, "predicted_popularity": 63.2, "explanations": []}

@app.post("/translate/upload")
async def translate_upload(file: UploadFile = File(...)):
    # TODO: run DSP translator on uploaded audio → pseudo-Spotify features
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    # stub return
    return {"features": {"tempo": 124.0, "valence": 0.42}}

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    # TODO: translate → predict
    return {"predicted_popularity": 60.1}

@app.post("/simulate")
def simulate(uri: str, playlists_delta: int = 0, followers_delta: int = 0, trends_delta: float = 0.0):
    # TODO: apply tweaks and rescore
    return {"track_uri": uri, "delta": +3.4}
