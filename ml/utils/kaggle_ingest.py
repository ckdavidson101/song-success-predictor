# ml/utils/kaggle_ingest.py
from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

# Columns you plan to use downstream (adjust names to the dataset you chose)
FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",
    "loudness",
    "key",
    "mode",
    "time_signature",
]
META_COLS = [
    "track_id",
    "track_uri",
    "isrc",
    "track_name",
    "artist_name",
    "album_release_date",
    "duration_ms",
]


def read_any_csv_dir(raw_dir: Path) -> pd.DataFrame:
    csvs = list(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")
    dfs = [pd.read_csv(p) for p in csvs]
    return pd.concat(dfs, ignore_index=True, sort=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/kaggle/spotify_1m", type=str)
    ap.add_argument("--out", default="data/interim/spotify_features.parquet", type=str)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = read_any_csv_dir(raw_dir)

    # Try to standardize column names (adapt to your dataset’s exact headers)
    rename_map = {
        "id": "track_id",
        "uri": "track_uri",
        "name": "track_name",
        "artists": "artist_name",
        "release_date": "album_release_date",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Keep only columns we need (ignore missing silently)
    keep = [c for c in META_COLS + FEATURE_COLS if c in df.columns]
    df = df[keep].copy()

    # Basic cleaning
    df = df.drop_duplicates(
        subset=[c for c in ["track_id", "track_uri", "isrc"] if c in df.columns]
    )
    if "album_release_date" in df.columns:
        df["album_release_date"] = pd.to_datetime(
            df["album_release_date"], errors="coerce"
        )

    # Minimal sanity filters (optional)
    if "duration_ms" in df.columns:
        df = df[
            (df["duration_ms"] > 10000) & (df["duration_ms"] < 15 * 60 * 1000)
        ]  # 10s..15min

    df.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(df):,} rows and {len(df.columns)} cols")


if __name__ == "__main__":
    main()
