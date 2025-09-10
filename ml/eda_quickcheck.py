#!/usr/bin/env python3
"""
Quick EDA script for Spotify features dataset.

Analyzes data/interim/spotify_features.parquet and prints summary statistics,
while saving key artifacts to data/interim/eda/ for reference.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def main():
    # Load data
    data_path = Path("data/interim/spotify_features.parquet")
    output_dir = Path("data/interim/eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 SPOTIFY FEATURES EDA QUICKCHECK")
    print("=" * 50)
    
    df = pd.read_parquet(data_path)
    
    # Basic shape and head
    print(f"\n📈 DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head(5))
    
    # Missingness analysis
    print(f"\n🔍 MISSINGNESS ANALYSIS")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"{col}: {pct:.2f}%")
    if missing_pct.max() == 0:
        print("No missing values found! ✅")
    
    # Year distribution (extract from album_release_date if available)
    print(f"\n📅 YEAR DISTRIBUTION")
    if "album_release_date" in df.columns:
        df["year"] = pd.to_datetime(df["album_release_date"], errors="coerce").dt.year
        year_2000_2023 = df[(df["year"] >= 2000) & (df["year"] <= 2023)]
        
        if len(year_2000_2023) > 0:
            year_counts = year_2000_2023["year"].value_counts().sort_index()
            print(f"Years 2000-2023: {len(year_2000_2023):,} tracks")
            print(f"Year range: {year_counts.index.min()}-{year_counts.index.max()}")
            print(f"Mean tracks per year: {year_counts.mean():.0f}")
            print(f"Peak year: {year_counts.idxmax()} ({year_counts.max():,} tracks)")
            
            # Save year counts
            year_counts.to_csv(output_dir / "year_counts.csv", header=["count"])
            print(f"💾 Saved: {output_dir}/year_counts.csv")
        else:
            print("No valid years found in 2000-2023 range")
    else:
        print("No album_release_date column found")
    
    # Top 30 genres
    print(f"\n🎵 TOP 30 GENRES")
    if "genre" in df.columns:
        genre_counts = df["genre"].value_counts().head(30)
        print(genre_counts)
        
        # Save top 30 genres
        genre_counts.to_csv(output_dir / "genre_top30.csv", header=["count"])
        print(f"💾 Saved: {output_dir}/genre_top30.csv")
    else:
        print("No genre column found")
    
    # Spotify audio features statistics
    print(f"\n🎶 SPOTIFY AUDIO FEATURES")
    audio_features = [
        "danceability", "energy", "valence", "acousticness", 
        "instrumentalness", "speechiness", "liveness", "tempo", "loudness"
    ]
    
    available_features = [f for f in audio_features if f in df.columns]
    if available_features:
        feature_stats = df[available_features].describe()
        print(feature_stats)
        
        # Save feature descriptions
        feature_stats.to_csv(output_dir / "feature_describe.csv")
        print(f"💾 Saved: {output_dir}/feature_describe.csv")
    else:
        print("No audio features found")
    
    # Categorical feature value counts
    print(f"\n🔢 CATEGORICAL FEATURES")
    categorical_features = ["key", "mode", "time_signature"]
    
    categorical_counts = {}
    for feature in categorical_features:
        if feature in df.columns:
            if feature == "key":
                # Sort by key index (0-11 for musical keys)
                counts = df[feature].value_counts().sort_index()
            else:
                counts = df[feature].value_counts()
            
            print(f"\n{feature.upper()}:")
            print(counts)
            categorical_counts[feature] = counts
    
    # Save categorical counts
    if categorical_counts:
        categorical_df = pd.DataFrame(categorical_counts)
        categorical_df.to_csv(output_dir / "class_counts.csv")
        print(f"💾 Saved: {output_dir}/class_counts.csv")
    
    # Duration analysis
    print(f"\n⏱️  DURATION ANALYSIS")
    if "duration_ms" in df.columns:
        duration_min = df["duration_ms"] / 1000 / 60  # Convert to minutes
        
        print(f"Duration (minutes):")
        print(f"  Min: {duration_min.min():.2f}")
        print(f"  Median: {duration_min.median():.2f}")
        print(f"  Max: {duration_min.max():.2f}")
        
        # Check tracks outside [10s, 15min] range
        outside_range = (df["duration_ms"] < 10000) | (df["duration_ms"] > 15*60*1000)
        pct_outside = (outside_range.sum() / len(df)) * 100
        
        print(f"Tracks outside [10s, 15min]: {outside_range.sum():,} ({pct_outside:.2f}%)")
        
        if pct_outside > 0:
            too_short = (df["duration_ms"] < 10000).sum()
            too_long = (df["duration_ms"] > 15*60*1000).sum()
            print(f"  Too short (<10s): {too_short:,}")
            print(f"  Too long (>15min): {too_long:,}")
    else:
        print("No duration_ms column found")
    
    print(f"\n✅ EDA COMPLETE")
    print(f"📁 Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()