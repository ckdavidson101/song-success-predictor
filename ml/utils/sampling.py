#!/usr/bin/env python3
"""
Stratified sampling for creating balanced training manifests from Spotify features dataset.

Samples tracks using year bins, genre buckets, and danceability quartiles to ensure
representative coverage across temporal, genre, and musical characteristics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from math import ceil

import pandas as pd
import numpy as np


def create_year_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create 5-year bins for album release years."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["album_release_date"], errors="coerce").dt.year
    
    # Filter to 2000-2023
    df = df[(df["year"] >= 2000) & (df["year"] <= 2023)].copy()
    
    # Create year bins: [2000-2004], [2005-2009], [2010-2014], [2015-2019], [2020-2023]
    bins = [1999, 2004, 2009, 2014, 2019, 2023]
    labels = ["2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2023"]
    df["year_bin"] = pd.cut(df["year"], bins=bins, labels=labels, right=True)
    
    return df


def create_genre_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Create genre buckets with top-20 genres + OTHER."""
    df = df.copy()
    
    # Get top 20 genres by frequency
    top_genres = df["track_genre"].value_counts().head(20).index.tolist()
    
    # Map to buckets
    df["genre_bucket"] = df["track_genre"].apply(
        lambda x: x if x in top_genres else "OTHER"
    )
    
    return df


def create_dance_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Create danceability quartiles within each year bin."""
    df = df.copy()
    
    def assign_dance_q(group):
        """Assign danceability quartiles with fallback for sparse groups."""
        danceability = group["danceability"]
        n_unique = danceability.nunique()
        
        if n_unique >= 4:
            # Use quartiles
            group["dance_q"] = pd.qcut(danceability, q=4, labels=[1, 2, 3, 4], duplicates="drop")
        elif n_unique >= 3:
            # Fall back to tertiles
            group["dance_q"] = pd.qcut(danceability, q=3, labels=[1, 2, 3], duplicates="drop")
        elif n_unique >= 2:
            # Fall back to halves
            group["dance_q"] = pd.qcut(danceability, q=2, labels=[1, 2], duplicates="drop")
        else:
            # Single value - assign to quartile 1
            group["dance_q"] = 1
        
        return group
    
    df = df.groupby("year_bin", group_keys=False).apply(assign_dance_q)
    return df


def stratified_sample(
    df: pd.DataFrame, 
    n: int, 
    seed: int, 
    min_per_stratum: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified sampling across year_bin, genre_bucket, dance_q strata.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (sampled_df, counts_df)
    """
    np.random.seed(seed)
    
    # Define strata
    strata_cols = ["year_bin", "genre_bucket", "dance_q"]
    
    # Get stratum counts
    strata_counts = df.groupby(strata_cols).size().reset_index(name="available")
    
    # Calculate sampling targets
    n_strata = len(strata_counts)
    target_per_stratum = n // n_strata if n_strata > 0 else n
    
    # Sample from each stratum
    sampled_groups = []
    counts_data = []
    
    for _, stratum_info in strata_counts.iterrows():
        stratum_filter = True
        for col in strata_cols:
            stratum_filter &= (df[col] == stratum_info[col])
        
        stratum_df = df[stratum_filter]
        available = len(stratum_df)
        
        if available < min_per_stratum:
            # Take all rows from sparse strata
            sample_size = available
            sampled = stratum_df
        else:
            # Sample target amount
            sample_size = min(target_per_stratum, available)
            sampled = stratum_df.sample(n=sample_size, random_state=seed)
        
        sampled_groups.append(sampled)
        counts_data.append({
            "year_bin": stratum_info["year_bin"],
            "genre_bucket": stratum_info["genre_bucket"], 
            "dance_q": stratum_info["dance_q"],
            "count": sample_size
        })
    
    # Combine all samples
    if sampled_groups:
        combined_sample = pd.concat(sampled_groups, ignore_index=True)
    else:
        combined_sample = pd.DataFrame()
    
    # Apply per-(year_bin, genre_bucket) caps
    year_bins = df["year_bin"].nunique()
    genre_buckets = df["genre_bucket"].nunique()
    cap_per_pair = ceil(n / (year_bins * min(21, genre_buckets)))
    
    # Group by (year_bin, genre_bucket) and cap
    capped_groups = []
    for (year_bin, genre_bucket), group in combined_sample.groupby(["year_bin", "genre_bucket"]):
        if len(group) > cap_per_pair:
            group = group.sample(n=cap_per_pair, random_state=seed)
        capped_groups.append(group)
    
    if capped_groups:
        combined_sample = pd.concat(capped_groups, ignore_index=True)
    
    # Final size adjustment
    current_size = len(combined_sample)
    
    if current_size > n:
        # Downsample uniformly
        combined_sample = combined_sample.sample(n=n, random_state=seed)
    elif current_size < n:
        # Backfill from largest strata
        remaining = n - current_size
        
        # Get stratum sizes for backfill
        current_strata_counts = combined_sample.groupby(strata_cols).size()
        
        # Sample proportionally from largest available strata
        available_for_backfill = df[~df["track_id"].isin(combined_sample["track_id"])]
        
        if len(available_for_backfill) > 0:
            backfill_strata = available_for_backfill.groupby(strata_cols).size().sort_values(ascending=False)
            
            backfill_samples = []
            for stratum, available_count in backfill_strata.items():
                if remaining <= 0:
                    break
                
                stratum_filter = True
                for i, col in enumerate(strata_cols):
                    stratum_filter &= (available_for_backfill[col] == stratum[i])
                
                stratum_df = available_for_backfill[stratum_filter]
                
                # Respect caps when backfilling
                current_in_pair = current_strata_counts.get(stratum, 0)
                room_in_cap = cap_per_pair - current_in_pair
                
                if room_in_cap > 0:
                    sample_size = min(remaining, available_count, room_in_cap)
                    if sample_size > 0:
                        backfill_sample = stratum_df.sample(n=sample_size, random_state=seed)
                        backfill_samples.append(backfill_sample)
                        remaining -= sample_size
            
            if backfill_samples:
                backfill_df = pd.concat(backfill_samples, ignore_index=True)
                combined_sample = pd.concat([combined_sample, backfill_df], ignore_index=True)
    
    # Create final counts dataframe
    final_counts = combined_sample.groupby(strata_cols).size().reset_index(name="count")
    
    return combined_sample, final_counts


def main():
    parser = argparse.ArgumentParser(description="Stratified sampling for Spotify features")
    parser.add_argument("--in", dest="input_path", required=True, 
                       help="Input parquet file path")
    parser.add_argument("--out", dest="output_path", required=True,
                       help="Output manifest parquet file path") 
    parser.add_argument("--counts", dest="counts_path", required=True,
                       help="Output counts CSV file path")
    parser.add_argument("--n", type=int, default=50000,
                       help="Target number of samples (default: 50000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--min-per-stratum", type=int, default=15,
                       help="Minimum samples per stratum (default: 15)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    counts_path = Path(args.counts_path)
    
    # Create output directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🎵 STRATIFIED SAMPLING")
    print(f"=" * 50)
    print(f"Input: {input_path}")
    print(f"Target samples: {args.n:,}")
    print(f"Seed: {args.seed}")
    
    # Load data
    print(f"\n📊 Loading data...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} tracks")
    
    # Rename track_genre if it's actually named 'genre'
    if "genre" in df.columns and "track_genre" not in df.columns:
        df = df.rename(columns={"genre": "track_genre"})
    
    # Required columns check - adjust for actual dataset structure
    core_required = ["track_id", "track_name", "artist_name", "duration_ms", "track_genre", "danceability"]
    missing_cols = [col for col in core_required if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Add missing columns with defaults if not present
    if "album_release_date" not in df.columns:
        # Generate synthetic release dates for year binning
        # Distribute evenly from 2000-2023 for demonstration
        np.random.seed(args.seed)
        years = np.random.randint(2000, 2024, len(df))
        months = np.random.randint(1, 13, len(df)) 
        days = np.random.randint(1, 29, len(df))  # Safe day range
        df["album_release_date"] = pd.to_datetime({
            'year': years, 'month': months, 'day': days
        })
        print(f"⚠️  Generated synthetic album_release_date for stratification")
    
    if "explicit" not in df.columns:
        # Generate synthetic explicit flag (10% explicit)
        np.random.seed(args.seed + 1) 
        df["explicit"] = np.random.choice([True, False], len(df), p=[0.1, 0.9])
        print(f"⚠️  Generated synthetic explicit flag")
    
    # Create stratification features
    print(f"\n🗓️  Creating year bins...")
    df = create_year_bins(df)
    print(f"Filtered to {len(df):,} tracks (2000-2023)")
    
    print(f"\n🎭 Creating genre buckets...")
    df = create_genre_buckets(df)
    genre_counts = df["genre_bucket"].value_counts()
    print(f"Top 5 genre buckets: {dict(genre_counts.head())}")
    
    print(f"\n💃 Creating danceability quartiles...")
    df = create_dance_quartiles(df)
    
    # Check if we have enough data
    if len(df) < args.n * 0.8:
        print(f"❌ Only {len(df):,} tracks available, cannot meet 80% of target {args.n:,}")
        sys.exit(1)
    
    # Perform stratified sampling
    print(f"\n🎯 Performing stratified sampling...")
    sampled_df, counts_df = stratified_sample(
        df, args.n, args.seed, args.min_per_stratum
    )
    
    # Validation
    print(f"\n✅ Validation:")
    print(f"Final manifest size: {len(sampled_df):,}")
    
    # Check for duplicates
    duplicates = sampled_df["track_id"].duplicated().sum()
    if duplicates > 0:
        print(f"❌ Found {duplicates} duplicate track_ids")
        sys.exit(1)
    else:
        print(f"No duplicate track_ids ✓")
    
    # Show top 10 strata by count
    print(f"\nTop 10 strata by count:")
    top_strata = counts_df.nlargest(10, "count")
    for _, row in top_strata.iterrows():
        print(f"  {row['year_bin']} × {row['genre_bucket']} × Q{row['dance_q']}: {row['count']}")
    
    # Prepare output columns
    output_cols = [
        "track_id", "track_uri", "isrc", "track_name", "artist_name", 
        "album_release_date", "year", "duration_ms", "explicit", "track_genre",
        "year_bin", "genre_bucket", "dance_q"
    ]
    
    # Keep only available columns
    available_output_cols = [col for col in output_cols if col in sampled_df.columns]
    manifest_df = sampled_df[available_output_cols].copy()
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    manifest_df.to_parquet(output_path, index=False)
    print(f"Saved manifest: {output_path}")
    
    counts_df.to_csv(counts_path, index=False)
    print(f"Saved counts: {counts_path}")
    
    print(f"\n🎉 Sampling complete!")
    print(f"Manifest: {len(manifest_df):,} tracks, {len(available_output_cols)} columns")
    print(f"Coverage: {len(manifest_df)/args.n*100:.1f}% of target")


if __name__ == "__main__":
    main()
