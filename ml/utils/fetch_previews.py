#!/usr/bin/env python3
"""
Preview fetcher for downloading 30-second MP3 previews from Spotify/Deezer.

Reads manifest, resolves preview URLs (Spotify first, then Deezer), downloads MP3s
to data/previews/ directory, and creates a preview manifest with status tracking.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def setup_session() -> requests.Session:
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )
    return session


def get_spotify_preview(
    session: requests.Session, track_id: str, track_name: str, artist_name: str
) -> dict[str, Any]:
    """
    Get preview URL from Spotify Web API (requires token from backend service).

    Returns dict with 'status', 'preview_url', 'source', 'error' keys.
    """
    try:
        # Use our backend service to get Spotify preview
        response = session.get(
            "http://localhost:8000/predict/spotify_track",
            params={"track_id": track_id},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            preview_url = data.get("preview_url")
            if preview_url:
                return {
                    "status": "success",
                    "preview_url": preview_url,
                    "source": "spotify",
                    "error": None,
                }
            else:
                return {
                    "status": "no_preview",
                    "preview_url": None,
                    "source": "spotify",
                    "error": "No preview URL in response",
                }
        else:
            return {
                "status": "error",
                "preview_url": None,
                "source": "spotify",
                "error": f"HTTP {response.status_code}",
            }
    except Exception as e:
        return {
            "status": "error",
            "preview_url": None,
            "source": "spotify",
            "error": str(e),
        }


def get_deezer_preview_by_isrc(
    session: requests.Session, isrc: str | None
) -> dict[str, Any]:
    """
    Get preview URL from Deezer API using ISRC lookup.

    Returns dict with 'status', 'preview_url', 'source', 'error' keys.
    """
    if not isrc or pd.isna(isrc):
        return {
            "status": "no_isrc",
            "preview_url": None,
            "source": "deezer_isrc",
            "error": "No ISRC available",
        }

    try:
        response = session.get(f"https://api.deezer.com/track/isrc:{isrc}", timeout=10)

        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                preview_url = data.get("preview")
                if preview_url:
                    return {
                        "status": "success",
                        "preview_url": preview_url,
                        "source": "deezer_isrc",
                        "error": None,
                    }
                else:
                    return {
                        "status": "no_preview",
                        "preview_url": None,
                        "source": "deezer_isrc",
                        "error": "No preview in response",
                    }
            else:
                return {
                    "status": "not_found",
                    "preview_url": None,
                    "source": "deezer_isrc",
                    "error": data.get("error", {}).get("message", "Unknown error"),
                }
        else:
            return {
                "status": "error",
                "preview_url": None,
                "source": "deezer_isrc",
                "error": f"HTTP {response.status_code}",
            }
    except Exception as e:
        return {
            "status": "error",
            "preview_url": None,
            "source": "deezer_isrc",
            "error": str(e),
        }


def get_deezer_preview_by_search(
    session: requests.Session, track_name: str, artist_name: str
) -> dict[str, Any]:
    """
    Get preview URL from Deezer API using track/artist search.

    Returns dict with 'status', 'preview_url', 'source', 'error' keys.
    """
    try:
        # Clean and encode search query
        query = f"{artist_name} {track_name}".strip()
        if not query:
            return {
                "status": "no_query",
                "preview_url": None,
                "source": "deezer_search",
                "error": "Empty search query",
            }

        response = session.get(
            "https://api.deezer.com/search/track",
            params={"q": query, "limit": "5"},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            tracks = data.get("data", [])

            # Look for tracks with previews
            for track in tracks:
                preview_url = track.get("preview")
                if preview_url:
                    return {
                        "status": "success",
                        "preview_url": preview_url,
                        "source": "deezer_search",
                        "error": None,
                    }

            return {
                "status": "no_preview",
                "preview_url": None,
                "source": "deezer_search",
                "error": f"No previews in {len(tracks)} search results",
            }
        else:
            return {
                "status": "error",
                "preview_url": None,
                "source": "deezer_search",
                "error": f"HTTP {response.status_code}",
            }
    except Exception as e:
        return {
            "status": "error",
            "preview_url": None,
            "source": "deezer_search",
            "error": str(e),
        }


def download_preview(
    session: requests.Session, preview_url: str, output_path: Path
) -> dict[str, Any]:
    """
    Download MP3 preview from URL to local file.

    Returns dict with 'status', 'file_size', 'error' keys.
    """
    try:
        response = session.get(preview_url, timeout=30, stream=True)
        response.raise_for_status()

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file in chunks
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = output_path.stat().st_size

        if file_size == 0:
            output_path.unlink(missing_ok=True)
            return {"status": "error", "file_size": 0, "error": "Downloaded empty file"}

        return {"status": "success", "file_size": file_size, "error": None}
    except Exception as e:
        # Clean up partial download
        if output_path.exists():
            output_path.unlink(missing_ok=True)

        return {"status": "error", "file_size": 0, "error": str(e)}


def process_track(track_data: dict[str, Any], previews_dir: Path) -> dict[str, Any]:
    """
    Process a single track: resolve preview URL and download MP3.

    Returns dict with all status information for manifest.
    """
    session = setup_session()
    track_id = track_data["track_id"]
    track_name = track_data["track_name"]
    artist_name = track_data["artist_name"]
    isrc = track_data.get("isrc")

    output_path = previews_dir / f"{track_id}.mp3"

    # Skip if already downloaded
    if output_path.exists():
        file_size = output_path.stat().st_size
        return {
            "track_id": track_id,
            "preview_status": "already_exists",
            "preview_source": "local",
            "preview_url": None,
            "download_status": "skipped",
            "file_size": file_size,
            "file_path": str(output_path),
            "error_message": None,
        }

    # Try resolution strategies in order
    strategies = [
        lambda: get_spotify_preview(session, track_id, track_name, artist_name),
        lambda: get_deezer_preview_by_isrc(session, isrc),
        lambda: get_deezer_preview_by_search(session, track_name, artist_name),
    ]

    preview_result = None
    for strategy in strategies:
        result = strategy()
        if result["status"] == "success":
            preview_result = result
            break
        preview_result = result  # Keep last attempt for error reporting

    # If no preview URL found, return failure
    if not preview_result or not preview_result.get("preview_url"):
        return {
            "track_id": track_id,
            "preview_status": preview_result["status"] if preview_result else "failed",
            "preview_source": preview_result["source"] if preview_result else "unknown",
            "preview_url": None,
            "download_status": "not_attempted",
            "file_size": 0,
            "file_path": None,
            "error_message": (
                preview_result["error"] if preview_result else "No preview URL found"
            ),
        }

    # Download the preview
    download_result = download_preview(
        session, preview_result["preview_url"], output_path
    )

    return {
        "track_id": track_id,
        "preview_status": preview_result["status"],
        "preview_source": preview_result["source"],
        "preview_url": preview_result["preview_url"],
        "download_status": download_result["status"],
        "file_size": download_result["file_size"],
        "file_path": (
            str(output_path) if download_result["status"] == "success" else None
        ),
        "error_message": download_result["error"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch preview MP3s for manifest tracks"
    )
    parser.add_argument(
        "--manifest",
        default="data/interim/manifest_v1.parquet",
        help="Input manifest parquet file",
    )
    parser.add_argument(
        "--previews-dir", default="data/previews", help="Output directory for MP3 files"
    )
    parser.add_argument(
        "--output",
        default="data/interim/previews_manifest.parquet",
        help="Output manifest with preview status",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of concurrent downloads",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of tracks to process (for testing)"
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    previews_dir = Path(args.previews_dir)
    output_path = Path(args.output)

    # Create output directories
    previews_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("🎵 PREVIEW FETCHER")
    print("=" * 50)
    print(f"Manifest: {manifest_path}")
    print(f"Previews dir: {previews_dir}")
    print(f"Max workers: {args.max_workers}")

    # Load manifest
    print("\n📊 Loading manifest...")
    df = pd.read_parquet(manifest_path)
    print(f"Loaded {len(df):,} tracks")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df):,} tracks for testing")

    # Check existing downloads
    existing_files = list(previews_dir.glob("*.mp3"))
    print(f"Found {len(existing_files)} existing MP3 files")

    # Convert to list of dicts for processing
    tracks = df.to_dict("records")

    print(f"\n🎯 Starting preview fetching with {args.max_workers} workers...")
    start_time = time.time()

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_track = {
            executor.submit(process_track, track, previews_dir): track
            for track in tracks
        }

        # Process completed tasks
        for future in as_completed(future_to_track):
            track = future_to_track[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if completed % 100 == 0 or completed == len(tracks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {completed:,}/{len(tracks):,} ({completed/len(tracks)*100:.1f}%) - {rate:.1f} tracks/sec"
                    )

            except Exception as e:
                print(f"  ❌ Error processing {track.get('track_id', 'unknown')}: {e}")
                results.append(
                    {
                        "track_id": track.get("track_id", "unknown"),
                        "preview_status": "error",
                        "preview_source": "unknown",
                        "preview_url": None,
                        "download_status": "error",
                        "file_size": 0,
                        "file_path": None,
                        "error_message": str(e),
                    }
                )
                completed += 1

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Merge with original manifest
    final_df = pd.merge(df, results_df, on="track_id", how="left")

    # Save preview manifest
    print("\n💾 Saving preview manifest...")
    final_df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")

    # Print summary statistics
    elapsed = time.time() - start_time
    print("\n📊 SUMMARY")
    print("=" * 30)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Average rate: {len(tracks)/elapsed:.1f} tracks/sec")

    status_counts = results_df["preview_status"].value_counts()
    print("\n🔍 Preview Resolution:")
    for status, count in status_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")

    source_counts = results_df[results_df["preview_status"] == "success"][
        "preview_source"
    ].value_counts()
    if len(source_counts) > 0:
        print("\n📡 Successful Sources:")
        for source, count in source_counts.items():
            pct = count / len(source_counts) * 100
            print(f"  {source}: {count:,} ({pct:.1f}%)")

    download_counts = results_df["download_status"].value_counts()
    print("\n⬇️  Download Status:")
    for status, count in download_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {status}: {count:,} ({pct:.1f}%)")

    # File size stats
    successful_downloads = results_df[results_df["download_status"] == "success"][
        "file_size"
    ]
    if len(successful_downloads) > 0:
        total_size_mb = successful_downloads.sum() / 1024 / 1024
        avg_size_kb = successful_downloads.mean() / 1024
        print("\n📁 File Stats:")
        print(f"  Total size: {total_size_mb:.1f} MB")
        print(f"  Average size: {avg_size_kb:.0f} KB")
        print(f"  Files downloaded: {len(successful_downloads):,}")

    print("\n✅ Preview fetching complete!")
    print(f"📁 MP3 files: {previews_dir}")
    print(f"📊 Manifest: {output_path}")


if __name__ == "__main__":
    main()
