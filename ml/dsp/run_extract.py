"""
Subset-friendly DSP feature extraction runner with MLflow logging.

Extracts DSP features from preview files, joins with Spotify labels, and logs
results to MLflow for experiment tracking.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from ml.dsp.extract import extract_features
from ml.infra.mlflow_utils import log_dict_as_artifact, set_tags, start_run


def load_and_filter_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load manifest and filter for successful downloads."""
    print(f"📊 Loading manifest from {manifest_path}")
    manifest_df = pd.read_parquet(manifest_path)

    # Filter for successful downloads
    success_mask = manifest_df["download_status"] == "success"
    filtered_df = manifest_df[success_mask].copy()

    print(
        f"Found {len(filtered_df):,} successful downloads out of {len(manifest_df):,} total"
    )

    # Keep only necessary columns
    columns_to_keep = ["track_id", "file_path"]
    filtered_df = filtered_df[columns_to_keep].copy()

    return filtered_df


def load_spotify_labels(labels_path: Path) -> pd.DataFrame:
    """Load Spotify features and keep target columns."""
    print(f"🎯 Loading Spotify labels from {labels_path}")
    labels_df = pd.read_parquet(labels_path)

    # Keep Spotify audio feature targets
    spotify_features = [
        "track_id",
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

    # Only keep columns that exist in the dataframe
    available_features = [col for col in spotify_features if col in labels_df.columns]
    labels_df = labels_df[available_features].copy()

    print(
        f"Loaded {len(labels_df):,} tracks with {len(available_features)} Spotify features"
    )
    return labels_df


def create_dataset(
    manifest_df: pd.DataFrame, labels_df: pd.DataFrame, limit: int, seed: int
) -> pd.DataFrame:
    """Join manifest with labels and optionally sample."""
    print("🔗 Joining manifest with labels...")

    # Inner join on track_id
    dataset = pd.merge(manifest_df, labels_df, on="track_id", how="inner")
    print(f"After join: {len(dataset):,} tracks with both previews and labels")

    # Sample if limit specified
    if limit > 0 and len(dataset) > limit:
        dataset = dataset.sample(n=limit, random_state=seed).reset_index(drop=True)
        print(f"Sampled {limit:,} tracks for processing")

    return dataset


def extract_features_safe(file_path: str) -> dict[str, Any]:
    """Safely extract features from a single file."""
    try:
        features = extract_features(file_path)
        # Get track_id from filename (remove .mp3 extension)
        track_id = Path(file_path).stem
        features["track_id"] = track_id
        features["_success"] = True
        features["_error"] = None
        return features
    except Exception as e:
        track_id = Path(file_path).stem
        return {"track_id": track_id, "_success": False, "_error": str(e)}


def parallel_extract_features(
    file_paths: list[str], workers: int
) -> tuple[list[dict], list[dict]]:
    """Extract features in parallel and separate successes/failures."""
    print(f"🚀 Starting parallel extraction with {workers} workers...")

    successes = []
    errors = []
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(extract_features_safe, file_path): file_path
            for file_path in file_paths
        }

        # Process results as they complete
        for future in as_completed(future_to_path):
            file_path = future_to_path[future]

            try:
                result = future.result()

                if result["_success"]:
                    # Remove internal flags before storing
                    del result["_success"]
                    del result["_error"]
                    successes.append(result)
                else:
                    errors.append(
                        {
                            "track_id": result["track_id"],
                            "file_path": file_path,
                            "error": result["_error"],
                        }
                    )

                completed += 1

                # Progress reporting
                if completed % 100 == 0 or completed == len(file_paths):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    success_rate = len(successes) / completed * 100
                    print(
                        f"  Progress: {completed:,}/{len(file_paths):,} "
                        f"({success_rate:.1f}% success, {rate:.1f} files/sec)"
                    )

            except Exception as e:
                track_id = Path(file_path).stem
                errors.append(
                    {"track_id": track_id, "file_path": file_path, "error": str(e)}
                )
                completed += 1

    return successes, errors


def save_results(successes: list[dict], errors: list[dict], out_path: Path) -> None:
    """Save successful extractions and errors to parquet files."""
    # Create output directory
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if successes:
        success_df = pd.DataFrame(successes)
        success_df.to_parquet(out_path, index=False)
        print(f"💾 Saved {len(successes):,} successful extractions to {out_path}")
    else:
        print("❌ No successful extractions to save")

    if errors:
        error_path = out_path.parent / "translator_dsp_errors.parquet"
        error_df = pd.DataFrame(errors)
        error_df.to_parquet(error_path, index=False)
        print(f"⚠️  Saved {len(errors):,} errors to {error_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract DSP features for translator training"
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to previews manifest parquet file"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to Spotify features parquet file"
    )
    parser.add_argument(
        "--out", required=True, help="Output path for DSP features parquet"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tracks to process (0 = no limit)",
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="Number of parallel workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--experiment", default="translator_baseline_dsp", help="MLflow experiment name"
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    labels_path = Path(args.labels)
    out_path = Path(args.out)

    print("🎵 DSP FEATURE EXTRACTION RUNNER")
    print("=" * 50)
    print(f"Manifest: {manifest_path}")
    print(f"Labels: {labels_path}")
    print(f"Output: {out_path}")
    print(f"Limit: {args.limit if args.limit > 0 else 'None'}")
    print(f"Workers: {args.workers}")
    print(f"Experiment: {args.experiment}")
    print()

    start_time = time.time()

    with start_run(args.experiment, run_name="dsp_extraction") as run:
        try:
            # Load and prepare data
            manifest_df = load_and_filter_manifest(manifest_path)
            labels_df = load_spotify_labels(labels_path)
            dataset = create_dataset(manifest_df, labels_df, args.limit, args.seed)

            if len(dataset) == 0:
                print("❌ No tracks to process after joining")
                return

            # Log parameters
            params = {
                "subset_n": len(dataset),
                "workers": args.workers,
                "dsp_recipe_version": "v1_librosa_baseline",
                "limit": args.limit,
                "seed": args.seed,
            }

            import mlflow

            mlflow.log_params(params)

            # Extract features
            file_paths = dataset["file_path"].tolist()
            successes, errors = parallel_extract_features(file_paths, args.workers)

            # Calculate metrics
            total_time = time.time() - start_time
            success_rate = len(successes) / len(dataset) * 100
            avg_extract_ms = (
                (total_time / len(dataset)) * 1000 if len(dataset) > 0 else 0
            )

            metrics = {
                "success_rate": success_rate,
                "avg_extract_ms": avg_extract_ms,
                "total_tracks": len(dataset),
                "successful_extractions": len(successes),
                "failed_extractions": len(errors),
                "total_time_sec": total_time,
            }

            mlflow.log_metrics(metrics)

            # Create and log schema artifact
            if successes:
                sample_features = successes[0]
                # Remove track_id for schema (it's not a DSP feature)
                feature_names = [k for k in sample_features.keys() if k != "track_id"]
                schema_info = {
                    "feature_count": len(feature_names),
                    "feature_names": sorted(feature_names),
                    "dsp_version": "v1_librosa_baseline",
                    "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                log_dict_as_artifact(schema_info, "dsp_schema.json")

            # Set additional tags
            set_tags(
                {
                    "data_type": "dsp_features",
                    "success_rate_pct": f"{success_rate:.1f}%",
                }
            )

            # Save results
            save_results(successes, errors, out_path)

            # Final summary
            print("\n✅ EXTRACTION COMPLETE")
            print(
                f"Success rate: {success_rate:.1f}% ({len(successes)}/{len(dataset)})"
            )
            print(f"Total time: {total_time:.1f}s")
            print(f"Average time per track: {avg_extract_ms:.0f}ms")
            print(f"MLflow run: {run.info.run_id}")

        except Exception as e:
            print(f"❌ Fatal error: {e}")
            import mlflow

            mlflow.log_param("fatal_error", str(e))
            raise


if __name__ == "__main__":
    main()
