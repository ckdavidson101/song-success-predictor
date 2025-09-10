"""
Baseline translator trainer (DSP-only) with MLflow logging.

Trains models to predict Spotify audio features from DSP features extracted
from 30-second audio previews.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold, train_test_split
from scipy.stats import spearmanr

from ml.infra.mlflow_utils import log_dict_as_artifact, start_run


# LightGBM availability check
def _check_lightgbm():
    """Check if LightGBM is available and working."""
    try:
        import lightgbm as lgb

        # Test that it can actually load
        _ = lgb.LGBMRegressor(n_estimators=1, verbosity=-1)
        return True, lgb
    except Exception:
        return False, None


HAS_LIGHTGBM, lgb = _check_lightgbm()
if not HAS_LIGHTGBM:
    warnings.warn("LightGBM not available, using RandomForest instead")


def load_and_join_data(features_path: Path, labels_path: Path) -> pd.DataFrame:
    """Load features and labels, join on track_id."""
    print(f"📊 Loading features from {features_path}")
    features_df = pd.read_parquet(features_path)

    print(f"🎯 Loading labels from {labels_path}")
    labels_df = pd.read_parquet(labels_path)

    # Keep only Spotify target columns
    target_columns = [
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

    # Add optional grouping columns if they exist
    optional_columns = ["artist_name", "track_name"]
    for col in optional_columns:
        if col in labels_df.columns:
            target_columns.append(col)

    available_targets = [col for col in target_columns if col in labels_df.columns]
    labels_df = labels_df[available_targets]

    print("Inner joining on track_id...")
    dataset = pd.merge(features_df, labels_df, on="track_id", how="inner")

    print(f"Dataset shape after join: {dataset.shape}")
    print(
        f"Available targets: {[col for col in available_targets if col != 'track_id']}"
    )

    return dataset


def create_splits(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.125,  # 0.125 of remaining 80% = 10% of total
    group_by: str = "",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits with optional grouped splitting."""

    if group_by and group_by in dataset.columns:
        print(f"Using grouped split by {group_by}")

        # Use GroupKFold for artist-based splitting
        groups = dataset[group_by]
        gkf = GroupKFold(n_splits=5)

        # Get first fold as test, second fold as val, rest as train
        splits = list(gkf.split(dataset, groups=groups))
        train_idx, test_idx = splits[0]

        # Split train further for validation
        train_data = dataset.iloc[train_idx]
        train_groups = train_data[group_by]

        if len(train_data) > 100:  # Only split if we have enough data
            gkf_val = GroupKFold(n_splits=4)
            val_splits = list(gkf_val.split(train_data, groups=train_groups))
            train_final_idx, val_idx = val_splits[0]

            train_df = train_data.iloc[train_final_idx]
            val_df = train_data.iloc[val_idx]
        else:
            # Too small for further splitting
            train_df = train_data
            val_df = train_data.sample(
                n=min(10, len(train_data) // 10), random_state=seed
            )

        test_df = dataset.iloc[test_idx]

    else:
        print("Using random split")
        # Standard train/test split
        train_df, test_df = train_test_split(
            dataset, test_size=test_size, random_state=seed, stratify=None
        )

        # Split train into train/val
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, random_state=seed, stratify=None
        )

    print(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def get_target_info() -> dict[str, dict]:
    """Define target variable types and ranges."""
    return {
        # Regression targets (0-1)
        "danceability": {"type": "regression", "range": (0, 1)},
        "energy": {"type": "regression", "range": (0, 1)},
        "valence": {"type": "regression", "range": (0, 1)},
        "acousticness": {"type": "regression", "range": (0, 1)},
        "instrumentalness": {"type": "regression", "range": (0, 1)},
        "speechiness": {"type": "regression", "range": (0, 1)},
        "liveness": {"type": "regression", "range": (0, 1)},
        # Regression targets (numeric)
        "tempo": {"type": "regression", "range": (0, 300)},
        "loudness": {"type": "regression", "range": (-60, 5)},
        # Classification targets
        "key": {"type": "classification", "classes": list(range(12))},
        "mode": {"type": "classification", "classes": [0, 1]},
        "time_signature": {"type": "classification", "classes": [3, 4, 5, 7]},
    }


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, target_name: str, target_info: dict
) -> Any:
    """Train a model for a specific target."""

    if target_info["type"] == "regression":
        if HAS_LIGHTGBM:
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:  # classification
        if HAS_LIGHTGBM:
            model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    return model


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Spearman correlation
    try:
        spearman_r, _ = spearmanr(y_true, y_pred)
        if np.isnan(spearman_r):
            spearman_r = 0.0
    except Exception:
        spearman_r = 0.0

    return {"mae": mae, "rmse": rmse, "spearman_r": spearman_r}


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {"accuracy": accuracy, "f1_macro": f1_macro}


def save_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, target_name: str, artifacts_dir: Path
) -> None:
    """Save confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {target_name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    cm_path = artifacts_dir / f"confusion_matrix_{target_name}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()


def clip_predictions(y_pred: np.ndarray, target_info: dict) -> np.ndarray:
    """Clip predictions to valid range."""
    if target_info["type"] == "regression" and "range" in target_info:
        min_val, max_val = target_info["range"]
        return np.clip(y_pred, min_val, max_val)
    return y_pred


def main():
    parser = argparse.ArgumentParser(description="Train translator models")
    parser.add_argument(
        "--features", required=True, help="Path to DSP features parquet"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to Spotify labels parquet"
    )
    parser.add_argument("--artifacts", required=True, help="Artifacts output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--experiment", default="translator_baseline_dsp", help="MLflow experiment"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit dataset size (0=no limit)"
    )
    parser.add_argument(
        "--group-by", default="", help="Column for grouped split (e.g. artist_name)"
    )

    args = parser.parse_args()

    features_path = Path(args.features)
    labels_path = Path(args.labels)
    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("🎵 TRANSLATOR TRAINING")
    print("=" * 50)
    print(f"Features: {features_path}")
    print(f"Labels: {labels_path}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Experiment: {args.experiment}")
    print(f"Model: {'LightGBM' if HAS_LIGHTGBM else 'RandomForest'}")
    print()

    with start_run(args.experiment, run_name="translator_training") as run:

        # Load and prepare data
        dataset = load_and_join_data(features_path, labels_path)

        # Sample if limit specified
        if args.limit > 0 and len(dataset) > args.limit:
            dataset = dataset.sample(n=args.limit, random_state=args.seed)
            print(f"Sampled {args.limit} tracks")

        # Get feature columns (exclude targets and metadata)
        target_info = get_target_info()
        exclude_cols = set(
            ["track_id"] + list(target_info.keys()) + ["artist_name", "track_name"]
        )
        feature_cols = [col for col in dataset.columns if col not in exclude_cols]

        print(f"Feature columns: {len(feature_cols)}")

        # Create splits
        train_df, val_df, test_df = create_splits(
            dataset, group_by=args.group_by, seed=args.seed
        )

        # Log parameters
        import mlflow

        mlflow.log_params(
            {
                "n_samples": len(dataset),
                "n_features": len(feature_cols),
                "model_type": "lightgbm" if HAS_LIGHTGBM else "random_forest",
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "group_by": args.group_by,
                "seed": args.seed,
            }
        )

        # Train and evaluate models for each target
        results = {}
        models = {}

        print("\n🚀 Training models...")

        for target_name, info in target_info.items():
            if target_name not in dataset.columns:
                print(f"⚠️  Skipping {target_name} (not in dataset)")
                continue

            print(f"\nTraining {target_name} ({info['type']})...")

            # Prepare data
            X_train = train_df[feature_cols]
            y_train = train_df[target_name].dropna()
            X_train = X_train.loc[y_train.index]

            X_val = val_df[feature_cols]
            y_val = val_df[target_name].dropna()
            X_val = X_val.loc[y_val.index]

            X_test = test_df[feature_cols]
            y_test = test_df[target_name].dropna()
            X_test = X_test.loc[y_test.index]

            if len(y_train) < 10:
                print(f"⚠️  Skipping {target_name} (insufficient data: {len(y_train)})")
                continue

            # Train model
            model = train_model(X_train, y_train, target_name, info)
            models[target_name] = model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            # Clip predictions to valid range
            y_train_pred = clip_predictions(y_train_pred, info)
            y_val_pred = clip_predictions(y_val_pred, info)
            y_test_pred = clip_predictions(y_test_pred, info)

            # Evaluate
            if info["type"] == "regression":
                train_metrics = evaluate_regression(y_train, y_train_pred)
                val_metrics = evaluate_regression(y_val, y_val_pred)
                test_metrics = evaluate_regression(y_test, y_test_pred)

                results[target_name] = {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                }

                # Log test metrics to MLflow
                for metric, value in test_metrics.items():
                    mlflow.log_metric(f"{target_name}_{metric}_test", value)

            else:  # classification
                train_metrics = evaluate_classification(y_train, y_train_pred)
                val_metrics = evaluate_classification(y_val, y_val_pred)
                test_metrics = evaluate_classification(y_test, y_test_pred)

                results[target_name] = {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                }

                # Log test metrics to MLflow
                for metric, value in test_metrics.items():
                    mlflow.log_metric(f"{target_name}_{metric}_test", value)

                # Save confusion matrix
                save_confusion_matrix(y_test, y_test_pred, target_name, artifacts_dir)

            # Save model
            model_path = artifacts_dir / f"model_{target_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save artifacts
        log_dict_as_artifact(results, "metrics.json")
        log_dict_as_artifact(
            {"feature_count": len(feature_cols), "feature_names": feature_cols},
            "feature_schema.json",
        )

        # Print summary table
        print("\n📊 RESULTS SUMMARY")
        print("=" * 80)
        print(
            f"{'Target':<20} {'Type':<12} {'Test MAE':<10} {'Test RMSE':<11} {'Test Spearman':<13} {'Test Acc':<10} {'Test F1':<10}"
        )
        print("-" * 80)

        for target_name, target_results in results.items():
            info = target_info[target_name]
            test_metrics = target_results["test"]

            if info["type"] == "regression":
                print(
                    f"{target_name:<20} {'Regression':<12} {test_metrics['mae']:<10.4f} {test_metrics['rmse']:<11.4f} {test_metrics['spearman_r']:<13.4f} {'-':<10} {'-':<10}"
                )
            else:
                print(
                    f"{target_name:<20} {'Classification':<12} {'-':<10} {'-':<11} {'-':<13} {test_metrics['accuracy']:<10.4f} {test_metrics['f1_macro']:<10.4f}"
                )

        print(f"\n✅ Training complete! MLflow run: {run.info.run_id}")
        print(f"📁 Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
