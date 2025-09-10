"""
MLflow utilities for experiment tracking and logging.

Provides helper functions for managing MLflow experiments, runs, and artifacts
with sensible defaults and environment-based configuration.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import mlflow


def get_or_create_experiment(name: str) -> str:
    """
    Get or create an MLflow experiment by name.

    Args:
        name: Name of the experiment

    Returns:
        Experiment ID as string
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


@contextmanager
def start_run(
    experiment_name: str, run_name: str | None = None, nested: bool = False
) -> Generator[mlflow.ActiveRun, None, None]:
    """
    Context manager for starting an MLflow run with automatic configuration.

    Automatically sets up tracking URI, experiment, and common tags including
    git commit hash and hostname.

    Args:
        experiment_name: Name of the experiment
        run_name: Optional name for the run
        nested: Whether this is a nested run

    Yields:
        Active MLflow run object
    """
    # Set tracking URI from environment or default to local ./mlruns
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Get or create experiment
    experiment_id = get_or_create_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, nested=nested
    ) as run:
        # Set common tags
        tags = {}

        # Add git commit hash if available
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            tags["git_commit"] = git_commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Add hostname
        try:
            tags["hostname"] = socket.gethostname()
        except Exception:
            pass

        # Set tags if we have any
        if tags:
            mlflow.set_tags(tags)

        yield run


def log_dict_as_artifact(d: dict[str, Any], path: str) -> None:
    """
    Log a dictionary as a JSON artifact.

    Args:
        d: Dictionary to log
        path: Path for the artifact (e.g., "config.json", "metrics/summary.json")
    """
    # Create temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(d, f, indent=2, default=str)
        temp_path = f.name

    try:
        # Log the artifact
        mlflow.log_artifact(temp_path, path)
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


def set_tags(tags: dict[str, str]) -> None:
    """
    Convenience function to set multiple tags.

    Args:
        tags: Dictionary of tag key-value pairs
    """
    mlflow.set_tags(tags)
