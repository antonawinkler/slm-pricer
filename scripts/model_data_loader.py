"""
Module for loading and processing model results data.
Provides utilities for loading model results from JSON files in the data folder.
"""

import glob
import json
from typing import Any, TypedDict


class ModelData(TypedDict):
    """Type definition for model data dictionary."""

    name: str
    guesses: list[float]
    truths: list[float]
    errors: list[float]
    cumulative_means: list[float]
    size: int


def load_model_data(
    start_index: int = 0,
    end_index: int | None = None,
    file_pattern: str = "data/[!_]*.json",
) -> list[ModelData]:
    """
    Load all model results from data folder and apply optional data slice.

    Args:
        data_slice: Optional slice object to apply to the data (e.g., slice(100, None))
                   If None, loads all data points.

    Returns:
        List of dictionaries containing model data with the following keys:
            - name: Model display name (from 'title' field or filename)
            - guesses: List of guess values
            - truths: List of truth values
            - errors: List of error values
            - cumulative_means: List of cumulative mean errors
            - size: Number of data points
    """
    data_slice = slice(start_index, end_index)

    data_files: list[str] = sorted(glob.glob(file_pattern))
    models_data: list[ModelData] = []

    for file_path in data_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            model_name: str = data["title"]
            errors: list[float] = data["errors"][data_slice]
            # For convenience calculate cumulative mean over the sequence
            cumulative_means: list[float] = [
                sum(errors[: i + 1]) / (i + 1) for i in range(len(errors))
            ]

            models_data.append(
                {
                    "name": model_name,
                    "errors": errors,
                    "guesses": data["guesses"][data_slice],
                    "truths": data["truths"][data_slice],
                    "cumulative_means": cumulative_means,
                    "size": len(errors),
                }
            )

    return models_data


if __name__ == "__main__":
    models: list[ModelData] = load_model_data()
    for m in models:
        print(f"Loaded model: {m['name']} with {m['size']} data points")
