"""
Utility functions for price prediction.

This module contains helper functions for price transformations,
configuration management, and other common operations.
"""

from typing import Optional

import numpy as np
import optuna


def transformed_prices(
    prices: np.ndarray, transform_type: Optional[str] = "log"
) -> np.ndarray:
    """Transform prices for training (log, unit, or none) and return as float32."""
    if transform_type == "log":
        result: np.ndarray = np.log1p(prices).astype("float32")
        return result
    elif transform_type == "unit":
        result_unit: np.ndarray = (prices / 1000).astype("float32")
        return result_unit
    elif transform_type is None or transform_type == "none":
        result_none: np.ndarray = prices.astype("float32")
        return result_none
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")


def convert_back_y(
    transformed_prices: np.ndarray, transform_type: Optional[str] = "log"
) -> np.ndarray:
    """Convert transformed prices back to original dollar space."""
    if transform_type == "log":
        result: np.ndarray = np.expm1(transformed_prices)
        return result
    elif transform_type == "unit":
        result_unit: np.ndarray = transformed_prices * 1000
        return result_unit
    elif transform_type is None or transform_type == "none":
        return transformed_prices
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")


def print_best_trials(study: optuna.Study, n: int = 5) -> None:
    """Print top N trials from Optuna study with hyperparameters and MAE."""
    print("\n" + "=" * 80)
    print(f"TOP {n} TRIALS")
    print("=" * 80)

    # Get top N trials sorted by value (MAE)
    top_trials = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float("inf")
    )[:n]

    for rank, trial in enumerate(top_trials, 1):
        print(f"\n{'─' * 80}")
        print(f"Rank {rank}: Trial #{trial.number}")
        print(f"{'─' * 80}")
        print(f"Validation MAE: ${trial.value:.2f}")
        print("\nHyperparameters:")

        params = trial.params
        lr = params.get("learning_rate", 0)
        wd = params.get("weight_decay", 0)
        db = params.get("dropout_base", 0)
        dim = params.get("initial_dim", 0)

        print(f"  learning_rate:  {lr:.2e}")
        print(f"  weight_decay:   {wd:.2e}")
        print(
            f"  dropout_base:   {db:.3f} (dropout1={min(2 * db, 0.5):.3f}, dropout2={db:.3f})"
        )
        print(f"  initial_dim:    {dim}")

    print("\n" + "=" * 80)
    print(
        f"\nBest Trial: #{study.best_trial.number} with Val MAE: ${study.best_value:.2f}"
    )
    print("=" * 80)
