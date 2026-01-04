"""
Utility functions for price prediction.

This module contains helper functions for price transformations,
configuration management, and other common operations.
"""

from typing import Callable, Optional

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


def create_early_stopping_fn(
    patience: int = 10,
    loss_patience: int = 2,
    loss_threshold: float = 1.1,
    warmup_epochs: int = 10,
) -> Callable[[int, float, float, float], tuple[bool, str]]:
    """Create an early stopping function.

    Stops training when either:
    1. Val MAE doesn't improve for `patience` epochs.
    2. Val loss increases significantly above minimum,

    Args:
        patience: Stop if val MAE doesn't improve for this many epochs
        loss_patience: Stop if val loss > loss_threshold * min_val_loss for this
            many consecutive epochs
        loss_threshold: Loss degradation ratio threshold (e.g., 1.1 = 10% above
            minimum)
        warmup_epochs: Wait this many epochs before checking

    Returns:
        Function with signature (epoch, train_loss, val_loss, val_mae)
        that returns (should_stop, reason)
    """
    best_val_mae = float("inf")
    min_val_loss = float("inf")
    epochs_without_mae_improvement = 0
    epochs_with_loss_degradation = 0

    def early_stopping_fn(
        epoch: int, train_loss: float, val_loss: float, val_mae: float
    ) -> tuple[bool, str]:
        nonlocal best_val_mae, min_val_loss
        nonlocal epochs_without_mae_improvement, epochs_with_loss_degradation

        if epoch < warmup_epochs:
            best_val_mae = min(best_val_mae, val_mae)
            min_val_loss = min(min_val_loss, val_loss)
            return False, ""

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_mae_improvement = 0
        else:
            epochs_without_mae_improvement += 1

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_with_loss_degradation = 0
        elif val_loss > min_val_loss * loss_threshold:
            epochs_with_loss_degradation += 1
        else:
            epochs_with_loss_degradation = 0

        if epochs_without_mae_improvement >= patience:
            return (
                True,
                f"Val MAE has not improved for {patience} epochs "
                f"(best: ${best_val_mae:.2f}, current: ${val_mae:.2f})",
            )

        if epochs_with_loss_degradation >= loss_patience:
            loss_ratio = val_loss / min_val_loss
            return (
                True,
                f"Val loss degraded for {loss_patience} epochs "
                f"(min: {min_val_loss:.4f}, current: {val_loss:.4f}, ratio: {loss_ratio:.2f}x)",
            )

        return False, ""

    return early_stopping_fn


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
