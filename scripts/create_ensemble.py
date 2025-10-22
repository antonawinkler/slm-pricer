"""
Create ensemble models by combining predictions from multiple models.
"""

import json
import random
from typing import Any

import numpy as np
import numpy.typing as npt
from model_data_loader import ModelData, load_model_data
from scipy.optimize import OptimizeResult, minimize  # type: ignore[import-untyped]


def _get_random_weights(size: int) -> list[float]:
    """Generate random weights that sum to 1."""
    weights: list[float] = [random.random() for _ in range(size)]
    total: float = sum(weights)
    return [w / total for w in weights]


def _calculate_mean_error(
    models: list[ModelData],
    weights: npt.NDArray[np.float64] | list[float],
    start: int = 0,
    end: int | None = None,
) -> float:
    """
    Calculate mean absolute error for a weighted ensemble.

    Args:
        models: List of model data dictionaries
        weights: List of weights for each model
        start: Start index for error calculation
        end: End index for error calculation (None = use all)

    Returns:
        Mean absolute error
    """
    size: int = min(model["size"] for model in models)
    if end is None:
        end = size

    combined_guesses: list[float] = []
    for i in range(end):
        weighted_guess: float = sum(
            models[j]["guesses"][i] * weights[j] for j in range(len(models))
        )
        combined_guesses.append(weighted_guess)

    truths: list[float] = models[0]["truths"][:end]
    errors: list[float] = [
        abs(combined_guesses[i] - truths[i]) for i in range(start, end)
    ]

    return sum(errors) / len(errors)


class EnsembleModel(ModelData):
    """Extended model data with weights."""

    weights: list[float]


def combine_models(
    models: list[ModelData],
    weights: list[float] | None = None,
    title: str | None = None,
    optimize: bool = False,
    optimization_range: tuple[int, int] | None = None,
) -> EnsembleModel:
    """
    Combine multiple models into an ensemble by weighted averaging of predictions.

    Args:
        models: List of model data dictionaries (from load_model_data)
        weights: List of weights for each model (must sum to 1). If None, uses equal weights.
        title: Title for the ensemble model. If None, generates one from model names.
        optimize: If True, use scipy.optimize to find optimal weights
        optimization_range: Tuple (start, end) for optimization. If None, uses all data points.
                          Can be used to optimize on a subset and test on full dataset.

    Returns:
        Dictionary containing the ensemble model data with the same structure as load_model_data:
    """
    if not models:
        raise ValueError("Must provide at least one model")

    size: int = min(model["size"] for model in models)
    if size != models[0]["size"]:
        print(
            f"Note: Models have different sizes. Using common size of {size} data points."
        )

    if optimize:
        if weights is not None:
            print("Warning: Provided weights will be ignored when optimize=True")

        print(f"Optimizing weights for {len(models)} models...")

        opt_start: int
        opt_end: int
        opt_start, opt_end = 0, size
        if optimization_range is not None:
            opt_start, opt_end = optimization_range
            print(f"  Optimizing on data range [{opt_start}:{opt_end}]")

        initial_weights: list[float] = _get_random_weights(len(models))

        def objective(w: npt.NDArray[np.float64]) -> float:
            return _calculate_mean_error(models, w, opt_start, opt_end)

        result: OptimizeResult = minimize(
            objective, initial_weights, method="Nelder-Mead"
        )
        weights = result.x.tolist()

        print("  Optimization complete!")
        print(f"  Optimal weights: {[f'{w:.4f}' for w in weights]}")
        print(f"  Mean error on optimization range: {result.fun:.2f}")

    elif weights is None:
        weights = [1.0 / len(models)] * len(models)

    if len(weights) != len(models):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of models ({len(models)})"
        )

    if title is None:
        model_names: list[str] = [m["name"] for m in models]
        title = "-".join(model_names)

    combined_guesses: list[float] = []
    for i in range(size):
        weighted_guess: float = sum(
            models[j]["guesses"][i] * weights[j] for j in range(len(models))
        )
        combined_guesses.append(weighted_guess)

    truths: list[float] = models[0]["truths"][:size]
    errors: list[float] = [abs(combined_guesses[i] - truths[i]) for i in range(size)]
    cumulative_means: list[float] = [
        sum(errors[: i + 1]) / (i + 1) for i in range(len(errors))
    ]

    ensemble_result: EnsembleModel = {
        "name": title,
        "errors": errors,
        "guesses": combined_guesses,
        "truths": truths,
        "cumulative_means": cumulative_means,
        "size": size,
        "weights": weights,
    }
    return ensemble_result


def save_ensemble(ensemble_data: EnsembleModel, filename: str) -> None:
    """
    Save ensemble model data to a JSON file.

    Args:
        ensemble_data: Dictionary from combine_models()
        filename: Path to save the JSON file
    """
    # Calculate additional metrics for completeness
    errors: list[float] = ensemble_data["errors"]
    truths: list[float] = ensemble_data["truths"]
    guesses: list[float] = ensemble_data["guesses"]

    # Create colors based on error thresholds
    colors: list[str] = []
    for i in range(len(errors)):
        error: float = errors[i]
        truth: float = truths[i]
        if error < 40 or error / truth < 0.2:
            colors.append("green")
        elif error < 80 or error / truth < 0.4:
            colors.append("orange")
        else:
            colors.append("red")

    output_data: dict[str, Any] = {
        "title": ensemble_data["name"],
        "size": ensemble_data["size"],
        "guesses": guesses,
        "truths": truths,
        "errors": errors,
        "colors": colors,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Saved ensemble model to {filename}")


def main() -> None:
    models: list[ModelData] = load_model_data()
    for m in models:
        print(f"Loaded model: {m['name']} with {m['size']} data points")
    ed_donner_model: ModelData = next(
        m for m in models if m["name"] == "ed-donner|attention|epoch?|LoRA_R32"
    )
    antonawinkler_all_linear_model: ModelData = next(
        m for m in models if m["name"] == "all-linear|batchsize64|3epochs|LoRA_R16"
    )

    ensemble: EnsembleModel = combine_models(
        [ed_donner_model, antonawinkler_all_linear_model],
        optimize=False,
        title="donner-winkler-ensemble",
    )
    mean_error: float = float(np.mean(ensemble["errors"]))
    print(f"Mean absolute error of donner-winkler ensemble: {mean_error:.2f}")
    save_ensemble(ensemble, "data/_ed-donner-antonawinkler.json")

    # no optimize over all models starting from a random weight initialization
    ensemble = combine_models(models, optimize=True)
    mean_error = float(np.mean(ensemble["errors"]))
    print(f"Mean absolute error of optimized ensemble: {mean_error:.2f}")


if __name__ == "__main__":
    main()
