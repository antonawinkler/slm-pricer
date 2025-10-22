"""
Visualize model results from the data folder.
Shows time series of cumulative mean error for all models.
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model_data_loader import ModelData, load_model_data


def parse_command_line_args() -> tuple[int, int | None]:
    """Parse command line arguments for start and end indices."""
    start_index: int = 0
    end_index: int | None = None

    if len(sys.argv) > 1:
        start_index = int(sys.argv[1])
        print(f"Using START_INDEX from command line: {start_index}")
    if len(sys.argv) > 2:
        end_index = int(sys.argv[2])
        print(f"Using END_INDEX from command line: {end_index}")

    return start_index, end_index


def get_range_suffix(start_index: int, end_index: int | None) -> str:
    """Generate filename suffix based on data range."""
    if start_index > 0 or end_index is not None:
        suffix: str = f"_{start_index}"
        if end_index is not None:
            suffix += f"-{end_index}"
        return suffix
    return ""


def plot_time_series(
    models_data: list[ModelData],
    start_index: int,
    end_index: int | None,
    range_suffix: str,
) -> None:
    """Create time series plot of cumulative mean error."""
    plt.figure(figsize=(14, 8))

    for model in models_data:
        # Plot all data (already sliced during loading)
        x_values = range(start_index, start_index + len(model["cumulative_means"]))
        y_values = model["cumulative_means"]
        plt.plot(x_values, y_values, label=model["name"], alpha=0.7, linewidth=1.5)

    plt.xlabel("Data Point Index", fontsize=12)
    plt.ylabel("Cumulative Mean Absolute Error ($)", fontsize=12)
    title: str = "Time Series of Cumulative Mean Error Across Models"
    if start_index > 0 or end_index is not None:
        range_str: str = (
            f"[{start_index}:{end_index if end_index is not None else 'end'}]"
        )
        title += f" (range {range_str})"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename: str = f"plots/model_comparison{range_suffix}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {filename}")


def plot_bar_chart(models_data: list[ModelData], range_suffix: str) -> None:
    """Create bar plot with mean absolute errors and standard error."""
    plt.figure(figsize=(12, 8))

    model_stats: list[dict[str, str | float]] = []
    for model in models_data:
        errors: list[float] = model["errors"]
        n: int = len(errors)
        mean_error: float = sum(errors) / n
        variance: float = sum((e - mean_error) ** 2 for e in errors) / n
        std_error: float = variance**0.5
        se_mean: float = std_error / (n**0.5)
        model_stats.append({"name": model["name"], "mean": mean_error, "se": se_mean})

    model_stats.sort(key=lambda x: float(x["mean"]))

    names: list[str] = [str(m["name"]) for m in model_stats]
    means: list[float] = [float(m["mean"]) for m in model_stats]
    ses: list[float] = [float(m["se"]) for m in model_stats]

    # Create bar plot with error bars
    x_pos = range(len(names))
    bars = plt.bar(
        x_pos,
        means,
        yerr=ses,
        capsize=5,
        alpha=0.7,
        color="steelblue",
        ecolor="black",
        error_kw={"alpha": 0.4},
    )

    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.xticks(x_pos, names, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Mean Absolute Error ($)", fontsize=12)
    plt.title(
        "Model Performance: Mean Absolute Error ± Standard Error",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    bar_filename: str = f"plots/model_comparison_bars{range_suffix}.png"
    plt.savefig(bar_filename, dpi=150, bbox_inches="tight")
    print(f"Saved bar plot to {bar_filename}")


def plot_relative_comparison(
    models_data: list[ModelData], baseline_name: str, range_suffix: str
) -> None:
    """Create relative comparison plot against baseline model with confidence intervals."""
    baseline_model: ModelData | None = None
    for model in models_data:
        if model["name"] == baseline_name:
            baseline_model = model
            break

    if baseline_model is None:
        print(
            f"Warning: Baseline model '{baseline_name}' not found. Skipping relative comparison plot."
        )
        return

    baseline_errors: list[float] = baseline_model["errors"]
    baseline_n: int = len(baseline_errors)
    baseline_mean: float = sum(baseline_errors) / baseline_n
    baseline_variance: float = (
        sum((e - baseline_mean) ** 2 for e in baseline_errors) / baseline_n
    )
    baseline_std: float = baseline_variance**0.5
    baseline_se: float = baseline_std / (baseline_n**0.5)

    relative_stats: list[dict[str, str | float]] = []
    for model in models_data:
        if model["name"] == baseline_name:
            continue

        errors: list[float] = model["errors"]
        n: int = len(errors)
        mean_error: float = sum(errors) / n
        variance: float = sum((e - mean_error) ** 2 for e in errors) / n
        std_error: float = variance**0.5
        se_mean: float = std_error / (n**0.5)

        diff: float = mean_error - baseline_mean
        combined_se: float = (se_mean**2 + baseline_se**2) ** 0.5

        relative_stats.append({"name": model["name"], "diff": diff, "se": combined_se})

    relative_stats.sort(key=lambda x: float(x["diff"]))

    names_rel: list[str] = [str(m["name"]) for m in relative_stats]
    diffs: list[float] = [float(m["diff"]) for m in relative_stats]
    ses_rel: list[float] = [float(m["se"]) for m in relative_stats]

    plt.figure(figsize=(12, 8))
    x_pos = range(len(names_rel))

    ci_levels: dict[str, tuple[float, str]] = {
        "80": (0.84, "orange"),
        "90": (1.28, "gold"),
        "95": (1.645, "blue"),
        "99": (2.33, "purple"),
    }

    colors: list[str] = ["green" if d < 0 else "red" for d in diffs]
    plt.bar(x_pos, diffs, alpha=0.7, color=colors)

    offset = 0.15
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for idx, (level, (z_score, color)) in enumerate(ci_levels.items()):
        for x, d, se in zip(x_pos, diffs, ses_rel):
            ci_low = d - z_score * se
            ci_high = d + z_score * se
            x_offset = x + offset * offsets[idx]
            plt.plot(
                [x_offset, x_offset],
                [ci_low, ci_high],
                color=color,
                linewidth=2,
                alpha=0.6,
                zorder=10,
            )

    plt.axhline(
        y=0,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"Baseline: {baseline_name}",
    )

    legend_elements = [
        Line2D([0], [0], color="orange", linewidth=2, label="80% CI"),
        Line2D([0], [0], color="gold", linewidth=2, label="90% CI"),
        Line2D([0], [0], color="blue", linewidth=2, label="95% CI"),
        Line2D([0], [0], color="purple", linewidth=2, label="99% CI"),
        Line2D(
            [0], [0], color="black", linewidth=1.5, label=f"Baseline: {baseline_name}"
        ),
    ]

    plt.xticks(x_pos, names_rel, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Difference in Mean Absolute Error ($)", fontsize=12)
    plt.title(
        f"Model Performance Relative to {baseline_name}", fontsize=14, fontweight="bold"
    )
    plt.legend(handles=legend_elements, loc="best")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    relative_filename: str = f"plots/model_comparison_relative{range_suffix}.png"
    plt.savefig(relative_filename, dpi=150, bbox_inches="tight")
    print(f"Saved relative comparison plot to {relative_filename}")


def plot_pairwise_comparison(
    models_data: list[ModelData], baseline_name: str, range_suffix: str
) -> None:
    """Create pairwise comparison plot with confidence intervals."""
    baseline_model: ModelData | None = None
    for model in models_data:
        if model["name"] == baseline_name:
            baseline_model = model
            break

    if baseline_model is None:
        return

    baseline_errors: list[float] = baseline_model["errors"]
    baseline_n: int = len(baseline_errors)

    pairwise_stats: list[dict[str, str | float]] = []
    for model in models_data:
        if model["name"] == baseline_name:
            continue

        errors: list[float] = model["errors"]
        n: int = min(len(errors), baseline_n)

        differences: list[float] = [errors[i] - baseline_errors[i] for i in range(n)]
        mean_diff: float = sum(differences) / n

        variance_diff: float = sum((d - mean_diff) ** 2 for d in differences) / n
        std_diff: float = variance_diff**0.5
        se_diff: float = std_diff / (n**0.5)

        pairwise_stats.append({"name": model["name"], "diff": mean_diff, "se": se_diff})

    pairwise_stats.sort(key=lambda x: float(x["diff"]))

    names_pair: list[str] = [str(m["name"]) for m in pairwise_stats]
    diffs_pair: list[float] = [float(m["diff"]) for m in pairwise_stats]
    ses_pair: list[float] = [float(m["se"]) for m in pairwise_stats]

    plt.figure(figsize=(12, 8))
    x_pos = range(len(names_pair))

    ci_levels: dict[str, tuple[float, str]] = {
        "80": (0.84, "orange"),
        "90": (1.28, "gold"),
        "95": (1.645, "blue"),
        "99": (2.33, "purple"),
    }

    colors_pair: list[str] = ["green" if d < 0 else "red" for d in diffs_pair]
    plt.bar(x_pos, diffs_pair, alpha=0.7, color=colors_pair)

    offset = 0.15
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for idx, (level, (z_score, color)) in enumerate(ci_levels.items()):
        for x, d, se in zip(x_pos, diffs_pair, ses_pair):
            ci_low = d - z_score * se
            ci_high = d + z_score * se
            x_offset = x + offset * offsets[idx]
            plt.plot(
                [x_offset, x_offset],
                [ci_low, ci_high],
                color=color,
                linewidth=2,
                alpha=0.6,
                zorder=10,
            )

    plt.axhline(
        y=0,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"Baseline: {baseline_name}",
    )

    legend_elements = [
        Line2D([0], [0], color="orange", linewidth=2, label="80% CI"),
        Line2D([0], [0], color="gold", linewidth=2, label="90% CI"),
        Line2D([0], [0], color="blue", linewidth=2, label="95% CI"),
        Line2D([0], [0], color="purple", linewidth=2, label="99% CI"),
        Line2D(
            [0], [0], color="black", linewidth=1.5, label=f"Baseline: {baseline_name}"
        ),
    ]

    plt.xticks(x_pos, names_pair, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Difference in Mean Absolute Error ($)", fontsize=12)
    plt.title(
        f"Model Performance Relative to {baseline_name}", fontsize=14, fontweight="bold"
    )
    plt.legend(handles=legend_elements, loc="best")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    pairwise_filename: str = f"plots/model_comparison_pairwise{range_suffix}.png"
    plt.savefig(pairwise_filename, dpi=150, bbox_inches="tight")
    print(f"Saved pairwise comparison plot to {pairwise_filename}")


def main() -> None:
    """Main function to orchestrate the visualization workflow."""
    start_index: int
    end_index: int | None
    start_index, end_index = parse_command_line_args()
    models_data: list[ModelData] = load_model_data(start_index, end_index)
    range_suffix: str = get_range_suffix(start_index, end_index)
    plot_time_series(models_data, start_index, end_index, range_suffix)
    plot_bar_chart(models_data, range_suffix)
    baseline_name: str = "ed-donner|attention|epoch?|LoRA_R32"
    plot_relative_comparison(models_data, baseline_name, range_suffix)
    plot_pairwise_comparison(models_data, baseline_name, range_suffix)


if __name__ == "__main__":
    main()
