"""
Training utilities for price prediction models.

This module contains functions for training and evaluating neural networks
on price prediction tasks.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import optuna
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import mean_absolute_error
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    convert_back_fn: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float]:
    """Evaluate model and return loss and MAE in original price space (dollars)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)

    # Convert predictions back to original price space for MAE calculation
    real_preds_transformed = np.array(all_preds)
    real_targets_transformed = np.array(all_targets)

    # Clip to avoid overflow in expm1
    real_preds_transformed = np.clip(real_preds_transformed, 0, 15)

    real_preds = convert_back_fn(real_preds_transformed)
    real_targets = convert_back_fn(real_targets_transformed)

    mae = mean_absolute_error(real_targets, real_preds)

    return {"loss": avg_loss, "mae": mae}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    convert_back_fn: Callable[[np.ndarray], np.ndarray],
    epochs: int = 100,
    grad_clip: float = 1.0,
    trial: Optional[optuna.Trial] = None,
    early_stopping_fn: Optional[
        Callable[[int, float, float, float], tuple[bool, str]]
    ] = None,
    verbose: bool = True,
) -> float:
    """Train model with early stopping and return best validation MAE in dollars.

    Supports Optuna pruning via trial parameter (median pruner only).
    Early stopping can be configured via early_stopping_fn.

    Args:
        early_stopping_fn: Optional function(epoch, train_loss, val_loss, val_mae)
            that returns (should_stop, reason). If returns True, training stops
            but trial completes normally with best MAE. Warmup handling is the
            responsibility of the function itself.
    """
    best_val_mae = float("inf")
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        val_metrics = evaluate_model(
            model, val_loader, criterion, device, convert_back_fn
        )
        val_loss = val_metrics["loss"]
        val_mae = val_metrics["mae"]

        if verbose:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: ${val_mae:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve_count = 0
            if verbose:
                print(f"  ✓ New best Val MAE: ${best_val_mae:.2f}")
        else:
            no_improve_count += 1

        # Check early stopping criterion
        if early_stopping_fn is not None:
            should_stop, reason = early_stopping_fn(
                epoch, avg_train_loss, val_loss, val_mae
            )
            if should_stop:
                if verbose:
                    print(
                        f"  ⏹ Early stopping at epoch {epoch + 1}: {reason}\n"
                        f"  Returning best Val MAE: ${best_val_mae:.2f}"
                    )
                break

        # Optuna pruning (median pruner only)
        if trial is not None:
            trial.report(val_mae, epoch)

            if trial.should_prune():
                if verbose:
                    print(f"  ✂ Trial pruned at epoch {epoch + 1} (median pruner)")
                raise optuna.TrialPruned()

    return best_val_mae


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: dict,
    filepath: Path | str,
    config: Optional[dict] = None,
) -> None:
    """Save model checkpoint with optimizer state, metrics, and optional config."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **metrics,
    }

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: Path | str,
    optimizer: Optional[Optimizer] = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Load checkpoint into model and optionally optimizer, return checkpoint dict."""
    checkpoint_raw = torch.load(filepath, map_location=device, weights_only=False)
    checkpoint: dict = checkpoint_raw
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def check_early_stopping(
    val_mae: float,
    best_val_mae: float,
    no_improve_count: int,
    patience: int,
) -> tuple[bool, str]:
    """Check if training should stop early based on validation MAE plateau.

    Args:
        val_mae: Current validation MAE (in dollars)
        best_val_mae: Best validation MAE seen so far
        no_improve_count: Number of epochs without improvement
        patience: Maximum epochs to wait without improvement

    Returns:
        (should_stop, reason) tuple
    """
    if no_improve_count >= patience:
        return (
            True,
            f"No improvement for {patience} epochs (best MAE: ${best_val_mae:.2f}, current: ${val_mae:.2f})",
        )

    return False, ""


def _get_model_registry() -> dict[str, type[nn.Module]]:
    """Return a mapping of model class names to their classes."""
    from slm_pricer.models import (
        MLP,
        PriceNetPostAct,
        PriceNetPreAct,
        ResidualNet,
        ResNetSimpleNN,
    )

    return {
        "ResidualNet": ResidualNet,
        "MLP": MLP,
        "PriceNetPostAct": PriceNetPostAct,
        "PriceNetPreAct": PriceNetPreAct,
        "ResNetSimpleNN": ResNetSimpleNN,
    }


def load_regression_head_from_hub(
    repo_id: str,
    filename: str = "best_model.pth",
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a pre-trained regression head from HuggingFace Hub.

    Supports all model classes in slm_pricer.models. Constructor kwargs
    are read from the checkpoint's "model_kwargs" field. Falls back to
    extracting input_dim/hidden_dim from config for old ResidualNet checkpoints.

    Returns:
        (model, metadata) where metadata contains config, metrics, epoch, phase.

    Example:
        >>> model, metadata = load_regression_head_from_hub(
        ...     "antonawinkler/slm-pricer-regressor-20260125"
        ... )
        >>> print(f"Pre-trained Val MAE: ${metadata['metrics']['val_mae']:.2f}")
    """
    registry = _get_model_registry()

    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_class_name = checkpoint.get("model_class", "ResidualNet")
    if model_class_name not in registry:
        raise ValueError(
            f"Unknown model class: {model_class_name}. "
            f"Available: {list(registry.keys())}"
        )

    model_cls = registry[model_class_name]

    # New format: model_kwargs stored explicitly in checkpoint
    if "model_kwargs" in checkpoint:
        model_kwargs = checkpoint["model_kwargs"]
    else:
        # Legacy format: extract from config
        model_kwargs = {}
        config = checkpoint.get("config", {})
        if "input_dim" in config:
            model_kwargs["input_dim"] = config["input_dim"]
        if "hidden_dim" in config:
            model_kwargs["hidden_dim"] = config["hidden_dim"]

    model = model_cls(**model_kwargs)

    if "regression_head_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["regression_head_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise KeyError("Checkpoint missing model state dict")

    model = model.to(device)

    metadata = {
        "config": checkpoint.get("training_config", checkpoint.get("config", {})),
        "metrics": checkpoint.get("metrics", {}),
        "epoch": checkpoint.get("epoch", None),
        "phase": checkpoint.get("phase", None),
    }

    for key in ["val_mae", "val_loss", "test_mae", "test_loss"]:
        if key in checkpoint:
            metadata["metrics"][key] = checkpoint[key]

    return model, metadata


def save_regression_head_to_hub(
    model: nn.Module,
    repo_id: str,
    metrics: dict[str, float],
    config: dict[str, Any],
    model_kwargs: dict[str, Any] | None = None,
    commit_message: str = "Upload regression head",
    private: bool = False,
    filename: str = "best_model.pth",
) -> None:
    """Save regression head to HuggingFace Hub with metadata.

    Args:
        model: Trained regression head model
        repo_id: HuggingFace repository ID (e.g., "antonawinkler/model-name")
        metrics: Performance metrics (val_mae, test_mae, etc.)
        config: Training configuration
        model_kwargs: Constructor kwargs needed to reconstruct the model.
            If None, load_regression_head_from_hub will fall back to
            extracting input_dim/hidden_dim from config (legacy behavior).
        commit_message: Git commit message
        private: Whether to make the repository private
        filename: Name for the checkpoint file

    Example:
        >>> save_regression_head_to_hub(
        ...     model=model,
        ...     repo_id="antonawinkler/slm-pricer-regressor",
        ...     metrics={"val_mae": 125.50, "test_mae": 130.20},
        ...     config={"learning_rate": 1e-4, "batch_size": 2048},
        ...     model_kwargs={"d_in": 3072, "d": 256, "n_layers": 1, "dropout": 0.1, "d_out": 1},
        ... )
    """
    import tempfile

    checkpoint: dict[str, Any] = {
        "regression_head_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
        "metrics": metrics,
        "training_config": config,
    }

    if model_kwargs is not None:
        checkpoint["model_kwargs"] = model_kwargs

    # Save checkpoint locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        torch.save(checkpoint, tmp_file.name)
        tmp_path = tmp_file.name

    try:
        # Upload to HuggingFace Hub
        api = HfApi()

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        except Exception:
            pass  # Repo might already exist

        # Upload file
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

        # Create a README with model card
        readme_content = f"""---
tags:
- regression
- price-prediction
- pytorch
metrics:
- mae
---

# {repo_id.split('/')[-1]}

Regression head trained for price prediction.

## Metrics

"""
        for key, value in metrics.items():
            if isinstance(value, float):
                readme_content += f"- **{key}**: {value:.2f}\n"
            else:
                readme_content += f"- **{key}**: {value}\n"

        readme_content += "\n## Configuration\n\n"
        for key, value in config.items():
            readme_content += f"- **{key}**: {value}\n"

        readme_content += f"\n## Usage\n\n```python\nfrom slm_pricer.training import load_regression_head_from_hub\n\nmodel, metadata = load_regression_head_from_hub(\"{repo_id}\")\nprint(f\"Val MAE: ${{metadata['metrics']['val_mae']:.2f}}\")\n```\n"

        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

    finally:
        # Clean up temp file
        import os

        os.unlink(tmp_path)
