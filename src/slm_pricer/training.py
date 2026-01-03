"""
Training utilities for price prediction models.

This module contains functions for training and evaluating neural networks
on price prediction tasks.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import optuna
import torch
import torch.nn as nn
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
    prune_on_increase: bool = True,
    increase_threshold: float = 1.1,
    verbose: bool = True,
) -> float:
    """Train model with early stopping and return best validation MAE in dollars.

    Supports Optuna pruning via trial parameter. Prunes if median pruner triggers
    or if val MAE increases significantly after warmup (epoch 10+).
    """
    best_val_mae = float("inf")
    no_improve_count = 0
    prev_val_mae = float("inf")

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

        if trial is not None:
            trial.report(val_mae, epoch)

            if trial.should_prune():
                if verbose:
                    print(f"  ✂ Trial pruned at epoch {epoch + 1} (median pruner)")
                raise optuna.TrialPruned()

            if prune_on_increase and epoch >= 10:
                if val_mae > prev_val_mae * increase_threshold:
                    if verbose:
                        print(
                            f"  ✂ Trial pruned at epoch {epoch + 1}: "
                            f"Val MAE increased from ${prev_val_mae:.2f} to ${val_mae:.2f} "
                            f"(>{increase_threshold}x threshold)"
                        )
                    raise optuna.TrialPruned()

        prev_val_mae = val_mae

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
