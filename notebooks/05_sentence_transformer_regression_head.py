# %% [markdown]
# # Sentence Transformer with Regression Head for Price Prediction
#
# ## Overview
#
# This notebook demonstrates using **sentence transformer models** as embedding generators combined with a **regression head** to predict prices from product descriptions.
#
# Building on the approach from notebook 03 (fine-tuned Llama with regression head), this notebook:
# 1. Uses sentence transformer models (configurable) to generate embeddings
# 2. Trains a regression head on top of those embeddings to predict prices
# 3. Supports multiple embedding models including custom fine-tuned ones
#
# ## Supported Models
#
# - `antonawinkler/slm-pricer-llama-3.2-3b` - Custom fine-tuned Llama 3.2 3B
# - `nvidia/llama-embed-nemotron-8b` - NVIDIA's Llama embedding model
# - Any sentence-transformers compatible model
#
# ## Key Advantages
#
# 1. **Faster inference**: Sentence transformers are optimized for embedding generation
# 2. **Better embeddings**: Models trained with contrastive learning for semantic similarity
# 3. **Flexible architecture**: Can swap embedding models easily
# 4. **Continuous predictions**: Regression head predicts any price value


# %%
!git clone https://github.com/antonawinkler/slm-pricer.git

%cd slm-pricer
!uv pip install .
%cd ..


# %%
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from google.colab import drive, userdata
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from slm_pricer.data import (
    PriceDataset,
    load_data_from_hf,
    load_embeddings,
    save_embeddings,
)
from slm_pricer.models import PriceRegressor
from slm_pricer.training import evaluate_model, train_model
from slm_pricer.utils import (
    convert_back_y,
    create_early_stopping_fn,
    transformed_prices,
)


# %%
@dataclass
class Config:
    """Configuration for sentence transformer with regression head."""

    # Embedding model configuration
    embedding_model_name: str = "antonawinkler/slm-pricer-llama-3.2-3b"  # or "nvidia/llama-embed-nemotron-8b"
    embedding_batch_size: int = 64
    max_seq_length: int = 128
    normalize_embeddings: bool = True

    # Data
    data_percent: int = 100

    # Data preprocessing
    y_transform: str = "log"  # Transform for target prices: "log", "unit", or None

    # Embedding caching
    use_cached_embeddings: bool = True
    cache_path: Path = Path("/content/drive/MyDrive/Pricer_Embeddings")
    cache_filename_root: str = "sentence_transformer_"  # Will append model name

    # Optuna settings
    run_optuna: bool = True
    n_trials: int = 50
    optuna_epochs: int = 50

    # Final training
    run_final_training: bool = True
    use_best_trial_params: bool = False

    # Training hyperparameters (used if not running Optuna)
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    hidden_dim1: int = 1024
    hidden_dim2: int = 256
    dropout: float = 0.1
    epochs: int = 100
    grad_clip: float = 1.0

    # Reproducibility
    seed: int = 42


CONFIG = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(CONFIG.seed)
np.random.seed(CONFIG.seed)


# %% [markdown]
# ## Import Data


# %%
df_train = load_data_from_hf(split="train", percent=CONFIG.data_percent)
df_val = load_data_from_hf(split="val", percent=100)
df_test = load_data_from_hf(split="test", percent=100)

print(
    f"Train: {len(df_train):,} samples | Val: {len(df_val):,} samples | Test: {len(df_test):,} samples"
)

y_train = transformed_prices(
    df_train["completion"].to_numpy(dtype="float32"), CONFIG.y_transform
)
y_val = transformed_prices(
    df_val["completion"].to_numpy(dtype="float32"), CONFIG.y_transform
)
y_test = transformed_prices(
    df_test["completion"].to_numpy(dtype="float32"), CONFIG.y_transform
)

print(f"Target ranges ({CONFIG.y_transform}-transformed):")
print(f"  Train: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"  Val:   [{y_val.min():.2f}, {y_val.max():.2f}]")
print(f"  Test:  [{y_test.min():.2f}, {y_test.max():.2f}]")


# %% [markdown]
# ## Generate Embeddings


# %%
if not CONFIG.use_cached_embeddings:
    print(f"Loading sentence transformer model: {CONFIG.embedding_model_name}...")

    model = SentenceTransformer(
        CONFIG.embedding_model_name,
        device=device,
        trust_remote_code=True,
    )

    if hasattr(model, "max_seq_length"):
        model.max_seq_length = CONFIG.max_seq_length

    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
else:
    print("Using cached embeddings")


# %%
def get_sentence_transformer_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Extract embeddings from sentence transformer model.

    Returns array of shape (N, embedding_dim).
    """
    print(f"Generating embeddings (batch size: {batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return embeddings


if not CONFIG.use_cached_embeddings:
    embeddings = {}
    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"\nProcessing {split} split...")
        embeddings[split] = get_sentence_transformer_embeddings(
            model,
            df["prompt"].tolist(),
            batch_size=CONFIG.embedding_batch_size,
            normalize=CONFIG.normalize_embeddings,
        )
        print(f"Shape: {embeddings[split].shape}")

    X_train = embeddings["train"]
    X_val = embeddings["val"]
    X_test = embeddings["test"]
else:
    print("Using cached embeddings")


# %% [markdown]
# ## Save Embeddings to Google Drive


# %%
if not CONFIG.use_cached_embeddings:
    drive.mount("/content/drive")

    # Create model-specific cache filename
    model_short_name = CONFIG.embedding_model_name.replace("/", "_").replace("-", "_")
    cache_filename = f"{CONFIG.cache_filename_root}{model_short_name}_"

    print(f"Saving embeddings to {CONFIG.cache_path}...")
    save_embeddings(
        embeddings={"train": X_train, "val": X_val, "test": X_test},
        cache_path=CONFIG.cache_path,
        cache_filename_root=cache_filename,
    )
else:
    print("Using cached embeddings")


# %% [markdown]
# ## Load Cached Embeddings


# %%
if CONFIG.use_cached_embeddings:
    drive.mount("/content/drive")

    model_short_name = CONFIG.embedding_model_name.replace("/", "_").replace("-", "_")
    cache_filename = f"{CONFIG.cache_filename_root}{model_short_name}_"

    print(f"Loading cached embeddings from {CONFIG.cache_path}...")

    embeddings = load_embeddings(
        cache_path=CONFIG.cache_path,
        cache_filename_root=cache_filename,
        splits=["train", "val", "test"],
    )

    X_train = embeddings["train"][: len(df_train)]
    X_val = embeddings["val"]
    X_test = embeddings["test"]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
else:
    print("Using generated embeddings")


# %% [markdown]
# ## Optuna Hyperparameter Optimization


# %%
def objective(trial: optuna.Trial) -> float:
    """Optuna objective optimizing regression head architecture and hyperparameters.

    Returns best validation MAE in dollars (lower is better).
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_dim1 = trial.suggest_int("hidden_dim1", 512, 2048, step=256)
    hidden_dim2 = trial.suggest_int("hidden_dim2", 128, 512, step=128)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 128, 512, step=128)

    print(f"\n{'=' * 80}")
    print(f"Trial {trial.number}: Testing hyperparameters:")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"  Hidden Dim 1: {hidden_dim1}")
    print(f"  Hidden Dim 2: {hidden_dim2}")
    print(f"  Dropout: {dropout:.3f}")
    print(f"  Weight Decay: {weight_decay:.2e}")
    print(f"  Batch Size: {batch_size}")
    print(f"{'=' * 80}\n")

    input_dim = X_train.shape[1]

    model = PriceRegressor(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout,
    ).to(device)

    train_loader = DataLoader(
        PriceDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PriceDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=CONFIG.optuna_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    early_stop_fn = create_early_stopping_fn(
        patience=10,
        loss_patience=3,
        loss_threshold=1.5,
        warmup_epochs=10,
    )

    best_val_mae = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        convert_back_fn=convert_back_y,
        epochs=CONFIG.optuna_epochs,
        grad_clip=CONFIG.grad_clip,
        trial=trial,
        early_stopping_fn=early_stop_fn,
        verbose=False,
    )

    print(f"\nTrial {trial.number} completed: Val MAE = ${best_val_mae:.2f}\n")

    return best_val_mae


if CONFIG.run_optuna:
    print(f"\n{'=' * 80}")
    print(f"Starting Optuna optimization with {CONFIG.n_trials} trials...")
    print(f"{'=' * 80}\n")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
        study_name="sentence_transformer_regression_head_optimization",
    )

    study.optimize(objective, n_trials=CONFIG.n_trials, show_progress_bar=True)

    print(f"\n{'=' * 80}")
    print("Optimization completed!")
    print(f"{'=' * 80}\n")
else:
    study = None


# %% [markdown]
# ## Optuna Results Analysis


# %%
if CONFIG.run_optuna and study is not None:
    print("\n" + "=" * 80)
    print("TOP 5 TRIALS")
    print("=" * 80)

    top_trials = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float("inf")
    )[:5]

    for rank, trial in enumerate(top_trials, 1):
        print(f"\n{'─' * 80}")
        print(f"Rank {rank}: Trial #{trial.number}")
        print(f"{'─' * 80}")
        print(f"Validation MAE: ${trial.value:.2f}")
        print("\nHyperparameters:")

        params = trial.params
        lr = params["learning_rate"]
        wd = params["weight_decay"]
        dropout = params["dropout"]
        hidden_dim1 = params["hidden_dim1"]
        hidden_dim2 = params["hidden_dim2"]
        bs = params["batch_size"]

        print(f"  learning_rate:  {lr:.2e}")
        print(f"  weight_decay:   {wd:.2e}")
        print(f"  dropout:        {dropout:.3f}")
        print(f"  hidden_dim1:    {hidden_dim1}")
        print(f"  hidden_dim2:    {hidden_dim2}")
        print(f"  batch_size:     {bs}")

    print("\n" + "=" * 80)
    print(
        f"\nBest Trial: #{study.best_trial.number} with Val MAE: ${study.best_value:.2f}"
    )
    print("=" * 80)

    print("\n" + "=" * 80)
    print("PARAMETER IMPORTANCE")
    print("=" * 80 + "\n")

    importance = optuna.importance.get_param_importances(study)
    for param, imp in importance.items():
        print(f"  {param}: {imp:.4f}")


# %% [markdown]
# ## Train Final Model


# %%
print("\n" + "=" * 80)
print("TRAINING FINAL MODEL")
print("=" * 80 + "\n")

if CONFIG.run_optuna and CONFIG.use_best_trial_params and study is not None:
    print("Using best trial hyperparameters from Optuna\n")
    best_params = study.best_params
    learning_rate = best_params["learning_rate"]
    hidden_dim1 = best_params["hidden_dim1"]
    hidden_dim2 = best_params["hidden_dim2"]
    dropout = best_params["dropout"]
    weight_decay = best_params["weight_decay"]
    batch_size = best_params["batch_size"]
else:
    print("Using manual configuration hyperparameters\n")
    learning_rate = CONFIG.learning_rate
    hidden_dim1 = CONFIG.hidden_dim1
    hidden_dim2 = CONFIG.hidden_dim2
    dropout = CONFIG.dropout
    weight_decay = CONFIG.weight_decay
    batch_size = CONFIG.batch_size

print("Hyperparameters:")
print(f"  Learning Rate: {learning_rate:.2e}")
print(f"  Hidden Dim 1: {hidden_dim1}")
print(f"  Hidden Dim 2: {hidden_dim2}")
print(f"  Dropout: {dropout:.3f}")
print(f"  Weight Decay: {weight_decay:.2e}")
print(f"  Batch Size: {batch_size}")
print()

input_dim = X_train.shape[1]

final_model = PriceRegressor(
    input_dim=input_dim,
    hidden_dim1=hidden_dim1,
    hidden_dim2=hidden_dim2,
    dropout=dropout,
).to(device)

train_loader = DataLoader(
    PriceDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    PriceDataset(X_val, y_val),
    batch_size=batch_size,
    shuffle=False,
)
test_loader = DataLoader(
    PriceDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False,
)

criterion = nn.MSELoss()
optimizer = optim.AdamW(
    final_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    epochs=CONFIG.epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
)

early_stop_fn = create_early_stopping_fn(
    patience=15,
    loss_patience=3,
    loss_threshold=1.5,
    warmup_epochs=10,
)

best_val_mae = train_model(
    model=final_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    convert_back_fn=convert_back_y,
    epochs=CONFIG.epochs,
    grad_clip=CONFIG.grad_clip,
    trial=None,
    early_stopping_fn=early_stop_fn,
    verbose=True,
)

print(f"\nFinal model Val MAE: ${best_val_mae:.2f}")


# %% [markdown]
# ## Test Evaluation


# %%
print("\n" + "=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80 + "\n")

test_metrics = evaluate_model(
    final_model, test_loader, criterion, device, convert_back_y
)

print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test MAE: ${test_metrics['mae']:.2f}")

print("\n" + "=" * 80)


# %% [markdown]
# ## Save Results and Model


# %%
results = {
    "timestamp": datetime.now().isoformat(),
    "test_mae": test_metrics["mae"],
    "test_loss": test_metrics["loss"],
    "val_mae": best_val_mae,
    "config": {
        "embedding_model_name": CONFIG.embedding_model_name,
        "max_seq_length": CONFIG.max_seq_length,
        "normalize_embeddings": CONFIG.normalize_embeddings,
        "data_percent": CONFIG.data_percent,
        "y_transform": CONFIG.y_transform,
        "epochs": CONFIG.epochs,
        "seed": CONFIG.seed,
    },
    "hyperparameters": {
        "learning_rate": learning_rate,
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    },
}

if CONFIG.run_optuna and study is not None:
    results["optuna"] = {
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_val_mae": study.best_value,
    }

model_short_name = CONFIG.embedding_model_name.replace("/", "_").replace("-", "_")
results_path = CONFIG.cache_path / f"sentence_transformer_{model_short_name}_results.json"

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {results_path}")

model_path = CONFIG.cache_path / f"sentence_transformer_{model_short_name}_model.pth"
checkpoint = {
    "model_state_dict": final_model.state_dict(),
    "model_config": {
        "input_dim": input_dim,
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2,
        "dropout": dropout,
    },
    "training_config": {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": CONFIG.epochs,
        "grad_clip": CONFIG.grad_clip,
    },
    "metrics": {
        "val_mae": best_val_mae,
        "test_mae": test_metrics["mae"],
        "test_loss": test_metrics["loss"],
    },
    "embedding_config": {
        "embedding_model_name": CONFIG.embedding_model_name,
        "max_seq_length": CONFIG.max_seq_length,
        "normalize_embeddings": CONFIG.normalize_embeddings,
    },
    "data_config": {
        "data_percent": CONFIG.data_percent,
        "y_transform": CONFIG.y_transform,
    },
    "seed": CONFIG.seed,
    "timestamp": datetime.now().isoformat(),
}

torch.save(checkpoint, model_path)
print(f"Model saved to {model_path}")


# %% [markdown]
# ## Test Predictions
#
# Generate predictions on example products to verify the model works correctly.


# %%
def predict_price(
    description: str,
    embedding_model: SentenceTransformer,
    regression_model: nn.Module,
    normalize: bool = True,
) -> float:
    """Predict price from product description.

    Returns predicted price in dollars.
    """
    embedding = embedding_model.encode(
        [description],
        normalize_embeddings=normalize,
        convert_to_numpy=False,
    )

    with torch.no_grad():
        embedding = embedding.to(device)
        log_price = regression_model(embedding).item()
        price = np.exp(log_price)

    return price


# Load model for predictions if using cached embeddings
if CONFIG.use_cached_embeddings:
    print(f"Loading sentence transformer model for predictions: {CONFIG.embedding_model_name}...")
    model = SentenceTransformer(
        CONFIG.embedding_model_name,
        device=device,
        trust_remote_code=True,
    )
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = CONFIG.max_seq_length

test_products = [
    "Apple AirPods Pro (2nd Generation) with MagSafe Charging Case",
    "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
    "USB-C Cable 6ft Fast Charging Cord",
    "Samsung Galaxy S24 Ultra 256GB Smartphone",
    "Paper Mate Ballpoint Pens, Medium Point, Black, 12 Pack",
]

print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80 + "\n")

print(f"{'Product Description':<60} {'Predicted Price':>15}")
print("=" * 80)

for product in test_products:
    predicted_price = predict_price(
        product,
        model,
        final_model,
        normalize=CONFIG.normalize_embeddings,
    )
    display_desc = product[:57] + "..." if len(product) > 60 else product
    print(f"{display_desc:<60} ${predicted_price:>14,.2f}")

print("\n" + "=" * 80)
