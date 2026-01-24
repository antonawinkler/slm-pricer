# %% [markdown]
# # Fine-Tuned Llama 3.1 with Regression Head for Price Prediction
#
# ## Overview
#
# In Ed Donner's LLM Engineering course, a Llama model was fine-tuned to predict prices based on Amazon product descriptions. The original approach used the most likely token (a number between 1 and 999) as the price prediction.
#
# This notebook demonstrates an improved approach: extracting the final hidden layer from the fine-tuned Llama model and adding a **regression head** (value head) to predict prices more accurately.
#
# ## Key Improvements
#
# 1. **Better accuracy**: Regression head can learn non-linear price transformations
# 2. **Flexible architecture**: Can optimize the regression head architecture independently


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
from huggingface_hub import HfApi
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
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
    """Configuration for fine-tuned Llama with regression head."""

    # Model configuration
    base_model_name: str = "meta-llama/Llama-3.1-8B"
    finetuned_model_name: str = "ed-donner/price-2025-11-28_18.47.07"
    finetuned_model_revision: str = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"

    # Embedding generation
    n_layers: int = 1  # Number of final Llama layers to use for embeddings
    embedding_batch_size: int = 32
    max_seq_length: int = 128

    # Data
    data_percent: int = 100  # Percentage of training data to use (1-100)

    # Data preprocessing
    y_transform: str = "log"  # Transform for target prices: "log", "unit", or None

    # Embedding caching
    use_cached_embeddings: bool = True  # Load pre-computed embeddings if available
    cache_path: Path = Path("/content/drive/MyDrive/Pricer_Embeddings")
    cache_filename_root: str = "llama_fine_tuned_"

    # Optuna settings
    run_optuna: bool = True  # Run hyperparameter optimization
    n_trials: int = 50
    optuna_epochs: int = 50  # Epochs per trial

    # Final training
    run_final_training: bool = True
    use_best_trial_params: bool = (
        False  # Use Optuna best params, or manual config above
    )

    # Training hyperparameters (used for final training if not running Optuna)
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

# Set device and seed
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
def load_fine_tuned_model_and_tokenizer(
    base_model_name: str,
    fine_tuned_model_name: str,
    revision: str | None = None,
    quantized: bool = False,
):
    """Load fine-tuned Llama model with optional quantization.

    Args:
        base_model_name: HuggingFace model name for base model
        fine_tuned_model_name: HuggingFace model name for fine-tuned adapter
        revision: Optional git revision for fine-tuned model
        quantized: If True, load in 4-bit quantization (slower for embeddings)

    Returns:
        Tuple of (model, tokenizer)

    Note: quantized=False is recommended for embedding generation (better performance).
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if quantized:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_name)
    fine_tuned_model = fine_tuned_model.merge_and_unload()

    return fine_tuned_model, tokenizer


if not CONFIG.use_cached_embeddings:
    print(
        "Loading fine-tuned Llama model (non-quantized for better embedding performance)..."
    )

    model, tokenizer = load_fine_tuned_model_and_tokenizer(
        base_model_name=CONFIG.base_model_name,
        fine_tuned_model_name=CONFIG.finetuned_model_name,
        revision=CONFIG.finetuned_model_revision,
        quantized=False,
    )

    model.eval()
else:
    print("Using cached embeddings")


# %%
def get_llama_embeddings(
    model,
    tokenizer,
    texts: list[str],
    n_layers: int = 1,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """Extract embeddings from final hidden layers of Llama model.

    Returns array of shape (N, n_layers, hidden_dim).
    """
    all_embeddings = []
    model.eval()

    print(f"Generating embeddings (batch size: {batch_size})...")

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            selected_layers = outputs.hidden_states[-n_layers:]

            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1

            layer_vectors = []
            for layer in selected_layers:
                vecs = layer[torch.arange(layer.shape[0]), last_token_indices]
                layer_vectors.append(vecs)

            batch_stack = torch.stack(layer_vectors, dim=1)
            all_embeddings.append(batch_stack.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


if not CONFIG.use_cached_embeddings:
    embeddings = {}
    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"\nProcessing {split} split...")
        embeddings[split] = get_llama_embeddings(
            model,
            tokenizer,
            df["prompt"].tolist(),
            n_layers=CONFIG.n_layers,
            batch_size=CONFIG.embedding_batch_size,
            max_length=CONFIG.max_seq_length,
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

    print(f"Saving embeddings to {CONFIG.cache_path}...")
    save_embeddings(
        embeddings={"train": X_train, "val": X_val, "test": X_test},
        cache_path=CONFIG.cache_path,
        cache_filename_root=CONFIG.cache_filename_root,
    )
else:
    print("Using cached embeddings")


# %% [markdown]
# ## Load Cached Embeddings


# %%
if CONFIG.use_cached_embeddings:
    drive.mount("/content/drive")

    print(f"Loading cached embeddings from {CONFIG.cache_path}...")

    embeddings = load_embeddings(
        cache_path=CONFIG.cache_path,
        cache_filename_root=CONFIG.cache_filename_root,
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

    if X_train.ndim == 3:
        input_dim = X_train.shape[1] * X_train.shape[2]
    else:
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
        study_name="llama_regression_head_optimization",
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
    print("\\n" + "=" * 80)
    print("TOP 5 TRIALS")
    print("=" * 80)

    top_trials = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float("inf")
    )[:5]

    for rank, trial in enumerate(top_trials, 1):
        print(f"\\n{'─' * 80}")
        print(f"Rank {rank}: Trial #{trial.number}")
        print(f"{'─' * 80}")
        print(f"Validation MAE: ${trial.value:.2f}")
        print("\\nHyperparameters:")

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

    print("\\n" + "=" * 80)
    print(
        f"\\nBest Trial: #{study.best_trial.number} with Val MAE: ${study.best_value:.2f}"
    )
    print("=" * 80)

    print("\\n" + "=" * 80)
    print("PARAMETER IMPORTANCE")
    print("=" * 80 + "\\n")

    importance = optuna.importance.get_param_importances(study)
    for param, imp in importance.items():
        print(f"  {param}: {imp:.4f}")


# %%
print("\\n" + "=" * 80)
print("TRAINING FINAL MODEL")
print("=" * 80 + "\\n")

if CONFIG.run_optuna and CONFIG.use_best_trial_params and study is not None:
    print("Using best trial hyperparameters from Optuna\\n")
    best_params = study.best_params
    learning_rate = best_params["learning_rate"]
    hidden_dim1 = best_params["hidden_dim1"]
    hidden_dim2 = best_params["hidden_dim2"]
    dropout = best_params["dropout"]
    weight_decay = best_params["weight_decay"]
    batch_size = best_params["batch_size"]
else:
    print("Using manual configuration hyperparameters\\n")
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

if X_train.ndim == 3:
    input_dim = X_train.shape[1] * X_train.shape[2]
else:
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

print(f"\\nFinal model Val MAE: ${best_val_mae:.2f}")


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


# %%
results = {
    "timestamp": datetime.now().isoformat(),
    "test_mae": test_metrics["mae"],
    "test_loss": test_metrics["loss"],
    "val_mae": best_val_mae,
    "config": {
        "base_model_name": CONFIG.base_model_name,
        "finetuned_model_name": CONFIG.finetuned_model_name,
        "n_layers": CONFIG.n_layers,
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

results_path = CONFIG.cache_path / "llama_regression_head_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {results_path}")

model_path = CONFIG.cache_path / "llama_regression_head_model.pth"
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
        "base_model_name": CONFIG.base_model_name,
        "finetuned_model_name": CONFIG.finetuned_model_name,
        "finetuned_model_revision": CONFIG.finetuned_model_revision,
        "n_layers": CONFIG.n_layers,
        "max_seq_length": CONFIG.max_seq_length,
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
# ## Upload Model to HuggingFace Hub
#
# This section uploads the trained regression head model to HuggingFace for sharing with the community.


# %%
from huggingface_hub import HfApi, create_repo
import shutil
import tempfile

HF_REPO_NAME = "your-username/llama-pricer-regression-head"
PRIVATE_REPO = False

print("\n" + "=" * 80)
print("UPLOADING MODEL TO HUGGINGFACE HUB")
print("=" * 80 + "\n")

try:
    hf_token = userdata.get("HF_TOKEN")
except Exception as e:
    print(f"⚠ Could not retrieve HF_TOKEN from secrets: {e}")
    print("Please set HF_TOKEN in Colab secrets or provide it manually")
    hf_token = None

if hf_token:
    with tempfile.TemporaryDirectory() as upload_dir:
        upload_path = Path(upload_dir)

        print("Preparing files for upload...")

        shutil.copy(model_path, upload_path / "model.pth")
        shutil.copy(results_path, upload_path / "results.json")

        readme_content = f"""---
license: apache-2.0
base_model: {CONFIG.finetuned_model_name}
tags:
- regression
- price-prediction
- llama
- fine-tuning
datasets:
- ed-donner/pricer
metrics:
- mae
library_name: pytorch
---

# Llama 3.1 Regression Head for Price Prediction

## Model Description

This model improves upon the approach taught in [Ed Donner's LLM Engineering Udemy course](https://www.udemy.com/course/llm-engineering/). 

In the course, a Llama 3.1 model was fine-tuned to predict product prices from Amazon descriptions by generating the most likely token (a number between 1 and 999). This approach limits predictions to discrete integer values.

**This model demonstrates a better approach**: Extract embeddings from the fine-tuned Llama model and add a **regression head** (value head) that predicts continuous price values.

## Key Improvements

1. **Continuous predictions**: Predicts any price value, not just integers 1-999
2. **Better accuracy**: Test MAE of ${test_metrics["mae"]:.2f} (vs. higher MAE from token-based approach)
3. **Flexible architecture**: Regression head can be optimized independently from the base model
4. **Log-space transformation**: Better handles wide price ranges ($0.01 to $10,000+)

## Model Architecture

**Base Model**: Fine-tuned Llama 3.1 8B ([{CONFIG.finetuned_model_name}]({CONFIG.finetuned_model_name}))

**Regression Head**:
- Input: Final hidden layer from Llama ({input_dim} dimensions)
- Hidden Layer 1: {hidden_dim1} units + BatchNorm + ReLU + Dropout({dropout})
- Hidden Layer 2: {hidden_dim2} units + BatchNorm + ReLU + Dropout({dropout})
- Output: 1 unit (price prediction in log-space)

**Total Parameters**: ~{sum(p.numel() for p in final_model.parameters()):,} (regression head only)

## Training Details

### Dataset
- Source: [{CONFIG.finetuned_model_name.split("/")[0]}/pricer](https://huggingface.co/datasets/ed-donner/pricer)
- Training samples: {len(df_train):,}
- Validation samples: {len(df_val):,}
- Test samples: {len(df_test):,}

### Hyperparameters
- Optimizer: AdamW
- Learning Rate: {learning_rate:.2e}
- Weight Decay: {weight_decay:.2e}
- Batch Size: {batch_size}
- Epochs: {CONFIG.epochs} (with early stopping)
- Scheduler: OneCycleLR (10% warmup, cosine annealing)
- Gradient Clipping: {CONFIG.grad_clip}
- Price Transform: {CONFIG.y_transform}-space
{"- Hyperparameter Tuning: Optuna (" + str(len(study.trials)) + " trials)" if CONFIG.run_optuna and study else ""}

### Training Results
- **Validation MAE**: ${best_val_mae:.2f}
- **Test MAE**: ${test_metrics["mae"]:.2f}
- **Test Loss**: {test_metrics["loss"]:.4f}

## Usage

### Installation

```bash
pip install torch transformers peft huggingface_hub
```

### Quick Start

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load the fine-tuned Llama model
base_model_name = "{CONFIG.base_model_name}"
finetuned_model_name = "{CONFIG.finetuned_model_name}"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)
llama_model = PeftModel.from_pretrained(base_model, finetuned_model_name)
llama_model = llama_model.merge_and_unload()
llama_model.eval()

# 2. Load the regression head
checkpoint = torch.load("model.pth", map_location="cpu")

class PriceRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim1),
            torch.nn.BatchNorm1d(hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.BatchNorm1d(hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim2, 1),
        )
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return self.network(x).squeeze(-1)

regression_head = PriceRegressor(**checkpoint["model_config"])
regression_head.load_state_dict(checkpoint["model_state_dict"])
regression_head.eval()

# 3. Predict price for a product description
def predict_price(description: str) -> float:
    inputs = tokenizer(
        description,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length={CONFIG.max_seq_length},
    ).to(llama_model.device)
    
    with torch.no_grad():
        outputs = llama_model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        last_token_idx = inputs["attention_mask"].sum(dim=1) - 1
        embedding = last_hidden_state[0, last_token_idx].unsqueeze(0).cpu()
        
        log_price = regression_head(embedding).item()
        price = np.exp(log_price)
        
    return price

# Example prediction
description = "Apple AirPods Pro (2nd Generation) with MagSafe Charging Case"
predicted_price = predict_price(description)
print(f"Predicted price: ${{{predicted_price:.2f}}}")
```

## Model Card Contact

For questions or feedback, please open an issue on the [GitHub repository](https://github.com/antonawinkler/slm-pricer).

## Citation

If you use this model, please cite Ed Donner's original course and dataset:

```bibtex
@misc{{donner2024llmengineering,
  author = {{Ed Donner}},
  title = {{LLM Engineering: Master AI & Large Language Models}},
  year = {{2024}},
  publisher = {{Udemy}},
  url = {{https://www.udemy.com/course/llm-engineering/}}
}}
```

## License

Apache 2.0 (same as base Llama model)
"""

        with open(upload_path / "README.md", "w") as f:
            f.write(readme_content)

        config_content = {
            "model_type": "regression_head",
            "architecture": "PriceRegressor",
            "base_model": CONFIG.finetuned_model_name,
            "base_model_revision": CONFIG.finetuned_model_revision,
            "framework": "pytorch",
            **checkpoint["model_config"],
            **checkpoint["embedding_config"],
            "metrics": checkpoint["metrics"],
        }

        with open(upload_path / "config.json", "w") as f:
            json.dump(config_content, f, indent=2)

        print(f"Files prepared:")
        print(
            f"  - model.pth ({(upload_path / 'model.pth').stat().st_size / 1024 / 1024:.2f} MB)"
        )
        print(f"  - results.json")
        print(f"  - README.md")
        print(f"  - config.json")

        print(f"\nUploading to HuggingFace: {HF_REPO_NAME}...")

        api = HfApi()

        try:
            create_repo(
                repo_id=HF_REPO_NAME,
                private=PRIVATE_REPO,
                token=hf_token,
                exist_ok=True,
            )
        except Exception as e:
            print(f"⚠ Warning creating repository: {e}")

        try:
            api.upload_folder(
                folder_path=upload_path,
                repo_id=HF_REPO_NAME,
                token=hf_token,
                commit_message=f"Upload regression head model (Test MAE: ${test_metrics['mae']:.2f})",
            )

            print(f"\n{'=' * 80}")
            print("✅ UPLOAD SUCCESSFUL!")
            print(f"{'=' * 80}\n")
            print(f"Model available at: https://huggingface.co/{HF_REPO_NAME}")
            print(f"\nModel Performance:")
            print(f"  - Test MAE: ${test_metrics['mae']:.2f}")
            print(f"  - Validation MAE: ${best_val_mae:.2f}")

        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
            print("\nPlease check:")
            print("  1. HF_TOKEN is set correctly in Colab secrets")
            print("  2. Repository name format: 'username/repo-name'")
            print("  3. You have write permissions to the repository")
else:
    print("\n⚠ Skipping upload (HF_TOKEN not available)")
    print("\nTo upload the model:")
    print("  1. Set HF_TOKEN in Colab secrets (🔑 icon in left sidebar)")
    print("  2. Update HF_REPO_NAME to your HuggingFace username/repo")
    print("  3. Re-run this cell")


# %% [markdown]
# ## Download and Use Model from HuggingFace
#
# This section demonstrates how to download and use the trained model from HuggingFace Hub.


# %%
from huggingface_hub import hf_hub_download
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DOWNLOAD_REPO_NAME = "your-username/llama-pricer-regression-head"

print("\n" + "=" * 80)
print("DOWNLOADING MODEL FROM HUGGINGFACE HUB")
print("=" * 80 + "\n")

print(f"Downloading from: {DOWNLOAD_REPO_NAME}...")

try:
    model_file = hf_hub_download(
        repo_id=DOWNLOAD_REPO_NAME, filename="model.pth", cache_dir="/content/hf_cache"
    )

    config_file = hf_hub_download(
        repo_id=DOWNLOAD_REPO_NAME,
        filename="config.json",
        cache_dir="/content/hf_cache",
    )

    print(f"model.pth: {model_file}")
    print(f"config.json: {config_file}")

except Exception as e:
    print(f"❌ Download failed: {e}")
    print("\nPlease check:")
    print("  1. Repository name is correct: 'username/repo-name'")
    print("  2. Repository is public or you have access")
    print("  3. Files exist in the repository")
    model_file = None
    config_file = None


# %%
if model_file and config_file:
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80 + "\n")

    with open(config_file, "r") as f:
        model_config = json.load(f)

    print(f"Model Configuration:")
    print(f"  Base Model: {model_config['base_model']}")
    print(f"  Input Dim: {model_config['input_dim']}")
    print(f"  Hidden Dim 1: {model_config['hidden_dim1']}")
    print(f"  Hidden Dim 2: {model_config['hidden_dim2']}")
    print(f"  Dropout: {model_config['dropout']}")
    print(f"\nMetrics:")
    print(f"  Validation MAE: ${model_config['metrics']['val_mae']:.2f}")
    print(f"  Test MAE: ${model_config['metrics']['test_mae']:.2f}")

    checkpoint = torch.load(model_file, map_location="cpu")

    downloaded_model = PriceRegressor(
        input_dim=checkpoint["model_config"]["input_dim"],
        hidden_dim1=checkpoint["model_config"]["hidden_dim1"],
        hidden_dim2=checkpoint["model_config"]["hidden_dim2"],
        dropout=checkpoint["model_config"]["dropout"],
    )

    downloaded_model.load_state_dict(checkpoint["model_state_dict"])
    downloaded_model.eval()

    print("\nLoading fine-tuned Llama model for embeddings...")
    base_model_name = checkpoint["embedding_config"]["base_model_name"]
    finetuned_model_name = checkpoint["embedding_config"]["finetuned_model_name"]

    embedding_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    embedding_tokenizer.pad_token = embedding_tokenizer.eos_token
    embedding_tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    embedding_model = PeftModel.from_pretrained(base_model, finetuned_model_name)
    embedding_model = embedding_model.merge_and_unload()
    embedding_model.eval()
else:
    print("\n⚠ Skipping model loading (download failed)")


# %%
if model_file and config_file:
    print("\n" + "=" * 80)
    print("TESTING DOWNLOADED MODEL WITH PREDICTIONS")
    print("=" * 80 + "\n")

    def predict_price_from_hf_model(description: str) -> float:
        """Predict price using downloaded model from HuggingFace.

        Returns predicted price in dollars.
        """
        inputs = embedding_tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=checkpoint["embedding_config"]["max_seq_length"],
        ).to(embedding_model.device)

        with torch.no_grad():
            outputs = embedding_model(**inputs, output_hidden_states=True)

            last_hidden_state = outputs.hidden_states[-1]

            attention_mask = inputs["attention_mask"]
            last_token_idx = attention_mask.sum(dim=1) - 1
            embedding = last_hidden_state[0, last_token_idx].unsqueeze(0).cpu()

            log_price = downloaded_model(embedding).item()
            price = np.exp(log_price)

        return price

    test_products = [
        "Apple AirPods Pro (2nd Generation) with MagSafe Charging Case",
        "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "USB-C Cable 6ft Fast Charging Cord",
        "Samsung Galaxy S24 Ultra 256GB Smartphone",
        "Paper Mate Ballpoint Pens, Medium Point, Black, 12 Pack",
    ]

    print(f"{'Product Description':<60} {'Predicted Price':>15}")
    print("=" * 80)

    for product in test_products:
        predicted_price = predict_price_from_hf_model(product)
        display_desc = product[:57] + "..." if len(product) > 60 else product
        print(f"{display_desc:<60} ${predicted_price:>14,.2f}")

    print("\n" + "=" * 80)
else:
    print("\n⚠ Skipping predictions (model not loaded)")
