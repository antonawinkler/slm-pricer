# %%
# !git clone https://github.com/antonawinkler/slm-pricer.git
# %cd slm-pricer
# !uv pip install .
# %cd ..

# %% [markdown]
# ## 📦 Setup: Imports and Configuration

# %%
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import sentence_transformers
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from slm_pricer.models import ResidualNet
from slm_pricer.training import check_early_stopping
from slm_pricer.utils import convert_back_y, transformed_prices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
@dataclass
class Config:
    """Configuration for embedding-based price prediction."""

    # Data configuration
    # Multiple dataset variations - val/test from first, train cycles through all
    dataset_names: tuple[str, ...] = (
        "ed-donner/items_prompts_full",
        # Add more dataset variations here
        # "ed-donner/items_prompts_full_v2",
        # "ed-donner/items_prompts_full_v3",
    )
    train_frac_pct: int = 50

    # Model configuration
    model_name: str = "antonawinkler/slm-pricer-llama-3.2-3b-embedding02"
    embeddings_base_path: pathlib.Path = pathlib.Path(
        "/content/drive/MyDrive/Pricer_Embeddings"
    )
    embedding_batch_size: int = 128

    # Training hyperparameters
    y_transform: str = "log"
    batch_size: int = 2048
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    dropout: float = 0.2
    epochs: int = 100
    patience: int = 30
    grad_clip: float = 1.0

    # Reproducibility
    seed: int = 42

    @property
    def model_safe_name(self) -> str:
        """Convert model name to filesystem-safe string."""
        return self.model_name.replace("/", "_").replace(":", "_")

    @property
    def embeddings_dir(self) -> pathlib.Path:
        """Directory where embeddings for this model are stored."""
        return self.embeddings_base_path / self.model_safe_name


CONFIG = Config()


# %%
# Mount Google Drive (Colab only)
try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except ImportError:
    pass


# %%
def load_data(split: str = "train", percent: int = 100, dataset_name: str | None = None) -> Dataset:
    """Load dataset split from HuggingFace."""
    if dataset_name is None:
        dataset_name = CONFIG.dataset_names[0]
    dataset = load_dataset(dataset_name, split=f"{split}[:{percent}%]")
    assert isinstance(dataset, Dataset)
    return dataset


# %%
class PriceDataset(TorchDataset):
    """PyTorch dataset for price regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# %%
def evaluate_comprehensive(
    model: nn.Module, data_loader: DataLoader, criterion: nn.Module
) -> dict[str, float]:
    """Evaluate model and return comprehensive metrics in dollars."""
    model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy().flatten().tolist())
            all_targets.extend(targets.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / len(data_loader)

    real_preds_transformed = np.array(all_preds)
    real_targets_transformed = np.array(all_targets)

    # Clip to prevent overflow on inverse transform (max ~$3M)
    real_preds_transformed = np.clip(real_preds_transformed, 0, 15)

    real_preds = convert_back_y(real_preds_transformed, CONFIG.y_transform)
    real_targets = convert_back_y(real_targets_transformed, CONFIG.y_transform)

    mae = mean_absolute_error(real_targets, real_preds)
    mape = mean_absolute_percentage_error(real_targets, real_preds)
    r2 = r2_score(real_targets, real_preds)

    errors = np.abs(real_targets - real_preds)
    median_error = float(np.median(errors))
    p90_error = float(np.percentile(errors, 90))

    return {
        "loss": avg_loss,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "median_error": median_error,
        "p90_error": p90_error,
    }


# %%
def training_loop(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
) -> None:
    """Train model with early stopping and evaluate on test set."""
    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80 + "\n")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_mape": [],
        "val_r2": [],
        "learning_rate": [],
    }

    best_val_mae = float("inf")
    no_improve_count = 0

    epoch_pbar = tqdm(range(CONFIG.epochs), desc="Training", unit="epoch", position=0)

    for epoch in epoch_pbar:
        # Create train loader for this epoch (cycles through datasets)
        train_loader = create_train_loader(epoch)

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_metrics = evaluate_comprehensive(model, val_loader, criterion)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_mape"].append(val_metrics["mape"])
        history["val_r2"].append(val_metrics["r2"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        epoch_pbar.set_postfix(
            {
                "MAE": f"${val_metrics['mae']:.2f}",
                "R²": f"{val_metrics['r2']:.3f}",
                "MAPE": f"{val_metrics['mape'] * 100:.2f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
            }
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            no_improve_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mae": best_val_mae,
                    "config": CONFIG,
                },
                os.path.join(CONFIG.embeddings_base_path, "best_model.pth"),
            )
            print(f"  ✓ New best MAE: ${best_val_mae:.2f}")
        else:
            no_improve_count += 1

        # Check early stopping
        should_stop, reason = check_early_stopping(
            val_mae=val_metrics["mae"],
            best_val_mae=best_val_mae,
            no_improve_count=no_improve_count,
            patience=CONFIG.patience,
        )

        if should_stop:
            print(f"\n⚠️  Early stopping: {reason}")
            break

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80 + "\n")

    checkpoint = torch.load(
        os.path.join(CONFIG.embeddings_base_path, "best_model.pth"), weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_comprehensive(model, test_loader, criterion)

    print("\n📊 Final Test Metrics:")
    print(f"  MAE:          ${test_metrics['mae']:.2f}")
    print(f"  MAPE:         {test_metrics['mape'] * 100:.2f}%")
    print(f"  R²:           {test_metrics['r2']:.4f}")


# %%
# Load all training dataset variations
print(f"Loading {len(CONFIG.dataset_names)} dataset variations...")
train_datasets = []
for i, dataset_name in enumerate(CONFIG.dataset_names):
    ds_train = load_data(split="train", percent=CONFIG.train_frac_pct, dataset_name=dataset_name)
    train_datasets.append(ds_train)
    print(f"  [{i+1}] {dataset_name}: {len(ds_train):,} samples")

# Val and test always from the first dataset
print(f"\nLoading val/test from: {CONFIG.dataset_names[0]}")
ds_test = load_data(split="test", percent=100)
ds_val = load_data(split="val", percent=100)

print(f"Validation set: {len(ds_val):,} samples")
print(f"Test set: {len(ds_test):,} samples")


# %%
def clean_prompt(text: str) -> str:
    """Remove prompt template text, leaving only product description."""
    return text.replace("What does this cost to the nearest dollar?\n\n", "").replace(
        "\n\nPrice is $", ""
    )


def get_embedding_path(split: str, dataset_idx: int = 0) -> pathlib.Path:
    """Get path for cached embeddings for a given split and dataset index."""
    if split in ("test", "val"):
        return CONFIG.embeddings_dir / f"{split}.npy"
    return CONFIG.embeddings_dir / f"{split}_{dataset_idx}.npy"


def load_or_create_embeddings() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Load embeddings from cache or generate them if they don't exist.

    Returns:
        Tuple of (X_train_all, X_test, X_val) where X_train_all is list of train embeddings

    Embeddings are automatically cached to disk for reuse. The cache key is based
    on the model name and dataset index.
    """
    # Check if embeddings need to be generated
    test_path = get_embedding_path("test")
    val_path = get_embedding_path("val")
    train_paths = [get_embedding_path("train", i) for i in range(len(train_datasets))]

    all_exist = test_path.exists() and val_path.exists() and all(p.exists() for p in train_paths)

    if all_exist:
        print(f"Loading cached embeddings from {CONFIG.embeddings_dir}")
        X_train_all = [np.load(p) for p in train_paths]
        X_test = np.load(test_path)
        X_val = np.load(val_path)
    else:
        print(
            f"Cached embeddings not found. Generating embeddings using {CONFIG.model_name}..."
        )

        # Create embeddings directory if it doesn't exist
        CONFIG.embeddings_dir.mkdir(parents=True, exist_ok=True)

        word_embedding_model = sentence_transformers.models.Transformer(
            CONFIG.model_name,
            max_seq_length=4096,
            model_args={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            },
        )

        pooling_model = sentence_transformers.models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_lasttoken=True,
        )

        embedder = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device="cuda"
        )
        embedder.tokenizer.pad_token = embedder.tokenizer.eos_token

        # Generate embeddings for all training datasets
        X_train_all = []
        for i, ds_train in enumerate(train_datasets):
            print(f"\nGenerating embeddings for train dataset {i+1}/{len(train_datasets)}...")
            X_train = embedder.encode(
                [clean_prompt(p) for p in ds_train["prompt"]],
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=CONFIG.embedding_batch_size,
            )
            X_train_all.append(X_train)
            np.save(train_paths[i], X_train)

        # Generate embeddings for test/val (only once, from first dataset)
        print("\nGenerating embeddings for test...")
        X_test = embedder.encode(
            [clean_prompt(p) for p in ds_test["prompt"]],
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=CONFIG.embedding_batch_size,
        )

        print("Generating embeddings for val...")
        X_val = embedder.encode(
            [clean_prompt(p) for p in ds_val["prompt"]],
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=CONFIG.embedding_batch_size,
        )

        # Save to cache
        print(f"\nSaving embeddings to {CONFIG.embeddings_dir}")
        np.save(test_path, X_test)
        np.save(val_path, X_val)

    print("\nEmbeddings shape:")
    for i, X_train in enumerate(X_train_all):
        print(f"  X_train[{i}]: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  X_val: {X_val.shape}")

    return X_train_all, X_test, X_val


# %%
X_train_all, X_test, X_val = load_or_create_embeddings()


# %%
# Transform prices for all training datasets
y_train_all = [
    transformed_prices(np.array(ds["completion"], dtype="float32"), CONFIG.y_transform)
    for ds in train_datasets
]

y_test = transformed_prices(
    np.array(ds_test["completion"], dtype="float32"), CONFIG.y_transform
)
y_val = transformed_prices(
    np.array(ds_val["completion"], dtype="float32"), CONFIG.y_transform
)

print("Targets shape:")
for i, y_train in enumerate(y_train_all):
    print(f"  y_train[{i}]: {y_train.shape}")
print(f"  y_test: {y_test.shape}")
print(f"  y_val: {y_val.shape}")


# %%
model = ResidualNet().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {total_params:,} parameters")


# %%
# Optional: Load pre-trained regression head from HuggingFace Hub
# Uncomment to use a pre-trained model instead of training from scratch
# from slm_pricer.training import load_regression_head_from_hub
#
# repo_id = "antonawinkler/slm-pricer-llama-3.2-3b-embedding02-regressor-20260125"
# model, metadata = load_regression_head_from_hub(repo_id, device=device)
#
# print(f"Loaded pre-trained model from {repo_id}")
# print(f"Pre-trained Val MAE: ${metadata['metrics'].get('val_mae', 'N/A')}")
# print(f"Pre-trained Test MAE: ${metadata['metrics'].get('test_mae', 'N/A')}")
# print(f"Training config: {metadata['config']}")


# %%
def create_train_loader(epoch: int) -> DataLoader:
    """Create train loader for a specific epoch, cycling through datasets.

    Datasets are shuffled before cycling to ensure different order each time.
    """
    # Shuffle dataset order for variety
    import random
    indices = list(range(len(X_train_all)))
    random.Random(CONFIG.seed + epoch).shuffle(indices)

    # Select dataset for this epoch (cycle through shuffled order)
    dataset_idx = indices[epoch % len(X_train_all)]
    X_train = X_train_all[dataset_idx]
    y_train = y_train_all[dataset_idx]

    dataset_name = CONFIG.dataset_names[dataset_idx]
    print(f"  Using dataset [{dataset_idx+1}]: {dataset_name.split('/')[-1]}")

    return DataLoader(
        PriceDataset(X_train, y_train), batch_size=CONFIG.batch_size, shuffle=True
    )


# Val and test loaders are fixed (always from first dataset)
val_loader = DataLoader(
    PriceDataset(X_val, y_val), batch_size=CONFIG.batch_size, shuffle=False
)
test_loader = DataLoader(
    PriceDataset(X_test, y_test), batch_size=CONFIG.batch_size, shuffle=False
)

print(f"Val batches: {len(val_loader):,}")
print(f"Test batches: {len(test_loader):,}")
print(f"\nTrain loaders will cycle through {len(CONFIG.dataset_names)} dataset(s)")

# Calculate steps per epoch for scheduler (use first dataset as reference)
temp_loader = create_train_loader(0)
steps_per_epoch = len(temp_loader)
del temp_loader

criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG.learning_rate,
    epochs=CONFIG.epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,
    anneal_strategy="cos",
)


# %%
training_loop(
    model, val_loader, test_loader, criterion, optimizer, scheduler
)


# %%
# Upload best model to HuggingFace Hub
from datetime import datetime

from slm_pricer.training import save_regression_head_to_hub

# Load the best model checkpoint
checkpoint = torch.load(
    os.path.join(CONFIG.embeddings_base_path, "best_model.pth"), weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])

# Generate model name with configuration and timestamp
timestamp = datetime.now().strftime("%Y%m%d")
base_model_name = CONFIG.model_name.split("/")[-1]
repo_id = f"antonawinkler/slm-pricer-{base_model_name}-regressor-{timestamp}"

print(f"Uploading model to HuggingFace Hub as: {repo_id}")

# Gather metrics from checkpoint
metrics = {
    "val_mae": checkpoint.get("mae", 0.0),
    "train_frac_pct": CONFIG.train_frac_pct,
}

# Gather configuration
config = {
    "dataset_name": CONFIG.dataset_name,
    "model_name": CONFIG.model_name,
    "y_transform": CONFIG.y_transform,
    "batch_size": CONFIG.batch_size,
    "learning_rate": CONFIG.learning_rate,
    "weight_decay": CONFIG.weight_decay,
    "dropout": CONFIG.dropout,
    "epochs": CONFIG.epochs,
    "patience": CONFIG.patience,
    "grad_clip": CONFIG.grad_clip,
    "input_dim": X_train.shape[1],
}

# Save to HuggingFace Hub with metadata
save_regression_head_to_hub(
    model=model,
    repo_id=repo_id,
    metrics=metrics,
    config=config,
    commit_message=f"Regression head trained on {CONFIG.dataset_name}",
    private=False,
)

print(f"✓ Model uploaded successfully to {repo_id}")
