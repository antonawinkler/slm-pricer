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
from slm_pricer.utils import convert_back_y, transformed_prices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
@dataclass
class Config:
    """Configuration for embedding-based price prediction."""

    # Data configuration
    dataset_name: str = "ed-donner/items_prompts_full"
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
def load_data(split: str = "train", percent: int = 100) -> Dataset:
    """Load dataset split from HuggingFace."""
    dataset = load_dataset(CONFIG.dataset_name, split=f"{split}[:{percent}%]")
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
    train_loader: DataLoader,
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
            if no_improve_count >= CONFIG.patience:
                print(
                    f"\n⚠️  Early stopping! No improvement for {CONFIG.patience} epochs."
                )
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
print("Loading datasets...")
ds_train = load_data(split="train", percent=CONFIG.train_frac_pct)
ds_test = load_data(split="test", percent=100)
ds_val = load_data(split="val", percent=100)

print(f"Train set: {len(ds_train):,} samples")
print(f"Test set: {len(ds_test):,} samples")
print(f"Validation set: {len(ds_val):,} samples")


# %%
def clean_prompt(text: str) -> str:
    """Remove prompt template text, leaving only product description."""
    return text.replace("What does this cost to the nearest dollar?\n\n", "").replace(
        "\n\nPrice is $", ""
    )


def get_embedding_path(split: str) -> pathlib.Path:
    """Get path for cached embeddings for a given split."""
    return CONFIG.embeddings_dir / f"{split}.npy"


def load_or_create_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings from cache or generate them if they don't exist.

    Returns:
        Tuple of (X_train, X_test, X_val) embeddings

    Embeddings are automatically cached to disk for reuse. The cache key is based
    on the model name, so switching models will regenerate embeddings.
    """
    train_path = get_embedding_path("train")
    test_path = get_embedding_path("test")
    val_path = get_embedding_path("val")

    # Check if all embedding files exist
    if train_path.exists() and test_path.exists() and val_path.exists():
        print(f"Loading cached embeddings from {CONFIG.embeddings_dir}")
        X_train = np.load(train_path)
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

        # Generate embeddings for each split
        X_train = embedder.encode(
            [clean_prompt(p) for p in ds_train["prompt"]],
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=CONFIG.embedding_batch_size,
        )

        X_test = embedder.encode(
            [clean_prompt(p) for p in ds_test["prompt"]],
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=CONFIG.embedding_batch_size,
        )

        X_val = embedder.encode(
            [clean_prompt(p) for p in ds_val["prompt"]],
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=CONFIG.embedding_batch_size,
        )

        # Save to cache
        print(f"Saving embeddings to {CONFIG.embeddings_dir}")
        np.save(train_path, X_train)
        np.save(test_path, X_test)
        np.save(val_path, X_val)

    # Truncate train embeddings to match dataset size
    X_train = X_train[: len(ds_train)]

    print("\nEmbeddings shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  X_val: {X_val.shape}")

    return X_train, X_test, X_val


# %%
X_train, X_test, X_val = load_or_create_embeddings()


# %%
y_train = transformed_prices(
    np.array(ds_train["completion"], dtype="float32"), CONFIG.y_transform
)
y_test = transformed_prices(
    np.array(ds_test["completion"], dtype="float32"), CONFIG.y_transform
)
y_val = transformed_prices(
    np.array(ds_val["completion"], dtype="float32"), CONFIG.y_transform
)

print("Targets shape:")
print(f"  y_train: {y_train.shape}")
print(f"  y_test: {y_test.shape}")
print(f"  y_val: {y_val.shape}")


# %%
model = ResidualNet().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {total_params:,} parameters")


# %%
train_loader = DataLoader(
    PriceDataset(X_train, y_train), batch_size=CONFIG.batch_size, shuffle=True
)
val_loader = DataLoader(
    PriceDataset(X_val, y_val), batch_size=CONFIG.batch_size, shuffle=False
)
test_loader = DataLoader(
    PriceDataset(X_test, y_test), batch_size=CONFIG.batch_size, shuffle=False
)

criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG.learning_rate,
    epochs=CONFIG.epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
)


# %%
training_loop(
    model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler
)
