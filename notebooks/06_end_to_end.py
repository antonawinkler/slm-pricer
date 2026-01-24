# %%
# End-to-End Training: Llama + Regression Head
#
# This script trains a Llama model with QLoRA and a regression head jointly.
# The last token embedding from Llama is fed into a ResNet regression head
# to predict prices directly from product descriptions.

# %%
# !git clone https://github.com/antonawinkler/slm-pricer.git
# %cd slm-pricer
# !uv pip install .
# %cd ..

# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import wandb
from google.colab import userdata
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from slm_pricer.data import load_data_from_hf
from slm_pricer.models import ResidualNet
from slm_pricer.utils import convert_back_y, transformed_prices

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class Config:
    """Configuration for end-to-end Llama + regression head training."""

    # Model
    base_model: str = "meta-llama/Llama-3.2-3B"
    max_seq_length: int = 128

    # Quantization
    use_4bit_quant: bool = True
    use_flash_attention: bool = True

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64  # 2 * lora_r
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    # Regression head (ResidualNet) - uses default parameters
    # ResidualNet has hardcoded input_dim=3072 (matches Llama 3.2 3B hidden size)

    # Training
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    gradient_accumulation_steps: int = 1

    # Warmup phase: train only regression head with frozen LoRA
    warmup_epochs: int = 1
    warmup_learning_rate: float = 1e-3  # Higher LR for head-only training

    # Data
    dataset_name: str = "ed-donner/items_prompts_full"
    data_percent: int = 100
    y_transform: str = "log"

    # Logging
    log_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    val_size: int = 1000

    # Output
    project_name: str = "slm-pricer-e2e"
    hub_user: str = "antonawinkler"
    log_to_wandb: bool = True

    # Reproducibility
    seed: int = 42


CONFIG = Config()

# %%
# =============================================================================
# SETUP
# =============================================================================

torch.manual_seed(CONFIG.seed)
np.random.seed(CONFIG.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
use_bf16 = capability[0] >= 8

print(f"Device: {device}")
print(f"Using bf16: {use_bf16}")

# %%
# Login to HuggingFace and Weights & Biases

hf_token = userdata.get("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

if CONFIG.log_to_wandb:
    wandb_api_key = userdata.get("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = wandb_api_key
    wandb.login()
    os.environ["WANDB_PROJECT"] = CONFIG.project_name


# %%
# =============================================================================
# DATASET
# =============================================================================


class TextPriceDataset(Dataset):
    """Dataset that returns tokenized text and price targets."""

    def __init__(
        self,
        texts: list[str],
        prices: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.prices = torch.tensor(prices, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        price = self.prices[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "price": price,
        }


# %%
# =============================================================================
# MODEL: LLAMA + REGRESSION HEAD
# =============================================================================


class LlamaWithRegressionHead(nn.Module):
    """Llama model with a regression head for price prediction.

    Extracts the last token embedding from Llama and feeds it through
    a ResidualNet regression head.
    """

    def __init__(self, llama_model: nn.Module) -> None:
        super().__init__()
        self.llama = llama_model
        self.regression_head = ResidualNet()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Get hidden states from Llama
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract last layer hidden states: (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[-1]

        # Get the last non-padding token embedding for each sequence
        # Find the position of the last real token using attention mask
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_embeddings = hidden_states[batch_indices, seq_lengths]

        # Cast to same dtype as regression head (Llama may output different dtype)
        target_dtype = next(self.regression_head.parameters()).dtype
        last_token_embeddings = last_token_embeddings.to(target_dtype)

        # Pass through regression head
        predictions = self.regression_head(last_token_embeddings)

        return predictions


# %%
# =============================================================================
# TRAINING UTILITIES
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    gradient_accumulation_steps: int,
    epoch: int,
    global_step: int,
    log_steps: int,
) -> tuple[float, int]:
    """Train for one epoch, return average loss and updated global step."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prices = batch["price"].to(device)

        predictions = model(input_ids, attention_mask)
        # Cast prices to same dtype as predictions for loss computation
        loss = criterion(predictions.squeeze(), prices.to(predictions.dtype))
        loss = loss / gradient_accumulation_steps

        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_steps == 0 and CONFIG.log_to_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item() * gradient_accumulation_steps,
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    }
                )

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        progress_bar.set_postfix(
            {"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"}
        )

    return total_loss / num_batches, global_step


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    convert_back_fn: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float]:
    """Evaluate model and return loss and MAE in dollars."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prices = batch["price"].to(device)

            predictions = model(input_ids, attention_mask)
            # Cast prices to same dtype as predictions for loss computation
            loss = criterion(predictions.squeeze(), prices.to(predictions.dtype))

            total_loss += loss.item()
            all_preds.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(prices.cpu().numpy())

    # Convert back to dollars
    preds_dollars = convert_back_fn(np.array(all_preds))
    targets_dollars = convert_back_fn(np.array(all_targets))
    mae = np.mean(np.abs(preds_dollars - targets_dollars))

    return {
        "loss": total_loss / len(val_loader),
        "mae": mae,
    }


# %%
# =============================================================================
# LOAD DATA
# =============================================================================

df_train = load_data_from_hf(
    split="train", percent=CONFIG.data_percent, dataset_name=CONFIG.dataset_name
)
df_val = load_data_from_hf(split="val", percent=100, dataset_name=CONFIG.dataset_name)
df_test = load_data_from_hf(split="test", percent=100, dataset_name=CONFIG.dataset_name)

# Limit validation size for faster evaluation during training
df_val = df_val.head(CONFIG.val_size)

print(f"Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# Transform prices
y_train = transformed_prices(
    df_train["completion"].to_numpy(dtype="float32"), CONFIG.y_transform
)
y_val = transformed_prices(
    df_val["completion"].to_numpy(dtype="float32"), CONFIG.y_transform
)
y_test = transformed_prices(
    df_test["completion"].to_numpy(dtype="float32"), CONFIG.y_transform
)

print(f"Price range (log-transformed): [{y_train.min():.2f}, {y_train.max():.2f}]")

# %%
# =============================================================================
# LOAD TOKENIZER AND MODEL
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(CONFIG.base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantization config
if CONFIG.use_4bit_quant:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

print(f"Loading {CONFIG.base_model}...")
base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG.base_model,
    quantization_config=quant_config,
    device_map="auto",
    attn_implementation="flash_attention_2" if CONFIG.use_flash_attention else None,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

# %%
# =============================================================================
# APPLY LORA
# =============================================================================

lora_config = LoraConfig(
    r=CONFIG.lora_r,
    lora_alpha=CONFIG.lora_alpha,
    lora_dropout=CONFIG.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=list(CONFIG.lora_target_modules),
)

base_model = get_peft_model(base_model, lora_config)
base_model.print_trainable_parameters()

# %%
# =============================================================================
# CREATE COMBINED MODEL
# =============================================================================

embedding_dim = base_model.config.hidden_size
print(f"Llama embedding dimension: {embedding_dim}")

model = LlamaWithRegressionHead(llama_model=base_model)

# Move regression head to the same device and dtype as Llama
# (Llama is on CUDA via device_map="auto" with bf16/fp16, but regression head defaults to CPU float32)
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
model.regression_head = model.regression_head.to(device=device, dtype=compute_dtype)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %%
# =============================================================================
# CREATE DATASETS AND DATALOADERS
# =============================================================================

train_dataset = TextPriceDataset(
    texts=df_train["prompt"].tolist(),
    prices=y_train,
    tokenizer=tokenizer,
    max_length=CONFIG.max_seq_length,
)

val_dataset = TextPriceDataset(
    texts=df_val["prompt"].tolist(),
    prices=y_val,
    tokenizer=tokenizer,
    max_length=CONFIG.max_seq_length,
)

test_dataset = TextPriceDataset(
    texts=df_test["prompt"].tolist(),
    prices=y_test,
    tokenizer=tokenizer,
    max_length=CONFIG.max_seq_length,
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, shuffle=False)

print(f"Train batches: {len(train_loader):,}")
print(f"Val batches: {len(val_loader):,}")
print(f"Test batches: {len(test_loader):,}")

# %%
# =============================================================================
# SETUP TRAINING
# =============================================================================

criterion = nn.MSELoss()


def freeze_lora_adapters(model: nn.Module) -> None:
    """Freeze all LoRA adapter parameters."""
    for name, param in model.llama.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = False


def unfreeze_lora_adapters(model: nn.Module) -> None:
    """Unfreeze all LoRA adapter parameters."""
    for name, param in model.llama.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Phase 1: Warmup - train only regression head
print("\n" + "=" * 80)
print("PHASE 1: WARMUP - Training regression head only (LoRA frozen)")
print("=" * 80)

freeze_lora_adapters(model)
print(f"Trainable parameters (warmup): {count_trainable_params(model):,}")

warmup_optimizer = torch.optim.AdamW(
    model.regression_head.parameters(),
    lr=CONFIG.warmup_learning_rate,
    weight_decay=CONFIG.weight_decay,
)

warmup_total_steps = (
    len(train_loader) * CONFIG.warmup_epochs // CONFIG.gradient_accumulation_steps
)

warmup_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    warmup_optimizer,
    max_lr=CONFIG.warmup_learning_rate,
    total_steps=warmup_total_steps,
    pct_start=0.1,
    anneal_strategy="cos",
)

print(f"Warmup epochs: {CONFIG.warmup_epochs}")
print(f"Warmup learning rate: {CONFIG.warmup_learning_rate:.2e}")
print(f"Warmup total steps: {warmup_total_steps:,}")

# Phase 2: Joint training - train both LoRA and regression head
print("\n" + "-" * 40)
print("PHASE 2 CONFIG: Joint training (LoRA + regression head)")
print("-" * 40)

joint_total_steps = (
    len(train_loader) * CONFIG.epochs // CONFIG.gradient_accumulation_steps
)

print(f"Joint epochs: {CONFIG.epochs}")
print(f"Joint learning rate: {CONFIG.learning_rate:.2e}")
print(f"Joint total steps: {joint_total_steps:,}")

# %%
# =============================================================================
# INITIALIZE WANDB
# =============================================================================

run_name = f"e2e-{datetime.now():%Y%m%d-%H%M%S}"

if CONFIG.log_to_wandb:
    wandb.init(
        project=CONFIG.project_name,
        name=run_name,
        config={
            "base_model": CONFIG.base_model,
            "lora_r": CONFIG.lora_r,
            "lora_alpha": CONFIG.lora_alpha,
            "regression_head": "ResidualNet",
            "batch_size": CONFIG.batch_size,
            "learning_rate": CONFIG.learning_rate,
            "warmup_learning_rate": CONFIG.warmup_learning_rate,
            "warmup_epochs": CONFIG.warmup_epochs,
            "joint_epochs": CONFIG.epochs,
        },
    )

# %%
# =============================================================================
# TRAINING LOOP
# =============================================================================

convert_back_fn = partial(convert_back_y, transform_type=CONFIG.y_transform)

best_val_mae = float("inf")
global_step = 0

# =============================================================================
# PHASE 1: WARMUP - Train regression head only (LoRA frozen)
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 1: WARMUP - Training regression head only")
print("=" * 80 + "\n")

for epoch in range(CONFIG.warmup_epochs):
    print(f"\nWarmup Epoch {epoch + 1}/{CONFIG.warmup_epochs}")
    print("-" * 40)

    # Train (LoRA is frozen, only regression head learns)
    train_loss, global_step = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=warmup_optimizer,
        scheduler=warmup_scheduler,
        criterion=criterion,
        device=device,
        grad_clip=CONFIG.max_grad_norm,
        gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,
        epoch=epoch,
        global_step=global_step,
        log_steps=CONFIG.log_steps,
    )

    # Evaluate
    val_metrics = evaluate(model, val_loader, criterion, device, convert_back_fn)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f} | Val MAE: ${val_metrics['mae']:.2f}")

    if CONFIG.log_to_wandb:
        wandb.log(
            {
                "phase": "warmup",
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/mae": val_metrics["mae"],
            }
        )

    # Track best model
    if val_metrics["mae"] < best_val_mae:
        best_val_mae = val_metrics["mae"]
        print(f"New best Val MAE: ${best_val_mae:.2f}")

    # Save checkpoint after each warmup epoch
    checkpoint_dir = f"./{run_name}-warmup-epoch-{epoch + 1}"
    print(f"Saving checkpoint to {checkpoint_dir}...")

    torch.save(
        {
            "regression_head_state_dict": model.regression_head.state_dict(),
            "epoch": epoch + 1,
            "phase": "warmup",
            "val_mae": val_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "train_loss": train_loss,
            "model_class": "ResidualNet",
        },
        f"{checkpoint_dir}-regression-head.pth",
    )
    print(
        f"Warmup checkpoint saved: epoch {epoch + 1}, Val MAE: ${val_metrics['mae']:.2f}"
    )

print("\n" + "=" * 80)
print(f"WARMUP COMPLETE - Best Val MAE: ${best_val_mae:.2f}")
print("=" * 80)

# =============================================================================
# PHASE 2: JOINT TRAINING - Train both LoRA and regression head
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 2: JOINT TRAINING - LoRA + regression head")
print("=" * 80 + "\n")

# Unfreeze LoRA adapters
unfreeze_lora_adapters(model)
print(f"Trainable parameters (joint): {count_trainable_params(model):,}")

# Create new optimizer for joint training (includes both LoRA and regression head)
joint_optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=CONFIG.learning_rate,
    weight_decay=CONFIG.weight_decay,
)

joint_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    joint_optimizer,
    max_lr=CONFIG.learning_rate,
    total_steps=joint_total_steps,
    pct_start=CONFIG.warmup_ratio,
    anneal_strategy="cos",
)

# Reset global step for joint training phase
global_step = 0

for epoch in range(CONFIG.epochs):
    print(f"\nJoint Epoch {epoch + 1}/{CONFIG.epochs}")
    print("-" * 40)

    # Train (both LoRA and regression head)
    train_loss, global_step = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=joint_optimizer,
        scheduler=joint_scheduler,
        criterion=criterion,
        device=device,
        grad_clip=CONFIG.max_grad_norm,
        gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,
        epoch=epoch,
        global_step=global_step,
        log_steps=CONFIG.log_steps,
    )

    # Evaluate
    val_metrics = evaluate(model, val_loader, criterion, device, convert_back_fn)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f} | Val MAE: ${val_metrics['mae']:.2f}")

    if CONFIG.log_to_wandb:
        wandb.log(
            {
                "phase": "joint",
                "epoch": CONFIG.warmup_epochs + epoch + 1,
                "train/epoch_loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/mae": val_metrics["mae"],
            }
        )

    # Track best model
    if val_metrics["mae"] < best_val_mae:
        best_val_mae = val_metrics["mae"]
        print(f"New best Val MAE: ${best_val_mae:.2f}")

    # Save checkpoint after each epoch
    checkpoint_dir = f"./{run_name}-joint-epoch-{epoch + 1}"
    print(f"Saving checkpoint to {checkpoint_dir}...")

    # Save LoRA adapter
    model.llama.save_pretrained(checkpoint_dir)

    # Save regression head
    torch.save(
        {
            "regression_head_state_dict": model.regression_head.state_dict(),
            "epoch": epoch + 1,
            "phase": "joint",
            "val_mae": val_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "train_loss": train_loss,
            "model_class": "ResidualNet",
        },
        f"{checkpoint_dir}/regression_head.pth",
    )
    print(f"Checkpoint saved: epoch {epoch + 1}, Val MAE: ${val_metrics['mae']:.2f}")

print("\n" + "=" * 80)
print("JOINT TRAINING COMPLETE")
print(f"Total epochs: {CONFIG.warmup_epochs} warmup + {CONFIG.epochs} joint")
print(f"Best Val MAE: ${best_val_mae:.2f}")
print("=" * 80)

# %%
# =============================================================================
# FINAL EVALUATION ON TEST SET
# =============================================================================

print("\n" + "=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80 + "\n")

test_metrics = evaluate(model, test_loader, criterion, device, convert_back_fn)

print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test MAE: ${test_metrics['mae']:.2f}")

if CONFIG.log_to_wandb:
    wandb.log(
        {
            "test/loss": test_metrics["loss"],
            "test/mae": test_metrics["mae"],
        }
    )
    wandb.finish()

# %%
# =============================================================================
# SAVE MODEL
# =============================================================================

hub_model_name = f"{CONFIG.hub_user}/{CONFIG.project_name}-{run_name}"
local_lora_path = f"./{run_name}-lora"
local_head_path = f"./{run_name}-regression-head.pth"

# Save the LoRA adapter locally
model.llama.save_pretrained(local_lora_path)
print(f"LoRA adapter saved to {local_lora_path}")

# Save the regression head locally
regression_head_checkpoint = {
    "regression_head_state_dict": model.regression_head.state_dict(),
    "model_class": "ResidualNet",
    "metrics": {
        "best_val_mae": best_val_mae,
        "test_mae": test_metrics["mae"],
        "test_loss": test_metrics["loss"],
    },
    "training_config": {
        "base_model": CONFIG.base_model,
        "y_transform": CONFIG.y_transform,
        "max_seq_length": CONFIG.max_seq_length,
    },
}
torch.save(regression_head_checkpoint, local_head_path)
print(f"Regression head saved to {local_head_path}")

# %%
# =============================================================================
# PUSH TO HUGGINGFACE HUB
# =============================================================================

from huggingface_hub import HfApi

print(f"\nPushing to HuggingFace Hub: {hub_model_name}")

# Push LoRA adapter
model.llama.push_to_hub(hub_model_name, private=True)
print(f"LoRA adapter pushed to {hub_model_name}")

# Push regression head checkpoint
api = HfApi()
api.upload_file(
    path_or_fileobj=local_head_path,
    path_in_repo="regression_head.pth",
    repo_id=hub_model_name,
    repo_type="model",
)
print(f"Regression head pushed to {hub_model_name}/regression_head.pth")

# Push tokenizer for convenience
tokenizer.push_to_hub(hub_model_name, private=True)
print(f"Tokenizer pushed to {hub_model_name}")

print("\n" + "=" * 80)
print(
    f"TRAINING COMPLETE - Model available at: https://huggingface.co/{hub_model_name}"
)
print("=" * 80)
