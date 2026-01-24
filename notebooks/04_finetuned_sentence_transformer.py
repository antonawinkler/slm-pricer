# %%
# !git clone https://github.com/antonawinkler/slm-pricer.git

# %cd slm-pricer
# !uv pip install .
# %cd ..

# %%
from __future__ import annotations

import os
import random
from dataclasses import astuple, dataclass, fields
from datetime import datetime
from typing import Any

import pandas as pd
import torch
import wandb
from datasets import Dataset  # type: ignore[import-untyped]
from google.colab import userdata  # type: ignore[import-untyped]
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    models,
)
from transformers import BitsAndBytesConfig

from slm_pricer.data import load_data_from_hf  # type: ignore[import-untyped]

# %%
# ============================================
# HYPERPARAMETERS - Configure all settings here
# ============================================

# Model Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B"
MAX_SEQ_LENGTH = 128
POOLING_MODE = "lasttoken"

# Quantization
USE_4BIT_QUANT = True
USE_FLASH_ATTENTION = True

# SimCSE Dropout (with augmentation, can use lower dropout)
SIMCSE_DROPOUT = 0.1

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32  # 2 * LORA_R
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training Hyperparameters
EPOCHS = 2
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 0.3

# Loss Function
USE_CACHED_LOSS = True  # True = more negatives but 2x slower
MINI_BATCH_SIZE = 128  # Only used if USE_CACHED_LOSS=True

# Logging & Checkpointing
LOGGING_STEPS = 5
EVAL_FREQUENCY = 5  # Evaluate N times per epoch
SAVE_FREQUENCY = 1  # Save N times per epoch

# Data
DATASET_NAME = "antonawinkler/two_items_full"  # Updated to two-item dataset
DATA_PERCENT = 100

# Data Augmentation
# Policy options:
#   "direct": sentence1 = summary_1, sentence2 = summary_2 (no augmentation)
#   "partition": Randomly assign fields to sentences, then add-back with probability
AUGMENTATION_POLICY = "direct"  # "direct", "partition", or "partition_shared"


@dataclass
class Sentence:
    title: str | None | bool | float = None
    category: str | None | bool | float = None
    brand: str | None | bool | float = None
    description: str | None | bool | float = None
    details: str | None | bool | float = None
    price: str | None | bool | float = None
    weight: str | None | bool | float = None

    @property
    def is_empty(self) -> bool:
        return all(value is None for value in astuple(self))

    def to_string(self) -> str:
        parts = []
        for field in fields(Sentence):
            value = getattr(self, field.name)
            if value is not None:
                parts.append(str(value))
        return "\n".join(parts)


ADD_BACK_PROB = Sentence(
    title=0.01,
    category=0.0,
    brand=0.0,
    description=0.01,
    details=0.01,
    price=0.01,
    weight=0.01,
)

USED_FIELDS = Sentence(
    title=True,
    category=True,
    brand=True,
    description=True,
    details=True,
    price=True,
    weight=True,
)

SHARED_FIELDS = Sentence(
    title=False,
    category=True,
    brand=True,
    description=False,
    details=False,
    price=False,
    weight=False,
)

# Output & Hub
OUTPUT_DIR = "llama-3.2-3b-lora-output"
MODEL_REPO_ID = "antonawinkler/slm-pricer-llama-3.2-3b"
RUN_NAME_BASE = "llama-3.2-3b-simcse-partition"
RUN_NAME = f"{RUN_NAME_BASE}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
WANDB_PROJECT = "slm-pricer-llama-3b"

print("=" * 50)
print("CONFIGURATION LOADED")
print("=" * 50)
print(f"Model: {MODEL_NAME}")
print(f"Run name: {RUN_NAME}")
print(f"SimCSE dropout: {SIMCSE_DROPOUT}")
print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
print(f"Batch size: {BATCH_SIZE}, Eval: {EVAL_BATCH_SIZE}")
print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
print(f"Cached loss: {USE_CACHED_LOSS}")
print(f"Augmentation policy: {AUGMENTATION_POLICY}")
if AUGMENTATION_POLICY == "partition":
    print(f"Add-back probability: {ADD_BACK_PROB}")
print("=" * 50)

# %%
wandb_api_key = userdata.get("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login()
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_WATCH"] = "gradients"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT_QUANT,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

word_embedding_model = models.Transformer(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    model_args={
        "quantization_config": bnb_config,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2" if USE_FLASH_ATTENTION else None,
    },
    config_args={
        "attention_dropout": SIMCSE_DROPOUT,
    },
)

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode=POOLING_MODE,
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# %%
df_train = load_data_from_hf(
    split="train", percent=DATA_PERCENT, dataset_name=DATASET_NAME
)
df_val = load_data_from_hf(
    split="validation", percent=DATA_PERCENT, dataset_name=DATASET_NAME
)


def create_direct_pair(row: dict[str, Any]) -> tuple[str, str]:
    """Direct mapping: sentence1 = summary_1, sentence2 = summary_2."""
    return row["summary_1"].strip(), row["summary_2"].strip()


def to_sentence(row: dict[str, Any], suffix: str = "") -> Sentence:
    summary = row["summary" + suffix].strip().split("\n")
    if len(summary) != 5:
        raise ValueError(f"Expected 5 lines in summary, {summary}")
    weight = row["weight" + suffix]
    return Sentence(
        title=summary[0],
        category=summary[1],
        brand=summary[2],
        description=summary[3],
        details=summary[4],
        price=f"Price: ${row['price' + suffix]:.2f}",
        weight=f"Weight: {weight} lbs" if weight != 0 else None,
    )


def create_partition_pair(
    row: dict[str, Any],
    add_back_prob: Sentence = ADD_BACK_PROB,
    used_fields: Sentence = USED_FIELDS,
    shared_fields: Sentence = SHARED_FIELDS,
) -> tuple[str, str]:
    """Create two augmented sentences using partition-then-add-back strategy."""
    # Get all lines from item 1
    original_sentence_1 = to_sentence(row, suffix="_1")
    original_sentence_2 = to_sentence(row, suffix="_2")
    sentence_1 = Sentence()
    sentence_2 = Sentence()

    while True:
        for field in fields(Sentence):
            if not getattr(used_fields, field.name):
                continue
            if getattr(shared_fields, field.name):
                setattr(
                    sentence_1, field.name, getattr(original_sentence_1, field.name)
                )
                setattr(
                    sentence_2, field.name, getattr(original_sentence_2, field.name)
                )
            elif random.random() < 0.5:
                setattr(
                    sentence_1, field.name, getattr(original_sentence_1, field.name)
                )
            else:
                setattr(
                    sentence_2, field.name, getattr(original_sentence_2, field.name)
                )
        if not sentence_1.is_empty and not sentence_2.is_empty:
            break

    # Step 2: Add items back with add_back_prob probability
    for field in fields(Sentence):
        if random.random() < getattr(add_back_prob, field.name):
            setattr(sentence_1, field.name, getattr(original_sentence_1, field.name))
        if random.random() < getattr(add_back_prob, field.name):
            setattr(sentence_2, field.name, getattr(original_sentence_2, field.name))

    return sentence_1.to_string(), sentence_2.to_string()


augmentation_policy = {
    "direct": create_direct_pair,
    "partition": create_partition_pair,
}[AUGMENTATION_POLICY]


def create_epoch_pairs(
    df: pd.DataFrame,
    policy_fn: Any,
    shuffle: bool = True,
) -> list[dict[str, str]]:
    """Generate sentence pairs for one epoch, optionally shuffling first."""
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    pairs = []
    for _, row in df.iterrows():
        s1, s2 = policy_fn(row)
        pairs.append({"sentence1": s1, "sentence2": s2})
    return pairs


def create_multi_epoch_dataset(
    df: pd.DataFrame,
    policy_fn: Any,
    n_epochs: int,
) -> Dataset:
    """Pre-generate sentence pairs for all epochs with shuffling between epochs."""
    all_pairs = []
    for epoch in range(n_epochs):
        print(f"Generating pairs for epoch {epoch + 1}/{n_epochs}...")
        epoch_pairs = create_epoch_pairs(df, policy_fn, shuffle=True)
        all_pairs.extend(epoch_pairs)
    return Dataset.from_list(all_pairs)


print(f"Pre-generating training data for {EPOCHS} epochs...")
train_dataset = create_multi_epoch_dataset(df_train, augmentation_policy, EPOCHS)

print("Generating validation data...")
eval_pairs = create_epoch_pairs(df_val, augmentation_policy, shuffle=False)
eval_dataset = Dataset.from_list(eval_pairs)

train_size = len(df_train)
eval_size = len(df_val)

print(
    f"Train dataset: {len(train_dataset):,} pairs ({train_size:,} samples x {EPOCHS} epochs)"
)
print(f"Validation dataset: {len(eval_dataset):,} pairs")
print(f"Augmentation policy: {AUGMENTATION_POLICY}")

# Show examples of augmented pairs
print("\nExample pairs:")
for i in range(min(3, len(train_dataset))):
    example = train_dataset[i]
    print(f"\n--- Example {i + 1} ---")
    print(f"Sentence 1: {example['sentence1']}")
    print(f"Sentence 2: {example['sentence2']}")

# %%
transformer_module = model[0]
tokenizer = transformer_module.tokenizer
if tokenizer.pad_token is None:  # type: ignore[union-attr]
    tokenizer.pad_token = tokenizer.eos_token  # type: ignore[union-attr]

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=LORA_TARGET_MODULES,
)

transformer_module.auto_model = get_peft_model(
    transformer_module.auto_model,  # type: ignore[arg-type]
    peft_config,
)

# %%
wandb.init(project=WANDB_PROJECT, name=RUN_NAME)

# %%
# Calculate steps - dataset already contains all epochs worth of data
total_samples = len(train_dataset)
total_steps = total_samples // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
EVAL_STEPS = total_steps // (EVAL_FREQUENCY * EPOCHS)
SAVE_STEPS = total_steps // (SAVE_FREQUENCY * EPOCHS)

print(f"Total samples (all epochs): {total_samples:,}")
print(f"Total steps: {total_steps:,}")
print(f"Eval steps: {EVAL_STEPS:,} ({EVAL_FREQUENCY} times per original epoch)")
print(f"Save steps: {SAVE_STEPS:,} ({SAVE_FREQUENCY} times per original epoch)")

# Choose loss function
train_loss: (
    losses.CachedMultipleNegativesRankingLoss | losses.MultipleNegativesRankingLoss
)
if USE_CACHED_LOSS:
    train_loss = losses.CachedMultipleNegativesRankingLoss(
        model=model, mini_batch_size=MINI_BATCH_SIZE
    )
    print(
        f"Using CachedMultipleNegativesRankingLoss: {BATCH_SIZE - 1} negatives (2x slower)"
    )
else:
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    print(f"Using MultipleNegativesRankingLoss: {BATCH_SIZE - 1} negatives")

args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # Dataset already contains all epochs
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    optim="adamw_torch",
    fp16=False,
    bf16=True,
    max_grad_norm=MAX_GRAD_NORM,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    report_to="wandb",
    run_name=RUN_NAME,
    push_to_hub=True,
    hub_model_id=MODEL_REPO_ID,
    hub_private_repo=True,
    hub_strategy="every_save",
)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    args=args,
)

trainer.train()

wandb.finish()
