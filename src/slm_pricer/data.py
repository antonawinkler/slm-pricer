"""
Data loading and preprocessing utilities for price prediction.

This module contains dataset classes and functions for loading and preparing
data for training price prediction models.
"""

from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


class PriceDataset(Dataset):
    """PyTorch dataset for price regression from embeddings."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_embeddings(
    cache_path: Path | str,
    cache_filename_root: str = "llama_fine_tuned_",
    splits: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """Load pre-computed embeddings from disk and return dict mapping splits to arrays."""
    if splits is None:
        splits = ["train", "val", "test"]

    cache_path = Path(cache_path)
    embeddings = {}

    for split in splits:
        filepath = cache_path / f"{cache_filename_root}{split}.npy"
        if not filepath.exists():
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        embeddings[split] = np.load(filepath, mmap_mode="r")
        print(f"Loaded {split} embeddings: {embeddings[split].shape}")

    return embeddings


def save_embeddings(
    embeddings: dict[str, np.ndarray],
    cache_path: Path | str,
    cache_filename_root: str = "llama_fine_tuned_",
) -> None:
    """Save embeddings dict to disk as .npy files."""
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    for split, emb_array in embeddings.items():
        filepath = cache_path / f"{cache_filename_root}{split}.npy"
        np.save(filepath, emb_array)
        print(f"Saved {split} embeddings to {filepath}")


def load_data_from_hf(
    split: Literal["train", "val", "test"] = "train",
    percent: int = 100,
    dataset_name: str = "ed-donner/items_prompts_full",
) -> pd.DataFrame:
    """Load data from HuggingFace dataset and return as pandas DataFrame."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=f"{split}[:{percent}%]")
    dataset = cast(HFDataset, dataset)
    return cast(pd.DataFrame, dataset.to_pandas())
