"""
Neural network architectures for price prediction from embeddings.

This module contains various architectures for predicting prices from
pre-computed language model embeddings.
"""

import torch
import torch.nn as nn


class CompressionPriceNet(nn.Module):
    """Compression-based price prediction network with progressive dimensionality reduction.

    Uses funnel architecture: progressively compresses embeddings with BatchNorm and dropout.
    dropout1 = 2 * dropout_base (capped at 0.5), dropout2 = dropout_base.
    """

    def __init__(
        self, dropout_base: float = 0.15, initial_dim: int = 256, input_dim: int = 3072
    ):
        super(CompressionPriceNet, self).__init__()

        dropout1 = min(2 * dropout_base, 0.5)  # Cap at 0.5 to avoid over-regularization
        dropout2 = dropout_base

        self.net = nn.Sequential(
            nn.Linear(input_dim, initial_dim),
            nn.BatchNorm1d(initial_dim),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(initial_dim, initial_dim // 2),
            nn.BatchNorm1d(initial_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(initial_dim // 2, initial_dim // 4),
            nn.ReLU(),
            nn.Linear(initial_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        result: torch.Tensor = self.net(x_flat)
        return result


class PriceRegressor(nn.Module):
    """Simple deep price regression network."""

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim1: int = 1024,
        hidden_dim2: int = 256,
        dropout: float = 0.1,
    ):
        super(PriceRegressor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class ShallowPriceNet(nn.Module):
    """Shallow price regression network with single hidden layer."""

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super(ShallowPriceNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and dropout."""

    def __init__(self, hidden_dim: int, dropout: float):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.dropout(self.activation(x + self.block(x)))
        return result


class HighResPriceNet(nn.Module):
    """High-resolution price network with residual connections."""

    def __init__(
        self, dropout: float = 0.2, hidden_dim: int = 512, input_dim: int = 4096
    ):
        super(HighResPriceNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=dropout),
            ResidualBlock(hidden_dim, dropout=dropout),
            ResidualBlock(hidden_dim, dropout=dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        result: torch.Tensor = self.head(x)
        return result


class InvertedBottleneckBlock(nn.Module):
    """Inverted bottleneck block that expands features before compressing.

    Allows network to untangle features in higher dimensional space before projecting down.
    """

    def __init__(self, dim: int, expansion_factor: float = 1.5, dropout: float = 0.2):
        super().__init__()
        expanded_dim = int(dim * expansion_factor)

        self.block = nn.Sequential(
            # Expansion phase
            nn.Linear(dim, expanded_dim),
            nn.BatchNorm1d(expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Projection phase
            nn.Linear(expanded_dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.block(x)
        return result


class TransformerMLP(nn.Module):
    """Transformer-inspired MLP with inverted bottleneck blocks."""

    def __init__(
        self,
        input_dim: int = 4096,
        num_blocks: int = 3,
        expansion_factor: float = 1.5,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.model_dim = input_dim

        self.blocks = nn.Sequential(
            *[
                InvertedBottleneckBlock(
                    dim=self.model_dim,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(self.model_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        result: torch.Tensor = self.head(x)
        return result
