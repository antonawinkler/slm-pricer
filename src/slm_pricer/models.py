"""
Neural network architectures for price prediction from embeddings.

This module contains various architectures for predicting prices from
pre-computed language model embeddings.

The MLP and ResNetSimpleNN classes are adapted from the RTDL
(Revisiting Deep Learning Models for Tabular Data) project:
    https://github.com/yandex-research/rtdl-revisiting-models

Original work Copyright 2021 Yandex LLC, licensed under Apache License 2.0.
See: https://www.apache.org/licenses/LICENSE-2.0

Modifications made for this project include parameter renaming and
simplification for the price prediction use case.
"""

import torch.nn as nn
from torch import Tensor


class MLPBlock(nn.Module):
    """Single MLP block: Dropout(ReLU(Linear(x))), optionally with skip connection."""

    def __init__(self, d: int, dropout: float, skip: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = skip

    def forward(self, x: Tensor) -> Tensor:
        if self.skip:
            return x + self.block(x)
        return self.block(x)


class MLP(nn.Module):
    """
    A concrete MLP implementation using nn.Module style (no F.functional).
    Only supports numerical features. Optionally supports skip connections.

    Architecture (skip_connection=False):
        MLP(x) = Linear(Block(...(Block(Linear(x)))))
        Block(x) = Dropout(ReLU(Linear(x)))

    Architecture (skip_connection=True):
        MLP(x) = Linear(Block(...(Block(Linear(x)))))
        Block(x) = x + Dropout(ReLU(Linear(x)))
    """

    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        n_layers: int,
        dropout: float,
        d_out: int,
        skip_connection: bool = False,
    ) -> None:
        super().__init__()

        self.first_layer = nn.Linear(d_in, d)

        self.layers = nn.Sequential(
            *[MLPBlock(d, dropout, skip=skip_connection) for _ in range(n_layers)]
        )

        self.head = nn.Linear(d, d_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.layers(x)
        x = self.head(x)
        return x


class PostActBlock(nn.Module):
    """
    Single-Layer Residual Block (Post-Activation).

    Architecture:
        x -> [Linear -> BN -> ReLU -> Dropout] -> + -> Output
             |____________________________________|

    References:
        - Chen et al. (2020). "Deep Residual Learning for Nonlinear Regression."
          Entropy, 22(2). Proposed replacing Conv layers with Dense layers for regression.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super(PostActBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class PriceNetPostAct(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 5000,
        n_blocks: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super(PriceNetPostAct, self).__init__()

        # Project raw embedding to hidden dimension
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual Backbone
        self.res_blocks = nn.Sequential(
            *[PostActBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = self.entry(x)
        x = self.res_blocks(x)
        return self.head(x)


class PreActBlock(nn.Module):
    """
    Single-Layer Pre-Activation Residual Block (Euler Step).

    Architecture:
        x -> [BN -> GELU -> Dropout -> Linear] -> + -> Output
             |____________________________________|

    References:
        - He, K., et al. (2016). "Identity Mappings in Deep Residual Networks."
          ECCV. Proved that Pre-Activation (BN before Weight) improves signal propagation.
        - Chen, R. T. Q., et al. (2018). "Neural Ordinary Differential Equations."
          NeurIPS. Formulates this block structure as a first-order Euler discretization:
          x_{t+1} = x_t + f(x_t).
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super(PreActBlock, self).__init__()

        # Normalization & Activation happen *before* the weight matrix
        self.norm_act = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),  # GELU is smoother than ReLU, better for regression
            nn.Dropout(dropout),
        )

        # The weight layer is the final operation of the block
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.norm_act(x)
        out = self.linear(out)
        return x + out


class PriceNetPreAct(nn.Module):
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 5000,
        n_blocks: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super(PriceNetPreAct, self).__init__()

        # Entry: Pure Linear Projection (Preserves raw embedding structure)
        self.entry = nn.Linear(input_dim, hidden_dim)

        # Residual Backbone (Euler Steps)
        self.res_blocks = nn.Sequential(
            *[PreActBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # Head: Must include Normalization first because PreAct blocks end un-normalized
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = self.entry(x)
        x = self.res_blocks(x)
        return self.head(x)


class ResNetBlock(nn.Module):
    """Single ResNet block: x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))"""

    def __init__(
        self, d: int, d_hidden: int, hidden_dropout: float, residual_dropout: float
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, d_hidden),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(d_hidden, d),
            nn.Dropout(residual_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResNetSimpleNN(nn.Module):
    """
    A concrete ResNet implementation using nn.Module style (no F.functional).

    Architecture:
        ResNet(x) = Prediction(ResNetBlock(...(ResNetBlock(Linear(x)))))
        ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))
        Prediction(x) = Linear(ReLU(BatchNorm(x)))
    """

    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        d_hidden: int,
        n_layers: int,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        self.first_layer = nn.Linear(d_in, d)

        self.layers = nn.Sequential(
            *[
                ResNetBlock(d, d_hidden, hidden_dropout, residual_dropout)
                for _ in range(n_layers)
            ]
        )

        self.prediction = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Linear(d, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.layers(x)
        x = self.prediction(x)
        return x.squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResidualNet(nn.Module):
    def __init__(
        self,
        dropout: float = 0.3,
        n_layers: int = 1,
        hidden_dim: int = 5000,
    ) -> None:
        super().__init__()
        input_dim = 3072 * n_layers

        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout), ResidualBlock(hidden_dim, dropout)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = self.entry(x)
        x = self.res_blocks(x)
        return self.head(x)
