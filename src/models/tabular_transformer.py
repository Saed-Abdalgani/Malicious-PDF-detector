"""FT-Transformer-style neural candidate for numerical PDF telemetry."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


class GEGLU(nn.Module):
    """Gated GELU feed-forward activation used by tabular transformer blocks."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = value.chunk(2, dim=-1)
        return content * functional.gelu(gate)


class FTTransformerBlock(nn.Module):
    """Pre-normalized attention and GEGLU residual block."""

    def __init__(
        self,
        token_dimension: int,
        heads: int,
        feedforward_multiplier: float,
        attention_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        hidden = max(8, int(token_dimension * feedforward_multiplier))
        self.attention_norm = nn.LayerNorm(token_dimension)
        self.attention = nn.MultiheadAttention(
            token_dimension,
            heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(residual_dropout)
        self.feedforward_norm = nn.LayerNorm(token_dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(token_dimension, hidden * 2),
            GEGLU(),
            nn.Dropout(residual_dropout),
            nn.Linear(hidden, token_dimension),
            nn.Dropout(residual_dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class FTTransformer(nn.Module):
    """Numerical feature tokenization plus feature-aware self-attention.

    Each scalar is projected with its own weight/bias and receives a learned
    feature-identity embedding. A CLS token is processed by pre-norm residual
    attention/GEGLU blocks and mapped to one malicious-class logit.
    """

    def __init__(
        self,
        input_dimension: int,
        *,
        token_dimension: int = 64,
        blocks: int = 3,
        heads: int = 8,
        feedforward_multiplier: float = 2.0,
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dimension < 1:
            raise ValueError("input_dimension must be positive.")
        if token_dimension % heads:
            raise ValueError("token_dimension must be divisible by heads.")
        if blocks < 1:
            raise ValueError("blocks must be positive.")
        self.input_dimension = int(input_dimension)
        self.token_dimension = int(token_dimension)
        self.numeric_weight = nn.Parameter(
            torch.empty(input_dimension, token_dimension)
        )
        self.numeric_bias = nn.Parameter(torch.zeros(input_dimension, token_dimension))
        self.feature_identity = nn.Parameter(
            torch.empty(input_dimension, token_dimension)
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, token_dimension))
        self.blocks = nn.ModuleList(
            FTTransformerBlock(
                token_dimension,
                heads,
                feedforward_multiplier,
                attention_dropout,
                residual_dropout,
            )
            for _ in range(blocks)
        )
        self.output_norm = nn.LayerNorm(token_dimension)
        self.output = nn.Linear(token_dimension, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.token_dimension)
        nn.init.uniform_(self.numeric_weight, -bound, bound)
        nn.init.normal_(self.feature_identity, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.numeric_bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def tokenize(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dimension:
            raise ValueError(
                f"Expected [batch, {self.input_dimension}] numerical input; "
                f"received {tuple(values.shape)}."
            )
        numeric = (
            values.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
            + self.numeric_bias.unsqueeze(0)
            + self.feature_identity.unsqueeze(0)
        )
        cls = self.cls_token.expand(values.shape[0], -1, -1)
        return torch.cat((cls, numeric), dim=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenize(values)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output(self.output_norm(tokens[:, 0]))

    @torch.no_grad()
    def predict_proba(
        self, values: torch.Tensor, *, batch_size: int = 8_192
    ) -> torch.Tensor:
        self.eval()
        outputs: list[torch.Tensor] = []
        for start in range(0, len(values), batch_size):
            outputs.append(torch.sigmoid(self(values[start : start + batch_size])))
        return torch.cat(outputs, dim=0)


class AsymmetricFocalLoss(nn.Module):
    """Weighted binary focal loss with separate positive/negative focusing."""

    def __init__(
        self,
        *,
        positive_weight: float = 1.0,
        gamma_positive: float = 1.0,
        gamma_negative: float = 4.0,
    ) -> None:
        super().__init__()
        if positive_weight <= 0 or gamma_positive < 0 or gamma_negative < 0:
            raise ValueError("Focal-loss weights and gamma values are invalid.")
        self.positive_weight = float(positive_weight)
        self.gamma_positive = float(gamma_positive)
        self.gamma_negative = float(gamma_negative)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype).reshape_as(logits)
        base = functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probability = torch.sigmoid(logits)
        probability_true = targets * probability + (1.0 - targets) * (1.0 - probability)
        gamma = (
            targets * self.gamma_positive
            + (1.0 - targets) * self.gamma_negative
        )
        class_weight = targets * self.positive_weight + (1.0 - targets)
        return (base * torch.pow(1.0 - probability_true, gamma) * class_weight).mean()
