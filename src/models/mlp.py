"""
mlp.py
------
PyTorch MLP classifier for malicious PDF detection, specifically designed
for INT8 post-training quantization compatibility (Phase 5).

Architecture:
    Input (37) -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.3)
               -> Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.2)
               -> Linear(32)  -> BatchNorm -> ReLU
               -> Linear(1)   -> raw logit (BCEWithLogitsLoss)

Design decisions:
    - BatchNorm BEFORE activation: stabilises quantised inference.
    - Progressive width reduction (128 -> 64 -> 32): prevents over-
      parameterisation on a 10K-sample dataset.
    - Moderate dropout: regularises without starving capacity.
    - BCEWithLogitsLoss: numerically stable sigmoid + BCE combo.
    - All operations on CPU (``device='cpu'`` enforced).

Usage:
    from src.models.mlp import MaliciousPDFClassifier, PDFDataset, train_mlp
    model = MaliciousPDFClassifier()
    history = train_mlp(model, train_loader, val_loader)
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.config import (
    DEVICE,
    RANDOM_SEED,
    TRAINED_MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Enforce reproducibility
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PDFDataset(Dataset):
    """PyTorch Dataset for PDF feature vectors.

    Args:
        X: Feature matrix, shape ``(n_samples, n_features)``.
        y: Labels, shape ``(n_samples,)``.  If None, returns features only.
    """

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

    def __len__(self) -> int:
        return len(self.X)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class MaliciousPDFClassifier(nn.Module):
    """3-hidden-layer MLP for binary PDF classification.

    Architecture::

        Input(37) -> [Linear(128) -> BN -> ReLU -> Dropout(0.3)]
                  -> [Linear(64)  -> BN -> ReLU -> Dropout(0.2)]
                  -> [Linear(32)  -> BN -> ReLU]
                  -> Linear(1)  # raw logit

    The output is a **raw logit** (no sigmoid).  Use ``BCEWithLogitsLoss``
    during training and ``torch.sigmoid(output)`` at inference for
    probability estimates.

    Args:
        input_dim: Number of input features. Default 37.
        dropout1: Dropout rate after first hidden layer. Default 0.3.
        dropout2: Dropout rate after second hidden layer. Default 0.2.
    """

    def __init__(
        self,
        input_dim: int = 37,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
        )

        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
        )

        self.block3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Linear(32, 1)

        # Initialize weights (Kaiming for ReLU layers)
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming initialization for ReLU-based layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape ``(batch, input_dim)``.

        Returns:
            torch.Tensor: Raw logits, shape ``(batch, 1)``.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions (applies sigmoid).

        Args:
            x: Input tensor, shape ``(batch, input_dim)``.

        Returns:
            torch.Tensor: Probabilities, shape ``(batch,)``.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze(-1)
        return probs


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class _EarlyStopping:
    """Monitors validation loss and triggers early stop after patience epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mlp(
    model: MaliciousPDFClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    device: str = DEVICE,
) -> Dict[str, Any]:
    """Train the MLP classifier with early stopping.

    Args:
        model: ``MaliciousPDFClassifier`` instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        max_epochs: Maximum training epochs. Default 100.
        learning_rate: Initial learning rate. Default 1e-3.
        weight_decay: AdamW weight decay. Default 1e-4.
        patience: Early stopping patience. Default 10.
        save_path: Where to save the best checkpoint. Defaults to
                   ``models/trained/mlp_best.pt``.
        device: Torch device string. Default ``'cpu'``.

    Returns:
        dict: Training history with keys:
            - ``train_loss``: list of per-epoch training losses
            - ``val_loss``: list of per-epoch validation losses
            - ``val_accuracy``: list of per-epoch validation accuracies
            - ``val_f1``: list of per-epoch validation F1 scores
            - ``best_epoch``: epoch index with best validation loss
            - ``best_val_loss``: best validation loss achieved
            - ``training_time_sec``: total training wall-clock time
            - ``epochs_trained``: number of epochs before stopping
    """
    if save_path is None:
        save_path = TRAINED_MODELS_DIR / "mlp_best.pt"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # Early stopping
    early_stop = _EarlyStopping(patience=patience)

    # History tracking
    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "training_time_sec": 0.0,
        "epochs_trained": 0,
    }

    logger.info(
        f"Starting MLP training: {max_epochs} max epochs, "
        f"lr={learning_rate}, patience={patience}, device={device}"
    )

    start_time = time.perf_counter()

    for epoch in range(max_epochs):
        # --- Training phase ---
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_X.size(0)
            train_samples += batch_X.size(0)

        train_loss = train_loss_sum / train_samples

        # --- Validation phase ---
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                val_loss_sum += loss.item() * batch_X.size(0)
                val_samples += batch_X.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                all_preds.extend(preds.squeeze().cpu().numpy())
                all_labels.extend(batch_y.squeeze().cpu().numpy())

        val_loss = val_loss_sum / val_samples
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        val_accuracy = (all_preds == all_labels).mean()

        # F1 score
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Update scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(val_accuracy))
        history["val_f1"].append(float(val_f1))

        # Save best model
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), save_path)

        # Logging (every 10 epochs + first + last)
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch+1:3d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {lr:.6f}"
            )

        # Early stopping check
        if early_stop.step(val_loss):
            logger.info(
                f"  Early stopping triggered at epoch {epoch+1} "
                f"(patience={patience})"
            )
            break

    history["training_time_sec"] = time.perf_counter() - start_time
    history["epochs_trained"] = epoch + 1

    # Load best checkpoint
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    logger.info(
        f"MLP training complete in {history['training_time_sec']:.2f}s | "
        f"{history['epochs_trained']} epochs | "
        f"Best epoch: {history['best_epoch']+1} | "
        f"Best val loss: {history['best_val_loss']:.4f}"
    )

    return history


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        batch_size: Batch size. Default 64.

    Returns:
        (train_loader, val_loader) tuple.
    """
    train_dataset = PDFDataset(X_train, y_train)
    val_dataset = PDFDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


def load_mlp(
    path: Optional[Union[str, Path]] = None,
    input_dim: int = 37,
) -> MaliciousPDFClassifier:
    """Load a trained MLP from a checkpoint file.

    Args:
        path: Path to the ``.pt`` checkpoint. Defaults to
              ``models/trained/mlp_best.pt``.
        input_dim: Input feature dimension. Default 37.

    Returns:
        MaliciousPDFClassifier: The loaded model in eval mode.
    """
    if path is None:
        path = TRAINED_MODELS_DIR / "mlp_best.pt"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {path}")

    model = MaliciousPDFClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()

    logger.info(f"MLP loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing MLP on synthetic data...")

    np.random.seed(RANDOM_SEED)
    X = np.random.randn(500, 37).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

    train_loader, val_loader = create_data_loaders(
        X[:400], y[:400], X[400:], y[400:], batch_size=32
    )

    model = MaliciousPDFClassifier(input_dim=37)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_mlp(
        model, train_loader, val_loader, max_epochs=30, patience=10
    )

    print("\nTraining complete!")
    print(f"  Epochs: {history['epochs_trained']}")
    print(f"  Best epoch: {history['best_epoch']+1}")
    print(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"  Final val F1: {history['val_f1'][-1]:.4f}")
