"""
PyTorch training with per-sample training dynamics for TrainStream.

Separates batch-level and epoch-level training to allow users to easily
implement custom training dynamics. The default_train_fn aggregates
confidence, AUM (margin), and forgetting events across epochs.

All functions operate on numpy arrays (X, y) and handle conversion
to/from torch tensors internally.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional, List


def auto_detect_device() -> torch.device:
    """
    Returns the best available device: CUDA > MPS > CPU.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_batch(
    model: nn.Module,
    batch_X: torch.Tensor,
    batch_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> tuple:
    """
    Single forward + backward pass on one mini-batch.

    Args:
        model: PyTorch model.
        batch_X: Tensor (batch_size, n_features).
        batch_y: Tensor (batch_size,) integer labels.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to use.

    Returns:
        logits: Tensor (batch_size, n_classes) — raw model output.
        loss_value: float — scalar loss.
    """
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()
    logits = model(batch_X)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()

    return logits, loss.item()


def train_one_epoch(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    batch_size: int = 256,
    binary: bool = False,
) -> Dict[str, Any]:
    """
    One full pass over the data in shuffled mini-batches.
    Computes per-sample metrics for training dynamics.

    Args:
        model: PyTorch model (must already be on device).
        X: np.ndarray (n_samples, n_features).
        y: np.ndarray (n_samples,) integer labels.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device.
        batch_size: Mini-batch size for SGD.
        binary: If True, use sigmoid for confidence. Otherwise softmax (default).

    Returns:
        Dict with:
            "confidence": np.ndarray (n_samples,) — P(correct class) this epoch.
            "margins": np.ndarray (n_samples,) — true logit minus 2nd-best logit.
            "predictions": np.ndarray (n_samples,) — predicted class this epoch.
            "loss": float — mean loss across all batches in this epoch.
    """
    n = len(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    indices = torch.arange(n)

    dataset = TensorDataset(X_tensor, y_tensor, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Accumulators for per-sample metrics (mapped back via original indices)
    confidence_all = np.zeros(n, dtype=np.float32)
    margins_all = np.zeros(n, dtype=np.float32)
    predictions_all = np.zeros(n, dtype=np.int64)
    losses = []

    model.train()

    for batch_X, batch_y, batch_idx in loader:
        logits, loss_val = train_one_batch(
            model, batch_X, batch_y, optimizer, criterion, device
        )
        losses.append(loss_val)

        # Compute per-sample metrics (no grad needed)
        with torch.no_grad():
            if binary:
                probs = torch.sigmoid(logits).squeeze(-1)
                # For binary: confidence = P(correct)
                conf = torch.where(
                    batch_y.to(device) == 1, probs, 1.0 - probs
                )
                preds = (probs > 0.5).long()
                # Margin: just the logit value for the true class
                margins = conf  # less meaningful for binary, but consistent
            else:
                probs = torch.softmax(logits, dim=1)
                # Confidence: P(correct class)
                conf = probs.gather(1, batch_y.to(device).unsqueeze(1)).squeeze(1)
                preds = probs.argmax(dim=1)

                # Margin: true class logit - 2nd best logit
                true_logits = logits.gather(
                    1, batch_y.to(device).unsqueeze(1)
                ).squeeze(1)
                logits_masked = logits.clone()
                logits_masked.scatter_(
                    1, batch_y.to(device).unsqueeze(1), float("-inf")
                )
                second_best = logits_masked.max(dim=1).values
                margins = true_logits - second_best

            # Write back to accumulators using original indices
            idx = batch_idx.numpy()
            confidence_all[idx] = conf.cpu().numpy()
            margins_all[idx] = margins.cpu().numpy()
            predictions_all[idx] = preds.cpu().numpy()

    return {
        "confidence": confidence_all,
        "margins": margins_all,
        "predictions": predictions_all,
        "loss": sum(losses) / len(losses) if losses else 0.0,
    }


def default_train_fn(
    X: np.ndarray,
    y: np.ndarray,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    epochs: int = 5,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    binary: bool = False,
) -> Dict[str, Any]:
    """
    Default train_fn for StreamLoop. Trains for multiple epochs and aggregates
    per-sample training dynamics across all epochs.

    Computes:
    - confidence: mean P(correct class) across epochs
    - aum: mean margin (true logit - 2nd best) across epochs (Area Under Margin)
    - forgetting: count of correct→incorrect transitions across consecutive epochs

    Args:
        X: np.ndarray (n_samples, n_features).
        y: np.ndarray (n_samples,) integer labels.
        model: PyTorch model.
        optimizer: Optimizer.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        epochs: Number of epochs per streaming step (default 5).
        batch_size: Mini-batch size for SGD (default 256).
        device: Device to use (default: auto-detect).
        binary: If True, use sigmoid instead of softmax (default False).

    Returns:
        train_info: dict with:
            "confidence": np.ndarray (n,) — mean P(correct) across epochs.
            "aum": np.ndarray (n,) — mean margin across epochs.
            "forgetting": np.ndarray (n,) int — forgetting event counts.
            "avg_loss": float — mean loss across all epochs.
            "per_epoch": list[dict] — raw per-epoch info for advanced analysis.
    """
    if device is None:
        device = auto_detect_device()

    model.to(device)

    n = len(X)
    confidence_sum = np.zeros(n, dtype=np.float32)
    margin_sum = np.zeros(n, dtype=np.float32)
    forgetting_counts = np.zeros(n, dtype=np.int32)
    prev_correct = None

    per_epoch = []
    total_loss = 0.0

    for epoch in range(epochs):
        epoch_info = train_one_epoch(
            model, X, y, optimizer, criterion, device,
            batch_size=batch_size, binary=binary,
        )
        per_epoch.append(epoch_info)

        confidence_sum += epoch_info["confidence"]
        margin_sum += epoch_info["margins"]
        total_loss += epoch_info["loss"]

        # Forgetting events: was correct last epoch, incorrect this epoch
        current_correct = epoch_info["predictions"] == y
        if prev_correct is not None:
            forgetting_events = prev_correct & ~current_correct
            forgetting_counts += forgetting_events.astype(np.int32)
        prev_correct = current_correct

    return {
        "confidence": confidence_sum / epochs,
        "aum": margin_sum / epochs,
        "forgetting": forgetting_counts,
        "avg_loss": total_loss / epochs,
        "per_epoch": per_epoch,
    }
