"""
Optional evaluation hooks for TrainStream.

Creates evaluation callbacks that can be passed to StreamLoop to track
model performance on a held-out test set after each streaming step.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Optional, Callable, Any
from sklearn.metrics import accuracy_score, f1_score


def make_eval_fn(
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_map: Optional[Dict[str, int]] = None,
    batch_size: int = 256,
    metrics: Optional[List[str]] = None,
) -> Callable[[nn.Module, torch.device], Dict[str, Any]]:
    """
    Creates an evaluation callback for StreamLoop.

    The returned function runs inference on the test set and computes
    classification metrics. It is called after each streaming step
    if passed as eval_fn to StreamLoop.

    Args:
        X_test: (n_test, n_features) held-out test features.
        y_test: (n_test,) held-out test labels (integers).
        label_map: Optional dict mapping label strings to integers.
                   If provided, per-class F1 scores are reported with
                   string keys. Otherwise, integer keys are used.
        batch_size: Batch size for inference (default 256).
        metrics: List of metrics to compute. Options:
                 "accuracy", "f1_macro", "f1_per_class".
                 Default: all three.

    Returns:
        Callable[[nn.Module, torch.device], dict] — evaluation function.

    Example:
        eval_fn = make_eval_fn(X_test, y_test, label_map=CLASS_TO_INT)
        engine = StreamLoop(..., eval_fn=eval_fn)
    """
    if metrics is None:
        metrics = ["accuracy", "f1_macro", "f1_per_class"]

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_numpy = y_test.copy()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_labels = np.unique(y_numpy)
    if label_map is not None:
        int_to_label = {v: k for k, v in label_map.items()}
    else:
        int_to_label = None

    def eval_fn(model: nn.Module, device: torch.device) -> Dict[str, Any]:
        model.eval()
        all_preds = []

        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        all_preds = np.array(all_preds)
        result = {}

        if "accuracy" in metrics:
            result["accuracy"] = accuracy_score(y_numpy, all_preds)

        if "f1_macro" in metrics:
            result["f1_macro"] = f1_score(
                y_numpy, all_preds, average="macro",
                labels=all_labels, zero_division=0,
            )

        if "f1_per_class" in metrics:
            f1_per = f1_score(
                y_numpy, all_preds, average=None,
                labels=all_labels, zero_division=0,
            )
            if int_to_label is not None:
                result["f1_per_class"] = {
                    int_to_label.get(int(c), c): score
                    for c, score in zip(all_labels, f1_per)
                }
            else:
                result["f1_per_class"] = {
                    int(c): score for c, score in zip(all_labels, f1_per)
                }

        model.train()
        return result

    return eval_fn
