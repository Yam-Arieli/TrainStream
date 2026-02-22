"""
StreamLoop — the main engine of TrainStream.

Orchestrates the streaming coreset training pipeline:
    stream chunk → merge with buffer → train → select coreset → repeat

All data flows as (X, y) numpy array pairs internally.
"""

import threading
import queue
import numpy as np
import torch.nn as nn
from typing import Optional, Callable, Dict, Any
from tqdm import tqdm

from .training import default_train_fn, auto_detect_device
from .selection import select_stratified, select_top
from .tracking import CoresetTracker


class StreamLoop:
    """
    Streaming coreset training engine.

    Pipeline per step:
        1. Load chunk (X_chunk, y_chunk) from stream (in background thread).
        2. Merge with coreset buffer via numpy concatenation.
           Track chunk_ids and within_chunk_idx for provenance.
        3. Train on merged data: train_fn(X, y, model, ...) -> train_info.
        4. Select best m samples: select_fn(X, y, m, scores, ...) -> indices.
        5. Update buffer: X_buffer, y_buffer = X[indices], y[indices].
        6. (Optional) Evaluate on held-out test set.
        7. Log to CoresetTracker.

    Args:
        stream_factory: Callable returning an iterator of (X, y) tuples.
        buffer_size: Coreset budget (m). Must be smaller than chunk size (k).
        model: PyTorch model (nn.Module).
        optimizer: PyTorch optimizer.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).

        train_fn: Custom training function. Default: default_train_fn.
                  Signature: (X, y, model, optimizer, criterion, **kwargs) -> train_info dict.
        epochs_per_step: Epochs to train per streaming step (default 5).
        batch_size: Mini-batch size for SGD (default 256).
        device: torch.device or None (auto-detect: CUDA > MPS > CPU).
        binary: If True, use sigmoid instead of softmax (default False).

        select_fn: Custom selection function. Default: select_stratified.
                   Signature: (X, y, m, scores, **kwargs) -> indices.
        sample_fn: Inner selection method for stratified selection.
                   Default: select_top.
        score_key: Which key in train_info to use as selection scores.
                   Options: "confidence", "aum", "forgetting". Default: "aum".
        max_chunk_fraction: Max fraction of coreset from any single chunk.
                           E.g., 0.3 means no chunk gets >30%. Default: None (no cap).
        confidence_decay: Decay factor for old buffer sample scores.
                         E.g., 0.9 means old scores are multiplied by 0.9 each step.
                         Default: None (no decay).
        min_per_class: Minimum samples per class in coreset (default 1).

        eval_fn: Optional evaluation callback. Created via make_eval_fn().
                 Signature: (model, device) -> dict.
        on_step_end: Optional callback after each step.
                     Signature: (step, train_info, eval_result) -> None.

    Example:
        from trainstream import StreamLoop
        from trainstream.streaming import from_numpy
        from trainstream.selection import weighted_sample_from_confidence

        engine = StreamLoop(
            stream_factory=from_numpy(X_train, y_train, chunk_size=2000),
            buffer_size=500,
            model=my_model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            score_key="confidence",
            sample_fn=weighted_sample_from_confidence,
            epochs_per_step=10,
        )
        result = engine.run()
    """

    def __init__(
        self,
        stream_factory: Callable,
        buffer_size: int,
        model: nn.Module,
        optimizer,
        criterion,
        # Training
        train_fn: Optional[Callable] = None,
        epochs_per_step: int = 5,
        batch_size: int = 256,
        device=None,
        binary: bool = False,
        # Selection
        select_fn: Optional[Callable] = None,
        sample_fn: Optional[Callable] = None,
        score_key: str = "aum",
        max_chunk_fraction: Optional[float] = None,
        confidence_decay: Optional[float] = None,
        min_per_class: int = 1,
        # Evaluation
        eval_fn: Optional[Callable] = None,
        # Callbacks
        on_step_end: Optional[Callable] = None,
    ):
        self.stream_factory = stream_factory
        self.m = buffer_size
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # Training config
        self.train_fn = train_fn or default_train_fn
        self.epochs_per_step = epochs_per_step
        self.batch_size = batch_size
        self.device = device if device is not None else auto_detect_device()
        self.binary = binary

        # Selection config
        self.select_fn = select_fn or select_stratified
        self.sample_fn = sample_fn or select_top
        self.score_key = score_key
        self.max_chunk_fraction = max_chunk_fraction
        self.confidence_decay = confidence_decay
        self.min_per_class = min_per_class

        # Evaluation
        self.eval_fn = eval_fn

        # Callbacks
        self.on_step_end = on_step_end

        # Internal state
        self._buffer_X = None
        self._buffer_y = None
        self._buffer_chunk_ids = None
        self._buffer_within_idx = None
        self._buffer_scores = None  # for confidence_decay

    def _background_loader(self, generator, q, stop_event):
        """Runs in a background thread to prefetch chunks."""
        try:
            for batch in generator:
                if stop_event.is_set():
                    break
                q.put(batch)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)  # end-of-stream sentinel

    def run(self) -> Dict[str, Any]:
        """
        Execute the full streaming pipeline until the stream is exhausted.

        Returns:
            dict with:
                "X": np.ndarray — final coreset features.
                "y": np.ndarray — final coreset labels.
                "tracker": CoresetTracker — full provenance history.
                "eval_history": list[dict] — per-step eval results (if eval_fn provided).
                "train_history": list[dict] — per-step train_info summaries.
        """
        tracker = CoresetTracker()
        train_history = []
        eval_history = []

        # Set up background loading
        q = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        stream = self.stream_factory()
        loader_thread = threading.Thread(
            target=self._background_loader,
            args=(stream, q, stop_event),
            daemon=True,
        )
        loader_thread.start()

        self.model.to(self.device)
        step = 0

        try:
            while True:
                item = q.get()

                if isinstance(item, Exception):
                    raise item
                if item is None:
                    break  # end of stream

                X_chunk, y_chunk = item
                chunk_size = len(X_chunk)

                # --- Merge ---
                if self._buffer_X is None:
                    # First chunk: no buffer yet
                    X_merged = X_chunk
                    y_merged = y_chunk
                    chunk_ids = np.full(chunk_size, step, dtype=np.int32)
                    within_idx = np.arange(chunk_size, dtype=np.int32)
                else:
                    X_merged = np.concatenate([self._buffer_X, X_chunk], axis=0)
                    y_merged = np.concatenate([self._buffer_y, y_chunk], axis=0)
                    new_chunk_ids = np.full(chunk_size, step, dtype=np.int32)
                    new_within_idx = np.arange(chunk_size, dtype=np.int32)
                    chunk_ids = np.concatenate(
                        [self._buffer_chunk_ids, new_chunk_ids]
                    )
                    within_idx = np.concatenate(
                        [self._buffer_within_idx, new_within_idx]
                    )

                # --- Train ---
                train_info = self.train_fn(
                    X_merged, y_merged,
                    self.model, self.optimizer, self.criterion,
                    epochs=self.epochs_per_step,
                    batch_size=self.batch_size,
                    device=self.device,
                    binary=self.binary,
                )

                # Extract scores for selection
                scores = train_info[self.score_key].copy()

                # --- Confidence decay ---
                # Apply decay to old buffer samples' scores so new samples
                # can compete more easily
                if (
                    self.confidence_decay is not None
                    and self._buffer_X is not None
                    and self._buffer_scores is not None
                ):
                    buffer_len = len(self._buffer_X)
                    # Blend: new score partially, old decayed score partially
                    # This gives old samples a disadvantage proportional to age
                    scores[:buffer_len] *= self.confidence_decay

                # --- Select ---
                selected_indices = self.select_fn(
                    X_merged, y_merged, self.m, scores,
                    sample_fn=self.sample_fn,
                    min_per_class=self.min_per_class,
                    chunk_ids=chunk_ids,
                    max_chunk_fraction=self.max_chunk_fraction,
                )

                # --- Update buffer ---
                self._buffer_X = X_merged[selected_indices]
                self._buffer_y = y_merged[selected_indices]
                self._buffer_chunk_ids = chunk_ids[selected_indices]
                self._buffer_within_idx = within_idx[selected_indices]
                self._buffer_scores = scores[selected_indices]

                # --- Track ---
                tracker.record(
                    step=step,
                    chunk_ids=self._buffer_chunk_ids,
                    within_chunk_idx=self._buffer_within_idx,
                    scores=self._buffer_scores,
                    extra={"avg_loss": train_info.get("avg_loss")},
                )

                # --- Train history ---
                step_summary = {
                    "step": step,
                    "avg_loss": train_info.get("avg_loss"),
                    "merged_size": len(X_merged),
                    "buffer_size": len(self._buffer_X),
                }
                train_history.append(step_summary)

                # --- Evaluate ---
                eval_result = None
                if self.eval_fn is not None:
                    eval_result = self.eval_fn(self.model, self.device)
                    eval_result["step"] = step
                    eval_history.append(eval_result)

                # --- Callback ---
                if self.on_step_end is not None:
                    self.on_step_end(step, train_info, eval_result)

                step += 1

        except KeyboardInterrupt:
            stop_event.set()
            print("\nStream stopped by user.")

        print(
            f"TrainStream finished. {step} steps completed. "
            f"Final buffer: {len(self._buffer_X) if self._buffer_X is not None else 0} samples."
        )

        return {
            "X": self._buffer_X,
            "y": self._buffer_y,
            "tracker": tracker,
            "eval_history": eval_history,
            "train_history": train_history,
        }
