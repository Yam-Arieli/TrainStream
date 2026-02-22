"""
Coreset provenance tracking for TrainStream.

Records which samples are in the coreset at each streaming step,
where they came from (chunk_id, within_chunk_index), and their
training dynamics scores at selection time.
"""

import numpy as np
from collections import Counter
from typing import Dict, Optional, Any


class CoresetTracker:
    """
    Tracks coreset composition and provenance across streaming steps.

    After each selection step, records:
    - Which chunk each coreset sample originally came from
    - The sample's index within its original chunk
    - Training dynamics scores (confidence, AUM, etc.) at selection time

    This enables post-hoc analysis: Which samples persisted across steps?
    Which chunks dominate the coreset? How do scores evolve?

    Example:
        tracker = CoresetTracker()
        # ... after streaming loop ...
        print(tracker.chunk_distribution())   # Counter({0: 150, 3: 100, ...})
        df = tracker.to_dataframe()           # Full history as DataFrame
    """

    def __init__(self):
        self.history = []

    def record(
        self,
        step: int,
        chunk_ids: np.ndarray,
        within_chunk_idx: np.ndarray,
        scores: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """
        Record coreset composition after a selection step.

        Args:
            step: Current streaming step index.
            chunk_ids: (m,) array — which chunk each coreset sample came from.
            within_chunk_idx: (m,) array — original index within that chunk.
            scores: (m,) array — the scores used for selection (optional).
            extra: Optional dict of additional per-step metadata
                   (e.g., avg_loss, eval metrics).
        """
        entry = {
            "step": step,
            "chunk_ids": chunk_ids.copy(),
            "within_chunk_idx": within_chunk_idx.copy(),
        }
        if scores is not None:
            entry["scores"] = scores.copy()
        if extra is not None:
            entry["extra"] = extra
        self.history.append(entry)

    def chunk_distribution(self, step: int = -1) -> Counter:
        """
        Returns a Counter of chunk_ids in the coreset at a given step.

        Args:
            step: Index into history (default -1 = final step).

        Returns:
            Counter mapping chunk_id -> count.
        """
        if not self.history:
            return Counter()
        return Counter(self.history[step]["chunk_ids"].tolist())

    def get_final_provenance(self) -> Dict[str, np.ndarray]:
        """
        Returns the provenance of the final coreset.

        Returns:
            Dict with keys:
                "chunk_ids": np.ndarray (m,)
                "within_chunk_idx": np.ndarray (m,)
                "scores": np.ndarray (m,) if recorded
        """
        if not self.history:
            return {}
        final = self.history[-1]
        result = {
            "chunk_ids": final["chunk_ids"],
            "within_chunk_idx": final["within_chunk_idx"],
        }
        if "scores" in final:
            result["scores"] = final["scores"]
        return result

    def to_dataframe(self):
        """
        Export full history as a pandas DataFrame for analysis.

        Each row represents one coreset sample at one step.
        Columns: step, chunk_id, within_chunk_idx, score (if available).

        Returns:
            pandas.DataFrame
        """
        import pandas as pd

        rows = []
        for entry in self.history:
            step = entry["step"]
            chunk_ids = entry["chunk_ids"]
            within_idx = entry["within_chunk_idx"]
            scores = entry.get("scores")

            for i in range(len(chunk_ids)):
                row = {
                    "step": step,
                    "chunk_id": chunk_ids[i],
                    "within_chunk_idx": within_idx[i],
                }
                if scores is not None:
                    row["score"] = scores[i]
                rows.append(row)

        return pd.DataFrame(rows)

    def __len__(self):
        return len(self.history)
