"""
Coreset selection methods for TrainStream.

All selection functions operate on scores (np.ndarray) and return indices.
The stratified wrapper handles splitting by class, calling the inner
selection function per class, and mapping indices back to the global array.

Staleness prevention:
- max_chunk_fraction: caps how much of the coreset any single chunk can occupy.
"""

import numpy as np
from typing import Callable, Optional, Dict


# ---------------------------------------------------------------------------
# Core selection functions
# All receive: scores (n_samples,), m (budget), **kwargs
# All return: np.ndarray of selected indices (into the scores array)
# ---------------------------------------------------------------------------


def select_random(scores: np.ndarray, m: int, **kwargs) -> np.ndarray:
    """
    Random baseline selection. Ignores scores entirely.

    Args:
        scores: (n_samples,) array — ignored.
        m: Number of samples to select.

    Returns:
        np.ndarray of m randomly chosen indices.
    """
    n = len(scores)
    m = min(m, n)
    return np.random.choice(n, size=m, replace=False)


def select_top(scores: np.ndarray, m: int, **kwargs) -> np.ndarray:
    """
    Select the indices with the highest scores.
    Use with confidence or AUM to keep the most "learned" samples.

    Args:
        scores: (n_samples,) array of per-sample scores.
        m: Number of samples to select.

    Returns:
        np.ndarray of m indices with highest scores.
    """
    n = len(scores)
    m = min(m, n)
    return np.argsort(scores)[-m:]


def select_by_forgetting(scores: np.ndarray, m: int, **kwargs) -> np.ndarray:
    """
    Sample proportional to forgetting counts.
    Higher forgetting count = more likely to be selected (most informative).

    Falls back to uniform sampling if all counts are zero.

    Args:
        scores: (n_samples,) array of forgetting event counts.
        m: Number of samples to select.

    Returns:
        np.ndarray of m selected indices.
    """
    n = len(scores)
    m = min(m, n)

    total = scores.sum()
    if total == 0:
        # No forgetting events: fall back to uniform
        return np.random.choice(n, size=m, replace=False)

    probs = scores.astype(np.float64)
    probs /= probs.sum()

    replace = m > (scores > 0).sum()
    return np.random.choice(n, size=m, replace=replace, p=probs)


# ---------------------------------------------------------------------------
# Logistic-weighted confidence sampling
# Ported from github.com/Yam-Arieli/confidence-sampling
# ---------------------------------------------------------------------------


def _logistic_curve(c: np.ndarray, k: float, c0: float) -> np.ndarray:
    """Numerator of logistic curve: 1 / (1 + exp(-k*(c - c0)))."""
    return 1.0 / (1.0 + np.exp(-k * (c - c0)))


def _compute_steepness(m: int, n: int, pi: float = 0.2,
                       k0: float = 1.0, epsilon: float = 1e-6) -> float:
    """Compute logistic steepness parameter k based on sample/budget ratio."""
    ratio = np.sqrt(m / (n + epsilon))
    k = k0 * ratio / (pi + epsilon)
    return max(k, 1.0)


def weighted_sample_from_confidence(
    scores: np.ndarray,
    m: int,
    k: Optional[float] = None,
    c0: float = 0.5,
    k0: float = 1.0,
    pi: float = 0.2,
    epsilon: float = 1e-6,
    **kwargs,
) -> np.ndarray:
    """
    Logistic-weighted quantile sampling biased toward high confidence.

    Applies a logistic curve to confidence scores, normalizes into weights,
    then draws one sample from each of m quantile bins. This ensures coverage
    across the full score distribution while biasing toward high-confidence samples.

    Args:
        scores: (n_samples,) confidence values (typically in [0, 1]).
        m: Number of samples to select.
        k: Logistic steepness. If None, computed from m/n ratio.
        c0: Logistic midpoint (default 0.5).
        k0: Base steepness scaling (default 1.0).
        pi: Steepness denominator (default 0.2).
        epsilon: Numerical stability constant.

    Returns:
        np.ndarray of m selected indices.
    """
    n = len(scores)
    m = min(m, n)

    if k is None:
        k = _compute_steepness(m, n, pi=pi, k0=k0, epsilon=epsilon)

    weights = _logistic_curve(scores, k, c0)
    weights = np.clip(weights, epsilon, None)

    # Normalize and transform into sampling weights
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min < epsilon:
        # All weights are equal: fall back to uniform
        probs = np.ones(n, dtype=np.float64)
    else:
        probs = (weights - w_min) / (w_max - w_min)
    probs = np.power(probs, 0.5)

    # Quantile-stratified sampling: one sample per quantile bin
    quantiles = np.linspace(0, 1, m + 1)
    sample_indices = []
    all_indices = np.arange(n)

    for i in range(m):
        q_min = np.quantile(probs, quantiles[i])
        q_max = np.quantile(probs, quantiles[i + 1])

        mask = (probs >= q_min) & (probs <= q_max)
        candidates = all_indices[mask]

        if len(candidates) > 0:
            chosen = np.random.choice(candidates, size=1, replace=False)
            sample_indices.append(int(chosen[0]))

    return np.array(sample_indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Stratified selection wrapper
# ---------------------------------------------------------------------------


def compute_class_budget(
    y: np.ndarray, m: int, min_per_class: int = 1
) -> Dict[int, int]:
    """
    Compute per-class sample counts proportional to class frequency.

    Args:
        y: (n_samples,) integer labels.
        m: Total coreset budget.
        min_per_class: Minimum samples per class (default 1).

    Returns:
        Dict mapping class_label -> count.

    Raises:
        ValueError: If m is too small to give min_per_class to each class.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    if m < n_classes * min_per_class:
        raise ValueError(
            f"Cannot allocate {m} samples with min_per_class={min_per_class} "
            f"across {n_classes} classes (need at least {n_classes * min_per_class})."
        )

    # Start with minimum allocation
    budget = {c: min_per_class for c in classes}
    remaining = m - sum(budget.values())

    # Distribute remaining proportionally
    freqs = counts / counts.sum()
    extra = np.floor(freqs * remaining).astype(int)
    for c, e in zip(classes, extra):
        budget[c] += e

    # Distribute leftover (due to floor rounding)
    leftover = m - sum(budget.values())
    if leftover > 0:
        # Give to classes with largest fractional remainders
        fractions = freqs * remaining - np.floor(freqs * remaining)
        top_classes = classes[np.argsort(-fractions)[:leftover]]
        for c in top_classes:
            budget[c] += 1

    return budget


def _enforce_chunk_cap(
    indices: np.ndarray,
    chunk_ids: np.ndarray,
    class_budget: int,
    max_chunk_fraction: float,
    all_scores: np.ndarray,
) -> np.ndarray:
    """
    Enforce max_chunk_fraction constraint on selected indices within one class.

    If any single chunk contributes more than max_chunk_fraction of the class
    budget, trim it and backfill with next-best samples from other chunks.
    If backfilling can't reach the budget (not enough candidates from other
    chunks), the cap is relaxed to fill the budget.

    All arrays are in class-local index space (i.e., chunk_ids[i] is the
    chunk ID of the i-th sample in this class).

    Args:
        indices: Initially proposed indices (class-local).
        chunk_ids: (n_class_samples,) chunk IDs for all samples in this class.
        class_budget: Total budget for this class.
        max_chunk_fraction: Max fraction per chunk.
        all_scores: (n_class_samples,) scores for all samples in this class.

    Returns:
        np.ndarray of adjusted class-local indices.
    """
    max_per_chunk = max(1, int(class_budget * max_chunk_fraction))
    selected_chunk_ids = chunk_ids[indices]

    # Check which chunks are over budget
    unique_chunks, chunk_counts = np.unique(selected_chunk_ids, return_counts=True)
    over_budget_chunks = set(unique_chunks[chunk_counts > max_per_chunk].tolist())

    if not over_budget_chunks:
        return indices

    # Phase 1: Keep up to max_per_chunk from over-budget chunks, keep all others
    keep = []
    chunk_counter = {}

    for idx in indices:
        cid = int(chunk_ids[idx])
        chunk_counter.setdefault(cid, 0)
        if cid in over_budget_chunks and chunk_counter[cid] >= max_per_chunk:
            continue
        keep.append(idx)
        chunk_counter[cid] += 1

    # Phase 2: Backfill from remaining candidates, respecting cap
    if len(keep) < class_budget:
        selected_set = set(keep)
        score_order = np.argsort(-all_scores)
        for candidate in score_order:
            if len(keep) >= class_budget:
                break
            if candidate in selected_set:
                continue
            cid = int(chunk_ids[candidate])
            chunk_counter.setdefault(cid, 0)
            if chunk_counter[cid] < max_per_chunk:
                keep.append(candidate)
                selected_set.add(candidate)
                chunk_counter[cid] += 1

    # Phase 3: If still short, relax the cap — fill by score regardless of chunk
    # This ensures we always meet the budget when possible.
    if len(keep) < class_budget:
        selected_set = set(keep)
        score_order = np.argsort(-all_scores)
        for candidate in score_order:
            if len(keep) >= class_budget:
                break
            if candidate not in selected_set:
                keep.append(candidate)
                selected_set.add(candidate)

    return np.array(keep[:class_budget], dtype=np.int64)


def select_stratified(
    X: np.ndarray,
    y: np.ndarray,
    m: int,
    scores: np.ndarray,
    sample_fn: Callable = select_top,
    min_per_class: int = 1,
    chunk_ids: Optional[np.ndarray] = None,
    max_chunk_fraction: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    """
    Stratified coreset selection: proportional budget per class, then apply
    sample_fn within each class.

    Args:
        X: (n_samples, n_features) feature matrix.
        y: (n_samples,) integer labels.
        m: Total coreset budget.
        scores: (n_samples,) per-sample scores for selection.
        sample_fn: Selection function to apply within each class.
                   Signature: (scores, m, **kwargs) -> indices.
                   Default: select_top.
        min_per_class: Minimum samples per class.
        chunk_ids: (n_samples,) optional — which chunk each sample came from.
                   Required if max_chunk_fraction is set.
        max_chunk_fraction: Optional float in (0, 1]. Max fraction of each
                           class's budget that any single chunk can occupy.
                           Set to e.g. 0.3 to prevent staleness.
        **kwargs: Passed through to sample_fn.

    Returns:
        np.ndarray of selected global indices (into X/y/scores).
    """
    n = len(X)
    if n <= m:
        return np.arange(n)

    budget = compute_class_budget(y, m, min_per_class=min_per_class)
    selected_global = []

    for cls, cls_budget in budget.items():
        if cls_budget <= 0:
            continue

        # Get indices of this class in the global array
        class_mask = y == cls
        global_indices = np.where(class_mask)[0]

        if len(global_indices) == 0:
            continue

        # Slice scores for this class
        class_scores = scores[global_indices]
        effective_budget = min(cls_budget, len(global_indices))

        # Apply inner selection function (returns local indices)
        local_indices = sample_fn(class_scores, effective_budget, **kwargs)
        local_indices = np.asarray(local_indices, dtype=np.int64)

        # Apply chunk cap if enabled
        if max_chunk_fraction is not None and chunk_ids is not None:
            class_chunk_ids = chunk_ids[global_indices]
            local_indices = _enforce_chunk_cap(
                local_indices,
                class_chunk_ids,
                effective_budget,
                max_chunk_fraction,
                class_scores,
            )

        # Map back to global indices
        selected_global.append(global_indices[local_indices])

    if not selected_global:
        return np.array([], dtype=np.int64)

    result = np.concatenate(selected_global)

    # Final trim if we overshoot due to rounding
    if len(result) > m:
        result = np.random.choice(result, size=m, replace=False)

    return result
