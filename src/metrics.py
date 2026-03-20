"""Ranking metrics: MAP, MRR, AP@k."""

import numpy as np
from sklearn.metrics import average_precision_score


def compute_map(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean Average Precision (single-query version = AP).

    For multi-query MAP, call per-query and average externally.
    Uses sklearn's average_precision_score (area under PR curve).

    Args:
        scores: [N] float array of sponsorship scores (higher = more sponsored).
        labels: [N] binary int/float array (1 = sponsored).

    Returns:
        AP score in [0, 1].
    """
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


def compute_mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean Reciprocal Rank.

    Args:
        scores: [N] float array.
        labels: [N] binary array.

    Returns:
        MRR in (0, 1].
    """
    order = np.argsort(-scores)
    ranked_labels = labels[order]
    hits = np.where(ranked_labels == 1)[0]
    if len(hits) == 0:
        return 0.0
    return float(1.0 / (hits[0] + 1))


def compute_ap_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """AP@k: Average Precision at cut-off k.

    Args:
        scores: [N] float array.
        labels: [N] binary array.
        k:      Cut-off value.

    Returns:
        AP@k in [0, 1].
    """
    if labels.sum() == 0:
        return 0.0

    order = np.argsort(-scores)[:k]
    ranked_labels = labels[order]

    precisions = []
    n_relevant = 0
    for i, rel in enumerate(ranked_labels, start=1):
        if rel == 1:
            n_relevant += 1
            precisions.append(n_relevant / i)

    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def compute_all_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    k_values: tuple[int, ...] = (10, 100, 1000, 10000),
) -> dict[str, float]:
    """Compute MAP, MRR, and AP@k for several k values.

    Args:
        scores: [N] float array.
        labels: [N] binary array.
        k_values: Cut-off values for AP@k.

    Returns:
        Dict with keys 'map', 'mrr', 'ap@10', 'ap@100', etc.
    """
    results = {
        "map": compute_map(scores, labels),
        "mrr": compute_mrr(scores, labels),
    }
    for k in k_values:
        results[f"ap@{k}"] = compute_ap_at_k(scores, labels, k)
    return results
