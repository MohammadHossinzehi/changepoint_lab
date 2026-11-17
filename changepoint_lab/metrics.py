"""Scoring a detected set of changepoints against the truth."""

from __future__ import annotations

import numpy as np


def hausdorff(true, pred) -> float:
    """Worst case localisation error between two changepoint sets."""
    a, b = np.asarray(sorted(true), float), np.asarray(sorted(pred), float)
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return float("inf")
    d = np.abs(a[:, None] - b[None, :])
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def precision_recall_f1(true, pred, margin: int = 5):
    """One to one matching within ``margin`` samples, maximum cardinality.

    Note that "match the closest pair first" is NOT optimal: with true
    [0, 3], pred [2, 5] and margin 2 it pairs 3 with 2 and strands both
    others. Because every window [t - margin, t + margin] has the same width,
    sweeping the true points left to right and giving each one the leftmost
    unused prediction inside its window is optimal (exchange argument), and
    the tests check it against brute force. Each true changepoint is matched
    at most once, so detecting a change twice is penalised.
    """
    true, pred = sorted(true), sorted(pred)
    if not pred and not true:
        return 1.0, 1.0, 1.0
    tp, j = 0, 0
    for t in true:
        while j < len(pred) and pred[j] < t - margin:
            j += 1
        if j < len(pred) and pred[j] <= t + margin:
            tp += 1
            j += 1
    precision = tp / len(pred) if pred else 1.0
    recall = tp / len(true) if true else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def labels(changepoints, n: int) -> np.ndarray:
    lab = np.zeros(n, dtype=np.int64)
    for c in sorted(changepoints):
        lab[c:] += 1
    return lab


def rand_index(true, pred, n: int) -> float:
    """Fraction of sample pairs on which the two segmentations agree about
    "same segment or not". Computed in O(n + K1*K2) from the contingency
    table instead of O(n^2) pairs."""
    if n < 2:
        return 1.0
    a, b = labels(true, n), labels(pred, n)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    comb = lambda v: (v * (v - 1) // 2).sum()
    same_both = comb(table)
    same_a, same_b = comb(table.sum(axis=1)), comb(table.sum(axis=0))
    total = n * (n - 1) // 2
    agree = total + 2 * same_both - same_a - same_b
    return float(agree / total)


def annotation_error(true, pred) -> int:
    return abs(len(pred) - len(true))


def score(true, pred, n: int, margin: int = 5) -> dict:
    p, r, f = precision_recall_f1(true, pred, margin)
    return {"precision": p, "recall": r, "f1": f, "hausdorff": hausdorff(true, pred),
            "rand_index": rand_index(true, pred, n),
            "annotation_error": annotation_error(true, pred)}
