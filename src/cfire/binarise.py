"""Binarisation of attribution scores: which features count as relevant for a sample (CFIRE's itemset input).

Every binariser maps an explanation matrix (n_samples, n_features) to a boolean matrix of the same shape and can be
passed to CFIRE as `expl_binarization_fn` (e.g. `functools.partial(binarise_topk, k=3)`).
Only positive scores can be relevant: a feature that speaks against the explained class is not part of its pattern.

- `binarise_threshold` (CFIRE default): normalised score >= threshold.
- `binarise_topk`: the k highest scores per sample, if positive.
- `binarise_topk_abs`: the k largest scores by magnitude per sample, if positive (a large negative score takes a
  place in the top k without becoming relevant).
"""
from __future__ import annotations

import numpy as np


def normalise(explanations: np.ndarray) -> np.ndarray:
    """each row divided by its largest absolute value (rows of zeros stay zero); returns a float copy"""
    e = np.array(explanations, dtype=float)
    peak = np.max(np.abs(e), axis=-1, keepdims=True)
    peak[peak == 0] = 1.0
    return e / peak


def relevance_scores(explanations: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """normalised scores with everything below `threshold` set to 0 (all negative scores for threshold > 0);
    binarise_threshold marks the non-zero entries. Also used for CFIRE(item_order="relevance")."""
    e = normalise(explanations)
    e[e < threshold] = 0.0
    return e


def binarise_threshold(explanations: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Binary relevance matrix from attribution scores (default binariser of CFIRE).

    Each row is divided by its largest absolute value; features with a normalised score below `threshold`
    (including all negative scores) are not relevant. Returns a boolean array of the same shape.
    """
    return relevance_scores(explanations, threshold) > 0


def _k(k: int | float, n_features: int) -> int:
    """k as a count, or as a fraction of the features (0 < k < 1, rounded up)"""
    if 0 < k < 1:
        return int(np.ceil(k * n_features))
    if k < 1 or k != int(k):
        raise ValueError(f"k must be a positive integer or a fraction in (0, 1), got {k!r}")
    return int(k)


def _topk_mask(ranking: np.ndarray, k: int) -> np.ndarray:
    """boolean mask of the k highest entries of each row of `ranking` (ties: numpy argsort order)"""
    mask = np.zeros(ranking.shape, dtype=bool)
    top = np.argsort(ranking, axis=-1)[:, -k:]
    np.put_along_axis(mask, top, True, axis=-1)
    return mask


def binarise_topk(explanations: np.ndarray, k: int | float) -> np.ndarray:
    """the k highest (signed) scores per sample are relevant if they are positive; k: count or fraction of features"""
    e = normalise(explanations)
    return _topk_mask(e, _k(k, e.shape[-1])) & (e > 0)


def binarise_topk_abs(explanations: np.ndarray, k: int | float) -> np.ndarray:
    """as binarise_topk, but the top k are ranked by magnitude; only the positive ones among them are relevant"""
    e = normalise(explanations)
    return _topk_mask(np.abs(e), _k(k, e.shape[-1])) & (e > 0)
