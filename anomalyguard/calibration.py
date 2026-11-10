"""Benign-only calibration and cross-layer fusion.

No anomaly label is used here: thresholds and p-values come from benign validation
packets only, so the false-positive budget is set without knowing what attacks look like.
"""
from __future__ import annotations

import numpy as np


def threshold_at_fpr(val_scores: np.ndarray, fpr: float) -> float:
    """Smallest score that flags at most ``fpr`` of benign validation packets."""
    v = np.asarray(val_scores, dtype=np.float64)
    u, c = np.unique(v, return_counts=True)
    tail = np.cumsum(c[::-1])[::-1] / len(v)          # tail[i] = P(v >= u[i])
    ok = np.flatnonzero(tail <= fpr)
    if len(ok) == 0:
        return float(np.nextafter(u[-1], np.inf))
    return float(u[ok[0]])  # rule: flag if score >= threshold; ties never exceed the budget


def empirical_pvalue(scores: np.ndarray, val_scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """Smoothed conformal p-value  p(x) = (#{v > s} + U * (#{v = s} + 1)) / (n + 1).

    ``U ~ Uniform(0, 1)`` breaks ties at random, so detectors with discrete scores (many
    benign packets share one score) still yield uniformly distributed benign p-values and a
    well-defined false-positive budget. The draw is seeded for reproducibility.
    """
    v = np.sort(np.asarray(val_scores, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    n = len(v)
    gt = n - np.searchsorted(v, s, side="right")
    eq = np.searchsorted(v, s, side="right") - np.searchsorted(v, s, side="left")
    u = np.random.default_rng(seed).random(len(s))
    return (gt + u * (eq + 1.0)) / (n + 1.0)


def fuse_pvalues(p_views: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni-corrected minimum p-value over the views that observed each packet.

    Returns the fused p-value and the index of the view that attains the minimum (used for
    layer attribution / alert routing). A view that did not observe a packet contributes 1.
    """
    P = np.ones((len(p_views), len(p_views[0])))
    for i, (p, m) in enumerate(zip(p_views, masks)):
        P[i, m] = p[m]
    n_obs = np.maximum(1, np.vstack(masks).sum(axis=0))
    best = P.argmin(axis=0)
    fused = np.minimum(1.0, P.min(axis=0) * n_obs)
    return fused, best
