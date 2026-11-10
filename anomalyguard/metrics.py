"""Evaluation metrics (attacks/anomalies are the positive class)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def detection_report(y: np.ndarray, names: np.ndarray, scores: np.ndarray, thr: float,
                     benign_name: str = "Normal") -> dict:
    """Threshold-free and thresholded metrics plus per-anomaly-type recall."""
    y = np.asarray(y).astype(int)
    flag = scores >= thr
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    rep = {
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.mean() < 1 else np.nan,
        "pr_auc": float(average_precision_score(y, scores)) if y.sum() else np.nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
    }
    rep["f1"] = (2 * tp / (2 * tp + fp + fn)) if tp else 0.0
    per = {}
    for n in np.unique(names):
        if n == benign_name:
            continue
        m = names == n
        per[str(n)] = float(flag[m].mean())
    rep["per_type_recall"] = per
    rep["macro_recall"] = float(np.mean(list(per.values()))) if per else np.nan
    return rep


def per_type_auc(y: np.ndarray, names: np.ndarray, scores: np.ndarray,
                 benign_name: str = "Normal") -> dict:
    """ROC-AUC of each anomaly type against all benign test packets."""
    ben = names == benign_name
    out = {}
    for n in np.unique(names):
        if n == benign_name:
            continue
        m = (names == n) | ben
        out[str(n)] = float(roc_auc_score((names[m] == n).astype(int), scores[m]))
    return out


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(-scores, kind="mergesort")[:k]
    return float(np.mean(np.asarray(y)[idx]))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarise(rows: list[dict], keys: list[str], metrics: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    g = df.groupby(keys)[metrics]
    return g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index()
