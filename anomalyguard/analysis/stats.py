"""Statistical comparison of detectors (Friedman test + Holm-corrected Wilcoxon tests).

Blocks are anomaly types (per-type recall at the 1 % FPR budget, averaged over seeds), so the
tests ask whether a detector is consistently better *across attack/anomaly types*.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from .. import config as C

REFERENCE = "AnomalyGuard-IF"


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def per_type_matrix(variant: str = "fused+propagation", metric: str = "recall") -> pd.DataFrame:
    per = pd.read_csv(C.RESULTS_DIR / "e2_per_type.csv")
    per = per[(per["variant"] == variant) | (per["detector"] == "HGB (supervised)")]
    cols = [c for c in per.columns if c.startswith(f"{metric}::")]
    m = per.groupby("detector")[cols].mean().T
    m.index = [c.split("::", 1)[1] for c in m.index]
    return m


def main():
    out = {}
    for metric in ("recall", "auc"):
        m = per_type_matrix(metric=metric)
        unsup = m.drop(columns=["HGB (supervised)"], errors="ignore")
        stat, p = friedmanchisquare(*[unsup[c].to_numpy() for c in unsup.columns])
        ranks = pd.DataFrame(np.vstack([rankdata(-row) for row in unsup.to_numpy()]),
                             columns=unsup.columns, index=unsup.index).mean().sort_values()
        raw = {}
        diffs = {}
        for c in unsup.columns:
            if c == REFERENCE:
                continue
            d = unsup[REFERENCE] - unsup[c]
            diffs[c] = float(d.median())
            raw[c] = float(wilcoxon(unsup[REFERENCE], unsup[c], zero_method="zsplit").pvalue) \
                if np.any(d != 0) else 1.0
        out[metric] = {
            "n_types": int(len(unsup)), "friedman_chi2": float(stat), "friedman_p": float(p),
            "average_rank": ranks.round(3).to_dict(),
            "wilcoxon_vs_reference_raw_p": raw,
            "wilcoxon_vs_reference_holm_p": holm(raw),
            "median_paired_difference_reference_minus_other": diffs,
        }
    bench = pd.read_csv(C.RESULTS_DIR / "e2_benchmark.csv")
    g = bench.groupby(["detector", "variant"])["macro_recall"]
    out["macro_recall_seed_summary"] = {
        f"{k[0]} | {k[1]}": {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                             "ci95_t": float(2.776 * v.std(ddof=1) / np.sqrt(len(v))), "n": int(len(v))}
        for k, v in g}
    with open(C.RESULTS_DIR / "stats.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("friedman_p", "average_rank", "wilcoxon_vs_reference_holm_p")}
                      for k, v in out.items() if k in ("recall", "auc")}, indent=1))


if __name__ == "__main__":
    main()
