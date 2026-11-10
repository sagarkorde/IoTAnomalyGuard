"""E3 - Component ablation of AnomalyGuard (RQ3).

Each row removes one component from the full configuration and re-runs the leakage-aware
evaluation for every seed. Components: label-free psi rule, causal context features,
cross-layer fusion, host-fingerprint features (TTL/window), alert propagation.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .. import config as C
from ..pipeline import anomalyguard_scores, load_with_context, sample_split
from .common import evaluate, layer_map, log

CONFIGS = [
    # name, detector params, context, fingerprint, fusion, propagation
    ("Full AnomalyGuard-IF", {"max_samples": "rule"}, True, True, True, True),
    ("- propagation", {"max_samples": "rule"}, True, True, True, False),
    ("- context features", {"max_samples": "rule"}, False, True, True, True),
    ("- cross-layer fusion", {"max_samples": "rule"}, True, True, False, True),
    ("- fingerprint features", {"max_samples": "rule"}, True, False, True, True),
    ("- psi rule (psi=256)", {"max_samples": 256}, True, True, True, True),
    ("IF-default (none of the above)", {"max_samples": 256}, False, True, False, False),
]


def main(seeds=C.SEEDS, detector="IF"):
    df = load_with_context()
    layers = layer_map(df)
    rows = []
    for seed in seeds:
        sp = sample_split(df, seed)
        for name, params, ctx, fp, fusion, prop in CONFIGS:
            out = anomalyguard_scores(sp, detector, seed, context=ctx, fingerprint=fp,
                                      fusion=fusion, det_params=params, propagation=prop)
            extra = out.get("prop_flags") if prop else None
            row, _ = evaluate(sp.test, out["score_test"], out["thr"], layers, extra)
            rows.append({"config": name, "detector": detector, "seed": seed, **row})
            log(f"seed {seed} {name:32s} macroR={row['macro_recall']:.3f} "
                f"macroAUC={row['macro_auc']:.3f} FPR={row['fpr']:.4f}")
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / f"e3_ablation_{detector}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS))
    ap.add_argument("--detector", default="IF")
    a = ap.parse_args()
    main(tuple(a.seeds), a.detector)
