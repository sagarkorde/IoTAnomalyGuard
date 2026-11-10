"""E5 - Robustness to unlabelled anomalies in the training capture (RQ4).

A fraction rho of the benign training pool is replaced by anomalous packets from the training
segments (all anomaly types, proportional to their size). Calibration still uses clean benign
validation traffic, i.e. we assume a small curated reference window is available.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .. import config as C
from ..pipeline import anomalyguard_scores, load_with_context, sample_split
from .common import evaluate, layer_map, log

RHOS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
DETECTORS = [("AnomalyGuard-IF", "IF", {"max_samples": "rule"}),
             ("IF-default", "IF", {"max_samples": 256}),
             ("OCSVM", "OCSVM", {})]


def main(seeds=(0, 1, 2)):
    df = load_with_context()
    layers = layer_map(df)
    rows = []
    for seed in seeds:
        for rho in RHOS:
            sp = sample_split(df, seed, contamination=rho)
            for label, det, params in DETECTORS:
                out = anomalyguard_scores(sp, det, seed, det_params=params, propagation=True)
                row, _ = evaluate(sp.test, out["score_test"], out["thr"], layers, out["prop_flags"])
                rows.append({"detector": label, "rho": rho, "seed": seed, **row})
                log(f"seed {seed} rho={rho:.3f} {label:16s} macroR={row['macro_recall']:.3f} FPR={row['fpr']:.4f}")
            pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e5_contamination.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    main(tuple(ap.parse_args().seeds))
