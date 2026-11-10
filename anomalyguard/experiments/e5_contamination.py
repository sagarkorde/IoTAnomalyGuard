"""E5 - Robustness to unlabelled anomalies in the training capture (RQ4).

Mode ``train`` (default): a fraction rho of the benign training pool is replaced by anomalous
packets from the training segments (all anomaly types, proportional to their size); calibration
still uses clean benign validation traffic, i.e. we assume a small curated reference window.
Mode ``calib``: the training pool is clean but a fraction rho of the calibration sample is
replaced by anomalies from the validation segments.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .. import config as C
from ..pipeline import anomalyguard_scores, load_with_context, sample_split
from .common import evaluate, layer_map, log

RHOS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
CALIB_RHOS = (0.0, 0.001, 0.005, 0.01, 0.02)
DETECTORS = [("AnomalyGuard-IF", "IF", {"max_samples": "rule"}),
             ("IF-default", "IF", {"max_samples": 256}),
             ("OCSVM", "OCSVM", {}),
             ("AE", "AE", {})]


def main(seeds=(0, 1, 2), detectors=None, mode="train", out_name="e5_contamination.csv"):
    df = load_with_context()
    layers = layer_map(df)
    rows = []
    for seed in seeds:
        for rho in (RHOS if mode == "train" else CALIB_RHOS):
            sp = (sample_split(df, seed, contamination=rho) if mode == "train"
                  else sample_split(df, seed, val_contamination=rho))
            for label, det, params in DETECTORS:
                if detectors and label not in detectors:
                    continue
                out = anomalyguard_scores(sp, det, seed, det_params=params, propagation=True)
                row, _ = evaluate(sp.test, out["score_test"], out["thr"], layers, out["prop_flags"])
                rows.append({"detector": label, "mode": mode, "rho": rho, "seed": seed, **row})
                log(f"seed {seed} rho={rho:.3f} {label:16s} macroR={row['macro_recall']:.3f} FPR={row['fpr']:.4f}")
            pd.DataFrame(rows).to_csv(C.RESULTS_DIR / out_name, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--detectors", nargs="*", default=["AnomalyGuard-IF", "IF-default", "OCSVM"])
    ap.add_argument("--mode", choices=["train", "calib"], default="train")
    ap.add_argument("--out", default="e5_contamination.csv")
    a = ap.parse_args()
    main(tuple(a.seeds), a.detectors, a.mode, a.out)
