"""E4 - Sensitivity of Isolation Forest to subsample size psi and ensemble size (RQ3).

The label-free psi rule is computed from the benign service mix only; this sweep shows where
it lands relative to the full curve (which *does* use labels and is therefore never used to
choose psi).
"""
from __future__ import annotations

import argparse

import pandas as pd

from .. import config as C
from ..pipeline import anomalyguard_scores, load_with_context, psi_mode_coverage, sample_split, service_of
from .common import evaluate, layer_map, log

PSI_GRID = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
TREE_GRID = (50, 100, 200, 400)


def main(seeds=(0, 1, 2)):
    df = load_with_context()
    layers = layer_map(df)
    rows = []
    for seed in seeds:
        sp = sample_split(df, seed)
        rule = psi_mode_coverage(service_of(sp.train), n_max=len(sp.train))
        for psi in PSI_GRID:
            out = anomalyguard_scores(sp, "IF", seed, det_params={"max_samples": psi})
            row, _ = evaluate(sp.test, out["score_test"], out["thr"], layers)
            rows.append({"sweep": "psi", "psi": psi, "n_estimators": 200, "seed": seed,
                         "psi_rule": rule, "fit_seconds": out["fit_seconds"],
                         "score_us_per_pkt": 1e6 * out["score_seconds_per_pkt"], **row})
            log(f"seed {seed} psi={psi:5d} macroR={row['macro_recall']:.3f} netR={row.get('recall_network', float('nan')):.3f}")
        for t in TREE_GRID:
            out = anomalyguard_scores(sp, "IF", seed, det_params={"max_samples": "rule", "n_estimators": t})
            row, _ = evaluate(sp.test, out["score_test"], out["thr"], layers)
            rows.append({"sweep": "trees", "psi": rule, "n_estimators": t, "seed": seed,
                         "psi_rule": rule, "fit_seconds": out["fit_seconds"],
                         "score_us_per_pkt": 1e6 * out["score_seconds_per_pkt"], **row})
            log(f"seed {seed} trees={t:4d} macroR={row['macro_recall']:.3f}")
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e4_psi_sweep.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    main(tuple(ap.parse_args().seeds))
