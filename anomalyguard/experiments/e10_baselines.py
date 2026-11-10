"""E10 - Further baselines requested in review, under exactly the E2 protocol.

* IF-strat: one Isolation Forest (psi = 256) per service context (MQTT / HTTP / other, from the
  server port) with Mondrian calibration, i.e. each service is calibrated on its own benign
  validation packets. This is the obvious alternative to the label-free psi rule.
* DIF: Deep Isolation Forest (Xu et al., 2023), NumPy re-implementation of the authors'
  reference code (see detectors.DIFDet).

Outputs: results/e10_baselines.csv, results/e10_per_type.csv (same columns as E2).
"""
from __future__ import annotations

import argparse

import pandas as pd

from .. import config as C
from ..pipeline import anomalyguard_scores, load_with_context, sample_split
from .common import evaluate, layer_map, log

DETECTORS = [
    ("IF-strat", "IF", {"max_samples": 256, "stratify": True}),
    ("DIF", "DIF", {}),
]


def main(seeds=C.SEEDS, detectors=None, suffix=""):
    df = load_with_context()
    layers = layer_map(df)
    rows, per_rows = [], []
    for seed in seeds:
        sp = sample_split(df, seed)
        for label, det, params in DETECTORS:
            if detectors and label not in detectors:
                continue
            out = anomalyguard_scores(sp, det, seed, det_params=params, propagation=True)
            for vname, extra in (("fused", None), ("fused+propagation", out["prop_flags"])):
                row, per = evaluate(sp.test, out["score_test"], out["thr"], layers, extra)
                meta = {"detector": label, "variant": vname, "seed": seed,
                        "fit_seconds": out["fit_seconds"],
                        "score_us_per_pkt": 1e6 * out["score_seconds_per_pkt"],
                        "prop_k": out.get("prop_k")}
                rows.append({**meta, **row}); per_rows.append({**meta, **per})
                log(f"seed {seed} {label:9s} {vname:18s} macroR={row['macro_recall']:.3f} "
                    f"net={row.get('recall_network'):.3f} macroAUC={row['macro_auc']:.3f} FPR={row['fpr']:.4f} "
                    f"fit={out['fit_seconds']:.1f}s")
            pd.DataFrame(rows).to_csv(C.RESULTS_DIR / f"e10_baselines{suffix}.csv", index=False)
            pd.DataFrame(per_rows).to_csv(C.RESULTS_DIR / f"e10_per_type{suffix}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS))
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--suffix", default="", help="output-file suffix (to run seed groups in parallel)")
    a = ap.parse_args()
    main(tuple(a.seeds), a.detectors, a.suffix)
