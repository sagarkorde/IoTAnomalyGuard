"""E2 - Main benchmark under the leakage-aware protocol (RQ2).

Every unsupervised detector is plugged into the same AnomalyGuard pipeline (behaviour-only
network+context view, telemetry view, benign-only calibration at a 1 % FPR budget,
Bonferroni p-value fusion). The supervised HGB model is a labelled upper bound.

Outputs: results/e2_benchmark.csv (one row per detector x seed), results/e2_per_type.csv,
results/e2_scores_seed0.parquet (test scores of seed 0 for figures).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config as C
from ..calibration import threshold_at_fpr
from ..detectors import make
from ..features import PortNovelty, TelemetryView
from ..pipeline import anomalyguard_scores, load_with_context, network_matrix, sample_split
from .common import environment, evaluate, layer_map, log, write_json

DETECTORS = [
    ("IF-default", "IF", {"max_samples": 256}),
    ("AnomalyGuard-IF", "IF", {"max_samples": "rule"}),
    ("ECOD", "ECOD", {}),
    ("HBOS", "HBOS", {}),
    ("LOF", "LOF", {}),
    ("OCSVM", "OCSVM", {}),
    ("AE", "AE", {}),
]


def supervised_reference(sp, seed):
    ports = PortNovelty().fit(sp.train)
    tel = TelemetryView().fit(sp.train)

    def mat(f):
        return pd.concat([network_matrix(f, ports), tel.transform(f)], axis=1)

    det = make("HGB", seed).fit(mat(sp.train_all), sp.train_all["label"].to_numpy())
    s_va = det.score(mat(sp.val))
    s_te = det.score(mat(sp.test))
    return s_te, threshold_at_fpr(s_va, C.FPR_BUDGET), det.fit_seconds


def main(seeds=C.SEEDS, detectors=None):
    df = load_with_context()
    layers = layer_map(df)
    rows, per_rows = [], []
    keep = {}
    for seed in seeds:
        sp = sample_split(df, seed)
        log(f"seed {seed}: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}")
        for label, det, params in DETECTORS:
            if detectors and label not in detectors:
                continue
            prop = True
            out = anomalyguard_scores(sp, det, seed, det_params=params, propagation=prop)
            variants = [("fused", None)]
            if prop:
                variants.append(("fused+propagation", out["prop_flags"]))
            for vname, extra in variants:
                row, per = evaluate(sp.test, out["score_test"], out["thr"], layers, extra)
                meta = {"detector": label, "variant": vname, "seed": seed,
                        "fit_seconds": out["fit_seconds"],
                        "score_us_per_pkt": 1e6 * out["score_seconds_per_pkt"],
                        "psi_net": out["params_net"].get("max_samples") if out["params_net"] else None,
                        "psi_tel": out["params_tel"].get("max_samples") if out["params_tel"] else None,
                        "prop_k": out.get("prop_k"), "prop_val_extra_fpr": out.get("prop_val_extra_fpr")}
                rows.append({**meta, **row})
                per_rows.append({**meta, **per})
                log(f"  {label:16s} {vname:18s} macroR={row['macro_recall']:.3f} "
                    f"macroAUC={row['macro_auc']:.3f} FPR={row['fpr']:.4f} fit={out['fit_seconds']:.1f}s")
            if seed == seeds[0] and label in ("AnomalyGuard-IF", "IF-default"):
                keep[f"{label}::fused"] = out["score_test"]
                keep[f"{label}::net"] = out["net_test"]
                keep[f"{label}::thr"] = np.full(len(sp.test), out["thr"])
                keep[f"{label}::attr"] = out["attribution"]
        if not detectors or "HGB" in detectors:
            s_te, thr, fit_s = supervised_reference(sp, seed)
            row, per = evaluate(sp.test, s_te, thr, layers)
            meta = {"detector": "HGB (supervised)", "variant": "single", "seed": seed,
                    "fit_seconds": fit_s, "score_us_per_pkt": np.nan,
                    "psi_net": None, "psi_tel": None, "prop_k": None, "prop_val_extra_fpr": None}
            rows.append({**meta, **row}); per_rows.append({**meta, **per})
            log(f"  HGB (supervised) macroR={row['macro_recall']:.3f} macroAUC={row['macro_auc']:.3f} FPR={row['fpr']:.4f}")
            if seed == seeds[0]:
                keep["HGB::fused"] = s_te
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e2_benchmark.csv", index=False)
        pd.DataFrame(per_rows).to_csv(C.RESULTS_DIR / "e2_per_type.csv", index=False)
    test0 = sample_split(df, seeds[0]).test
    keepdf = pd.DataFrame(keep)
    keepdf["label"] = test0["label"].to_numpy()
    keepdf["name"] = test0["name"].astype(str).to_numpy()
    keepdf["layer"] = test0["layer"].to_numpy()
    keepdf.to_parquet(C.RESULTS_DIR / "e2_scores_seed0.parquet", index=False)
    write_json(environment(), C.RESULTS_DIR / "environment.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS))
    ap.add_argument("--detectors", nargs="*", default=None)
    a = ap.parse_args()
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    main(tuple(a.seeds), a.detectors)
