"""E13 - Does drift-aware recalibration restore the false-positive budget?

Per-capture protocol of E2 (fusion, no session association). The benign test periods of the
NormalTrafficMQTT and NormalTrafficHTTP captures are cut into ten time-ordered parts of equal
packet count. ``static`` keeps the validation threshold; ``rolling`` recomputes the threshold of
part i from the benign packets of part i-1 of the same capture (a recent window assumed to have
been vetted as benign), at the same 1 % budget. We report the FPR of parts 1-9 under both
policies and, as a cost indicator, the mean macro recall over all test anomalies when the rolling
thresholds are applied instead of the static one.
Output: results/e13_recalibration.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..calibration import threshold_at_fpr
from ..pipeline import anomalyguard_scores, load_with_context, sample_split
from .common import log

DETECTORS = [("AnomalyGuard-IF", "IF", {"max_samples": "rule"}), ("AE", "AE", {}), ("OCSVM", "OCSVM", {})]
STREAMS = ("NormalTrafficMQTT", "NormalTrafficHTTP")


def macro_recall(test, flags):
    y = test["label"].to_numpy() == 1
    names = test["name"].astype(str).to_numpy()
    return float(np.mean([flags[y & (names == n)].mean() for n in np.unique(names[y])]))


def main(seeds=C.SEEDS):
    df = load_with_context()
    rows = []
    for seed in seeds:
        sp = sample_split(df, seed)
        te = sp.test
        cap = te["capture_group"].astype(str).to_numpy()
        ben = te["label"].to_numpy() == 0
        t = te["t"].to_numpy()
        for label, det, params in DETECTORS:
            out = anomalyguard_scores(sp, det, seed, det_params=params, propagation=False)
            s, thr = out["score_test"], out["thr"]
            static_mr = macro_recall(te, s >= thr)
            rolling_thr = []
            for c in STREAMS:
                m = np.flatnonzero(ben & (cap == c))
                m = m[np.argsort(t[m], kind="stable")]
                parts = np.array_split(m, 10)
                for i in range(1, 10):
                    thr_i = threshold_at_fpr(s[parts[i - 1]], C.FPR_BUDGET)
                    rolling_thr.append(thr_i)
                    rows.append({"seed": seed, "detector": label, "stream": c, "part": i,
                                 "fpr_static": float((s[parts[i]] >= thr).mean()),
                                 "fpr_rolling": float((s[parts[i]] >= thr_i).mean()),
                                 "thr_ratio": float(thr_i / thr) if thr else np.nan,
                                 "macro_recall_static": static_mr,
                                 "macro_recall_rolling_thr": macro_recall(te, s >= thr_i)})
            r = pd.DataFrame([x for x in rows if x["seed"] == seed and x["detector"] == label])
            log(f"seed {seed} {label:16s} max part FPR static={r.fpr_static.max():.3f} rolling={r.fpr_rolling.max():.3f} "
                f"macroR static={static_mr:.3f} rolling(mean)={r.macro_recall_rolling_thr.mean():.3f}")
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e13_recalibration.csv", index=False)


if __name__ == "__main__":
    main()
