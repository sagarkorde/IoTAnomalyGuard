"""E7 - Triage value of the context layer (RQ6).

(a) Alert routing: each fused alert is attributed to the view with the smallest p-value
    (network vs telemetry) and routed to a network-security or a device/clinical queue.
    We compare the route with the true anomaly layer.
(b) Base-rate analysis: alert precision implied by the measured recall and FPR at
    realistic anomaly prevalences.
(c) Incident grouping: alerts from the same (capture, source host) separated by less than
    60 s form one incident; we report alerts per incident (analyst workload).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config as C
from ..pipeline import anomalyguard_scores, load_with_context, sample_split
from .common import log

PREVALENCES = (0.001, 0.01, 0.05, 0.10)
GAP_S = 60.0


def incidents(test: pd.DataFrame, flags: np.ndarray) -> pd.DataFrame:
    a = test.loc[flags, ["capture_group", "src", "t", "label", "name"]].copy()
    a["capture_group"] = a["capture_group"].astype(str)
    a["src"] = a["src"].astype(str)
    a = a.sort_values(["capture_group", "src", "t"])
    new = (a["capture_group"] != a["capture_group"].shift()) | (a["src"] != a["src"].shift()) \
        | (a["t"].diff() > GAP_S)
    a["incident"] = np.cumsum(new.to_numpy())
    return a


def main(seeds=C.SEEDS):
    df = load_with_context()
    route_rows, prev_rows, inc_rows = [], [], []
    for seed in seeds:
        sp = sample_split(df, seed)
        out = anomalyguard_scores(sp, "IF", seed, det_params={"max_samples": "rule"}, propagation=True)
        fused = out["score_test"] >= out["thr"]
        flags = fused | out["prop_flags"]
        layer = sp.test["layer"].to_numpy()
        label = sp.test["label"].to_numpy()
        # (a) routing - propagated alerts are network-security alerts by construction
        route = np.where(out["prop_flags"] & ~fused, 0, out["attribution"])  # 0 network, 1 telemetry
        for L in (1, 2, 3):
            m = flags & (layer == L)
            route_rows.append({"seed": seed, "true_layer": C.LAYER_NAMES[L],
                               "alerts": int(m.sum()),
                               "routed_network": int(np.sum(m & (route == 0))),
                               "routed_device_clinical": int(np.sum(m & (route == 1)))})
        m0 = flags & (label == 0)
        route_rows.append({"seed": seed, "true_layer": "Normal (false alerts)", "alerts": int(m0.sum()),
                           "routed_network": int(np.sum(m0 & (route == 0))),
                           "routed_device_clinical": int(np.sum(m0 & (route == 1)))})
        # (b) base rate
        rec = flags[label == 1].mean()
        fpr = flags[label == 0].mean()
        for pi in PREVALENCES:
            prec = rec * pi / (rec * pi + fpr * (1 - pi))
            prev_rows.append({"seed": seed, "prevalence": pi, "recall": rec, "fpr": fpr,
                              "precision": prec, "false_alerts_per_true_alert": (1 - prec) / prec})
        # (c) incidents
        inc = incidents(sp.test, flags)
        g = inc.groupby("incident").agg(n=("t", "size"), mal=("label", "max"))
        inc_rows.append({"seed": seed, "alerts": int(flags.sum()), "incidents": int(len(g)),
                         "true_incidents": int((g["mal"] == 1).sum()),
                         "false_incidents": int((g["mal"] == 0).sum()),
                         "alerts_per_incident": float(g["n"].mean()),
                         "false_alerts": int(m0.sum())})
        log(f"seed {seed}: alerts={flags.sum()} incidents={len(g)} rec={rec:.3f} fpr={fpr:.4f}")
    pd.DataFrame(route_rows).to_csv(C.RESULTS_DIR / "e7_routing.csv", index=False)
    pd.DataFrame(prev_rows).to_csv(C.RESULTS_DIR / "e7_base_rate.csv", index=False)
    pd.DataFrame(inc_rows).to_csv(C.RESULTS_DIR / "e7_incidents.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS))
    main(tuple(ap.parse_args().seeds))
