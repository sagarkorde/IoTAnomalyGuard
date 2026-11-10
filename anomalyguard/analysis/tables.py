"""Build the paper's tables (as CSV with formatted strings) from results/*.csv."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C

R = C.RESULTS_DIR
OUT = R / "tables"
ORDER = ["IF-default", "AnomalyGuard-IF", "ECOD", "HBOS", "LOF", "OCSVM", "AE", "HGB (supervised)"]


def ms(s: pd.Series, d: int = 3) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "-"
    return f"{s.mean():.{d}f} ± {s.std(ddof=1):.{d}f}" if len(s) > 1 else f"{s.mean():.{d}f}"


def t_dataset():
    rep = json.load(open(C.PROCESSED_DIR / "xiomt_report.json"))
    df = pd.read_parquet(C.PROCESSED_DIR / "xiomt.parquet", columns=["name", "layer", "split", "capture_group"])
    g = df.groupby("name", observed=True)
    t = pd.DataFrame({
        "Layer": g["layer"].agg(lambda s: C.LAYER_NAMES[int(s.mode().iloc[0])]),
        "Captures": g["capture_group"].nunique(),
        "Packets": g.size(),
        "Test packets": g["split"].agg(lambda s: int((s == 2).sum())),
    }).reset_index().rename(columns={"name": "Class"})
    t["_o"] = t["Layer"].map({"Normal": 0, "Network": 1, "Perception": 2, "Application": 3})
    t = t.sort_values(["_o", "Class"]).drop(columns="_o")
    t.to_csv(OUT / "T1_dataset.csv", index=False)
    return rep


def t_audit():
    p = pd.read_csv(R / "e1_probes.csv")
    q = pd.read_csv(R / "e1_protocols.csv")
    rows = []
    names = {"probe:absence_spelling": "Probe: absence spelling + telemetry presence (no behaviour)",
             "probe:host_identity": "Probe: host identity (IP, MAC, client ID, TTL, window)",
             "naive:HGB:random-split": "Supervised HGB, all columns, random split",
             "naive:HGB:chronological": "Supervised HGB, all columns, chronological split",
             "leakage-aware:HGB:chronological": "Supervised HGB, behaviour-only views, chronological split",
             "naive:IF-default:random-split": "IF (psi=256), all columns, random split",
             "naive:IF-default:chronological": "IF (psi=256), all columns, chronological split"}
    for k, label in names.items():
        sub = pd.concat([p, q])
        sub = sub[sub["setting"] == k]
        if len(sub) == 0:
            continue
        rows.append({"Setting": label, "AUROC": ms(sub["roc_auc"], 4), "Accuracy": ms(sub["accuracy"], 4),
                     "F1": ms(sub["f1"], 4), "Macro recall": ms(sub["macro_recall"], 3)})
    pd.DataFrame(rows).to_csv(OUT / "T2_audit.csv", index=False)


def t_benchmark():
    b = pd.read_csv(R / "e2_benchmark.csv")
    rows = []
    for det in ORDER:
        for var in ("fused", "fused+propagation", "single"):
            sub = b[(b["detector"] == det) & (b["variant"] == var)]
            if len(sub) == 0:
                continue
            rows.append({
                "Detector": det, "Variant": {"fused": "fusion", "fused+propagation": "fusion + propagation",
                                             "single": "single model"}[var],
                "Macro recall": ms(sub["macro_recall"]), "Macro AUROC": ms(sub["macro_auc"]),
                "PR-AUC": ms(sub["pr_auc"]), "Test FPR (%)": ms(100 * sub["fpr"], 2),
                "Network": ms(sub["recall_network"], 2), "Perception": ms(sub["recall_perception"], 2),
                "Application": ms(sub["recall_application"], 2),
            })
    pd.DataFrame(rows).to_csv(OUT / "T3_benchmark.csv", index=False)


def t_per_type():
    from .stats import per_type_matrix
    m = per_type_matrix()
    layer = pd.read_parquet(C.PROCESSED_DIR / "xiomt.parquet", columns=["name", "layer"]) \
        .groupby("name", observed=True)["layer"].agg(lambda s: C.LAYER_NAMES[int(s.mode().iloc[0])])
    m.insert(0, "Layer", [layer[n] for n in m.index])
    m["_o"] = m["Layer"].map({"Network": 1, "Perception": 2, "Application": 3})
    m = m.sort_values(["_o"]).drop(columns="_o")
    cols = ["Layer"] + [c for c in ORDER if c in m.columns]
    m = m[cols]
    m.index.name = "Anomaly type"
    m.round(2).to_csv(OUT / "T4_per_type_recall.csv")


def t_ablation():
    a = pd.read_csv(R / "e3_ablation_IF.csv")
    rows = []
    for cfg, sub in a.groupby("config", sort=False):
        rows.append({"Configuration": cfg, "Macro recall": ms(sub["macro_recall"]),
                     "Macro AUROC": ms(sub["macro_auc"]), "Test FPR (%)": ms(100 * sub["fpr"], 2),
                     "Network": ms(sub["recall_network"], 2), "Perception": ms(sub["recall_perception"], 2),
                     "Application": ms(sub["recall_application"], 2)})
    pd.DataFrame(rows).to_csv(OUT / "T5_ablation.csv", index=False)


def t_cost_selection():
    if (R / "e6_cost.csv").exists():
        c = pd.read_csv(R / "e6_cost.csv")
        c = c.assign(**{"Fit (s)": c["fit_s_median"].round(2),
                        "Throughput (pkt/s)": c["throughput_pkts_per_s"].round(0).astype(int),
                        "Latency (us/pkt)": c["score_us_per_pkt_median"].round(2),
                        "Model (KiB)": c["model_kib"].round(0).astype(int)})
        c[["detector", "Fit (s)", "Throughput (pkt/s)", "Latency (us/pkt)", "Model (KiB)"]] \
            .rename(columns={"detector": "Detector"}).to_csv(OUT / "T6_cost.csv", index=False)
    if (R / "e8_view_selection.csv").exists():
        s = pd.read_csv(R / "e8_view_selection.csv")
        rows = []
        for var, sub in s.groupby("variant"):
            rows.append({"Variant": var, "Selected (net/tel)": ", ".join(
                sorted(set(f"{a}/{b}" for a, b in zip(sub["net"], sub["tel"])))),
                "Macro recall": ms(sub["macro_recall"]), "Macro AUROC": ms(sub["macro_auc"]),
                "Test FPR (%)": ms(100 * sub["fpr"], 2), "Network": ms(sub["recall_network"], 2),
                "Perception": ms(sub["recall_perception"], 2), "Application": ms(sub["recall_application"], 2)})
        pd.DataFrame(rows).to_csv(OUT / "T8_view_selection.csv", index=False)


def t_triage():
    if not (R / "e7_routing.csv").exists():
        return
    r = pd.read_csv(R / "e7_routing.csv").groupby("true_layer")[["alerts", "routed_network", "routed_device_clinical"]].sum()
    r["Network queue (%)"] = (100 * r["routed_network"] / r["alerts"]).round(1)
    r["Device/clinical queue (%)"] = (100 * r["routed_device_clinical"] / r["alerts"]).round(1)
    r.to_csv(OUT / "T7_routing.csv")
    b = pd.read_csv(R / "e7_base_rate.csv").groupby("prevalence")[["precision", "false_alerts_per_true_alert"]].agg(["mean", "std"])
    b.to_csv(OUT / "T7_base_rate.csv")
    i = pd.read_csv(R / "e7_incidents.csv")
    i.describe().to_csv(OUT / "T7_incidents.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (t_dataset, t_audit, t_benchmark, t_per_type, t_ablation, t_cost_selection, t_triage):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)


if __name__ == "__main__":
    main()
