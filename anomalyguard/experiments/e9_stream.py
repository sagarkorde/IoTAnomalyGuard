"""E9 - Deployment-style evaluation without capture identifiers.

In the main protocol every X-IoMT capture is its own stream: causal context and session
association are keyed by (capture, host), so an attack never shares a timeline with benign
traffic of the same hosts. A deployed sensor sees one stream. This experiment builds such a
stream and re-runs the top detectors under two protocols with identical code:

* ``per-capture``  - the main protocol (context and association keyed by capture).
* ``stream``       - the benign MQTT and HTTP captures and all six network-attack captures form
                     ONE stream with no capture key. Each attack capture is time-shifted, with
                     its internal timing intact, to a random start inside the benign MQTT test
                     day (one attack per equal slot, seeded), so attack packets interleave with
                     benign traffic of the same broker (10.42.0.1). Context features and session
                     association are recomputed on this stream. In addition the
                     ``ApplicationAnomaliesClinical`` capture is held out entirely: none of its
                     benign packets is used for training or calibration, and all of them are
                     test packets (an unseen benign capture).

Application- and perception-anomaly captures keep their own streams in both protocols: their
timestamps are synthetic (five captures share the same 24-h clock) or overlap the benign capture
of the same devices, so merging them would invent concurrent copies of a device.

Recall is measured on every packet of each attack capture (not only its last 20 %), in both
protocols. Extra metrics: benign FPR inside the attack windows (attack span +/- 10 s) and outside them,
FPR on the held-out benign capture, FPR per tenth of the benign test period (drift), and the
realised benign FPR of the fused conformal p-values at several nominal levels (coverage).

Outputs: results/e9_stream.csv, results/e9_stream_per_type.csv, results/e9_offsets.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config as C
from ..data import load_processed
from ..features import context_view
from ..pipeline import anomalyguard_scores, sample_split
from .common import evaluate, layer_map, log, write_json

DETECTORS = [
    ("AnomalyGuard-IF", "IF", {"max_samples": "rule"}),
    ("AE", "AE", {}),
    ("OCSVM", "OCSVM", {}),
    ("DIF", "DIF", {}),
]
DEFAULT_DETECTORS = ("AnomalyGuard-IF", "AE", "OCSVM")
ATTACKS = ["NetworkAnomaliesMQTTDoS", "NetworkAnomaliesMQTTBruteForce", "NetworkAnomaliesSYNFlood",
           "NetworkAnomaliesFINFlood", "NetworkAnomaliesReconAttack", "NetworkAnomaliesSlowloris"]
BENIGN_STREAM = ["NormalTrafficMQTT", "NormalTrafficHTTP"]
HELD_OUT = "ApplicationAnomaliesClinical"
HOST_PAD_S = 10.0
ALPHAS = (0.001, 0.005, 0.01, 0.02, 0.05)


def build(protocol: str, seed: int):
    """Return (frame with context, attack windows, offsets)."""
    df = load_processed()
    windows, offsets = [], {}
    # In both protocols every packet of a network-attack capture is a test packet (attack rows
    # are never used for unsupervised training), so that recall covers attack onsets and every
    # attack packet can drive session association.
    split = df["split"].to_numpy().copy()
    split[np.isin(df["capture_group"].astype(str).to_numpy(), ATTACKS)] = 2
    df["split"] = split.astype(df["split"].dtype)
    if protocol == "stream":
        cg = df["capture_group"].astype(str).to_numpy().copy()
        t = df["t"].to_numpy().copy()
        test_day = df[(df["capture_group"] == "NormalTrafficMQTT") & (df["split"] == 2)]["t"]
        lo, hi = float(test_day.min()), float(test_day.max())
        rng = np.random.default_rng(10_000 + seed)
        order = rng.permutation(len(ATTACKS))
        slot = (hi - lo) / len(ATTACKS)
        for j, k in enumerate(order):
            name = ATTACKS[k]
            m = cg == name
            a0, a1 = t[m].min(), t[m].max()
            dur = a1 - a0
            start = lo + j * slot + rng.uniform(HOST_PAD_S, max(HOST_PAD_S, slot - dur - HOST_PAD_S))
            t[m] = t[m] - a0 + start
            windows.append((start - HOST_PAD_S, start + dur + HOST_PAD_S))
            offsets[name] = {"start": float(start), "duration_s": float(dur)}
        merged = np.isin(cg, ATTACKS + BENIGN_STREAM)
        cg[merged] = "STREAM"
        df["t"] = t
        df["capture_group"] = pd.Categorical(cg)
        split = df["split"].to_numpy().copy()
        split[df["capture_group"].astype(str).to_numpy() == HELD_OUT] = 2
        df["split"] = split.astype(df["split"].dtype)
    ctx = context_view(df)
    return pd.concat([df, ctx.set_index(df.index)], axis=1), windows, offsets


def extra_metrics(test: pd.DataFrame, flags: np.ndarray, flags_prop: np.ndarray, p_fused: np.ndarray,
                  windows, protocol: str) -> dict:
    y = test["label"].to_numpy()
    ben = y == 0
    t = test["t"].to_numpy()
    cap = test["capture_group"].astype(str).to_numpy()
    out = {}
    inwin = np.zeros(len(test), bool)
    for a, b in windows:
        inwin |= (t >= a) & (t <= b)
    for tag, f in (("", flags), ("_prop", flags_prop)):
        out[f"fpr{tag}"] = float(f[ben].mean())
        if protocol == "stream":
            out[f"fpr_in_attack_windows{tag}"] = float(f[ben & inwin].mean()) if (ben & inwin).any() else np.nan
            out[f"fpr_outside_windows{tag}"] = float(f[ben & ~inwin].mean())
        ho = ben & (cap == HELD_OUT)
        out[f"fpr_clinical_capture{tag}"] = float(f[ho].mean()) if ho.any() else np.nan
    out["n_benign_in_windows"] = int((ben & inwin).sum())
    out["n_benign_clinical_capture"] = int((ben & (cap == HELD_OUT)).sum())
    # drift: FPR per tenth of the benign MQTT and HTTP test periods
    names = test["name"].astype(str).to_numpy()
    for capname in BENIGN_STREAM:
        orig = test["orig_capture"].to_numpy() == capname
        m = ben & orig & (names == "Normal")
        if m.sum() < 100:
            continue
        tt = t[m]
        edges = np.quantile(tt, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(edges, tt, side="right") - 1, 0, 9)
        fd = np.array([flags_prop[m][dec == i].mean() for i in range(10)])
        key = "mqtt" if "MQTT" in capname else "http"
        out[f"drift_{key}_min"] = float(fd.min())
        out[f"drift_{key}_max"] = float(fd.max())
        out[f"drift_{key}_first"] = float(fd[0])
        out[f"drift_{key}_last"] = float(fd[-1])
    for a in ALPHAS:
        out[f"coverage_{a}"] = float((p_fused[ben] <= a).mean())
    return out


def main(protocols=("per-capture", "stream"), seeds=C.SEEDS, detectors=DEFAULT_DETECTORS, suffix=""):
    rows, per_rows, offs = [], [], {}
    for protocol in protocols:
        for seed in seeds:
            df, windows, offsets = build(protocol, seed)
            raw = load_processed()[["capture_group"]]
            df["orig_capture"] = raw["capture_group"].astype(str).to_numpy()
            layers = layer_map(df)
            offs[f"{protocol}:{seed}"] = offsets
            sp = sample_split(df, seed)
            log(f"{protocol} seed {seed}: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}")
            for label, det, params in DETECTORS:
                if label not in detectors:
                    continue
                out = anomalyguard_scores(sp, det, seed, det_params=params, propagation=True)
                flags = out["score_test"] >= out["thr"]
                flags_prop = flags | out["prop_flags"]
                p_fused = np.exp(-out["score_test"])
                ex = extra_metrics(sp.test, flags, flags_prop, p_fused, windows, protocol)
                for vname, extra in (("fused", None), ("fused+propagation", out["prop_flags"])):
                    row, per = evaluate(sp.test, out["score_test"], out["thr"], layers, extra)
                    meta = {"protocol": protocol, "detector": label, "variant": vname, "seed": seed,
                            "prop_k": out.get("prop_k"), "psi_net": out["params_net"].get("max_samples")}
                    rows.append({**meta, **row, **ex})
                    per_rows.append({**meta, **per})
                    log(f"  {label:16s} {vname:18s} macroR={row['macro_recall']:.3f} net={row.get('recall_network', np.nan):.3f} "
                        f"FPR={row['fpr']:.4f} in-win={ex.get('fpr_in_attack_windows_prop', np.nan):.4f} "
                        f"clin={ex['fpr_clinical_capture_prop']:.4f} k={out.get('prop_k')}")
            pd.DataFrame(rows).to_csv(C.RESULTS_DIR / f"e9_stream{suffix}.csv", index=False)
            pd.DataFrame(per_rows).to_csv(C.RESULTS_DIR / f"e9_stream_per_type{suffix}.csv", index=False)
            write_json(offs, C.RESULTS_DIR / f"e9_offsets{suffix}.json")
            del df, sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="*", default=["per-capture", "stream"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS))
    ap.add_argument("--detectors", nargs="*", default=list(DEFAULT_DETECTORS))
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    main(tuple(a.protocols), tuple(a.seeds), tuple(a.detectors), a.suffix)
