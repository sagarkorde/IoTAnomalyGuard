"""E11 - End-to-end cost of the complete AnomalyGuard pipeline (single thread).

Stages timed per packet on the full test set (seed 0), after the models are fitted:
  context   causal per-source context for every packet of the dataset (numba; JIT warm-up excluded)
  views     behaviour-only network view + telemetry view of the test packets
  scoring   network-view and telemetry-view detector scores
  fusion    conformal p-values and Bonferroni fusion
  assoc     session association (alert propagation)
The pipeline is run in vectorised batches; a packet-at-a-time implementation would add call
overhead. Packet parsing (pcap -> fields) is outside the scope, as X-IoMT ships parsed fields.
Run with OMP_NUM_THREADS=1 (see Makefile). Output: results/e11_pipeline_cost.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config as C
from ..calibration import empirical_pvalue, fuse_pvalues, threshold_at_fpr
from ..data import load_processed
from ..features import PortNovelty, TelemetryView, context_view
from ..pipeline import (_resolve_params, calibrate_propagation, load_with_context, network_matrix,
                        propagate_alerts, run_view, sample_split, service_of)
from .common import log

DETECTORS = [("AnomalyGuard-IF", "IF", {"max_samples": "rule", "n_jobs": 1}),
             ("AE", "AE", {}), ("OCSVM", "OCSVM", {}), ("DIF", "DIF", {})]
REPEATS = 3


def repeats(label):
    return 1 if label == "DIF" else REPEATS


def main(seed: int = 0):
    raw = load_processed()
    context_view(raw.iloc[:5000])                       # numba JIT warm-up
    t_ctx = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); context_view(raw); t_ctx.append(time.perf_counter() - t0)
    ctx_us = 1e6 * float(np.median(t_ctx)) / len(raw)
    df = load_with_context()
    sp = sample_split(df, seed)
    rows = []
    for label, det, params in DETECTORS:
        ports = PortNovelty().fit(sp.train); tel = TelemetryView().fit(sp.train)
        Xtr, Xva = network_matrix(sp.train, ports), network_matrix(sp.val, ports)
        mtr, mva = TelemetryView.mask(sp.train), TelemetryView.mask(sp.val)
        prm_n = _resolve_params(det, params, service_of(sp.train), len(Xtr))
        prm_t = _resolve_params(det, params, service_of(sp.train[mtr]), int(mtr.sum()))
        net = run_view(det, seed, Xtr, Xva, Xva.iloc[:10], **prm_n)
        tv = run_view(det, seed, tel.transform(sp.train[mtr]), tel.transform(sp.val[mva]),
                      tel.transform(sp.val[mva].iloc[:10]), **prm_t)
        Xvs = network_matrix(sp.val_stream, ports)
        f_vs = empirical_pvalue(net.model.score(Xvs), net.val, seed + 4) <= C.FPR_BUDGET
        k = calibrate_propagation(sp.val_stream, f_vs)
        st = {"views": [], "scoring": [], "fusion": [], "assoc": []}
        te = sp.test
        for _ in range(repeats(label)):
            t0 = time.perf_counter()
            Xte = network_matrix(te, ports); mte = TelemetryView.mask(te); Tte = tel.transform(te[mte])
            t1 = time.perf_counter()
            s_net = net.model.score(Xte); s_tel = tv.model.score(Tte)
            t2 = time.perf_counter()
            p_net = empirical_pvalue(s_net, net.val, seed)
            p_tel = np.ones(len(te)); p_tel[mte] = empirical_pvalue(s_tel, tv.val, seed + 2)
            fused, _ = fuse_pvalues([p_net, p_tel], [np.ones(len(te), bool), mte])
            t3 = time.perf_counter()
            prop = propagate_alerts(te, p_net <= C.FPR_BUDGET, 10.0, k)
            t4 = time.perf_counter()
            for key, a, b in (("views", t0, t1), ("scoring", t1, t2), ("fusion", t2, t3), ("assoc", t3, t4)):
                st[key].append(b - a)
        n = len(te)
        row = {"detector": label, "n_test": n, "fit_seconds": net.fit_seconds + tv.fit_seconds,
               "context_us_per_pkt": ctx_us}
        for key, v in st.items():
            row[f"{key}_us_per_pkt"] = 1e6 * float(np.median(v)) / n
        row["total_us_per_pkt"] = sum(row[f"{k}_us_per_pkt"] for k in ("context", "views", "scoring", "fusion", "assoc"))
        row["throughput_pkts_per_s"] = 1e6 / row["total_us_per_pkt"]
        rows.append(row)
        log(f"{label}: " + " ".join(f"{k}={v:.2f}" for k, v in row.items() if k.endswith('_us_per_pkt'))
            + f" -> {row['throughput_pkts_per_s']:.0f} pkt/s")
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e11_pipeline_cost.csv", index=False)


if __name__ == "__main__":
    main()
