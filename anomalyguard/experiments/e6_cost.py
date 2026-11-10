"""E6 - Computational cost (RQ5): fit time, single-thread scoring throughput, model size.

Measured on the network+context view (the view every packet passes through).
"""
from __future__ import annotations

import os
import pickle
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from .. import config as C
from ..detectors import make
from ..features import PortNovelty
from ..pipeline import load_with_context, network_matrix, psi_mode_coverage, sample_split, service_of
from .common import environment, log, write_json

N_SCORE = 100_000
REPEATS = 3


def main(seed: int = 0):
    df = load_with_context()
    # reference load: the highest packet rate (packets per wall-clock second) of any capture
    sec = np.floor(df["t"].to_numpy()).astype(np.int64)
    per_sec = pd.DataFrame({"capture_group": df["capture_group"].astype(str).to_numpy(), "sec": sec}) \
        .groupby(["capture_group", "sec"]).size()
    peak = per_sec.groupby(level=0).max().sort_values(ascending=False).rename("peak_pkts_per_s")
    peak.to_csv(C.RESULTS_DIR / "e6_peak_rate.csv")
    log(f"peak packet rate: {int(peak.iloc[0])} pkt/s ({peak.index[0]})")
    sp = sample_split(df, seed)
    ports = PortNovelty().fit(sp.train)
    Xtr = network_matrix(sp.train, ports).to_numpy(np.float64)
    Xte = network_matrix(sp.test.iloc[:N_SCORE], ports).to_numpy(np.float64)
    rule = psi_mode_coverage(service_of(sp.train), n_max=len(Xtr))
    configs = [("IF-default", "IF", {"max_samples": 256}),
               ("AnomalyGuard-IF", "IF", {"max_samples": rule}),
               ("ECOD", "ECOD", {}), ("HBOS", "HBOS", {}), ("LOF", "LOF", {}),
               ("OCSVM", "OCSVM", {}), ("AE", "AE", {})]
    rows = []
    for label, det, params in configs:
        fits, scores = [], []
        for r in range(REPEATS):
            d = make(det, seed + r, **params)
            if det == "IF":
                d.params["n_jobs"] = 1
            t0 = time.perf_counter(); d.fit(Xtr); fits.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); d.score(Xte); scores.append(time.perf_counter() - t0)
        size = len(pickle.dumps(d.model if hasattr(d, "model") else d))
        rows.append({"detector": label, "fit_s_median": float(np.median(fits)),
                     "score_us_per_pkt_median": 1e6 * float(np.median(scores)) / len(Xte),
                     "throughput_pkts_per_s": len(Xte) / float(np.median(scores)),
                     "model_kib": size / 1024, "n_train": len(Xtr), "n_features": Xtr.shape[1]})
        log(f"{label:16s} fit={rows[-1]['fit_s_median']:.2f}s  "
            f"{rows[-1]['throughput_pkts_per_s']:.0f} pkt/s  {rows[-1]['model_kib']:.0f} KiB")
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e6_cost.csv", index=False)
    write_json(environment(), C.RESULTS_DIR / "environment_cost.json")


if __name__ == "__main__":
    main()
