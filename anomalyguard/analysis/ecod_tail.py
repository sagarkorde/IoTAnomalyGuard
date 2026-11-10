"""Diagnostic for RQ2: which benign packets set ECOD's 1 % threshold on the network view?

Fits ECOD exactly as in E2 (network view with context, benign training pool of seed 0), scores the
benign validation packets and describes the top 1 % that define the threshold: their service mix and
the prevalence of the HTTP-response and data-payload indicators compared with the training pool.
Writes results/ecod_tail.json.
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..detectors import make
from ..experiments.common import write_json
from ..features import PortNovelty
from ..pipeline import load_with_context, network_matrix, sample_split, service_of

INDICATORS = ("http_has_response", "proto_has_data")


def main(seed: int = 0):
    df = load_with_context()
    sp = sample_split(df, seed)
    ports = PortNovelty().fit(sp.train)
    Xtr = network_matrix(sp.train, ports)
    Xva = network_matrix(sp.val, ports)
    det = make("ECOD", seed)
    det.fit(Xtr.to_numpy(np.float64))
    s_va = det.score(Xva.to_numpy(np.float64))
    thr = float(np.quantile(s_va, 1 - C.FPR_BUDGET))
    top = s_va >= thr
    svc_tr, svc_va = service_of(sp.train), service_of(sp.val)
    te = sp.test
    s_te = det.score(network_matrix(te, ports).to_numpy(np.float64))
    anom = te["label"].to_numpy() == 1
    out = {
        "seed": seed,
        "threshold_q99_val": thr,
        "n_top": int(top.sum()),
        "service_share_train": {k: float((svc_tr == k).mean()) for k in ("mqtt", "http", "other")},
        "service_share_top_val": {k: float((svc_va[top] == k).mean()) for k in ("mqtt", "http", "other")},
        "indicator_prevalence_train": {c: float((Xtr[c].to_numpy() > 0).mean()) for c in INDICATORS},
        "indicator_prevalence_top_val": {c: float((Xva[c].to_numpy()[top] > 0).mean()) for c in INDICATORS},
        "share_top_val_with_any_indicator": float(
            (np.column_stack([Xva[c].to_numpy()[top] > 0 for c in INDICATORS]).any(axis=1)).mean()),
        "share_test_anomalies_above_threshold": float((s_te[anom] >= thr).mean()),
    }
    write_json(out, C.RESULTS_DIR / "ecod_tail.json")
    print(out)


if __name__ == "__main__":
    main()
