"""Which benign packets are flagged inside the attack windows of the E9 stream? (seed 0)

For AnomalyGuard-IF and AE, reports for benign in-window false alerts (before session association)
the share sent by the attacked gateway 10.42.0.1 (vs. sent to it), and the mean of every causal
context feature for flagged versus unflagged benign in-window packets and for benign packets outside
the windows. Output: results/e9_collateral.json
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..experiments.common import write_json
from ..experiments.e9_stream import build
from ..pipeline import anomalyguard_scores, sample_split

GATEWAY = "10.42.0.1"


def main(seed: int = 0):
    df, windows, _ = build("stream", seed)
    sp = sample_split(df, seed)
    te = sp.test
    t = te["t"].to_numpy()
    inwin = np.zeros(len(te), bool)
    for a, b in windows:
        inwin |= (t >= a) & (t <= b)
    ben = te["label"].to_numpy() == 0
    gw = (te["src"].astype(str).to_numpy() == GATEWAY) | (te["dst"].astype(str).to_numpy() == GATEWAY)
    out = {"seed": seed, "benign_in_windows": int((ben & inwin).sum()),
           "share_benign_in_windows_involving_gateway": float(gw[ben & inwin].mean())}
    for label, det, params in (("AnomalyGuard-IF", "IF", {"max_samples": "rule"}), ("AE", "AE", {})):
        o = anomalyguard_scores(sp, det, seed, det_params=params, propagation=True)
        f = o["score_test"] >= o["thr"]
        fp = f & ben & inwin
        nf = ~f & ben & inwin
        out[label] = {
            "fp_in_windows": int(fp.sum()),
            "share_fp_involving_gateway": float(gw[fp].mean()) if fp.any() else None,
            "median_dst_fanin_flagged": float(np.median(te["ctx_log_dst_fanin_1s"].to_numpy()[fp])) if fp.any() else None,
            "median_dst_fanin_unflagged": float(np.median(te["ctx_log_dst_fanin_1s"].to_numpy()[nf])),
            "median_src_n1s_flagged": float(np.median(te["ctx_log_n_1s"].to_numpy()[fp])) if fp.any() else None,
            "median_src_n1s_unflagged": float(np.median(te["ctx_log_n_1s"].to_numpy()[nf])),
            "attribution_network_share": float((o["attribution"][fp] == 0).mean()) if fp.any() else None,
            "share_fp_sent_by_gateway": float((te["src"].astype(str).to_numpy()[fp] == GATEWAY).mean()) if fp.any() else None,
            "share_unflagged_sent_by_gateway": float((te["src"].astype(str).to_numpy()[nf] == GATEWAY).mean()),
            "ctx_means": {c: {"flagged": float(np.mean(te[c].to_numpy()[fp])) if fp.any() else None,
                              "unflagged_in_window": float(np.mean(te[c].to_numpy()[nf])),
                              "benign_outside": float(np.mean(te[c].to_numpy()[ben & ~inwin]))}
                          for c in te.columns if c.startswith("ctx_")},
        }
    write_json(out, C.RESULTS_DIR / "e9_collateral.json")
    print(out)


if __name__ == "__main__":
    main()
