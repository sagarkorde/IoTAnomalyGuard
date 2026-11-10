"""E12 - Does the label-free psi rule track the best subsample size when the service mix changes?

The benign training (100k), calibration (40k) and benign test (40k) samples are re-drawn so that
HTTP makes up a share s of each (the rest keeps its natural MQTT/other ratio). For every s, an
Isolation Forest on the network view (with context, no fusion, no session association) is run
over a grid of subsample sizes and calibrated at a 1 % budget; we record network-attack recall.
The rule's choice psi*(s) = 2^ceil(log2(256 / pi_min)) is compared with the best grid value.
Output: results/e12_psi_mix.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..calibration import threshold_at_fpr
from ..features import PortNovelty
from ..pipeline import load_with_context, network_matrix, psi_mode_coverage, run_view, service_of
from .common import log

SHARES = (0.025, 0.05, 0.10, 0.20, 0.40)
PSI_GRID = (256, 512, 1024, 2048, 4096, 8192, 16384)
N_TRAIN, N_VAL, N_TEST = 100_000, 40_000, 40_000


def mix(frame: pd.DataFrame, share: float, n: int, rng) -> pd.DataFrame:
    svc = service_of(frame)
    http = np.flatnonzero(svc == "http"); rest = np.flatnonzero(svc != "http")
    nh = int(round(share * n))
    pick = np.r_[rng.choice(http, nh, replace=False), rng.choice(rest, n - nh, replace=False)]
    return frame.iloc[np.sort(pick)]


def main(seeds=(0, 1)):
    df = load_with_context()
    ben = df[df["label"] == 0]
    net_L = [k for k, v in C.LAYER_NAMES.items() if v == "Network"][0]
    net_attacks = df[(df["split"] == 2) & (df["label"] == 1) & (df["layer"] == net_L)]
    rows = []
    for seed in seeds:
        for share in SHARES:
            rng = np.random.default_rng(seed * 100 + int(share * 1000))
            tr = mix(ben[ben["split"] == 0], share, N_TRAIN, rng)
            va = mix(ben[ben["split"] == 1], share, N_VAL, rng)
            tb = mix(ben[ben["split"] == 2], share, N_TEST, rng)
            ports = PortNovelty().fit(tr)
            Xtr, Xva = network_matrix(tr, ports), network_matrix(va, ports)
            Xte = network_matrix(pd.concat([tb, net_attacks]), ports)
            y = np.r_[np.zeros(len(tb)), np.ones(len(net_attacks))]
            names = np.r_[np.array(["Normal"] * len(tb)), net_attacks["name"].astype(str).to_numpy()]
            rule = psi_mode_coverage(service_of(tr), n_max=len(tr))
            for psi in PSI_GRID:
                r = run_view("IF", seed, Xtr, Xva, Xte, max_samples=psi)
                thr = threshold_at_fpr(r.val, C.FPR_BUDGET)
                flag = r.test >= thr
                rec = {n: float(flag[names == n].mean()) for n in np.unique(names[y == 1])}
                rows.append({"seed": seed, "http_share": share, "psi": psi, "psi_rule": rule,
                             "network_macro_recall": float(np.mean(list(rec.values()))),
                             "network_recall": float(flag[y == 1].mean()), "fpr": float(flag[y == 0].mean())})
                log(f"seed {seed} share={share:.3f} psi={psi:5d} (rule {rule}) netR={rows[-1]['network_macro_recall']:.3f} "
                    f"FPR={rows[-1]['fpr']:.4f}")
            pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e12_psi_mix.csv", index=False)


if __name__ == "__main__":
    main()
