"""E8 - Validation-selected detector per view (RQ2, practical configuration).

E2 shows that no single detector is best on every layer. Choosing a per-view detector by
looking at test results would be test-set snooping, so here the (network, telemetry) detector
pair is selected on the *validation* split - chronologically before the test split - using its
labelled anomalies, and only the selected pair is then evaluated on the untouched test split.
This deliberately uses validation labels for model selection and is reported as such.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import config as C
from ..calibration import empirical_pvalue, fuse_pvalues, threshold_at_fpr
from ..features import PortNovelty, TelemetryView
from ..metrics import detection_report
from ..pipeline import (_resolve_params, calibrate_propagation, load_with_context, network_matrix,
                        propagate_alerts, run_view, sample_split, service_of)
from .common import evaluate, layer_map, log

CANDIDATES = {"IF": {"max_samples": "rule"}, "OCSVM": {}, "AE": {}, "LOF": {}}


def validation_eval_set(df: pd.DataFrame, sp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    parts = [sp.val]
    for _, g in df[(df["split"] == 1) & (df["label"] == 1)].groupby("name", observed=True):
        n = min(len(g), C.TEST_PER_TYPE_CAP)
        parts.append(g.iloc[np.sort(rng.choice(len(g), n, replace=False))])
    return pd.concat(parts)


def main(seeds=C.SEEDS):
    df = load_with_context()
    layers = layer_map(df)
    rows, sel_rows = [], []
    for seed in seeds:
        sp = sample_split(df, seed)
        veval = validation_eval_set(df, sp, seed)
        ports = PortNovelty().fit(sp.train)
        tel = TelemetryView().fit(sp.train)
        Xtr, Xca = network_matrix(sp.train, ports), network_matrix(sp.val, ports)
        Xve, Xte = network_matrix(veval, ports), network_matrix(sp.test, ports)
        Xvs = network_matrix(sp.val_stream, ports)
        mtr, mca, mve, mte = (TelemetryView.mask(f) for f in (sp.train, sp.val, veval, sp.test))
        net, telv = {}, {}
        for name, params in CANDIDATES.items():
            prm = _resolve_params(name, params, service_of(sp.train), len(Xtr))
            r = run_view(name, seed, Xtr, Xca, np.vstack([Xve, Xte]), extra=Xvs, **prm)
            net[name] = dict(cal=r.val, ve=r.test[:len(Xve)], te=r.test[len(Xve):], vs=r.extra)
            prm = _resolve_params(name, params, service_of(sp.train[mtr]), int(mtr.sum()))
            T = np.vstack([tel.transform(veval[mve]), tel.transform(sp.test[mte])])
            r = run_view(name, seed, tel.transform(sp.train[mtr]), tel.transform(sp.val[mca]), T, **prm)
            telv[name] = dict(cal=r.val, ve=r.test[:int(mve.sum())], te=r.test[int(mve.sum()):])
            log(f"seed {seed}: fitted {name}")

        def fused(nn, tt, part):
            mask = mve if part == "ve" else mte
            n = len(mask)
            p_n = empirical_pvalue(net[nn][part], net[nn]["cal"], seed)
            p_t = np.ones(n); p_t[mask] = empirical_pvalue(telv[tt][part], telv[tt]["cal"], seed + 2)
            p_nc = empirical_pvalue(net[nn]["cal"], net[nn]["cal"], seed + 1)
            p_tc = np.ones(len(sp.val)); p_tc[mca] = empirical_pvalue(telv[tt]["cal"], telv[tt]["cal"], seed + 3)
            f, best = fuse_pvalues([p_n, p_t], [np.ones(n, bool), mask])
            fc, _ = fuse_pvalues([p_nc, p_tc], [np.ones(len(sp.val), bool), mca])
            return -np.log(f), threshold_at_fpr(-np.log(fc), C.FPR_BUDGET), p_n

        yv, nv = veval["label"].to_numpy(), veval["name"].astype(str).to_numpy()
        scores = {}
        for nn, tt in itertools.product(CANDIDATES, CANDIDATES):
            s, thr, _ = fused(nn, tt, "ve")
            scores[(nn, tt)] = detection_report(yv, nv, s, thr)["macro_recall"]
            sel_rows.append({"seed": seed, "net": nn, "tel": tt, "val_macro_recall": scores[(nn, tt)]})
        nn, tt = max(scores, key=scores.get)
        s, thr, p_n = fused(nn, tt, "te")
        f_vs = empirical_pvalue(net[nn]["vs"], net[nn]["cal"], seed + 4) <= C.FPR_BUDGET
        k = calibrate_propagation(sp.val_stream, f_vs)
        prop = propagate_alerts(sp.test, p_n <= C.FPR_BUDGET, 10.0, k)
        for vname, extra in (("fused", None), ("fused+propagation", prop)):
            row, _ = evaluate(sp.test, s, thr, layers, extra)
            rows.append({"seed": seed, "net": nn, "tel": tt, "variant": vname, "prop_k": k,
                         "val_macro_recall": scores[(nn, tt)], **row})
        log(f"seed {seed}: selected net={nn} tel={tt} (val macroR={scores[(nn, tt)]:.3f}) "
            f"test macroR={rows[-1]['macro_recall']:.3f} FPR={rows[-1]['fpr']:.4f}")
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "e8_view_selection.csv", index=False)
        pd.DataFrame(sel_rows).to_csv(C.RESULTS_DIR / "e8_selection_grid.csv", index=False)


if __name__ == "__main__":
    main()
