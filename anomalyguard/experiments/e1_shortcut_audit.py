"""E1 - Shortcut audit and protocol inflation (RQ1).

(a) Probe classifiers trained on feature sets that carry *no traffic behaviour*
    (absent-value spellings + telemetry presence bits; host identity fields).
    If they separate normal from anomalous packets, the dataset admits shortcuts.
(b) The same supervised and unsupervised models under a typical "naive" protocol
    (every column label-encoded, random 70/30 row split) versus the leakage-aware protocol
    (behaviour-only views, chronological split).

Outputs: results/e1_probes.csv, results/e1_protocols.csv, results/e1_naive_importance.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .. import config as C
from ..calibration import threshold_at_fpr
from ..features import PortNovelty, TelemetryView, artefact_views, naive_view
from ..metrics import detection_report
from ..pipeline import load_with_context, network_matrix
from .common import log, write_json

PER_CLASS_TRAIN = 20_000


def _balanced_train_index(df, mask, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for _, g in df[mask].groupby("name", observed=True):
        take = min(len(g), PER_CLASS_TRAIN * (5 if g["label"].iloc[0] == 0 else 1))
        idx.append(rng.choice(g.index.to_numpy(), take, replace=False))
    return np.sort(np.concatenate(idx))


def _supervised_eval(X, y, names, tr, te, seed, tag):
    m = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X.loc[tr], y[tr])
    p = m.predict_proba(X.loc[te])[:, 1]
    rep = detection_report(y[te], names[te], p, 0.5)
    return m, {"setting": tag, "seed": seed, "roc_auc": roc_auc_score(y[te], p),
               "accuracy": accuracy_score(y[te], p >= 0.5), "f1": f1_score(y[te], p >= 0.5),
               "macro_recall": rep["macro_recall"], "fpr": rep["fpr"], "n_features": X.shape[1]}


def main(seeds=(0, 1, 2)):
    df = load_with_context()
    y = df["label"].to_numpy()
    names = df["name"].astype(str).to_numpy()
    chrono_tr = df["split"].to_numpy() == 0
    chrono_te = df["split"].to_numpy() == 2
    probes, protos, importances = [], [], []
    views = artefact_views(df)
    naive = naive_view(df)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        tr_c = _balanced_train_index(df, chrono_tr, seed)
        te_c = np.flatnonzero(chrono_te)
        # (a) probes on artefact-only feature sets, chronological split
        for vname, X in views.items():
            _, r = _supervised_eval(X, y, names, tr_c, te_c, seed, f"probe:{vname}")
            probes.append(r); log(f"seed {seed} {r['setting']}: AUROC={r['roc_auc']:.4f} acc={r['accuracy']:.4f}")

        # (b) naive protocol: every column, random 70/30 row split
        perm = rng.permutation(len(df))
        cut = int(0.7 * len(df))
        rnd_tr_mask = np.zeros(len(df), bool); rnd_tr_mask[perm[:cut]] = True
        tr_r = _balanced_train_index(df, rnd_tr_mask, seed)
        te_r = np.sort(perm[cut:])
        m, r = _supervised_eval(naive, y, names, tr_r, te_r, seed, "naive:HGB:random-split")
        protos.append(r); log(f"seed {seed} {r['setting']}: AUROC={r['roc_auc']:.4f} acc={r['accuracy']:.4f}")
        if seed == seeds[0]:
            sub = rng.choice(te_r, 20_000, replace=False)
            pi = permutation_importance(m, naive.loc[sub], y[sub], n_repeats=3, random_state=seed,
                                        scoring="roc_auc", n_jobs=1)
            importances = sorted(zip(naive.columns, pi.importances_mean), key=lambda t: -t[1])
        _, r = _supervised_eval(naive, y, names, tr_c, te_c, seed, "naive:HGB:chronological")
        protos.append(r); log(f"seed {seed} {r['setting']}: AUROC={r['roc_auc']:.4f}")

        # leakage-aware supervised reference: behaviour-only views, chronological split
        ben_tr = df.loc[tr_c][df.loc[tr_c, "label"] == 0]
        ports = PortNovelty().fit(ben_tr)
        tel = TelemetryView().fit(ben_tr)
        Xb = pd.concat([network_matrix(df, ports), tel.transform(df)], axis=1)
        _, r = _supervised_eval(Xb, y, names, tr_c, te_c, seed, "leakage-aware:HGB:chronological")
        protos.append(r); log(f"seed {seed} {r['setting']}: AUROC={r['roc_auc']:.4f}")

        # unsupervised IF (default psi) under the naive protocol
        for tag, tr_mask, te_idx in (("naive:IF-default:random-split", rnd_tr_mask, te_r),
                                     ("naive:IF-default:chronological", chrono_tr, te_c)):
            ben = np.flatnonzero(tr_mask & (y == 0))
            fit_idx = rng.choice(ben, min(C.TRAIN_CAP, len(ben)), replace=False)
            val_idx = rng.choice(np.flatnonzero((df["split"].to_numpy() == 1) & (y == 0) & ~tr_mask)
                                 if tag.endswith("chronological") else
                                 np.setdiff1d(ben, fit_idx), C.VAL_CAP, replace=False)
            f = IsolationForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=2)
            f.fit(naive.iloc[fit_idx])
            s_te = -f.score_samples(naive.iloc[te_idx])
            thr = threshold_at_fpr(-f.score_samples(naive.iloc[val_idx]), C.FPR_BUDGET)
            rep = detection_report(y[te_idx], names[te_idx], s_te, thr)
            protos.append({"setting": tag, "seed": seed, "roc_auc": rep["roc_auc"],
                           "accuracy": accuracy_score(y[te_idx], s_te >= thr), "f1": rep["f1"],
                           "macro_recall": rep["macro_recall"], "fpr": rep["fpr"],
                           "n_features": naive.shape[1]})
            log(f"seed {seed} {tag}: AUROC={rep['roc_auc']:.4f} macroR={rep['macro_recall']:.3f}")

        pd.DataFrame(probes).to_csv(C.RESULTS_DIR / "e1_probes.csv", index=False)
        pd.DataFrame(protos).to_csv(C.RESULTS_DIR / "e1_protocols.csv", index=False)
    pd.DataFrame(importances, columns=["feature", "auroc_drop"]).to_csv(
        C.RESULTS_DIR / "e1_naive_importance.csv", index=False)

    # descriptive evidence: how absence is spelled in each capture group
    tok_cols = [c for c in df.columns if c.startswith("tok__")]
    spell = df.groupby("capture_group", observed=True)[tok_cols].agg(lambda s: float((s == 2).mean()))
    spell.to_csv(C.RESULTS_DIR / "e1_zero_dot_share_by_capture.csv")
    pres = df.groupby("capture_group", observed=True)[list(C.TELEMETRY_COLS)].agg(lambda s: float(s.notna().mean()))
    pres.to_csv(C.RESULTS_DIR / "e1_telemetry_presence_by_capture.csv")
    hosts = df.groupby("capture_group", observed=True)["src"].agg(lambda s: sorted(set(map(str, s))))
    write_json({k: v for k, v in hosts.items()}, C.RESULTS_DIR / "e1_sources_by_capture.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args()
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    main(tuple(a.seeds))
