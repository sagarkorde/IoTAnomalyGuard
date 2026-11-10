"""Shared, seed-controlled sampling and the AnomalyGuard scoring pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .calibration import empirical_pvalue, fuse_pvalues, threshold_at_fpr
from .data import load_processed
from .detectors import make
from .features import PortNovelty, TelemetryView, context_view, net_view

CTX_PREFIX = "ctx_"


def load_with_context(cache: Path | None = None) -> pd.DataFrame:
    """Processed table plus causal context features (cached as parquet)."""
    cache = Path(cache) if cache else C.PROCESSED_DIR / "xiomt_ctx.parquet"
    df = load_processed()
    if cache.exists():
        ctx = pd.read_parquet(cache)
    else:
        ctx = context_view(df)
        ctx.to_parquet(cache, index=False)
    return pd.concat([df, ctx.set_index(df.index)], axis=1)


# ------------------------------------------------------------------------------ splits
@dataclass
class Split:
    train: pd.DataFrame      # benign training pool (unsupervised detectors)
    val: pd.DataFrame        # benign validation pool (calibration only)
    test: pd.DataFrame       # every packet of every test segment (benign + anomalies)
    train_all: pd.DataFrame  # labelled training packets of all classes (supervised reference)
    val_stream: pd.DataFrame  # complete benign validation stream (propagation calibration)


def sample_split(df: pd.DataFrame, seed: int, train_cap: int = C.TRAIN_CAP,
                 val_cap: int = C.VAL_CAP, test_cap_per_type: int | None = None,
                 contamination: float = 0.0) -> Split:
    """Seeded sub-sampling of the chronological splits.

    ``contamination`` > 0 replaces that fraction of the benign training pool by anomalous
    packets drawn from the training segments (all types, proportional to their size), to
    emulate an unlabelled training capture that silently contains attacks.
    """
    rng = np.random.default_rng(seed)

    def cap(frame: pd.DataFrame, n: int | None) -> pd.DataFrame:
        if n is None or len(frame) <= n:
            return frame
        return frame.iloc[np.sort(rng.choice(len(frame), n, replace=False))]

    tr_ben = df[(df["split"] == 0) & (df["label"] == 0)]
    tr_bad = df[(df["split"] == 0) & (df["label"] == 1)]
    n_bad = int(round(contamination * train_cap))
    train = cap(tr_ben, train_cap - n_bad)
    if n_bad:
        train = pd.concat([train, cap(tr_bad, n_bad)])
    val = cap(df[(df["split"] == 1) & (df["label"] == 0)], val_cap)
    te = df[df["split"] == 2]
    if test_cap_per_type:
        parts = [cap(g, test_cap_per_type if n != "Normal" else None)
                 for n, g in te.groupby("name", observed=True)]
        te = pd.concat(parts)
    tr_all = [cap(tr_ben, train_cap)]
    for _, g in tr_bad.groupby("name", observed=True):
        tr_all.append(cap(g, 2 * C.TEST_PER_TYPE_CAP))
    val_stream = df[(df["split"] == 1) & (df["label"] == 0)]
    return Split(train=train, val=val, test=te, train_all=pd.concat(tr_all), val_stream=val_stream)


# --------------------------------------------------------------------- service context
def service_of(frame: pd.DataFrame) -> np.ndarray:
    """Coarse service context of each packet from its server port (no labels involved)."""
    p = PortNovelty.server_port(frame)
    return np.where(p == 1883, "mqtt", np.where(p == 8080, "http", "other"))


def psi_mode_coverage(services: np.ndarray, base: int = 256, min_share: float = 0.01,
                      n_max: int | None = None) -> int:
    """Label-free choice of the Isolation-Forest subsample size.

    Every benign service mode holding at least ``min_share`` of the training traffic should
    contribute about ``base`` points to each tree:  psi = 2^ceil(log2(base / pi_min)).
    """
    share = pd.Series(services).value_counts(normalize=True)
    share = share[share >= min_share]
    pi_min = float(share.min()) if len(share) else 1.0
    psi = int(2 ** np.ceil(np.log2(base / pi_min)))
    psi = max(base, psi)
    if n_max:
        psi = min(psi, n_max)
    return psi


# ------------------------------------------------------------------------------ views
def network_matrix(frame: pd.DataFrame, ports: PortNovelty, context: bool = True,
                   fingerprint: bool = True) -> pd.DataFrame:
    X = net_view(frame, ports, fingerprint=fingerprint)
    if context:
        ctx = frame[[c for c in frame.columns if c.startswith(CTX_PREFIX)]]
        X = pd.concat([X, ctx.astype(np.float32)], axis=1)
    return X


def _resolve_params(det_name: str, params: dict, services: np.ndarray, n: int) -> dict:
    p = dict(params)
    if det_name == "IF" and p.get("max_samples") == "rule":
        p["max_samples"] = psi_mode_coverage(services, n_max=n)
    return p


@dataclass
class ViewResult:
    val: np.ndarray
    test: np.ndarray
    fit_seconds: float
    score_seconds_per_pkt: float
    params: dict


def run_view(det_name: str, seed: int, Xtr, Xva, Xte, ytr=None, extra=None, **params) -> ViewResult:
    det = make(det_name, seed, **params)
    det.fit(Xtr, ytr)
    t0 = time.perf_counter()
    s_te = det.score(Xte)
    dt = (time.perf_counter() - t0) / max(1, len(Xte))
    s_va = det.score(Xva)
    res = ViewResult(s_va, s_te, det.fit_seconds, dt, params)
    res.extra = det.score(extra) if extra is not None else None
    res.model = det
    return res


def anomalyguard_scores(sp: Split, det_name: str = "IF", seed: int = 0, context: bool = True,
                        fingerprint: bool = True, fusion: bool = True,
                        det_params: dict | None = None, propagation: bool = False,
                        window_s: float = 10.0) -> dict:
    """Score every test packet with the network and telemetry views and fuse them.

    With ``fusion=False`` a single detector is fitted on the concatenation of all views
    (absent telemetry -> benign median), which is the usual "one model" design; its score
    then also drives alert propagation.
    """
    det_params = det_params or {}
    ports = PortNovelty().fit(sp.train)
    tel = TelemetryView().fit(sp.train)
    Xtr = network_matrix(sp.train, ports, context, fingerprint)
    Xva = network_matrix(sp.val, ports, context, fingerprint)
    Xte = network_matrix(sp.test, ports, context, fingerprint)
    Xvs = network_matrix(sp.val_stream, ports, context, fingerprint) if propagation else None
    out = {"net_feature_names": list(Xtr.columns)}

    if not fusion:
        def z(frame, X):
            return pd.concat([X, tel.transform(frame)], axis=1)
        Ztr, Zva, Zte = z(sp.train, Xtr), z(sp.val, Xva), z(sp.test, Xte)
        Zvs = z(sp.val_stream, Xvs) if propagation else None
        prm = _resolve_params(det_name, det_params, service_of(sp.train), len(Ztr))
        r = run_view(det_name, seed, Ztr, Zva, Zte, extra=Zvs, **prm)
        p_te = empirical_pvalue(r.test, r.val, seed)
        out.update(score_test=r.test, score_val=r.val,
                   thr=threshold_at_fpr(r.val, C.FPR_BUDGET), net_test=r.test, net_val=r.val,
                   thr_net=threshold_at_fpr(r.val, C.FPR_BUDGET),
                   attribution=np.zeros(len(Zte), int), fit_seconds=r.fit_seconds,
                   score_seconds_per_pkt=r.score_seconds_per_pkt, params_net=r.params,
                   params_tel=None, net_model=r.model)
        net_extra, net_val, p_net_te = r.extra, r.val, p_te
    else:
        prm_n = _resolve_params(det_name, det_params, service_of(sp.train), len(Xtr))
        net = run_view(det_name, seed, Xtr, Xva, Xte, extra=Xvs, **prm_n)
        mtr, mva, mte = (TelemetryView.mask(f) for f in (sp.train, sp.val, sp.test))
        prm_t = _resolve_params(det_name, det_params, service_of(sp.train[mtr]), int(mtr.sum()))
        tv = run_view(det_name, seed, tel.transform(sp.train[mtr]), tel.transform(sp.val[mva]),
                      tel.transform(sp.test[mte]), **prm_t)
        p_net_te = empirical_pvalue(net.test, net.val, seed)
        p_net_va = empirical_pvalue(net.val, net.val, seed + 1)
        p_tel_te = np.ones(len(sp.test)); p_tel_te[mte] = empirical_pvalue(tv.test, tv.val, seed + 2)
        p_tel_va = np.ones(len(sp.val)); p_tel_va[mva] = empirical_pvalue(tv.val, tv.val, seed + 3)
        fused_te, best_te = fuse_pvalues([p_net_te, p_tel_te], [np.ones(len(sp.test), bool), mte])
        fused_va, _ = fuse_pvalues([p_net_va, p_tel_va], [np.ones(len(sp.val), bool), mva])
        s_te, s_va = -np.log(fused_te), -np.log(fused_va)
        tel_test = np.full(len(sp.test), np.nan); tel_test[mte] = tv.test
        out.update(
            score_test=s_te, score_val=s_va, thr=threshold_at_fpr(s_va, C.FPR_BUDGET),
            net_test=net.test, net_val=net.val, thr_net=threshold_at_fpr(net.val, C.FPR_BUDGET),
            tel_test=tel_test, tel_mask_test=mte, attribution=best_te,
            fit_seconds=net.fit_seconds + tv.fit_seconds,
            score_seconds_per_pkt=net.score_seconds_per_pkt, params_net=net.params,
            params_tel=tv.params, net_model=net.model, tel_model=tv.model,
        )
        net_extra, net_val = net.extra, net.val

    if propagation:
        f_vs = empirical_pvalue(net_extra, net_val, seed + 4) <= C.FPR_BUDGET
        f_te = p_net_te <= C.FPR_BUDGET
        k = calibrate_propagation(sp.val_stream, f_vs, window_s)
        out["prop_k"] = k
        out["prop_flags"] = propagate_alerts(sp.test, f_te, window_s, k)
        out["prop_val_extra_fpr"] = float((propagate_alerts(sp.val_stream, f_vs, window_s, k)
                                           & ~f_vs).mean())
        out["net_flags_test"] = f_te
    return out


# ----------------------------------------------------------------- context propagation
def propagate_alerts(frame: pd.DataFrame, net_flags: np.ndarray, window_s: float = 10.0,
                     min_alerts: int = 1) -> np.ndarray:
    """Session-association rule (causal).

    A packet is flagged when its destination host raised at least ``min_alerts``
    network-view alerts *as a source* within the preceding ``window_s`` seconds, inside the
    same capture. This links a victim's replies to the host whose traffic triggered the
    alerts. ``min_alerts`` suppresses propagation from a busy benign host that raises an
    occasional false alert; it is calibrated on benign validation traffic only
    (``calibrate_propagation``).
    """
    g = frame["capture_group"].astype(str).to_numpy()
    src = frame["src"].astype(str).to_numpy()
    dst = frame["dst"].astype(str).to_numpy()
    t = frame["t"].to_numpy()
    out = np.zeros(len(frame), dtype=bool)
    alert_idx = np.flatnonzero(net_flags)
    if len(alert_idx) == 0:
        return out
    a = pd.DataFrame({"g": g[alert_idx], "h": src[alert_idx], "t": t[alert_idx]})
    times = {k: np.sort(v.to_numpy()) for k, v in a.groupby(["g", "h"])["t"]}
    q = pd.DataFrame({"g": g, "h": dst, "t": t, "i": np.arange(len(frame))})
    for key, sub in q.groupby(["g", "h"]):
        A = times.get(key)
        if A is None:
            continue
        T = sub["t"].to_numpy()
        hi = np.searchsorted(A, T, side="right")          # alerts with time <= T
        lo = np.searchsorted(A, T - window_s, side="left")  # alerts with time >= T - W
        out[sub["i"].to_numpy()[(hi - lo) >= min_alerts]] = True
    return out


def calibrate_propagation(val_stream: pd.DataFrame, val_flags: np.ndarray, window_s: float = 10.0,
                          extra_fpr: float = 0.001,
                          grid=(1, 2, 3, 5, 10, 20, 50, 100, 200)) -> int:
    """Smallest ``min_alerts`` whose propagation adds at most ``extra_fpr`` false alerts on
    the *complete* benign validation stream (no anomaly labels are used)."""
    base = val_flags.astype(bool)
    for k in grid:
        add = propagate_alerts(val_stream, base, window_s, k) & ~base
        if add.mean() <= extra_fpr:
            return int(k)
    return int(grid[-1])
