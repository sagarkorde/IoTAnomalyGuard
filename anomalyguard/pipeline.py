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
                 contamination: float = 0.0, val_contamination: float = 0.0) -> Split:
    """Seeded sub-sampling of the chronological splits.

    ``contamination`` > 0 replaces that fraction of the benign training pool by anomalous
    packets drawn from the training segments (all types, proportional to their size), to
    emulate an unlabelled training capture that silently contains attacks.
    ``val_contamination`` > 0 does the same for the calibration (validation) sample, with
    anomalies drawn from the validation segments.
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
    n_vbad = int(round(val_contamination * val_cap))
    val = cap(df[(df["split"] == 1) & (df["label"] == 0)], val_cap - n_vbad)
    if n_vbad:
        val = pd.concat([val, cap(df[(df["split"] == 1) & (df["label"] == 1)], n_vbad)])
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


def run_view_stratified(det_name: str, seed: int, Xtr, Xva, Xte, s_tr, s_va, s_te, extra=None,
                        s_ex=None, **params) -> ViewResult:
    """Service-stratified variant (Mondrian calibration): one detector per service context,
    each calibrated on the benign validation packets of its own service. Returned scores are
    -log of the per-service conformal p-values, so the pooled calibration downstream is a
    monotone re-use of them."""
    params = {k: v for k, v in params.items() if k != "stratify"}
    Xtr, Xva, Xte = (np.asarray(x, dtype=np.float64) for x in (Xtr, Xva, Xte))
    Xex = None if extra is None else np.asarray(extra, dtype=np.float64)
    val, test = np.zeros(len(Xva)), np.zeros(len(Xte))
    ex = None if Xex is None else np.zeros(len(Xex))
    fit_s, score_s, n_sc = 0.0, 0.0, 0
    services = [s for s in np.unique(s_tr) if (s_tr == s).sum() >= 2]
    fallback = max(services, key=lambda s: (s_tr == s).sum())
    for s in np.unique(np.concatenate([s_tr, s_va, s_te] + ([s_ex] if s_ex is not None else []))):
        m_tr = s_tr == (s if s in services else fallback)
        prm = dict(params)
        if "max_samples" in prm and isinstance(prm["max_samples"], int):
            prm["max_samples"] = min(prm["max_samples"], int(m_tr.sum()))
        det = make(det_name, seed, **prm)
        det.fit(Xtr[m_tr])
        fit_s += det.fit_seconds
        mv, mt = s_va == s, s_te == s
        ref = det.score(Xva[mv]) if mv.any() else det.score(Xtr[m_tr])
        if mv.any():
            val[mv] = -np.log(empirical_pvalue(ref, ref, seed + 1))
        if mt.any():
            t0 = time.perf_counter()
            st = det.score(Xte[mt])
            score_s += time.perf_counter() - t0; n_sc += int(mt.sum())
            test[mt] = -np.log(empirical_pvalue(st, ref, seed))
        if Xex is not None and (s_ex == s).any():
            me = s_ex == s
            ex[me] = -np.log(empirical_pvalue(det.score(Xex[me]), ref, seed + 5))
    res = ViewResult(val, test, fit_s, score_s / max(1, n_sc), dict(params, stratify=True))
    res.extra = ex
    res.model = None
    return res


def _view(det_name, seed, params, frames, Xtr, Xva, Xte, extra=None):
    """Dispatch to the pooled or the service-stratified view runner."""
    if params.get("stratify"):
        ftr, fva, fte, fex = frames
        return run_view_stratified(det_name, seed, Xtr, Xva, Xte, service_of(ftr), service_of(fva),
                                   service_of(fte), extra=extra,
                                   s_ex=None if fex is None else service_of(fex), **params)
    return run_view(det_name, seed, Xtr, Xva, Xte, extra=extra, **params)


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
        net = _view(det_name, seed, prm_n, (sp.train, sp.val, sp.test, sp.val_stream if propagation else None),
                    Xtr, Xva, Xte, extra=Xvs)
        mtr, mva, mte = (TelemetryView.mask(f) for f in (sp.train, sp.val, sp.test))
        prm_t = _resolve_params(det_name, det_params, service_of(sp.train[mtr]), int(mtr.sum()))
        tv = _view(det_name, seed, prm_t, (sp.train[mtr], sp.val[mva], sp.test[mte], None),
                   tel.transform(sp.train[mtr]), tel.transform(sp.val[mva]), tel.transform(sp.test[mte]))
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
        k_pair = calibrate_propagation(sp.val_stream, f_vs, window_s, mode="pair")
        out["prop_k_pair"] = k_pair
        out["prop_flags_pair"] = propagate_alerts(sp.test, f_te, window_s, k_pair, mode="pair")
    return out


# ----------------------------------------------------------------- context propagation
def propagate_alerts(frame: pd.DataFrame, net_flags: np.ndarray, window_s: float = 10.0,
                     min_alerts: int = 1, mode: str = "host") -> np.ndarray:
    """Session-association rule (causal).

    A packet is flagged when its destination host raised at least ``min_alerts``
    network-view alerts *as a source* within the preceding ``window_s`` seconds, inside the
    same capture. This links a victim's replies to the host whose traffic triggered the
    alerts. ``min_alerts`` suppresses propagation from a busy benign host that raises an
    occasional false alert; it is calibrated on benign validation traffic only
    (``calibrate_propagation``).

    ``mode="pair"`` is the conservative host-pair variant: packet i (source a, destination d) is
    flagged only if d raised at least ``min_alerts`` alerts *addressed to a* in the window, so
    a victim's replies to an attacker are linked but its traffic with other hosts is not.
    """
    g = frame["capture_group"].astype(str).to_numpy()
    src = frame["src"].astype(str).to_numpy()
    dst = frame["dst"].astype(str).to_numpy()
    t = frame["t"].to_numpy()
    out = np.zeros(len(frame), dtype=bool)
    alert_idx = np.flatnonzero(net_flags)
    if len(alert_idx) == 0:
        return out
    if mode == "pair":
        a = pd.DataFrame({"g": g[alert_idx], "h": src[alert_idx], "p": dst[alert_idx], "t": t[alert_idx]})
        keys = ["g", "h", "p"]
        q = pd.DataFrame({"g": g, "h": dst, "p": src, "t": t, "i": np.arange(len(frame))})
    else:
        a = pd.DataFrame({"g": g[alert_idx], "h": src[alert_idx], "t": t[alert_idx]})
        keys = ["g", "h"]
        q = pd.DataFrame({"g": g, "h": dst, "t": t, "i": np.arange(len(frame))})
    times = {k: np.sort(v.to_numpy()) for k, v in a.groupby(keys)["t"]}
    for key, sub in q.groupby(keys):
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
                          grid=(1, 2, 3, 5, 10, 20, 50, 100, 200), mode: str = "host") -> int:
    """Smallest ``min_alerts`` whose propagation adds at most ``extra_fpr`` false alerts on
    the *complete* benign validation stream (no anomaly labels are used)."""
    base = val_flags.astype(bool)
    for k in grid:
        add = propagate_alerts(val_stream, base, window_s, k, mode) & ~base
        if add.mean() <= extra_fpr:
            return int(k)
    return int(grid[-1])
