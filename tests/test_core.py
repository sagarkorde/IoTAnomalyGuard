"""Unit tests for the components whose correctness every reported number depends on.

These tests use synthetic data only, so they run without downloading X-IoMT.
"""
import numpy as np
import pandas as pd
import pytest

from anomalyguard.calibration import empirical_pvalue, fuse_pvalues, threshold_at_fpr
from anomalyguard.data import TOK_NAN, TOK_PRESENT, TOK_ZERO, TOK_ZERO_DOT, _token_style, capture_group
from anomalyguard.detectors import make
from anomalyguard.features import PortNovelty, TelemetryView, context_view
from anomalyguard.metrics import detection_report
from anomalyguard.pipeline import calibrate_propagation, propagate_alerts, psi_mode_coverage


# ----------------------------------------------------------------------------- data
def test_capture_group_strips_part_and_version():
    assert capture_group("NormalTrafficMQTT_part_012_v11.csv") == "NormalTrafficMQTT"
    assert capture_group("ApplicationAnomaliesClinical_v11.csv") == "ApplicationAnomaliesClinical"


def test_token_style_distinguishes_absent_spellings():
    s = pd.Series(["POST", "0", "0.0", None, "nan"], dtype="string")
    assert list(_token_style(s)) == [TOK_PRESENT, TOK_ZERO, TOK_ZERO_DOT, TOK_NAN, TOK_NAN]


# ---------------------------------------------------------------------- calibration
def test_threshold_respects_budget_with_heavy_ties():
    rng = np.random.default_rng(0)
    v = np.r_[np.zeros(9_000), rng.random(1_000)]          # 90 % tied benign scores
    for fpr in (0.001, 0.01, 0.05, 0.2):
        thr = threshold_at_fpr(v, fpr)
        assert np.mean(v >= thr) <= fpr + 1e-12


def test_smoothed_pvalues_are_uniform_on_benign():
    rng = np.random.default_rng(1)
    cal = np.round(rng.normal(size=5_000), 1)               # discrete scores -> ties
    new = np.round(rng.normal(size=5_000), 1)
    p = empirical_pvalue(new, cal, seed=0)
    # P(p <= a) ~= a for exchangeable benign points (tie-breaking makes it exact in law)
    for a in (0.01, 0.05, 0.1, 0.5):
        assert abs(np.mean(p <= a) - a) < 0.02


def test_fusion_bonferroni_and_attribution():
    p1 = np.array([0.5, 0.001, 0.2])
    p2 = np.array([0.004, 0.5, 1.0])
    m1 = np.array([True, True, True])
    m2 = np.array([True, True, False])
    fused, best = fuse_pvalues([p1, p2], [m1, m2])
    assert np.allclose(fused, [0.008, 0.002, 0.2])
    assert list(best) == [1, 0, 0]


# --------------------------------------------------------------------------- metrics
def test_detection_report_counts():
    y = np.array([0, 0, 0, 1, 1, 1])
    names = np.array(["Normal"] * 3 + ["A", "A", "B"])
    s = np.array([0.1, 0.2, 0.9, 0.95, 0.1, 0.99])
    r = detection_report(y, names, s, 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 2)
    assert r["per_type_recall"] == {"A": 0.5, "B": 1.0}
    assert r["macro_recall"] == 0.75


# -------------------------------------------------------------------------- features
def _toy_stream():
    t = np.array([0.0, 0.5, 1.2, 1.3, 5.0, 0.0, 0.1])
    return pd.DataFrame({
        "capture_group": pd.Categorical(["g"] * 5 + ["h"] * 2),
        "src": pd.Categorical(["a", "a", "a", "b", "a", "a", "a"]),
        "dst": pd.Categorical(["x"] * 7),
        "t": t,
        "tcp.flags.syn": [1, 0, 1, 1, 0, 1, 1], "tcp.flags.fin": [0] * 7,
        "tcp.flags.reset": [0] * 7, "tcp.flags.push": [0] * 7,
        "frame.len": [60.0] * 7, "tcp.dstport": [80, 81, 82, 80, 80, 80, 80],
        "tcp.srcport": [5000] * 7,
    })


def test_context_features_are_causal():
    df = _toy_stream()
    full = context_view(df)
    # Removing the last packet of stream (g, a) must not change any earlier feature value.
    trimmed = context_view(df.drop(index=4))
    pd.testing.assert_frame_equal(full.drop(index=4), trimmed, check_dtype=False)


def test_context_streams_do_not_mix_captures():
    df = _toy_stream()
    ctx = context_view(df)
    # First packet of capture h (index 5) has no history even though capture g used src "a".
    assert ctx.loc[5, "ctx_log_n_10s"] == pytest.approx(np.log1p(1))
    # Packet 2 of (g, a) at t=1.2 sees packets at 0.5 and 1.2 inside the 1 s window.
    assert ctx.loc[2, "ctx_log_n_1s"] == pytest.approx(np.log1p(2))


def test_port_novelty_learned_from_benign_only():
    tr = pd.DataFrame({"tcp.srcport": [50000] * 60, "tcp.dstport": [1883] * 60})
    te = pd.DataFrame({"tcp.srcport": [50001, 50002], "tcp.dstport": [1883, 8000]})
    out = PortNovelty(min_count=50).fit(tr).transform(te)
    assert list(out["port_known"]) == [1.0, 0.0]


def test_telemetry_absence_is_uninformative():
    tr = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in ("Temperature", "HR")})
    tv = TelemetryView(cols=("Temperature", "HR")).fit(tr)
    te = pd.DataFrame({"Temperature": [np.nan], "HR": [2.0]})
    assert tv.transform(te).iloc[0].tolist() == [2.0, 2.0]   # imputed to benign median


# -------------------------------------------------------------------------- pipeline
def test_psi_rule():
    services = np.array(["mqtt"] * 900 + ["http"] * 100)
    assert psi_mode_coverage(services) == 4096            # 256 / 0.1 = 2560 -> 4096
    assert psi_mode_coverage(np.array(["mqtt"] * 1000)) == 256
    assert psi_mode_coverage(services, n_max=3000) == 3000


def test_propagation_requires_min_alerts_and_is_causal():
    df = pd.DataFrame({
        "capture_group": ["g"] * 6, "src": ["atk", "atk", "atk", "vic", "vic", "vic"],
        "dst": ["vic", "vic", "vic", "atk", "atk", "atk"],
        "t": [0.0, 1.0, 2.0, 2.5, 30.0, -1.0],
    })
    flags = np.array([True, True, True, False, False, False])
    out = propagate_alerts(df, flags, window_s=10.0, min_alerts=3)
    # reply at 2.5 s follows 3 alerts in the window; reply at 30 s is outside it;
    # reply at -1 s precedes every alert (causality)
    assert list(out[3:]) == [True, False, False]
    assert not propagate_alerts(df, flags, window_s=10.0, min_alerts=4).any()


def test_propagation_calibration_on_benign():
    df = pd.DataFrame({"capture_group": ["g"] * 4, "src": ["a", "b", "a", "b"],
                       "dst": ["b", "a", "b", "a"], "t": [0.0, 0.1, 0.2, 0.3]})
    flags = np.array([True, False, False, False])
    # with k=1 the replies to "a" would add false alerts (50 %); k=2 adds none
    assert calibrate_propagation(df, flags, extra_fpr=0.0, grid=(1, 2)) == 2


# -------------------------------------------------------------------------- detectors
@pytest.mark.parametrize("name", ["IF", "ECOD", "HBOS", "LOF", "OCSVM", "AE"])
def test_detectors_rank_obvious_outliers_higher(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2_000, 4))
    Xo = rng.normal(loc=8.0, size=(20, 4))
    det = make(name, 0).fit(X)
    assert np.median(det.score(Xo)) > np.quantile(det.score(X[:500]), 0.99)


def test_isolation_forest_score_matches_paper_definition():
    """score = 2^(-E[h(x)]/c(psi)) = -score_samples(x) in scikit-learn."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1_000, 3))
    det = make("IF", 0, max_samples=256).fit(X)
    s = det.score(X[:10])
    assert np.all((s > 0) & (s < 1))
    assert np.allclose(s, -det.model.score_samples(X[:10]))
