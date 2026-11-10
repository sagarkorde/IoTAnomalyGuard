"""Numerical check of our Deep IF scoring against the authors' reference scoring function.

``_reference_cal_score`` is copied, with comments removed, from ``algorithms/dif.py`` of
https://github.com/xuhongzuo/deep-iforest (Apache License 2.0, (c) the DIF authors). The
reference's neural representation (bias-free linear layers with N(0, 1) weights and tanh between
layers, no output activation) is plain matrix algebra and is reproduced in DIFDet._forward.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length

from anomalyguard.detectors import DIFDet, make


def _reference_cal_score(xx, clf):
    depth_sum = np.zeros(xx.shape[0])
    deviations = np.zeros((xx.shape[0], len(clf.estimators_)))
    for ii, estimator_tree in enumerate(clf.estimators_):
        tree = estimator_tree.tree_
        if tree.node_count == 1:
            continue
        feature_lst, threshold_lst = tree.feature.copy(), tree.threshold.copy()
        leaves_index = estimator_tree.apply(xx)
        node_indicator = estimator_tree.decision_path(xx)
        n_samples_leaf = estimator_tree.tree_.n_node_samples[leaves_index]
        d = (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
        depth_sum += d
        node_indicator = np.array(node_indicator.todense())
        value_mat = np.array([xx[i][feature_lst] for i in range(xx.shape[0])])
        value_mat[:, np.where(feature_lst == -2)[0]] = -2
        th_mat = np.array([threshold_lst for _ in range(xx.shape[0])])
        mat = np.abs(value_mat - th_mat) * node_indicator
        exist = (mat != 0)
        deviations[:, ii] = mat.sum(axis=1) / (exist.sum(axis=1) + 1e-6)
    scores = 2 ** (-depth_sum / (len(clf.estimators_) * _average_path_length([clf.max_samples_])))
    return scores * np.mean(deviations, axis=1)


def test_deaas_matches_reference_scoring():
    rng = np.random.default_rng(3)
    R = np.tanh(rng.normal(size=(1500, 20)))
    clf = IsolationForest(n_estimators=6, max_samples=256, random_state=0).fit(R)
    Q = np.tanh(rng.normal(size=(400, 20)) * 1.5)
    np.testing.assert_allclose(DIFDet._deaas(clf, Q), _reference_cal_score(Q, clf), rtol=1e-10, atol=1e-12)


def test_dif_ensemble_score_is_mean_of_reference_scores():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1200, 7))
    d = make("DIF", 0, n_ensemble=4).fit(X)
    Q = rng.normal(size=(300, 7)) * 2
    Z = d.scaler.transform(Q)
    ref = np.mean([_reference_cal_score(np.tanh(sc.transform(d._forward(Z, W))), f)
                   for W, sc, f in zip(d.nets, d.rep_scalers, d.forests)], axis=0)
    np.testing.assert_allclose(d.score(Q), ref, rtol=1e-9, atol=1e-12)
