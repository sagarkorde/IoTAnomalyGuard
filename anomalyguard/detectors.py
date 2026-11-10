"""Detectors behind one interface: ``fit(X)`` then ``score(X)`` (higher = more anomalous).

Unsupervised detectors are fitted on benign training packets only. The supervised
reference (``HGB``) is fitted on labelled training packets of all classes and is reported
only as an upper bound on what labels can buy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _clean(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


@dataclass
class Detector:
    name: str
    seed: int = 0
    params: dict = field(default_factory=dict)
    fit_seconds: float = 0.0

    # --- subclasses implement _fit / _score
    def fit(self, X, y=None):
        t0 = time.perf_counter()
        self._fit(_clean(X), y)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def score(self, X) -> np.ndarray:
        return np.asarray(self._score(_clean(X)), dtype=np.float64)


class IForest(Detector):
    """Isolation Forest; the score is s(x, psi) = 2^(-E[h(x)]/c(psi)) = -score_samples(x)."""

    def _fit(self, X, y):
        p = dict(n_estimators=200, max_samples=256, max_features=1.0, n_jobs=2)
        p.update(self.params)
        self.model = IsolationForest(random_state=self.seed, **p).fit(X)

    def _score(self, X):
        return -self.model.score_samples(X)


class ECODDet(Detector):
    def _fit(self, X, y):
        from pyod.models.ecod import ECOD
        self.model = ECOD().fit(X)

    def _score(self, X):
        return self.model.decision_function(X)


class HBOSDet(Detector):
    def _fit(self, X, y):
        from pyod.models.hbos import HBOS
        self.model = HBOS(n_bins=self.params.get("n_bins", 20)).fit(X)

    def _score(self, X):
        return self.model.decision_function(X)


class LOFDet(Detector):
    """LOF in novelty mode on de-duplicated, standardised benign samples."""

    def _fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Z = np.unique(np.round(self.scaler.transform(X), 6), axis=0)
        cap = self.params.get("cap", 20_000)
        if len(Z) > cap:
            Z = Z[np.random.default_rng(self.seed).choice(len(Z), cap, replace=False)]
        self.n_fit_ = len(Z)
        self.model = LocalOutlierFactor(n_neighbors=self.params.get("k", 20), novelty=True,
                                        n_jobs=2).fit(Z)

    def _score(self, X):
        return -self.model.score_samples(self.scaler.transform(X))


class OCSVMDet(Detector):
    """One-class SVM with an RBF kernel approximated by Nystroem features (scalable)."""

    def _fit(self, X, y):
        d = X.shape[1]
        self.model = make_pipeline(
            StandardScaler(),
            Nystroem(gamma=1.0 / d, n_components=self.params.get("n_components", 300),
                     random_state=self.seed),
            SGDOneClassSVM(nu=self.params.get("nu", 0.05), random_state=self.seed),
        ).fit(X)

    def _score(self, X):
        return -self.model.decision_function(X)


class AEDet(Detector):
    """Under-complete autoencoder (MLP) - score is the mean squared reconstruction error."""

    def _fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Z = np.clip(self.scaler.transform(X), -20, 20)
        d = Z.shape[1]
        h = max(4, d // 2)
        b = max(2, d // 4)
        self.model = MLPRegressor(hidden_layer_sizes=(h, b, h), activation="relu",
                                  max_iter=self.params.get("max_iter", 60), early_stopping=True,
                                  random_state=self.seed, batch_size=512,
                                  learning_rate_init=1e-3).fit(Z, Z)

    def _score(self, X):
        Z = np.clip(self.scaler.transform(X), -20, 20)
        return np.mean((self.model.predict(Z) - Z) ** 2, axis=1)


class DIFDet(Detector):
    """Deep Isolation Forest (Xu et al., IEEE TKDE 2023), re-implemented in NumPy after the
    authors' reference code (github.com/xuhongzuo/deep-iforest), independent-network variant:
    ``n_ensemble`` randomly initialised MLPs (d-500-100-20, tanh, N(0, 1) weights, no bias) map
    the min-max scaled input to representations, which are standardised and squashed with tanh;
    an isolation forest with ``n_estimators`` trees and ``max_samples`` = 256 is fitted on each,
    and the deviation-enhanced score 2^(-mean depth / c(psi)) x mean |z_f - threshold| along the
    path is averaged over the ensemble. Unlike the reference, both scalers are fitted once on the
    training data (the reference fits them on the data set being scored), so scoring is inductive."""

    def _fit(self, X, y):
        p = dict(n_ensemble=50, n_estimators=6, max_samples=256, hidden=(500, 100), rep_dim=20)
        p.update(self.params)
        self.p = p
        rng = np.random.default_rng(self.seed)
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler().fit(X)          # the reference min-max scales its inputs
        Z = self.scaler.transform(X)
        dims = [Z.shape[1], *p["hidden"], p["rep_dim"]]
        self.nets, self.rep_scalers, self.forests = [], [], []
        for i in range(p["n_ensemble"]):
            W = [rng.standard_normal((dims[j], dims[j + 1])) for j in range(len(dims) - 1)]
            R = self._forward(Z, W)
            sc = StandardScaler().fit(R)
            f = IsolationForest(n_estimators=p["n_estimators"], max_samples=min(p["max_samples"], len(Z)),
                                random_state=int(rng.integers(0, 2**31 - 1)), n_jobs=1)
            f.fit(np.tanh(sc.transform(R)))
            self.nets.append(W); self.rep_scalers.append(sc); self.forests.append(f)

    @staticmethod
    def _forward(Z, W):
        h = Z
        for j, w in enumerate(W):
            h = h @ w
            if j < len(W) - 1:
                h = np.tanh(h)
        return h

    @staticmethod
    def _deaas(forest, R):
        from sklearn.ensemble._iforest import _average_path_length
        n = len(R)
        depth_sum = np.zeros(n)
        devs = np.zeros((n, len(forest.estimators_)))
        for ii, est in enumerate(forest.estimators_):
            tree = est.tree_
            if tree.node_count == 1:
                continue
            leaves = est.apply(R)
            ind = est.decision_path(R).tocsr()
            plen = np.diff(ind.indptr)
            depth_sum += plen + _average_path_length(tree.n_node_samples[leaves]) - 1.0
            rows = np.repeat(np.arange(n), plen)
            nodes = ind.indices
            feat = tree.feature[nodes]
            internal = feat >= 0
            dev = np.abs(R[rows[internal], feat[internal]] - tree.threshold[nodes[internal]])
            nz = dev != 0
            s = np.bincount(rows[internal][nz], weights=dev[nz], minlength=n)
            c = np.bincount(rows[internal][nz], minlength=n)
            devs[:, ii] = s / (c + 1e-6)
        c_psi = _average_path_length([forest.max_samples_])[0]
        return 2.0 ** (-depth_sum / (len(forest.estimators_) * c_psi)) * devs.mean(axis=1)

    def _score(self, X, batch: int = 50_000):
        Z = self.scaler.transform(X)
        out = np.zeros(len(Z))
        for lo in range(0, len(Z), batch):           # batches bound the memory of the 500-unit layer
            Zb = Z[lo:lo + batch]
            for W, sc, f in zip(self.nets, self.rep_scalers, self.forests):
                out[lo:lo + batch] += self._deaas(f, np.tanh(sc.transform(self._forward(Zb, W))))
        return out / len(self.nets)


class HGBSupervised(Detector):
    """Supervised reference (labels of all classes) - an upper bound, not a competitor."""

    def _fit(self, X, y):
        self.model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, class_weight="balanced",
            random_state=self.seed).fit(X, y)

    def _score(self, X):
        return self.model.predict_proba(X)[:, 1]


UNSUPERVISED = {
    "IF": IForest,
    "ECOD": ECODDet,
    "HBOS": HBOSDet,
    "LOF": LOFDet,
    "OCSVM": OCSVMDet,
    "AE": AEDet,
    "DIF": DIFDet,
}


def make(name: str, seed: int, **params) -> Detector:
    if name == "HGB":
        return HGBSupervised(name="HGB", seed=seed, params=params)
    return UNSUPERVISED[name](name=name, seed=seed, params=params)
