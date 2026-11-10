"""Helpers shared by the experiment scripts."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..metrics import detection_report, per_type_auc
from ..pipeline import propagate_alerts

LAYER_OF = {}  # anomaly name -> layer name, filled lazily from the data


def layer_map(df: pd.DataFrame) -> dict:
    m = df.groupby("name", observed=True)["layer"].agg(lambda s: int(s.mode().iloc[0]))
    return {str(k): C.LAYER_NAMES[int(v)] for k, v in m.items()}


def evaluate(test: pd.DataFrame, scores: np.ndarray, thr: float, layers: dict,
             extra_flags: np.ndarray | None = None) -> tuple[dict, dict]:
    """Return (summary row, per-type recall dict) for one scored test set."""
    y = test["label"].to_numpy()
    names = test["name"].astype(str).to_numpy()
    if extra_flags is not None:
        # fold extra (propagated) alerts into the score so that the report counts them
        scores = np.where(extra_flags, np.maximum(scores, thr), scores)
    rep = detection_report(y, names, scores, thr)
    aucs = per_type_auc(y, names, scores)
    row = {k: rep[k] for k in ("roc_auc", "pr_auc", "recall", "precision", "fpr", "f1",
                               "macro_recall", "tp", "fp", "fn", "tn")}
    row["macro_auc"] = float(np.mean(list(aucs.values())))
    by_layer: dict[str, list] = {}
    for n, r in rep["per_type_recall"].items():
        by_layer.setdefault(layers[n], []).append(r)
    for L, v in by_layer.items():
        row[f"recall_{L.lower()}"] = float(np.mean(v))
    per = {f"recall::{k}": v for k, v in rep["per_type_recall"].items()}
    per.update({f"auc::{k}": v for k, v in aucs.items()})
    return row, per


def environment() -> dict:
    import sklearn, scipy, numba
    try:
        import pyod
        pv = pyod.__version__
    except Exception:  # pragma: no cover
        pv = None
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__, "numba": numba.__version__, "pyod": pv,
        "xiomt_commit": C.XIOMT_COMMIT, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


__all__ = ["evaluate", "layer_map", "environment", "write_json", "log", "propagate_alerts"]
