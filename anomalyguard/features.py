"""Feature views used by AnomalyGuard.

Four views are built from the processed table:

* ``net``  - stateless per-packet behaviour (header flags, sizes, protocol indicators,
             service-port novelty). Host identity and capture metadata are excluded.
* ``ctx``  - causal per-source temporal context (rates, flag mixes, port fan-out) computed
             only from packets that precede or coincide with the scored packet.
* ``tel``  - device and patient telemetry, scored only on packets that carry telemetry.
* ``naive`` / ``artefact`` - deliberately leaky views used only by the shortcut audit.
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from . import config as C
from .data import TEXT_COLS, TOK_NAN, TOK_ZERO_DOT

NET_NUMERIC = [
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset", "tcp.flags.push",
    "mqtt.msgtype", "mqtt.topic_len", "mqtt.qos", "mqtt.retain", "mqtt.dupflag",
    "mqtt.conflag.cleansess", "http_is_post", "http_is_get", "http_has_response",
    "mqtt_has_payload", "mqtt_has_connect", "mqtt_has_conack", "proto_has_mqtt",
    "proto_has_http", "proto_has_data", "proto_depth",
]
NET_LOG = ["frame.len", "tcp.len", "mqtt.len", "http.content_length"]
FINGERPRINT = ["ip.ttl", "tcp.window_size"]   # host/OS fingerprints, ablated separately


class PortNovelty:
    """Service-port features learned from benign training traffic only.

    This is the learned counterpart of rule R4 of the original design ("first appearance of
    a destination port"): a port counts as *known* if it served at least ``min_count``
    benign training packets.
    """

    def __init__(self, min_count: int = 50):
        self.min_count = min_count
        self.known_: set[int] = set()

    @staticmethod
    def server_port(df: pd.DataFrame) -> np.ndarray:
        sp = df["tcp.srcport"].fillna(0).to_numpy(np.int64)
        dp = df["tcp.dstport"].fillna(0).to_numpy(np.int64)
        return np.minimum(sp, dp)

    def fit(self, df_benign_train: pd.DataFrame) -> "PortNovelty":
        vc = pd.Series(self.server_port(df_benign_train)).value_counts()
        self.known_ = set(int(p) for p in vc[vc >= self.min_count].index)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.server_port(df)
        return pd.DataFrame({
            "port_known": np.isin(p, list(self.known_)).astype(np.float32),
            "port_wellknown": (p < 1024).astype(np.float32),
            "port_registered": ((p >= 1024) & (p < 49152)).astype(np.float32),
        }, index=df.index)


def net_view(df: pd.DataFrame, ports: PortNovelty, fingerprint: bool = True) -> pd.DataFrame:
    X = df[NET_NUMERIC].astype(np.float32).fillna(0.0).copy()
    for c in NET_LOG:
        X[c] = np.log1p(df[c].clip(lower=0).fillna(0.0).astype(np.float32))
    if fingerprint:
        X["ip.ttl"] = df["ip.ttl"].fillna(0.0).astype(np.float32)
        X["tcp.window_size"] = np.log1p(df["tcp.window_size"].clip(lower=0).fillna(0.0)).astype(np.float32)
    X = pd.concat([X, ports.transform(df)], axis=1)
    return X


# ----------------------------------------------------------------------------- context
@numba.njit(cache=True)
def _distinct_last_k(keys: np.ndarray, group_start: np.ndarray, k: int) -> np.ndarray:
    """Number of distinct values among the last ``k`` packets of the same stream."""
    n = keys.shape[0]
    out = np.empty(n, dtype=np.float32)
    buf = np.empty(k, dtype=np.int64)
    for i in range(n):
        lo = max(group_start[i], i - k + 1)
        m = i - lo + 1
        for j in range(m):
            buf[j] = keys[lo + j]
        s = np.sort(buf[:m])
        d = 1
        for j in range(1, m):
            if s[j] != s[j - 1]:
                d += 1
        out[i] = d
    return out


def _stream_index(df: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Order rows by (keys, time); return the permutation and per-row stream start index."""
    order = np.lexsort([df["t"].to_numpy()] + [df[k].cat.codes.to_numpy() if hasattr(df[k], "cat")
                                                  else df[k].to_numpy() for k in reversed(keys)])
    sk = np.zeros(len(df), dtype=np.int64)
    for k in keys:
        codes = (df[k].cat.codes.to_numpy() if hasattr(df[k], "cat") else df[k].to_numpy())
        sk = sk * (int(codes.max()) + 2) + (codes.astype(np.int64) + 1)
    sk_sorted = sk[order]
    new = np.ones(len(df), dtype=bool)
    new[1:] = sk_sorted[1:] != sk_sorted[:-1]
    start_pos = np.maximum.accumulate(np.where(new, np.arange(len(df)), 0))
    return order, start_pos


def context_view(df: pd.DataFrame) -> pd.DataFrame:
    """Causal per-source context features (computed once on the whole processed table).

    Streams are keyed by (capture_group, src) so packets from different captures never mix.
    Every statistic uses only packets with timestamp <= the current packet's timestamp.
    """
    order, start = _stream_index(df, ["capture_group", "src"])
    t = df["t"].to_numpy()[order]
    out = {}
    dt = np.diff(t, prepend=np.nan)
    first = np.r_[True, start[1:] != start[:-1]]
    dt[first] = 3600.0
    out["ctx_log_dt_prev"] = np.log1p(np.clip(dt, 0, 3600.0))
    syn = df["tcp.flags.syn"].fillna(0).to_numpy(np.float64)[order]
    fin = df["tcp.flags.fin"].fillna(0).to_numpy(np.float64)[order]
    rst = df["tcp.flags.reset"].fillna(0).to_numpy(np.float64)[order]
    psh = df["tcp.flags.push"].fillna(0).to_numpy(np.float64)[order]
    flen = df["frame.len"].fillna(0).to_numpy(np.float64)[order]
    idx = np.arange(len(t))
    cs = {name: np.r_[0.0, np.cumsum(v)] for name, v in
          [("syn", syn), ("fin", fin), ("rst", rst), ("psh", psh), ("len", flen)]}
    for w in C.CTX_WINDOWS_S:
        lo = _window_counts_streamwise(t, start, w)
        n = idx - lo + 1
        out[f"ctx_log_n_{int(w)}s"] = np.log1p(n)
        if w == max(C.CTX_WINDOWS_S):
            for name in ("syn", "fin", "rst", "psh"):
                out[f"ctx_frac_{name}_{int(w)}s"] = (cs[name][idx + 1] - cs[name][lo]) / n
            out[f"ctx_log_meanlen_{int(w)}s"] = np.log1p((cs["len"][idx + 1] - cs["len"][lo]) / n)
    dport = df["tcp.dstport"].fillna(-1).to_numpy(np.int64)[order]
    sport = df["tcp.srcport"].fillna(-1).to_numpy(np.int64)[order]
    out[f"ctx_distinct_dport_last{C.CTX_LAST_K}"] = _distinct_last_k(dport, start, C.CTX_LAST_K)
    out[f"ctx_distinct_sport_last{C.CTX_LAST_K}"] = _distinct_last_k(sport, start, C.CTX_LAST_K)

    # Fan-in towards the destination (all sources) in the last second.
    order_d, start_d = _stream_index(df, ["capture_group", "dst"])
    td = df["t"].to_numpy()[order_d]
    lo_d = _window_counts_streamwise(td, start_d, 1.0)
    fan = np.empty(len(td))
    fan[order_d] = np.log1p(np.arange(len(td)) - lo_d + 1)

    res = pd.DataFrame(index=df.index)
    for k, v in out.items():
        col = np.empty(len(t), dtype=np.float32)
        col[order] = v
        res[k] = col
    res["ctx_log_dst_fanin_1s"] = fan.astype(np.float32)
    return res


def _window_counts_streamwise(t: np.ndarray, start: np.ndarray, w: float) -> np.ndarray:
    """Index of the first packet inside (t - w, t] for every packet, computed per stream."""
    n = len(t)
    lo = np.empty(n, dtype=np.int64)
    bounds = np.flatnonzero(np.r_[True, start[1:] != start[:-1]])
    bounds = np.r_[bounds, n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = t[a:b]
        lo[a:b] = a + np.searchsorted(seg, seg - w, side="right")
    return lo


# --------------------------------------------------------------------------- telemetry
class TelemetryView:
    """Telemetry fields with benign-median imputation of fields a packet does not carry.

    Packets carrying no telemetry at all are not scored by this view (``mask``). Imputing
    absent fields with the benign median makes *absence* uninformative, so the view can
    only react to the values that are actually reported.
    """

    def __init__(self, cols=C.TELEMETRY_COLS):
        self.cols = list(cols)
        self.median_: pd.Series | None = None

    def fit(self, df_benign_train: pd.DataFrame) -> "TelemetryView":
        self.median_ = df_benign_train[self.cols].median()
        return self

    @staticmethod
    def mask(df: pd.DataFrame, cols=C.TELEMETRY_COLS) -> np.ndarray:
        return df[list(cols)].notna().any(axis=1).to_numpy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.cols].astype(np.float32).fillna(self.median_.astype(np.float32))


# ------------------------------------------------------------------------ leaky views
def naive_view(df: pd.DataFrame) -> pd.DataFrame:
    """A typical "use every column" pipeline: label-encode text, fill missing with 0.

    Timestamps and IP addresses are dropped, as most published pipelines do; everything else
    (including raw absent-value spellings, MACs and client IDs) is kept.
    """
    from .data import NUMERIC_COLS
    cols = {c: df[c].astype(np.float32).fillna(0.0) for c in NUMERIC_COLS}
    for c in TEXT_COLS:
        if c in ("ip.src_host", "ip.dst_host"):
            continue
        cols[f"raw__{c}"] = df[f"raw__{c}"].astype(np.float32)
    return pd.DataFrame(cols, index=df.index)


def artefact_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Feature sets that carry *no* traffic behaviour, used to probe for shortcuts."""
    spelling = {}
    for c in TEXT_COLS:
        tok = df[f"tok__{c}"].to_numpy()
        spelling[f"zero_dot__{c}"] = (tok == TOK_ZERO_DOT).astype(np.float32)
        spelling[f"nan_tok__{c}"] = (tok == TOK_NAN).astype(np.float32)
    for c in C.TELEMETRY_COLS:
        spelling[f"present__{c}"] = df[c].notna().to_numpy().astype(np.float32)
    identity = {
        "src": df["src"].cat.codes.astype(np.float32),
        "dst": df["dst"].cat.codes.astype(np.float32),
        "mac": df["raw__MAC_extracted"].astype(np.float32),
        "clientid": df["raw__mqtt.clientid"].astype(np.float32),
        "ip.ttl": df["ip.ttl"].fillna(0).astype(np.float32),
        "tcp.window_size": df["tcp.window_size"].fillna(0).astype(np.float32),
    }
    return {
        "absence_spelling": pd.DataFrame(spelling, index=df.index),
        "host_identity": pd.DataFrame(identity, index=df.index),
    }
