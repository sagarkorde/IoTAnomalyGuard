"""Loading, cleaning and chronological splitting of X-IoMT v1.1.

Design rules (documented in the paper's methodology section):
* every row keeps its capture group (CSV file name without the ``_part_NNN`` suffix);
* rows without a ground-truth label are dropped and counted;
* absent-value tokens ("0", "0.0", "nan", ...) in text columns are canonicalised, and
  the raw spelling is kept separately (``tok__*``) only for the shortcut audit;
* each capture group is split chronologically (60/20/20) so that no model ever trains on
  packets that occur after the packets it is evaluated on.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

TEXT_COLS = (
    "ip.src_host", "ip.dst_host", "http.request.method", "http.request.full_uri",
    "http.request.version", "http.referer", "http.file_data", "http.response", "mqtt.msg",
    "mqtt.topic", "mqtt.protoname", "mqtt.conack.flags", "http.user_agent", "http.host",
    "mqtt.clientid", "frame.protocols", "MAC_extracted",
)

NUMERIC_COLS = (
    "tcp.srcport", "tcp.dstport", "tcp.seq", "tcp.ack", "tcp.flags.syn", "tcp.flags.ack",
    "tcp.flags.fin", "tcp.flags.reset", "tcp.flags.push", "tcp.flags.urg", "tcp.len",
    "http.request.uri.query", "http.content_length", "mqtt.msgtype", "mqtt.topic_len",
    "mqtt.conflags", "mqtt.conflag.cleansess", "mqtt.len", "mqtt.proto_len", "mqtt.ver",
    "mqtt.msg_decoded_as", "mqtt.hdrflags", "mqtt.qos", "mqtt.retain", "mqtt.dupflag",
    "ip.ttl", "tcp.window_size", "frame.len",
) + C.TELEMETRY_COLS

# Token style codes used by the shortcut audit.
TOK_PRESENT, TOK_ZERO, TOK_ZERO_DOT, TOK_NAN = 0, 1, 2, 3


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_raw_files(raw_dir: Path = C.RAW_DIR) -> list[Path]:
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files under {raw_dir}. Run scripts/fetch_data.sh first or set XIOMT_DIR.")
    return files


def capture_group(file_name: str) -> str:
    stem = file_name.replace("_v11.csv", "").replace(".csv", "")
    parts = stem.split("_part_")
    return parts[0]


def _token_style(s: pd.Series) -> np.ndarray:
    raw = s.astype("string")
    out = np.full(len(s), TOK_PRESENT, dtype=np.int8)
    out[(raw == "0").fillna(False).to_numpy()] = TOK_ZERO
    out[(raw == "0.0").fillna(False).to_numpy()] = TOK_ZERO_DOT
    out[raw.isna().to_numpy() | (raw.str.lower() == "nan").fillna(False).to_numpy()
        | (raw == "").fillna(False).to_numpy()] = TOK_NAN
    return out


def _canonical_text(s: pd.Series) -> pd.Series:
    raw = s.astype("string")
    return raw.mask(raw.isin(C.ABSENT_TOKENS) | raw.isna())


def load_raw(raw_dir: Path = C.RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Read every CSV, attach provenance columns and drop unlabeled rows."""
    frames, manifest = [], {}
    for f in list_raw_files(raw_dir):
        df = pd.read_csv(f, low_memory=False, dtype={c: "string" for c in TEXT_COLS})
        df["capture_file"] = f.name
        df["capture_group"] = capture_group(f.name)
        df["row_in_file"] = np.arange(len(df), dtype=np.int64)
        manifest[f.name] = {"rows": int(len(df)), "sha256": sha256_file(f)}
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    n_total = len(df)
    unlabeled = df[C.LABEL_COL].isna()
    report = {
        "rows_total": int(n_total),
        "rows_unlabeled_dropped": int(unlabeled.sum()),
        "unlabeled_by_file": {k: int(v) for k, v in
                              df.loc[unlabeled, "capture_file"].value_counts().items()},
        "files": manifest,
    }
    df = df.loc[~unlabeled].reset_index(drop=True)
    return df, report


def build_processed(raw_dir: Path = C.RAW_DIR, out_dir: Path = C.PROCESSED_DIR) -> Path:
    """Create ``xiomt.parquet`` with canonical, raw-token and split columns."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, report = load_raw(raw_dir)

    out = pd.DataFrame({
        "capture_file": df["capture_file"].astype("category"),
        "capture_group": df["capture_group"].astype("category"),
        "row_in_file": df["row_in_file"],
        "t": df[C.TIME_COL].astype("int64").to_numpy() / 1e6,     # seconds since epoch
        "label": df[C.LABEL_COL].astype("int8"),
        "name": df[C.NAME_COL].astype("category"),
        "layer": df[C.LAYER_COL].astype("int8"),
        "src": df["ip.src_host"].astype("category"),
        "dst": df["ip.dst_host"].astype("category"),
    })
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # Canonical text fields reduced to behaviour-level indicators.
    method = _canonical_text(df["http.request.method"])
    protos = df["frame.protocols"].astype("string").fillna("")
    out["http_is_post"] = (method == "POST").fillna(False).astype("int8")
    out["http_is_get"] = (method == "GET").fillna(False).astype("int8")
    out["http_has_response"] = _canonical_text(df["http.response"]).notna().astype("int8")
    out["mqtt_has_payload"] = _canonical_text(df["mqtt.msg"]).notna().astype("int8")
    out["mqtt_has_connect"] = _canonical_text(df["mqtt.protoname"]).notna().astype("int8")
    out["mqtt_has_conack"] = _canonical_text(df["mqtt.conack.flags"]).notna().astype("int8")
    out["proto_has_mqtt"] = protos.str.contains(":mqtt", regex=False).astype("int8")
    out["proto_has_http"] = protos.str.contains(":http", regex=False).astype("int8")
    out["proto_has_data"] = protos.str.contains(":data", regex=False).astype("int8")
    out["proto_depth"] = protos.str.count(":").astype("int8")

    # Raw spellings, kept only for the shortcut audit and the "naive" protocol.
    for c in TEXT_COLS:
        out[f"tok__{c}"] = _token_style(df[c])
        out[f"raw__{c}"] = df[c].astype("string").fillna("<NA>").astype("category").cat.codes.astype("int32")

    # Chronological 60/20/20 split inside every capture group.
    out = out.sort_values(["capture_group", "t", "capture_file", "row_in_file"], kind="mergesort")
    out = out.reset_index(drop=True)
    pos = out.groupby("capture_group", observed=True).cumcount().to_numpy()
    size = out.groupby("capture_group", observed=True)["t"].transform("size").to_numpy()
    frac = pos / size
    a, b, _ = C.SPLIT_FRACTIONS
    split = np.where(frac < a, 0, np.where(frac < a + b, 1, 2)).astype("int8")
    out["split"] = split  # 0 train, 1 validation, 2 test

    path = out_dir / "xiomt.parquet"
    out.to_parquet(path, index=False)

    report["rows_after_cleaning"] = int(len(out))
    report["split_counts"] = {
        name: {k: int(v) for k, v in grp.value_counts().sort_index().items()}
        for name, grp in out.groupby("name", observed=True)["split"]
    }
    report["xiomt_commit"] = C.XIOMT_COMMIT
    with open(out_dir / "xiomt_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return path


def load_processed(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else C.PROCESSED_DIR / "xiomt.parquet"
    if not path.exists():
        build_processed()
    return pd.read_parquet(path)


if __name__ == "__main__":  # pragma: no cover
    p = build_processed()
    print("wrote", p)
