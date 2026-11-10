"""Central configuration for every AnomalyGuard experiment.

All experiment scripts import their constants from here so that a single file
documents every choice that can change a reported number.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Raw X-IoMT v1.1 CSVs ("OriginalDatasetUpdated" folder of the authors' repository).
RAW_DIR = Path(os.environ.get(
    "XIOMT_DIR", REPO_ROOT / "data" / "raw" / "X-IoMTDataset" / "OriginalDatasetUpdated"))
PROCESSED_DIR = Path(os.environ.get("AG_PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
RESULTS_DIR = Path(os.environ.get("AG_RESULTS_DIR", REPO_ROOT / "results"))

# Upstream dataset pin (https://github.com/RuiPintoUBI/X-IoMTDataset).
XIOMT_REPO = "https://github.com/RuiPintoUBI/X-IoMTDataset"
XIOMT_COMMIT = "2e25990dc414b4666585d9a91a945379e86dc0fc"

# Chronological split of every capture: first 60 % train, next 20 % validation, last 20 % test.
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)

# Seeds used for every stochastic component (sub-sampling and model randomness).
SEEDS = (0, 1, 2, 3, 4)

# False-positive budget used to calibrate every detector on benign validation traffic only.
FPR_BUDGET = 0.01

# Sub-sampling caps (compute budget: 2 CPU cores, 7 GB RAM).
TRAIN_CAP = 100_000          # benign training pool per seed
TEST_BENIGN_CAP = 60_000     # benign test packets per seed
TEST_PER_TYPE_CAP = 4_000    # test packets per anomaly type per seed
VAL_CAP = 60_000             # benign validation packets per seed

# Causal per-source context windows.
CTX_WINDOWS_S = (1.0, 10.0)
CTX_LAST_K = 64

# Token strings that encode "field absent" in the raw CSVs. The raw files use both
# "0" and "0.0" (and empty strings); which spelling appears depends on the capture,
# which makes the raw tokens a capture fingerprint (see experiments/e1_shortcut_audit.py).
ABSENT_TOKENS = ("0", "0.0", "nan", "NaN", "", "None")

LABEL_COL = "anomaly.label"
NAME_COL = "anomaly.name"
LAYER_COL = "anomaly.layer"
TIME_COL = "frame.time"

LAYER_NAMES = {0: "Normal", 1: "Perception", 2: "Network", 3: "Application"}

TELEMETRY_COLS = ("Temperature", "Memory", "In_Temperature", "CPU", "RSSI",
                  "HR", "PULSE", "RESP", "SpO2")

# Columns that identify a host, a session or a capture rather than behaviour.
IDENTITY_COLS = ("frame.time", "ip.src_host", "ip.dst_host", "MAC_extracted",
                 "mqtt.clientid", "http.host", "http.user_agent", "tcp.seq", "tcp.ack")
