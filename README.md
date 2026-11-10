# AnomalyGuard

**Shortcut-aware evaluation and context-aware triage of unsupervised anomaly detection
(Isolation Forest and baselines) on cross-layer Internet-of-Medical-Things traffic.**

This repository contains the complete, seeded code behind every number, table and figure of
the AnomalyGuard study. It downloads a public dataset at a pinned commit, verifies it by
SHA-256, and regenerates all results with one command.

## What the code does

| Step | Module | Purpose |
|---|---|---|
| Data | `anomalyguard/data.py` | Loads X-IoMT v1.1, drops unlabeled rows, canonicalises absent-value tokens, keeps raw spellings only for the audit, splits every capture chronologically 60/20/20 |
| Views | `anomalyguard/features.py` | Behaviour-only **network view**, causal per-source **context** features, **telemetry view** (scored only where telemetry exists), and deliberately leaky **naive/artefact** views for the audit |
| Detectors | `anomalyguard/detectors.py` | Isolation Forest, ECOD, HBOS, LOF, One-Class SVM (Nyström + SGD), MLP autoencoder, and a supervised HGB reference behind one `fit/score` interface |
| Calibration | `anomalyguard/calibration.py` | Benign-only thresholds, smoothed conformal p-values, Bonferroni cross-layer fusion with view attribution |
| Pipeline | `anomalyguard/pipeline.py` | Label-free Isolation-Forest subsample rule (ψ), cross-layer scoring, causal alert propagation with a benign-calibrated trigger |
| Experiments | `anomalyguard/experiments/e1…e8` | Shortcut audit, benchmark, ablation, ψ sweep, contamination, cost, triage, validation-selected views |
| Analysis | `anomalyguard/analysis/` | Statistics (Friedman, Holm-corrected Wilcoxon), tables and figures |
| Tests | `tests/test_core.py` | 20 unit tests (calibration under ties, causality of context features, propagation, fusion, detector sanity) |

## Reproduce

Requirements: Python 3.13, `git`, ~1 GB free disk, 2+ CPU cores, 8 GB RAM.

```bash
pip install -r requirements.txt
make data        # sparse-clones X-IoMT v1.1 at commit 2e25990 and verifies 109 SHA-256 sums
make processed   # builds data/processed/xiomt.parquet (+ cleaning report)
make test        # unit tests
make all         # every experiment, statistics, tables and figures (~1.5 h on 2 cores)
```

Single steps: `make e1` … `make e8`, `make stats`, `make tables`, `make figures`.
A five-minute smoke run: `make quick`.

Every stochastic component is seeded (`SEEDS = (0, 1, 2, 3, 4)` in `anomalyguard/config.py`);
all thresholds are calibrated on benign validation traffic at a 1 % false-positive budget.
`results/environment.json` records the exact library versions and platform used.

## Dataset

X-IoMT v1.1 (`OriginalDatasetUpdated`), R. P. O. Pinto, B. M. C. Silva, P. R. M. Inácio,
<https://github.com/RuiPintoUBI/X-IoMTDataset>, pinned to commit
`2e25990dc414b4666585d9a91a945379e86dc0fc`. Please cite the dataset authors:
*IEEE Internet of Things Journal*, doi:10.1109/JIOT.2026.3664373, and *Proc. NCA 2025*,
doi:10.1109/NCA67271.2025.00047. After removing 2,661 unlabeled rows the cleaned table has
1,579,704 packets, matching the size published by the dataset authors.

## Protocol in one paragraph

Each capture is split chronologically (60 % train, 20 % validation, 20 % test). Unsupervised
detectors see only benign training packets; thresholds come from benign validation packets
only; the full test segments (315,938 packets) are scored. Host identity (IPs, MACs, MQTT
client IDs, timestamps, sequence numbers) never enters a model; absent-value tokens are
canonicalised so that their capture-specific spelling cannot be exploited. Results are
reported as mean ± s.d. over five seeds, per anomaly type and macro-averaged.

## Layout

```
anomalyguard/            library code (config, data, features, detectors, calibration, pipeline)
anomalyguard/experiments e1_shortcut_audit … e8_view_selection
anomalyguard/analysis    stats.py, tables.py, figures.py
data/MANIFEST.sha256     checksums of the 109 raw CSV files
results/                 CSV/JSON outputs, results/tables/, results/figures/
scripts/                 fetch_data.sh, run_experiments.sh
tests/                   unit tests
```
