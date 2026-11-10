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
| Experiments | `anomalyguard/experiments/e1…e8` | Shortcut audit, benchmark, ablation, ψ sweep, contamination, cost (incl. peak packet rate), triage, validation-selected views |
| Analysis | `anomalyguard/analysis/` | Statistics (Friedman, Holm-corrected Wilcoxon), tables, figures, ECOD threshold diagnostic |
| Tests | `tests/test_core.py` | 20 unit tests (calibration under ties, causality of context features, propagation, fusion, detector sanity) |

## Reproduce

Requirements: Python 3.13, `git`, ~1 GB free disk, 2+ CPU cores, 8 GB RAM.

```bash
pip install -r requirements.txt
make data        # sparse-clones X-IoMT v1.1 at commit 2e25990 and verifies 109 SHA-256 sums
make processed   # builds data/processed/xiomt.parquet (+ cleaning report)
make test        # unit tests
make all         # every experiment, statistics, tables and figures (about 1 h on 2 cores)
```

Single steps: `make e1` … `make e8`, `make diag` (ECOD threshold diagnostic), `make stats`,
`make tables`, `make figures`. Run E2 before `make figures` (the figures reuse its score dump).
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

## Headline results

All numbers below are produced by the code in this repository (`results/`), mean over five seeds
unless stated otherwise; thresholds are calibrated on benign validation traffic at a 1 % budget.

* **Shortcut audit (E1).** A classifier that sees only how absent values are spelled and which
  telemetry fields are present separates normal from anomalous test packets with AUROC
  0.989; the conventional all-columns pipeline with a random split reports 99.84 % accuracy.
* **Benchmark (E2)** on the full chronological test segments (315,938 packets):

| Detector (fusion + session association) | Macro recall | Macro AUROC | Test FPR (%) | Network | Perception | Application |
|---|---|---|---|---|---|---|
| AE | 0.890 ± 0.018 | 0.946 | 1.07 | 0.998 | 0.858 | 0.812 |
| OCSVM | 0.837 ± 0.011 | 0.952 | 1.01 | 0.978 | 0.772 | 0.745 |
| AnomalyGuard-IF | 0.785 ± 0.018 | 0.916 | 1.08 | 0.995 | 0.241 | 0.838 |
| LOF | 0.467 ± 0.031 | 0.893 | 1.54 | 0.114 | 0.864 | 0.599 |
| HBOS | 0.438 ± 0.012 | 0.838 | 1.06 | 0.259 | 0.250 | 0.672 |
| IF-default | 0.380 ± 0.051 | 0.890 | 1.15 | 0.175 | 0.038 | 0.702 |
| ECOD | 0.058 ± 0.007 | 0.920 | 0.00 | 0.152 | 0.007 | 0.000 |
| HGB (supervised) | 0.913 ± 0.006 | 0.991 | 1.06 | 1.000 | 0.709 | 0.926 |

* **Ablation (E3).** Plain IF with ψ = 256 reaches a macro recall of 0.074;
  the full AnomalyGuard-IF reaches 0.785. Reverting only the ψ rule gives
  0.380, removing causal context 0.450, removing fusion
  0.446.
* **Contamination (E5, three seeds).** With 10 % unlabelled anomalies in the training pool,
  AnomalyGuard-IF falls from 0.789 to 0.741 and OCSVM from
  0.833 to 0.513.

The autoencoder and the one-class SVM obtain a higher macro recall than AnomalyGuard-IF, mainly on
perception-layer (device) faults; no unsupervised detector separates the 'Clinical' anomalies.
`results/stats.json` holds the Friedman and Holm-corrected Wilcoxon tests.

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
