# Reproduce every table and figure of the AnomalyGuard study.
#   make all        -> data, tests, all experiments, tables, figures (about 1 h on 2 CPU cores)
#   make quick      -> tests + one-seed smoke run of the main benchmark (~5 min)
PY ?= python3
export PYTHONPATH := $(CURDIR)
export PYTHONHASHSEED := 0

.PHONY: all data processed test e1 e2 e3 e4 e5 e6 e7 e8 diag stats tables figures quick clean

all: data processed test e1 e2 e3 e4 e5 e6 e7 e8 diag stats tables figures

data:
	bash scripts/fetch_data.sh

processed:
	$(PY) -m anomalyguard.data

test:
	$(PY) -m pytest -q tests

e1:  ; $(PY) -m anomalyguard.experiments.e1_shortcut_audit
e2:  ; $(PY) -m anomalyguard.experiments.e2_benchmark
e3:  ; $(PY) -m anomalyguard.experiments.e3_ablation
e4:  ; $(PY) -m anomalyguard.experiments.e4_psi_sweep
e5:  ; $(PY) -m anomalyguard.experiments.e5_contamination
e6:  ; $(PY) -m anomalyguard.experiments.e6_cost
e7:  ; $(PY) -m anomalyguard.experiments.e7_triage
e8:  ; $(PY) -m anomalyguard.experiments.e8_view_selection
diag: ; $(PY) -m anomalyguard.analysis.ecod_tail

stats:   ; $(PY) -m anomalyguard.analysis.stats
tables:  ; $(PY) -m anomalyguard.analysis.tables
figures: ; $(PY) -m anomalyguard.analysis.figures

quick: test
	$(PY) -m anomalyguard.experiments.e2_benchmark --seeds 0 --detectors IF-default AnomalyGuard-IF

clean:
	rm -rf data/processed results/*.csv results/*.json results/figures
