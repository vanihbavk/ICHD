PYTHON_BIN := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTHON = PYTHONPATH=src $(PYTHON_BIN)

.PHONY: fetch-data extract episodes evaluate experiments external plots test all pipeline clean

all: evaluate experiments plots test

pipeline: extract episodes evaluate experiments external plots test

fetch-data:
	$(PYTHON) scripts/download_data.py

extract:
	$(PYTHON) -m ichd.extract --workers 4

episodes:
	$(PYTHON) -m ichd.episodes

evaluate:
	$(PYTHON) -m ichd.evaluate

experiments:
	$(PYTHON) experiments/exp1_lvef_anchor.py
	$(PYTHON) experiments/exp2_trending_concordance.py
	$(PYTHON) experiments/exp3_svr_independence.py
	$(PYTHON) experiments/exp4_noise_ceiling.py
	$(PYTHON) experiments/exp5_gls_reconciliation.py
	$(PYTHON) experiments/exp6_locked_holdout.py
	$(PYTHON) experiments/exp7_temporal_and_vitaldb_audit.py
	$(PYTHON) experiments/exp8_identity_isolation.py

external:
	$(PYTHON) experiments/exp9_vitaldb_validation.py
	$(PYTHON) experiments/exp10_vitaldb_shift_grid.py

plots:
	$(PYTHON) -m ichd.plots

test:
	$(PYTHON) -m pytest -v tests/

clean:
	rm -rf __pycache__ .pytest_cache src/ichd/__pycache__ experiments/__pycache__ scripts/__pycache__ tests/__pycache__
