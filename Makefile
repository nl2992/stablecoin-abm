.PHONY: install test lint fmt calibrate counterfactuals joint-analysis train sweep e2e clean

# Calibration sentinel — downstream targets depend on this file existing.
# If calibration hasn't been run, `make counterfactuals` will first run calibration.
CALIBRATION_SENTINEL := experiments/results/calibration/calibration_report.json

install:
	pip install -e ".[dev]" || pip install -e . && pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

# ── Critical path (order matters) ─────────────────────────────────────────────

# Step 1: SMM calibration against empirical moments from stablecoin-contagion-network
# GATE: must produce ≥ 3/4 passing moments before proceeding
calibrate $(CALIBRATION_SENTINEL):
	python scripts/run_calibration.py --n-seeds 20 --maxiter 80

# Step 2: Per-hub counterfactuals (40 seeds × all hubs, ~30 min)
# Hard-blocked by calibration sentinel — won't run on an uncalibrated sim
counterfactuals: $(CALIBRATION_SENTINEL)
	python scripts/run_counterfactuals.py --n-seeds 40 --n-steps 150

# Step 3: Joint analysis: predicted vs. causal hub ranking + headline figure
joint-analysis: experiments/results/counterfactual/causal_hub_ranking.csv
	python scripts/run_joint_analysis.py

# ── Secondary ──────────────────────────────────────────────────────────────────

# RL training
train:
	python -m stablesim.rl.ppo --config configs/base.yaml

# Intervention × scenario sweep
sweep: $(CALIBRATION_SENTINEL)
	python -m stablesim.experiments.sweep --config configs/interventions.yaml

# ── CI ─────────────────────────────────────────────────────────────────────────

# End-to-end smoke test (2 seeds, 2 hubs, toy episode — fast, runs on every commit)
e2e:
	pytest tests/test_e2e_smoke.py -v

# Full suite
test:
	pytest tests/ -v

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
