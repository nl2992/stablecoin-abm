.PHONY: install test lint fmt calibrate counterfactuals joint-analysis train sweep clean

install:
	pip install -e ".[dev]" || pip install -e . && pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

# Run SMM calibration against empirical moments from stablecoin-contagion-network
calibrate:
	python scripts/run_calibration.py --n-seeds 20 --maxiter 80

# Run per-hub counterfactuals (40 seeds × all hubs, ~30 min)
counterfactuals:
	python scripts/run_counterfactuals.py --n-seeds 40 --n-steps 150

# Joint analysis: predicted vs. causal hub ranking + headline figure
joint-analysis:
	python scripts/run_joint_analysis.py

# Train RL policies (PPO) for arbitrageur / redeemer agents
train:
	python -m stablesim.rl.ppo --config configs/base.yaml

# Sweep interventions x StressBench scenarios
sweep:
	python -m stablesim.experiments.sweep --config configs/interventions.yaml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
