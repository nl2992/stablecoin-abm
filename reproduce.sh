#!/usr/bin/env bash
# Reproduce every ABM result and figure end-to-end.
# Requires: the shared venv at ../.venv, AND the GNN repo's outputs (run its reproduce.sh first:
# this consumes ../stablecoin-contagion-gnn/exports/* and data/processed/graphs/*).
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

echo "[1/8] causal counterfactual join (lead-lag W, 4/4 calibration)"
$PY scripts/run_netcontagion_join.py

echo "[2/8] balance-sheet exposure join (documented W) + concordance"
$PY scripts/run_exposure_join.py

echo "[3/8] intervention / welfare sweep + policy comparison"
$PY scripts/run_intervention_sweep.py

echo "[4/8] multi-episode generalization"
$PY scripts/run_multi_episode_join.py

echo "[5/8] robustness (calibration uncertainty) + welfare matrix"
$PY scripts/run_robustness_welfare.py

echo "[6/8] placebo control + adaptive-redeemer + two-agent + predictive-causality"
$PY scripts/run_placebo_control.py
$PY scripts/run_adaptive_robustness.py
$PY scripts/run_two_agent_robustness.py
$PY scripts/run_predictive_causality.py

echo "[7/8] RL regulator (PPO)"
$PY scripts/run_rl_regulator.py

echo "[8/8] tests"
$PY -m pytest tests/ -q -p no:warnings

echo "Done. Results in experiments/results/netcontagion/."
