# Appendix: Reproducibility

## A.1 Repository manifest

| Repo | Commit hash | Purpose |
|---|---|---|
| `nl2992/stablecoin-contagion-network` | [TBD] | Predicted hub rankings, empirical moments |
| `nl2992/stablecoin-abm` | [TBD] | ABM simulator, calibration, counterfactuals |

Both repos are public on GitHub.

## A.2 Environment

```bash
# Python 3.11+ required
cd stablecoin-abm
pip install -r requirements.txt
pip install -e .
```

Key dependency versions:
- numpy >= 1.26
- scipy >= 1.12
- stable-baselines3 >= 2.3
- gymnasium >= 0.29

## A.3 Exact commands to reproduce end-to-end

```bash
# Step 1: Verify Phase 0 gate (engine sanity)
cd stablecoin-abm
pytest tests/test_phase0_gate.py -v

# Step 2: Run calibration
python scripts/run_calibration.py \
    --n-seeds 20 \
    --maxiter 80 \
    --event usdc_svb_2023
# Output: experiments/results/calibration/calibration_report.{md,json,csv}
# GATE: must pass before proceeding

# Step 3: Run counterfactuals (40 seeds × all hubs × 150 steps ≈ 20-40 min)
python scripts/run_counterfactuals.py \
    --n-seeds 40 \
    --n-steps 150 \
    --event usdc_svb_2023
# Output: experiments/results/counterfactual/causal_hub_ranking.{csv,json}

# Step 4: Joint analysis + headline figure
python scripts/run_joint_analysis.py
# Output: experiments/results/joint_analysis/
#         paper/figures/fig_headline_scatter.{pdf,png}

# Step 5: Intervention sweep
make sweep
# Output: experiments/results/sweep_results.csv

# Step 6: Run regression CI gate
pytest tests/test_calibration_regression.py -v
```

## A.4 Seeds and randomness

All experiments use `np.random.default_rng(seed)` with explicit seed passed from
the outer loop.  The seed range is `[base_seed, base_seed + n_seeds)` where
`base_seed=0` by default.  Results are reproducible given the same seed range
and software versions.

## A.5 Hub ranking schema version

The hub ranking from repo 1 uses `table_node_centrality.csv` with schema:

```
node_id, out_degree_w, in_degree_w, eigenvector, betweenness, role, event_id
```

The composite predicted_importance is:
```
0.6 × eigenvector + 0.3 × out_degree_w_norm + 0.1 × betweenness_norm
```

If the repo-1 schema changes, re-export via `stablesim.counterfactual.hub_loader.load_hub_rankings()`.

## A.6 Data manifest hashes

[To be filled after final data freeze]

| File | SHA-256 | Description |
|---|---|---|
| `configs/calibration_targets.json` | [hash] | Locked empirical targets |
| `results/tables/table_node_centrality.csv` (repo 1) | [hash] | Predicted hub rankings |
| `experiments/results/counterfactual/causal_hub_ranking.csv` | [hash] | Causal rankings |
| `experiments/results/joint_analysis/agreement_metrics.json` | [hash] | Agreement metrics |
