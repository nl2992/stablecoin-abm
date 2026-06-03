#!/usr/bin/env python
"""Run the full per-hub counterfactual sweep and produce the causal hub ranking.

Usage:
    python scripts/run_counterfactuals.py [--n-seeds 40] [--scenario ust_style_bank_run]

Outputs to experiments/results/counterfactual/:
    causal_hub_ranking.csv   — nodes ranked by Δcontagion with SEs
    causal_hub_ranking.json  — machine-readable for joint analysis

This should be run AFTER calibration passes (make calibrate).
~40 seeds × ~8 hubs × 2 (baseline + intervened) × 150 steps ≈ 10-30 minutes.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stablesim.counterfactual.hub_loader import load_hub_rankings
from stablesim.counterfactual.runner import run_all_hubs, N_SEEDS_DEFAULT
from stablesim.counterfactual.ranking import causal_hub_ranking, save_ranking, ranking_summary_table
from stablesim.scenarios.loader import load_stressbench_scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--scenario", default=None, help="Scenario name filter")
    parser.add_argument("--event", default=None, help="Filter hub list by event_id from repo 1")
    parser.add_argument("--output-dir", default="experiments/results/counterfactual")
    parser.add_argument("--max-hubs", type=int, default=None, help="Limit hub count for testing")
    args = parser.parse_args()

    # Load hub list from repo 1
    hubs = load_hub_rankings(event_id=args.event)
    if args.max_hubs:
        hubs = hubs[:args.max_hubs]
    print(f"Loaded {len(hubs)} hubs. Running counterfactuals with n_seeds={args.n_seeds}...")

    # Pick scenario
    scenarios = load_stressbench_scenarios()
    if args.scenario:
        scenario = next((s for s in scenarios if s.name == args.scenario), scenarios[0])
    else:
        # Default: most severe synthetic scenario
        scenario = next(
            (s for s in scenarios if s.name not in ("no_shock_baseline", "baseline")),
            scenarios[0],
        )
    print(f"Scenario: {scenario.name} ({len(scenario)} events)")

    # Run all counterfactuals
    results = run_all_hubs(hubs, scenario, n_seeds=args.n_seeds, n_steps=args.n_steps, verbose=True)

    # Rank and save
    df = causal_hub_ranking(results)
    save_ranking(df, args.output_dir)

    print("\n## Causal Hub Ranking\n")
    print(ranking_summary_table(df))

    sig = df[df["significant_p05"]].shape[0]
    print(f"\n{sig}/{len(df)} hubs show statistically significant contagion reduction (p < 0.05, one-sided)")


if __name__ == "__main__":
    main()
