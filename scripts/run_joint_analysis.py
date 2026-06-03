#!/usr/bin/env python
"""Compute joint analysis: predicted hub ranking vs. causal hub ranking.

Usage:
    python scripts/run_joint_analysis.py

Reads:
    experiments/results/counterfactual/causal_hub_ranking.csv

Outputs to experiments/results/joint_analysis/:
    agreement_metrics.json
    divergence_case.json
    joint_analysis_report.md
    joint_analysis_table.csv
    paper/figures/fig_headline_scatter.pdf + .png
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from stablesim.analysis.comparison import compute_agreement, find_divergence_case, save_agreement_report
from stablesim.analysis.figures import plot_headline_scatter, save_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-csv", default="experiments/results/counterfactual/causal_hub_ranking.csv")
    parser.add_argument("--output-dir", default="experiments/results/joint_analysis")
    parser.add_argument("--figures-dir", default="paper/figures")
    args = parser.parse_args()

    rp = Path(args.ranking_csv)
    if not rp.exists():
        print(f"ERROR: {rp} not found. Run scripts/run_counterfactuals.py first.")
        sys.exit(1)

    df = pd.read_csv(rp)
    print(f"Loaded causal hub ranking: {len(df)} nodes")

    # Agreement metrics
    metrics = compute_agreement(df)
    divergence = find_divergence_case(df)

    print("\n" + metrics.interpretation())
    print(f"\nLargest divergence (spurious hub): {divergence['node_id']}")
    print(f"  Predicted importance: {divergence['predicted_importance']:.3f}")
    print(f"  Causal effect: {divergence['delta_contagion']:.4f}")
    print(f"  Explanation: {divergence['likely_explanation'][:200]}...")

    save_agreement_report(metrics, divergence, df, args.output_dir)

    # Headline figure
    fig = plot_headline_scatter(df)
    save_figure(fig, "fig_headline_scatter", args.figures_dir)
    print(f"\nHeadline figure saved to {args.figures_dir}/")


if __name__ == "__main__":
    main()
