"""Causal hub ranking from counterfactual results.

Ranks nodes by delta_contagion (estimated causal effect of intervention).
Exports a DataFrame suitable for the joint analysis in analysis/comparison.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .runner import CounterfactualResult


def causal_hub_ranking(results: list[CounterfactualResult]) -> pd.DataFrame:
    """Convert counterfactual results to a ranked DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        causal_rank, node_id, node_type, role, predicted_importance,
        delta_contagion, se, t_stat, p_value_one_sided, significant_p05,
        baseline_mean, intervened_mean, intervention_type, n_seeds, scenario
    """
    rows = [r.summary_dict() for r in results]
    df = pd.DataFrame(rows)
    df = df.sort_values("delta_contagion", ascending=False).reset_index(drop=True)
    df.insert(0, "causal_rank", range(1, len(df) + 1))
    return df


def save_ranking(
    df: pd.DataFrame,
    output_dir: str | Path = "experiments/results/counterfactual",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "causal_hub_ranking.csv", index=False)
    # Also save as JSON for the joint analysis module
    records = df.to_dict(orient="records")
    with open(out / "causal_hub_ranking.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved causal hub ranking to {out}/")


def ranking_summary_table(df: pd.DataFrame) -> str:
    """Return a Markdown table of the causal ranking with SEs."""
    cols = [
        "causal_rank", "node_id", "role", "delta_contagion",
        "se", "t_stat", "significant_p05", "predicted_importance"
    ]
    sub = df[cols].copy()
    sub["delta_contagion"] = sub["delta_contagion"].map("{:.4f}".format)
    sub["se"] = sub["se"].map("{:.4f}".format)
    sub["t_stat"] = sub["t_stat"].map("{:.2f}".format)
    sub["predicted_importance"] = sub["predicted_importance"].map("{:.3f}".format)
    return sub.to_markdown(index=False)
