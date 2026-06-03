"""Causal hub ranking from counterfactual results.

Ranks nodes by delta_c (estimated causal effect of intervention) and exports
a DataFrame suitable for the joint analysis in analysis/comparison.py.

Uses PairedResult from inference.py (replaces the old CounterfactualResult).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .inference import PairedResult


def causal_hub_ranking(results: list[PairedResult]) -> pd.DataFrame:
    """Convert FDR-corrected PairedResults to a ranked DataFrame.

    Returns
    -------
    pd.DataFrame with columns: causal_rank + all fields from PairedResult.to_row().
    Sorted by delta_c descending (largest causal reduction first).
    """
    rows = [r.to_row() for r in results]
    df = pd.DataFrame(rows)

    # Rename delta_c → delta_contagion for downstream compatibility
    if "delta_c" in df.columns:
        df = df.rename(columns={
            "delta_c": "delta_contagion",
            "se_paired": "se",
        })

    df = df.sort_values("delta_contagion", ascending=False).reset_index(drop=True)
    df.insert(0, "causal_rank", range(1, len(df) + 1))
    return df


def save_ranking(
    df: pd.DataFrame,
    output_dir: str | Path = "experiments/results/counterfactual",
    stamp_kwargs: dict | None = None,
) -> None:
    """Save ranking CSV + JSON and optionally write a provenance stamp."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "causal_hub_ranking.csv"
    df.to_csv(csv_path, index=False)

    json_path = out / "causal_hub_ranking.json"
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    if stamp_kwargs is not None:
        from ..utils.stamp import stamp_artifact
        stamp_artifact(csv_path, **stamp_kwargs)

    print(f"Saved causal hub ranking to {out}/")


def ranking_summary_table(df: pd.DataFrame) -> str:
    """Markdown table of the causal ranking with inference columns."""
    want = [
        "causal_rank", "node_id", "delta_contagion", "se",
        "t_stat", "q_value", "significant_fdr", "underpowered",
        "pair_corr", "n_pairs",
    ]
    cols = [c for c in want if c in df.columns]
    sub = df[cols].copy()
    for col in ("delta_contagion", "se", "t_stat", "q_value", "pair_corr"):
        if col in sub.columns:
            sub[col] = sub[col].map("{:.4f}".format)
    return sub.to_markdown(index=False)
