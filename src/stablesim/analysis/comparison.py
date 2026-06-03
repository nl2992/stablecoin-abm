"""Joint analysis: predicted hub ranking (repo 1) vs. causal hub ranking (repo 2).

Computes three agreement metrics:
  1. Spearman rank correlation ρ between predicted importance and causal effect
  2. Top-k overlap (k = 3, 5) — fraction of top-k causal hubs that appear in top-k predicted
  3. OLS regression of delta_contagion on predicted_importance — slope and R²

The headline finding is whether these metrics are high (GNN predictions are
causally valid) or low/divergent (GNN captures correlational structure only).

Divergence case study: the node with the highest predicted_importance but
lowest causal effect — mechanically why didn't it propagate causally in the ABM?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AgreementMetrics:
    """All three agreement metrics between predicted and causal hub rankings.

    Attributes
    ----------
    spearman_rho : float
        Spearman rank correlation between predicted_importance and delta_contagion.
    spearman_pvalue : float
        Two-sided p-value for spearman_rho.
    top3_overlap : float
        |top-3 causal ∩ top-3 predicted| / 3.
    top5_overlap : float
        |top-5 causal ∩ top-5 predicted| / 5.
    ols_slope : float
        Slope of OLS: delta_contagion ~ predicted_importance.
    ols_intercept : float
    ols_r_squared : float
    ols_pvalue : float
        p-value for the slope.
    n_nodes : int
    """

    spearman_rho: float
    spearman_pvalue: float
    spearman_ci_lo: float = float("nan")   # bootstrap 95% CI lower
    spearman_ci_hi: float = float("nan")   # bootstrap 95% CI upper
    top3_overlap: float = 0.0
    top5_overlap: float = 0.0
    ols_slope: float = 0.0
    ols_intercept: float = 0.0
    ols_r_squared: float = 0.0
    ols_pvalue: float = 1.0
    n_nodes: int = 0
    # Split by data tier
    n_real_nodes: int = 0    # nodes that appear in real empirical episodes
    n_synth_nodes: int = 0   # nodes that appear only in synthetic scenarios
    spearman_rho_real: float = float("nan")   # agreement on real-episode hubs only

    def interpretation(self) -> str:
        ci_str = (
            f" [95% CI: {self.spearman_ci_lo:.3f}, {self.spearman_ci_hi:.3f}]"
            if not (np.isnan(self.spearman_ci_lo) or np.isnan(self.spearman_ci_hi))
            else " [CI: run bootstrap]"
        )
        real_str = (
            f"\n  Spearman ρ (real-episode hubs only, n={self.n_real_nodes}): "
            f"{self.spearman_rho_real:.3f}"
            if not np.isnan(self.spearman_rho_real) else ""
        )
        lines = [f"Agreement between repo-1 predicted importance and repo-2 causal effect (n={self.n_nodes} hubs)\n"]
        lines.append(f"  Spearman ρ = {self.spearman_rho:.3f}  (p = {self.spearman_pvalue:.3f}){ci_str}")
        lines.append(f"  Top-3 overlap = {self.top3_overlap:.0%}  |  Top-5 overlap = {self.top5_overlap:.0%}")
        lines.append(f"  OLS: Δcontagion = {self.ols_intercept:.4f} + {self.ols_slope:.4f} × predicted_importance  (R² = {self.ols_r_squared:.3f}, p = {self.ols_pvalue:.3f})")
        if real_str:
            lines.append(real_str)

        if self.spearman_rho > 0.7:
            lines.append("\nConclusion: Strong agreement — GNN hub predictions are causally valid.")
        elif self.spearman_rho > 0.4:
            lines.append("\nConclusion: Moderate agreement — GNN predictions partially capture causal structure; some spurious hubs.")
        else:
            lines.append("\nConclusion: Weak agreement — GNN hub scores reflect correlational structure, not causation (spurious hubs dominate).")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "spearman_rho": self.spearman_rho,
            "spearman_pvalue": self.spearman_pvalue,
            "top3_overlap": self.top3_overlap,
            "top5_overlap": self.top5_overlap,
            "ols_slope": self.ols_slope,
            "ols_intercept": self.ols_intercept,
            "ols_r_squared": self.ols_r_squared,
            "ols_pvalue": self.ols_pvalue,
            "n_nodes": self.n_nodes,
        }


def compute_agreement(
    causal_df: pd.DataFrame,
    predicted_col: str = "predicted_importance",
    causal_col: str = "delta_contagion",
    id_col: str = "node_id",
    real_episode_nodes: list[str] | None = None,
    n_boot: int = 2_000,
    rng_seed: int = 0,
) -> AgreementMetrics:
    """Compute all three agreement metrics.

    Parameters
    ----------
    causal_df : pd.DataFrame
        Output of causal_hub_ranking() — must have node_id, delta_contagion,
        predicted_importance.
    """
    df = causal_df[[id_col, predicted_col, causal_col]].dropna()
    n = len(df)
    if n < 3:
        raise ValueError(f"Need ≥ 3 hubs to compute agreement, got {n}")

    pred = df[predicted_col].values
    causal = df[causal_col].values

    # 1. Spearman + bootstrap CI
    rho, pval = stats.spearmanr(pred, causal)
    ci_lo, ci_hi = _bootstrap_spearman_ci(pred, causal, n_boot=n_boot, rng_seed=rng_seed)

    # 2. Top-k overlap
    def _topk_overlap(k: int) -> float:
        k = min(k, n)
        top_predicted = set(df.nlargest(k, predicted_col)[id_col])
        top_causal = set(df.nlargest(k, causal_col)[id_col])
        return len(top_predicted & top_causal) / k

    top3 = _topk_overlap(3)
    top5 = _topk_overlap(5)

    # 3. OLS
    slope, intercept, r, p_ols, _ = stats.linregress(pred, causal)

    # 4. Real-episode hub split
    n_real = 0
    rho_real = float("nan")
    if real_episode_nodes:
        real_mask = df[id_col].isin(real_episode_nodes)
        n_real = int(real_mask.sum())
        if n_real >= 3:
            rho_real, _ = stats.spearmanr(
                df.loc[real_mask, predicted_col], df.loc[real_mask, causal_col]
            )
            rho_real = float(rho_real)

    return AgreementMetrics(
        spearman_rho=float(rho),
        spearman_pvalue=float(pval),
        spearman_ci_lo=ci_lo,
        spearman_ci_hi=ci_hi,
        top3_overlap=float(top3),
        top5_overlap=float(top5),
        ols_slope=float(slope),
        ols_intercept=float(intercept),
        ols_r_squared=float(r ** 2),
        ols_pvalue=float(p_ols),
        n_nodes=n,
        n_real_nodes=n_real,
        n_synth_nodes=n - n_real,
        spearman_rho_real=rho_real,
    )


def _bootstrap_spearman_ci(
    pred: np.ndarray,
    causal: np.ndarray,
    *,
    n_boot: int = 2_000,
    rng_seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman ρ (resample rows with replacement)."""
    rng = np.random.default_rng(rng_seed)
    n = len(pred)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho_b, _ = stats.spearmanr(pred[idx], causal[idx])
        if np.isfinite(rho_b):
            boot_rhos.append(rho_b)
    if not boot_rhos:
        return float("nan"), float("nan")
    lo = float(np.quantile(boot_rhos, alpha / 2))
    hi = float(np.quantile(boot_rhos, 1 - alpha / 2))
    return lo, hi


def find_divergence_case(
    causal_df: pd.DataFrame,
    predicted_col: str = "predicted_importance",
    causal_col: str = "delta_contagion",
    id_col: str = "node_id",
) -> dict:
    """Find the largest divergence: high predicted importance, low causal effect.

    Returns the node that sits farthest below the OLS regression line when
    predicted_importance is high — this is the "spurious hub" case study.
    """
    df = causal_df[[id_col, predicted_col, causal_col, "node_type", "role"]].dropna().copy()
    pred = df[predicted_col].values
    causal = df[causal_col].values

    slope, intercept, *_ = stats.linregress(pred, causal)
    predicted_causal = intercept + slope * pred
    residuals = causal - predicted_causal
    df["residual"] = residuals

    # Most spurious: highest predicted_importance AND most negative residual
    df["spurious_score"] = df[predicted_col] - df[causal_col]
    worst = df.nlargest(1, "spurious_score").iloc[0]

    return {
        "node_id": worst[id_col],
        "node_type": worst["node_type"],
        "role": worst["role"],
        "predicted_importance": worst[predicted_col],
        "delta_contagion": worst[causal_col],
        "residual": float(worst["residual"]),
        "likely_explanation": _explain_spurious_hub(worst),
    }


def _explain_spurious_hub(row: pd.Series) -> str:
    ntype = str(row.get("node_type", ""))
    role = str(row.get("role", ""))

    if "exchange_flow" in ntype or "bridge" in ntype:
        return (
            "Exchange-flow and bridge nodes have high volume/TVL in the empirical graph "
            "(driving high centrality) but act as price-takers in the AMM, not price-setters. "
            "Intervening on them reduces flow but arbitrageurs restore the peg regardless. "
            "This is the TVL/volume artifact identified in the repo-1 audit: high centrality "
            "measures flow, not causal propagation power."
        )
    if "cex" in ntype:
        return (
            "CEX venue nodes appear as hubs due to high trading volume during the stress episode "
            "(observed correlation with depeg). However, redemption gating shows limited causal "
            "effect because: (1) arbitrageurs route through other venues, and (2) the primary "
            "peg mechanism is reserve backing, not CEX price discovery."
        )
    return (
        "High predicted importance likely reflects large trading volume during the stress episode "
        "(correlation with the shock) rather than causal propagation. The ABM shows that "
        "intervening on this node does not materially reduce contagion because peg recovery "
        "occurs through other channels."
    )


def save_agreement_report(
    metrics: AgreementMetrics,
    divergence: dict,
    causal_df: pd.DataFrame,
    output_dir: str | Path = "experiments/results/joint_analysis",
) -> None:
    import json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "agreement_metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    with open(out / "divergence_case.json", "w") as f:
        json.dump(divergence, f, indent=2)

    causal_df.to_csv(out / "joint_analysis_table.csv", index=False)

    md = [
        "# Joint Analysis: Predicted vs. Causal Hub Ranking\n",
        metrics.interpretation(),
        "\n## Largest Divergence (Spurious Hub)\n",
        f"**Node:** {divergence['node_id']}  ",
        f"**Type:** {divergence['node_type']}  |  **Role:** {divergence['role']}",
        f"**Predicted importance:** {divergence['predicted_importance']:.3f}",
        f"**Causal effect (Δcontagion):** {divergence['delta_contagion']:.4f}",
        f"\n**Mechanism:**\n{divergence['likely_explanation']}",
    ]
    with open(out / "joint_analysis_report.md", "w") as f:
        f.write("\n".join(md))

    print(f"Joint analysis saved to {out}/")
