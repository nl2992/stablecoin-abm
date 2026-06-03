"""Headline figures for the paper.

Fig 1 — Headline: predicted importance (x) vs. causal effect (y), one point per hub.
         Agreement on the diagonal; spurious hubs off it.
Fig 2 — Calibration overlay: simulated vs. empirical peg path / half-life.
Fig 3 — Per-intervention midprice/peg evolution (Gu Fig 3 analog).
Fig 4 — Welfare decomposition matrix (Gu / JaxMARL Fig 5 analog).
Fig 5 — Regime / threshold figure (where behaviour flips).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Figure 1 — Headline scatter: predicted vs. causal

def plot_headline_scatter(
    causal_df: pd.DataFrame,
    predicted_col: str = "predicted_importance",
    causal_col: str = "delta_contagion",
    id_col: str = "node_id",
    role_col: str = "role",
    episode_nodes: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Predicted importance (x) vs. causal effect (y) scatter.

    Points near the diagonal: GNN predictions are causally valid.
    Points in the lower-right quadrant: spurious hubs (high predicted, low causal).

    Parameters
    ----------
    episode_nodes : list of node_ids to highlight (real empirical episodes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    role_colors = {
        "originator": "#d62728",
        "amplifier":  "#1f77b4",
        "mixed":       "#7f7f7f",
    }

    for _, row in causal_df.iterrows():
        color = role_colors.get(str(row.get(role_col, "mixed")), "#7f7f7f")
        is_episode = episode_nodes and row[id_col] in episode_nodes
        marker = "*" if is_episode else "o"
        size = 140 if is_episode else 70
        ax.scatter(row[predicted_col], row[causal_col], c=color, marker=marker,
                   s=size, zorder=3, edgecolors="white", linewidths=0.5)

    # OLS regression line
    from scipy import stats
    x = causal_df[predicted_col].values
    y = causal_df[causal_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        slope, intercept, *_ = stats.linregress(x[mask], y[mask])
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "k--", linewidth=1.2, alpha=0.6, label="OLS fit")

    # Node labels
    for _, row in causal_df.iterrows():
        ax.annotate(
            row[id_col].replace("_", "\n"),
            (row[predicted_col], row[causal_col]),
            fontsize=7, ha="left", va="bottom", alpha=0.75,
            xytext=(4, 4), textcoords="offset points",
        )

    # Quadrant shading
    x_mid = np.nanmedian(causal_df[predicted_col])
    y_mid = np.nanmedian(causal_df[causal_col])
    ax.axvline(x_mid, color="gray", linewidth=0.5, alpha=0.4)
    ax.axhline(y_mid, color="gray", linewidth=0.5, alpha=0.4)
    ax.fill_between([x_mid, causal_df[predicted_col].max() * 1.05],
                    causal_df[causal_col].min() * 1.2, y_mid,
                    alpha=0.05, color="#d62728", label="Spurious hub region")

    # Legend
    patches = [mpatches.Patch(color=c, label=r) for r, c in role_colors.items()]
    if episode_nodes:
        patches.append(plt.scatter([], [], marker="*", s=140, c="gray", label="Real episode node"))
    ax.legend(handles=patches, fontsize=8, loc="upper left")

    ax.set_xlabel("Predicted hub importance (repo 1: eigenvector centrality)", fontsize=11)
    ax.set_ylabel("Causal effect Δcontagion (repo 2: ABM counterfactual)", fontsize=11)
    ax.set_title(
        "Predicted importance vs. causal effect — one point per hub\n"
        "Agreement near diagonal; spurious hubs lower-right",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Calibration overlay

def plot_calibration_overlay(
    simulated_paths: list[np.ndarray],
    empirical_path: Optional[np.ndarray] = None,
    title: str = "Calibration overlay: simulated vs. empirical peg path",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Simulated peg-deviation paths vs. empirical reference."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    for path in simulated_paths:
        ax.plot(path, color="steelblue", alpha=0.3, linewidth=0.8)

    if simulated_paths:
        sim_median = np.median(simulated_paths, axis=0)
        ax.plot(sim_median, color="steelblue", linewidth=2.0, label="Simulated (median)")

    if empirical_path is not None:
        ax.plot(empirical_path, color="crimson", linewidth=1.5, linestyle="--", label="Empirical")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Depeg (price − 1)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Per-intervention peg evolution

def plot_intervention_peg_evolution(
    histories: dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Peg evolution for each intervention label (Gu Fig 3 analog)."""
    cmap = plt.get_cmap("tab10")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    for i, (label, df) in enumerate(histories.items()):
        ax.plot(df["step"], df["depeg"], color=cmap(i), linewidth=1.2, label=label, alpha=0.85)

    ax.set_xlabel("Step")
    ax.set_ylabel("Depeg (price − 1)")
    ax.set_title("Peg evolution by intervention")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Welfare decomposition matrix

def plot_welfare_matrix(
    sweep_df: pd.DataFrame,
    agent_types: list[str] = ("arbitrageur", "redeemer", "lp", "issuer"),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Welfare by agent type × intervention (Gu / JaxMARL Fig 5 analog)."""
    import seaborn as sns

    welfare_cols = [f"welfare_{t}" for t in agent_types if f"welfare_{t}" in sweep_df.columns]
    pivot_data = {}
    for col in welfare_cols:
        agent = col.replace("welfare_", "")
        pivot_data[agent] = sweep_df.groupby("intervention")[col].mean()

    pivot = pd.DataFrame(pivot_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.6)))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, ax=ax, linewidths=0.5,
    )
    ax.set_title("Welfare decomposition: cumulative P&L by agent type × intervention\n(positive = gains, negative = losses)")
    ax.set_ylabel("Intervention")
    ax.set_xlabel("Agent type")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save all figures

def save_figure(fig: plt.Figure, name: str, output_dir: str | Path = "paper/figures") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved {name}.pdf / .png to {out}/")
