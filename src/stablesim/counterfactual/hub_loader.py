"""Load hub rankings from stablecoin-contagion-network (repo 1) outputs.

Reads:
  - table_node_centrality.csv    — predicted importance (eigenvector, out_degree)
  - table_event_study_summary_*.csv — empirical transmission rank per node per event
  - table_propagation_intensity.csv — event-level propagation score

Falls back to synthetic hub data if repo 1 results are not found.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .hub_interventions import HubNode, NodeType

# Default sibling repo path
_REPO1_ROOT = Path(__file__).parents[6] / "stablecoin-contagion-network"

# Node-type classifier: map node_id substring → NodeType
_NODE_TYPE_MAP = {
    "curve_": NodeType.DEX_POOL,
    "uniswap_": NodeType.DEX_POOL,
    "mint_burn": NodeType.MINT_BURN,
    "bridge": NodeType.BRIDGE,
    "exchange_flows": NodeType.EXCHANGE_FLOW,
    "binance": NodeType.CEX_VENUE,
    "coinbase": NodeType.CEX_VENUE,
    "kraken": NodeType.CEX_VENUE,
}


def _classify_node(node_id: str) -> NodeType:
    for pattern, ntype in _NODE_TYPE_MAP.items():
        if pattern in node_id:
            return ntype
    return NodeType.CEX_VENUE  # default


def load_hub_rankings(
    repo1_root: str | Path | None = None,
    event_id: str | None = None,
) -> list[HubNode]:
    """Load hub list from repo 1 node-centrality table.

    Returns nodes sorted by descending predicted importance (eigenvector centrality).

    Parameters
    ----------
    repo1_root : path to stablecoin-contagion-network repo root.
    event_id : filter to specific event (None = aggregate across events, de-duplicate).
    """
    root = Path(repo1_root or _REPO1_ROOT)
    centrality_path = root / "results" / "tables" / "table_node_centrality.csv"

    if centrality_path.exists():
        try:
            return _load_from_centrality(centrality_path, event_id)
        except Exception as e:
            print(f"Warning: could not read {centrality_path}: {e}. Using synthetic hubs.")

    return _synthetic_hubs()


def _load_from_centrality(path: Path, event_id: str | None) -> list[HubNode]:
    df = pd.read_csv(path)

    if event_id:
        df = df[df["event_id"] == event_id]

    # Aggregate across events: max eigenvector per node
    agg = (
        df.groupby("node_id")
        .agg(
            eigenvector=("eigenvector", "max"),
            out_degree_w=("out_degree_w", "max"),
            betweenness=("betweenness", "max"),
            role=("role", "first"),
        )
        .reset_index()
    )

    # Composite predicted importance: 0.6*eigenvector + 0.3*out_degree_w_norm + 0.1*betweenness
    agg["out_deg_norm"] = agg["out_degree_w"] / (agg["out_degree_w"].max() + 1e-9)
    agg["btwn_norm"] = agg["betweenness"] / (agg["betweenness"].max() + 1e-9)
    agg["predicted_importance"] = (
        0.6 * agg["eigenvector"]
        + 0.3 * agg["out_deg_norm"]
        + 0.1 * agg["btwn_norm"]
    )
    agg = agg.sort_values("predicted_importance", ascending=False).reset_index(drop=True)

    hubs = []
    for _, row in agg.iterrows():
        hubs.append(
            HubNode(
                node_id=row["node_id"],
                name=row["node_id"],
                predicted_importance=float(row["predicted_importance"]),
                node_type=_classify_node(row["node_id"]),
                role=str(row["role"]),
                eigenvector=float(row["eigenvector"]),
                out_degree_w=float(row["out_degree_w"]),
                event_ids=[event_id] if event_id else list(df[df["node_id"] == row["node_id"]]["event_id"].unique()),
            )
        )
    return hubs


def load_propagation_intensity(repo1_root: str | Path | None = None) -> pd.DataFrame:
    """Return the propagation intensity table from repo 1."""
    root = Path(repo1_root or _REPO1_ROOT)
    path = root / "results" / "tables" / "table_propagation_intensity.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_event_study_ranks(
    event_id: str,
    repo1_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load empirical transmission ranks from the event study summary table.

    Returns DataFrame with columns: node_id, transmission_rank, event_id.
    """
    root = Path(repo1_root or _REPO1_ROOT)
    path = root / "results" / "tables" / f"table_event_study_summary_{event_id}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            return df[["node_id", "transmission_rank", "event_id"]].dropna(subset=["transmission_rank"])
        except Exception:
            pass
    return pd.DataFrame(columns=["node_id", "transmission_rank", "event_id"])


def _synthetic_hubs() -> list[HubNode]:
    """Fallback synthetic hub list mirroring the real node set from repo 1."""
    return [
        HubNode("usdc_coinbase",       "USDC Coinbase",        0.85, NodeType.CEX_VENUE,    "amplifier",  0.614, 4.043),
        HubNode("ust_binance",         "UST Binance",          0.72, NodeType.CEX_VENUE,    "amplifier",  0.559, 0.162),
        HubNode("curve_3pool",         "Curve 3pool",          0.61, NodeType.DEX_POOL,     "mixed",      0.530, 0.120),
        HubNode("eth_bridge_flows",    "ETH Bridge",           0.53, NodeType.BRIDGE,       "mixed",      0.529, 2.276),
        HubNode("uniswap_usdc_usdt",   "Uniswap USDC/USDT",   0.47, NodeType.DEX_POOL,     "amplifier",  0.248, 2.351),
        HubNode("usdt_binance",        "USDT Binance",         0.40, NodeType.CEX_VENUE,    "originator", 0.353, 0.258),
        HubNode("usdc_mint_burn",      "USDC Mint/Burn",       0.31, NodeType.MINT_BURN,    "mixed",      0.205, 1.001),
        HubNode("eth_usdc_flows",      "ETH USDC Flows",       0.26, NodeType.EXCHANGE_FLOW, "mixed",     0.303, 2.938),
    ]
