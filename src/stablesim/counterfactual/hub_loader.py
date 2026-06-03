"""Load hub rankings from stablecoin-contagion-network (repo 1) outputs.

CRITICAL: synthetic hub fallback is EXPLICITLY DISABLED by default.
Passing --allow-synthetic-hubs on scripts or allow_synthetic=True here
is required. Any output artifact generated from synthetic data is labelled
"SYNTHETIC — not for paper figures" in its stamp file.

Raises
------
FileNotFoundError
    If repo-1 centrality table is not found and allow_synthetic=False.
ValueError
    If the table schema has drifted from the locked contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .hub_interventions import HubNode, NodeType

# --------------------------------------------------------------------------- #
# Locked schema contract — fail loudly on mismatch                            #
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "node_id", "out_degree_w", "in_degree_w",
    "eigenvector", "betweenness", "role", "event_id",
})
SCHEMA_VERSION = "v1.0"   # bump when REQUIRED_COLUMNS changes

# --------------------------------------------------------------------------- #
# Default path to sibling repo                                                 #
# --------------------------------------------------------------------------- #
_REPO1_ROOT = Path(__file__).parents[6] / "stablecoin-contagion-network"

# --------------------------------------------------------------------------- #
# Node-type classifier                                                         #
# --------------------------------------------------------------------------- #
_NODE_TYPE_MAP: list[tuple[str, NodeType]] = [
    ("curve_",          NodeType.DEX_POOL),
    ("uniswap_",        NodeType.DEX_POOL),
    ("mint_burn",       NodeType.MINT_BURN),
    ("bridge",          NodeType.BRIDGE),
    ("exchange_flows",  NodeType.EXCHANGE_FLOW),
    ("binance",         NodeType.CEX_VENUE),
    ("coinbase",        NodeType.CEX_VENUE),
    ("kraken",          NodeType.CEX_VENUE),
]


def _classify_node(node_id: str) -> NodeType:
    for pattern, ntype in _NODE_TYPE_MAP:
        if pattern in node_id:
            return ntype
    return NodeType.CEX_VENUE  # conservative default


def _validate_schema(df: pd.DataFrame, path: Path) -> None:
    """Raise ValueError if the table is missing required columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"table_node_centrality.csv schema mismatch at {path}. "
            f"Missing columns: {sorted(missing)}. "
            f"Schema version {SCHEMA_VERSION} requires: {sorted(REQUIRED_COLUMNS)}. "
            f"Update REQUIRED_COLUMNS or re-export from repo 1."
        )


# --------------------------------------------------------------------------- #
# Public loader                                                                 #
# --------------------------------------------------------------------------- #
def load_hub_rankings(
    repo1_root: str | Path | None = None,
    event_id: str | None = None,
    allow_synthetic: bool = False,
) -> list[HubNode]:
    """Load hub list from repo 1's node-centrality table.

    Parameters
    ----------
    repo1_root : path to stablecoin-contagion-network root.
    event_id : filter to a specific event (None = aggregate across all events).
    allow_synthetic : if True, fall back to synthetic hubs when repo 1 is missing.
                      NEVER set True for paper-figure runs.

    Returns
    -------
    list[HubNode] sorted by predicted_importance descending.

    Raises
    ------
    FileNotFoundError if repo-1 table missing and allow_synthetic=False.
    ValueError if table schema has drifted.
    """
    root = Path(repo1_root or _REPO1_ROOT)
    path = root / "results" / "tables" / "table_node_centrality.csv"

    if not path.exists():
        if allow_synthetic:
            import warnings
            warnings.warn(
                f"table_node_centrality.csv not found at {path}. "
                "Falling back to SYNTHETIC hub list. "
                "This is suitable for smoke-testing ONLY. "
                "Set allow_synthetic=False (default) for paper runs.",
                UserWarning,
                stacklevel=2,
            )
            return _synthetic_hubs()
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Run repo 1 (stablecoin-contagion-network) to generate it, "
            f"or pass allow_synthetic=True for smoke-testing ONLY."
        )

    df = pd.read_csv(path)
    _validate_schema(df, path)

    if event_id:
        df = df[df["event_id"] == event_id]
        if df.empty:
            raise ValueError(f"No rows for event_id={event_id!r} in {path}")

    return _build_hubs(df, event_id)


def _build_hubs(df: pd.DataFrame, event_id: Optional[str]) -> list[HubNode]:
    """Aggregate across events and compute composite predicted importance."""
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

    # Composite: 0.6 × eigenvector + 0.3 × out_degree_norm + 0.1 × betweenness_norm
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
        node_id = row["node_id"]
        all_events = (
            df[df["node_id"] == node_id]["event_id"].unique().tolist()
            if event_id is None else [event_id]
        )
        hubs.append(HubNode(
            node_id=node_id,
            name=node_id,
            predicted_importance=float(row["predicted_importance"]),
            node_type=_classify_node(node_id),
            role=str(row["role"]),
            eigenvector=float(row["eigenvector"]),
            out_degree_w=float(row["out_degree_w"]),
            event_ids=all_events,
        ))
    return hubs


def load_propagation_intensity(repo1_root: str | Path | None = None) -> pd.DataFrame:
    """Return the propagation intensity table from repo 1 (may be empty if not found)."""
    root = Path(repo1_root or _REPO1_ROOT)
    path = root / "results" / "tables" / "table_propagation_intensity.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_event_study_ranks(
    event_id: str,
    repo1_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load empirical transmission ranks from the event-study summary table."""
    root = Path(repo1_root or _REPO1_ROOT)
    path = root / "results" / "tables" / f"table_event_study_summary_{event_id}.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "transmission_rank" in df.columns:
            return df[["node_id", "transmission_rank", "event_id"]].dropna(subset=["transmission_rank"])
    return pd.DataFrame(columns=["node_id", "transmission_rank", "event_id"])


# --------------------------------------------------------------------------- #
# Synthetic hub list — smoke-test only, always labelled                        #
# --------------------------------------------------------------------------- #
def _synthetic_hubs() -> list[HubNode]:
    """Synthetic hub list mirroring the real node set from repo 1.

    ONLY for smoke-testing the pipeline. Never use for paper figures.
    """
    return [
        HubNode("usdc_coinbase",       "USDC Coinbase",         0.85, NodeType.CEX_VENUE,    "amplifier",  0.614, 4.043),
        HubNode("ust_binance",         "UST Binance",           0.72, NodeType.CEX_VENUE,    "amplifier",  0.559, 0.162),
        HubNode("curve_3pool",         "Curve 3pool",           0.61, NodeType.DEX_POOL,     "mixed",      0.530, 0.120),
        HubNode("eth_bridge_flows",    "ETH Bridge",            0.53, NodeType.BRIDGE,       "mixed",      0.529, 2.276),
        HubNode("uniswap_usdc_usdt",   "Uniswap USDC/USDT",    0.47, NodeType.DEX_POOL,     "amplifier",  0.248, 2.351),
        HubNode("usdt_binance",        "USDT Binance",          0.40, NodeType.CEX_VENUE,    "originator", 0.353, 0.258),
        HubNode("usdc_mint_burn",      "USDC Mint/Burn",        0.31, NodeType.MINT_BURN,    "mixed",      0.205, 1.001),
        HubNode("eth_usdc_flows",      "ETH USDC Flows",        0.26, NodeType.EXCHANGE_FLOW,"mixed",      0.303, 2.938),
    ]
