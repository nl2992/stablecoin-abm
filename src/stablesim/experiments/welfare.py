"""Welfare decomposition and LP impermanent-loss metrics.

The "who pays for peg stability" angle: for each intervention, decompose
welfare (cumulative P&L) by agent type and compute LP IL vs. hold.

This is both a standalone result (if the headline comparison underdelivers)
and the mechanism evidence for why interventions work (following Gu §8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class WelfareDecomposition:
    """Welfare by agent type for one episode.

    Attributes
    ----------
    intervention : str
    scenario : str
    seed : int
    arbitrageur_pnl : float
    redeemer_pnl : float
    lp_pnl : float          — includes impermanent loss
    lp_il : float           — impermanent loss component only
    issuer_pnl : float
    noise_pnl : float
    total_pnl : float
    """

    intervention: str
    scenario: str
    seed: int
    arbitrageur_pnl: float = 0.0
    redeemer_pnl: float = 0.0
    lp_pnl: float = 0.0
    lp_il: float = 0.0
    issuer_pnl: float = 0.0
    noise_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        return (
            self.arbitrageur_pnl + self.redeemer_pnl
            + self.lp_pnl + self.issuer_pnl + self.noise_pnl
        )

    def to_dict(self) -> dict:
        return {
            "intervention": self.intervention,
            "scenario": self.scenario,
            "seed": self.seed,
            "welfare_arbitrageur": self.arbitrageur_pnl,
            "welfare_redeemer": self.redeemer_pnl,
            "welfare_lp": self.lp_pnl,
            "lp_il": self.lp_il,
            "welfare_issuer": self.issuer_pnl,
            "welfare_noise": self.noise_pnl,
            "welfare_total": self.total_pnl,
        }


def extract_welfare(agents: list, intervention: str, scenario: str, seed: int) -> WelfareDecomposition:
    """Extract welfare decomposition from a completed episode's agent list."""
    from ..agents.arbitrageur import Arbitrageur
    from ..agents.redeemer import Redeemer
    from ..agents.lp import LPAgent
    from ..agents.issuer import IssuerAgent
    from ..agents.noise import NoiseTrader

    arb_pnl = sum(a.cumulative_pnl for a in agents if isinstance(a, Arbitrageur))
    red_pnl = sum(a.cumulative_pnl for a in agents if isinstance(a, Redeemer))
    lp_agents = [a for a in agents if isinstance(a, LPAgent)]
    lp_pnl = sum(a.cumulative_pnl for a in lp_agents)
    # LP impermanent loss is stored in cumulative_pnl (negative component)
    # The LP records IL as record_pnl(il) where il = actual - hold_value
    # Approximate: lp_il = lp_pnl (this is the IL-only component from our LPAgent impl)
    lp_il = lp_pnl  # conservative: all LP P&L here is IL minus subsidies
    iss_pnl = sum(a.cumulative_pnl for a in agents if isinstance(a, IssuerAgent))
    noise_pnl = sum(a.cumulative_pnl for a in agents if isinstance(a, NoiseTrader))

    return WelfareDecomposition(
        intervention=intervention,
        scenario=scenario,
        seed=seed,
        arbitrageur_pnl=arb_pnl,
        redeemer_pnl=red_pnl,
        lp_pnl=lp_pnl,
        lp_il=lp_il,
        issuer_pnl=iss_pnl,
        noise_pnl=noise_pnl,
    )


def lp_il_vs_hold(
    pool_x_entry: float,
    pool_y_entry: float,
    pool_x_exit: float,
    pool_y_exit: float,
    entry_price: float = 1.0,
) -> float:
    """Compute LP impermanent loss vs. a simple hold strategy.

    IL = (value of LP share at exit) − (value of equivalent hold at exit price)

    For a 50/50 constant-product benchmark:
        hold_value = x_entry * p_exit + y_entry
        lp_value   = x_exit * p_exit + y_exit
    """
    if pool_x_exit <= 0 or pool_y_exit <= 0:
        return 0.0
    exit_price = pool_x_exit / max(pool_y_exit, 1e-9)  # approx spot price
    hold_value = pool_x_entry * exit_price + pool_y_entry
    lp_value = pool_x_exit * exit_price + pool_y_exit
    return lp_value - hold_value


def welfare_summary_table(welfare_records: list[WelfareDecomposition]) -> pd.DataFrame:
    """Aggregate welfare by (intervention, scenario) across seeds.

    Returns mean ± std for each agent type.
    """
    df = pd.DataFrame([w.to_dict() for w in welfare_records])
    numeric_cols = [c for c in df.columns if c not in ("intervention", "scenario", "seed")]
    grp = df.groupby(["intervention", "scenario"])
    mean = grp[numeric_cols].mean().add_suffix("_mean")
    std = grp[numeric_cols].std().add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def intervention_welfare_ranking(
    welfare_df: pd.DataFrame,
    key_metric: str = "welfare_total_mean",
) -> pd.DataFrame:
    """Rank interventions by total welfare impact (most welfare-preserving first)."""
    if key_metric not in welfare_df.columns:
        key_metric = [c for c in welfare_df.columns if "total_mean" in c][0]
    return welfare_df.sort_values(key_metric, ascending=False).reset_index(drop=True)
