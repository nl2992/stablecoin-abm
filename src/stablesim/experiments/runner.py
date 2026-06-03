"""Single-episode simulation runner."""

from __future__ import annotations

import numpy as np

from ..agents.arbitrageur import Arbitrageur
from ..agents.issuer import IssuerAgent
from ..agents.lp import LPAgent
from ..agents.noise import NoiseTrader
from ..agents.redeemer import Redeemer
from ..engine.market import MultiVenueMarket
from ..engine.amm import StableswapAMM
from ..engine.redemption import RedemptionChannel
from ..engine.reserve import ReserveModel
from ..scenarios.schedule import ShockSchedule
from .interventions import InterventionConfig


def run_episode(
    scenario: ShockSchedule,
    intervention: InterventionConfig,
    n_steps: int = 100,
    n_noise_traders: int = 5,
    rng_seed: int | None = None,
) -> dict:
    """Run one episode and return outcome metrics.

    Returns
    -------
    dict with keys: scenario, intervention, history_df, metrics
    """
    rng = np.random.default_rng(rng_seed)
    kw = intervention.to_market_kwargs()

    pools = [StableswapAMM()]
    market = MultiVenueMarket(
        pools=pools,
        redemption=RedemptionChannel(**kw["redemption"]),
        reserve=ReserveModel(rng=rng, **kw["reserve"]),
        rng=rng,
    )

    lp_subsidy = kw.get("lp_subsidy_rate", 0.0)
    agents = (
        [Arbitrageur("arb_0"), Arbitrageur("arb_1")]
        + [Redeemer("red_0"), Redeemer("red_1")]
        + [LPAgent("lp_0", subsidy_rate=lp_subsidy), LPAgent("lp_1", subsidy_rate=lp_subsidy)]
        + [IssuerAgent()]
        + [NoiseTrader(f"noise_{i}", rng=rng) for i in range(n_noise_traders)]
    )

    for step in range(n_steps):
        shock_events = scenario.events_at(step)
        shock = shock_events[0] if shock_events else None
        snap = market.step(shock=shock)
        for agent in agents:
            agent.act(market, snap)

    df = market.history_df()
    from ..analysis.metrics import compute_metrics
    metrics = compute_metrics(df, agents)

    return {
        "scenario": scenario.name,
        "intervention": intervention.label,
        "history": df,
        "metrics": metrics,
    }
