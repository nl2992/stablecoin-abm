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
    *,
    pool_amp: float = 100.0,
    noise_size: float = 1_000.0,
    shock_scale: float = 1.0,
    reserve_speed: float | None = None,
    contagion_coupling: float = 0.0,
    common_flow_vol: float = 0.0,
) -> dict:
    """Run one episode and return outcome metrics.

    Calibration knobs (thread through SMM so simulated moments respond to params):
      pool_amp      — stableswap amplification; lower => price depegs more easily.
      noise_size    — mean noise-trade size (USD); drives baseline price volatility.
      shock_scale   — multiplies scenario shock magnitudes; drives contagion magnitude.
      reserve_speed — override ReserveModel OU mean-reversion speed.

    Returns
    -------
    dict with keys: scenario, intervention, history, metrics
    """
    rng = np.random.default_rng(rng_seed)
    kw = intervention.to_market_kwargs()
    reserve_kw = dict(kw["reserve"])
    if reserve_speed is not None:
        reserve_kw["speed"] = reserve_speed

    # TWO venues: the cross-venue arbitrageur only acts when >=2 pools exist, so a
    # single pool silently disables all peg-restoring arbitrage. Two coupled pools also
    # give the cross-venue correlation moment. Shocks hit pool 0; arb (capital-capped)
    # partially transmits stress to pool 1 -> realistic contagion + partial cross-venue rho.
    # Shallow ($50k) pools so realistic flow and shocks move the mid-price.
    pools = [StableswapAMM(reserves=(50_000.0, 50_000.0), amp=pool_amp),
             StableswapAMM(reserves=(50_000.0, 50_000.0), amp=pool_amp)]
    market = MultiVenueMarket(
        pools=pools,
        redemption=RedemptionChannel(**kw["redemption"]),
        reserve=ReserveModel(rng=rng, **reserve_kw),
        rng=rng,
        contagion_coupling=contagion_coupling,
        common_flow_vol=common_flow_vol,
    )

    lp_subsidy = kw.get("lp_subsidy_rate", 0.0)
    agents = (
        [Arbitrageur("arb_0"), Arbitrageur("arb_1")]
        + [Redeemer("red_0"), Redeemer("red_1")]
        + [LPAgent("lp_0", subsidy_rate=lp_subsidy), LPAgent("lp_1", subsidy_rate=lp_subsidy)]
        + [IssuerAgent()]
        + [NoiseTrader(f"noise_{i}", rng=rng, trade_size_mean=noise_size,
                       trade_size_std=0.5 * noise_size) for i in range(n_noise_traders)]
    )

    for step in range(n_steps):
        shock_events = scenario.events_at(step)
        shock = shock_events[0] if shock_events else None
        if shock is not None and shock_scale != 1.0:
            from ..scenarios.schedule import ShockEvent
            shock = ShockEvent(step=shock.step, kind=shock.kind,
                               magnitude=shock.magnitude * shock_scale,
                               pool_idx=getattr(shock, "pool_idx", 0),
                               label=getattr(shock, "label", ""))
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
