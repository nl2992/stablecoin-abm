"""Engine-specific wiring for uniform ablation.

This is the ONLY module that touches MultiVenueMarket internals.
intervention_spec.py is engine-agnostic; this adapter resolves the
`# ADAPT:` placeholders for our Curve stableswap + redemption engine.

Uniform ablation semantics per NodeType
=========================================
DEX_POOL      → scale pool reserves by (1-alpha). At alpha=1 the pool is
                near-empty and cannot route meaningful flow (price impact
                collapses to near-zero; arb bots get nothing from it).

CEX_VENUE     → scale throughput as: fee_bps = alpha × 5000 (up to 50%),
                delay_steps = round(alpha × MAX_DELAY). At alpha=1 the channel
                is effectively sealed (massive fee + long delay).

MINT_BURN     → scale reserve_usd by (1-alpha). At alpha=1 the reserve is
                exhausted and cannot honour any redemptions.

BRIDGE /      → scale noise-trader trade_prob by (1-alpha). At alpha=1
EXCHANGE_FLOW   noise traders are silent, eliminating cross-venue pressure.

Invariants
==========
• alpha = 0.0 is an exact no-op (early return, nothing touched).
• alpha > 0 must make the node strictly less capable of transmitting stress.
• Every branch must be deterministic given (market, hub, alpha, agents, seed).

Tests
=====
• test_inference.py::test_alpha_zero_is_noop
• test_inference.py::test_alpha_one_makes_node_inert
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .hub_interventions import NodeType

if TYPE_CHECKING:
    from ..engine.market import MultiVenueMarket
    from .hub_interventions import HubNode

_MAX_DELAY = 200      # steps; effectively infinite for a 150-step episode
_MAX_FEE_BPS = 5000   # 50% fee at alpha=1
_EPSILON = 1e-9       # floor for scaling (avoid exact-zero AMM reserves)


def apply_ablation(
    market: "MultiVenueMarket",
    hub: "HubNode",
    alpha: float,
    agents: list,
) -> None:
    """Apply uniform ablation of intensity alpha to hub in market (in place).

    Parameters
    ----------
    market : MultiVenueMarket — will be mutated.
    hub : HubNode — determines which component is targeted.
    alpha : float in [0, 1] — ablation dose.  0.0 is a guaranteed no-op.
    agents : list of BaseAgent — needed for BRIDGE/EXCHANGE_FLOW branches.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if alpha == 0.0:
        return  # guaranteed no-op — baseline arm

    scale = max(1.0 - alpha, _EPSILON)
    ntype = hub.node_type

    if ntype == NodeType.DEX_POOL:
        _ablate_dex_pool(market, scale)

    elif ntype == NodeType.CEX_VENUE:
        _ablate_cex_venue(market, alpha)

    elif ntype == NodeType.MINT_BURN:
        _ablate_mint_burn(market, scale)

    elif ntype in (NodeType.BRIDGE, NodeType.EXCHANGE_FLOW):
        _ablate_flow(agents, scale)

    else:
        raise ValueError(f"No ablation handler for NodeType {ntype}")


# --------------------------------------------------------------------------- #
# Per-type implementations                                                    #
# --------------------------------------------------------------------------- #

def _ablate_dex_pool(market: "MultiVenueMarket", scale: float) -> None:
    """Scale all AMM pool reserves by scale = (1-alpha)."""
    for pool in market.pools:
        pool.x = pool.x * scale
        pool.y = pool.y * scale
        pool._D = pool._compute_D()


def _ablate_cex_venue(market: "MultiVenueMarket", alpha: float) -> None:
    """Reduce redemption-channel throughput proportional to alpha."""
    redemption = market.redemption
    redemption.fee_bps = max(redemption.fee_bps, alpha * _MAX_FEE_BPS)
    redemption.delay_steps = max(redemption.delay_steps, round(alpha * _MAX_DELAY))
    if alpha >= 1.0:
        # Full ablation: cap queue to 1 (one pending order max) as belt-and-suspenders
        if redemption.max_queue == 0:          # 0 means unlimited → set a cap
            redemption.max_queue = 1


def _ablate_mint_burn(market: "MultiVenueMarket", scale: float) -> None:
    """Scale reserve_usd by scale; exhaustion flag will catch the zero case."""
    market.redemption.reserve_usd = market.redemption.reserve_usd * scale
    market.reserve.ratio = market.reserve.ratio * scale


def _ablate_flow(agents: list, scale: float) -> None:
    """Scale noise-trader trade_prob by scale."""
    from ..agents.noise import NoiseTrader
    for agent in agents:
        if isinstance(agent, NoiseTrader):
            agent.trade_prob = max(0.0, agent.trade_prob * scale)


# --------------------------------------------------------------------------- #
# Dose-response sweep helper                                                  #
# --------------------------------------------------------------------------- #

def dose_response_ablate(
    market_factory,
    agent_factory,
    hub: "HubNode",
    alphas: tuple[float, ...],
    episode_fn,
    n_seeds: int = 20,
) -> dict[float, list[float]]:
    """Run episode_fn at each alpha in alphas over n_seeds.

    Returns: {alpha: [contagion_magnitude per seed]}
    Used to verify monotone dose-response before trusting the headline ranking.
    """
    results: dict[float, list[float]] = {a: [] for a in alphas}
    for seed in range(n_seeds):
        for alpha in alphas:
            market = market_factory(seed=seed)
            agents = agent_factory(seed=seed)
            apply_ablation(market, hub, alpha, agents)
            mag = episode_fn(market, agents, seed)
            results[alpha].append(mag)
    return results
