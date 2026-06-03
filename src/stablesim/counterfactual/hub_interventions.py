"""Per-node intervention harness.

Maps each hub node from repo 1's network graph to an ABM component and
applies a targeted intervention.  This is the mechanism that makes repo 2 a
causal counterfactual oracle rather than just a simulation.

Node-to-ABM mapping:
  DEX_POOL         → AMM pool (circuit breaker or liquidity drain)
  CEX_VENUE        → RedemptionChannel (fee gate + queue)
  MINT_BURN        → IssuerAgent + reserve transparency boost
  BRIDGE           → Noise trader activity scaling (bridge = cross-chain flow)
  EXCHANGE_FLOW    → Arbitrageur min_spread increase (flow reduction)

Counterfactual question:
  "If hub X had been subject to intervention Y during episode E,
   how much would contagion have changed?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine.market import MultiVenueMarket


class NodeType(Enum):
    DEX_POOL = "dex_pool"
    CEX_VENUE = "cex_venue"
    MINT_BURN = "mint_burn"
    BRIDGE = "bridge"
    EXCHANGE_FLOW = "exchange_flow"


class InterventionType(Enum):
    CIRCUIT_BREAKER = "circuit_breaker"     # Halt AMM pool trading for N steps
    LIQUIDITY_DRAIN = "liquidity_drain"     # Remove fraction of pool liquidity
    REDEMPTION_GATE = "redemption_gate"     # Fee + queue + delay on redemptions
    TRANSPARENCY_BOOST = "transparency_boost"  # Force daily reserve disclosure
    FLOW_REDUCTION = "flow_reduction"       # Reduce arbitrageur activity (bridge/flow nodes)
    COMBINED = "combined"                   # CB + gate (strongest intervention)


# Default interventions by node type (from ROADMAP intervention knobs)
_DEFAULT_INTERVENTIONS: dict[NodeType, InterventionType] = {
    NodeType.DEX_POOL: InterventionType.CIRCUIT_BREAKER,
    NodeType.CEX_VENUE: InterventionType.REDEMPTION_GATE,
    NodeType.MINT_BURN: InterventionType.TRANSPARENCY_BOOST,
    NodeType.BRIDGE: InterventionType.FLOW_REDUCTION,
    NodeType.EXCHANGE_FLOW: InterventionType.FLOW_REDUCTION,
}


@dataclass
class HubNode:
    """A node from repo 1's contagion network.

    Parameters
    ----------
    node_id : str
        Repo 1 identifier (e.g. "curve_3pool", "usdc_coinbase").
    name : str
        Human-readable label.
    predicted_importance : float
        Composite hub importance from repo 1 (0–1, higher = more important).
    node_type : NodeType
        ABM component mapping.
    role : str
        Repo 1 role label: "originator", "amplifier", or "mixed".
    eigenvector : float
        Eigenvector centrality from repo 1.
    out_degree_w : float
        Weighted out-degree from repo 1.
    event_ids : list[str]
        Events in which this node appears as a hub.
    """

    node_id: str
    name: str
    predicted_importance: float
    node_type: NodeType = NodeType.CEX_VENUE
    role: str = "mixed"
    eigenvector: float = 0.0
    out_degree_w: float = 0.0
    event_ids: list[str] = field(default_factory=list)

    def default_intervention(self) -> InterventionType:
        return _DEFAULT_INTERVENTIONS.get(self.node_type, InterventionType.REDEMPTION_GATE)


@dataclass
class HubInterventionParams:
    """Intervention parameters applied when this hub is targeted."""

    intervention_type: InterventionType
    # Circuit breaker
    cb_threshold: float = 0.02      # very sensitive (fires early)
    cb_duration: int = 20
    # Redemption gate
    gate_fee_bps: float = 200.0
    gate_queue_len: int = 10
    gate_delay_steps: int = 6
    # Liquidity drain
    drain_fraction: float = 0.50
    # Flow reduction (as arbitrageur spread multiplier)
    flow_reduction_factor: float = 5.0  # multiply min_spread by this
    # Transparency
    transparency_freq: int = 1
    transparency_noise: float = 0.005
    # Pool index to intervene on (for multi-pool markets)
    pool_idx: int = 0


def build_intervention_params(
    hub: HubNode,
    intervention_type: InterventionType | None = None,
) -> HubInterventionParams:
    """Return default intervention params for a hub node."""
    itype = intervention_type or hub.default_intervention()
    return HubInterventionParams(intervention_type=itype)


def apply_hub_intervention(
    market: "MultiVenueMarket",
    hub: HubNode,
    params: HubInterventionParams | None = None,
    current_step: int = 0,
) -> None:
    """Mutate market in-place to apply the hub intervention.

    Called at step 0 of each counterfactual episode to install the intervention
    before any shocks or agent actions run.
    """
    if params is None:
        params = build_intervention_params(hub)

    itype = params.intervention_type

    if itype == InterventionType.CIRCUIT_BREAKER:
        # Install a very sensitive circuit breaker on the target pool
        market.redemption.cb_threshold = params.cb_threshold
        market.redemption.cb_duration = params.cb_duration

    elif itype == InterventionType.LIQUIDITY_DRAIN:
        # Remove fraction of pool liquidity (simulates hub removing its TVL)
        pool_idx = min(params.pool_idx, len(market.pools) - 1)
        pool = market.pools[pool_idx]
        frac = min(params.drain_fraction, 0.99)
        pool.remove_liquidity(frac)

    elif itype == InterventionType.REDEMPTION_GATE:
        # Add fee + queue + delay to the redemption channel
        market.redemption.fee_bps = max(market.redemption.fee_bps, params.gate_fee_bps)
        if market.redemption.max_queue == 0:
            market.redemption.max_queue = params.gate_queue_len
        market.redemption.delay_steps = max(market.redemption.delay_steps, params.gate_delay_steps)

    elif itype == InterventionType.TRANSPARENCY_BOOST:
        # Force frequent reserve disclosure
        market.reserve.transparency_freq = params.transparency_freq
        market.reserve.transparency_noise = params.transparency_noise

    elif itype == InterventionType.FLOW_REDUCTION:
        # Reduce arbitrageur sensitivity (models reduced cross-chain / exchange flow)
        # Implemented by installing a thicker min_spread in the market; agents respect this
        # via the `market._arb_min_spread_override` attribute
        market._arb_min_spread_override = getattr(market, "_arb_min_spread_override", 0.001) * params.flow_reduction_factor

    elif itype == InterventionType.COMBINED:
        # Apply both circuit breaker and redemption gate
        apply_hub_intervention(
            market,
            hub,
            HubInterventionParams(
                intervention_type=InterventionType.CIRCUIT_BREAKER,
                cb_threshold=params.cb_threshold,
                cb_duration=params.cb_duration,
            ),
            current_step,
        )
        apply_hub_intervention(
            market,
            hub,
            HubInterventionParams(
                intervention_type=InterventionType.REDEMPTION_GATE,
                gate_fee_bps=params.gate_fee_bps,
                gate_queue_len=params.gate_queue_len,
                gate_delay_steps=params.gate_delay_steps,
            ),
            current_step,
        )
