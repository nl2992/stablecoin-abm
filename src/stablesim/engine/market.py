"""Multi-venue market coordinator.

Holds one or more StableswapAMM pools plus a primary RedemptionChannel.
Each step: apply exogenous shock → run agent actions → settle redemptions → record state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .amm import StableswapAMM
from .redemption import RedemptionChannel
from .reserve import ReserveModel

if TYPE_CHECKING:
    from ..scenarios.schedule import ShockEvent


class MultiVenueMarket:
    """Coordinates AMM pools, redemption channel, and reserve model.

    Parameters
    ----------
    pools : list[StableswapAMM]
        One or more stableswap pools (index 0 is primary).
    redemption : RedemptionChannel
    reserve : ReserveModel
    rng : np.random.Generator | None
    """

    def __init__(
        self,
        pools: list[StableswapAMM] | None = None,
        redemption: RedemptionChannel | None = None,
        reserve: ReserveModel | None = None,
        rng: np.random.Generator | None = None,
        contagion_coupling: float = 0.0,
        common_flow_vol: float = 0.0,
    ) -> None:
        self.pools = pools or [StableswapAMM()]
        self.redemption = redemption or RedemptionChannel()
        self.reserve = reserve or ReserveModel()
        self.rng = rng or np.random.default_rng()
        # Fraction of a venue-0 shock that transmits DIRECTLY to other venues
        # (cross-venue information/flow channel, beyond arbitrage). Controls the
        # empirical cross-venue correlation moment.
        self.contagion_coupling = float(contagion_coupling)
        # Common (market-wide) order-flow factor: each step a single signed flow is
        # applied in the SAME direction to every venue. This is the shared "value"
        # innovation that makes venues co-move (positive cross-venue correlation) and
        # is the dominant source of baseline price volatility. Without it the only
        # cross-venue link is arbitrage, which anti-correlates venues (a spread trade).
        self.common_flow_vol = float(common_flow_vol)
        self.step_count = 0
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Prices

    def prices(self) -> list[float]:
        """Spot price (stablecoin / USD) from each pool."""
        return [p.price() for p in self.pools]

    def mid_price(self) -> float:
        """Volume-weighted average price across pools (equal-weight here)."""
        ps = self.prices()
        return float(np.mean(ps))

    def depeg(self) -> float:
        """Signed depeg: mid_price - 1.0."""
        return self.mid_price() - 1.0

    # ------------------------------------------------------------------
    # Step

    def step(self, shock: "ShockEvent | None" = None) -> dict:
        """Advance the market by one time step.

        1. Advance reserve OU process.
        2. Apply optional exogenous shock (price / liquidity / reserve).
        3. Check circuit breaker.
        4. Settle due redemption orders.
        5. Record and return state snapshot.
        """
        self.reserve.step()

        # Common market-wide flow (shared value innovation) — same direction on all pools.
        # Mean-reverting (OU-like) so it supplies baseline volatility + cross-venue
        # co-movement WITHOUT random-walking pool inventory to infinity: the flow has a
        # random innovation plus a pull back toward peg proportional to current deviation.
        if self.common_flow_vol > 0:
            dev = self.mid_price() - 1.0
            f = float(self.rng.normal(0.0, self.common_flow_vol)) - 0.5 * dev
            for pool in self.pools:
                try:
                    if f > 0:
                        pool.swap_x_for_y(min(f, 0.2) * pool.x)
                    elif f < 0:
                        pool.swap_y_for_x(min(-f, 0.2) * pool.y)
                except Exception:
                    pass

        if shock is not None:
            self._apply_shock(shock)

        price = self.mid_price()
        self.redemption.check_and_trigger(price, self.step_count)
        settled = self.redemption.settle(self.step_count)

        snapshot = self._snapshot(settled)
        self._history.append(snapshot)
        self.step_count += 1
        return snapshot

    def _apply_shock(self, shock: "ShockEvent") -> None:
        """Apply an exogenous shock to the specified pool or reserve."""
        kind = shock.kind
        pool_idx = getattr(shock, "pool_idx", 0)
        pool = self.pools[pool_idx]

        # Build the list of (pool, magnitude) to shock: the target pool at full
        # magnitude, plus every other pool at coupling x magnitude (cross-venue channel).
        targets = [(pool, shock.magnitude)]
        if self.contagion_coupling > 0 and kind in ("sell_pressure", "buy_pressure", "liquidity_removal"):
            for k, other in enumerate(self.pools):
                if k != pool_idx:
                    targets.append((other, shock.magnitude * self.contagion_coupling))

        for tgt, mag in targets:
            if kind == "sell_pressure":
                try:
                    tgt.swap_x_for_y(mag * tgt.x)
                except Exception:
                    pass
            elif kind == "buy_pressure":
                try:
                    tgt.swap_y_for_x(mag * tgt.y)
                except Exception:
                    pass
            elif kind == "liquidity_removal":
                tgt.remove_liquidity(min(mag, 0.99))
        if kind == "reserve_drop":
            self.reserve.ratio = max(0.0, self.reserve.ratio - shock.magnitude)

    def _snapshot(self, settled: list[dict]) -> dict:
        return {
            "step": self.step_count,
            "prices": self.prices(),
            "mid_price": self.mid_price(),
            "depeg": self.depeg(),
            "reserve_ratio": self.reserve.ratio,
            "reserve_perceived": self.reserve.perceived_backing,
            "queue_depth": self.redemption.queue_depth(),
            "settled_count": len(settled),
            "pool_states": [p.state() for p in self.pools],
        }

    def history_df(self):
        """Return simulation history as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._history)
