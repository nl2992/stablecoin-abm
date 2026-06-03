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
    ) -> None:
        self.pools = pools or [StableswapAMM()]
        self.redemption = redemption or RedemptionChannel()
        self.reserve = reserve or ReserveModel()
        self.rng = rng or np.random.default_rng()
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

        if kind == "sell_pressure":
            # Force-sell stablecoin into pool (drives price down)
            amount = shock.magnitude * pool.x
            try:
                pool.swap_x_for_y(amount)
            except Exception:
                pass
        elif kind == "buy_pressure":
            amount = shock.magnitude * pool.y
            try:
                pool.swap_y_for_x(amount)
            except Exception:
                pass
        elif kind == "liquidity_removal":
            frac = min(shock.magnitude, 0.99)
            pool.remove_liquidity(frac)
        elif kind == "reserve_drop":
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
