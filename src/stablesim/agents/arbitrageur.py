"""Cross-venue arbitrageur.

Heuristic policy: if spread between AMM pools exceeds threshold, swap on the
cheaper venue and sell on the more expensive one.  RL-trained variant uses the
same observation space but the policy comes from a PPO checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import BaseAgent

if TYPE_CHECKING:
    from ..engine.market import MultiVenueMarket


class Arbitrageur(BaseAgent):
    """Heuristic or RL-driven cross-venue arbitrageur.

    Parameters
    ----------
    min_spread : float
        Minimum price spread (fraction) that triggers arbitrage.
    max_trade_frac : float
        Maximum fraction of own wealth to deploy per step.
    policy : callable | None
        If provided, replaces heuristic — called as policy(obs) → action.
    """

    def __init__(
        self,
        agent_id: str,
        wealth: float = 100_000.0,
        min_spread: float = 0.002,
        max_trade_frac: float = 0.05,
        policy=None,
    ) -> None:
        super().__init__(agent_id, wealth)
        self.min_spread = min_spread
        self.max_trade_frac = max_trade_frac
        self.policy = policy

    def act(self, market: "MultiVenueMarket", obs: dict) -> None:
        if self.policy is not None:
            action = self.policy(obs)
            self._execute_rl_action(market, action)
            return

        prices = obs.get("prices", market.prices())
        if len(prices) < 2:
            return

        i_buy = int(np.argmin(prices))
        i_sell = int(np.argmax(prices))
        spread = prices[i_sell] - prices[i_buy]

        if spread < self.min_spread:
            return

        trade_size = min(self.wealth * self.max_trade_frac, market.pools[i_buy].y * 0.05)
        if trade_size <= 0:
            return

        try:
            dx = market.pools[i_buy].swap_y_for_x(trade_size)
            revenue = market.pools[i_sell].swap_x_for_y(dx)
            profit = revenue - trade_size
            self.record_pnl(profit)
        except (ValueError, ZeroDivisionError):
            pass

    def _execute_rl_action(self, market: "MultiVenueMarket", action) -> None:
        """Map RL action vector to market operations (stub for PPO integration)."""
        # action[0] = pool index to buy from (discretized)
        # action[1] = fraction of wealth to deploy
        pool_idx = int(np.clip(action[0], 0, len(market.pools) - 1))
        frac = float(np.clip(action[1], 0, self.max_trade_frac))
        trade_size = self.wealth * frac
        if trade_size < 1.0:
            return
        try:
            dx = market.pools[pool_idx].swap_y_for_x(trade_size)
            # Sell on primary pool (index 0) if not same venue
            sell_idx = 0 if pool_idx != 0 else min(1, len(market.pools) - 1)
            revenue = market.pools[sell_idx].swap_x_for_y(dx)
            self.record_pnl(revenue - trade_size)
        except (ValueError, ZeroDivisionError):
            pass
