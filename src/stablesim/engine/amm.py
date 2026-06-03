"""Curve-style stableswap AMM (2-token).

Invariant (n=2):
    A * 4 * (x + y) + D = 4 * A * D + D^3 / (4 * x * y)

where A is the amplification coefficient.  Higher A → tighter peg range,
lower slippage near equilibrium, larger losses on large imbalances.
"""

from __future__ import annotations

import numpy as np


class StableswapAMM:
    """Two-token stableswap pool (stablecoin ↔ stablecoin or stablecoin ↔ USD).

    Parameters
    ----------
    reserves : (float, float)
        Initial reserves (x, y).
    amp : float
        Amplification coefficient A.  Typical range 10–500.
    fee_bps : float
        Swap fee in basis points (e.g. 4 = 0.04%).
    """

    def __init__(
        self,
        reserves: tuple[float, float] = (1_000_000.0, 1_000_000.0),
        amp: float = 100.0,
        fee_bps: float = 4.0,
    ) -> None:
        self.x, self.y = float(reserves[0]), float(reserves[1])
        self.amp = float(amp)
        self.fee = fee_bps / 10_000.0
        self._D = self._compute_D()

    # ------------------------------------------------------------------
    # Invariant

    def _compute_D(self, tol: float = 1e-9, max_iter: int = 256) -> float:
        """Newton solve for the invariant D given current (x, y)."""
        S = self.x + self.y
        if S == 0:
            return 0.0
        A4 = 4 * self.amp
        D = S
        for _ in range(max_iter):
            D_prev = D
            D3_over_4xy = D**3 / (4 * self.x * self.y)
            D = D * (A4 * S + 2 * D3_over_4xy) / ((A4 - 1) * D + 3 * D3_over_4xy)
            if abs(D - D_prev) < tol:
                break
        return D

    def _y_given_x(self, x_new: float, tol: float = 1e-9, max_iter: int = 256) -> float:
        """Given new reserve x, find y that satisfies the invariant D."""
        D = self._D
        A4 = 4 * self.amp
        b = x_new + D / A4 - D
        c = -(D**3) / (A4 * 4 * x_new)
        y = D
        for _ in range(max_iter):
            y_prev = y
            y = (y * y - c) / (2 * y + b)
            if abs(y - y_prev) < tol:
                break
        return y

    # ------------------------------------------------------------------
    # Public interface

    def price(self) -> float:
        """Analytic marginal price dy/dx (spot price of x in terms of y).

        Derived via implicit differentiation of the stableswap invariant:
            F(x, y) = 4A(x+y) + D − 4AD − D³/(4xy) = 0
            ∂F/∂x = 4A + D³/(4x²y)
            ∂F/∂y = 4A + D³/(4xy²)
            price  = (∂F/∂x) / (∂F/∂y)

        At x=y (equilibrium): ∂F/∂x = ∂F/∂y → price = 1.0 exactly.
        """
        D3 = self._D ** 3
        dFdx = 4.0 * self.amp + D3 / (4.0 * self.x ** 2 * self.y)
        dFdy = 4.0 * self.amp + D3 / (4.0 * self.x * self.y ** 2)
        return dFdx / dFdy

    def swap_x_for_y(self, dx: float) -> float:
        """Swap dx units of x into y.  Returns dy received (net of fee)."""
        if dx <= 0:
            raise ValueError("dx must be positive")
        x_new = self.x + dx
        y_new = self._y_given_x(x_new)
        dy_gross = self.y - y_new
        dy_net = dy_gross * (1 - self.fee)
        self.x = x_new
        self.y = y_new + (dy_gross - dy_net)  # fee stays in pool
        self._D = self._compute_D()
        return dy_net

    def swap_y_for_x(self, dy: float) -> float:
        """Swap dy units of y into x.  Returns dx received (net of fee)."""
        if dy <= 0:
            raise ValueError("dy must be positive")
        # Symmetric: swap token roles temporarily
        self.x, self.y = self.y, self.x
        dx = self.swap_x_for_y(dy)
        self.x, self.y = self.y, self.x
        return dx

    def add_liquidity(self, dx: float, dy: float) -> float:
        """Add (dx, dy) liquidity.  Returns LP tokens minted (proportional to D)."""
        D_before = self._D
        self.x += dx
        self.y += dy
        D_after = self._compute_D()
        self._D = D_after
        lp_minted = D_after - D_before if D_before > 0 else D_after
        return lp_minted

    def remove_liquidity(self, lp_fraction: float) -> tuple[float, float]:
        """Remove lp_fraction of pool liquidity.  Returns (dx, dy) withdrawn."""
        if not 0 < lp_fraction <= 1:
            raise ValueError("lp_fraction must be in (0, 1]")
        dx = self.x * lp_fraction
        dy = self.y * lp_fraction
        self.x -= dx
        self.y -= dy
        self._D = self._compute_D()
        return dx, dy

    def state(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "D": self._D,
            "price": self.price(),
            "amp": self.amp,
        }
