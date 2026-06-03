"""Reserve backing model for the stablecoin issuer.

Two layers:
  1. True backing ratio (r_t): an OU process representing erosion of reserve
     quality through credit risk, yield-bearing assets losing value, etc.
     Agents cannot observe r_t directly — only the disclosed signal.
  2. Disclosure: issuer publishes a noisy signal of r_t at configurable
     frequency (transparency_freq=0 → fully opaque).

Exhaustion:
  When r_t ≤ exhaustion_threshold, the reserve is treated as exhausted —
  the RedemptionChannel will cap payouts and the market records the event.
"""

from __future__ import annotations

import numpy as np


class ReserveModel:
    """Stochastic reserve backing with controlled disclosure and exhaustion logic.

    Parameters
    ----------
    initial_ratio : float
        Starting r_0 (1.0 = fully backed).
    mean_ratio : float
        Long-run mean backing ratio (theta in OU).
    speed : float
        OU mean-reversion speed kappa.
    vol : float
        Volatility of the backing ratio (sigma in OU).
    transparency_freq : int
        Disclose true ratio every N steps (0 = never disclose; opaque issuer).
    transparency_noise : float
        Std dev of Gaussian noise added to each disclosed ratio.
    exhaustion_threshold : float
        r_t below this level → reserve declared exhausted.
    rng : np.random.Generator | None
    """

    def __init__(
        self,
        initial_ratio: float = 1.0,
        mean_ratio: float = 1.0,
        speed: float = 0.05,
        vol: float = 0.02,
        transparency_freq: int = 0,
        transparency_noise: float = 0.0,
        exhaustion_threshold: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.ratio = float(initial_ratio)
        self.mean_ratio = float(mean_ratio)
        self.speed = float(speed)
        self.vol = float(vol)
        self.transparency_freq = int(transparency_freq)
        self.transparency_noise = float(transparency_noise)
        self.exhaustion_threshold = float(exhaustion_threshold)
        self.rng = rng or np.random.default_rng()
        self._step: int = 0
        self._last_disclosed: float | None = None
        self._exhausted_at: int | None = None

    # ------------------------------------------------------------------
    # Dynamics

    def step(self, dt: float = 1.0) -> None:
        """Advance the backing ratio by one step (Euler-Maruyama OU)."""
        dW = self.rng.normal(0.0, np.sqrt(dt))
        self.ratio += self.speed * (self.mean_ratio - self.ratio) * dt + self.vol * dW
        self.ratio = max(0.0, self.ratio)
        self._step += 1

        # Disclosure signal
        if self.transparency_freq > 0 and self._step % self.transparency_freq == 0:
            noise = (
                self.rng.normal(0.0, self.transparency_noise)
                if self.transparency_noise > 0.0
                else 0.0
            )
            self._last_disclosed = float(np.clip(self.ratio + noise, 0.0, None))

        # Exhaustion detection
        if self._exhausted_at is None and self.ratio <= self.exhaustion_threshold:
            self._exhausted_at = self._step

    def shock(self, delta: float) -> None:
        """Apply an instantaneous shock to the backing ratio (e.g. reserve haircut)."""
        self.ratio = max(0.0, self.ratio + delta)
        if self._exhausted_at is None and self.ratio <= self.exhaustion_threshold:
            self._exhausted_at = self._step

    # ------------------------------------------------------------------
    # Observability

    @property
    def disclosed_ratio(self) -> float | None:
        """Most recently disclosed backing ratio (None if never disclosed)."""
        return self._last_disclosed

    @property
    def perceived_backing(self) -> float:
        """Agent-observable backing: disclosed if available, else prior mean."""
        return self._last_disclosed if self._last_disclosed is not None else self.mean_ratio

    @property
    def is_exhausted(self) -> bool:
        """True once ratio has fallen to or below the exhaustion threshold."""
        return self._exhausted_at is not None

    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "ratio": self.ratio,
            "disclosed": self._last_disclosed,
            "perceived": self.perceived_backing,
            "is_exhausted": self.is_exhausted,
            "exhausted_at": self._exhausted_at,
            "step": self._step,
        }
