"""Gymnasium environment wrapping MultiVenueMarket for RL training.

Observation space: [mid_price, depeg, reserve_perceived, queue_depth/max,
                    pool_0_x, pool_0_y, agent_wealth_norm]
Action space: continuous [0,1]^2  (pool_idx_frac, trade_size_frac)

The agent trained here is the arbitrageur by default; swap for redeemer by
passing agent_type="redeemer" to control which agent's policy is learned.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..engine.market import MultiVenueMarket
from ..engine.amm import StableswapAMM
from ..engine.redemption import RedemptionChannel
from ..engine.reserve import ReserveModel
from ..agents.arbitrageur import Arbitrageur
from ..agents.redeemer import Redeemer
from ..agents.lp import LPAgent
from ..agents.noise import NoiseTrader
from ..scenarios.schedule import ShockSchedule


class StablecoinEnv(gym.Env):
    """Single-agent Gymnasium env for training arbitrageur or redeemer policies.

    Parameters
    ----------
    scenario : ShockSchedule | None
        Shock schedule for the episode.  None = no shocks.
    agent_type : str
        "arbitrageur" or "redeemer".
    max_steps : int
        Episode length.
    market_kwargs : dict
        Passed to MultiVenueMarket constructor.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ShockSchedule | None = None,
        agent_type: str = "arbitrageur",
        max_steps: int = 200,
        market_kwargs: dict | None = None,
        rng_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.agent_type = agent_type
        self.max_steps = max_steps
        self.market_kwargs = market_kwargs or {}
        self._rng_seed = rng_seed

        # Obs: [price, depeg, reserve_perceived, queue_norm, x0_norm, y0_norm, wealth_norm]
        obs_dim = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: [pool_frac ∈ [0,1], trade_size_frac ∈ [0,1]]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self._market: MultiVenueMarket | None = None
        self._rl_agent: Arbitrageur | Redeemer | None = None
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed if seed is not None else self._rng_seed)

        pools = [StableswapAMM(rng=rng) if False else StableswapAMM()]
        self._market = MultiVenueMarket(
            pools=pools,
            redemption=RedemptionChannel(**self.market_kwargs.get("redemption", {})),
            reserve=ReserveModel(rng=rng, **self.market_kwargs.get("reserve", {})),
            rng=rng,
        )

        # Background agents (non-RL)
        self._noise_traders = [
            NoiseTrader(f"noise_{i}", rng=rng) for i in range(3)
        ]
        self._lp_agents = [LPAgent("lp_0")]

        if self.agent_type == "arbitrageur":
            self._rl_agent = Arbitrageur("rl_arb", policy=None)
        else:
            self._rl_agent = Redeemer("rl_redeemer", policy=None)

        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        shock_events = self.scenario.events_at(self._step_count) if self.scenario else []
        shock = shock_events[0] if shock_events else None

        snap = self._market.step(shock=shock)
        obs_dict = snap

        # RL agent acts with current action
        wealth_before = self._rl_agent.wealth
        self._rl_agent.policy = lambda o: action  # inject action
        self._rl_agent.act(self._market, obs_dict)
        self._rl_agent.policy = None
        wealth_after = self._rl_agent.wealth
        reward = float(wealth_after - wealth_before)

        # Background agents
        for agent in self._noise_traders + self._lp_agents:
            agent.act(self._market, obs_dict)

        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        obs = self._get_obs()
        return obs, reward, terminated, False, snap

    def _get_obs(self) -> np.ndarray:
        if self._market is None:
            return np.zeros(7, dtype=np.float32)
        snap = self._market._snapshot([])
        pool = self._market.pools[0]
        scale = 1_000_000.0
        return np.array([
            snap["mid_price"],
            snap["depeg"],
            snap["reserve_perceived"],
            snap["queue_depth"] / max(self._market.redemption.max_queue or 1, 1),
            pool.x / scale,
            pool.y / scale,
            self._rl_agent.wealth / 100_000.0 if self._rl_agent else 1.0,
        ], dtype=np.float32)
