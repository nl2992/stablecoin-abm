"""Multi-seed counterfactual runner.

For each hub node, runs N_SEEDS paired episodes (baseline vs. intervened) and
estimates the causal effect as:

    Δcontagion_h = E[baseline_magnitude] − E[intervened_magnitude]

with standard error from the variance of the difference:

    SE = sqrt(Var[baseline]/N + Var[intervened]/N)

Mirroring the 40k-sim averaging discipline from Gu et al. (2023):
single runs are meaningless; report distributions and t-statistics.

Usage:
    from stablesim.counterfactual.runner import run_counterfactual
    result = run_counterfactual(hub_node, scenario, n_seeds=40)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..agents.arbitrageur import Arbitrageur
from ..agents.issuer import IssuerAgent
from ..agents.lp import LPAgent
from ..agents.noise import NoiseTrader
from ..agents.redeemer import Redeemer
from ..engine.amm import StableswapAMM
from ..engine.market import MultiVenueMarket
from ..engine.redemption import RedemptionChannel
from ..engine.reserve import ReserveModel
from ..experiments.interventions import BASELINE
from ..scenarios.schedule import ShockSchedule
from .hub_interventions import HubNode, HubInterventionParams, apply_hub_intervention, build_intervention_params

if TYPE_CHECKING:
    from .hub_interventions import InterventionType

N_SEEDS_DEFAULT = 40     # Match Gu et al. discipline
N_SEEDS_FAST = 8         # For CI / quick checks


@dataclass
class CounterfactualResult:
    """Per-hub counterfactual outcome with standard errors.

    Attributes
    ----------
    hub : HubNode
    baseline_magnitudes : np.ndarray
        Peak |depeg| per seed in no-intervention runs.
    intervened_magnitudes : np.ndarray
        Peak |depeg| per seed in intervened runs.
    delta_contagion : float
        E[baseline] − E[intervened].  Positive = intervention reduced contagion.
    se : float
        Standard error of delta_contagion.
    t_stat : float
        t = delta_contagion / se.
    p_value_one_sided : float
        One-sided p-value (H1: intervention reduces contagion).
    """

    hub: HubNode
    baseline_magnitudes: np.ndarray
    intervened_magnitudes: np.ndarray
    intervention_type: str
    n_seeds: int
    n_steps: int
    scenario_name: str

    # Derived on __post_init__
    delta_contagion: float = field(init=False)
    se: float = field(init=False)
    t_stat: float = field(init=False)
    p_value_one_sided: float = field(init=False)
    baseline_mean: float = field(init=False)
    intervened_mean: float = field(init=False)

    def __post_init__(self) -> None:
        from scipy import stats

        b = self.baseline_magnitudes
        i = self.intervened_magnitudes
        self.baseline_mean = float(np.mean(b))
        self.intervened_mean = float(np.mean(i))
        self.delta_contagion = self.baseline_mean - self.intervened_mean
        self.se = float(np.sqrt(np.var(b, ddof=1) / len(b) + np.var(i, ddof=1) / len(i)))
        self.t_stat = self.delta_contagion / max(self.se, 1e-9)
        # One-sided: H1 delta > 0
        self.p_value_one_sided = float(stats.t.sf(self.t_stat, df=2 * (len(b) - 1)))

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value_one_sided < alpha

    def summary_dict(self) -> dict:
        return {
            "node_id": self.hub.node_id,
            "node_type": self.hub.node_type.value,
            "role": self.hub.role,
            "predicted_importance": self.hub.predicted_importance,
            "intervention_type": self.intervention_type,
            "baseline_mean": self.baseline_mean,
            "intervened_mean": self.intervened_mean,
            "delta_contagion": self.delta_contagion,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value_one_sided": self.p_value_one_sided,
            "significant_p05": self.is_significant(0.05),
            "n_seeds": self.n_seeds,
            "scenario": self.scenario_name,
        }


def _build_market(rng: np.random.Generator, redemption_kwargs: dict = {}, reserve_kwargs: dict = {}) -> MultiVenueMarket:
    return MultiVenueMarket(
        pools=[StableswapAMM(), StableswapAMM(reserves=(900_000, 1_100_000), amp=100)],
        redemption=RedemptionChannel(**redemption_kwargs),
        reserve=ReserveModel(rng=rng, **reserve_kwargs),
        rng=rng,
    )


def _build_agents(rng: np.random.Generator) -> list:
    return [
        Arbitrageur("arb_0"), Arbitrageur("arb_1"),
        Redeemer("red_0"), Redeemer("red_1"),
        LPAgent("lp_0"), LPAgent("lp_1"),
        IssuerAgent(),
        NoiseTrader("noise_0", rng=rng),
        NoiseTrader("noise_1", rng=rng),
        NoiseTrader("noise_2", rng=rng),
    ]


def _run_single(
    scenario: ShockSchedule,
    n_steps: int,
    rng_seed: int,
    hub: HubNode | None = None,
    params: HubInterventionParams | None = None,
) -> dict:
    """Run one episode (baseline if hub is None, intervened otherwise)."""
    rng = np.random.default_rng(rng_seed)
    market = _build_market(rng)
    agents = _build_agents(rng)

    if hub is not None:
        apply_hub_intervention(market, hub, params, current_step=0)

    for step in range(n_steps):
        shock_events = scenario.events_at(step)
        shock = shock_events[0] if shock_events else None
        snap = market.step(shock=shock)
        for agent in agents:
            agent.act(market, snap)

    df = market.history_df()
    return {
        "contagion_magnitude": float(df["depeg"].abs().max()),
        "ou_half_life": _ou_hl(df["depeg"].values),
        "welfare": {type(a).__name__: a.cumulative_pnl for a in agents},
    }


def _ou_hl(depeg: np.ndarray) -> float:
    from ..analysis.metrics import compute_ou_half_life
    return compute_ou_half_life(depeg)


def run_counterfactual(
    hub: HubNode,
    scenario: ShockSchedule,
    n_seeds: int = N_SEEDS_DEFAULT,
    n_steps: int = 150,
    intervention_type: "InterventionType | None" = None,
    base_seed: int = 0,
    verbose: bool = False,
) -> CounterfactualResult:
    """Run paired baseline/intervened episodes for one hub.

    Parameters
    ----------
    hub : HubNode
        The hub to intervene on.
    scenario : ShockSchedule
        Exogenous shock schedule for both runs.
    n_seeds : int
        Number of independent seeds.  Use 40+ for publishable estimates.
    intervention_type : InterventionType | None
        Override the default intervention for this hub type.
    """
    params = build_intervention_params(hub, intervention_type)
    itype_str = params.intervention_type.value

    baseline_mags, intervened_mags = [], []

    for i in range(n_seeds):
        seed = base_seed + i
        if verbose and i % 10 == 0:
            print(f"  hub={hub.node_id} seed={seed}/{n_seeds}")

        b = _run_single(scenario, n_steps, seed, hub=None, params=None)
        iv = _run_single(scenario, n_steps, seed, hub=hub, params=params)

        baseline_mags.append(b["contagion_magnitude"])
        intervened_mags.append(iv["contagion_magnitude"])

    return CounterfactualResult(
        hub=hub,
        baseline_magnitudes=np.array(baseline_mags),
        intervened_magnitudes=np.array(intervened_mags),
        intervention_type=itype_str,
        n_seeds=n_seeds,
        n_steps=n_steps,
        scenario_name=scenario.name,
    )


def run_all_hubs(
    hubs: list[HubNode],
    scenario: ShockSchedule,
    n_seeds: int = N_SEEDS_DEFAULT,
    n_steps: int = 150,
    verbose: bool = True,
) -> list[CounterfactualResult]:
    """Run counterfactual for every hub in the list."""
    results = []
    for hub in hubs:
        if verbose:
            print(f"Running counterfactual: {hub.node_id} ({hub.node_type.value})")
        r = run_counterfactual(hub, scenario, n_seeds=n_seeds, n_steps=n_steps, verbose=verbose)
        results.append(r)
    return results
