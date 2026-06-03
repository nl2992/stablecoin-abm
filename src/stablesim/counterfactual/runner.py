"""Multi-seed counterfactual runner — paired design with correct inference.

Three key changes from the original:
  1. UNIFORM ABLATION as primary treatment (intervention_spec.py).
     Type-specific knobs are secondary only.  All hubs get the same alpha dose,
     making delta-C comparable across the node universe.

  2. PAIRED standard error (inference.py).
     Baseline and intervened use the SAME seed, so the variance of the within-
     seed difference drives the SE.  Using a two-sample SE was wrong.

  3. BH FDR correction via summarize_sweep().
     Never claim significance on raw p-values across multiple hubs.

Usage:
    from stablesim.counterfactual.runner import run_all_hubs, N_SEEDS_DEFAULT
    results = run_all_hubs(hubs, scenario, n_seeds=N_SEEDS_DEFAULT)
    # results: list[PairedResult], FDR-corrected, sorted by delta_c desc
"""

from __future__ import annotations

import numpy as np

from ..agents.arbitrageur import Arbitrageur
from ..agents.issuer import IssuerAgent
from ..agents.lp import LPAgent
from ..agents.noise import NoiseTrader
from ..agents.redeemer import Redeemer
from ..engine.amm import StableswapAMM
from ..engine.market import MultiVenueMarket
from ..engine.redemption import RedemptionChannel
from ..engine.reserve import ReserveModel
from ..scenarios.schedule import ShockSchedule
from .ablation_adapter import apply_ablation
from .hub_interventions import HubNode
from .inference import PairedResult, required_n, summarize_sweep
from .intervention_spec import DOSE_GRID, primary_intervention

N_SEEDS_DEFAULT = 40    # justify with required_n() on calibrated noise
N_SEEDS_FAST = 8        # CI / smoke tests


# --------------------------------------------------------------------------- #
# Market and agent factories — deterministic given seed                       #
# --------------------------------------------------------------------------- #

def _make_market(rng: np.random.Generator) -> MultiVenueMarket:
    return MultiVenueMarket(
        pools=[
            StableswapAMM(),
            StableswapAMM(reserves=(900_000, 1_100_000), amp=100),
        ],
        redemption=RedemptionChannel(),
        reserve=ReserveModel(rng=rng),
        rng=rng,
    )


def _make_agents(rng: np.random.Generator) -> list:
    return [
        Arbitrageur("arb_0"), Arbitrageur("arb_1"),
        Redeemer("red_0"), Redeemer("red_1"),
        LPAgent("lp_0"), LPAgent("lp_1"),
        IssuerAgent(),
        NoiseTrader("noise_0", rng=rng),
        NoiseTrader("noise_1", rng=rng),
        NoiseTrader("noise_2", rng=rng),
    ]


def _run_single_episode(
    scenario: ShockSchedule,
    n_steps: int,
    seed: int,
    hub: HubNode | None,
    alpha: float,
) -> float:
    """Run one episode and return contagion magnitude.

    If hub is not None, applies uniform ablation at the given alpha BEFORE
    any shocks or agent actions.  Alpha=0.0 is a guaranteed no-op so the
    baseline arm and the control arm use identical code paths.
    """
    rng = np.random.default_rng(seed)
    market = _make_market(rng)
    agents = _make_agents(rng)

    # Apply ablation (or no-op baseline) — must come before any steps
    if hub is not None:
        apply_ablation(market, hub, alpha, agents)

    for step in range(n_steps):
        shock_events = scenario.events_at(step)
        shock = shock_events[0] if shock_events else None
        snap = market.step(shock=shock)
        for agent in agents:
            agent.act(market, snap)

    df = market.history_df()
    return float(df["depeg"].abs().max())


# --------------------------------------------------------------------------- #
# Per-hub runner                                                               #
# --------------------------------------------------------------------------- #

def run_hub_paired(
    hub: HubNode,
    scenario: ShockSchedule,
    *,
    alpha: float = 1.0,
    n_seeds: int = N_SEEDS_DEFAULT,
    n_steps: int = 150,
    base_seed: int = 0,
    verbose: bool = False,
) -> tuple[list[float], list[float]]:
    """Collect paired (baseline, intervened) contagion magnitudes.

    CRITICAL: baseline and intervened use the SAME seed so they are paired.
    This is what makes the paired SE valid.
    """
    baseline_C: list[float] = []
    intervened_C: list[float] = []

    for i in range(n_seeds):
        seed = base_seed + i
        if verbose and i % 10 == 0:
            print(f"  {hub.node_id}  seed {seed}/{base_seed + n_seeds - 1}")

        # Same seed for both arms -- the paired design
        b = _run_single_episode(scenario, n_steps, seed, hub=None, alpha=0.0)
        c = _run_single_episode(scenario, n_steps, seed, hub=hub, alpha=alpha)

        baseline_C.append(b)
        intervened_C.append(c)

    return baseline_C, intervened_C


# --------------------------------------------------------------------------- #
# Dose-response check                                                          #
# --------------------------------------------------------------------------- #

def run_dose_response(
    hub: HubNode,
    scenario: ShockSchedule,
    *,
    n_seeds: int = 20,
    n_steps: int = 150,
    alphas: tuple[float, ...] = DOSE_GRID,
    base_seed: int = 0,
) -> dict[float, list[float]]:
    """Run episode at each dose alpha for the hub.

    Used to verify monotone dose-response before trusting the headline ranking.
    Each alpha shares seeds with the alpha=0 control so the comparison is paired.
    """
    results: dict[float, list[float]] = {}
    for alpha in alphas:
        mags = []
        for i in range(n_seeds):
            seed = base_seed + i
            m = _run_single_episode(scenario, n_steps, seed, hub=hub, alpha=alpha)
            mags.append(m)
        results[alpha] = mags
    return results


# --------------------------------------------------------------------------- #
# Sweep over all hubs                                                          #
# --------------------------------------------------------------------------- #

def run_all_hubs(
    hubs: list[HubNode],
    scenario: ShockSchedule,
    *,
    alpha: float = 1.0,
    n_seeds: int = N_SEEDS_DEFAULT,
    n_steps: int = 150,
    fdr: float = 0.05,
    verbose: bool = True,
) -> list[PairedResult]:
    """Run paired counterfactual for every hub, return FDR-corrected results.

    Returns
    -------
    list[PairedResult] sorted by delta_c descending (largest causal effect first).
    Each result has q_value and significant_fdr set.
    """
    per_hub: dict[str, tuple[list[float], list[float]]] = {}

    for hub in hubs:
        if verbose:
            print(f"Counterfactual: {hub.node_id} ({hub.node_type.value})")
        b, c = run_hub_paired(hub, scenario, alpha=alpha, n_seeds=n_seeds,
                              n_steps=n_steps, verbose=verbose)
        per_hub[hub.node_id] = (b, c)

    return summarize_sweep(per_hub, fdr=fdr, seed=1)


def power_check(
    pilot_hub: HubNode,
    pilot_scenario: ShockSchedule,
    target_effect: float,
    *,
    n_pilot: int = 10,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Run a pilot to estimate SD of paired differences, then compute required_n.

    Call this BEFORE the full sweep to justify N.  If required_n >> N_SEEDS_DEFAULT,
    the headline test is underpowered and you should document that.

    Returns
    -------
    n_required : int
    """
    import numpy as np
    scenario = pilot_scenario
    b, c = run_hub_paired(pilot_hub, scenario, alpha=1.0, n_seeds=n_pilot)
    d = np.array(b) - np.array(c)
    sd_d = float(d.std(ddof=1))
    n_req = required_n(sd_d, target_effect, alpha=alpha, power=power)
    print(
        f"Pilot (n={n_pilot}): sd_d={sd_d:.4f}, "
        f"to detect δC={target_effect:.3f} at {power:.0%} power need N={n_req} "
        f"(planned: {N_SEEDS_DEFAULT})"
    )
    if n_req > N_SEEDS_DEFAULT:
        print(
            f"  WARNING: planned N={N_SEEDS_DEFAULT} is underpowered. "
            f"Either increase seeds or flag non-significant results as 'underpowered', not 'null'."
        )
    return n_req
