"""Simulated Method of Moments (SMM) calibration.

Tunes the parameter vector θ = [reserve_speed, reserve_vol, arb_min_spread,
noise_trade_prob, noise_trade_size_mean] until the simulated moments match the
empirical targets locked in configs/calibration_targets.json.

Moment targets:
  1. Calm OU half-life   (no-shock baseline, ~3 steps at 5 min/step)
  2. Crisis contagion magnitude  (shock episode, peak |depeg|)
  3. Cross-venue ρ̂      (Pearson correlation of pool prices during shock)
  4. Baseline price vol  (std of Δprice in calm regime)

Loss function: weighted MSE in relative terms.
Optimizer: scipy.optimize.differential_evolution (global, gradient-free).

Usage:
    from stablesim.calibration.smm import SMMCalibrator
    cal = SMMCalibrator()
    best, report = cal.fit(n_seeds=20, maxiter=80)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from ..analysis.metrics import compute_ou_half_life
from ..experiments.interventions import BASELINE
from ..experiments.runner import run_episode
from ..scenarios.loader import load_stressbench_scenarios
from ..scenarios.schedule import ShockSchedule

# ---------------------------------------------------------------------------
# Parameter space

PARAM_NAMES = [
    "reserve_speed",       # OU kappa for backing ratio
    "reserve_vol",         # OU sigma
    "arb_min_spread",      # min price spread to trigger arbitrage
    "noise_trade_prob",    # Bernoulli prob each step
    "noise_trade_size",    # mean trade size (USD)
]

PARAM_BOUNDS = [
    (0.01, 0.30),          # reserve_speed
    (0.005, 0.05),         # reserve_vol
    (0.0005, 0.01),        # arb_min_spread
    (0.10, 0.70),          # noise_trade_prob
    (500.0, 10_000.0),     # noise_trade_size
]

# ---------------------------------------------------------------------------

def _load_targets(targets_path: str | Path | None = None) -> dict:
    if targets_path is None:
        targets_path = Path(__file__).parents[4] / "configs" / "calibration_targets.json"
    with open(targets_path) as f:
        return json.load(f)


def _params_to_dict(params: np.ndarray) -> dict:
    return dict(zip(PARAM_NAMES, params.tolist()))


def _simulate_moments(
    params: np.ndarray,
    shock_scenario: ShockSchedule,
    baseline_scenario: ShockSchedule,
    n_seeds: int,
    n_steps: int,
) -> dict:
    """Simulate one parameter configuration and return moment estimates."""
    p = _params_to_dict(params)

    calm_half_lives, calm_vols, crisis_magnitudes, cross_rhos = [], [], [], []

    for seed in range(n_seeds):
        # --- Calm run (no shock) ---
        r_calm = run_episode(baseline_scenario, BASELINE, n_steps=n_steps, rng_seed=seed)
        df_c = r_calm["history"]
        prices_c = df_c["mid_price"].values
        calm_half_lives.append(compute_ou_half_life(prices_c - 1.0))
        calm_vols.append(float(np.std(np.diff(prices_c))))

        # --- Crisis run ---
        r_crisis = run_episode(shock_scenario, BASELINE, n_steps=n_steps, rng_seed=seed)
        df_k = r_crisis["history"]
        crisis_magnitudes.append(float(df_k["depeg"].abs().max()))

        # Cross-venue ρ̂: correlation between pool prices if ≥ 2 pools
        pool_states = df_k.get("pool_states", None)
        if pool_states is not None and len(df_k) > 5:
            try:
                p0 = [s[0]["price"] for s in df_k["pool_states"]]
                p1 = [s[-1]["price"] for s in df_k["pool_states"]]
                rho = float(np.corrcoef(p0, p1)[0, 1])
                cross_rhos.append(rho if np.isfinite(rho) else 0.0)
            except Exception:
                cross_rhos.append(0.0)
        else:
            cross_rhos.append(0.0)

    def _safe_median(lst):
        vals = [v for v in lst if np.isfinite(v)]
        return float(np.median(vals)) if vals else 0.0

    return {
        "calm_ou_half_life": _safe_median(calm_half_lives),
        "baseline_price_vol": _safe_median(calm_vols),
        "contagion_magnitude": _safe_median(crisis_magnitudes),
        "cross_venue_rho": _safe_median(cross_rhos),
    }


class SMMCalibrator:
    """Simulated Method of Moments calibrator.

    Parameters
    ----------
    targets_path : str | Path | None
        Path to calibration_targets.json.  Defaults to configs/calibration_targets.json.
    event_name : str
        Which event to calibrate against (must exist in targets JSON).
    n_steps : int
        Steps per simulated episode.
    """

    def __init__(
        self,
        targets_path: str | Path | None = None,
        event_name: str = "usdc_svb_2023",
        n_steps: int = 150,
    ) -> None:
        self.targets_json = _load_targets(targets_path)
        self.abm_targets = self.targets_json["abm_moment_targets"]
        self.tolerances = self.targets_json["calibration_tolerances"]
        self.event_name = event_name
        self.n_steps = n_steps

        scenarios = load_stressbench_scenarios()
        self._shock = next(
            (s for s in scenarios if s.name not in ("no_shock_baseline", "baseline")),
            scenarios[0],
        )
        self._baseline = next(
            (s for s in scenarios if s.name in ("no_shock_baseline", "baseline")),
            ShockSchedule(name="baseline"),
        )

        # Moment weights (relative importance in loss)
        self._weights = {
            "calm_ou_half_life": 1.0,
            "baseline_price_vol": 2.0,
            "contagion_magnitude": 2.0,
            "cross_venue_rho": 0.5,
        }

    def _loss(self, params: np.ndarray, n_seeds: int) -> float:
        moments = _simulate_moments(
            params, self._shock, self._baseline, n_seeds, self.n_steps
        )
        t = self.abm_targets
        losses = {
            "calm_ou_half_life": (
                (moments["calm_ou_half_life"] - t["calm_ou_half_life_steps"])
                / max(t["calm_ou_half_life_steps"], 1e-9)
            ) ** 2,
            "baseline_price_vol": (
                (moments["baseline_price_vol"] - t["baseline_price_vol"])
                / max(t["baseline_price_vol"], 1e-9)
            ) ** 2,
            "contagion_magnitude": (
                (moments["contagion_magnitude"] - t["contagion_magnitude_high"])
                / max(t["contagion_magnitude_high"], 1e-9)
            ) ** 2,
            "cross_venue_rho": (
                (moments["cross_venue_rho"] - t["cross_venue_rho_crisis"])
                / max(t["cross_venue_rho_crisis"], 1e-9)
            ) ** 2,
        }
        return sum(self._weights[k] * v for k, v in losses.items())

    def fit(
        self,
        n_seeds: int = 20,
        maxiter: int = 80,
        popsize: int = 10,
        verbose: bool = True,
    ) -> tuple[dict[str, float], "CalibrationReport"]:
        """Run differential evolution.

        Returns
        -------
        best_params : dict
        report : CalibrationReport
        """
        from .report import CalibrationReport

        t0 = time.time()
        result = differential_evolution(
            lambda p: self._loss(p, n_seeds=n_seeds),
            PARAM_BOUNDS,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-4,
            seed=42,
            disp=verbose,
        )
        elapsed = time.time() - t0

        best_params = _params_to_dict(result.x)
        final_moments = _simulate_moments(
            result.x, self._shock, self._baseline, n_seeds=n_seeds * 2, n_steps=self.n_steps
        )

        report = CalibrationReport(
            best_params=best_params,
            simulated_moments=final_moments,
            targets=self.abm_targets,
            tolerances=self.tolerances,
            optimizer_result=result,
            elapsed_seconds=elapsed,
        )
        return best_params, report
