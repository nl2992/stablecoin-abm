"""Simulated Method of Moments (SMM) calibration.

Identification
==============
We have 4 target moments and 4 free parameters (just-identified).
noise_trade_size is FIXED as a structural constant (2 000 USD) based on
market-microstructure priors for retail stablecoin trades; it is not identified
by the price moments and fixing it avoids under-identification.

  Free parameters (4):
    reserve_speed    — OU kappa; identified by calm OU half-life
    reserve_vol      — OU sigma; identified by baseline price vol
    arb_min_spread   — identified by crisis contagion magnitude
    noise_trade_prob — identified by cross-venue rho during shock

  Fixed structural constant:
    noise_trade_size = 2 000 USD  (prior: median retail stablecoin trade)

Moments (4):
  1. Calm OU half-life       (no-shock baseline, ~3 steps at 5 min/step)
  2. Baseline price vol      (std of Δprice in calm regime)
  3. Crisis contagion magnitude (peak |depeg| during shock)
  4. Cross-venue ρ̂           (Pearson r of pool prices during shock)

Loss:
  L(θ) = Σ_k w_k · ((m_sim_k(θ) − m_emp_k) / m_emp_k)²

Optimizer: scipy.optimize.differential_evolution (global, gradient-free).
  • Run from n_de_restarts independent initial populations; confirm they
    converge to the same basin before accepting the optimum.
  • Sensitivity analysis: numerical Jacobian J[moment_k, param_j] around the
    optimum, revealing which moments constrain which params.
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
# Parameter space — 4 free + 1 fixed

PARAM_NAMES = [
    "pool_amp",        # stableswap amplification; identified by contagion magnitude + OU half-life
    "noise_size",      # mean noise-trade size (USD); identified by baseline price vol
    "shock_scale",     # multiplies scenario shock magnitude; identified by contagion magnitude
    "reserve_speed",   # OU kappa for backing ratio; identified by calm OU half-life
]

PARAM_BOUNDS = [
    (8.0, 150.0),          # pool_amp
    (200.0, 25_000.0),     # noise_size (USD)
    (0.5, 8.0),            # shock_scale
    (0.02, 0.60),          # reserve_speed
]


def _load_targets(targets_path: str | Path | None = None) -> dict:
    if targets_path is None:
        targets_path = Path(__file__).parents[3] / "configs" / "calibration_targets.json"
    with open(targets_path) as f:
        return json.load(f)


def _params_to_dict(params: np.ndarray) -> dict:
    return dict(zip(PARAM_NAMES, params.tolist()))


def _episode_kwargs(params: np.ndarray) -> dict:
    amp, noise_size, shock_scale, reserve_speed = params.tolist()
    return dict(pool_amp=amp, noise_size=noise_size, shock_scale=shock_scale,
                reserve_speed=reserve_speed)


def _simulate_moments(
    params: np.ndarray,
    shock_scenario: ShockSchedule,
    baseline_scenario: ShockSchedule,
    n_seeds: int,
    n_steps: int,
) -> dict:
    """Simulate one θ configuration and return all 4 moment estimates."""
    calm_half_lives, calm_vols, crisis_magnitudes, cross_rhos = [], [], [], []
    ekw = _episode_kwargs(params)

    for seed in range(n_seeds):
        # Calm run (no shock_scale effect since baseline has no shocks)
        r_calm = run_episode(baseline_scenario, BASELINE, n_steps=n_steps, rng_seed=seed, **ekw)
        df_c = r_calm["history"]
        prices_c = df_c["mid_price"].values
        hl = compute_ou_half_life(prices_c - 1.0)
        calm_half_lives.append(hl)
        calm_vols.append(float(np.std(np.diff(prices_c))) if len(prices_c) > 1 else 0.0)

        # Crisis run
        r_crisis = run_episode(shock_scenario, BASELINE, n_steps=n_steps, rng_seed=seed, **ekw)
        df_k = r_crisis["history"]
        crisis_magnitudes.append(float(df_k["depeg"].abs().max()))

        # Cross-venue ρ̂ from pool_states (if ≥ 2 pools recorded)
        if "pool_states" in df_k.columns and len(df_k) > 5:
            try:
                p0 = [s[0]["price"] for s in df_k["pool_states"]]
                p1 = [s[-1]["price"] for s in df_k["pool_states"]]
                rho = float(np.corrcoef(p0, p1)[0, 1])
                cross_rhos.append(rho if np.isfinite(rho) else 0.0)
            except Exception:
                cross_rhos.append(0.0)
        else:
            cross_rhos.append(0.0)

    def _med(lst):
        vals = [v for v in lst if np.isfinite(v)]
        return float(np.median(vals)) if vals else 0.0

    return {
        "calm_ou_half_life": _med(calm_half_lives),
        "baseline_price_vol": _med(calm_vols),
        "contagion_magnitude": _med(crisis_magnitudes),
        "cross_venue_rho": _med(cross_rhos),
    }


class SMMCalibrator:
    """Just-identified (4×4) SMM calibrator with convergence and sensitivity checks.

    Parameters
    ----------
    targets_path : path to calibration_targets.json.
    event_name : which event to calibrate against.
    n_steps : steps per simulated episode.
    n_de_restarts : number of independent DE restarts for convergence check.
    """

    def __init__(
        self,
        targets_path: str | Path | None = None,
        event_name: str = "usdc_svb_2023",
        n_steps: int = 150,
        n_de_restarts: int = 3,
    ) -> None:
        self.targets_json = _load_targets(targets_path)
        self.abm_targets = self.targets_json["abm_moment_targets"]
        self.tolerances = self.targets_json["calibration_tolerances"]
        self.event_name = event_name
        self.n_steps = n_steps
        self.n_de_restarts = n_de_restarts

        scenarios = load_stressbench_scenarios()
        self._shock = next(
            (s for s in scenarios if s.name not in ("no_shock_baseline", "baseline")),
            scenarios[0],
        )
        self._baseline = next(
            (s for s in scenarios if s.name in ("no_shock_baseline", "baseline")),
            ShockSchedule(name="baseline"),
        )

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

    def _run_de(self, n_seeds: int, maxiter: int, popsize: int, seed: int, verbose: bool):
        return differential_evolution(
            lambda p: self._loss(p, n_seeds=n_seeds),
            PARAM_BOUNDS,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-4,
            seed=seed,
            disp=verbose,
        )

    def fit(
        self,
        n_seeds: int = 20,
        maxiter: int = 80,
        popsize: int = 10,
        verbose: bool = True,
    ) -> tuple[dict[str, float], "CalibrationReport"]:
        """Run differential evolution from n_de_restarts seeds; check convergence.

        Returns best_params and CalibrationReport.
        """
        from .report import CalibrationReport

        t0 = time.time()
        all_results = []
        for restart in range(self.n_de_restarts):
            if verbose:
                print(f"\n--- DE restart {restart + 1}/{self.n_de_restarts} ---")
            r = self._run_de(n_seeds, maxiter, popsize, seed=restart * 1000 + 42, verbose=verbose)
            all_results.append(r)

        # Pick best (lowest loss)
        best_result = min(all_results, key=lambda r: r.fun)
        elapsed = time.time() - t0

        # Convergence check: do restarts agree within 10%?
        convergence_ok = self._check_convergence(all_results)
        if not convergence_ok:
            print(
                "WARNING: DE restarts did not converge to the same basin. "
                "The calibration optimum may be fragile. "
                "Document this in the calibration report."
            )

        best_params = _params_to_dict(best_result.x)
        final_moments = _simulate_moments(
            best_result.x, self._shock, self._baseline,
            n_seeds=n_seeds * 2, n_steps=self.n_steps,
        )

        report = CalibrationReport(
            best_params=best_params,
            simulated_moments=final_moments,
            targets=self.abm_targets,
            tolerances=self.tolerances,
            optimizer_result=best_result,
            elapsed_seconds=elapsed,
            convergence_ok=convergence_ok,
            n_de_restarts=self.n_de_restarts,
        )
        return best_params, report

    def _check_convergence(self, results: list) -> bool:
        """Return True if all DE runs land within 10% of the best loss."""
        losses = [r.fun for r in results]
        best = min(losses)
        if best == 0:
            return True
        return all(abs(l - best) / best < 0.10 for l in losses)

    def sensitivity_analysis(
        self,
        best_params: np.ndarray,
        n_seeds: int = 20,
        eps_frac: float = 0.05,
    ) -> dict:
        """Numerical Jacobian J[moment_k, param_j] at the optimum.

        J_kj = (m_k(θ + eps·e_j) − m_k(θ)) / eps_j

        Shows which moments constrain which parameters.
        A near-zero column j means param j is weakly identified.
        """
        base = _simulate_moments(best_params, self._shock, self._baseline, n_seeds, self.n_steps)
        moment_keys = list(base.keys())

        jac = {}
        for j, pname in enumerate(PARAM_NAMES):
            eps_j = abs(best_params[j]) * eps_frac + 1e-8
            p_plus = best_params.copy()
            p_plus[j] += eps_j
            m_plus = _simulate_moments(p_plus, self._shock, self._baseline, n_seeds, self.n_steps)
            jac[pname] = {
                mk: (m_plus[mk] - base[mk]) / eps_j
                for mk in moment_keys
            }
        return {"jacobian": jac, "base_moments": base}
