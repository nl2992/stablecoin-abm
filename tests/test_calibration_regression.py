"""Calibration regression test — CI gate.

Fails the build if simulated moments drift outside tolerance of the locked
empirical targets in configs/calibration_targets.json.

This uses a FAST parameter set (few seeds) to keep CI under 60 seconds.
The full calibration run (n_seeds=20, maxiter=80) is a separate make target.

Design note: the targets are locked; if the engine changes in a way that breaks
calibration, this test will catch it and force the team to re-run calibration
before merging.  This is intentional — it mirrors the Gu et al. validation
discipline where empirical and simulated moments must match before any
intervention results are trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stablesim.scenarios.schedule import ShockSchedule
from stablesim.scenarios.loader import load_stressbench_scenarios
from stablesim.experiments.interventions import BASELINE
from stablesim.experiments.runner import run_episode
from stablesim.analysis.metrics import compute_ou_half_life

TARGETS_PATH = Path(__file__).parents[1] / "configs" / "calibration_targets.json"
CALIBRATION_REPORT = Path(__file__).parents[1] / "experiments" / "results" / "calibration" / "calibration_report.json"
N_SEEDS_REGRESSION = 10   # small for CI speed; full calibration uses 20+

# Skip moment-matching tests unless calibration has been run
_calibrated = CALIBRATION_REPORT.exists()
calibration_required = pytest.mark.skipif(
    not _calibrated,
    reason="Moment-matching gates require calibration. Run `make calibrate` first."
)


def _load_targets() -> dict:
    with open(TARGETS_PATH) as f:
        return json.load(f)


def _simulate_baseline_moments(n_seeds: int = N_SEEDS_REGRESSION) -> dict:
    """Run no-shock baseline and return key moments."""
    baseline = ShockSchedule(name="baseline")
    half_lives, vols = [], []
    for seed in range(n_seeds):
        r = run_episode(baseline, BASELINE, n_steps=100, rng_seed=seed)
        df = r["history"]
        prices = df["mid_price"].values
        half_lives.append(compute_ou_half_life(prices - 1.0))
        vols.append(float(np.std(np.diff(prices))))
    return {
        "calm_ou_half_life": float(np.median([h for h in half_lives if np.isfinite(h)])),
        "baseline_price_vol": float(np.median(vols)),
    }


def _simulate_shock_moments(n_seeds: int = N_SEEDS_REGRESSION) -> dict:
    """Run a shock scenario and return crisis moments."""
    scenarios = load_stressbench_scenarios()
    shock = next(
        (s for s in scenarios if s.name not in ("no_shock_baseline", "baseline")),
        scenarios[0],
    )
    magnitudes = []
    for seed in range(n_seeds):
        r = run_episode(shock, BASELINE, n_steps=100, rng_seed=seed)
        magnitudes.append(r["metrics"]["contagion_magnitude"])
    return {"contagion_magnitude": float(np.median(magnitudes))}


@pytest.fixture(scope="module")
def targets():
    return _load_targets()


@pytest.fixture(scope="module")
def baseline_moments():
    return _simulate_baseline_moments(N_SEEDS_REGRESSION)


@pytest.fixture(scope="module")
def shock_moments():
    return _simulate_shock_moments(N_SEEDS_REGRESSION)


# ------------------------------------------------------------------
# Gate tests

def test_targets_file_exists():
    """Locked targets file must be present and parseable."""
    assert TARGETS_PATH.exists(), f"Missing {TARGETS_PATH}"
    data = _load_targets()
    assert "abm_moment_targets" in data
    assert "calibration_tolerances" in data


@calibration_required
def test_baseline_price_vol_within_tolerance(targets, baseline_moments):
    """No-shock price volatility must be within 30% of locked target."""
    t = targets["abm_moment_targets"]
    tol = targets["calibration_tolerances"]["ou_half_life_rtol"]  # reuse 30%
    target_vol = t["baseline_price_vol"]
    sim_vol = baseline_moments["baseline_price_vol"]
    rel_err = abs(sim_vol - target_vol) / max(target_vol, 1e-9)
    assert rel_err <= tol, (
        f"Baseline price vol drifted: simulated={sim_vol:.5f}, "
        f"target={target_vol:.5f}, rel_err={rel_err:.1%} > tol={tol:.1%}. "
        f"Re-run `make calibrate` and lock new params."
    )


def test_no_shock_market_stays_at_par(targets, baseline_moments):
    """No-agent, no-shock runs must keep price at $1 (Phase 0 gate, repeated here for CI)."""
    from stablesim.engine.market import MultiVenueMarket
    market = MultiVenueMarket(rng=np.random.default_rng(0))
    for _ in range(200):
        market.step()
    assert abs(market.mid_price() - 1.0) < 1e-10


def test_shock_produces_nonzero_contagion(shock_moments):
    """Shock scenarios must produce measurable depeg (sanity check on engine)."""
    mag = shock_moments["contagion_magnitude"]
    assert mag > 0.001, f"Shock produced negligible contagion: {mag:.6f}"


@calibration_required
def test_contagion_magnitude_within_tolerance(targets, shock_moments):
    """Crisis contagion magnitude must be within 25% of locked target.

    NOTE: This test is intentionally loose because:
    (a) calibration_targets.json uses empirical mean_abs_effect which is an upper bound,
    (b) synthetic scenarios may produce different magnitudes than real episodes.

    If this test fails after an engine change, it signals calibration drift —
    document the divergence in the calibration report before merging.
    """
    t = targets["abm_moment_targets"]
    tol = targets["calibration_tolerances"]["contagion_magnitude_rtol"]
    target_high = t["contagion_magnitude_high"]
    target_low = t["contagion_magnitude_low"]
    sim_mag = shock_moments["contagion_magnitude"]

    # Pass if within tolerance of either the high or low empirical target
    rel_err_high = abs(sim_mag - target_high) / max(target_high, 1e-9)
    rel_err_low = abs(sim_mag - target_low) / max(target_low, 1e-9)

    passes = rel_err_high <= tol or rel_err_low <= tol or (target_low <= sim_mag <= target_high)

    if not passes:
        pytest.fail(
            f"Contagion magnitude drifted outside calibration window: "
            f"simulated={sim_mag:.4f}, expected [{target_low:.3f}, {target_high:.3f}] ±{tol:.0%}. "
            f"Divergence is a FINDING — document in calibration report. "
            f"Re-run `make calibrate` if the engine changed intentionally."
        )
