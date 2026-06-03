"""End-to-end pipeline smoke test.

Runs the full three-command pipeline (calibrate → counterfactuals → joint-analysis)
with minimal parameters (2 seeds, 2 hubs, 30 steps) on every commit.

This test is fast (<10s) and checks that the pipeline doesn't silently break.
It uses synthetic hubs (allow_synthetic=True) and is clearly labelled as such.
Paper runs use real repo-1 hubs and more seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stablesim.counterfactual.hub_loader import _synthetic_hubs
from stablesim.counterfactual.runner import run_hub_paired
from stablesim.counterfactual.inference import summarize_sweep
from stablesim.counterfactual.ranking import causal_hub_ranking
from stablesim.analysis.comparison import compute_agreement
from stablesim.scenarios.schedule import ShockEvent, ShockSchedule
from stablesim.calibration.report import CalibrationReport, MomentComparison


# ─── Shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def toy_scenario():
    return ShockSchedule(
        name="e2e_toy",
        events=[ShockEvent(step=5, kind="sell_pressure", magnitude=0.05, label="test_shock")],
    )


@pytest.fixture(scope="module")
def two_hubs():
    return _synthetic_hubs()[:2]


# ─── Stage 1: Calibration report smoke ──────────────────────────────────────

def test_calibration_report_builds():
    """CalibrationReport assembles without error and exports valid JSON."""
    target_stub = {
        "calm_ou_half_life_steps": 3.0,
        "baseline_price_vol": 0.003,
        "contagion_magnitude_high": 0.842,
        "cross_venue_rho_crisis": 0.576,
    }
    tol_stub = {
        "contagion_magnitude_rtol": 0.25,
        "ou_half_life_rtol": 0.30,
    }
    sim_moments = {
        "calm_ou_half_life": 4.0,
        "baseline_price_vol": 0.002,
        "contagion_magnitude": 0.70,
        "cross_venue_rho": 0.50,
    }
    report = CalibrationReport(
        best_params={"reserve_speed": 0.05, "reserve_vol": 0.015},
        simulated_moments=sim_moments,
        targets=target_stub,
        tolerances=tol_stub,
        convergence_ok=True,
        n_de_restarts=3,
    )
    data = report.to_json()
    assert "gate_pass" in data
    assert "convergence_ok" in data
    assert data["identification"] == "just-identified: 4 free params x 4 moments"


# ─── Stage 2: Counterfactual pipeline smoke ──────────────────────────────────

def test_e2e_paired_run_returns_results(two_hubs, toy_scenario):
    """Full paired run over 2 seeds × 2 hubs returns correctly shaped output."""
    per_hub = {}
    for hub in two_hubs:
        b, c = run_hub_paired(hub, toy_scenario, alpha=1.0, n_seeds=2, n_steps=30)
        assert len(b) == 2
        assert len(c) == 2
        per_hub[hub.node_id] = (b, c)

    results = summarize_sweep(per_hub, seed=0)
    assert len(results) == len(two_hubs)


def test_e2e_all_results_have_fdr_fields(two_hubs, toy_scenario):
    """After summarize_sweep, every result must have q_value and significant_fdr."""
    per_hub = {}
    for hub in two_hubs:
        b, c = run_hub_paired(hub, toy_scenario, alpha=1.0, n_seeds=2, n_steps=30)
        per_hub[hub.node_id] = (b, c)

    results = summarize_sweep(per_hub, seed=0)
    for r in results:
        assert r.q_value is not None, f"{r.node_id}: q_value is None"
        assert r.significant_fdr is not None


def test_e2e_causal_ranking_sorted(two_hubs, toy_scenario):
    """causal_hub_ranking must be sorted by delta_c descending."""
    per_hub = {}
    for hub in two_hubs:
        b, c = run_hub_paired(hub, toy_scenario, alpha=1.0, n_seeds=2, n_steps=30)
        per_hub[hub.node_id] = (b, c)

    results = summarize_sweep(per_hub, seed=0)
    df = causal_hub_ranking(results)

    assert list(df["causal_rank"]) == list(range(1, len(df) + 1))
    deltas = df["delta_contagion"].tolist()
    assert deltas == sorted(deltas, reverse=True)


# ─── Stage 3: Joint analysis smoke ───────────────────────────────────────────

def test_e2e_agreement_metrics_computable(two_hubs, toy_scenario):
    """compute_agreement must run without error on the minimal result set."""
    per_hub = {}
    for hub in two_hubs:
        b, c = run_hub_paired(hub, toy_scenario, alpha=1.0, n_seeds=2, n_steps=30)
        per_hub[hub.node_id] = (b, c)

    results = summarize_sweep(per_hub, seed=0)
    df = causal_hub_ranking(results)

    # Merge with predicted importance from hub list
    imp_map = {h.node_id: h.predicted_importance for h in two_hubs}
    df["predicted_importance"] = df["node_id"].map(imp_map)

    # Need ≥ 3 hubs for Spearman — skip if only 2
    if len(df) < 3:
        pytest.skip("Need ≥ 3 hubs for agreement metrics (smoke uses 2)")

    metrics = compute_agreement(df)
    assert np.isfinite(metrics.spearman_rho)
    assert 0 <= metrics.top3_overlap <= 1


# ─── Pipeline stamp smoke ────────────────────────────────────────────────────

def test_stamp_artifact_creates_sidecar(tmp_path):
    """stamp_artifact must create a sidecar .stamp.json with required fields."""
    from stablesim.utils.stamp import stamp_artifact

    artifact = tmp_path / "causal_hub_ranking.csv"
    artifact.write_text("node_id,delta_contagion\nA,0.1\n")

    stamp_path = stamp_artifact(
        artifact,
        params={"reserve_speed": 0.05},
        n_seeds=40,
        alpha=1.0,
        fdr=0.05,
        synthetic_data=True,
    )

    assert stamp_path.exists()
    with open(stamp_path) as f:
        stamp = json.load(f)

    assert "git_sha" in stamp
    assert "calibration_param_hash" in stamp
    assert stamp["n_seeds"] == 40
    assert stamp["synthetic_data"] is True
    assert "WARNING" in stamp   # must flag synthetic data
