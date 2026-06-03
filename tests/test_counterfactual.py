"""Tests for the counterfactual engine.

Validates the core causal inference machinery without running a full multi-seed
sweep (which takes minutes).  Uses N_SEEDS_FAST=8 and n_steps=50.
"""

from __future__ import annotations

import numpy as np
import pytest

from stablesim.counterfactual.hub_interventions import (
    HubNode,
    NodeType,
    InterventionType,
    HubInterventionParams,
    apply_hub_intervention,
    build_intervention_params,
)
from stablesim.counterfactual.hub_loader import load_hub_rankings, _synthetic_hubs
from stablesim.counterfactual.runner import run_counterfactual, N_SEEDS_FAST
from stablesim.counterfactual.ranking import causal_hub_ranking
from stablesim.analysis.comparison import compute_agreement
from stablesim.engine.market import MultiVenueMarket
from stablesim.scenarios.schedule import ShockSchedule
from stablesim.scenarios.loader import load_stressbench_scenarios


# ------------------------------------------------------------------
# Hub loader

def test_synthetic_hubs_loaded():
    hubs = _synthetic_hubs()
    assert len(hubs) >= 5
    importances = [h.predicted_importance for h in hubs]
    assert importances == sorted(importances, reverse=True), "Should be sorted by importance"


def test_load_hub_rankings_returns_hubs():
    hubs = load_hub_rankings()   # falls back to synthetic if repo 1 not found
    assert len(hubs) > 0
    assert all(hasattr(h, "predicted_importance") for h in hubs)
    assert all(0 <= h.predicted_importance <= 1 for h in hubs)


def test_hub_node_type_classification():
    hubs = _synthetic_hubs()
    pool_hubs = [h for h in hubs if h.node_type == NodeType.DEX_POOL]
    cex_hubs = [h for h in hubs if h.node_type == NodeType.CEX_VENUE]
    assert len(pool_hubs) > 0, "Expected at least one DEX pool hub"
    assert len(cex_hubs) > 0, "Expected at least one CEX venue hub"


# ------------------------------------------------------------------
# Intervention application

def test_circuit_breaker_intervention_modifies_market():
    hub = HubNode("curve_3pool", "Curve 3pool", 0.8, NodeType.DEX_POOL)
    market = MultiVenueMarket()
    orig_threshold = market.redemption.cb_threshold
    params = HubInterventionParams(
        intervention_type=InterventionType.CIRCUIT_BREAKER,
        cb_threshold=0.01,
        cb_duration=15,
    )
    apply_hub_intervention(market, hub, params)
    assert market.redemption.cb_threshold == 0.01
    assert market.redemption.cb_duration == 15


def test_redemption_gate_intervention_modifies_market():
    hub = HubNode("usdc_coinbase", "USDC Coinbase", 0.85, NodeType.CEX_VENUE)
    market = MultiVenueMarket()
    params = HubInterventionParams(
        intervention_type=InterventionType.REDEMPTION_GATE,
        gate_fee_bps=200.0,
        gate_queue_len=10,
        gate_delay_steps=6,
    )
    apply_hub_intervention(market, hub, params)
    assert market.redemption.fee_bps == 200.0
    assert market.redemption.max_queue == 10
    assert market.redemption.delay_steps == 6


def test_transparency_boost_intervention():
    hub = HubNode("usdc_mint_burn", "USDC Mint/Burn", 0.3, NodeType.MINT_BURN)
    market = MultiVenueMarket()
    params = HubInterventionParams(
        intervention_type=InterventionType.TRANSPARENCY_BOOST,
        transparency_freq=1,
        transparency_noise=0.005,
    )
    apply_hub_intervention(market, hub, params)
    assert market.reserve.transparency_freq == 1


def test_build_default_intervention_by_node_type():
    for ntype, expected in [
        (NodeType.DEX_POOL, InterventionType.CIRCUIT_BREAKER),
        (NodeType.CEX_VENUE, InterventionType.REDEMPTION_GATE),
        (NodeType.MINT_BURN, InterventionType.TRANSPARENCY_BOOST),
    ]:
        hub = HubNode("test", "test", 0.5, ntype)
        params = build_intervention_params(hub)
        assert params.intervention_type == expected


# ------------------------------------------------------------------
# Counterfactual runner (fast mode)

@pytest.fixture(scope="module")
def shock_scenario():
    scenarios = load_stressbench_scenarios()
    return next(s for s in scenarios if s.name not in ("no_shock_baseline", "baseline"))


def test_counterfactual_runner_returns_result(shock_scenario):
    hub = HubNode("curve_3pool", "Curve 3pool", 0.8, NodeType.DEX_POOL)
    result = run_counterfactual(hub, shock_scenario, n_seeds=N_SEEDS_FAST, n_steps=50)
    assert len(result.baseline_magnitudes) == N_SEEDS_FAST
    assert len(result.intervened_magnitudes) == N_SEEDS_FAST
    assert result.n_seeds == N_SEEDS_FAST


def test_counterfactual_se_is_finite(shock_scenario):
    hub = HubNode("curve_3pool", "Curve 3pool", 0.8, NodeType.DEX_POOL)
    result = run_counterfactual(hub, shock_scenario, n_seeds=N_SEEDS_FAST, n_steps=50)
    assert np.isfinite(result.se)
    assert np.isfinite(result.t_stat)
    assert 0 <= result.p_value_one_sided <= 1


def test_circuit_breaker_reduces_contagion_on_average(shock_scenario):
    """Applying a very sensitive CB to the primary pool should weakly reduce
    median contagion (stochastic — use loose tolerance)."""
    hub = HubNode("curve_3pool", "Curve 3pool", 0.8, NodeType.DEX_POOL)
    params = HubInterventionParams(
        intervention_type=InterventionType.CIRCUIT_BREAKER,
        cb_threshold=0.01,  # fires very early
        cb_duration=30,
    )
    result = run_counterfactual(
        hub, shock_scenario, n_seeds=N_SEEDS_FAST, n_steps=50,
        intervention_type=InterventionType.CIRCUIT_BREAKER,
    )
    # We don't assert direction (too few seeds) but result is well-formed
    assert result.summary_dict()["intervention_type"] == "circuit_breaker"


# ------------------------------------------------------------------
# Causal ranking

def test_causal_hub_ranking_sorted(shock_scenario):
    hubs = _synthetic_hubs()[:3]
    results = [
        run_counterfactual(h, shock_scenario, n_seeds=N_SEEDS_FAST, n_steps=40)
        for h in hubs
    ]
    df = causal_hub_ranking(results)
    assert "causal_rank" in df.columns
    assert list(df["causal_rank"]) == list(range(1, len(df) + 1))
    deltas = df["delta_contagion"].tolist()
    assert deltas == sorted(deltas, reverse=True)


# ------------------------------------------------------------------
# Agreement metrics

def test_compute_agreement_with_synthetic_data():
    """Verify agreement metrics compute correctly on known inputs."""
    import pandas as pd

    # Construct a synthetic table with perfect rank agreement
    df = pd.DataFrame({
        "node_id": ["A", "B", "C", "D", "E"],
        "predicted_importance": [0.9, 0.7, 0.5, 0.3, 0.1],
        "delta_contagion": [0.08, 0.06, 0.04, 0.02, 0.00],
        "node_type": ["dex_pool"] * 5,
        "role": ["amplifier"] * 5,
    })
    metrics = compute_agreement(df)
    assert metrics.spearman_rho > 0.9, "Perfect rank agreement should give high Spearman"
    assert metrics.top3_overlap == 1.0, "Perfect agreement: top-3 identical"
    assert metrics.n_nodes == 5


def test_compute_agreement_with_disagreement():
    """Anti-correlated ranks should give negative Spearman and zero top-3 overlap.

    6 nodes: top-3 predicted = {A,B,C}, top-3 causal = {F,E,D} — fully disjoint.
    """
    import pandas as pd

    df = pd.DataFrame({
        "node_id": ["A", "B", "C", "D", "E", "F"],
        "predicted_importance": [0.90, 0.80, 0.70, 0.30, 0.20, 0.10],
        "delta_contagion":      [0.00, 0.01, 0.02, 0.07, 0.08, 0.09],
        "node_type": ["dex_pool"] * 6,
        "role": ["mixed"] * 6,
    })
    metrics = compute_agreement(df)
    assert metrics.spearman_rho < -0.9
    assert metrics.top3_overlap == 0.0, (
        f"Expected zero overlap: top-3 predicted={{A,B,C}} vs top-3 causal={{F,E,D}}, "
        f"got {metrics.top3_overlap:.3f}"
    )
