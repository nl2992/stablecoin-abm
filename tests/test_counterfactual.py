"""Tests for the counterfactual engine — updated for paired-design runner.

Uses N_SEEDS_FAST and 50-step episodes to stay under 30 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from stablesim.counterfactual.hub_interventions import (
    HubNode, NodeType, InterventionType,
    HubInterventionParams, apply_hub_intervention, build_intervention_params,
)
from stablesim.counterfactual.hub_loader import load_hub_rankings, _synthetic_hubs
from stablesim.counterfactual.runner import run_hub_paired, N_SEEDS_FAST
from stablesim.counterfactual.inference import summarize_sweep
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
    assert importances == sorted(importances, reverse=True)


def test_load_hub_rankings_requires_explicit_flag_when_missing(tmp_path):
    """Missing repo-1 table must raise FileNotFoundError by default."""
    with pytest.raises(FileNotFoundError):
        load_hub_rankings(repo1_root=str(tmp_path), allow_synthetic=False)


def test_load_hub_rankings_synthetic_allowed(tmp_path):
    """With allow_synthetic=True, returns synthetic hubs and emits a warning."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hubs = load_hub_rankings(repo1_root=str(tmp_path), allow_synthetic=True)
    assert len(hubs) > 0
    assert any("SYNTHETIC" in str(warning.message) for warning in w)


def test_load_hub_rankings_real_data():
    """If repo-1 data exists, load it and validate schema."""
    from pathlib import Path
    repo1 = Path(__file__).parents[2] / "stablecoin-contagion-network"
    if not (repo1 / "results" / "tables" / "table_node_centrality.csv").exists():
        pytest.skip("Repo-1 data not present in this environment")
    hubs = load_hub_rankings(repo1_root=str(repo1))
    assert len(hubs) > 0
    assert all(0 <= h.predicted_importance <= 1 for h in hubs)


def test_hub_node_type_classification():
    hubs = _synthetic_hubs()
    pool_hubs = [h for h in hubs if h.node_type == NodeType.DEX_POOL]
    cex_hubs = [h for h in hubs if h.node_type == NodeType.CEX_VENUE]
    assert len(pool_hubs) > 0
    assert len(cex_hubs) > 0


# ------------------------------------------------------------------
# Intervention application (type-specific knobs — secondary analysis only)

def test_circuit_breaker_intervention_modifies_market():
    hub = HubNode("curve_3pool", "Curve 3pool", 0.8, NodeType.DEX_POOL)
    market = MultiVenueMarket()
    params = HubInterventionParams(
        intervention_type=InterventionType.CIRCUIT_BREAKER,
        cb_threshold=0.01, cb_duration=15,
    )
    apply_hub_intervention(market, hub, params)
    assert market.redemption.cb_threshold == 0.01
    assert market.redemption.cb_duration == 15


def test_redemption_gate_intervention_modifies_market():
    hub = HubNode("usdc_coinbase", "USDC Coinbase", 0.85, NodeType.CEX_VENUE)
    market = MultiVenueMarket()
    params = HubInterventionParams(
        intervention_type=InterventionType.REDEMPTION_GATE,
        gate_fee_bps=200.0, gate_queue_len=10, gate_delay_steps=6,
    )
    apply_hub_intervention(market, hub, params)
    assert market.redemption.fee_bps == 200.0
    assert market.redemption.max_queue == 10
    assert market.redemption.delay_steps == 6


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
# Paired counterfactual runner

@pytest.fixture(scope="module")
def shock_scenario():
    scenarios = load_stressbench_scenarios()
    return next(s for s in scenarios if s.name not in ("no_shock_baseline", "baseline"))


def test_paired_runner_returns_aligned_arrays(shock_scenario):
    hub = HubNode("curve_3pool", "Curve 3pool", 0.8, NodeType.DEX_POOL)
    b, c = run_hub_paired(hub, shock_scenario, alpha=1.0, n_seeds=N_SEEDS_FAST, n_steps=50)
    assert len(b) == N_SEEDS_FAST
    assert len(c) == N_SEEDS_FAST
    # All values must be finite non-negative
    assert all(np.isfinite(v) and v >= 0 for v in b)
    assert all(np.isfinite(v) and v >= 0 for v in c)


def test_summarize_sweep_produces_fdr_corrected_results(shock_scenario):
    hubs = _synthetic_hubs()[:3]
    per_hub = {}
    for hub in hubs:
        b, c = run_hub_paired(hub, shock_scenario, alpha=1.0, n_seeds=N_SEEDS_FAST, n_steps=50)
        per_hub[hub.node_id] = (b, c)
    results = summarize_sweep(per_hub, seed=0)
    assert len(results) == 3
    for r in results:
        assert r.q_value is not None
        assert np.isfinite(r.se_paired)
        assert np.isfinite(r.t_stat)
        assert 0 <= r.p_one_sided <= 1


def test_causal_ranking_sorted_by_delta_c(shock_scenario):
    hubs = _synthetic_hubs()[:3]
    per_hub = {}
    for hub in hubs:
        b, c = run_hub_paired(hub, shock_scenario, alpha=1.0, n_seeds=N_SEEDS_FAST, n_steps=50)
        per_hub[hub.node_id] = (b, c)
    results = summarize_sweep(per_hub, seed=0)
    df = causal_hub_ranking(results)
    assert list(df["causal_rank"]) == list(range(1, len(df) + 1))
    deltas = df["delta_contagion"].tolist()
    assert deltas == sorted(deltas, reverse=True)


# ------------------------------------------------------------------
# Agreement metrics

def test_compute_agreement_with_synthetic_data():
    import pandas as pd
    df = pd.DataFrame({
        "node_id": ["A", "B", "C", "D", "E"],
        "predicted_importance": [0.9, 0.7, 0.5, 0.3, 0.1],
        "delta_contagion": [0.08, 0.06, 0.04, 0.02, 0.00],
        "node_type": ["dex_pool"] * 5,
        "role": ["amplifier"] * 5,
    })
    metrics = compute_agreement(df)
    assert metrics.spearman_rho > 0.9
    assert metrics.top3_overlap == 1.0
    assert metrics.n_nodes == 5


def test_compute_agreement_with_disagreement():
    """Anti-correlated ranks: top-3 predicted={A,B,C} and top-3 causal={F,E,D} are disjoint."""
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
    assert metrics.top3_overlap == 0.0


def test_bootstrapped_spearman_ci_is_finite():
    import pandas as pd
    df = pd.DataFrame({
        "node_id": ["A", "B", "C", "D", "E"],
        "predicted_importance": [0.9, 0.7, 0.5, 0.3, 0.1],
        "delta_contagion": [0.08, 0.06, 0.04, 0.02, 0.00],
        "node_type": ["dex_pool"] * 5,
        "role": ["amplifier"] * 5,
    })
    metrics = compute_agreement(df, n_boot=500)
    assert np.isfinite(metrics.spearman_ci_lo)
    assert np.isfinite(metrics.spearman_ci_hi)
    assert metrics.spearman_ci_lo <= metrics.spearman_rho <= metrics.spearman_ci_hi
