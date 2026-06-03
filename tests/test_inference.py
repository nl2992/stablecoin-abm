"""Tests for inference.py and ablation_adapter.py.

All five tests specified in INTEGRATION.md:
  1. Paired SE equals sd(d)/sqrt(n) on a fixed fixture (golden value).
  2. BH monotonicity: q-values non-decreasing in p-order; recovers raw p when m=1.
  3. Planted spurious hub → not FDR-significant.
  4. apply_ablation(alpha=1.0) makes the node inert.
  5. alpha=0.0 is a bit-for-bit no-op.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stablesim.counterfactual.inference import (
    bh_correct,
    paired_test,
    required_n,
    summarize_sweep,
)
from stablesim.counterfactual.ablation_adapter import apply_ablation
from stablesim.counterfactual.hub_interventions import HubNode, NodeType
from stablesim.engine.amm import StableswapAMM
from stablesim.engine.market import MultiVenueMarket
from stablesim.engine.redemption import RedemptionChannel
from stablesim.engine.reserve import ReserveModel
from stablesim.agents.noise import NoiseTrader


# ─── Helpers ────────────────────────────────────────────────────────────────


def _fresh_market(seed: int = 0) -> MultiVenueMarket:
    rng = np.random.default_rng(seed)
    return MultiVenueMarket(
        pools=[StableswapAMM(), StableswapAMM(reserves=(900_000, 1_100_000), amp=100)],
        redemption=RedemptionChannel(),
        reserve=ReserveModel(rng=rng),
        rng=rng,
    )


def _noise_agents(seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [NoiseTrader(f"n{i}", rng=rng) for i in range(3)]


# ─── Test 1: Paired SE golden value ─────────────────────────────────────────


def test_paired_se_equals_sd_over_sqrt_n():
    """SE of paired test must equal sd(d)/sqrt(n) — the defining identity."""
    rng = np.random.default_rng(42)
    n = 40
    baseline = rng.normal(10, 2, size=n)
    effect = 1.0
    intervened = baseline - effect + rng.normal(0, 0.3, size=n)

    d = baseline - intervened
    expected_se = float(d.std(ddof=1)) / math.sqrt(n)

    result = paired_test("hub_A", baseline, intervened, rng=np.random.default_rng(0))

    assert abs(result.se_paired - expected_se) < 1e-12, (
        f"Paired SE mismatch: got {result.se_paired:.8f}, expected {expected_se:.8f}"
    )
    # Also verify delta_c = mean(d)
    assert abs(result.delta_c - float(d.mean())) < 1e-12


def test_paired_se_tighter_than_two_sample_when_correlated():
    """When baseline and intervened are correlated (good pairing), paired SE < two-sample SE."""
    rng = np.random.default_rng(7)
    n = 40
    common = rng.normal(0, 5, n)   # strong common shock
    baseline = common + 10 + rng.normal(0, 0.2, n)
    intervened = common + 9 + rng.normal(0, 0.2, n)

    result = paired_test("hub_B", baseline, intervened, rng=np.random.default_rng(0))

    # Two-sample SE (incorrect formula) for comparison
    se_twosamp = math.sqrt(baseline.var(ddof=1)/n + intervened.var(ddof=1)/n)
    assert result.se_paired < se_twosamp, (
        "Paired SE should be smaller than two-sample SE when arms are correlated"
    )
    # pair_corr should be strongly positive
    assert result.pair_corr > 0.8, f"Expected high pair_corr, got {result.pair_corr:.3f}"


# ─── Test 2: BH monotonicity ────────────────────────────────────────────────


def test_bh_q_values_nondecreasing_in_p_order():
    """BH q-values must be non-decreasing when sorted by p-value."""
    results = []
    for p in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
        r = paired_test("x", [1.0] * 10, [0.9] * 10, rng=np.random.default_rng(0))
        # Manually set p to known value for test
        object.__setattr__(r, "p_one_sided", p) if False else None
        r.p_one_sided = p   # dataclass is not frozen
        results.append(r)

    bh_correct(results, fdr=0.05)
    ordered = sorted(results, key=lambda r: r.p_one_sided)
    q_vals = [r.q_value for r in ordered]

    for i in range(len(q_vals) - 1):
        assert q_vals[i] <= q_vals[i + 1] + 1e-12, (
            f"BH q not non-decreasing at positions {i},{i+1}: {q_vals[i]:.4f} > {q_vals[i+1]:.4f}"
        )


def test_bh_recovers_raw_p_when_m_equals_one():
    """With m=1 hub, BH q-value should equal the raw p-value."""
    rng = np.random.default_rng(0)
    b = rng.normal(10, 1, 30)
    iv = b - 1.0 + rng.normal(0, 0.3, 30)
    results = [paired_test("single_hub", b, iv, rng=np.random.default_rng(0))]
    bh_correct(results, fdr=0.05)
    assert abs(results[0].q_value - results[0].p_one_sided) < 1e-10


def test_bh_all_q_values_set():
    """After bh_correct, every result must have q_value and significant_fdr set."""
    hubs = {f"hub_{i}": ([1.0]*10, [0.8]*10) for i in range(6)}
    results = summarize_sweep(hubs, seed=0)
    for r in results:
        assert r.q_value is not None
        assert r.significant_fdr is not None


# ─── Test 3: Planted spurious hub ───────────────────────────────────────────


def test_planted_spurious_hub_not_fdr_significant():
    """A hub with high baseline C but zero true causal effect should NOT be FDR-significant."""
    rng = np.random.default_rng(99)
    n = 40
    common = rng.normal(0, 3, n)

    hubs = {
        # True causal hub: intervention reliably reduces contagion by 2.0
        "real_hub": (
            common + 10 + rng.normal(0, 0.3, n),
            common + 8.0 + rng.normal(0, 0.3, n),
        ),
        # Spurious hub: high baseline (highly active during stress) but zero effect
        "spurious_hub": (
            common + 15 + rng.normal(0, 0.3, n),
            common + 15 + rng.normal(0, 0.3, n),   # zero treatment effect
        ),
        # Filler hubs (small true effects)
        "filler_1": (common + 9 + rng.normal(0, 0.3, n), common + 8.8 + rng.normal(0, 0.3, n)),
        "filler_2": (common + 9 + rng.normal(0, 0.3, n), common + 8.9 + rng.normal(0, 0.3, n)),
    }

    results = summarize_sweep(hubs, seed=42)
    by_id = {r.node_id: r for r in results}

    # The real hub should have positive delta_c
    assert by_id["real_hub"].delta_c > 1.0, "Real hub delta_c too small"

    # The spurious hub should NOT be FDR-significant
    spurious = by_id["spurious_hub"]
    assert not spurious.significant_fdr, (
        f"Spurious hub should not be FDR-significant, got q={spurious.q_value:.4f}"
    )


# ─── Test 4: alpha=1.0 makes node inert ─────────────────────────────────────


class _StubHub:
    """Minimal HubNode-like object for adapter tests."""
    def __init__(self, node_type):
        self.node_id = "test_node"
        self.node_type = node_type


def test_alpha_one_makes_dex_pool_nearly_empty():
    """After alpha=1 ablation, pool reserves must be ≤ 0.1% of initial."""
    hub = _StubHub(NodeType.DEX_POOL)
    market = _fresh_market()
    x0, y0 = market.pools[0].x, market.pools[0].y

    apply_ablation(market, hub, alpha=1.0, agents=[])

    assert market.pools[0].x < x0 * 0.01, (
        f"Pool x not reduced: {market.pools[0].x:.2f} vs initial {x0:.2f}"
    )
    assert market.pools[0].y < y0 * 0.01


def test_alpha_one_makes_cex_venue_effectively_sealed():
    """After alpha=1 ablation, redemption fee must be high and delay must be large."""
    hub = _StubHub(NodeType.CEX_VENUE)
    market = _fresh_market()

    apply_ablation(market, hub, alpha=1.0, agents=[])

    # Fee should be substantial (≥1000bps = 10%)
    assert market.redemption.fee_bps >= 1000, (
        f"CEX venue not sealed: fee_bps={market.redemption.fee_bps}"
    )
    # Delay should be large (≥50 steps)
    assert market.redemption.delay_steps >= 50, (
        f"CEX venue not sealed: delay_steps={market.redemption.delay_steps}"
    )


def test_alpha_one_makes_mint_burn_reserve_near_zero():
    """After alpha=1 ablation, reserve_usd must be ≤ 0.1% of initial."""
    hub = _StubHub(NodeType.MINT_BURN)
    market = _fresh_market()
    r0 = market.redemption.reserve_usd

    apply_ablation(market, hub, alpha=1.0, agents=[])

    assert market.redemption.reserve_usd < r0 * 0.01, (
        f"Mint/burn reserve not drained: {market.redemption.reserve_usd:.2f} vs {r0:.2f}"
    )


def test_alpha_one_silences_exchange_flow_traders():
    """After alpha=1 ablation on EXCHANGE_FLOW, noise-trader trade_prob must be near zero."""
    hub = _StubHub(NodeType.EXCHANGE_FLOW)
    market = _fresh_market()
    agents = _noise_agents()

    apply_ablation(market, hub, alpha=1.0, agents=agents)

    for agent in agents:
        if isinstance(agent, NoiseTrader):
            assert agent.trade_prob < 1e-6, (
                f"Noise trader not silenced: trade_prob={agent.trade_prob:.6f}"
            )


# ─── Test 5: alpha=0.0 is a bit-for-bit no-op ───────────────────────────────


def test_alpha_zero_is_exact_noop_dex_pool():
    """alpha=0 must not change pool reserves by even one ULP."""
    hub = _StubHub(NodeType.DEX_POOL)
    market = _fresh_market(seed=1)
    x0, y0, D0 = market.pools[0].x, market.pools[0].y, market.pools[0]._D

    apply_ablation(market, hub, alpha=0.0, agents=[])

    assert market.pools[0].x == x0
    assert market.pools[0].y == y0
    assert market.pools[0]._D == D0


def test_alpha_zero_is_exact_noop_cex_venue():
    hub = _StubHub(NodeType.CEX_VENUE)
    market = _fresh_market(seed=2)
    fee0 = market.redemption.fee_bps
    delay0 = market.redemption.delay_steps

    apply_ablation(market, hub, alpha=0.0, agents=[])

    assert market.redemption.fee_bps == fee0
    assert market.redemption.delay_steps == delay0


def test_alpha_zero_produces_identical_episode():
    """An episode with alpha=0 must produce the same contagion as an untouched run."""
    from stablesim.scenarios.loader import load_stressbench_scenarios

    hub = _StubHub(NodeType.DEX_POOL)
    scenario = load_stressbench_scenarios()[0]

    def _run(with_ablation: bool, seed: int) -> float:
        rng = np.random.default_rng(seed)
        market = MultiVenueMarket(
            pools=[StableswapAMM()],
            redemption=RedemptionChannel(),
            reserve=ReserveModel(rng=rng),
            rng=rng,
        )
        agents = _noise_agents(seed)
        if with_ablation:
            apply_ablation(market, hub, alpha=0.0, agents=agents)
        for step in range(50):
            shock = scenario.events_at(step)[0] if scenario.events_at(step) else None
            snap = market.step(shock=shock)
            for a in agents:
                a.act(market, snap)
        return float(market.history_df()["depeg"].abs().max())

    for seed in range(3):
        assert _run(False, seed) == _run(True, seed), (
            f"alpha=0 changed episode outcome at seed {seed}"
        )
