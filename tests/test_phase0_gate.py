"""Phase 0 gate tests.

Gate criterion (from ROADMAP.md):
  "With no shock and no agents acting, pegs sit at $1 and the invariant is conserved."

Additional accounting gate:
  "Redemption accounting balances to the cent."

All tests here must pass before Phase 1 work begins.
"""

from __future__ import annotations

import numpy as np
import pytest

from stablesim.engine.amm import StableswapAMM
from stablesim.engine.market import MultiVenueMarket
from stablesim.engine.redemption import RedemptionChannel
from stablesim.engine.reserve import ReserveModel


# =============================================================================
# AMM: invariant D conservation
# =============================================================================


def test_D_exact_at_initialisation():
    """D should equal 2 × reserve for equal-reserve initialisation."""
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100, fee_bps=0)
    assert abs(amm._D - 2_000_000) < 1e-6


def test_invariant_conserved_zero_fee():
    """D must be unchanged (< 0.01 ppm drift) across many zero-fee round-trip swaps."""
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100, fee_bps=0)
    D0 = amm._D
    rng = np.random.default_rng(0)
    for _ in range(100):
        size = rng.uniform(1_000, 50_000)
        amm.swap_x_for_y(size)
        amm.swap_y_for_x(size * 0.999)   # slightly less due to price impact
    drift_ppm = abs(amm._D - D0) / D0 * 1e6
    assert drift_ppm < 0.01, f"D drifted {drift_ppm:.4f} ppm with zero fees"


def test_D_nondecreasing_with_fees():
    """Fees add to pool reserves, so D should never decrease with positive fees."""
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100, fee_bps=4)
    D0 = amm._D
    for _ in range(50):
        amm.swap_x_for_y(10_000)
        amm.swap_y_for_x(8_000)
    assert amm._D >= D0 - 1e-6, "D should not decrease when fees are positive"


# =============================================================================
# AMM: equilibrium price
# =============================================================================


def test_analytic_price_exactly_one_at_par():
    """Analytic price formula must return exactly 1.0 when x == y."""
    for amp in [10, 100, 500]:
        amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=amp, fee_bps=0)
        p = amm.price()
        assert abs(p - 1.0) < 1e-12, f"amp={amp}: price={p}"


def test_price_moves_away_from_par_after_large_sell():
    """Selling stablecoin x into the pool must push price below $1."""
    amm = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=100)
    amm.swap_x_for_y(300_000)
    assert amm.price() < 1.0


def test_higher_amp_tighter_price_impact():
    """Higher amplification should produce less price impact per unit swapped."""
    amm_lo = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=10)
    amm_hi = StableswapAMM(reserves=(1_000_000, 1_000_000), amp=500)
    amm_lo.swap_x_for_y(100_000)
    amm_hi.swap_x_for_y(100_000)
    assert amm_hi.price() > amm_lo.price(), "Higher A should maintain tighter peg"


# =============================================================================
# Market: no-shock, no-agent equilibrium
# =============================================================================


def test_no_agents_no_shock_price_stays_at_par():
    """Core Phase 0 gate: running 200 steps with no agents and no shocks must
    leave the AMM price at $1 (reserve OU evolves, but no one trades)."""
    rng = np.random.default_rng(42)
    market = MultiVenueMarket(rng=rng)
    for _ in range(200):
        market.step(shock=None)
    price = market.mid_price()
    assert abs(price - 1.0) < 1e-10, f"Expected $1 peg, got {price}"


def test_depeg_returns_zero_without_agents():
    """depeg() must be 0.0 at all times with no agents and no shocks."""
    market = MultiVenueMarket(rng=np.random.default_rng(7))
    for _ in range(100):
        snap = market.step()
        assert abs(snap["depeg"]) < 1e-10, f"Unexpected depeg at step {snap['step']}"


# =============================================================================
# RedemptionChannel: accounting invariants
# =============================================================================


def test_mint_increases_reserve_and_supply_equally():
    """mint(x) must increase reserve_usd and total_supply by exactly x (1:1)."""
    ch = RedemptionChannel(initial_reserve_usd=1_000_000, initial_supply=1_000_000)
    sc = ch.mint(10_000)
    assert sc == 10_000
    assert abs(ch.reserve_usd - 1_010_000) < 1e-9
    assert abs(ch.total_supply - 1_010_000) < 1e-9
    assert abs(ch.backing_ratio - 1.0) < 1e-9


def test_redeem_zero_fee_balances_to_the_cent():
    """Zero-fee mint then full redeem must return reserve and supply to initial."""
    ch = RedemptionChannel(
        initial_reserve_usd=1_000_000,
        initial_supply=1_000_000,
        fee_bps=0,
        delay_steps=0,
    )
    deposit = 5_000.0
    ch.mint(deposit)
    ch.submit("agent", deposit, current_step=0)
    settled = ch.settle(current_step=0)

    assert len(settled) == 1
    assert abs(settled[0]["usd_net"] - deposit) < 1e-9, "Net payout ≠ deposit (zero fee)"
    assert abs(ch.reserve_usd - 1_000_000) < 1e-9, "Reserve not restored"
    assert abs(ch.total_supply - 1_000_000) < 1e-9, "Supply not restored"


def test_redeem_with_fee_accounting():
    """With a non-zero fee the fee stays in the reserve; net payout is correct."""
    fee_bps = 50  # 0.5%
    ch = RedemptionChannel(
        initial_reserve_usd=1_000_000,
        initial_supply=1_000_000,
        fee_bps=fee_bps,
        delay_steps=0,
    )
    ch.mint(10_000)
    ch.submit("agent", 10_000, current_step=0)
    settled = ch.settle(0)

    fee_rate = fee_bps / 10_000
    expected_net = 10_000 * (1 - fee_rate)
    expected_fee = 10_000 * fee_rate

    assert abs(settled[0]["usd_net"] - expected_net) < 1e-9
    assert abs(settled[0]["fee_usd"] - expected_fee) < 1e-9
    # Fee stays in reserve: net reserve change = −usd_net (not −usd_gross)
    # After mint(10_000) → reserve = 1_010_000
    # After settle: reserve -= usd_net → 1_010_000 − expected_net
    assert abs(ch.reserve_usd - (1_010_000 - expected_net)) < 1e-9


def test_circuit_breaker_blocks_redemptions():
    """When circuit breaker fires, submit() must return False."""
    ch = RedemptionChannel(cb_threshold=0.05, cb_duration=10, fee_bps=0, delay_steps=0)
    ch.trigger_circuit_breaker(current_step=0)
    accepted = ch.submit("agent", 1000, current_step=5)
    assert not accepted, "Redemption should be blocked during circuit breaker"
    # After cb expires, should be accepted again
    accepted_after = ch.submit("agent", 1000, current_step=11)
    assert accepted_after


def test_reserve_exhaustion_blocks_new_submissions():
    """Once reserve is exhausted, no new orders should be accepted."""
    ch = RedemptionChannel(
        initial_reserve_usd=100,
        initial_supply=1_000,
        fee_bps=0,
        delay_steps=0,
    )
    ch.reserve_usd = 0.0   # force exhaustion
    accepted = ch.submit("agent", 10, current_step=0)
    assert not accepted, "Should not accept redemption on exhausted reserve"


# =============================================================================
# ReserveModel: exhaustion logic
# =============================================================================


def test_reserve_exhaustion_triggered():
    """Setting ratio to 0 should flag the model as exhausted."""
    res = ReserveModel(exhaustion_threshold=0.05)
    res.shock(-1.0)   # drop ratio to max(0, 1.0-1.0)=0
    assert res.is_exhausted


def test_reserve_disclosure_emitted_at_correct_frequency():
    """Disclosure should be None until the first multiple of transparency_freq."""
    res = ReserveModel(transparency_freq=5, transparency_noise=0.0, rng=np.random.default_rng(0))
    assert res.disclosed_ratio is None
    for i in range(1, 5):
        res.step()
        assert res.disclosed_ratio is None, f"Disclosed too early at step {i}"
    res.step()   # step 5
    assert res.disclosed_ratio is not None
    assert abs(res.disclosed_ratio - res.ratio) < 1e-9, "No-noise disclosure should be exact"
