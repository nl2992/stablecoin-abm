# Agent Specification

## Overview

All agents share a common interface via `BaseAgent.act(market, obs)` called each step.
Heuristic policies are the default.  RL-trained policies are injected via `agent.policy = fn`.

---

## Agent spec table

| Agent | Observation | Action | Utility | Heuristic trigger |
|---|---|---|---|---|
| `Arbitrageur` | prices per pool, wealth | (pool_idx, trade_size_frac) | P&L per step | spread > `min_spread` |
| `Redeemer` | depeg, step, queue depth | redemption_frac ∈ [0,1] | P&L from redeeming at par vs AMM price | depeg < −`trigger_depeg` |
| `LPAgent` | depeg, pool reserves | add or remove (frac) | P&L − impermanent loss + subsidy | add if \|depeg\| < threshold; remove if \|depeg\| > threshold |
| `IssuerAgent` | depeg | USD deployed for buyback | Peg stability (exogenous) | depeg < −`intervention_threshold` |
| `NoiseTrader` | none | random buy/sell (size) | None (background flow) | Bernoulli(`trade_prob`) each step |

---

## Arbitrageur (`agents/arbitrageur.py`)

**Role:** Profits from price discrepancies between pools, and between AMM and primary redemption ($1).

**Heuristic policy:**
1. Find pools with `max_price` and `min_price`.
2. If `spread = max_price − min_price > min_spread`, buy on cheap pool, sell on expensive pool.
3. Trade size = `min(wealth × max_trade_frac, 5% of target pool's y reserve)`.

**RL action space:** `Box([0,1]²)` — (pool_idx_frac, trade_size_frac).

**Myopic correctness test (Phase 1 gate):**
After inducing a 5% depeg in one pool, heuristic arbitrageur must measurably shrink the spread over ≤ 20 steps.

---

## Redeemer (`agents/redeemer.py`)

**Role:** Profits by buying stablecoins at AMM discount and redeeming them at par ($1) through the primary channel (net of gating fee/delay).

**Heuristic policy:**
- If `depeg < −trigger_depeg` and stablecoin balance > 0:
  submit `redemption_frac × stablecoin_holdings` to RedemptionChannel.
- Does nothing when peg is at or above par.

**RL action space:** `Box([0,1])` — redemption fraction.

**Net-position accounting:**
- `stablecoin` balance decreases on submission.
- `wealth` increases when settlement arrives (via `receive_settlement(net_usd)`).

---

## LPAgent (`agents/lp.py`)

**Role:** Provides liquidity during calm; withdraws during stress to avoid impermanent loss.

**Heuristic policy:**
- Add liquidity when `|depeg| < add_threshold` (pool near equilibrium).
- Remove liquidity when `|depeg| > remove_threshold` (pool stressed).
- Per-step LP subsidy earned on `lp_tokens` held (if `subsidy_rate > 0`).

**IL tracking:**
Impermanent loss recorded as `(actual pool value at removal) − (hold value at entry price)`.

**Welfare metric:** cumulative IL is the key "who pays" signal for LP incentive interventions.

---

## IssuerAgent (`agents/issuer.py`)

**Role:** Defends peg via open-market operations (buybacks from primary pool) when depeg exceeds threshold.

**Policy:** buy `intervention_size` USD of stablecoins from pool when `depeg < −intervention_threshold`.

**Note:** Issuer does not participate in RL training; it represents the policy lever being studied (intervention), not an RL agent.

---

## NoiseTrader (`agents/noise.py`)

**Role:** Background order flow providing realistic price variance in the no-shock baseline.

**Policy:** each step, with probability `trade_prob`, place a buy or sell of size ~ N(`trade_size_mean`, `trade_size_std²`) on a random pool.

**Calibration role:** `trade_prob` and `trade_size_mean` are among the parameters tuned in Phase 2 to match empirical baseline price volatility.

---

## P&L and welfare accounting

Each agent tracks `cumulative_pnl` via `record_pnl(delta)`.

`compute_metrics()` in `analysis/metrics.py` aggregates:

```python
welfare_by_type = {
    "arbitrageur": sum P&L across all Arbitrageur agents,
    "redeemer":    sum P&L across all Redeemer agents,
    "lp":          sum P&L (IL-adjusted) across all LPAgent agents,
    "issuer":      sum P&L across all IssuerAgent agents,
    "noise":       sum P&L across all NoiseTrader agents,
}
```

The welfare-decomposition matrix (Gu / JaxMARL Fig 5 analog) shows which agent types bear the cost of each intervention — the core "mechanism + who-pays" framing of the paper.
