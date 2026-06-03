# Engine Specification

## Overview

The `stablesim` engine models a multi-venue stablecoin market comprising:

1. **Stableswap AMM pools** — on-chain DEX liquidity for stablecoin swaps
2. **Redemption channel** — the issuer's primary mint/redeem facility at $1 face value
3. **Reserve model** — stochastic backing ratio with controlled disclosure

Time is discrete.  Each step calls `MultiVenueMarket.step(shock?)`.

---

## 1. StableswapAMM (`engine/amm.py`)

### Invariant

Two-token Curve stableswap invariant (n = 2):

```
4A(x + y) + D = 4AD + D³ / (4xy)
```

where:
- `x`, `y` — pool token balances
- `D` — invariant (solved numerically via Newton's method)
- `A` — amplification coefficient (typical range 10–500)

### Analytic marginal price

Derived via implicit differentiation:

```
∂F/∂x = 4A + D³/(4x²y)
∂F/∂y = 4A + D³/(4xy²)
price  = (∂F/∂x) / (∂F/∂y)
```

At equilibrium (x = y): `∂F/∂x = ∂F/∂y` → price = 1.0 **exactly**.

### Fee mechanics

- Swap fee `fee_bps` in basis points (e.g. 4 = 0.04%)
- Fee amount remains in the pool → D strictly increases with each fee-bearing swap
- Net output to trader: `dy_net = dy_gross × (1 − fee)`

### Phase 0 gate

| Condition | Expected | Tolerance |
|---|---|---|
| Price at equal reserves | 1.0 | < 1e-12 |
| D drift across 100 zero-fee round-trips | 0.0 | < 0.01 ppm |
| D non-decreasing with fees | ≥ D₀ | exact |

---

## 2. RedemptionChannel (`engine/redemption.py`)

### Balance sheet

| Variable | Meaning |
|---|---|
| `reserve_usd` | USD held in the issuer's redemption reserve |
| `total_supply` | Stablecoins in circulation (minted − redeemed) |
| `backing_ratio` | `reserve_usd / total_supply` |

### Mint

```
mint(usd_in):
    stablecoins = usd_in          # 1:1 issuance
    reserve_usd += usd_in
    total_supply += stablecoins
```

### Redeem (settlement)

```
settle(step):
    for each ready order (amount = stablecoins to burn):
        usd_gross = min(amount, reserve_usd)
        fee_usd   = usd_gross × fee_rate
        usd_net   = usd_gross − fee_usd      → paid to redeemer
        reserve_usd  −= usd_net              # fee stays in reserve
        total_supply −= usd_gross
```

**Accounting invariant (zero fee, fully backed, no exhaustion):**

```
mint(X) then full redeem(X):
    reserve_usd  → initial value  (±1e-9)
    total_supply → initial value  (±1e-9)
```

### Gating controls

| Knob | Effect |
|---|---|
| `fee_bps` | Reduces usd_net paid to redeemer; fee stays in reserve |
| `max_queue` | Caps pending orders; excess submissions rejected |
| `delay_steps` | Order settled only after ≥ delay_steps since submission |
| `cb_threshold` | `|price−1|` that triggers circuit breaker |
| `cb_duration` | Steps the halt remains active |

---

## 3. ReserveModel (`engine/reserve.py`)

### Backing ratio dynamics

Euler-Maruyama discretisation of an Ornstein–Uhlenbeck process:

```
dr_t = κ(θ − r_t) dt + σ dW_t
```

| Parameter | Symbol | Typical value |
|---|---|---|
| `speed` | κ | 0.05 |
| `mean_ratio` | θ | 1.0 |
| `vol` | σ | 0.015 |

### Disclosure

- True ratio `r_t` is unobservable to agents
- Issuer publishes a signal every `transparency_freq` steps:
  `signal = r_t + N(0, transparency_noise²)`
- `perceived_backing` = last disclosed ratio, or prior mean `θ` if never disclosed

### Exhaustion

```
is_exhausted = (r_t ≤ exhaustion_threshold)
```

When exhausted: `RedemptionChannel.submit()` returns False; redemptions halt.

---

## 4. MultiVenueMarket (`engine/market.py`)

### Step sequence

```
1. reserve.step()           — advance OU process
2. _apply_shock(event)      — if shock fired this step
3. redemption.check_and_trigger(price, step)
4. settled = redemption.settle(step)
5. record snapshot
```

### Snapshot keys

```python
{
    "step": int,
    "prices": [float],          # per-pool spot price
    "mid_price": float,         # equal-weight average
    "depeg": float,             # mid_price − 1.0
    "reserve_ratio": float,     # true OU ratio
    "reserve_perceived": float, # agent-observable backing
    "queue_depth": int,
    "settled_count": int,
    "pool_states": [dict],      # per-pool x, y, D, price, A
}
```

### Shock kinds

| `kind` | Effect |
|---|---|
| `sell_pressure` | swap `magnitude × pool.x` stablecoins into pool |
| `buy_pressure` | swap `magnitude × pool.y` USD into pool |
| `liquidity_removal` | remove `magnitude` fraction of pool liquidity |
| `reserve_drop` | subtract `magnitude` from `reserve.ratio` directly |
