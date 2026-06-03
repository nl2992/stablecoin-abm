# Section 2: Method

## 2.1 Two-system design

```
Repo 1 (stablecoin-contagion-network)         Repo 2 (stablecoin-abm)
─────────────────────────────────────         ──────────────────────────
Real tick data → graph → GNN                  Calibrated ABM
Node centrality ranking (predicted)    →→→    Counterfactual engine
Empirical moments (OU half-life, ρ̂)  →→→    Calibration targets
StressBench shock schedule             →→→    Exogenous shock schedule
```

The two repos share only the shock schedule and the calibration moments.
The causal ranking is computed entirely in repo 2 and then compared against repo 1's
predicted ranking.  This separation prevents circularity.

## 2.2 ABM architecture

### Market mechanism
- **N = 2 stableswap AMM pools** (Curve invariant, A = 100, fee = 4bps)
- **Primary redemption channel**: mint/redeem at $1 face value with configurable gating
- **Reserve model**: OU backing ratio with controlled disclosure (transparency knob)

### Agent population
- **Arbitrageurs** (2): cross-venue price equalisation; PPO-trainable
- **Redeemers** (2): exploit AMM/redemption price gap; PPO-trainable
- **LPs** (2): add/withdraw liquidity based on depeg signal
- **Issuer** (1): open-market operations when depeg exceeds threshold
- **Noise traders** (3): background order flow (calibrated to match empirical price vol)

### Discrete time
- 1 step ≈ 5 minutes real time (derived from calibration)
- Episode length: 150 steps (12.5 hours) — covers full depeg–recovery cycle

## 2.3 Calibration protocol (Phase 2 gate)

Simulated Method of Moments (SMM) minimises:

$$L(\theta) = \sum_k w_k \left(\frac{m_k^{sim}(\theta) - m_k^{emp}}{m_k^{emp}}\right)^2$$

Target moments:
| Moment $m_k$ | Empirical value | Weight $w_k$ | Source |
|---|---|---|---|
| Calm OU half-life | 3.0 steps | 1.0 | IAQF analysis |
| Crisis contagion magnitude | 0.842 | 2.0 | `mean_abs_effect` (repo 1) |
| Baseline price vol | 0.003 | 2.0 | Tick data calibration |
| Cross-venue ρ̂ (crisis) | 0.576 | 0.5 | FEVD share (TVP-VAR, repo 1) |

Gate: ≥ 3/4 moments within tolerance before any intervention results are reported.

## 2.4 Counterfactual protocol

For each hub node h and shock scenario s:

1. **Baseline**: run N=40 independent episodes with shock schedule s and no intervention.
   Record contagion_magnitude_i for each seed i.

2. **Counterfactual**: run N=40 episodes with the same shock schedule and intervention on node h.
   Record contagion_magnitude_i^h for each seed i.

3. **Causal effect**:
   $$\Delta C_h = \bar{C}^{baseline} - \bar{C}^{intervened}$$
   $$SE = \sqrt{\frac{Var[C^{baseline}]}{N} + \frac{Var[C^{intervened}]}{N}}$$

4. Report $t_h = \Delta C_h / SE$ and one-sided p-value.

Node-to-intervention mapping:
| Node type | Intervention | Knob settings |
|---|---|---|
| DEX pool (curve, uniswap) | Circuit breaker | cb_threshold=0.02, cb_duration=20 |
| CEX venue (binance, coinbase, kraken) | Redemption gate | fee=200bps, queue=10, delay=6 steps |
| Mint/burn | Transparency boost | freq=1 step, noise=0.005 |
| Bridge / exchange flow | Flow reduction | arb_min_spread × 5 |

## 2.5 Agreement metrics

- **Spearman ρ**: rank correlation between predicted_importance and delta_contagion
- **Top-k overlap** (k=3,5): |top-k predicted ∩ top-k causal| / k
- **OLS regression**: delta_contagion ~ predicted_importance (slope, R², p-value)
