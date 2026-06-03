# Section 3: Calibration and Validation

## 3.1 Why calibration is the critical path

No intervention result is trustworthy until the ABM reproduces the empirical stylized facts
from the real episodes.  This is not a formality — it is the load-bearing step that separates
a credible causal claim from a simulation exercise.

The calibration section makes or breaks the paper.  Divergences are findings, not failures:
if the ABM cannot match a moment, that divergence tells us something about the mechanism
the real market has that the ABM lacks.  Document it honestly.

## 3.2 Calibration results

[Table 1 — to be filled after running make calibrate]

| Moment | Empirical | Simulated | Rel. error | Tolerance | Pass? |
|---|---|---|---|---|---|
| Calm OU half-life (steps) | 3.0 | [X.X] | [X%] | 30% | [✅/❌] |
| Crisis contagion magnitude | 0.842 | [X.XXX] | [X%] | 25% | [✅/❌] |
| Baseline price vol | 0.003 | [X.XXX] | [X%] | 30% | [✅/❌] |
| Cross-venue ρ̂ (crisis) | 0.576 | [X.XXX] | [X%] | 30% | [✅/❌] |

Overall gate: [PASS / FAIL] ([X]/4 moments within tolerance)

## 3.3 Calibrated parameters

[To be filled from calibration_report.json]

| Parameter | Value | Interpretation |
|---|---|---|
| reserve_speed (κ) | [X.XXX] | OU mean-reversion for backing ratio |
| reserve_vol (σ) | [X.XXX] | Backing ratio volatility |
| arb_min_spread | [X.XXXXX] | Min spread triggering arbitrage |
| noise_trade_prob | [X.XX] | Background order flow rate |
| noise_trade_size | [X,XXX] | Mean trade size (USD) |

## 3.4 Calibration overlay (Fig A1)

[Figure: simulated peg path (50 seeds, median + 10th/90th pct) vs. empirical basis-vs-usd
from stablecoin-contagion-network for the usdc_svb_2023 episode]

## 3.5 Divergences as findings

[To be filled — any moment that fails tolerance gets an explanation here]

If the calm OU half-life is too short: the noise traders are too active or the reserve OU
process adds unmodelled variance.  Mitigation: increase arb_min_spread (thicker bid/ask).

If the crisis contagion magnitude is too low: the shock scenarios are calibrated to reproduce
structural shocks (reserve haircut, liquidity drain), but real episodes may have coordination
dynamics (runs) that the heuristic agents don't capture fully.  Mitigation: train RL redeemers
(Phase 3) and verify they produce sharper run dynamics.

If cross-venue ρ̂ is too low: the ABM has only 2 pools (v1 scope); real episodes involve
5–11 nodes.  The limited venue count mechanically reduces cross-venue correlation.
Document as a model limitation (Section 7).
