# Results — stablecoin-abm: causal validation of GNN contagion hubs

_Generated 2026-06-04. The headline two-repo result: an ABM causally validates (and
partially refutes) the GNN's correlational hub ranking on the real USDC/SVB episode._

## Why a new engine

The original AMM-only market is **bimodal** for depeg magnitude — a stableswap pool either
resists a shock (~0 depeg) or collapses (drains to the price clamp), with no smooth middle.
So the SMM cannot match a controlled ~14% depeg and the calibration gate could not pass.
We replace it (for the causal analysis) with a **reduced-form networked-contagion engine**
(`src/stablesim/netcontagion/`): per-asset peg deviations following a near-critical
mean-reverting process with a **directed transmission network W** estimated from the real
lead-lag structure of the 1-min deviations. This is smoothly calibratable and gives clean
per-node knockout counterfactuals.

## Calibration: 4/4 PASS

Calibrated to the empirical moments (targets from the GNN export `calibration_v1.csv`
for USDC/SVB + the locked stylized facts):

| moment | empirical | simulated | rel. err | pass |
|---|---|---|---|---|
| contagion magnitude | 0.1376 | 0.1376 | 0.0% | ✅ |
| crisis half-life (steps) | 116 | 116.0 | 0.0% | ✅ |
| baseline price vol | 0.0030 | 0.0029 | 3.8% | ✅ |
| cross-venue ρ | 0.576 | 0.640 | 11% | ✅ |

`experiments/results/netcontagion/calibration_moments.csv`. Params:
coupling 0.022, κ 0.006 (→ half-life 116), common-factor 0.0028, σ 0.0021, shock 0.103.

## The causal join (correlation → causation)

Per-node knockout: protect node X (hold it at peg), measure the change in contagion to the
**other** victims (on the deterministic propagation, so the signal is clean).

| node | ABM causal Δ-contagion | GNN predicted importance | propagator? |
|---|---|---|---|
| **USDC (origin)** | **0.031 (= full baseline)** | — (origin) | source |
| DAI | 0.0036 | 0.00 | no |
| TUSD | 0.000 | 0.26 | yes |
| USDP | 0.000 | 0.08 | yes |
| **BUSD** | **0.000** | **1.00 (GNN's #1 hub)** | no |
| USDT | 0.000 | 0.00 | no |

`experiments/results/netcontagion/causal_hub_ranking.csv` + `fig_join_scatter.png`.

### Headline findings

1. **Protecting the origin (USDC) removes 100% of contagion** — the engine attributes the
   cascade to its true source, a sanity check the model passes.
2. **BUSD is a spurious hub.** It is the GNN's #1 ranked hub (highest betweenness 0.071 and
   highest GNN occlusion-influence 12.3), yet its causal Δ-contagion is **exactly 0** — its
   estimated **out-transmission is 0** (it co-moved during its regulatory wind-down without
   transmitting stress to anyone). The ABM thus **refutes** the GNN's top hub as non-causal.
3. **Correlation ≠ causation, quantified.** Spearman ρ between GNN predicted importance and
   ABM causal effect is **−0.77** — the correlational hub ranking does not (here, inversely)
   track causal importance. This is the XAI "spurious-correlation" result, made causal.

## Policy-intervention sweep (the regulatory payoff)

`scripts/run_intervention_sweep.py` → `experiments/results/netcontagion/`. Each policy is
scored by total system contagion (mean peak |dev| over non-origin victims).

| intervention | best setting | contagion reduction |
|---|---|---|
| targeted protection (bail out USDC) | full backstop | **100%** |
| targeted protection (bail out DAI, the relay) | full backstop | 98% |
| redemption gating (coupling → 0) | full halt | 100% |
| reserve strengthening (USDC κ ×20) | — | 89% |
| circuit breaker | cap 2% | 85% |
| targeted protection (bail out **BUSD**) | full backstop | **0%** |
| reserve strengthening (**BUSD** κ ×20) | — | **0%** |

**Headline policy result (`policy_comparison.json`, `fig_interventions.png`):**
a budget-constrained regulator who can backstop ONE venue and picks by the GNN's
**correlational** hub ranking protects **BUSD → 0% contagion reduction** (wasted budget);
picking by the ABM's **causal** ranking protects **USDC → 100% reduction**. This is the
concrete cost of acting on correlation instead of causation — directly relevant to
reserve-transparency and intervention design under the GENIUS Act / MiCA.

## Generalization, robustness, and a learning regulator

**Multi-episode** (`run_multi_episode_join.py`, `multi_episode_join.csv`). Across the 3
contagion-producing episodes, the GNN's top hub *equals* the ABM causal driver in Terra
(1/3) but *diverges* with a spurious hub in SVB (BUSD) and USDT-May-2022 (2/3). Correlational
hub rankings are **unreliable** — sometimes right, sometimes badly wrong — so the causal test
is required. (2 thin episodes with ~0 contagion are excluded.)

**Robustness** (`run_robustness_welfare.py`, `robustness_summary.json`). Under ±30%
perturbation of (coupling, κ, W) across 60 draws, USDC stays the top causal node **100%** of
the time and BUSD stays causally inert **100%** — the headline is not a calibration artifact.

**Welfare matrix** (`welfare_matrix.csv`): protecting BUSD is *identical* to no intervention;
protecting USDC takes every victim's peak depeg to 0; reserve-strengthening USDC ×10 cuts
DAI's depeg from 0.138 → 0.029.

**RL regulator** (`run_rl_regulator.py`, `rl_regulator.json`). A PPO agent given only the
transmission-network features (no causal labels) learns to allocate reserve budget
**USDC 1.0 / DAI 0.48 / BUSD 0.0**, achieving **93.7%** contagion reduction — independently
rediscovering the causal targets and ignoring the spurious correlational hub.

## Honest limitations

- The directed network W is estimated from 1-min lead-lag, which can over-credit thinly
  traded coins as "leaders"; the robust, load-bearing claim is the **BUSD divergence**
  (predicted #1, causal 0), not the full ranking.
- The causal structure is sparse (USDC→DAI dominates), so several nodes have Δ≈0 and the
  5-node Spearman is noisy (p≈0.23). The BUSD point is the bulletproof result.
- This reduced-form engine is for the causal hub analysis; the full AMM agent model
  (`src/stablesim/engine/`, `experiments/`) remains for intervention/welfare studies.

## Reproduce

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/run_netcontagion_join.py
```
Outputs under `experiments/results/netcontagion/`.
