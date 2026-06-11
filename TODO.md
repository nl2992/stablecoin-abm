# TODO — stablecoin-abm

> **STATUS (2026-06-11): SUBMISSION-READY.** All four submission gates below are passed; paper compiles at 8pp (ICAIF '26, ACM sigconf). The calibrated counterfactual refutes the companion GNN's top hub (BUSD causal effect = 0; USDC = 100%), with DebtRank, placebo, and strategic-agent robustness checks and a seed-averaged RL regulator. Planning notes below are a historical record of the gate-clearing pass.

---

# Calibrated Networked Stablecoin ABM — From Correlation to Causation

## The paper's claim and what would falsify it

**Primary claim**: USDC is the causal origin of SVB contagion (causal effect = 100%); BUSD
is a correlational hub with no causal effect (0%). The GNN's hub ranking is wrong for SVB
because correlation is not causation.

**Secondary claim**: A PPO regulator, trained without knowledge of this causal ranking,
independently learns to protect USDC and ignore BUSD — validating the causal finding from
a different direction.

**What a reviewer needs to see to believe it**:
1. The BUSD=0%, USDC=100% finding holds across multiple episodes, not just SVB.
2. The RL regulator reliably converges to USDC protection across seeds — not one lucky run.
3. The "causal vs correlational" policy difference is quantified: K=1 budget on GNN ranking = 0%
   reduction; K=1 on ABM ranking = 100% reduction.
4. The calibration claim (4/4 moments matched) has uncertainty bounds, not just point estimates.

Without (1), the paper's generalization claim rests on a single episode. Without (2), "RL
independently learns" could be a single seed. Without (3), the policy recommendation is verbal
not numerical. Without (4), calibration matching could be a point estimate that falls apart with
noise.

---

## SUBMISSION GATES — must be done before any draft goes out

All core results are already committed. These gates are about making existing results
statistically defensible and multi-episode.

---

### GATE 1 — Multi-episode agreement table

**Why this gate exists**: The paper's generalization claim is "the ABM validates the GNN
across episodes." That claim needs a table, not a sentence. The data is already in
`multi_episode_join.csv` — this is a synthesis task, not a modeling task.

**What to code**: `scripts/compile_multi_episode_table.py`
```
Load: experiments/results/netcontagion/multi_episode_join.csv

For each of the 5 episodes:
  - classify: high_contagion (>= threshold) or low_contagion
  - record: GNN top hub, ABM causal top hub, agreement (True/False)
  - note: spurious_hub label if GNN hub has ABM causal effect < 10%

Expected structure:
  UST_Terra:      high_contagion, GNN=USDC, ABM=USDC, agreement=True
  USDC_SVB:       high_contagion, GNN=BUSD (spurious), ABM=USDC, agreement=False
  USDT_May2022:   high_contagion, GNN=USDC (spurious), ABM=USDT, agreement=False
  DAI_FTX:        low_contagion, N/A (model not applicable below threshold)
  BUSD_winddown:  low_contagion, N/A

Print: LaTeX table \label{tab:multi_episode}
Save: experiments/results/netcontagion/multi_episode_table.tex
```

**What to run**:
```bash
python scripts/compile_multi_episode_table.py
```

**Target result**: 5-row table showing 2 high-contagion episodes with spurious GNN hubs
(confirmed), 1 episode where GNN IS correct (UST/Terra), 2 low-contagion episodes not
applicable. The UST/Terra concordance is the honest positive — it shows the ABM as a real
test, not one that always says "GNN is wrong."

Key sentence for paper: "Of the three high-contagion episodes, two contain a spurious GNN hub;
in the third (UST/Terra), the GNN hub is the causal origin — a concordance that validates the
ABM as a discriminating test rather than a refutation machine."

**Write into paper**: New Table in §4 (Multi-Episode Validation), replacing any single-episode
framing. This table is the paper's empirical backbone.

---

### GATE 2 — RL training convergence with 5 seeds

**Why this gate exists**: `rl_regulator.json` shows the final allocation but not the learning
curve. "PPO independently learns to protect USDC" becomes unverifiable if only one seed is
reported. Five seeds with mean ± std is the minimum for this claim to be credible.

**What to code**: `scripts/run_rl_convergence.py`
```
Re-run PPO training with 5 random seeds (or load checkpoints if available):

For each seed, record per-episode training trajectory:
  - cumulative reward
  - USDC allocation (should rise from ~1/N to ~1.0)
  - BUSD allocation (should fall from ~1/N to ~0.0)
  - contagion_reduction_pct

Save: experiments/results/netcontagion/rl_convergence.csv

Plot 3 panels:
  (a) USDC allocation vs training timestep; all 5 seeds overlaid + mean
  (b) BUSD allocation vs training timestep; all 5 seeds overlaid + mean
  (c) Contagion reduction % vs timestep; converges to ~93%

Report: mean ± std across 5 seeds for final contagion_reduction_pct, USDC allocation,
  BUSD allocation
```

**What to run**:
```bash
python scripts/run_rl_convergence.py --n_seeds 5 --timesteps 12000
```

**Target result**: All 5 seeds converge to USDC allocation >= 0.9 and BUSD allocation <= 0.1.
Mean contagion reduction across seeds >= 90% ± 5%. If any seed fails to converge, report and
investigate.

**Write into paper**: Training convergence figure in §5.3 (RL Regulator). Key sentence:
"Across 5 random seeds, PPO converges to >= 0.9 USDC allocation within X timesteps (mean
contagion reduction Y ± Z%), confirming the causal discovery is reproducible."

---

### GATE 3 — Budget-constrained allocation comparison (K=1, 2)

**Why this gate exists**: The paper's policy claim is "a regulator following the correlational
GNN ranking wastes its intervention budget." This must be quantified as a table, not stated
as a conclusion. K=1 (one intervention) is the strongest possible demonstration: GNN = 0%
reduction, ABM = 100% reduction, same budget.

**What to code**: `scripts/run_budget_optimization.py`
```
Load: experiments/results/netcontagion/intervention_sweep.csv

For K = 1, 2, 3:
  Enumerate all C(n, K) venue combinations
  
  For each combination: extract contagion_reduction_pct from intervention_sweep.csv
  
  Compute contagion_reduction for:
    1. Greedy optimal K-subset (upper bound)
    2. GNN-guided K-subset (protect top K by correlational importance)
    3. ABM-guided K-subset (protect top K by causal Delta_X)
    4. RL regulator top-K by learned allocation from rl_regulator.json

  Save: experiments/results/netcontagion/budget_allocation.csv

Expected K=1 results:
  Optimal:     ~100%
  ABM-guided:  100% (USDC)
  RL:          ~94% (learned USDC)
  GNN-guided:  ~0% (BUSD, the spurious hub)
```

**What to run**:
```bash
python scripts/run_budget_optimization.py
```

**Target result**: At K=1: GNN=0%, ABM=100%, RL≈94%. At K=2: ABM≈100%, GNN still < 50%.
The numerical gap at K=1 is the paper's policy headline — it is not rhetorical once quantified.

**Write into paper**: New Table in §5 (Policy Implications): "Contagion reduction (%) under
three allocation strategies, by budget K." Caption: "A regulator following the GNN ranking
achieves X% reduction with K=1; the ABM-guided regulator achieves 100% with K=1 alone."

---

### GATE 4 — Calibration uncertainty bounds

**Why this gate exists**: "4/4 moments matched" is a point estimate. A reviewer correctly asks:
does it hold across simulation noise? Running 100+ simulations and reporting the CI around each
moment confirms calibration is robust, not lucky.

**What to code**: `scripts/run_calibration_uncertainty.py`
```
Re-run calibrated simulation N = 500 times with different random seeds
(same calibrated parameters, different stochastic realizations)

For each of the 4 calibration moments, record:
  - Mean across 500 runs
  - 95% CI (2.5th and 97.5th percentile)
  - Empirical target value

Report: table of (moment_name, empirical_target, sim_mean, sim_95CI_lower, sim_95CI_upper)
        Flag: is the empirical target within the simulation CI?

Save: experiments/results/netcontagion/calibration_uncertainty.csv
```

**What to run**:
```bash
python scripts/run_calibration_uncertainty.py --n_sims 500
```

**Target result**: All 4 empirical targets fall within the simulation 95% CI. Report as:
"Calibration moments: 4/4 empirical targets lie within the simulation 95% CI across 500
runs [Table X]."

**Write into paper**: Replace "4/4 moments matched" with the table. The statement becomes
auditable, not a claim.

---

## STRONG — Should be done; significantly strengthens submission

---

### S1 — Intervention timing sensitivity

**Closes**: "Why does early warning matter if the intervention is instantaneous?"

**What to code**: `scripts/run_intervention_timing.py`
```
Using calibrated SVB model:
For delay_steps in {0, 5, 10, 20, 40, 80} (each step ≈ 5 min):
  Apply USDC backstop at step (40 + delay_steps) instead of at shock onset (step 40)
  Measure: contagion_reduction_pct vs no-intervention baseline

Plot: pct_reduction vs delay (minutes); mark the "50% effectiveness" threshold
Save: experiments/results/netcontagion/intervention_timing.csv
```

**What to run**:
```bash
python scripts/run_intervention_timing.py
```

**Target result**: Intervention >= 50% effective up to D minutes of delay (target: D >= 150 min
= 2.5 hours). This directly connects to the companion GNN/HMM papers' detection leads.

**Write into paper**: Figure in §5 (Policy Implications). Caption: "Effectiveness threshold at
D minutes means the companion paper's 24h early warning leaves ample response time."

---

### S2 — Partial backstop cost-effectiveness curve

**Closes**: "Full backstop is unrealistic — what about partial interventions?"

**What to code**: `scripts/run_partial_backstop.py`
```
For USDC (causal origin):
  Sweep kappa multiplier: {1.5, 2.0, 3.0, 5.0, 10.0, full}
  Record: pct_reduction, intervention_cost_proxy (multiplier × USDC_mcap)

For BUSD (spurious hub):
  Same sweep → shows ANY level of BUSD intervention ≈ 0% reduction

Plot: pct_reduction vs kappa multiplier; two lines (USDC vs BUSD)
Save: experiments/results/netcontagion/partial_backstop.csv
```

**What to run**:
```bash
python scripts/run_partial_backstop.py
```

**Target result**: USDC 5× reserve: ~64% reduction at fraction of full-backstop cost.
BUSD 10×: < 5% reduction. The two lines diverge immediately from kappa=1.5.

**Write into paper**: Figure in §5 "Intervention cost-effectiveness curve." The BUSD flat-near-
zero line is as important as the USDC slope.

---

### S3 — Welfare decomposition by agent type

**Closes**: "Who wins and who loses under each intervention?"

**What to code**: `scripts/run_welfare_analysis.py`
```
Load: experiments/results/netcontagion/welfare_matrix.csv

For each scenario (no-intervention, USDC backstop, BUSD backstop):
  Compute welfare by agent type: stablecoin holders, arbitrageurs, liquidity providers

Pareto check: does USDC backstop Pareto-dominate no-intervention?
  (every agent type at least as well off)

Plot: 3-column bar chart (scenarios) × agent types as grouped bars
Save: experiments/results/netcontagion/welfare_analysis.json
```

**What to run**:
```bash
python scripts/run_welfare_analysis.py
```

**Target result**: USDC backstop Pareto-dominates no-intervention. BUSD backstop: BUSD holders
benefit; all other agents no better off. This is the strongest policy statement in the paper.

**Write into paper**: Table in §5 — welfare decomposition matrix with Pareto-dominance note.

---

## EXTENSIONS — Do if time permits before camera-ready

- **Cross-crisis RL policy transfer (Plan E)**: Test SVB-trained RL on Terra parameters.
  Expected to work because both crises have USDC as causal origin. Good if paper needs more
  generalization evidence beyond the multi-episode table.

- **GENIUS Act / MiCA policy translation (Plan H)**: Map 3 numbered recommendations to
  specific provisions. High-value if submitting to a venue where regulatory relevance is scored.

---

## Ordered execution sequence

```
GATE 1 (multi-episode table) → first; uses existing data, no new runs required
GATE 4 (calibration uncertainty) → can run in parallel with GATE 1; longest compute time
GATE 2 (RL convergence) → run after confirming GATE 1; 5-seed training takes ~30 min
GATE 3 (budget allocation) → depends on intervention_sweep.csv being confirmed; fast
S1, S2, S3 → run after all gates pass; can parallelize all three
Extensions → only if time permits after core is done
```

---

## Non-negotiable checklist before submission

- [ ] Multi-episode table with 5 episodes, concordance column, spurious-hub labels (GATE 1)
- [ ] RL convergence figure with 5 seeds; mean ± std contagion reduction reported (GATE 2)
- [ ] Budget table: GNN vs ABM at K=1 showing 0% vs 100% reduction (GATE 3)
- [ ] Calibration moments: 4/4 empirical targets within 95% CI across 500 simulations (GATE 4)
- [ ] The UST/Terra concordance (where GNN IS right) is reported as a concordance, not omitted
- [ ] No-intervention baseline anchors all comparison tables

---

## Loop run 2026-06-11 (Loop 1)

### Status: prose quality pass complete; all 4 GATES confirmed in paper

**Completed this run:**
- Fixed calibration-uncertainty semicolon (CI`; BUSD` → CI`. BUSD`)
- Fixed seed-stable semicolon (seed-stable`:` → seed-stable`.`)
- Rewrote Step-two colon (Step two`:` → `checks each candidate's`)
- Fixed cost-of-skipping colon (concrete`:` → concrete`.`)
- Fixed correlational-not-causal colon (causal`:` → causal`.`)
- Fixed lesson-generalises colon (generalises`:` → generalises`.`)
- Fixed learnable colon (correct`:` → correct`.`)
- Fixed two prose semicolons (discarded`;` → discarded`.`, `$93.7\%$; seed-stable` → `$93.7\%$, seed-stable`)
- Committed `86a1f0c`, pushed to GitHub

**GATES status**: GATE 1 (multi-episode table) ✅, GATE 2 (RL convergence) ✅, GATE 3 (budget table) ✅, GATE 4 (calibration uncertainty) ✅
**Task 3 status**: All prior-session additions (Terra case study, state-dependent noise, DebtRank, CIs) confirmed in paper — COMPLETE
**Remaining**: None
