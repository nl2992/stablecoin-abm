# TODO — stablecoin-abm

Goal: a calibrated agent-based stablecoin market that **causally validates** the GNN's
hub predictions and measures which interventions reduce depeg contagion and at what cost.
Template paper: Gu, Wang, Wellman et al. — *The Effect of Liquidity on the Spoofability
of Financial Markets* (ICAIF'24). Killer framing (already in `paper/outline.md`):
**"From Correlation to Causation."**

The engine, agents, calibration, RL, and counterfactual code all exist. **What is missing
is executed results** — `experiments/results/` is empty. Dependency-ordered below; the
gates are non-negotiable (intervention results are meaningless before calibration passes).

---

## P0 — Calibration is the credibility gate (blocks everything downstream)

- [ ] Run `make calibrate` and produce the **simulated-vs-empirical moments table**:
      OU half-life (calm ~3.2 min → crisis ~600 min), cross-asset ρ̂, LP-loss magnitude,
      peg-deviation distribution. Targets live in `configs/calibration_targets.json`.
- [ ] **Gate:** simulated moments fall within empirical bands on ≥3 of 4 stylized facts.
      If not, fix engine/agents — do **not** proceed to interventions.
- [ ] Produce the **calibration overlay figure** (simulated vs empirical peg path). This
      single figure sells the ABM's realism — it is the figure reviewers look for first.
- [ ] Finish the **per-agent myopic-correctness tests** (ROADMAP Phase 1, unchecked):
      arbitrageur trades toward peg when profitable after fees, etc.

## P1 — The join: causal validation of GNN hubs (your unique contribution)

- [ ] Load the GNN hub ranking (`counterfactual/hub_loader.py` reads
      `../stablecoin-contagion-gnn/exports/hub_ranking_v1_*.json`).
- [ ] Run **per-node counterfactuals** (`scripts/run_counterfactuals.py`): for each hub,
      intervene (remove/dampen the node) and measure Δcontagion with standard errors.
- [ ] Compute the **agreement result** (`scripts/run_joint_analysis.py`): Spearman ρ,
      top-k overlap, OLS slope between GNN-predicted importance and ABM causal effect.
      This is the headline scatter (Fig 1).
- [ ] **Divergence case study:** take the largest predicted-vs-causal gap — the spurious
      hub the GNN flagged via the volume/TVL confound — and explain *mechanically* why
      intervening on it does **not** reduce contagion. This is the XAI money result.

## P2 — Intervention sweep (the policy paper)

- [ ] Run `make sweep` — scenarios × interventions × many seeds, with standard errors.
      Four knobs: reserve transparency, redemption gating, circuit breaker, LP incentives.
- [ ] Outcomes per run: contagion magnitude, peg-recovery half-life, LP impermanent loss,
      **welfare by agent type** (the decomposition matrix, Fig 4).
- [ ] **Find the regime flip** (Gu's "critical liquidity" analog): the transparency/gating
      threshold where contagion behaviour qualitatively changes. Regime discovery — not
      just a monotone curve — is what made the Gu paper land.
- [ ] **Mechanism analysis** (Gu §8 analog): for each effective intervention, explain *why*
      it works, not just the effect size.

## P3 — RL (make it non-decorative)

- [ ] Train PPO arbitrageur/redeemer (`make train`) vs the heuristic background population.
- [ ] **Gate:** RL policy beats its heuristic counterpart on its own objective,
      reproducibly. If it doesn't, say so and keep RL as a robustness arm — don't oversell.
- [ ] Re-run the key intervention under RL-adaptive agents: do interventions that work
      against heuristic agents still work when agents *adapt*? (Lucas-critique robustness —
      a depth point none of the template papers fully address.)

## P4 — Breadth & robustness (Continuous-Time-RL lesson: many scenarios)

- [ ] Robustness sweep: vary agent-population mix, shock severity, calibration uncertainty.
- [ ] Anchor **every** comparison against no-intervention + no-RL baseline.
- [ ] Confirm intervention *rankings* survive calibration uncertainty and seed variation
      (Table 6). Rankings that flip under uncertainty must be reported as such.

## P5 — Paper + release

- [ ] Assemble paper from `paper/sections/` (intro, methods, calibration, hub-agreement,
      divergence case study, threats, negative-result framing — all stubs exist).
- [ ] Tie policy implications to **GENIUS Act / MiCA** (reuse IAQF regulatory framing).
- [ ] Release simulator + configs + sweep runner; reproducibility appendix
      (`paper/appendix_reproducibility.md`).

---

## Credibility checklist (from ROADMAP — reviewer-facing, non-negotiable)
1. Calibration validation passes (sim reproduces empirical half-lives + ρ̂).
2. GNN-vs-ABM hub agreement computed; ≥1 spurious hub explained mechanically.
3. ≥1 intervention shows significant contagion reduction **with a stated mechanism**.
4. Intervention rankings robust to calibration uncertainty + seeds (with SEs).
5. RL beats heuristics, or RL is explicitly framed as robustness only.
6. No-intervention + no-RL baseline anchors all comparisons.
