# Paper Outline — Causal Counterfactual Hubs in Stablecoin Contagion

## Working title
"From Correlation to Causation: An Agent-Based Counterfactual Analysis of Stablecoin Contagion Hubs"

## Target venue
ICAIF 2025 / AAAI AIFIN track; fallback: Journal of Financial Stability

## Core argument
GNN-based hub detection identifies nodes that are *correlated* with contagion propagation.
We ask: are those nodes *causally* responsible?  We answer by calibrating an ABM to the same
empirical episodes, then running per-node counterfactuals (intervene on hub X, measure Δcontagion).
Agreement between predicted and causal rankings validates the GNN; divergence identifies spurious hubs
driven by the volume/TVL confound identified in our own data audit.

## The two results
1. **Agreement result**: how well do GNN-predicted hub rankings predict ABM causal rankings?
   (Spearman ρ, top-k overlap, regression)
2. **Mechanism result**: for the largest divergence (spurious hub), explain mechanically why
   intervening on it doesn't reduce contagion — this is the XAI-style "black box vs. mechanism"
   contribution.

---

## Section structure

| Section | Content | Key figure/table |
|---|---|---|
| 1. Introduction | Prediction vs. causation; XAI spurious-correlation problem; policy stakes (GENIUS Act / MiCA) | — |
| 2. Method | Two-system design; ABM architecture; calibration strategy; counterfactual protocol | System diagram |
| 3. Calibration & validation | SMM fit; simulated-vs-empirical moments table; honest divergences | Table 1 (moments); Fig A1 (overlay) |
| 4. Hub agreement | Spearman, top-k, regression; headline scatter | Fig 1 (headline scatter); Table 2 |
| 5. Divergence case study | Mechanistic explanation of the largest spurious hub | Fig 2 (mechanism diagram) |
| 6. Intervention sweep | Knob × scenario sweep; welfare decomposition | Fig 3 (peg evolution); Fig 4 (welfare matrix) |
| 7. Limitations | ABM fidelity; n=7 episodes; synthetic caveat; calibration uncertainty | — |
| 8. Conclusion | Policy implications; regulatory framing | — |
| Appendix | Reproducibility: commands, seeds, hashes | — |

---

## Figure plan (matching artefact list from ROADMAP)

| Figure | Description | Script |
|---|---|---|
| Fig 1 | Headline: predicted importance vs. causal effect (one point per hub) | `run_joint_analysis.py` |
| Fig 2 | Mechanism diagram: why the spurious hub doesn't propagate causally | Hand-drawn / TikZ |
| Fig 3 | Per-intervention peg evolution across StressBench scenarios | `analysis/figures.py` |
| Fig 4 | Welfare decomposition matrix (agent type × intervention) | `analysis/figures.py` |
| Fig A1 | Calibration overlay: simulated vs. empirical peg path | `run_calibration.py` |
| Fig A2 | RL training curves (if Phase 3 complete) | `rl/ppo.py` |

---

## Table plan

| Table | Description |
|---|---|
| Table 1 | Calibration moments: simulated vs. empirical (±tolerance, pass/fail) |
| Table 2 | Hub agreement metrics: Spearman, top-3 overlap, top-5 overlap, OLS slope |
| Table 3 | Causal hub ranking with delta_contagion, SE, t-stat, p-value |
| Table 4 | Intervention sweep results: contagion magnitude by intervention × scenario |
| Table 5 | Welfare decomposition: mean P&L by agent type × intervention |
| Table 6 | Robustness: intervention ranking under calibration uncertainty and seed variation |
