# Section 8: Negative-Result Framing Contingency

Pre-register both framings before running. Knowing which story you're telling
before seeing results keeps the paper honest. Both are publishable.

---

## Scenario A: Predicted and causal hubs AGREE (ρ > 0.6, top-3 overlap ≥ 67%)

**The story:** "GNN interpretability validated by causal oracle."

The GNN attention/centrality scores correctly identify causally potent hubs.
The XAI concern (spurious correlation) does not materialise in this domain.
Eigenvector centrality / FEVD shares are sufficient statistics for causal propagation power.

**Headline:** "We show that graph-neural-network hub predictions for stablecoin contagion are
causally valid: Spearman ρ = [X] (95% CI [Y, Z]) between predicted importance and
ABM counterfactual effect. Circuit-breaker policy should target the top-ranked hubs."

**Contribution framing:**
1. First causal validation of GNN-based hub detection in decentralised finance.
2. The ABM counterfactual framework can serve as a pre-deployment test for future XAI systems.
3. Welfare decomposition shows which agent type bears the cost of targeting the correct hub.

**Risks:** Agreement may hold only for synthetic hubs, not real-episode hubs. If so:
"Agreement is driven by synthetic scenarios; real-episode hubs show weaker agreement [ρ_real = X],
consistent with the spurious-hub audit in repo 1." This is still publishable — the split
between synthetic and real is itself a finding.

---

## Scenario B: Predicted and causal hubs DISAGREE (ρ < 0.4 or top-3 overlap = 0)

**The story:** "Attention misleads; here's the mechanism."

GNN hub scores reflect activity patterns during the stress episode (correlation with the shock)
rather than causal propagation power. The volume/TVL confound identified in repo 1's data audit
is the leading mechanistic explanation.

**Headline:** "We find that GNN-predicted hub importance has Spearman ρ = [X] (95% CI [Y, Z])
with ABM causal effect — consistent with near-independence. The leading spurious hub is [node_id],
whose high centrality reflects [exchange flow volume / TVL / balanced buy-sell flow] during the
stress episode rather than causal propagation. Mechanically: [routing-around / flow-absorption /
balanced-flow argument]. Targeting [node_id] for circuit-breaking would cost [welfare_lp] per
episode in LP impermanent loss with no measurable contagion reduction."

**Contribution framing:**
1. First demonstration that GNN hub scores are spurious predictors of causal contagion power
   in the stablecoin domain.
2. Mechanism explanation distinguishing correlational from causal hubs.
3. Policy implication: circuit-breaker regulation targeting GNN-predicted hubs may be
   misallocated; ABM counterfactual validation should precede deployment.

**Risks:** The disagreement may be partly due to ABM fidelity limitations (Section 7.1).
If the agreement is weak and the ABM has known calibration divergences, explicitly bound the
conclusion: "Subject to ABM fidelity limitations documented in Section 3.5, we find evidence
that..."

---

## Decision rule (pre-registered)

Run joint analysis after counterfactuals. Based on Spearman ρ (all hubs, bootstrapped CI):

| ρ (all hubs) | ρ (real-episode hubs) | Story |
|---|---|---|
| > 0.6 | > 0.6 | Scenario A: GNN validated |
| > 0.6 | < 0.4 | Mixed: agreement is synthetic-driven; XAI caveat applies to real episodes |
| 0.4–0.6 | any | Moderate agreement; report both validated and spurious hubs |
| < 0.4 | < 0.4 | Scenario B: GNN misleads; mechanism section is the paper |

Do NOT change the framing after seeing the number. The above table is the pre-registered
decision rule. Record the actual ρ and which row it falls into before writing the Discussion.
