# Section 1: Introduction

## Hook
Stablecoin depegs propagate across venues in minutes.  Regulators and protocol designers need
to know *where to intervene* — which hubs to circuit-break, gate, or force transparent.
Graph neural networks can rank these hubs by predicted importance.  But is importance in a
correlation graph the same as causal propagation power?  We show it often is not.

## The prediction–causation gap
- GNN hub detection identifies nodes whose activity correlates with contagion (high eigenvector
  centrality, high FEVD share, high out-degree).
- But correlation reflects joint response to a common shock, not causal propagation.
- The classic example: exchange-flow nodes appear as hubs because they are *active during* stress,
  not because *intervening on them* would have reduced stress.
- This is the XAI spurious-correlation problem applied to financial networks.

## Our approach: causal counterfactual oracle
We build a calibrated ABM that reproduces the same empirical episodes the GNN was trained on.
Then, for each GNN-predicted hub, we run a counterfactual: remove or gate that hub and measure
Δcontagion in the ABM.  The difference reveals whether the predicted hub is causally potent.

## What we find (preview)
[To be filled after results]
- Spearman ρ between predicted and causal rankings: [X.XX] (p = [X.XX])
- Top-3 overlap: [X%]
- Largest spurious hub: [node_id] — high GNN importance, low causal effect
- Mechanism: [volume/TVL confound OR routing-around OR peg-mechanism argument]

## Policy stakes
The GENIUS Act (2025) and MiCA (EU 2024) both contemplate circuit-breaker requirements for
systemically important stablecoins.  If the targeted nodes are spurious hubs, the circuit
breaker is costly (LP exits, redemption disruption) without reducing contagion.  Our mechanism
analysis specifies which intervention lever works on which hub type.

## Contributions
1. First causal counterfactual analysis of stablecoin contagion hubs.
2. Empirical validation of GNN hub predictions via ABM counterfactuals.
3. Mechanism explanation for the largest GNN–ABM divergence.
4. Welfare decomposition showing who pays for each intervention design.
