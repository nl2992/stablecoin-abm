# Section 5: Divergence Case Study

## 5.1 The largest spurious hub

[To be filled from experiments/results/joint_analysis/divergence_case.json]

**Node**: [node_id]
**GNN-predicted importance**: [X.XXX] (rank [Y] of [N])
**ABM causal effect**: Δcontagion = [X.XXXX] (rank [Z] of [N])
**Divergence score**: [X.XXX]

## 5.2 Why it appears as a hub in repo 1

[node_id] has high [eigenvector / out_degree / betweenness] centrality in the empirical
contagion graph for the [event_id] episode.  During the stress event:

- [Empirical fact 1: e.g., "trade volume through this node spiked 8× above baseline"]
- [Empirical fact 2: e.g., "Granger-causal relationship to 4 other nodes significant at FDR 5%"]
- [Empirical fact 3: e.g., "FEVD share of X% at horizon 10"]

These correlations are real and statistically sound.  But they reflect joint response to
the common shock, not causal propagation from this node.

## 5.3 Why intervening on it doesn't reduce contagion in the ABM

**Mechanism** (following the Gu §8 belief-stabilisation approach):

The ABM counterfactual applies [circuit_breaker / redemption_gate / transparency_boost] to [node_id].
Contagion magnitude changes by Δ = [X.XXXX] ± [SE] (t = [X.XX], p = [X.XX]).

Explanation:

1. **Routing-around**: [If CEX venue] When the redemption channel for [node_id] is gated,
   redeemers route to alternative venues.  Arbitrageurs re-establish the peg through the
   surviving AMM pools.  Net effect on cross-venue spread: negligible.

2. **Volume/TVL confound**: [If exchange-flow/bridge] [node_id] appears central because
   its flow volume is large.  But large volume in the ABM means the noise traders proxying
   this flow don't drive prices — they are price-takers relative to the AMM invariant.
   Reducing their activity doesn't move the AMM price because the invariant absorbs the
   flow change into reserves, not into price.

3. **Shock absorption**: The stableswap AMM with high A (=100) has very tight price impact
   near equilibrium.  A node that contributes large but balanced flows (equal buy/sell)
   appears active but does not push the price off peg.  The GNN's centrality measures
   do not distinguish directional from balanced flow.

## 5.4 Mechanism diagram (Fig 2)

[Hand-drawn or TikZ diagram showing:
  left panel: correlational view (GNN sees both node and depeg move together)
  right panel: causal view (ABM shows the common shock causes both; blocking node doesn't stop depeg)]

## 5.5 Regulatory implication

Targeting [node_id] with a circuit breaker would impose real costs:
- LPs exit the gated pool (welfare_lp: [−$X per episode])
- Redeemers face delay costs (welfare_redeemer: [−$X per episode])
- Net contagion reduction: [negligible / X%]

A circuit breaker on [the correct causal hub, rank 1] achieves [Y%] contagion reduction
at the same or lower welfare cost.  Mis-targeting the spurious hub wastes the regulatory
budget.
