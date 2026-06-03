# Section 4: Hub Agreement Results

## 4.1 Predicted vs. causal hub ranking

[Table 2 — to be filled after running scripts/run_joint_analysis.py]

| Metric | Value | Interpretation |
|---|---|---|
| Spearman ρ | [X.XX] | — |
| p-value | [X.XXX] | — |
| Top-3 overlap | [X%] | — |
| Top-5 overlap | [X%] | — |
| OLS slope | [X.XX] | Δcontagion per unit predicted_importance |
| OLS R² | [X.XX] | — |

## 4.2 Headline figure (Fig 1)

[scatter: x = predicted_importance (eigenvector centrality, repo 1),
          y = delta_contagion (ABM counterfactual, repo 2),
          colour = node role (originator/amplifier/mixed),
          star = real-episode nodes]

Points near the diagonal: GNN predictions are causally valid.
Points in the lower-right: spurious hubs.

## 4.3 Interpretation

[Template — fill in after results]

**If ρ > 0.7**: Strong agreement.  The GNN hub scores are causally informative.
Eigenvector centrality captures the causal propagation structure, not just
volume/TVL correlation.  Recommended interventions target the top-ranked hubs.

**If 0.4 < ρ < 0.7**: Moderate agreement.  DEX pool hubs (curve_3pool, uniswap)
show strong agreement; CEX venue hubs show weaker agreement (routing-around
effect: blocking one redemption venue redirects flow to others).

**If ρ < 0.4**: Weak agreement.  GNN hub scores reflect the volume/TVL confound
identified in the repo-1 audit — high-activity nodes during a stress episode
score high on centrality measures but do not causally amplify contagion.
The XAI claim is not supported.

## 4.4 Causal hub ranking (Table 3)

[To be filled from experiments/results/counterfactual/causal_hub_ranking.csv]

| Rank | Node | Type | Role | Δcontagion | SE | t-stat | p (one-sided) | Predicted rank |
|---|---|---|---|---|---|---|---|---|
| 1 | [node] | [type] | [role] | [X.XXXX] | [X.XXXX] | [X.XX] | [X.XXX] | [Y] |
| ... | | | | | | | | |
