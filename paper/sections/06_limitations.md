# Section 6: Limitations

## 6.1 ABM fidelity

The v1 ABM has 2 AMM pools and 1 redemption channel.  Real episodes involve
5–11 nodes across CEX, DEX, and cross-chain bridge venues.  Implications:

- **Cross-venue ρ̂ is mechanically lower** than the empirical 5-venue value.
  We treat this as a known limitation and report it in the calibration section.
- **Routing-around effects are underestimated**: with only 2 pools, intervention
  on pool 1 routes all flow to pool 2.  Real markets have more alternative routes,
  which would strengthen the spurious-hub result (less causal effect per node).
- **v2 extension**: add order-book venue + full MARL (Section 6.4 of ROADMAP).

## 6.2 Episode count

We calibrate against [5] real stress episodes across [3] event types.
[7] total real-node observations (usdc_svb: 3, terra: 2, ftx: 2).
This is small by econometric standards; our t-statistics account for this via
correct SE computation, but power is limited for hubs that appear in only 1 episode.

## 6.3 Synthetic scenario caveat

The StressBench scenarios are calibrated to reproduce empirical moments but are
not reconstructions of the real episodes.  The counterfactual results are therefore
statements about the *mechanism class* (e.g., "reserve haircut + liquidity withdrawal
shock") rather than about the specific historical episode.  This is standard for ABMs
and is a feature (the ABM can generalise) not a bug.

## 6.4 Calibration uncertainty

We report point estimates from the SMM optimiser.  A full sensitivity analysis
(bootstrap the calibration across shock scenarios and parameter neighbourhoods) is
in the robustness section.  If the intervention ranking is stable across the
calibration uncertainty band, the result is robust; if it flips, we report that.

## 6.5 RL scope

Phase 3 RL agents (PPO-trained arbitrageur/redeemer) were [trained / not yet trained]
at time of writing.  If trained, we verify they beat heuristics before including
RL agents in the main counterfactual sweep.  If not trained, the heuristic agents
provide a conservative lower bound on contagion (RL agents would likely produce
sharper run dynamics and higher peak depegs, strengthening the intervention effect).
