# Section 7: Threats to Validity

Honest documentation of the limitations that reviewers will check.
Pre-registering these before seeing results prevents selective reporting.

---

## 7.1 ABM fidelity ceiling

**Threat:** The ABM has 2 pools and 1 redemption channel. Real episodes involve 5–11 nodes
across CEX, DEX, and bridge venues. The cross-venue ρ̂ is mechanically bounded by the number
of pools, so the calibration will always underfit the real multi-venue propagation.

**Mitigation:** Document in calibration report. Treat cross-venue ρ̂ tolerance as loose (30%)
and flag this moment if it fails. The v2 extension (order-book venue + MARL) addresses this
but is out of scope for the causal ranking result.

**Impact on conclusions:** Underestimated ρ̂ means the model is MORE conservative about
contagion spread. If a hub shows significant Δcontagion despite conservative propagation,
the real effect is likely larger, not smaller.

---

## 7.2 Episode count (n=7 real-node observations)

**Threat:** 7 real-node observations across 5 events is too small for reliable Spearman ρ
estimation. A single ρ=0.6 with n=7 has a 95% CI of roughly [−0.15, 0.93] (by bootstrap).
Report the CI prominently; single-point ρ without a CI is uninterpretable.

**Mitigation:** Report bootstrapped Spearman CI (implemented in `comparison.py::_bootstrap_spearman_ci`).
Report ρ separately for real-episode hubs and synthetic hubs; agreement that holds only on
synthetics is not evidence about real markets.

**Impact on conclusions:** Small n means the paper's primary claim should be about mechanism
(which intervention type works on which hub type) rather than precise effect-size estimates.

---

## 7.3 Intervention-strength confound (resolved)

**Threat (original):** Type-specific knobs made δC a mixture of node importance and knob
strength. DEX pools got liquidity removal; CEX venues got fee gates — mechanically different
treatments, confounding the ranking.

**Resolution:** Primary treatment is now uniform ablation at alpha=1.0 across all node types.
Type-specific knobs are demoted to secondary/policy analysis. The headline causal ranking
uses comparable treatments.

**Residual threat:** Even uniform ablation may have different _realizability_ per node type
(e.g., fully draining a pool is more disruptive than fully gating a CEX if there are more
alternative pools than alternative venues). Mitigate by running dose-response curves and
checking that δC scales monotonically with alpha for the top hubs.

---

## 7.4 Under-identified SMM (resolved)

**Threat (original):** 5 parameters × 4 moments was under-identified.

**Resolution:** `noise_trade_size` fixed at $2000 (structural prior from retail trade-size
literature), leaving 4 free parameters × 4 moments (just-identified). Document in calibration
report as: "identification: just-identified: 4 free params × 4 moments."

**Residual threat:** Just-identified SMM has no degrees of freedom to test over-identification.
The calibration report includes a local sensitivity analysis (numerical Jacobian around the
optimum) showing which moments constrain which parameters. If any column of J is near-zero,
that parameter is weakly identified — report it.

---

## 7.5 Calibration tolerance chosen before seeing results

**Threat:** Tolerances chosen after seeing results are the calibration equivalent of p-hacking.

**Mitigation:** Tolerances are locked in `configs/calibration_targets.json` BEFORE any
calibration runs. The file has a `_generated` timestamp. Do not modify tolerances after
seeing calibration output. If the calibration consistently fails a moment, document the
divergence as a finding (Section 3.5), not a reason to loosen the tolerance.

---

## 7.6 Synthetic scenario caveat

**Threat:** StressBench scenarios reproduce the structural shock type (reserve haircut +
liquidity drain) but are not reconstructions of the real episodes. Counterfactual results
are about the mechanism class, not the specific historical episode.

**Mitigation:** Standard for ABMs; the Gu et al. spoofing paper uses the same approach.
Make clear in the paper that results are statements about mechanism, not history.

---

## 7.7 Multiple comparisons on hub ranking

**Threat:** Testing ~8 hubs at α=0.05 raw yields at least 0.4 expected false positives.

**Mitigation:** BH FDR correction implemented in `inference.py::bh_correct()`. All
significance claims use q-values, not raw p-values. Report `pair_corr` as evidence that
the pairing is real (if pair_corr ≈ 0, the paired SE is invalid and all p-values are suspect).

---

## 7.8 Power at N=40 seeds

**Threat:** N=40 paired seeds may be underpowered for small causal effects. Gu et al. used
40k simulations; calling our N=40 "40k-seed discipline" is inaccurate.

**Mitigation:** Run `power_check()` before the full sweep to compute required_n() given the
pilot noise. Report MDE at N=40. Flag hubs with `underpowered=True` explicitly rather than
calling them null. If required_n >> 40, either increase seeds or acknowledge power limitation.
