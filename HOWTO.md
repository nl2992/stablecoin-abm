# HOW-TO — stablecoin-abm

Concrete execution guide for `TODO.md`. Each block = commands + what "good" looks like
+ the failure mode to watch. Setup:

```bash
pip install -r requirements.txt && pip install -e .
pytest tests/    # incl. test_phase0_gate, test_calibration_regression, test_counterfactual
```

The ordering is a hard dependency chain: **calibration → join → sweep**. Do not run the
sweep until calibration passes its gate; uncalibrated intervention numbers are noise.

---

## P0 — Calibration (the gate)

```bash
make calibrate        # wraps scripts/run_calibration.py (SMM over the moment set)
```
- *Good:* `experiments/results/` gets a calibration report with a moments table
  (simulated vs empirical, ±tolerance, pass/fail) and an overlay figure.
- *Gate:* ≥3 of 4 stylized facts inside empirical bands — OU half-life calm≈3.2 min /
  crisis≈600 min, cross-asset ρ̂, LP loss, peg-deviation tails.
- *Failure mode:* SMM "fits" by collapsing to a degenerate parameter (e.g. zero noise so
  peg never moves). Inspect the fitted params; check `calibration/smm.py` identification
  (you already fixed an SMM identification bug — verify moments are sensitive to each param
  via a one-at-a-time perturbation before trusting the fit).
- *Sells the paper:* the overlay figure. If simulated and empirical peg paths visibly
  track during a crisis episode, reviewers believe everything downstream.

Then close the Phase-1 gap:
```bash
pytest tests/test_agents.py      # add myopic-correctness assertions if missing
```

---

## P1 — The join (correlation → causation)

```bash
# 1. ensure the GNN exported a hub ranking
ls ../stablecoin-contagion-gnn/exports/hub_ranking_v1_*.json

# 2. per-node counterfactuals: intervene on each hub, measure Δcontagion
python scripts/run_counterfactuals.py

# 3. agreement: predicted importance vs causal effect
python scripts/run_joint_analysis.py
```
- *Headline (Fig 1):* scatter of GNN-predicted hub importance (x) vs ABM causal
  Δcontagion (y), one point per node. Report Spearman ρ, top-3/top-5 overlap, OLS slope.
- *Good agreement (high ρ):* "the GNN's correlational hubs are genuinely causal" — clean win.
- *The better story is the outlier:* the node high on x but low on y. That is the
  **spurious hub** the GNN flagged via the volume/TVL confound. Open it up:
  - run the intervention, show Δcontagion ≈ 0 with its SE;
  - trace the mechanism — high centrality but the node is a *sink*, not a *transmitter*
    (it co-moves with stress but doesn't propagate it). Draw the mechanism diagram (Fig 2).
- *Failure mode:* counterfactual effects all within noise ⇒ shocks too weak or seeds too
  few. Increase shock severity / seeds until no-intervention contagion is well above the
  SE floor, then re-run. Use `counterfactual/inference.py` for the SEs.

---

## P2 — Intervention sweep (the policy result)

```bash
make sweep        # scenarios × interventions × seeds; configs/interventions.yaml
```
- *Outcomes per cell:* contagion magnitude, peg half-life, LP impermanent loss, welfare by
  agent type. Build the welfare-decomposition matrix (Fig 4) and outcome-vs-knob curves (Fig 3).
- *Find the regime flip:* sweep one knob (start with redemption gating or transparency
  frequency) finely and look for the threshold where contagion drops sharply or where the
  *winner among agent types flips*. Gu's paper won on identifying two regimes — replicate
  that move: name the regimes and the boundary.
- *Mechanism (do not skip):* for each intervention that works, write the causal chain —
  e.g. "transparency reduces redemption panic because the issuer signal lowers the
  redeemers' depeg-probability estimate, shrinking the redemption surge that exhausts the
  reserve." Effect size without mechanism reads as curve-fitting to reviewers.
- *Failure mode:* an intervention helps one agent while silently hurting another. That's a
  *finding*, not a bug — surface it in the welfare matrix; it's the "at what cost to whom"
  half of the research question.

---

## P3 — RL (make it earn its place)

```bash
make train        # PPO via stable-baselines3, rl/ppo.py + rl/env.py
```
- *Gate:* learned policy beats the heuristic arbitrageur/redeemer on P&L, reproducibly
  across seeds. Plot training curves + RL-vs-heuristic comparison.
- *The depth move:* re-run your best intervention from P2 with RL-adaptive agents replacing
  heuristic ones. If the intervention still reduces contagion when agents *adapt to it*,
  that's a Lucas-critique-robust result — a genuine edge over static-agent ABM papers.
- *Failure mode:* PPO won't converge (sparse reward / bad action scaling). Keep RL as an
  explicit robustness arm and say so — decorative RL that's oversold is a reviewer red flag;
  honestly-scoped RL is fine.

---

## P4 — Robustness

Re-run P2 varying: agent-population mix, shock severity, and calibration parameters drawn
from their posterior/uncertainty band. The deliverable (Table 6) is whether the
intervention **ranking** is stable. Report rankings that flip — don't hide them.
Every comparison anchors to the no-intervention + no-RL baseline.

---

## P5 — Paper + release

Sections already stubbed in `paper/sections/`. Fill in order: calibration (03) →
hub-agreement (04) → divergence case study (05) → intervention sweep → threats (07) →
negative-result framing (08). Tie policy implications to GENIUS Act / MiCA. Release the
simulator, `configs/`, and the sweep runner; complete `paper/appendix_reproducibility.md`
with exact commands, seeds, and the calibration hash.

**Dependency on the GNN repo:** P1 cannot run until
`../stablecoin-contagion-gnn/exports/hub_ranking_v1_*.json` exists and names a spurious-hub
candidate. Coordinate the two repos around that single JSON contract.
