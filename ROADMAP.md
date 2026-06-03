# ROADMAP — stablecoin-abm

**Goal:** a calibrated, agent-based stablecoin market with RL policies, used to measure which
interventions reduce depeg contagion and at what cost to whom.  This is the heavier of the two
builds — the value is the mechanism + policy story (the Gu spoofing paper is the template),
but it only counts if the simulator is calibrated to the empirical facts first.  Phases are
dependency-ordered; week estimates are effort, not calendar.

**Scope discipline:** build an AMM-only v1 (stableswap pools + redemption channel).  Add an
order-book venue and full MARL only after v1 produces a calibrated, validated result.

---

## Phase 0 — Engine core (~week 0–2)

- [x] `engine/amm.py`: Curve-style stableswap invariant with fees; ≥ 2 pools.
- [x] `engine/redemption.py`: primary mint/redeem at $1 against a reserve; redemption accounting.
- [x] `engine/reserve.py`: backing ratio (support fractional / yield-bearing later); exhaustion logic.
- [x] Event loop + discrete time-stepping; deterministic seeding; full state logging.
- [x] Unit tests: invariant conservation, no-arbitrage equilibrium in calm, redemption accounting balances to the cent.

**Gate:** with no shock and no agents acting, pegs sit at $1 and the invariant is conserved.
**Artefacts:** `docs/engine_spec.md`; passing invariant/accounting test suite.

---

## Phase 1 — Agents (~week 2–3)

- [x] Heuristic agents (`agents/`): arbitrageur (cross-venue + redemption arb),
  LP (add/withdraw on signal), redeemer (redeem on depeg net of gating cost),
  noise traders, issuer/reserve (publishes transparency signal, honours redemptions until
  reserve exhausts).
- [x] Define obs / action / utility for each; net-position and P&L accounting.
- [ ] Per-agent unit tests for myopic correctness (e.g. arbitrageur trades toward peg when
  profitable after fees).

**Gate:** heuristic arbitrage measurably shrinks an induced depeg in a single pool.
**Artefacts:** `docs/agent_spec.md`; per-agent tests.

---

## Phase 2 — Scenarios & calibration (~week 3–5) — THE credibility gate

- [x] `scenarios/`: StressBench shock loader — reserve haircut, redemption surge, liquidity
  withdrawal, correlated cross-asset shock — as an exogenous event schedule.
- [ ] `calibration/`: tune a small parameter set (via SMM / grid / Bayesian) so no-intervention
  runs reproduce empirical moments from the contagion-network + IAQF work:
  - peg-deviation distribution by regime,
  - OU half-life (calm ≈ minutes → crisis ≈ hours; IAQF numbers: ~3.2 min → ~600 min),
  - cross-asset propagation ρ̂,
  - LP-loss magnitude.
- [ ] If the sim cannot reproduce these, stop and fix the engine/agents — intervention results
  are meaningless until it does.

**Gate:** simulated moments fall within empirical confidence bands on ≥ 3 of 4 stylized facts.
**Artefacts:** calibration report — simulated-vs-empirical moments table + overlay figures;
the validation figure that sells the ABM's realism.

---

## Phase 3 — RL agents (~week 5–7)

- [x] `rl/`: Gymnasium wrapper; PPO via stable-baselines3.
- [ ] Train arbitrageur and/or redeemer to maximise P&L vs. heuristic background population.
- [ ] Verify RL agents converge and beat heuristic counterparts.

**Gate:** trained RL policy outperforms heuristic baseline on its own objective, reproducibly.
**Artefacts:** training curves; RL-vs-heuristic comparison; learned-policy behaviour plots.

---

## Phase 4 — Intervention experiments (~week 7–9) — THE paper

- [x] Four intervention knobs as env params: reserve transparency, redemption gating,
  circuit breaker, LP incentives.
- [ ] Sweep scenarios × interventions × many seeds; report standard errors.
- [ ] Outcomes: contagion magnitude, peg-recovery half-life, LP impermanent loss, welfare by agent type.
- [ ] Find regime/threshold where behaviour flips (Gu: critical liquidity; here: transparency/gating analog).
- [ ] Mechanism analysis — explain *why* each intervention works (Gu §8 analog).

**Gate:** ≥ 1 intervention shows statistically significant contagion reduction with a clear mechanism.
**Artefacts:** outcome-vs-knob curves; midprice/peg-evolution figures; welfare-decomposition matrix;
regime figure; mechanism diagram.

---

## Phase 5 — Robustness & paper (~week 9–12)

- [ ] Robustness: vary agent-population mix, shock severity, calibration uncertainty.
- [ ] Anchor every comparison against no-intervention + no-RL baseline.
- [ ] Paper: ABM design, calibration/validation, RL, intervention results, mechanisms, policy implications
  (tie to GENIUS Act / MiCA, reusing the IAQF regulatory framing).

**Gate:** intervention conclusions survive calibration-uncertainty and seed variation.
**Artefacts:** released simulator + configs; reproducible sweep runner; paper.

---

## Credibility checklist (reviewer-facing gates)

- [ ] Calibration validation passes — sim reproduces empirical half-lives and ρ̂ **(non-negotiable)**.
- [ ] RL agents beat heuristic baselines (else RL is decorative).
- [ ] Intervention rankings robust to calibration uncertainty + seeds, with standard errors.
- [ ] Each intervention result carries a mechanism explanation, not just an effect size.
- [ ] A no-intervention + no-RL baseline anchors all comparisons.

---

## Artefact master list

| Type | Items |
|---|---|
| Tables | agent spec · calibration moments (sim vs empirical) · intervention sweep results · welfare decomposition · robustness |
| Figures | calibration overlay (peg path / half-life) · RL training curves · learned-policy behaviour · per-intervention midprice/peg evolution · outcome-vs-knob curves · welfare matrix · mechanism diagram |
| Code/data | released simulator + configs · sweep runner · calibration dataset |

---

## Empirical calibration targets (from stablecoin-contagion-network / IAQF)

| Moment | Calm regime | Crisis regime |
|---|---|---|
| OU half-life | ~3.2 min | ~600 min |
| Cross-asset ρ̂ | low | high (propagation) |
| LP loss magnitude | small | large |
| Peg-deviation distribution | tight | fat-tailed |
