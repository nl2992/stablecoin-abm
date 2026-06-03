"""
inference.py
------------
Correct statistical inference for paired counterfactual experiments.

Three fixes over the original design:

1. PAIRED standard error.
   Baseline and intervened arms share the same seed (same fundamental path,
   same agent draws), so they are PAIRED. The variance of interest is the
   variance of the within-seed difference d_i = C_baseline_i - C_intervened_i,
   NOT the two-sample variance. Using a two-sample SE here inflates the
   denominator inconsistently and, more often, the independent-sample formula
   understates the true precision OR overstates significance depending on the
   sign of the cross-arm correlation. Always use the paired SE.

2. POWER / minimum detectable effect.
   With N seeds you can only resolve effects above a floor. Report the MDE so a
   null result is read as "underpowered" vs. "genuinely no effect", and so the
   reader knows whether N=40 was ever enough.

3. MULTIPLE COMPARISONS.
   You test many hubs. A raw one-sided p < 0.05 across ~30 hubs yields false
   "significant" hubs by construction. Apply Benjamini-Hochberg FDR control
   across the hub set and report q-values alongside p-values.

Plus bootstrap CIs on delta-C (robust to the non-normal, heavy-tailed sim
output that financial contagion magnitudes typically have).

Pure numpy/scipy; no engine dependency. Unit-tested at the bottom under
`if __name__ == "__main__"`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:  # scipy optional; fall back to normal approx
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Result container -- superset of your existing CounterfactualResult fields   #
# --------------------------------------------------------------------------- #
@dataclass
class PairedResult:
    node_id: str
    n_pairs: int
    delta_c: float            # mean within-seed (baseline - intervened); >0 = intervention reduces contagion
    se_paired: float          # paired standard error of delta_c
    t_stat: float
    p_one_sided: float        # H1: intervention REDUCES contagion (delta_c > 0)
    ci_lo: float              # bootstrap CI lower
    ci_hi: float              # bootstrap CI upper
    cohens_dz: float          # paired effect size
    mde_80: float             # minimum detectable effect at 80% power, this N
    underpowered: bool        # |delta_c| < mde_80 and not significant
    pair_corr: float          # corr(baseline, intervened) across seeds -- sanity check
    # filled in by bh_correct():
    q_value: float | None = None
    significant_fdr: bool | None = None

    def to_row(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Core paired test                                                            #
# --------------------------------------------------------------------------- #
def paired_test(
    node_id: str,
    baseline: Sequence[float],
    intervened: Sequence[float],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> PairedResult:
    """Paired one-sided test that the intervention reduces contagion magnitude.

    Parameters
    ----------
    baseline, intervened : arrays of contagion magnitude C, ALIGNED BY SEED.
        baseline[i] and intervened[i] MUST come from the same seed i.
    alpha : significance level for the MDE/power calc and reporting.
    power : target power for the minimum-detectable-effect calc.
    n_boot : bootstrap resamples for the CI.

    Convention: delta_c = baseline - intervened. delta_c > 0 means the
    intervention LOWERED contagion (the hub matters). H1 is delta_c > 0.
    """
    b = np.asarray(baseline, dtype=float)
    x = np.asarray(intervened, dtype=float)
    if b.shape != x.shape:
        raise ValueError("baseline and intervened must be aligned & same length")
    n = b.size
    if n < 2:
        raise ValueError("need >= 2 paired observations")
    if rng is None:
        rng = np.random.default_rng(0)

    d = b - x                              # paired differences
    delta_c = float(d.mean())
    sd_d = float(d.std(ddof=1))            # sample SD of the differences
    se = sd_d / math.sqrt(n)               # PAIRED standard error (the fix)

    # cross-arm correlation -- diagnostic. High positive corr is expected for
    # well-paired seeds and is exactly why the paired SE is tighter than the
    # two-sample SE. If this is ~0, your seeds aren't actually pairing the runs.
    if sd_d > 0 and b.std() > 0 and x.std() > 0:
        pair_corr = float(np.corrcoef(b, x)[0, 1])
    else:
        pair_corr = float("nan")

    # one-sided p-value, H1: delta_c > 0
    if se == 0:
        t_stat = math.inf if delta_c > 0 else (-math.inf if delta_c < 0 else 0.0)
        p = 0.0 if delta_c > 0 else 1.0
    else:
        t_stat = delta_c / se
        if _HAVE_SCIPY:
            p = float(_scipy_stats.t.sf(t_stat, df=n - 1))   # P(T >= t)
        else:
            p = float(_normal_sf(t_stat))                    # normal approx

    # paired effect size (Cohen's dz)
    dz = delta_c / sd_d if sd_d > 0 else float("inf")

    # minimum detectable effect at target power, given THIS n and sd_d
    mde = _mde_paired(sd_d, n, alpha=alpha, power=power)

    underpowered = (p > alpha) and (abs(delta_c) < mde)

    # bootstrap CI on delta_c (resample the paired differences)
    ci_lo, ci_hi = _bootstrap_ci(d, n_boot=n_boot, alpha=alpha, rng=rng)

    return PairedResult(
        node_id=node_id,
        n_pairs=n,
        delta_c=delta_c,
        se_paired=se,
        t_stat=float(t_stat),
        p_one_sided=p,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        cohens_dz=float(dz),
        mde_80=mde,
        underpowered=bool(underpowered),
        pair_corr=pair_corr,
    )


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg FDR across the hub set                                   #
# --------------------------------------------------------------------------- #
def bh_correct(results: list[PairedResult], *, fdr: float = 0.05) -> list[PairedResult]:
    """Add q-values and FDR-significance flags across ALL tested hubs, in place.

    Standard Benjamini-Hochberg step-up on the one-sided p-values. Call this
    ONCE over the full set of hub results before claiming any hub is
    significant.
    """
    m = len(results)
    if m == 0:
        return results
    order = sorted(range(m), key=lambda i: results[i].p_one_sided)
    # raw BH q: p_(k) * m / k, then enforce monotonicity from the top
    q_raw = [0.0] * m
    for rank, i in enumerate(order, start=1):
        q_raw[i] = results[i].p_one_sided * m / rank
    # monotone non-decreasing in p-order
    running_min = 1.0
    for i in reversed(order):
        running_min = min(running_min, q_raw[i])
        results[i].q_value = float(min(running_min, 1.0))
        results[i].significant_fdr = bool(results[i].q_value <= fdr)
    return results


# --------------------------------------------------------------------------- #
# Power helpers                                                               #
# --------------------------------------------------------------------------- #
def _mde_paired(sd_d: float, n: int, *, alpha: float, power: float) -> float:
    """Minimum detectable paired mean difference (one-sided) at given power."""
    if sd_d == 0 or n < 2:
        return 0.0
    z_a = _z(1 - alpha)
    z_b = _z(power)
    return (z_a + z_b) * sd_d / math.sqrt(n)


def required_n(sd_d: float, target_effect: float, *, alpha: float = 0.05,
               power: float = 0.80) -> int:
    """How many paired seeds to detect `target_effect` at given power.

    Use this BEFORE the big sweep to justify N. If this returns 4000 and you
    planned to run 40, you know the headline test is underpowered.
    """
    if target_effect <= 0:
        raise ValueError("target_effect must be > 0")
    z_a = _z(1 - alpha)
    z_b = _z(power)
    n = ((z_a + z_b) * sd_d / target_effect) ** 2
    return int(math.ceil(n))


# --------------------------------------------------------------------------- #
# Bootstrap CI                                                                #
# --------------------------------------------------------------------------- #
def _bootstrap_ci(d: np.ndarray, *, n_boot: int, alpha: float,
                  rng: np.random.Generator) -> tuple[float, float]:
    n = d.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


# --------------------------------------------------------------------------- #
# Normal-dist helpers (avoid hard scipy dependency)                           #
# --------------------------------------------------------------------------- #
def _z(p: float) -> float:
    """Inverse standard normal CDF."""
    if _HAVE_SCIPY:
        return float(_scipy_stats.norm.ppf(p))
    # Acklam's rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)


def _normal_sf(z: float) -> float:
    """Survival function of standard normal (1 - CDF)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


# --------------------------------------------------------------------------- #
# How the runner uses this (replaces the ad-hoc t-stat in runner.py)          #
# --------------------------------------------------------------------------- #
def summarize_sweep(
    per_hub: dict[str, tuple[Sequence[float], Sequence[float]]],
    *,
    alpha: float = 0.05,
    fdr: float = 0.05,
    power: float = 0.80,
    seed: int = 0,
) -> list[PairedResult]:
    """End-to-end: per-hub paired test + BH correction across the hub set.

    `per_hub` maps node_id -> (baseline_C_by_seed, intervened_C_by_seed),
    where the two arrays are aligned by seed. Returns FDR-corrected results
    sorted by delta_c descending (largest causal reduction first).
    """
    rng = np.random.default_rng(seed)
    results = [
        paired_test(node_id, base, interv, alpha=alpha, power=power, rng=rng)
        for node_id, (base, interv) in per_hub.items()
    ]
    bh_correct(results, fdr=fdr)
    results.sort(key=lambda r: r.delta_c, reverse=True)
    return results


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate 3 hubs over 40 paired seeds.
    # Strong common per-seed shock => high pairing correlation => paired SE
    # is much tighter than the naive two-sample SE.
    n = 40
    common = rng.normal(0, 5.0, size=n)          # shared seed-level shock
    hubs = {
        # true effect +1.0 contagion-units (intervention helps)
        "curve_3pool":   (common + 10 + rng.normal(0, 0.5, n),
                          common + 9.0 + rng.normal(0, 0.5, n)),
        # true effect ~0 (spurious hub: high baseline, no causal role)
        "ust_binance":   (common + 12 + rng.normal(0, 0.5, n),
                          common + 12 + rng.normal(0, 0.5, n)),
        # small true effect +0.3, likely underpowered at n=40
        "usdc_coinbase": (common + 8 + rng.normal(0, 0.5, n),
                          common + 7.7 + rng.normal(0, 0.5, n)),
    }

    res = summarize_sweep(hubs, seed=1)
    print(f"{'node':14s} {'dC':>7s} {'SE':>6s} {'t':>7s} "
          f"{'p':>7s} {'q':>7s} {'mde80':>7s} {'pairR':>6s} sig?")
    for r in res:
        print(f"{r.node_id:14s} {r.delta_c:7.3f} {r.se_paired:6.3f} "
              f"{r.t_stat:7.2f} {r.p_one_sided:7.4f} {r.q_value:7.4f} "
              f"{r.mde_80:7.3f} {r.pair_corr:6.2f} {r.significant_fdr}")

    # Power planning example: to detect a 0.3-unit effect at this noise level,
    # how many seeds do we actually need?
    sd_d = float(np.std(np.asarray(hubs["usdc_coinbase"][0])
                        - np.asarray(hubs["usdc_coinbase"][1]), ddof=1))
    print(f"\nTo detect dC=0.30 at 80% power need N = "
          f"{required_n(sd_d, 0.30)} paired seeds (planned 40).")
