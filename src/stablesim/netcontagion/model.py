"""
Networked-contagion model.

State: per-asset peg deviation d_i(t) = price_i(t) - 1  (units: fraction; 0.01 = 100 bps).

Dynamics (one step, dt):
    d_i(t+1) = d_i(t)
               - kappa_i * d_i(t) * dt                      # reserve pull back to peg
               + coupling * Σ_j W[i,j] * stress(d_j(t))     # directed contagion inflow
               + shock_i(t)                                 # exogenous depeg injection
               + common * eps_market(t)                     # market-wide common factor
               + sigma * eps_i(t)                           # idiosyncratic noise

where stress(d) = d if |d| >= stress_thr else 0  (only a *stressed* venue transmits),
and W[i,j] is the DIRECTED transmission weight j -> i (j leads/causes i), estimated
from the real lead-lag structure of the empirical 1-min deviations.

Per-node counterfactual: "protect node X" clamps d_X(t)=0 for all t (X neither depegs
nor transmits). Delta-contagion_X = baseline_contagion - contagion(protect=X). A true
propagator (the origin / a transmitter) has large Delta; a *spurious* hub — central /
correlated but not a transmitter — has Delta ~ 0. That is the causal test the GNN's
correlational hub ranking cannot perform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def estimate_transmission_matrix(
    dev: Dict[str, np.ndarray],
    nodes: List[str],
    max_lag: int = 30,
    stress_bps: float = 25.0,
) -> np.ndarray:
    """Directed transmission matrix W[i,j] = strength of j -> i (j leads i).

    Estimated from real 1-min peg deviations (bps): for each ordered pair, the
    positive part of the lead-lag correlation at the lag where j *leads* i, gated so
    that only genuinely leading, co-stressed pairs get weight. Rows are L1-normalised
    so each receiver's total inflow is comparable.
    """
    n = len(nodes)
    W = np.zeros((n, n))
    series = {k: np.asarray(v, float) for k, v in dev.items()}
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j or ni not in series or nj not in series:
                continue
            a = series[ni]
            b = series[nj]
            L = min(len(a), len(b))
            a, b = a[:L], b[:L]
            best = 0.0
            for lag in range(1, max_lag + 1):
                # b leads a by `lag`: corr(a[lag:], b[:-lag]) over rows finite in BOTH
                x, y = a[lag:], b[:-lag]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < max_lag + 20:
                    continue
                xs, ys = x[ok], y[ok]
                if xs.std() < 1e-9 or ys.std() < 1e-9:
                    continue
                # gate: the leader j must actually be stressed somewhere in-window
                if np.nanmax(np.abs(ys)) < stress_bps:
                    continue
                r = float(np.corrcoef(xs, ys)[0, 1])
                if r > best:
                    best = r
            W[i, j] = max(0.0, best)
    # NET directionality: co-moving stablecoins show lead-lag in BOTH directions; keep
    # only the dominant (net-lead) direction so a true transmitter (leads its followers)
    # is distinguished from a co-mover/sink. W[i,j] = relu(corr_ji_lead - corr_ij_lead).
    W = np.maximum(0.0, W - W.T)
    # row-normalise (receiver inflow), keep zeros for isolated nodes
    rs = W.sum(axis=1, keepdims=True)
    W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    return W


@dataclass
class ContagionNetwork:
    nodes: List[str]
    W: np.ndarray                      # (N, N) directed transmission j->i
    kappa: float = 0.15               # reserve recovery speed (per step)
    coupling: float = 1.0             # global contagion gain
    stress_thr: float = 0.0015        # |d| below which a venue does not transmit (fraction)
    sigma: float = 0.0008             # idiosyncratic noise std
    common: float = 0.0015            # common market-factor std
    panic_gain: float = 0.0           # origin-driven confidence spillover (see below)
    redemption_feedback: float = 0.0  # ENDOGENOUS adaptive redeemer: redemptions accelerate
    redeem_thr: float = 0.01          #   nonlinearly once a coin is depegged past redeem_thr,
    dt: float = 1.0                   #   worsening its own depeg (a self-reinforcing run)
    kappa_node: Optional[np.ndarray] = None  # optional per-node recovery (reserve strength)

    def __post_init__(self):
        self.N = len(self.nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        if self.kappa_node is None:
            self.kappa_node = np.full(self.N, self.kappa, float)

    def simulate(self, shock_node: str, shock_size: float, shock_step: int = 20,
                 n_steps: int = 200, seed: int = 0,
                 protect: Optional[str] = None, noise: bool = True,
                 cb_threshold: Optional[float] = None,
                 kappa_scale: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Return d (n_steps, N). `protect` clamps that node's deviation to 0 throughout.
        With noise=False the run is the DETERMINISTIC shock propagation (no idiosyncratic
        or common factor) — used for the causal knockout so Δ-contagion is not
        contaminated by noise extrema over the horizon."""
        rng = np.random.default_rng(seed)
        d = np.zeros((n_steps, self.N))
        s_idx = self.idx.get(shock_node, -1)
        p_idx = self.idx.get(protect, -1) if protect else -1
        cvol = self.common if noise else 0.0
        svol = self.sigma if noise else 0.0
        # interventions: per-node recovery boost (reserve strengthening) + circuit breaker
        kn = self.kappa_node.copy()
        if kappa_scale:
            for nd, sc in kappa_scale.items():
                if nd in self.idx:
                    kn[self.idx[nd]] = kn[self.idx[nd]] * sc
        cap = cb_threshold if cb_threshold is not None else 0.6
        for t in range(1, n_steps):
            prev = d[t - 1].copy()
            stressed = np.where(np.abs(prev) >= self.stress_thr, prev, 0.0)
            inflow = self.coupling * (self.W @ stressed)
            common_eps = rng.normal(0.0, cvol) if cvol else 0.0
            cur = (prev
                   - kn * prev * self.dt
                   + inflow
                   + common_eps
                   + (rng.normal(0.0, svol, self.N) if svol else 0.0))
            # Origin-driven panic / confidence spillover: the market-wide flight from
            # stablecoins is CAUSED by the origin's distress, so it scales with the origin's
            # current depeg and vanishes when the origin is held at peg. This is the channel
            # by which non-exposed coins (no balance-sheet link) still depeg — and the reason
            # protecting the origin (not a co-mover) removes that contagion too.
            if self.panic_gain > 0 and s_idx >= 0:
                panic = self.panic_gain * abs(prev[s_idx])
                cur -= panic
                cur[s_idx] += panic  # origin's own dynamics are its shock + recovery, not panic
            # Endogenous adaptive redeemer: once a coin is depegged past redeem_thr, panic
            # redemptions accelerate and push its price further from peg — a self-reinforcing
            # run whose intensity RESPONDS to the state (the Lucas-critique channel).
            if self.redemption_feedback > 0:
                excess = np.maximum(np.abs(prev) - self.redeem_thr, 0.0)
                # push a depegged coin FURTHER from peg (more negative when already negative)
                cur += self.redemption_feedback * excess * np.sign(prev)
            if s_idx >= 0 and t == shock_step:
                cur[s_idx] -= shock_size  # depeg the origin downward
            if p_idx >= 0:
                cur[p_idx] = 0.0          # protected node held at peg
            # circuit breaker caps the depeg magnitude; default cap prevents blow-up.
            d[t] = np.clip(cur, -cap, cap)
        return d

    # --------- moments (match the empirical calibration targets) ----------
    def moments(self, shock_node: str, shock_size: float, n_seeds: int = 20,
                n_steps: int = 200, shock_step: int = 40) -> dict:
        peak_mags, half_lives, rhos, base_vols = [], [], [], []
        origin_i = self.idx.get(shock_node, -1)
        # crisis half-life: the origin receives no inflow (it is the source), so after the
        # impulse it is a pure OU decay at rate kappa -> half-life = ln(2)/kappa (exact).
        ko = float(self.kappa_node[origin_i]) if origin_i >= 0 else float(self.kappa)
        det_hl = float(np.log(2) / ko) if ko > 0 else float("nan")
        for seed in range(n_seeds):
            d = self.simulate(shock_node, shock_size, shock_step, n_steps, seed)
            others = [k for k in range(self.N) if k != origin_i]
            # contagion magnitude: peak |dev| over non-origin nodes (deterministic part)
            d_o = self.simulate(shock_node, shock_size, shock_step, n_steps, seed, noise=False)
            peak_mags.append(float(np.abs(d_o[:, others]).max()))
            half_lives.append(det_hl)
            # baseline vol: std of step changes pre-shock
            pre = d[:shock_step]
            base_vols.append(float(np.std(np.diff(pre[:, others].mean(axis=1)))) if shock_step > 3 else 0.0)
            # cross-venue rho: mean pairwise corr of returns during contagion window
            rhos.append(_crisis_rho(d, others, shock_step))
        return {
            "contagion_magnitude": float(np.median(peak_mags)),
            "crisis_half_life": float(np.median([h for h in half_lives if np.isfinite(h)] or [np.nan])),
            "baseline_price_vol": float(np.median(base_vols)),
            "cross_venue_rho": float(np.median(rhos)),
        }

    def contagion_over(self, shock_node: str, shock_size: float, measure: List[str],
                       protect: Optional[str] = None, n_steps: int = 250,
                       shock_step: int = 40, cb_threshold: Optional[float] = None,
                       kappa_scale: Optional[Dict[str, float]] = None) -> float:
        """Mean peak |dev| over a FIXED `measure` node-set, on the deterministic
        propagation. Used for causal knockout AND intervention outcomes: comparing
        intervention vs none over the SAME measure set isolates the intervention's effect."""
        d = self.simulate(shock_node, shock_size, shock_step, n_steps, seed=0,
                          protect=protect, noise=False, cb_threshold=cb_threshold,
                          kappa_scale=kappa_scale)
        cols = [self.idx[m] for m in measure if m in self.idx]
        if not cols:
            return 0.0
        return float(np.abs(d[:, cols]).max(axis=0).mean())

    def causal_delta(self, shock_node: str, shock_size: float, node: str) -> float:
        """Causal Δ-contagion of protecting `node`: change in contagion to the OTHER
        victims (excluding origin and `node` from the measured set consistently)."""
        measure = [n for n in self.nodes if n not in (shock_node, node)]
        base = self.contagion_over(shock_node, shock_size, measure, protect=None)
        prot = self.contagion_over(shock_node, shock_size, measure, protect=node)
        return base - prot


def _recovery_half_life(series: np.ndarray, shock_step: int) -> float:
    """Steps for |dev| to decay to half its post-shock peak."""
    post = np.abs(series[shock_step:])
    if len(post) < 3 or post.max() <= 0:
        return float("nan")
    peak = post.max()
    pk = int(np.argmax(post))
    for k in range(pk, len(post)):
        if post[k] <= 0.5 * peak:
            return float(k - pk)
    return float("nan")


def _crisis_rho(d: np.ndarray, others: List[int], shock_step: int) -> float:
    """Median pairwise correlation of returns over the post-shock contagion window."""
    win = d[shock_step:shock_step + 60, others]
    if win.shape[0] < 5 or win.shape[1] < 2:
        return 0.0
    r = np.diff(win, axis=0)
    if (r.std(axis=0) < 1e-12).any():
        return 0.0
    C = np.corrcoef(r.T)
    iu = np.triu_indices(len(others), k=1)
    vals = C[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else 0.0


def episode_moments(dev_bps: Dict[str, np.ndarray], origin: str) -> dict:
    """Empirical moments from a real episode's 1-min deviations (bps), for reference."""
    nodes = list(dev_bps.keys())
    others = [n for n in nodes if n != origin]
    arrs = {n: np.asarray(v, float) / 1e4 for n, v in dev_bps.items()}  # bps -> fraction
    peak = max((np.abs(arrs[n]).max() for n in others), default=0.0)
    return {"contagion_magnitude": float(peak), "n_nodes": len(nodes)}
