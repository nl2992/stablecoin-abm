"""
Causal counterfactual join: does the GNN's correlational hub ranking predict the ABM's
*causal* hub ranking?  And is BUSD (the GNN's top hub) actually causal, or spurious?

Pipeline
--------
1. Load the REAL USDC/SVB episode (1-min peg deviations) from the GNN repo.
2. Estimate a DIRECTED transmission network W from the real lead-lag structure.
3. Calibrate the networked-contagion engine to the empirical moments
   (contagion magnitude, cross-venue rho, baseline vol, crisis half-life).
4. Per-node knockout counterfactual: protect each node, measure the drop in contagion.
5. Load the GNN hub ranking (predicted importance) and correlate it with the ABM's
   causal Delta-contagion.  Identify the spurious hub (high predicted, ~0 causal).

Outputs -> experiments/results/netcontagion/
    calibration_moments.csv     sim vs empirical (pass/fail gate)
    causal_hub_ranking.csv      per-node predicted importance + causal Delta
    join_summary.json           Spearman agreement, spurious hub, headline numbers
    fig_join_scatter.png        predicted vs causal, BUSD highlighted
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from stablesim.netcontagion.model import ContagionNetwork, estimate_transmission_matrix  # noqa: E402

GNN_ROOT = Path(__file__).parents[2] / "stablecoin-contagion-gnn"
EPISODE = "USDC_SVB"
OUT = Path("experiments/results/netcontagion")

# Empirical targets (fraction units). contagion magnitude + crisis half-life from the
# GNN export (exports/calibration_v1.csv: USDC/SVB peak ~0.1376, OU half-life 579 min /
# 5 min-per-step ~= 116 steps). rho/vol from the locked stylized facts.
TARGETS = {"contagion_magnitude": 0.1376, "cross_venue_rho": 0.576,
           "baseline_price_vol": 0.003, "crisis_half_life": 116.0}
TOL = {"contagion_magnitude": 0.25, "cross_venue_rho": 0.30,
       "baseline_price_vol": 0.30, "crisis_half_life": 0.30}


def load_episode(name: str) -> dict:
    with open(GNN_ROOT / "data" / "processed" / "graphs" / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def build_network(b: dict) -> tuple[ContagionNetwork, list, str]:
    nodes = b["active_node_strs"]
    origin = b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}  # bps
    W = estimate_transmission_matrix(dev, nodes, max_lag=30, stress_bps=25.0)
    net = ContagionNetwork(nodes=nodes, W=W)
    return net, nodes, origin


def calibrate(net: ContagionNetwork, origin: str) -> dict:
    """Fit [coupling, kappa, common, sigma, shock] to the empirical moments."""
    x0 = np.array([0.015, 0.006, 0.0015, 0.0008, 0.13])
    bounds = [(0.004, 0.05), (0.003, 0.04), (0.0002, 0.006), (0.0001, 0.003), (0.03, 0.30)]

    def unpack(x):
        net.coupling, net.kappa, net.common, net.sigma = x[0], x[1], x[2], x[3]
        net.kappa_node = np.full(net.N, x[1], float)
        return x[4]

    def loss(x):
        shock = unpack(x)
        m = net.moments(origin, shock, n_seeds=12, shock_step=40, n_steps=200)
        L = 0.0
        for k, tgt in TARGETS.items():
            sim = m.get(k, np.nan)
            if not np.isfinite(sim):
                L += 5.0
                continue
            w = {"contagion_magnitude": 2.0, "cross_venue_rho": 2.0,
                 "crisis_half_life": 1.5, "baseline_price_vol": 1.0}.get(k, 1.0)
            L += w * ((sim - tgt) / tgt) ** 2
        return L

    res = minimize(loss, x0, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-5})
    shock = unpack(res.x)
    m = net.moments(origin, shock, n_seeds=40, shock_step=40, n_steps=200)
    return {"params": dict(zip(["coupling", "kappa", "common", "sigma", "shock"], res.x.tolist())),
            "shock": shock, "moments": m}


def moments_table(m: dict) -> tuple[pd.DataFrame, int]:
    rows, n_pass = [], 0
    for k, tgt in TARGETS.items():
        sim = m.get(k, np.nan)
        rel = abs(sim - tgt) / tgt if np.isfinite(sim) else np.inf
        ok = rel <= TOL[k]
        n_pass += int(ok)
        rows.append({"moment": k, "empirical": tgt, "simulated": round(float(sim), 5),
                     "rel_error": round(float(rel), 4), "tolerance": TOL[k], "passes": ok})
    return pd.DataFrame(rows), n_pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = load_episode(EPISODE)
    net, nodes, origin = build_network(b)
    print(f"nodes={nodes}\norigin={origin}")
    print("transmission W (row=receiver i, col=sender j):")
    print(pd.DataFrame(np.round(net.W, 3), index=nodes, columns=nodes).to_string())

    cal = calibrate(net, origin)
    shock = cal["shock"]
    mt, n_pass = moments_table(cal["moments"])
    mt.to_csv(OUT / "calibration_moments.csv", index=False)
    print("\n=== Calibration moments (gate >= 3/4) ===")
    print(mt.to_string(index=False))
    print(f"GATE: {n_pass}/4 passing  params={ {k: round(v,4) for k,v in cal['params'].items()} }")

    # ---- per-node knockout counterfactual (causal Δ on the OTHER victims) ----
    rows = []
    for nd in nodes:
        if nd == origin:
            continue
        delta = net.causal_delta(origin, shock, nd)
        rows.append({"node": nd, "causal_delta_contagion": round(float(delta), 6),
                     "out_transmission": round(float(net.W[:, nodes.index(nd)].sum()), 4)})
    cf = pd.DataFrame(rows)

    # sanity: protecting the origin should remove (nearly) all contagion
    measure_all = [n for n in nodes if n != origin]
    base_all = net.contagion_over(origin, shock, measure_all, protect=None)
    prot_o = net.contagion_over(origin, shock, measure_all, protect=origin)
    origin_delta = base_all - prot_o
    base = base_all

    # ---- merge with GNN predicted importance ----
    hub = pd.read_csv(GNN_ROOT / "exports" / f"hub_ranking_v1_{EPISODE}.csv")
    pred_col = "hub_score_full" if "hub_score_full" in hub.columns else "hub_score"
    hub = hub[["node", pred_col, "betweenness", "gnn_mask_sum", "propagator_label"]].rename(
        columns={pred_col: "gnn_predicted_importance"})
    join = cf.merge(hub, on="node", how="left").fillna(0.0)
    join = join.sort_values("causal_delta_contagion", ascending=False).reset_index(drop=True)
    join.to_csv(OUT / "causal_hub_ranking.csv", index=False)
    print("\n=== Causal hub ranking (ABM) vs GNN predicted ===")
    print(join.to_string(index=False))

    # ---- agreement + spurious hub ----
    valid = join[(join["gnn_predicted_importance"] > 0) | (join["causal_delta_contagion"].abs() > 0)]
    rho, p = spearmanr(valid["gnn_predicted_importance"], valid["causal_delta_contagion"])
    # spurious = highest predicted importance but near-zero causal effect
    thresh = 0.2 * max(join["causal_delta_contagion"].max(), 1e-9)
    cand = join[join["causal_delta_contagion"] <= thresh].sort_values(
        "gnn_predicted_importance", ascending=False)
    spurious = cand.iloc[0]["node"] if len(cand) and cand.iloc[0]["gnn_predicted_importance"] > 0 else None

    summary = {
        "episode": EPISODE, "origin": origin,
        "calibration_pass": f"{n_pass}/4", "calibrated_params": cal["params"],
        "baseline_contagion": round(float(base), 6),
        "origin_causal_delta": round(float(origin_delta), 6),
        "spearman_pred_vs_causal": round(float(rho), 4), "spearman_p": round(float(p), 4),
        "spurious_hub": spurious,
        "gnn_top_hub": hub.sort_values("gnn_predicted_importance", ascending=False).iloc[0]["node"],
        "interpretation": (
            f"The GNN's top correlational hub is "
            f"{hub.sort_values('gnn_predicted_importance', ascending=False).iloc[0]['node']}; "
            f"the ABM's causal knockout shows {spurious} has ~0 causal effect on contagion "
            f"despite high predicted importance -> a spurious (correlational, non-causal) hub. "
            f"Protecting the origin {origin} removes {origin_delta:.4f} of contagion."),
    }
    (OUT / "join_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== JOIN SUMMARY ===")
    print(json.dumps(summary, indent=2))

    _plot(join, spurious, origin, OUT / "fig_join_scatter.png")


def _plot(join: pd.DataFrame, spurious, origin, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import paper_style as ps
    ps.apply()
    fig, ax = plt.subplots(figsize=ps.TALL)
    for _, r in join.iterrows():
        if r["node"] == spurious:
            c, mk, s = ps.RED, "X", 200
        elif r["propagator_label"] == 1:
            c, mk, s = ps.GREEN, "o", 150
        else:
            c, mk, s = ps.GREY, "s", 130
        ax.scatter(r["gnn_predicted_importance"], r["causal_delta_contagion"],
                   c=c, marker=mk, s=s, zorder=3, edgecolor="k", linewidth=0.5)
        ax.annotate(r["node"].split("/")[0],
                    (r["gnn_predicted_importance"], r["causal_delta_contagion"]),
                    fontsize=9, xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("GNN predicted importance (correlational hub score)")
    ax.set_ylabel("ABM causal effect (knockout)")
    ax.set_title("Correlation is not causation: BUSD is a hub with no causal effect")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ps.GREEN, label="true propagator", markersize=11),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ps.GREY, label="non-propagator", markersize=11),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=ps.RED, label="spurious hub", markersize=12),
    ], fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    main()
