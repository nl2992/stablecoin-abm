"""
Multi-episode generalization of the correlation->causation join.

Runs the calibrate + per-node causal-knockout pipeline on EVERY fetchable episode that has
>=3 active nodes and a GNN hub export, then asks: does the pattern "the GNN's top
correlational hub is NOT the ABM's causal driver (often a spurious, non-transmitting
co-mover)" recur across crises?

Per-episode empirical targets come from the GNN export `calibration_v1.csv`
(peak depeg -> contagion magnitude; OU half-life/5min -> crisis half-life).

Outputs -> experiments/results/netcontagion/
    multi_episode_join.csv     one row per episode (GNN top hub, ABM causal top, spurious, rho)
    multi_episode_join.json    full per-node detail
    fig_multi_episode.png      per-episode predicted-vs-causal summary
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
OUT = Path("experiments/results/netcontagion")
TOL = {"contagion_magnitude": 0.25, "cross_venue_rho": 0.30,
       "baseline_price_vol": 0.30, "crisis_half_life": 0.30}


def episode_targets(name: str) -> dict:
    cal = pd.read_csv(GNN_ROOT / "exports" / "calibration_v1.csv")
    row = cal[cal["episode"] == name]
    mag = 0.1376
    hl = 116.0
    if len(row):
        r = row.iloc[0]
        if np.isfinite(r.get("peak_depeg_bps", np.nan)):
            mag = float(min(max(r["peak_depeg_bps"] / 1e4, 0.02), 0.5))
        ou = r.get("ou_half_life_min", np.nan)
        if np.isfinite(ou) and 5 < ou < 5000:        # sane OU only; else default
            hl = float(ou / 5.0)
    return {"contagion_magnitude": mag, "cross_venue_rho": 0.576,
            "baseline_price_vol": 0.003, "crisis_half_life": hl}


def calibrate(net: ContagionNetwork, origin: str, targets: dict) -> float:
    x0 = np.array([0.015, np.log(2) / targets["crisis_half_life"], 0.0015, 0.0008, 0.13])
    bounds = [(0.004, 0.06), (0.001, 0.05), (0.0002, 0.006), (0.0001, 0.003), (0.02, 0.35)]

    def unpack(x):
        net.coupling, net.kappa, net.common, net.sigma = x[0], x[1], x[2], x[3]
        net.kappa_node = np.full(net.N, x[1], float)
        return x[4]

    def loss(x):
        shock = unpack(x)
        m = net.moments(origin, shock, n_seeds=10, shock_step=40, n_steps=240)
        L = 0.0
        for k, tgt in targets.items():
            sim = m.get(k, np.nan)
            w = {"contagion_magnitude": 2.0, "cross_venue_rho": 2.0,
                 "crisis_half_life": 1.5}.get(k, 1.0)
            L += w * (((sim - tgt) / tgt) ** 2 if np.isfinite(sim) else 5.0)
        return L

    res = minimize(loss, x0, method="Nelder-Mead", bounds=bounds,
                   options={"maxiter": 300, "fatol": 1e-5})
    return unpack(res.x)


def run_episode(name: str) -> dict | None:
    pkl = GNN_ROOT / "data/processed/graphs" / f"{name}.pkl"
    hub_csv = GNN_ROOT / "exports" / f"hub_ranking_v1_{name}.csv"
    if not pkl.exists() or not hub_csv.exists():
        return None
    b = pickle.load(open(pkl, "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    if len(nodes) < 3:
        return None
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    net = ContagionNetwork(nodes=nodes, W=W)
    targets = episode_targets(name)
    shock = calibrate(net, origin, targets)
    m = net.moments(origin, shock, n_seeds=30)
    n_pass = sum(int(abs(m[k] - targets[k]) / targets[k] <= TOL[k]) for k in targets if np.isfinite(m.get(k, np.nan)))

    # causal deltas (non-origin)
    deltas = {nd: net.causal_delta(origin, shock, nd) for nd in nodes if nd != origin}
    hub = pd.read_csv(hub_csv)
    pred_col = "hub_score_full" if "hub_score_full" in hub.columns else "hub_score"
    pred = dict(zip(hub["node"], hub[pred_col]))

    nonorigin = [n for n in nodes if n != origin]
    gnn_top = max(nonorigin, key=lambda n: pred.get(n, 0.0))
    causal_top = max(nonorigin, key=lambda n: deltas.get(n, 0.0))
    maxd = max(deltas.values()) if deltas else 0.0
    low_contagion = m["contagion_magnitude"] < 0.02  # episode produced ~no contagion

    # spurious hub = the GNN's top-predicted node that (a) is NOT the causal top and
    # (b) has near-zero causal effect (< 15% of the strongest node's). Only meaningful
    # when the episode actually produced contagion.
    spurious = None
    if not low_contagion and maxd > 1e-6:
        if gnn_top != causal_top and deltas.get(gnn_top, 0.0) < 0.15 * maxd and pred.get(gnn_top, 0) > 0:
            spurious = gnn_top

    pv = [pred.get(n, 0.0) for n in nonorigin]
    cv = [deltas.get(n, 0.0) for n in nonorigin]
    rho = (float(spearmanr(pv, cv)[0]) if len(nonorigin) >= 3 and np.std(pv) > 0
           and np.std(cv) > 0 and not low_contagion else float("nan"))

    return {
        "episode": name, "origin": origin.split("/")[0], "n_nodes": len(nodes),
        "calibration_pass": f"{n_pass}/4", "sim_contagion": round(m["contagion_magnitude"], 4),
        "low_contagion": low_contagion,
        "gnn_top_hub": gnn_top.split("/")[0],
        "gnn_top_is_causal_top": (gnn_top == causal_top) if not low_contagion else None,
        "abm_causal_top": causal_top.split("/")[0] if not low_contagion else None,
        "spurious_hub": spurious.split("/")[0] if spurious else None,
        "spearman_pred_vs_causal": round(rho, 3) if np.isfinite(rho) else None,
        "_detail": {n.split("/")[0]: {"pred": round(pred.get(n, 0.0), 3),
                                      "causal_delta": round(deltas.get(n, 0.0), 5)} for n in nonorigin},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = ["USDC_SVB", "UST_Terra", "USDT_May2022", "DAI_FTX", "BUSD_winddown"]
    results = []
    for ep in episodes:
        print(f"--- {ep} ---")
        r = run_episode(ep)
        if r is None:
            print("  skipped (no export or <3 nodes)")
            continue
        results.append(r)
        print(" ", {k: v for k, v in r.items() if k != "_detail"})
    if not results:
        print("no episodes ran"); return
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_detail"} for r in results])
    df.to_csv(OUT / "multi_episode_join.csv", index=False)
    (OUT / "multi_episode_join.json").write_text(json.dumps(results, indent=2))

    rich = df[~df["low_contagion"]]              # episodes with real contagion
    n = len(rich)
    diverge = int((rich["gnn_top_is_causal_top"] == False).sum())  # noqa: E712
    agree = int((rich["gnn_top_is_causal_top"] == True).sum())     # noqa: E712
    spurious_found = int(rich["spurious_hub"].notna().sum())
    print("\n=== MULTI-EPISODE SUMMARY ===")
    print(df.to_string(index=False))
    print(f"\nAmong {n} contagion-producing episodes: GNN top hub == ABM causal top in "
          f"{agree}/{n} (correlation happens to be right), but DIVERGES in {diverge}/{n} "
          f"(a spurious hub identified in {spurious_found}/{n}). "
          f"{len(df) - n} episodes produced ~no contagion (degenerate, excluded). "
          f"=> correlational hub rankings are unreliable; the causal test is required to "
          f"tell which regime you are in.")
    _plot(results, OUT / "fig_multi_episode.png")


def _plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.apply()
    fig, axes = plt.subplots(1, len(results), figsize=(3.15 * len(results), 3.5), squeeze=False)
    for ax, r in zip(axes[0], results):
        det = r["_detail"]
        for nd, d in det.items():
            spurious = (nd == r["spurious_hub"])
            ax.scatter(d["pred"], d["causal_delta"], s=120,
                       c=ps.RED if spurious else ps.BLUE,
                       marker="X" if spurious else "o", edgecolor="k", lw=0.5, zorder=3)
            ax.annotate(nd, (d["pred"], d["causal_delta"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_title(f'{r["episode"]}\n(rho={r["spearman_pred_vs_causal"]})',
                     fontsize=9)
        ax.set_xlabel("GNN predicted", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("ABM causal effect", fontsize=8)
    fig.suptitle("Correlation versus causation across crises (red X = spurious hub)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    main()
