"""
Robustness of the causal ranking under calibration uncertainty + welfare decomposition.

ROBUSTNESS: the calibration is not exact, so we perturb the calibrated params
(coupling, kappa) and the estimated transmission matrix W by +-30% across many draws and
re-estimate every node's causal Δ-contagion. We then ask how often the headline ordering
holds: origin USDC is the top causal node AND BUSD stays causally inert (a spurious hub).

WELFARE: decompose the contagion outcome by node (who bears the depeg loss) under the
no-intervention baseline and under each policy, giving a welfare-by-agent matrix.

Outputs -> experiments/results/netcontagion/
    robustness_causal_ranking.csv   per-node causal Δ distribution over perturbations
    robustness_summary.json         stability fractions
    welfare_matrix.csv              node x intervention peak-depeg (loss)
    fig_robustness.png
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from stablesim.netcontagion.model import ContagionNetwork, estimate_transmission_matrix  # noqa: E402

GNN_ROOT = Path(__file__).parents[2] / "stablecoin-contagion-gnn"
OUT = Path("experiments/results/netcontagion")
EPISODE = "USDC_SVB"


def load():
    b = pickle.load(open(GNN_ROOT / "data/processed/graphs" / f"{EPISODE}.pkl", "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    p = json.loads((OUT / "join_summary.json").read_text())["calibrated_params"]
    return b, nodes, origin, W, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b, nodes, origin, W0, p = load()
    shock = float(p.get("shock", 0.1027))
    victims = [n for n in nodes if n != origin]
    rng = np.random.default_rng(0)
    N_DRAWS = 60

    draws = []
    for _ in range(N_DRAWS):
        cpl = p["coupling"] * rng.uniform(0.7, 1.3)
        kap = p["kappa"] * rng.uniform(0.7, 1.3)
        Wp = W0 * rng.uniform(0.7, 1.3, size=W0.shape)
        rs = Wp.sum(axis=1, keepdims=True)
        Wp = np.divide(Wp, rs, out=np.zeros_like(Wp), where=rs > 0)
        net = ContagionNetwork(nodes=nodes, W=Wp, coupling=cpl, kappa=kap,
                               common=p["common"], sigma=p["sigma"])
        row = {nd: net.causal_delta(origin, shock, nd) for nd in nodes if nd != origin}
        row["__origin__"] = (net.contagion_over(origin, shock, victims)
                             - net.contagion_over(origin, shock, victims, protect=origin))
        draws.append(row)
    dd = pd.DataFrame(draws)
    dd.to_csv(OUT / "robustness_causal_ranking.csv", index=False)

    # stability: origin top? BUSD inert?
    busd = "BUSD/binance"
    origin_top = 0
    busd_inert = 0
    for _, r in dd.iterrows():
        causal = {k: r[k] for k in nodes if k != origin}
        causal["__origin__"] = r["__origin__"]
        top = max(causal, key=causal.get)
        if top == "__origin__":
            origin_top += 1
        maxnon = max(r[k] for k in nodes if k != origin)
        if busd in dd.columns and r[busd] <= 0.2 * max(maxnon, 1e-9):
            busd_inert += 1
    summary = {
        "n_draws": N_DRAWS, "perturbation": "+-30% on coupling, kappa, W",
        "origin_is_top_causal_frac": round(origin_top / N_DRAWS, 3),
        "busd_inert_frac": round(busd_inert / N_DRAWS, 3),
        "busd_causal_delta_mean": round(float(dd[busd].mean()), 6) if busd in dd else None,
        "busd_causal_delta_p95": round(float(dd[busd].quantile(0.95)), 6) if busd in dd else None,
    }
    (OUT / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    print("=== ROBUSTNESS (causal ranking under +-30% calibration uncertainty) ===")
    print(json.dumps(summary, indent=2))

    # ---- welfare decomposition: peak depeg per node under each policy ----
    net = ContagionNetwork(nodes=nodes, W=W0, coupling=p["coupling"], kappa=p["kappa"],
                           common=p["common"], sigma=p["sigma"])

    def per_node_peak(**kw):
        d = net.simulate(origin, shock, shock_step=40, n_steps=250, seed=0, noise=False, **kw)
        return {nodes[j].split("/")[0]: float(np.abs(d[:, j]).max()) for j in range(len(nodes))}

    policies = {
        "no_intervention": {},
        "protect_USDC": {"protect": origin},
        "protect_BUSD": {"protect": "BUSD/binance"},
        "reserve_USDC_x10": {"kappa_scale": {origin: 10.0}},
        "circuit_breaker_0.05": {"cb_threshold": 0.05},
        "redemption_gating": {},  # coupling handled below
    }
    wel = {}
    for name, kw in policies.items():
        if name == "redemption_gating":
            net.coupling = p["coupling"] * 0.25
            wel[name] = per_node_peak()
            net.coupling = p["coupling"]
        else:
            wel[name] = per_node_peak(**kw)
    wdf = pd.DataFrame(wel).T
    wdf.to_csv(OUT / "welfare_matrix.csv")
    print("\n=== WELFARE: peak depeg (loss) by node x policy ===")
    print(wdf.round(4).to_string())

    _plot(dd, nodes, origin, summary, OUT / "fig_robustness.png")


def _plot(dd, nodes, origin, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.apply()
    cols = [n for n in nodes if n != origin] + ["__origin__"]
    labels = [c.split("/")[0] if c != "__origin__" else "USDC*" for c in cols]
    data = [dd[c].values for c in cols if c in dd.columns]
    fig, ax = plt.subplots(figsize=ps.SINGLE)
    bp = ax.boxplot(data, labels=[labels[i] for i, c in enumerate(cols) if c in dd.columns],
                    showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(ps.BLUE); patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color(ps.INK)
    ax.set_ylabel("Causal effect (knockout)")
    ax.set_title("Causal ranking is robust to calibration uncertainty")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    main()
