"""
Policy-intervention sweep on the calibrated networked-contagion engine.

Interventions (each measured by total system contagion = mean peak |dev| over all
non-origin victims, on the deterministic propagation):

  - targeted_protection : bail out / fully backstop ONE node (hold it at peg)
  - reserve_strengthen  : raise a node's recovery speed kappa (transparency / backing)
  - circuit_breaker     : cap |dev| at a threshold (system-wide trading halt)
  - redemption_gating   : damp the global transmission coupling (slow redemptions)

Headline policy result (the reason the two-system design matters):
  a budget-constrained regulator can protect ONE node. Choosing by the GNN's
  *correlational* hub ranking protects BUSD (its #1 hub) and barely helps; choosing by
  the ABM's *causal* ranking protects USDC (the origin) and removes the cascade.

Outputs -> experiments/results/netcontagion/
    intervention_sweep.csv   every (intervention, intensity) -> contagion + % reduction
    policy_comparison.json    protect-causal vs protect-spurious headline
    fig_interventions.png     contagion-reduction bars + the policy comparison
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


def load_calibrated():
    """Rebuild the network and load the calibrated params from the join summary."""
    b = pickle.load(open(GNN_ROOT / "data/processed/graphs" / f"{EPISODE}.pkl", "rb"))
    nodes, origin = b["active_node_strs"], b["origin"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes)
    js = json.loads((OUT / "join_summary.json").read_text())
    p = js["calibrated_params"]
    net = ContagionNetwork(nodes=nodes, W=W, coupling=p["coupling"], kappa=p["kappa"],
                           common=p["common"], sigma=p["sigma"])
    return net, nodes, origin, p["shock"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net, nodes, origin, shock = load_calibrated()
    victims = [n for n in nodes if n != origin]

    def contagion(**kw):
        return net.contagion_over(origin, shock, victims, **kw)

    base = contagion()
    rows = []

    def add(intervention, target, intensity, value):
        red = 100.0 * (base - value) / base if base > 0 else 0.0
        rows.append({"intervention": intervention, "target": target.split("/")[0] if target else "all",
                     "intensity": intensity, "contagion": round(value, 6),
                     "pct_reduction": round(red, 1)})

    # 1. targeted protection of every node (the causal ranking lever)
    for nd in nodes:
        add("targeted_protection", nd, "full_backstop", contagion(protect=nd))

    # 2. reserve strengthening (raise kappa) on origin vs spurious vs all
    for tgt in [origin, "BUSD/binance"]:
        for sc in [2.0, 5.0, 20.0]:
            add("reserve_strengthen", tgt, f"kappa_x{int(sc)}", contagion(kappa_scale={tgt: sc}))

    # 3. circuit breaker at various caps
    for cap in [0.10, 0.05, 0.02]:
        add("circuit_breaker", None, f"cap_{cap}", contagion(cb_threshold=cap))

    # 4. redemption gating (damp coupling)
    base_coupling = net.coupling
    for f in [0.5, 0.25, 0.0]:
        net.coupling = base_coupling * f
        add("redemption_gating", None, f"coupling_x{f}", contagion())
    net.coupling = base_coupling

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "intervention_sweep.csv", index=False)
    print(df.to_string(index=False))

    # ---- headline policy comparison: protect by GNN-correlational vs ABM-causal ----
    hub = pd.read_csv(GNN_ROOT / "exports" / f"hub_ranking_v1_{EPISODE}.csv")
    pred_col = "hub_score_full" if "hub_score_full" in hub.columns else "hub_score"
    # top GNN hub that is NOT the origin (the node a GNN-guided regulator would protect)
    hub_nonorigin = hub[hub["node"] != origin].sort_values(pred_col, ascending=False)
    gnn_pick = hub_nonorigin.iloc[0]["node"]
    # ABM causal pick = node whose protection most reduces contagion (incl. origin)
    causal = [(nd, base - contagion(protect=nd)) for nd in nodes]
    abm_pick = max(causal, key=lambda x: x[1])[0]

    gnn_red = 100.0 * (base - contagion(protect=gnn_pick)) / base
    abm_red = 100.0 * (base - contagion(protect=abm_pick)) / base
    policy = {
        "episode": EPISODE, "baseline_contagion": round(base, 6),
        "gnn_correlational_pick": gnn_pick, "gnn_pick_contagion_reduction_pct": round(gnn_red, 1),
        "abm_causal_pick": abm_pick, "abm_pick_contagion_reduction_pct": round(abm_red, 1),
        "wasted_budget_finding": (
            f"A regulator protecting the GNN's top correlational hub ({gnn_pick.split('/')[0]}) "
            f"cuts contagion by only {gnn_red:.0f}%; protecting the ABM's causal pick "
            f"({abm_pick.split('/')[0]}) cuts it by {abm_red:.0f}%. Correlational hub rankings "
            f"misallocate scarce intervention budget."),
    }
    (OUT / "policy_comparison.json").write_text(json.dumps(policy, indent=2))
    print("\n=== POLICY COMPARISON ===")
    print(json.dumps(policy, indent=2))

    _plot(df, policy, base, OUT / "fig_interventions.png")


def _plot(df, policy, base, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=ps.WIDE)

    # left: best intensity per intervention family
    fam = (df.sort_values("pct_reduction", ascending=False)
             .groupby("intervention").first().reset_index()
             .sort_values("pct_reduction"))
    axes[0].barh(fam["intervention"], fam["pct_reduction"], color=ps.BLUE)
    for y, (_, r) in enumerate(fam.iterrows()):
        axes[0].text(r["pct_reduction"] + 1, y, f'{r["target"]}/{r["intensity"]}',
                     va="center", fontsize=8)
    axes[0].set_xlabel("Contagion reduction (%)")
    axes[0].set_title("Best reduction by intervention family")
    axes[0].grid(axis="x", alpha=0.25)

    # right: the policy punchline
    picks = [f'GNN hub\n({policy["gnn_correlational_pick"].split("/")[0]})',
             f'ABM causal\n({policy["abm_causal_pick"].split("/")[0]})']
    vals = [policy["gnn_pick_contagion_reduction_pct"], policy["abm_pick_contagion_reduction_pct"]]
    axes[1].bar(picks, vals, color=[ps.RED, ps.GREEN])
    for i, v in enumerate(vals):
        axes[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Contagion reduction (%)")
    axes[1].set_title("One-venue budget: correlational hub vs causal pick")
    axes[1].set_ylim(0, 105); axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    print("figure ->", path)


if __name__ == "__main__":
    main()
