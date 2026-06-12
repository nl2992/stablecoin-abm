"""
Near-criticality check for the calibrated networked-contagion model (USDC/SVB).

The paper's "Why near-critical" claim is that the calibrated system sits close to,
but below, the threshold of instability. This script makes that claim a committed,
reproducible number. In the full-stress regime (every |d_i| >= stress_thr) the
homogeneous propagation in model.py linearizes exactly to

    d(t+1) = M d(t) + forcing,   M = (1 - kappa*dt) I + coupling * W,

(the origin-driven panic term is forcing proportional to the origin's own decaying
deviation, not a feedback loop through the other nodes). The system is sub-critical
iff the spectral radius rho(M) < 1; "near-critical" means rho(M) is close to 1.

Inputs (all committed):
  - W: rebuilt with estimate_transmission_matrix exactly as scripts/run_netcontagion_join.py
       (same episode pickle, max_lag=30, stress_bps=25.0)
  - calibrated params: experiments/results/netcontagion/join_summary.json

Output: experiments/results/netcontagion/near_criticality.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from stablesim.netcontagion.model import estimate_transmission_matrix  # noqa: E402

GNN_ROOT = _ROOT.parent / "stablecoin-contagion-gnn"
EPISODE = "USDC_SVB"
OUT = _ROOT / "experiments/results/netcontagion"


def main() -> None:
    with open(GNN_ROOT / "data" / "processed" / "graphs" / f"{EPISODE}.pkl", "rb") as fh:
        b = pickle.load(fh)
    nodes = b["active_node_strs"]
    dev = {n: np.asarray(b["dev_bps_1m"][n], float) for n in nodes}
    W = estimate_transmission_matrix(dev, nodes, max_lag=30, stress_bps=25.0)

    params = json.loads((OUT / "join_summary.json").read_text())["calibrated_params"]
    kappa, coupling = params["kappa"], params["coupling"]
    dt = 1.0

    M = (1.0 - kappa * dt) * np.eye(len(nodes)) + coupling * W
    eig_M = np.linalg.eigvals(M)
    eig_W = np.linalg.eigvals(W)
    rho_M = float(np.max(np.abs(eig_M)))
    rho_W = float(np.max(np.abs(eig_W)))

    res = {
        "episode": EPISODE,
        "nodes": nodes,
        "kappa": kappa,
        "coupling": coupling,
        "dt": dt,
        "spectral_radius_W": round(rho_W, 6),
        "spectral_radius_M": round(rho_M, 6),
        "distance_to_criticality": round(1.0 - rho_M, 6),
        "subcritical": bool(rho_M < 1.0),
        "note": "M = (1-kappa*dt)I + coupling*W is the exact linearization of the "
                "full-stress homogeneous propagation in model.py. rho(M)<1 => shocks "
                "decay (sub-critical); rho(M) close to 1 => slow decay (near-critical). "
                "The panic term pi*|d_origin| is forcing from the origin's own decaying "
                "deviation, not feedback through other nodes.",
    }
    out_path = OUT / "near_criticality.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
