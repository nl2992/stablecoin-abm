"""Calibration report: simulated vs. empirical moments, explicit pass/fail.

This is the section that makes or breaks the causal claims.  Divergences are
documented as findings, not buried.  The report is both machine-readable
(JSON/CSV) and human-readable (Markdown).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MomentComparison:
    """One row in the calibration report."""

    moment: str
    empirical: float
    simulated: float
    rel_error: float
    abs_error: float
    tolerance: float
    passes: bool
    note: str = ""

    @classmethod
    def compute(
        cls,
        moment: str,
        empirical: float,
        simulated: float,
        rtol: float = 0.25,
        note: str = "",
    ) -> "MomentComparison":
        abs_err = abs(simulated - empirical)
        rel_err = abs_err / max(abs(empirical), 1e-9)
        return cls(
            moment=moment,
            empirical=empirical,
            simulated=simulated,
            rel_error=rel_err,
            abs_error=abs_err,
            tolerance=rtol,
            passes=rel_err <= rtol,
            note=note,
        )


@dataclass
class CalibrationReport:
    """Full calibration report with pass/fail gating.

    Parameters
    ----------
    best_params : dict
        Calibrated parameter vector.
    simulated_moments : dict
        Moments from the calibrated simulator.
    targets : dict
        Empirical moment targets (from calibration_targets.json).
    tolerances : dict
        Per-moment tolerance thresholds.
    optimizer_result : Any
        Raw scipy OptimizeResult.
    elapsed_seconds : float
        Wall time for calibration run.
    """

    best_params: dict
    simulated_moments: dict
    targets: dict
    tolerances: dict
    optimizer_result: Any = field(default=None, repr=False)
    elapsed_seconds: float = 0.0
    convergence_ok: bool = True
    n_de_restarts: int = 1
    _comparisons: list[MomentComparison] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_comparisons()

    def _build_comparisons(self) -> None:
        t = self.targets
        s = self.simulated_moments
        rtol_hl = self.tolerances.get("ou_half_life_rtol", 0.30)
        rtol_cm = self.tolerances.get("contagion_magnitude_rtol", 0.25)
        rtol_vol = 0.30

        self._comparisons = [
            MomentComparison.compute(
                "calm_ou_half_life (steps)",
                empirical=t.get("calm_ou_half_life_steps", 3.0),
                simulated=s.get("calm_ou_half_life", float("nan")),
                rtol=rtol_hl,
                note="Steps for depeg to decay by half in no-shock baseline (1 step ≈ 5 min)",
            ),
            MomentComparison.compute(
                "crisis_contagion_magnitude",
                empirical=t.get("contagion_magnitude_high", 0.842),
                simulated=s.get("contagion_magnitude", float("nan")),
                rtol=rtol_cm,
                note="Peak |depeg| during shock episode; empirical from mean_abs_effect in repo 1",
            ),
            MomentComparison.compute(
                "baseline_price_vol",
                empirical=t.get("baseline_price_vol", 0.003),
                simulated=s.get("baseline_price_vol", float("nan")),
                rtol=rtol_vol,
                note="Std of Δprice in no-shock baseline",
            ),
            MomentComparison.compute(
                "cross_venue_rho (crisis)",
                empirical=t.get("cross_venue_rho_crisis", 0.576),
                simulated=s.get("cross_venue_rho", float("nan")),
                rtol=0.30,
                note="Pearson correlation of pool prices during shock; empirical from FEVD share (repo 1 TVP-VAR)",
            ),
        ]

    # ------------------------------------------------------------------
    # Pass/fail summary

    @property
    def n_passing(self) -> int:
        return sum(c.passes for c in self._comparisons)

    @property
    def n_total(self) -> int:
        return len(self._comparisons)

    @property
    def overall_pass(self) -> bool:
        """Pass if ≥ 3 of 4 moments are within tolerance (ROADMAP gate)."""
        return self.n_passing >= 3

    def comparisons_df(self) -> pd.DataFrame:
        rows = [asdict(c) for c in self._comparisons]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Divergence analysis

    def divergences(self) -> list[MomentComparison]:
        """Moments that failed.  Treat as findings, not failures to bury."""
        return [c for c in self._comparisons if not c.passes]

    def divergence_notes(self) -> str:
        """Plain-English explanation of each divergence for honest documentation."""
        divs = self.divergences()
        if not divs:
            return "No divergences — all moments within tolerance."
        lines = ["## Calibration Divergences\n"]
        lines.append(
            "The following moments failed tolerance gates. These are findings, "
            "not failures: they reveal ABM mechanisms that may differ from the "
            "real market and should be discussed in the Limitations section.\n"
        )
        for c in divs:
            lines.append(
                f"- **{c.moment}**: simulated={c.simulated:.4f}, "
                f"empirical={c.empirical:.4f}, "
                f"relative error={c.rel_error:.1%} (tolerance={c.tolerance:.1%})"
            )
            if c.note:
                lines.append(f"  _{c.note}_")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation

    def to_markdown(self) -> str:
        df = self.comparisons_df()
        gate = "✅ PASS" if self.overall_pass else "❌ FAIL"
        conv = "✅ converged" if self.convergence_ok else "⚠️ NOT converged — results fragile"
        lines = [
            f"# Calibration Report\n",
            f"**Gate: {gate}** ({self.n_passing}/{self.n_total} moments within tolerance)\n",
            f"**DE convergence ({self.n_de_restarts} restarts):** {conv}\n",
            f"**Identification:** 4 free params × 4 moments (just-identified; noise_trade_size fixed at $2k structural prior)\n",
            f"**Elapsed:** {self.elapsed_seconds:.0f}s\n",
            f"\n## Calibrated Parameters\n",
        ]
        for k, v in self.best_params.items():
            lines.append(f"- `{k}` = {v:.6g}")
        lines.append("\n## Moment Comparison\n")
        lines.append(df[["moment", "empirical", "simulated", "rel_error", "tolerance", "passes"]]
                     .to_markdown(index=False, floatfmt=".4f"))
        lines.append("\n")
        lines.append(self.divergence_notes())
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "gate_pass": self.overall_pass,
            "n_passing": self.n_passing,
            "n_total": self.n_total,
            "elapsed_seconds": self.elapsed_seconds,
            "convergence_ok": self.convergence_ok,
            "n_de_restarts": self.n_de_restarts,
            "identification": "just-identified: 4 free params x 4 moments",
            "best_params": self.best_params,
            "simulated_moments": self.simulated_moments,
            "comparisons": [asdict(c) for c in self._comparisons],
        }

    def save(self, output_dir: str | Path = "experiments/results/calibration") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "calibration_report.md", "w") as f:
            f.write(self.to_markdown())
        with open(out / "calibration_report.json", "w") as f:
            json.dump(self.to_json(), f, indent=2)
        self.comparisons_df().to_csv(out / "calibration_moments.csv", index=False)
        print(f"Calibration report saved to {out}/")
