"""
intervention_spec.py
--------------------
Comparable interventions for counterfactual hub testing.

Problem this solves
====================
The original design mapped each NodeType to *its own default* intervention
(DEX_POOL -> remove liquidity, CEX_VENUE -> gate, etc.). That confounds the
causal hub ranking: delta-C then measures a mixture of (node importance) and
(how strong that node's particular knob happens to be). A pool that looks
"more causal" than a venue might just be getting a stronger treatment.

Fix
===
Define ONE primary treatment that is mechanically comparable across all node
types: a dose `alpha in [0, 1]` that removes that fraction of the node's
capacity to transmit stress. alpha = 1.0 is full ablation (node cannot
participate at all); alpha = 0.0 is the untouched baseline. Every node type
implements the same dose semantics, so delta-C at a fixed alpha is comparable
across the whole node universe.

Type-specific knobs (gating, circuit breaker, LP subsidy, ...) are retained
but DEMOTED to a secondary analysis -- run them only after the uniform-ablation
ranking is established, to answer "which real-world lever achieves the ablation
effect" rather than "which node matters".

Primary spec for the paper  : UNIFORM_ABLATION at alpha = 1.0
Robustness for the paper    : dose-response sweep over DOSE_GRID
Secondary / policy analysis : type-specific knobs (kept, not on critical path)

Integration
===========
Engine-specific wiring lives in ablation_adapter.py (the `# ADAPT:` surface).
This module is engine-agnostic; it defines treatment semantics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Re-export NodeType from hub_interventions so callers get one import
from .hub_interventions import NodeType


# --------------------------------------------------------------------------- #
# Treatment kinds                                                              #
# --------------------------------------------------------------------------- #
class TreatmentKind(str, Enum):
    UNIFORM_ABLATION = "uniform_ablation"  # PRIMARY -- comparable across types
    TYPE_SPECIFIC = "type_specific"        # SECONDARY -- policy realism only


# Dose grid for the dose-response robustness check.
# alpha = 0.0 is baseline (no-op); 1.0 is full ablation.
DOSE_GRID: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.00)

# The single dose used for the headline causal-hub ranking.
PRIMARY_DOSE: float = 1.0


@dataclass(frozen=True)
class Intervention:
    """A fully specified, applicable treatment."""
    kind: TreatmentKind
    alpha: float = PRIMARY_DOSE       # dose for UNIFORM_ABLATION
    knob: str | None = None           # name of the type-specific knob, if any
    knob_value: float | None = None   # value for that knob
    label: str = ""                   # human-readable, stamped onto outputs

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.kind is TreatmentKind.TYPE_SPECIFIC and self.knob is None:
            raise ValueError("type-specific intervention requires a knob name")


# --------------------------------------------------------------------------- #
# Builders -- what the runner iterates over                                   #
# --------------------------------------------------------------------------- #
def primary_intervention(alpha: float = PRIMARY_DOSE) -> Intervention:
    """The headline treatment: full (or dosed) uniform ablation."""
    return Intervention(
        kind=TreatmentKind.UNIFORM_ABLATION,
        alpha=alpha,
        label=f"uniform_ablation@alpha={alpha:g}",
    )


def baseline_intervention() -> Intervention:
    """alpha = 0 no-op, used as the paired control arm."""
    return Intervention(
        kind=TreatmentKind.UNIFORM_ABLATION,
        alpha=0.0,
        label="baseline",
    )


def dose_response_interventions() -> list[Intervention]:
    """One intervention per dose, for the monotonicity / dose-response check."""
    return [primary_intervention(a) for a in DOSE_GRID]


# Default type-specific knobs (secondary analysis only, NOT the causal ranking)
DEFAULT_TYPE_KNOBS: dict[NodeType, tuple[str, float]] = {
    NodeType.DEX_POOL:       ("lp_subsidy_rate", 0.01),
    NodeType.CEX_VENUE:      ("gate_fee_bps", 500.0),
    NodeType.MINT_BURN:      ("gate_delay_steps", 24),
    NodeType.BRIDGE:         ("cb_threshold", 0.02),
    NodeType.EXCHANGE_FLOW:  ("gate_queue_len", 100),
}


def type_specific_intervention(node_type: NodeType) -> Intervention:
    """The realistic per-type lever for the secondary policy analysis."""
    knob, value = DEFAULT_TYPE_KNOBS[node_type]
    return Intervention(
        kind=TreatmentKind.TYPE_SPECIFIC,
        knob=knob,
        knob_value=value,
        label=f"{node_type.value}:{knob}={value:g}",
    )
