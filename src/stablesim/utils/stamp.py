"""Artifact provenance stamping.

Every CSV/JSON output that feeds the paper should have a sidecar .stamp.json
recording: calibration param hash, repo-1 schema version, N seeds, alpha,
FDR level, git SHA, timestamp.

The appendix_reproducibility.md promises this; this module delivers it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _param_hash(params: dict) -> str:
    """Short SHA-256 of sorted param dict (first 12 chars)."""
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def stamp_artifact(
    path: str | Path,
    *,
    params: dict | None = None,
    repo1_schema_version: str | None = None,
    n_seeds: int | None = None,
    alpha: float | None = None,
    fdr: float | None = None,
    synthetic_data: bool = False,
    extra: dict | None = None,
) -> Path:
    """Write a sidecar .stamp.json next to path.

    Parameters
    ----------
    path : the artifact being stamped (CSV, JSON, etc.).
    params : calibrated parameter dict → hashed.
    repo1_schema_version : schema version of hub_loader.SCHEMA_VERSION at run time.
    n_seeds : number of simulation seeds used.
    alpha : ablation dose.
    fdr : BH FDR level.
    synthetic_data : True if synthetic hub fallback was used — flags output.
    extra : any additional key-value pairs.

    Returns
    -------
    Path to the stamp file.
    """
    stamp: dict[str, Any] = {
        "artifact": str(path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "synthetic_data": synthetic_data,
    }
    if synthetic_data:
        stamp["WARNING"] = "SYNTHETIC — not for paper figures. Re-run with real repo-1 data."

    if params is not None:
        stamp["calibration_param_hash"] = _param_hash(params)
    if repo1_schema_version is not None:
        stamp["repo1_schema_version"] = repo1_schema_version
    if n_seeds is not None:
        stamp["n_seeds"] = n_seeds
    if alpha is not None:
        stamp["ablation_alpha"] = alpha
    if fdr is not None:
        stamp["bh_fdr"] = fdr
    if extra:
        stamp.update(extra)

    stamp_path = Path(str(path) + ".stamp.json")
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=2)

    return stamp_path
