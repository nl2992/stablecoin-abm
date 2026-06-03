#!/usr/bin/env python
"""Run the full SMM calibration and save the report.

Usage:
    python scripts/run_calibration.py [--n-seeds 20] [--maxiter 80] [--event usdc_svb_2023]

Outputs to experiments/results/calibration/:
    calibration_report.md   — human-readable with pass/fail gate
    calibration_report.json — machine-readable for downstream use
    calibration_moments.csv — per-moment comparison table
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stablesim.calibration.smm import SMMCalibrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--event", default="usdc_svb_2023")
    parser.add_argument("--output-dir", default="experiments/results/calibration")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"Running SMM calibration: event={args.event}, n_seeds={args.n_seeds}, maxiter={args.maxiter}")
    calibrator = SMMCalibrator(event_name=args.event)
    best_params, report = calibrator.fit(
        n_seeds=args.n_seeds,
        maxiter=args.maxiter,
        verbose=not args.quiet,
    )

    report.save(args.output_dir)
    print("\n" + report.to_markdown())

    gate = "PASS" if report.overall_pass else "FAIL"
    print(f"\nCalibration gate: {gate} ({report.n_passing}/{report.n_total} moments within tolerance)")
    if not report.overall_pass:
        print("WARNING: Calibration gate failed. Document divergences in calibration report.")
        print("         Do NOT proceed to intervention sweep until ≥3/4 moments pass.")
        sys.exit(1)


if __name__ == "__main__":
    main()
