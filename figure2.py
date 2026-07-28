#!/usr/bin/env python3
"""Regenerate all six panels of the Gaussian Figure 2 benchmark.

The numerical engine is the validated vectorized implementation in
``run_figure_2_drive_code.py``.  It preserves the manuscript objective while
using bounded L-BFGS-B and analytic derivatives instead of slow Powell
finite-difference searches.  Trials are deterministic and resumable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


SUITE_DIR = Path(__file__).resolve().parent
ENGINE = SUITE_DIR / "run_figure_2_drive_code.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help=(
            "Trials per noise level and panel. Use 1000 for the high-precision "
            "run used during the audit."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--workers", type=int, default=min(12, os.cpu_count() or 1)
    )
    parser.add_argument("--optimizer-iterations", type=int, default=180)
    parser.add_argument("--optimizer-starts", type=int, default=2)
    parser.add_argument("--covariance-epsilon", type=float, default=1e-2)
    parser.add_argument("--include-differential-entropy", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the PDF and PNG from an existing trial CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUITE_DIR / "output" / "figure_2",
    )
    return parser.parse_args()


def redraw(output_dir: Path) -> None:
    """Use the engine's publication formatter without rerunning simulations."""

    sys.path.insert(0, str(SUITE_DIR))
    import run_figure_2_drive_code as engine

    raw_csv = output_dir / "figure_2_trials.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"No Figure 2 checkpoint found at {raw_csv}"
        )
    summary = pd.read_csv(raw_csv)
    engine.plot_figure(
        summary,
        output_dir / "figure_2.pdf",
        output_dir / "figure_2.png",
    )
    print(f"SAVED {output_dir / 'figure_2.pdf'}")
    print(f"SAVED {output_dir / 'figure_2.png'}")


def main() -> None:
    args = parse_args()
    if min(
        args.simulations,
        args.workers,
        args.optimizer_iterations,
        args.optimizer_starts,
    ) < 1:
        raise ValueError("simulation and optimizer counts must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        redraw(output_dir)
        return

    command = [
        sys.executable,
        str(ENGINE),
        "--simulations",
        str(args.simulations),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--maxiter",
        str(args.optimizer_iterations),
        "--starts",
        str(args.optimizer_starts),
        "--epsilon",
        str(args.covariance_epsilon),
        "--output-dir",
        str(output_dir),
    ]
    if args.include_differential_entropy:
        command.append("--include-differential-entropy")
    subprocess.run(command, check=True, cwd=SUITE_DIR)


if __name__ == "__main__":
    main()
