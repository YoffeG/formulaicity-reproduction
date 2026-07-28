#!/usr/bin/env python3
"""Validate the figure mathematics and clean-run numerical baselines.

The validator deliberately compares numerical summaries rather than PDF or
PNG bytes. Vector files can differ in embedded metadata or font rendering
across operating systems even when every scientific result is identical.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_figure1_math import main as audit_figure_1_math


ROOT = Path(__file__).resolve().parent
EXPECTED_DIR = ROOT / "expected"
FIGURE_1_KEYS = (
    "panel",
    "dimensions",
    "p",
    "p_form",
    "fraction",
    "method",
)
FIGURE_1_VALUES = ("mean", "std")
FIGURE_6_FILES = {
    "a": "Leviticus_figure6_half_l12_n3_f500_panel-a_metrics.json",
    "b": "Leviticus_figure6_half_l6_n5_f500_panel-b_metrics.json",
}


def validate_environment(lock_path: Path) -> list[str]:
    """Return version mismatches relative to the fully pinned lock file."""

    mismatches: list[str] = []
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        package, expected = line.split("==", maxsplit=1)
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{package}: not installed; expected {expected}")
            continue
        if observed != expected:
            mismatches.append(
                f"{package}: installed {observed}; expected {expected}"
            )
    return mismatches


def validate_figure_1(output_root: Path, atol: float, rtol: float) -> None:
    expected = pd.read_csv(EXPECTED_DIR / "figure_1_summary.csv")
    observed_path = output_root / "figure_1" / "figure_1_summary.csv"
    if not observed_path.exists():
        raise FileNotFoundError(
            f"Figure 1 summary is missing: {observed_path}"
        )
    observed = pd.read_csv(observed_path)
    expected = expected.drop(columns=("index",), errors="ignore")
    observed = observed.drop(columns=("index",), errors="ignore")
    expected = expected.sort_values(list(FIGURE_1_KEYS)).reset_index(drop=True)
    observed = observed.sort_values(list(FIGURE_1_KEYS)).reset_index(drop=True)

    if len(expected) != len(observed):
        raise AssertionError(
            f"Figure 1 row count differs: {len(observed)} vs {len(expected)}"
        )
    if not expected[list(FIGURE_1_KEYS)].equals(
        observed[list(FIGURE_1_KEYS)]
    ):
        raise AssertionError("Figure 1 condition keys do not match")
    if not np.array_equal(
        expected["count"].to_numpy(), observed["count"].to_numpy()
    ):
        raise AssertionError("Figure 1 simulation counts do not match")

    for column in FIGURE_1_VALUES:
        expected_values = expected[column].to_numpy(dtype=float)
        observed_values = observed[column].to_numpy(dtype=float)
        if not np.allclose(
            observed_values,
            expected_values,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        ):
            difference = np.nanmax(
                np.abs(observed_values - expected_values)
            )
            raise AssertionError(
                f"Figure 1 {column} differs; maximum absolute difference "
                f"is {difference:.12g}"
            )
    print("PASS Figure 1: all 324 numerical summary rows match")


def compare_json(
    expected: object,
    observed: object,
    path: str,
    atol: float,
    rtol: float,
) -> None:
    """Recursively compare a metrics document with numeric tolerances."""

    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise AssertionError(f"{path}: expected a mapping")
        if set(expected) != set(observed):
            raise AssertionError(
                f"{path}: keys differ: "
                f"{set(expected) ^ set(observed)}"
            )
        for key in expected:
            compare_json(
                expected[key],
                observed[key],
                f"{path}.{key}",
                atol,
                rtol,
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise AssertionError(f"{path}: list shape differs")
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed, strict=True)
        ):
            compare_json(
                expected_item,
                observed_item,
                f"{path}[{index}]",
                atol,
                rtol,
            )
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        if not np.isclose(
            float(observed),
            float(expected),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        ):
            raise AssertionError(
                f"{path}: observed {observed!r}, expected {expected!r}"
            )
        return
    if observed != expected:
        raise AssertionError(
            f"{path}: observed {observed!r}, expected {expected!r}"
        )


def validate_figure_6(output_root: Path, atol: float, rtol: float) -> None:
    for panel, observed_name in FIGURE_6_FILES.items():
        expected_path = (
            EXPECTED_DIR / f"figure_6_panel_{panel}_metrics.json"
        )
        observed_path = output_root / "figure_6" / observed_name
        if not observed_path.exists():
            raise FileNotFoundError(
                f"Figure 6 panel {panel} metrics are missing: {observed_path}"
            )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        observed = json.loads(observed_path.read_text(encoding="utf-8"))
        compare_json(
            expected,
            observed,
            f"figure_6.panel_{panel}",
            atol,
            rtol,
        )
    print("PASS Figure 6: both Multinomial panel metrics match")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output",
        help="Directory containing figure_1/ and figure_6/ outputs.",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--skip-environment",
        action="store_true",
        help="Do not require the exact versions in requirements-lock.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_environment:
        mismatches = validate_environment(ROOT / "requirements-lock.txt")
        if mismatches:
            raise RuntimeError(
                "Environment does not match requirements-lock.txt:\n- "
                + "\n- ".join(mismatches)
            )
        print("PASS Environment: Python packages match requirements-lock.txt")
    audit_figure_1_math()
    output_root = args.output_root.resolve()
    validate_figure_1(
        output_root, args.absolute_tolerance, args.relative_tolerance
    )
    validate_figure_6(
        output_root, args.absolute_tolerance, args.relative_tolerance
    )
    print("PASS Reproduction: Figures 1 and 6 match the clean-run baselines")


if __name__ == "__main__":
    main()
