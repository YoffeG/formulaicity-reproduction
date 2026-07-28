#!/usr/bin/env python3
"""Regenerate the Genesis, Exodus, and Leviticus parameter-sweep figures.

Raw n-gram counts are retained.  The information method uses a Multinomial
likelihood by default, and k-means receives that same count matrix.  Each of
the 260 parameter combinations per book is checkpointed independently.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parent
ENGINE = SUITE_DIR / "run_figures_3_5.py"
BOOKS = ("genesis", "exodus", "leviticus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book",
        choices=("all", *BOOKS),
        default="all",
        help="Run all three books or a single figure.",
    )
    parser.add_argument(
        "--score-model",
        choices=("multinomial", "binary", "binomial"),
        default="multinomial",
        help=(
            "Multinomial is the coherent model for the raw textual counts. "
            "The other choices are retained only for controlled comparisons."
        ),
    )
    parser.add_argument("--optimizer-restarts", type=int, default=6)
    parser.add_argument("--optimizer-iterations", type=int, default=200)
    parser.add_argument("--kmeans-restarts", type=int, default=100)
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Number of books to process in parallel.",
    )
    parser.add_argument("--limit-combinations", type=int)
    parser.add_argument("--combination-start", type=int, default=0)
    parser.add_argument("--combination-stop", type=int, default=260)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--formulaicity-gap-weight", type=float, default=0.0)
    parser.add_argument("--formulaicity-gap-bits", type=float, default=0.0)
    parser.add_argument("--formulaicity-smoothing", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            SUITE_DIR / "output" / "figures_3_5"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SUITE_DIR / "data",
        help="Self-contained biblical corpus and annotation directory.",
    )
    return parser.parse_args()


def command_for(book: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ENGINE),
        "--book",
        book,
        "--data-dir",
        str(args.data_dir.resolve()),
        "--score-model",
        args.score_model,
        "--optimizer-restarts",
        str(args.optimizer_restarts),
        "--optimizer-iterations",
        str(args.optimizer_iterations),
        "--kmeans-restarts",
        str(args.kmeans_restarts),
        "--combination-start",
        str(args.combination_start),
        "--combination-stop",
        str(args.combination_stop),
        "--formulaicity-gap-weight",
        str(args.formulaicity_gap_weight),
        "--formulaicity-gap-bits",
        str(args.formulaicity_gap_bits),
        "--formulaicity-smoothing",
        str(args.formulaicity_smoothing),
        "--output-dir",
        str(output_dir),
    ]
    if args.limit_combinations is not None:
        command.extend(
            ("--limit-combinations", str(args.limit_combinations))
        )
    if args.plot_only:
        command.append("--plot-only")
    return command


def run_book(
    book: str, args: argparse.Namespace, output_dir: Path
) -> tuple[str, int]:
    completed = subprocess.run(
        command_for(book, args, output_dir),
        cwd=SUITE_DIR,
        check=False,
    )
    return book, completed.returncode


def main() -> None:
    args = parse_args()
    if min(
        args.optimizer_restarts,
        args.optimizer_iterations,
        args.kmeans_restarts,
        args.jobs,
    ) < 1:
        raise ValueError("optimizer, k-means, and job counts must be positive")
    if not 0 <= args.combination_start < args.combination_stop <= 260:
        raise ValueError("combination range must lie within 0..260")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    books = BOOKS if args.book == "all" else (args.book,)
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(books))) as executor:
        futures = {
            executor.submit(run_book, book, args, output_dir): book
            for book in books
        }
        for future in as_completed(futures):
            book, return_code = future.result()
            if return_code:
                failures.append((book, return_code))
            else:
                print(f"[{book}] complete", flush=True)

    metadata = {
        "books": list(books),
        "score_model": args.score_model,
        "matrix_representation": (
            "binary presence"
            if args.score_model == "binary"
            else "raw n-gram counts"
        ),
        "same_matrix_for_both_methods": True,
        "optimizer_restarts": args.optimizer_restarts,
        "optimizer_iterations": args.optimizer_iterations,
        "kmeans_restarts": args.kmeans_restarts,
        "formulaicity_gap_weight": args.formulaicity_gap_weight,
        "formulaicity_gap_bits": args.formulaicity_gap_bits,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Book runs failed: {failures}")


if __name__ == "__main__":
    main()
