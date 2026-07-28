#!/usr/bin/env python3
"""Regenerate the two-panel Leviticus Figure 6.

Panel (a) is the highest-MCC Multinomial configuration
``ell=12, n=3, f=500``.  Panel (b) is the selected high-significance
H-formulaic configuration ``ell=6, n=5, f=500``.  Each main panel shows
Hebrew feature importance with half-sample uncertainty; each inset shows the
correct leave-one-out self-information distributions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation


SUITE_DIR = Path(__file__).resolve().parent
PANEL_ENGINE = SUITE_DIR / "plot_figure6_h_formulaic_panel.py"
PANELS = (
    ("a", 12, 3, 500),
    ("b", 6, 5, 500),
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-model",
        choices=("multinomial", "binary", "binomial"),
        default="multinomial",
    )
    parser.add_argument("--optimizer-restarts", type=int, default=6)
    parser.add_argument("--optimizer-iterations", type=int, default=200)
    parser.add_argument("--subsamples", type=int, default=500)
    parser.add_argument("--significance-permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Combine already generated panel PDFs without refitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUITE_DIR / "output" / "figure_6",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SUITE_DIR / "data",
        help="Self-contained biblical corpus and annotation directory.",
    )
    return parser.parse_args()


def panel_stem(label: str, window: int, ngram: int, features: int) -> str:
    return (
        f"Leviticus_figure6_half_l{window}_n{ngram}_f{features}"
        f"_panel-{label}"
    )


def run_panel(
    specification: tuple[str, int, int, int],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    label, window, ngram, features = specification
    command = [
        sys.executable,
        str(PANEL_ENGINE),
        "--window",
        str(window),
        "--ngram",
        str(ngram),
        "--features",
        str(features),
        "--panel-label",
        label,
        "--data-dir",
        str(args.data_dir.resolve()),
        "--score-model",
        args.score_model,
        "--optimizer-restarts",
        str(args.optimizer_restarts),
        "--optimizer-iterations",
        str(args.optimizer_iterations),
        "--subsamples",
        str(args.subsamples),
        "--significance-permutations",
        str(args.significance_permutations),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True, cwd=SUITE_DIR)


def page_size(page) -> tuple[float, float]:
    return float(page.mediabox.width), float(page.mediabox.height)


def combine_panels(output_dir: Path) -> Path:
    """Stack the two single-page vector PDFs with a six-point gap."""

    panel_paths = [
        output_dir / f"{panel_stem(*specification)}.pdf"
        for specification in PANELS
    ]
    if any(not path.exists() for path in panel_paths):
        missing = [str(path) for path in panel_paths if not path.exists()]
        raise FileNotFoundError(f"Missing Figure 6 panels: {missing}")

    top_page = PdfReader(panel_paths[0]).pages[0]
    bottom_page = PdfReader(panel_paths[1]).pages[0]
    top_width, top_height = page_size(top_page)
    bottom_width, bottom_height = page_size(bottom_page)
    gap = 6.0
    output_width = max(top_width, bottom_width)
    output_height = top_height + gap + bottom_height

    writer = PdfWriter()
    canvas = writer.add_blank_page(
        width=output_width, height=output_height
    )
    canvas.merge_transformed_page(
        bottom_page,
        Transformation().translate(
            (output_width - bottom_width) / 2.0
            - float(bottom_page.mediabox.left),
            -float(bottom_page.mediabox.bottom),
        ),
    )
    canvas.merge_transformed_page(
        top_page,
        Transformation().translate(
            (output_width - top_width) / 2.0
            - float(top_page.mediabox.left),
            bottom_height + gap - float(top_page.mediabox.bottom),
        ),
    )
    writer.add_metadata(
        {
            "/Title": "Leviticus formulaicity clustering panels",
            "/Subject": "Feature importance and self-information distributions",
        }
    )
    output_pdf = output_dir / "figure_6.pdf"
    with output_pdf.open("wb") as stream:
        writer.write(stream)
    return output_pdf


def render_pdf(pdf: Path) -> Path | None:
    """Render the combined PDF to PNG when Poppler is available."""

    executable = shutil.which("pdftoppm")
    if executable is None:
        return None
    prefix = pdf.with_suffix("")
    subprocess.run(
        [
            str(executable),
            "-png",
            "-r",
            "180",
            "-singlefile",
            str(pdf),
            str(prefix),
        ],
        check=True,
    )
    return prefix.with_suffix(".png")


def main() -> None:
    args = parse_args()
    if min(
        args.optimizer_restarts,
        args.optimizer_iterations,
        args.subsamples,
        args.significance_permutations,
        args.jobs,
    ) < 1:
        raise ValueError("optimizer and resampling counts must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.combine_only:
        with ThreadPoolExecutor(max_workers=min(args.jobs, 2)) as executor:
            futures = [
                executor.submit(run_panel, panel, args, output_dir)
                for panel in PANELS
            ]
            for future in futures:
                future.result()

    output_pdf = combine_panels(output_dir)
    output_png = render_pdf(output_pdf)
    metadata = {
        "score_model": args.score_model,
        "panels": [
            {
                "label": label,
                "window": window,
                "ngram": ngram,
                "features": features,
            }
            for label, window, ngram, features in PANELS
        ],
        "self_information": (
            "count-weighted leave-one-out negative log2 probability"
        ),
        "subsamples": args.subsamples,
        "significance_permutations": args.significance_permutations,
        "outputs": {
            "pdf": str(output_pdf),
            "png": str(output_png) if output_png else None,
        },
    }
    (output_dir / "figure_6_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"SAVED {output_pdf}")
    if output_png:
        print(f"SAVED {output_png}")


if __name__ == "__main__":
    main()
