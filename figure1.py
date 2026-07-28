#!/usr/bin/env python3
"""Regenerate Figure 1: the sparse Bernoulli simulation benchmark.

The script uses the exact nine parameter panels and nine fractions of
formulaic dimensions from the manuscript code.  Every trial is deterministic,
all methods receive the same binary matrix, and results are checkpointed after
each batch.  Re-running the same command resumes missing trials.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import matthews_corrcoef
from sklearn.mixture import GaussianMixture


SUITE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUITE_DIR))

from formulaicity_optimization import optimize_legacy_partition


FRACTIONS = np.round(np.arange(0.1, 1.0, 0.1), 1)
METHODS = ("Cross-Entropy", "K-means", "GMM", "DBSCAN")
COLORS = {
    "Cross-Entropy": "#d62728",
    "K-means": "#ff7f0e",
    "GMM": "#e6cf00",
    "DBSCAN": "#2ca02c",
}
LINESTYLES = {
    "Cross-Entropy": "-",
    "K-means": "-",
    "GMM": "-.",
    "DBSCAN": "--",
}


@dataclass(frozen=True)
class Panel:
    dimensions: int
    probability: float
    formulaic_probability: float

    @property
    def bias(self) -> float:
        return self.formulaic_probability - self.probability


PANELS = (
    Panel(200, 0.10, 0.20),
    Panel(200, 0.05, 0.15),
    Panel(200, 0.01, 0.11),
    Panel(50, 0.10, 0.20),
    Panel(50, 0.05, 0.15),
    Panel(50, 0.01, 0.11),
    Panel(20, 0.50, 0.60),
    Panel(20, 0.50, 0.80),
    Panel(20, 0.50, 1.00),
)


@dataclass(frozen=True)
class Config:
    simulations: int
    samples_per_cluster: int
    seed: int
    workers: int
    optimizer_restarts: int
    optimizer_iterations: int


def normalized_mcc(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Return the label-invariant MCC scale used by the paper."""

    return 50.0 + 50.0 * abs(
        float(matthews_corrcoef(reference, predicted))
    )


def generate_sparse_bernoulli(
    panel: Panel,
    fraction: float,
    samples_per_cluster: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the uniform and biased binary clusters from the source code."""

    biased_dimensions = max(2, int(panel.dimensions * fraction))
    noise = 0.01
    ordinary_probabilities = np.clip(
        rng.normal(panel.probability, noise, panel.dimensions), 0.0, 1.0
    )
    formulaic_probabilities = np.concatenate(
        (
            np.clip(
                rng.normal(
                    panel.formulaic_probability,
                    noise,
                    biased_dimensions,
                ),
                0.0,
                1.0,
            ),
            np.clip(
                rng.normal(
                    panel.probability,
                    noise,
                    panel.dimensions - biased_dimensions,
                ),
                0.0,
                1.0,
            ),
        )
    )
    ordinary = rng.binomial(
        1,
        ordinary_probabilities,
        size=(samples_per_cluster, panel.dimensions),
    )
    formulaic = rng.binomial(
        1,
        formulaic_probabilities,
        size=(samples_per_cluster, panel.dimensions),
    )

    # The original generator imposes a dependency in half of the formulaic
    # samples by copying one selected biased coordinate to another.
    for sample in formulaic:
        if rng.random() < 0.5:
            first, second = rng.choice(
                biased_dimensions, size=2, replace=False
            )
            sample[second] = sample[first]

    samples = np.vstack((formulaic, ordinary)).astype(float)
    labels = np.concatenate(
        (
            np.ones(samples_per_cluster, dtype=int),
            np.zeros(samples_per_cluster, dtype=int),
        )
    )
    return samples, labels


def run_trial(task: tuple[int, int, int, Panel, float, Config]) -> list[dict]:
    """Run all four clustering methods on one shared simulated matrix."""

    panel_index, fraction_index, simulation, panel, fraction, config = task
    seed = int(
        np.random.SeedSequence(
            [config.seed, panel_index, fraction_index, simulation]
        ).generate_state(1, dtype=np.uint64)[0]
    )
    rng = np.random.default_rng(seed)
    samples, truth = generate_sparse_bernoulli(
        panel, fraction, config.samples_per_cluster, rng
    )

    information = optimize_legacy_partition(
        samples,
        score_model="binary",
        n_init=config.optimizer_restarts,
        max_iter=config.optimizer_iterations,
        random_state=seed,
    )
    predictions: dict[str, np.ndarray] = {
        "Cross-Entropy": information.labels,
        "K-means": KMeans(
            n_clusters=2,
            n_init=100,
            max_iter=100000,
            tol=1e-3,
            random_state=seed % (2**32 - 1),
        ).fit_predict(samples),
        "GMM": GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=seed % (2**32 - 1),
        ).fit_predict(samples),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5).fit_predict(samples),
    }

    common = {
        "panel": panel_index + 1,
        "dimensions": panel.dimensions,
        "p": panel.probability,
        "p_form": panel.formulaic_probability,
        "fraction": fraction,
        "simulation": simulation,
        "seed": seed,
    }
    return [
        {
            **common,
            "method": method,
            "mcc": normalized_mcc(truth, prediction),
            "information_loss": (
                information.loss if method == "Cross-Entropy" else np.nan
            ),
            "optimizer_success": (
                information.success if method == "Cross-Entropy" else ""
            ),
        }
        for method, prediction in predictions.items()
    ]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _probability(value: float) -> str:
    if np.isclose(value, 1.0):
        return "1"
    return f"{value:.2f}".rstrip("0")


def plot_figure(results: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Draw the complete 3-by-3 panel figure from checkpointed trials."""

    expected = {
        (index + 1, float(fraction), method)
        for index in range(len(PANELS))
        for fraction in FRACTIONS
        for method in METHODS
    }
    observed = set(
        results[["panel", "fraction", "method"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    missing = expected - observed
    if missing:
        raise RuntimeError(
            f"Cannot plot the full figure: {len(missing)} conditions are missing"
        )

    configure_plotting()
    figure, axes = plt.subplots(
        3, 3, figsize=(12.0, 9.0), sharex=True, constrained_layout=False
    )
    legend_handles = []

    for panel_index, (axis, panel) in enumerate(
        zip(axes.flat, PANELS), start=1
    ):
        panel_rows = results[results["panel"] == panel_index]
        all_bounds: list[float] = []
        for method in METHODS:
            method_rows = panel_rows[panel_rows["method"] == method]
            grouped = method_rows.groupby("fraction", sort=True)["mcc"]
            means = grouped.mean().reindex(FRACTIONS).to_numpy()
            stds = grouped.std(ddof=0).reindex(FRACTIONS).to_numpy()
            all_bounds.extend(means - stds)
            all_bounds.extend(means + stds)
            axis.fill_between(
                FRACTIONS,
                means - stds,
                means + stds,
                color=COLORS[method],
                alpha=0.25,
                linewidth=0,
            )
            (line,) = axis.plot(
                FRACTIONS,
                means,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.4,
                label=method,
            )
            if panel_index == 1:
                legend_handles.append(line)

        lower = float(np.nanmin(all_bounds))
        upper = float(np.nanmax(all_bounds))
        span = max(upper - lower, 3.0)
        axis.set_ylim(lower - 0.04 * span, upper + 0.04 * span)
        axis.set_xlim(0.06, 0.94)
        axis.set_xticks((0.2, 0.4, 0.6, 0.8))
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.tick_params(which="major", direction="out", length=5, width=0.9)
        axis.tick_params(which="minor", direction="out", length=2.7, width=0.7)
        axis.grid(False)
        axis.set_title(
            rf"$d={panel.dimensions},\ p={_probability(panel.probability)},"
            rf"\ p_{{\mathrm{{form}}}}="
            rf"{_probability(panel.formulaic_probability)}$",
            pad=5,
        )

    figure.supxlabel("Fraction of Formulaic Dimensions", y=0.072)
    figure.supylabel("MCC [%]", x=0.025)
    figure.legend(
        legend_handles,
        METHODS,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=4,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.8,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.992,
        top=0.96,
        bottom=0.132,
        wspace=0.15,
        hspace=0.28,
    )
    pdf = output_dir / "figure_1.pdf"
    png = output_dir / "figure_1.png"
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(png, dpi=260, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    return pdf, png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--samples-per-cluster", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--workers", type=int, default=min(12, os.cpu_count() or 1)
    )
    parser.add_argument("--optimizer-restarts", type=int, default=6)
    parser.add_argument("--optimizer-iterations", type=int, default=200)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUITE_DIR / "output" / "figure_1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.simulations,
        args.samples_per_cluster,
        args.workers,
        args.optimizer_restarts,
        args.optimizer_iterations,
    ) < 1:
        raise ValueError("simulation and optimizer counts must be positive")
    config = Config(
        simulations=args.simulations,
        samples_per_cluster=args.samples_per_cluster,
        seed=args.seed,
        workers=args.workers,
        optimizer_restarts=args.optimizer_restarts,
        optimizer_iterations=args.optimizer_iterations,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "figure_1_trials.csv"

    existing = pd.read_csv(raw_csv) if raw_csv.exists() else pd.DataFrame()
    if not existing.empty:
        existing["fraction"] = existing["fraction"].round(1)
    completed = set()
    if not existing.empty:
        completed = set(
            existing[["panel", "fraction", "simulation"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

    tasks = []
    if not args.plot_only:
        for panel_index, panel in enumerate(PANELS):
            for fraction_index, fraction in enumerate(FRACTIONS):
                for simulation in range(config.simulations):
                    key = (panel_index + 1, float(fraction), simulation)
                    if key not in completed:
                        tasks.append(
                            (
                                panel_index,
                                fraction_index,
                                simulation,
                                panel,
                                float(fraction),
                                config,
                            )
                        )

    new_rows: list[dict] = []
    if tasks:
        context = mp.get_context("spawn")
        with context.Pool(processes=config.workers) as pool:
            for count, rows in enumerate(
                pool.imap_unordered(run_trial, tasks, chunksize=1), start=1
            ):
                new_rows.extend(rows)
                if count % 25 == 0 or count == len(tasks):
                    combined = pd.concat(
                        (existing, pd.DataFrame(new_rows)), ignore_index=True
                    )
                    combined.to_csv(raw_csv, index=False)
                    print(f"completed {count}/{len(tasks)} new trials", flush=True)

    results = pd.concat(
        (existing, pd.DataFrame(new_rows)), ignore_index=True
    )
    if results.empty:
        raise FileNotFoundError(
            f"No checkpoint found at {raw_csv}; run without --plot-only first"
        )
    results = results.drop_duplicates(
        subset=("panel", "fraction", "simulation", "method"), keep="last"
    )
    results["fraction"] = results["fraction"].round(1)
    results.to_csv(raw_csv, index=False)
    summary_csv = output_dir / "figure_1_summary.csv"
    (
        results.groupby(
            ["panel", "dimensions", "p", "p_form", "fraction", "method"],
            as_index=False,
        )["mcc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_csv(summary_csv, index=False)
    )
    pdf, png = plot_figure(results, output_dir)

    metadata = {
        "config": asdict(config),
        "likelihood": "independent Bernoulli feature activations",
        "matrix": "the same binary matrix is supplied to all four methods",
        "uncertainty": "one population standard deviation across simulations",
        "outputs": {
            "pdf": str(pdf),
            "png": str(png),
            "raw_csv": str(raw_csv),
            "summary_csv": str(summary_csv),
        },
    }
    metadata_path = output_dir / "figure_1_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
