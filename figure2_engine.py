#!/usr/bin/env python3
"""Numerical simulation and plotting engine for Figure 2.

The accompanying methods module implements the Gaussian data generator and
comparison algorithms. The two continuous optimization problems use
vectorized objectives with automatic gradients and bounded L-BFGS-B so that
the 2,000-weight panels can be evaluated efficiently.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

import figure2_methods


METHODS_MODULE_PATH = Path(__file__).resolve().parent / "figure2_methods.py"
PANEL_SPECS = (
    (50, 20),
    (50, 5),
    (50, 2),
    (10, 20),
    (10, 10),
    (10, 2),
)
NOISE_LEVELS = np.array((0.0, 0.1, 0.2, 0.5, 0.7, 0.9))
METHODS = (
    "Self-Information",
    r"$L_2$ Norm",
    "Differential Entropy",
    "Gaussian Mixture",
    "Reg. EM",
    "CEC",
)
COLORS = {
    "Self-Information": "#d62728",
    r"$L_2$ Norm": "#ff7f0e",
    "Differential Entropy": "#e6b800",
    "Gaussian Mixture": "#2ca02c",
    "Reg. EM": "#1f77b4",
    "CEC": "#9467bd",
}
WORKER_METHODS_MODULE = None


@dataclass(frozen=True)
class Config:
    simulations: int
    seed: int
    workers: int
    maxiter: int
    starts: int
    epsilon: float
    include_differential_entropy: bool


def regularized_em_vectorized(
    data,
    k,
    target_cov=None,
    max_iter=10000,
    tol=1e-4,
    reg_param=1e-2,
):
    """Vectorized regularized EM without N x d x d intermediate arrays."""

    indices = np.random.choice(len(data), k, replace=False)
    means = data[indices]
    base_covariance = np.cov(data.T) + np.eye(data.shape[1]) * 1e-3
    covariances = [base_covariance.copy() for _ in range(k)]
    weights = np.ones(k) / k
    responsibilities = np.zeros((len(data), k))

    for _ in range(max_iter):
        for cluster in range(k):
            distribution = multivariate_normal(
                mean=means[cluster],
                cov=covariances[cluster],
                allow_singular=True,
            )
            responsibilities[:, cluster] = (
                weights[cluster] * distribution.pdf(data)
            )
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        new_means = []
        new_covariances = []
        new_weights = []
        for cluster in range(k):
            response = responsibilities[:, cluster]
            effective_weight = response.sum()
            new_weights.append(effective_weight / len(data))
            mean = (
                response[:, None] * data
            ).sum(axis=0) / effective_weight
            new_means.append(mean)
            centered = data - mean
            covariance = (
                centered.T @ (response[:, None] * centered)
            ) / effective_weight
            if target_cov is not None:
                covariance = (
                    reg_param * target_cov
                    + (1.0 - reg_param) * covariance
                )
            covariance += np.eye(data.shape[1]) * 1e-3
            new_covariances.append(covariance)

        new_means = np.asarray(new_means)
        new_weights = np.asarray(new_weights)
        if np.linalg.norm(new_means - means) < tol:
            break
        means = new_means
        covariances = new_covariances
        weights = new_weights

    return np.argmax(responsibilities, axis=1), covariances


def cross_entropy_clustering_vectorized(
    data,
    k,
    max_iter=100,
    tol=1e-6,
):
    """Vectorized CEC without N x d x d intermediate arrays."""

    indices = np.random.choice(len(data), k, replace=False)
    means = data[indices]
    base_covariance = np.cov(data.T) + np.eye(data.shape[1]) * 1e-3
    covariances = [base_covariance.copy() for _ in range(k)]
    previous_means = means.copy()
    responsibilities = np.zeros((len(data), k))

    for _ in range(max_iter):
        cross_entropy = np.zeros((len(data), k))
        for cluster in range(k):
            distribution = multivariate_normal(
                mean=means[cluster],
                cov=covariances[cluster],
            )
            cross_entropy[:, cluster] = distribution.logpdf(data)
        responsibilities = np.exp(
            cross_entropy - cross_entropy.max(axis=1, keepdims=True)
        )
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        new_means = []
        new_covariances = []
        for cluster in range(k):
            response = responsibilities[:, cluster]
            effective_weight = response.sum()
            mean = (
                response[:, None] * data
            ).sum(axis=0) / effective_weight
            centered = data - mean
            covariance = (
                centered.T @ (response[:, None] * centered)
            ) / effective_weight
            covariance += np.eye(data.shape[1]) * 1e-3
            new_means.append(mean)
            new_covariances.append(covariance)
        means = np.asarray(new_means)
        covariances = new_covariances
        if np.linalg.norm(means - previous_means) < tol:
            break
        previous_means = means.copy()

    return np.argmax(responsibilities, axis=1), covariances


def initialize_worker_module():
    global WORKER_METHODS_MODULE
    WORKER_METHODS_MODULE = figure2_methods
    WORKER_METHODS_MODULE.regularized_em = regularized_em_vectorized
    WORKER_METHODS_MODULE.cross_entropy_clustering = (
        cross_entropy_clustering_vectorized
    )


def normalized_mcc(module, predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(module.norm_mcc(predicted, truth))


def bounded_lbfgsb(
    objective,
    starts: Iterable[np.ndarray],
    maxiter: int,
) -> tuple[np.ndarray, float, bool]:
    value_gradient = value_and_grad(objective)

    def wrapped(weights: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_gradient(weights)
        return float(value), np.asarray(gradient, dtype=float)

    best = None
    for start in starts:
        result = minimize(
            wrapped,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            jac=True,
            bounds=[(1e-6, 1.0)] * len(start),
            options={
                "maxiter": maxiter,
                "maxfun": maxiter * 4,
                "maxls": 30,
                "ftol": 1e-10,
                "gtol": 1e-6,
            },
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("All optimization starts failed")
    return np.asarray(best.x), float(best.fun), bool(best.success)


def make_starts(
    n_samples: int,
    rng: np.random.Generator,
    count: int,
) -> list[np.ndarray]:
    starts = [rng.uniform(0.0, 1.0, n_samples)]
    for _ in range(1, count):
        starts.append(rng.uniform(0.05, 0.95, n_samples))
    return starts


def optimize_self_information(
    samples: np.ndarray,
    rng: np.random.Generator,
    config: Config,
) -> tuple[np.ndarray, float, bool]:
    """Evaluate the manuscript self-information objective efficiently.

    This reproduces:
      std(H_all) * mean(H_all) / ||S||
    where the Gaussian is fitted with continuous sample weights and NumPy's
    default unbiased weighted-covariance normalization.
    """

    x = anp.asarray(samples)
    n_samples, dim = samples.shape
    identity = anp.eye(dim)
    constant = dim * anp.log(2.0 * anp.pi)

    def objective(weights):
        weight_sum = anp.sum(weights)
        normalized = weights / weight_sum
        mean = anp.sum(normalized[:, None] * x, axis=0)
        centered = x - mean
        covariance_numerator = anp.dot(
            centered.T, normalized[:, None] * centered
        )
        unbiased_denominator = 1.0 - anp.sum(normalized * normalized)
        covariance = (
            covariance_numerator / unbiased_denominator
            + config.epsilon * identity
        )
        logdet = anp.linalg.slogdet(covariance)[1]
        solved = anp.linalg.solve(covariance, centered.T).T
        self_information = 0.5 * (
            constant + logdet + anp.sum(centered * solved, axis=1)
        )
        return (
            anp.std(self_information)
            * anp.mean(self_information)
            / anp.linalg.norm(weights)
        )

    weights, loss, success = bounded_lbfgsb(
        objective,
        make_starts(n_samples, rng, config.starts),
        config.maxiter,
    )
    return np.round(weights).astype(int), loss, success


def optimize_l2(
    samples: np.ndarray,
    rng: np.random.Generator,
    config: Config,
) -> tuple[np.ndarray, float, bool]:
    radii = anp.asarray(np.linalg.norm(samples, axis=1))
    n_samples = len(samples)

    def objective(weights):
        weighted_radii = weights * radii
        return (
            anp.std(weighted_radii)
            * anp.mean(weighted_radii)
            / anp.linalg.norm(weights)
        )

    weights, loss, success = bounded_lbfgsb(
        objective,
        make_starts(n_samples, rng, config.starts),
        config.maxiter,
    )
    return np.round(weights).astype(int), loss, success


def run_one(
    dim: int,
    coefficient: int,
    noise: float,
    seed: int,
    config: Config,
) -> dict:
    module = WORKER_METHODS_MODULE
    if module is None:
        initialize_worker_module()
        module = WORKER_METHODS_MODULE
    np.random.seed(seed % (2**32 - 1))
    rng = np.random.default_rng(seed)
    n_per_class = dim * coefficient
    samples, labels, _, _ = module.generate_data(
        dim, n_per_class, n_per_class, noise=noise, plot=False
    )

    self_labels, self_loss, self_success = optimize_self_information(
        samples, rng, config
    )
    l2_labels, l2_loss, l2_success = optimize_l2(samples, rng, config)

    scores: dict[str, float] = {
        "Self-Information": normalized_mcc(module, self_labels, labels),
        r"$L_2$ Norm": normalized_mcc(module, l2_labels, labels),
        "Differential Entropy": np.nan,
    }
    failures: dict[str, str] = {}

    comparison_calls = {
        "Gaussian Mixture": lambda: module.gmm_clustering(samples, labels)[0],
        "Reg. EM": lambda: module.regularized_em(
            samples, 2, np.cov(samples.T)
        )[0],
        "CEC": lambda: module.cross_entropy_clustering(samples, 2)[0],
    }
    if config.include_differential_entropy:
        comparison_calls["Differential Entropy"] = (
            lambda: module.differential_entropic_clustering(
                samples, 2, max_iter=100
            )[0]
        )
    for method, call in comparison_calls.items():
        try:
            scores[method] = normalized_mcc(module, call(), labels)
        except Exception as exc:
            scores[method] = np.nan
            failures[method] = f"{type(exc).__name__}: {exc}"

    return {
        "dimension": dim,
        "coefficient": coefficient,
        "samples_per_class": n_per_class,
        "noise": noise,
        "seed": seed,
        "scores": scores,
        "self_information_loss": self_loss,
        "l2_loss": l2_loss,
        "self_information_success": self_success,
        "l2_success": l2_success,
        "failures": failures,
    }


def worker(task):
    dim, coefficient, noise, seed, config = task
    original_seed = seed
    last_error = None
    for attempt in range(5):
        retry_seed = (
            original_seed
            + attempt * 0x9E3779B97F4A7C15
        ) % (2**64)
        try:
            result = run_one(
                dim,
                coefficient,
                noise,
                retry_seed,
                config,
            )
            result["seed"] = original_seed
            result["effective_seed"] = retry_seed
            result["retries"] = attempt
            return result
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            last_error = exc

    return {
        "dimension": dim,
        "coefficient": coefficient,
        "samples_per_class": dim * coefficient,
        "noise": noise,
        "seed": original_seed,
        "effective_seed": retry_seed,
        "retries": 5,
        "scores": {method: np.nan for method in METHODS},
        "self_information_loss": np.nan,
        "l2_loss": np.nan,
        "self_information_success": False,
        "l2_success": False,
        "failures": {
            "trial": f"{type(last_error).__name__}: {last_error}"
        },
    }


def summarize(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        for method, score in result["scores"].items():
            rows.append(
                {
                    "dimension": result["dimension"],
                    "coefficient": result["coefficient"],
                    "samples_per_class": result["samples_per_class"],
                    "noise": result["noise"],
                    "seed": result["seed"],
                    "method": method,
                    "mcc": score,
                }
            )
    return pd.DataFrame(rows)


def plot_figure(summary: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), sharex=True, sharey=True)

    legend_handles = []
    for panel_index, (axis, (dim, coefficient)) in enumerate(
        zip(axes.flat, PANEL_SPECS)
    ):
        panel = summary[
            (summary["dimension"] == dim)
            & (summary["coefficient"] == coefficient)
        ]
        for method in METHODS:
            method_data = panel[panel["method"] == method]
            if not method_data["mcc"].notna().any():
                continue
            grouped = method_data.groupby("noise", sort=True)["mcc"]
            means = grouped.mean().reindex(NOISE_LEVELS).to_numpy()
            stds = grouped.std(ddof=0).reindex(NOISE_LEVELS).to_numpy()
            color = COLORS[method]
            axis.fill_between(
                NOISE_LEVELS,
                means - stds,
                means + stds,
                color=color,
                alpha=0.20,
                linewidth=0,
            )
            (line,) = axis.plot(
                NOISE_LEVELS,
                means,
                color=color,
                linewidth=2.0,
                label=method,
            )
            if panel_index == 0:
                legend_handles.append(line)

        n_per_class = dim * coefficient
        axis.set_title(
            rf"$d={dim}$, $n_1=n_2={n_per_class}$",
            fontsize=14,
            pad=5,
        )
        axis.set_xlim(-0.02, 0.92)
        axis.set_ylim(48, 102)
        axis.set_xticks((0.0, 0.2, 0.4, 0.6, 0.8))
        axis.set_yticks((50, 60, 70, 80, 90, 100))
        axis.tick_params(direction="out", length=4, width=0.8)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    figure.supxlabel("Noise", fontsize=14, y=0.105)
    figure.supylabel("MCC [%]", fontsize=14, x=0.025)

    figure.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.5,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.96,
        bottom=0.17,
        wspace=0.12,
        hspace=0.20,
    )
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output_png, dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--maxiter", type=int, default=180)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument(
        "--include-differential-entropy",
        action="store_true",
        help="Include the optional Differential Entropy comparator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "output"
        / "figure_2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        simulations=args.simulations,
        seed=args.seed,
        workers=args.workers,
        maxiter=args.maxiter,
        starts=args.starts,
        epsilon=args.epsilon,
        include_differential_entropy=args.include_differential_entropy,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_sequence = np.random.SeedSequence(config.seed)
    condition_count = len(PANEL_SPECS) * len(NOISE_LEVELS)
    condition_sequences = seed_sequence.spawn(condition_count)
    tasks = []
    condition_index = 0
    for dim, coefficient in PANEL_SPECS:
        for noise in NOISE_LEVELS:
            seeds = condition_sequences[condition_index].generate_state(
                config.simulations, dtype=np.uint64
            )
            condition_index += 1
            for seed in seeds:
                tasks.append((dim, coefficient, float(noise), int(seed), config))

    raw_csv = output_dir / "figure_2_trials.csv"
    aggregate_csv = output_dir / "figure_2_summary.csv"
    output_pdf = output_dir / "figure_2.pdf"
    output_png = output_dir / "figure_2.png"
    existing_summary = (
        pd.read_csv(raw_csv)
        if raw_csv.exists()
        else pd.DataFrame()
    )
    completed_keys = set()
    if not existing_summary.empty:
        completed_keys = set(
            existing_summary[
                ["dimension", "coefficient", "noise", "seed"]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        tasks = [
            task
            for task in tasks
            if (task[0], task[1], task[2], task[3]) not in completed_keys
        ]
        print(
            f"resuming after {len(completed_keys)}/"
            f"{condition_count * config.simulations} trials",
            flush=True,
        )

    context = mp.get_context("spawn")
    results = []
    with context.Pool(
        processes=config.workers,
        initializer=initialize_worker_module,
    ) as pool:
        for newly_completed, result in enumerate(
            pool.imap_unordered(worker, tasks, chunksize=1),
            start=1,
        ):
            results.append(result)
            completed = len(completed_keys) + newly_completed
            total = condition_count * config.simulations
            if completed % 500 == 0 or newly_completed == len(tasks):
                print(f"completed {completed}/{total}", flush=True)
            if completed % 2000 == 0:
                checkpoint = pd.concat(
                    [existing_summary, summarize(results)],
                    ignore_index=True,
                )
                checkpoint.to_csv(raw_csv, index=False)

    summary = pd.concat(
        [existing_summary, summarize(results)],
        ignore_index=True,
    )
    summary.to_csv(raw_csv, index=False)
    (
        summary.groupby(
            [
                "dimension",
                "coefficient",
                "samples_per_class",
                "noise",
                "method",
            ],
            as_index=False,
        )["mcc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_csv(aggregate_csv, index=False)
    )
    plot_figure(summary, output_pdf, output_png)

    failures = [
        {
            "dimension": result["dimension"],
            "coefficient": result["coefficient"],
            "noise": result["noise"],
            "seed": result["seed"],
            "failures": result["failures"],
        }
        for result in results
        if result["failures"]
    ]
    metadata = {
        "methods_module": str(METHODS_MODULE_PATH),
        "methods_module_modified": METHODS_MODULE_PATH.stat().st_mtime,
        "config": asdict(config),
        "objective": "Manuscript Gaussian std(H_all)*mean(H_all)/||S||",
        "data_generator": "Gaussian generator in figure2_methods.py",
        "comparators": (
            "Methods in figure2_methods.py; Differential Entropy "
            + (
                "enabled"
                if config.include_differential_entropy
                else "disabled"
            )
        ),
        "optimizer": "Algebraically equivalent vectorized objective; bounded L-BFGS-B",
        "resumed_from_trials": len(completed_keys),
        "failures": failures,
        "outputs": {
            "pdf": str(output_pdf),
            "png": str(output_png),
            "raw_csv": str(raw_csv),
            "aggregate_csv": str(aggregate_csv),
        },
    }
    (output_dir / "figure_2_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps(metadata["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
