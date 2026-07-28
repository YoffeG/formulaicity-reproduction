"""Generate one Leviticus analysis panel for Figure 6.

The main axes show the features most strongly associated with the fitted
formulaic cluster.  Error bars are the standard deviation across repeated
half-sample estimates. The inset shows the leave-one-out
self-information distributions of the fitted clusters.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
    VPacker,
)
from matplotlib.patches import Rectangle
from scipy import sparse
from scipy.stats import norm

from formulaicity_optimization import optimize_partition
from figures3_5_engine import (
    DATA_DIR,
    count_matrix,
    load_book,
    normalized_mcc,
    original_running_windows,
    ranked_features,
    word_ngrams,
)


WINDOW_WIDTH = 22
NGRAM_SIZE = 3
FEATURE_COUNT = 100
N_FEATURES_TO_PLOT = 12
N_SUBSAMPLES = 500
SMOOTHING = 0.5
FONT_SIZE = 14
N_SIGNIFICANCE_PERMUTATIONS = 100_000

BLUE = "#0000FF"
ORANGE = "#FFA500"
INSET_BACKGROUND = "white"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pdf"


def self_information_distribution(
    samples: sparse.csr_matrix,
    labels: np.ndarray,
    cluster: int,
) -> np.ndarray:
    """Return count-weighted leave-one-out self-information in bits."""

    cluster_samples = samples[labels == cluster].tocsr()
    cluster_counts = np.asarray(cluster_samples.sum(axis=0)).ravel()
    cluster_total = float(cluster_counts.sum())
    vocabulary_size = samples.shape[1]
    values = np.empty(cluster_samples.shape[0], dtype=float)

    for row_index in range(cluster_samples.shape[0]):
        start = cluster_samples.indptr[row_index]
        stop = cluster_samples.indptr[row_index + 1]
        active = cluster_samples.indices[start:stop]
        row_counts = cluster_samples.data[start:stop]
        row_total = float(row_counts.sum())
        held_in_counts = cluster_counts[active] - row_counts + SMOOTHING
        held_in_total = (
            cluster_total - row_total + SMOOTHING * vocabulary_size
        )
        probabilities = held_in_counts / held_in_total
        values[row_index] = -float(
            np.sum(row_counts * np.log2(probabilities)) / row_total
        )

    return values


def empirical_mean_difference_significance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_permutations: int = N_SIGNIFICANCE_PERMUTATIONS,
    random_state: int = 0,
) -> tuple[float, float, bool]:
    """Return empirical p, Gaussian Z, and whether Z is a lower bound."""

    pooled = np.concatenate((first, second))
    first_size = len(first)
    observed = abs(float(first.mean() - second.mean()))
    total_sum = float(pooled.sum())
    mean_scale = (1.0 / first_size) + (1.0 / len(second))
    rng = np.random.default_rng(random_state)
    exceedances = 0
    batch_size = 2_000

    for start in range(0, n_permutations, batch_size):
        current_batch = min(batch_size, n_permutations - start)
        random_keys = rng.random((current_batch, len(pooled)))
        selected = np.argpartition(
            random_keys, first_size - 1, axis=1
        )[:, :first_size]
        first_sums = pooled[selected].sum(axis=1)
        permuted_differences = np.abs(
            first_sums * mean_scale - total_sum / len(second)
        )
        exceedances += int(np.count_nonzero(
            permuted_differences >= observed
        ))

    empirical_p = (exceedances + 1.0) / (n_permutations + 1.0)
    z_score = float(norm.isf(empirical_p / 2.0))
    return empirical_p, z_score, exceedances == 0


def permutation_null_z_score(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    """Return the randomization-null Z-score and its two-sided normal p-value."""

    pooled = np.concatenate((first, second))
    observed = abs(float(first.mean() - second.mean()))
    null_standard_deviation = float(
        pooled.std(ddof=1)
        * np.sqrt((1.0 / len(first)) + (1.0 / len(second)))
    )
    z_score = observed / null_standard_deviation
    return z_score, float(2.0 * norm.sf(z_score))


def paired_window_ngrams(
    table: pd.DataFrame,
    corpus: dict,
) -> tuple[
    list[tuple[tuple[str, ...], ...]],
    list[tuple[tuple[str, ...], ...]],
]:
    """Build aligned morphological and surface-form running-window n-grams."""

    morph_verses: list[tuple[str, ...]] = []
    text_verses: list[tuple[str, ...]] = []
    for row in table.itertuples(index=False):
        verse = corpus[str(row.Book)][int(row.Chapter)][int(row.Verse)]
        morph = tuple(verse["morph"])
        text = tuple(verse["text"])
        if len(morph) != len(text):
            raise RuntimeError(
                f"Unaligned verse {row.Book} {row.Chapter}:{row.Verse}"
            )
        morph_verses.append(morph)
        text_verses.append(text)

    morph_windows = original_running_windows(morph_verses, WINDOW_WIDTH)
    text_windows = original_running_windows(text_verses, WINDOW_WIDTH)
    morph_ngrams = [
        word_ngrams(tokens, NGRAM_SIZE) for tokens in morph_windows
    ]
    text_ngrams = [
        word_ngrams(tokens, NGRAM_SIZE) for tokens in text_windows
    ]
    for morph_document, text_document in zip(
        morph_ngrams, text_ngrams, strict=True
    ):
        if len(morph_document) != len(text_document):
            raise RuntimeError("Morphological and Hebrew n-grams lost alignment")
    return morph_ngrams, text_ngrams


def surface_realizations(
    morph_documents: list[tuple[tuple[str, ...], ...]],
    text_documents: list[tuple[tuple[str, ...], ...]],
) -> dict[tuple[str, ...], Counter[tuple[str, ...]]]:
    """Collect Hebrew realizations observed for every morphological n-gram."""

    mappings: dict[
        tuple[str, ...], Counter[tuple[str, ...]]
    ] = defaultdict(Counter)
    for morph_document, text_document in zip(
        morph_documents, text_documents, strict=True
    ):
        for morph_feature, text_feature in zip(
            morph_document, text_document, strict=True
        ):
            mappings[morph_feature][text_feature] += 1
    return mappings


def feature_importance_with_uncertainty(
    samples: sparse.csr_matrix,
    labels: np.ndarray,
    formulaic_cluster: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate mean cluster contrasts and half-sample uncertainty."""

    dense_samples = samples.toarray()
    rng = np.random.default_rng(0)
    all_indices = np.arange(samples.shape[0])
    estimates = np.empty((N_SUBSAMPLES, samples.shape[1]), dtype=float)

    for simulation in range(N_SUBSAMPLES):
        selected = rng.choice(
            all_indices, size=samples.shape[0] // 2, replace=False
        )
        selected_labels = labels[selected]
        formulaic_rows = dense_samples[
            selected[selected_labels == formulaic_cluster]
        ]
        other_rows = dense_samples[
            selected[selected_labels != formulaic_cluster]
        ]
        estimates[simulation] = (
            formulaic_rows.mean(axis=0) - other_rows.mean(axis=0)
        )

    return estimates.mean(axis=0), estimates.std(axis=0, ddof=1)


def rtl_display(text: str) -> str:
    """Reverse a Hebrew label for Matplotlib's left-to-right text renderer."""

    return text[::-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one Leviticus Figure 6 panel with a feature-importance "
            "bar plot and a leave-one-out self-information inset."
        )
    )
    parser.add_argument("--window", type=int, default=WINDOW_WIDTH)
    parser.add_argument("--ngram", type=int, default=NGRAM_SIZE)
    parser.add_argument("--features", type=int, default=FEATURE_COUNT)
    parser.add_argument("--panel-label", choices=("a", "b"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the Leviticus CSV and TOTHT_corpus_full.",
    )
    parser.add_argument(
        "--score-model",
        choices=("binary", "binomial", "multinomial"),
        default="multinomial",
        help=(
            "Likelihood used to fit the partition. Textual feature counts "
            "should normally use multinomial (the default)."
        ),
    )
    parser.add_argument("--optimizer-restarts", type=int, default=6)
    parser.add_argument("--optimizer-iterations", type=int, default=200)
    parser.add_argument("--subsamples", type=int, default=N_SUBSAMPLES)
    parser.add_argument(
        "--significance-permutations",
        type=int,
        default=N_SIGNIFICANCE_PERMUTATIONS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    global WINDOW_WIDTH, NGRAM_SIZE, FEATURE_COUNT
    global N_SUBSAMPLES, N_SIGNIFICANCE_PERMUTATIONS, OUTPUT_DIR, DATA_DIR
    args = parse_args()
    WINDOW_WIDTH = args.window
    NGRAM_SIZE = args.ngram
    FEATURE_COUNT = args.features
    N_SUBSAMPLES = args.subsamples
    N_SIGNIFICANCE_PERMUTATIONS = args.significance_permutations
    OUTPUT_DIR = args.output_dir.resolve()
    DATA_DIR = args.data_dir.resolve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(DATA_DIR / "Leviticus _PH_14-3-2022.csv")
    with (DATA_DIR / "TOTHT_corpus_full").open("rb") as stream:
        corpus = pickle.load(stream)

    morph_documents, text_documents = paired_window_ngrams(table, corpus)
    features = ranked_features(morph_documents)[:FEATURE_COUNT]
    _, p_labels = load_book("leviticus")
    samples, valid_p_labels = count_matrix(
        morph_documents, p_labels, features
    )
    selected_feature_set = set(features)
    valid_indices = np.asarray(
        [
            index
            for index, document in enumerate(morph_documents)
            if any(feature in selected_feature_set for feature in document)
        ],
        dtype=int,
    )
    if not np.array_equal(valid_p_labels, p_labels[valid_indices]):
        raise RuntimeError("Filtered expert-label alignment failed")

    optimizer = optimize_partition(
        samples,
        score_model=args.score_model,
        n_init=args.optimizer_restarts,
        max_iter=args.optimizer_iterations,
        random_state=args.seed,
    )
    mcc_score = normalized_mcc(valid_p_labels, optimizer.labels)
    distributions = {
        cluster: self_information_distribution(
            samples, optimizer.labels, cluster
        )
        for cluster in (0, 1)
    }
    formulaic_cluster = min(
        distributions, key=lambda cluster: distributions[cluster].mean()
    )
    non_formulaic_cluster = 1 - formulaic_cluster
    h_labels = table["H"].to_numpy(dtype=int)[valid_indices]
    p_labels_filtered = table["P"].to_numpy(dtype=int)[valid_indices]
    formulaic_mask = optimizer.labels == formulaic_cluster
    h_share_formulaic = float(h_labels[formulaic_mask].mean())
    h_share_other = float(h_labels[~formulaic_mask].mean())
    p_share_formulaic = float(p_labels_filtered[formulaic_mask].mean())
    p_share_other = float(p_labels_filtered[~formulaic_mask].mean())
    if h_share_formulaic > h_share_other:
        formulaic_source = "H-aligned"
        non_formulaic_source = "P-aligned"
    else:
        formulaic_source = "P-aligned"
        non_formulaic_source = "H-aligned"

    mean_differences, difference_uncertainties = (
        feature_importance_with_uncertainty(
            samples, optimizer.labels, formulaic_cluster
        )
    )
    formulaic_features = np.flatnonzero(mean_differences > 0)
    ordered_features = formulaic_features[
        np.argsort(mean_differences[formulaic_features])[::-1]
    ][:N_FEATURES_TO_PLOT]
    normalization = float(mean_differences[ordered_features[0]])
    normalized_importance = (
        mean_differences[ordered_features] / normalization
    )
    normalized_uncertainty = (
        difference_uncertainties[ordered_features] / normalization
    )

    mappings = surface_realizations(morph_documents, text_documents)
    hebrew_features: list[str] = []
    mapping_support: list[int] = []
    mapping_total: list[int] = []
    for feature_index in ordered_features:
        feature = features[feature_index]
        realizations = mappings[feature]
        hebrew, support = realizations.most_common(1)[0]
        hebrew_features.append(" ".join(hebrew))
        mapping_support.append(support)
        mapping_total.append(sum(realizations.values()))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(10, 7.2))
    x_positions = np.arange(len(ordered_features))
    axis.bar(
        x_positions,
        normalized_importance,
        yerr=normalized_uncertainty,
        width=0.68,
        color=ORANGE,
        alpha=0.42,
        ecolor="black",
        capsize=7,
        error_kw={"elinewidth": 1.8, "capthick": 1.8},
    )
    axis.set_ylabel("Norm. feature importance")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [rtl_display(label) for label in hebrew_features],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    axis.text(
        0.0,
        1.02,
        (
            (f"({args.panel_label})  " if args.panel_label else "")
            + f"ℓ = {WINDOW_WIDTH}, n = {NGRAM_SIZE}, "
            f"f = {FEATURE_COUNT}, MCC = {mcc_score:.1f}%"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    axis.set_ylim(
        0,
        max(1.08, float(np.max(normalized_importance + normalized_uncertainty)) * 1.08),
    )
    axis.grid(False)

    inset = axis.inset_axes([0.55, 0.57, 0.42, 0.38])
    non_formulaic = distributions[non_formulaic_cluster]
    formulaic = distributions[formulaic_cluster]
    empirical_p, empirical_z, z_is_lower_bound = (
        empirical_mean_difference_significance(
            non_formulaic,
            formulaic,
            n_permutations=N_SIGNIFICANCE_PERMUTATIONS,
            random_state=args.seed,
        )
    )
    null_z_score, null_normal_p = permutation_null_z_score(
        non_formulaic, formulaic
    )
    combined = np.concatenate((non_formulaic, formulaic))
    bin_edges = np.histogram_bin_edges(combined, bins=24)
    inset.hist(
        non_formulaic,
        bins=bin_edges,
        weights=np.full(len(non_formulaic), 1.0 / len(non_formulaic)),
        color=BLUE,
        alpha=0.70,
        edgecolor=BLUE,
        linewidth=0.45,
        label=non_formulaic_source,
    )
    inset.hist(
        formulaic,
        bins=bin_edges,
        weights=np.full(len(formulaic), 1.0 / len(formulaic)),
        color=ORANGE,
        alpha=0.70,
        edgecolor=ORANGE,
        linewidth=0.45,
        label=formulaic_source,
    )
    inset.set_xlabel("Self-information (bits)")
    inset.set_ylabel("Probability")
    inset.set_facecolor(INSET_BACKGROUND)
    inset.grid(False)
    legend_rows = []
    for label, color in (
        (non_formulaic_source, BLUE),
        (formulaic_source, ORANGE),
    ):
        swatch = DrawingArea(20, 11, 0, 0)
        swatch.add_artist(
            Rectangle(
                (0, 1),
                20,
                9,
                facecolor=color,
                edgecolor=color,
                linewidth=0.45,
                alpha=0.70,
            )
        )
        legend_rows.append(
            HPacker(
                children=[
                    swatch,
                    TextArea(label, textprops={"fontsize": FONT_SIZE}),
                ],
                align="center",
                pad=0,
                sep=6,
            )
        )
    legend_on_left = formulaic_source == "H-aligned"
    legend_content = VPacker(
        children=legend_rows,
        align="left" if legend_on_left else "right",
        pad=0,
        sep=2,
    )
    inset.add_artist(AnchoredOffsetbox(
        child=legend_content,
        loc="upper left" if legend_on_left else "upper right",
        bbox_to_anchor=(
            (0.02, 0.84) if legend_on_left else (0.98, 0.84)
        ),
        bbox_transform=inset.transAxes,
        frameon=False,
        pad=0,
        borderpad=0,
    ))
    inset.text(
        0.98,
        0.94,
        rf"${null_z_score:.1f}\sigma$",
        transform=inset.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE,
    )

    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.94,
        bottom=0.29,
    )
    output_stem = (
        f"Leviticus_figure6_half_l{WINDOW_WIDTH}_"
        f"n{NGRAM_SIZE}_f{FEATURE_COUNT}"
    )
    if args.panel_label:
        output_stem += f"_panel-{args.panel_label}"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    figure.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)

    feature_table = pd.DataFrame(
        {
            "rank": np.arange(1, len(ordered_features) + 1),
            "morphological_feature": [
                " ".join(features[index]) for index in ordered_features
            ],
            "hebrew_realization": hebrew_features,
            "realization_support": mapping_support,
            "all_realizations": mapping_total,
            "normalized_importance": normalized_importance,
            "normalized_uncertainty": normalized_uncertainty,
        }
    )
    csv_path = OUTPUT_DIR / f"{output_stem}_features.csv"
    feature_table.to_csv(csv_path, index=False)
    metrics_path = OUTPUT_DIR / f"{output_stem}_metrics.json"
    metrics = {
        "panel": args.panel_label,
        "window": WINDOW_WIDTH,
        "ngram": NGRAM_SIZE,
        "features": FEATURE_COUNT,
        "score_model": args.score_model,
        "seed": args.seed,
        "optimizer_restarts": args.optimizer_restarts,
        "optimizer_iterations": args.optimizer_iterations,
        "optimizer_success": bool(optimizer.success),
        "optimizer_loss": float(optimizer.loss),
        "normalized_mcc": float(mcc_score),
        "formulaic_cluster": int(formulaic_cluster),
        "formulaic_source": formulaic_source,
        "non_formulaic_source": non_formulaic_source,
        "h_share_formulaic": h_share_formulaic,
        "h_share_other": h_share_other,
        "p_share_formulaic": p_share_formulaic,
        "p_share_other": p_share_other,
        "formulaic_mean_self_information_bits": float(formulaic.mean()),
        "non_formulaic_mean_self_information_bits": float(
            non_formulaic.mean()
        ),
        "empirical_two_sided_p": empirical_p,
        "empirical_equivalent_z": empirical_z,
        "empirical_z_is_lower_bound": z_is_lower_bound,
        "permutation_null_z": null_z_score,
        "permutation_normal_two_sided_p": null_normal_p,
        "subsamples": N_SUBSAMPLES,
        "significance_permutations": N_SIGNIFICANCE_PERMUTATIONS,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"normalized MCC={mcc_score:.6f}")
    print(
        f"formulaic={formulaic_source}; H shares="
        f"{h_share_formulaic:.6f} vs {h_share_other:.6f}; "
        f"P shares={p_share_formulaic:.6f} vs {p_share_other:.6f}"
    )
    print(
        f"self-information means: {formulaic_source}="
        f"{formulaic.mean():.6f}, {non_formulaic_source}="
        f"{non_formulaic.mean():.6f}"
    )
    print(
        f"two-sided empirical p={empirical_p:.8g}; "
        f"equivalent Z{'>' if z_is_lower_bound else '='}"
        f"{empirical_z:.6f} sigma; "
        f"permutation-null Z={null_z_score:.6f} sigma; "
        f"two-sided normal-tail p={null_normal_p:.8g}"
    )
    print(f"SAVED {pdf_path}")
    print(f"SAVED {png_path}")
    print(f"SAVED {csv_path}")
    print(f"SAVED {metrics_path}")


if __name__ == "__main__":
    main()
