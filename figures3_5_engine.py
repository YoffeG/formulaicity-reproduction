"""Numerical analysis and plotting engine for Figures 3--5.

The analysis follows the specifications reported in the article:

* morphological TOTHT tokens;
* the original ``Corpus.running_window`` boundaries;
* word n-grams n=1..5;
* window widths 2,3,4,6,8,10,12,14,18,22,24,26,28;
* the 100, 300, 500, or all most frequent features;
* either binary-presence matrices with a Bernoulli loss or cumulative count
  matrices with a Binomial/composite or proper Multinomial loss;
* normalized absolute MCC, 50 + 50 |MCC|.

Implementation details are optimized for the exhaustive sweep: n-gram matrices
are sparse, feature rankings are reused, K-means performs its restarts in one
fit, and every completed combination is checkpointed to CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef

from formulaicity_optimization import optimize_partition


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "figures_3_5"

NGRAMS = (1, 2, 3, 4, 5)
WINDOWS = (2, 3, 4, 6, 8, 10, 12, 14, 18, 22, 24, 26, 28)
FEATURE_COUNTS = (100, 300, 500, None)
INTERVALS = ((50, 74), (75, 84), (85, 89), (90, 95), (96, 100))

BOOKS = {
    "genesis": {
        "figure": 3,
        "title": "Genesis: genealogical lists vs. narrative",
        "csv": "Genesis_Lists_Narrative.csv",
        "label": "P",
        "pdf": "genesis_gen_results_new.pdf",
    },
    "exodus": {
        "figure": 4,
        "title": "Exodus: P vs. non-P",
        "csv": "Exodus_P-nonP_Roemer_AB.csv",
        "label": "P",
        "pdf": "Exodus_results_new.pdf",
    },
    "leviticus": {
        "figure": 5,
        "title": "Leviticus: P vs. H",
        "csv": "Leviticus _PH_14-3-2022.csv",
        "label": "P",
        "pdf": "leviticus_results_new.pdf",
    },
}


def normalized_mcc(reference: np.ndarray, predicted: np.ndarray) -> float:
    return 50.0 + 50.0 * abs(matthews_corrcoef(reference, predicted))


def load_book(book: str) -> tuple[list[tuple[str, ...]], np.ndarray]:
    config = BOOKS[book]
    table = pd.read_csv(DATA_DIR / config["csv"])
    labels = table[config["label"]].to_numpy(dtype=int)
    with open(DATA_DIR / "TOTHT_corpus_full", "rb") as stream:
        corpus = pickle.load(stream)

    documents: list[tuple[str, ...]] = []
    for row in table.itertuples(index=False):
        documents.append(
            tuple(corpus[str(row.Book)][int(row.Chapter)][int(row.Verse)]["morph"])
        )
    return documents, labels


def original_running_windows(
    documents: list[tuple[str, ...]], width: int
) -> list[tuple[str, ...]]:
    """Vector-free equivalent of the published Corpus.running_window method."""

    count = len(documents)
    windows: list[tuple[str, ...]] = []
    for index in range(count):
        start = index - min(index, width)
        stop = index + min(count - index - 1, width)
        tokens: list[str] = []
        for adjacent in range(start, stop):
            tokens.extend(documents[adjacent])
        windows.append(tuple(tokens))
    return windows


def word_ngrams(tokens: tuple[str, ...], n: int) -> tuple[tuple[str, ...], ...]:
    if len(tokens) < n:
        return ()
    return tuple(zip(*(tokens[offset:] for offset in range(n))))


def ranked_features(
    documents: list[tuple[tuple[str, ...], ...]]
) -> list[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for document in documents:
        counts.update(document)
    return [feature for feature, _ in counts.most_common()]


def count_matrix(
    documents: list[tuple[tuple[str, ...], ...]],
    labels: np.ndarray,
    features: list[tuple[str, ...]],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    feature_index = {feature: index for index, feature in enumerate(features)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    valid_indices: list[int] = []

    output_row = 0
    for original_index, document in enumerate(documents):
        counts = Counter(feature for feature in document if feature in feature_index)
        if not counts:
            continue
        valid_indices.append(original_index)
        for feature, count in counts.items():
            rows.append(output_row)
            columns.append(feature_index[feature])
            values.append(count)
        output_row += 1

    matrix = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(output_row, len(features)),
        dtype=float,
    )
    return matrix, labels[np.asarray(valid_indices, dtype=int)]


def result_key(n: int, window: int, feature_count: int | None) -> str:
    feature_label = "all" if feature_count is None else str(feature_count)
    return f"n-{n}_l-{window}_f-{feature_label}"


def load_completed(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        return {row["combination"]: row for row in csv.DictReader(stream)}


def append_result(path: Path, row: dict[str, object]) -> None:
    write_header = not path.exists()
    if write_header:
        fieldnames = tuple(row)
    else:
        with path.open(newline="") as stream:
            fieldnames = tuple(next(csv.reader(stream)))
        if set(fieldnames) != set(row):
            raise ValueError(
                f"Checkpoint columns in {path} do not match this run"
            )
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def run_book(
    book: str,
    score_model: str,
    optimizer_restarts: int,
    optimizer_iterations: int,
    kmeans_restarts: int,
    limit_combinations: int | None,
    output_dir: Path,
    formulaicity_gap_weight: float,
    formulaicity_gap_bits: float,
    formulaicity_smoothing: float,
    combination_start: int,
    combination_stop: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / (
        f"figure_{BOOKS[book]['figure']}_{book}_results.csv"
    )
    completed = load_completed(results_path)
    default_results_path = OUTPUT_DIR / (
        f"figure_{BOOKS[book]['figure']}_{book}_results.csv"
    )
    reusable_kmeans = (
        load_completed(default_results_path)
        if (
            score_model != "binary"
            and default_results_path.resolve() != results_path.resolve()
        )
        else {}
    )
    base_documents, base_labels = load_book(book)
    completed_this_run = 0
    combination_index = 0

    print(
        f"[{book}] {len(base_documents)} verses; "
        f"{len(completed)}/260 combinations already checkpointed; "
        f"score model={score_model}; "
        f"gap weight={formulaicity_gap_weight:g}, "
        f"target={formulaicity_gap_bits:g} bits",
        flush=True,
    )

    for window in WINDOWS:
        window_documents = original_running_windows(base_documents, window)
        for ngram in NGRAMS:
            ngram_documents = [
                word_ngrams(document, ngram) for document in window_documents
            ]
            feature_ranking = ranked_features(ngram_documents)

            for feature_count in FEATURE_COUNTS:
                current_index = combination_index
                combination_index += 1
                if (
                    current_index < combination_start
                    or current_index >= combination_stop
                ):
                    continue
                combination = result_key(ngram, window, feature_count)
                if combination in completed:
                    continue
                if (
                    limit_combinations is not None
                    and completed_this_run >= limit_combinations
                ):
                    return results_path

                started = time.perf_counter()
                selected_features = (
                    feature_ranking
                    if feature_count is None
                    else feature_ranking[:feature_count]
                )
                matrix, labels = count_matrix(
                    ngram_documents, base_labels, selected_features
                )
                information_matrix = matrix
                if score_model == "binary":
                    information_matrix = matrix.copy()
                    information_matrix.data.fill(1.0)

                optimizer = optimize_partition(
                    information_matrix,
                    score_model=score_model,
                    n_init=optimizer_restarts,
                    max_iter=optimizer_iterations,
                    random_state=0,
                    formulaicity_gap_weight=formulaicity_gap_weight,
                    formulaicity_gap_bits=formulaicity_gap_bits,
                    formulaicity_smoothing=formulaicity_smoothing,
                )
                information_mcc = normalized_mcc(labels, optimizer.labels)

                if combination in reusable_kmeans:
                    kmeans_mcc = float(
                        reusable_kmeans[combination]["kmeans_mcc"]
                    )
                else:
                    kmeans = KMeans(
                        n_clusters=2,
                        init="k-means++",
                        n_init=kmeans_restarts,
                        max_iter=100000,
                        tol=1e-3,
                        random_state=0,
                    )
                    kmeans_labels = kmeans.fit_predict(information_matrix)
                    kmeans_mcc = normalized_mcc(labels, kmeans_labels)
                elapsed = time.perf_counter() - started

                row = {
                    "combination": combination,
                    "ngram": ngram,
                    "window": window,
                    "features": "all" if feature_count is None else feature_count,
                    "n_samples": matrix.shape[0],
                    "n_features": matrix.shape[1],
                    "score_model": score_model,
                    "matrix_representation": (
                        "binary_presence"
                        if score_model == "binary"
                        else "counts"
                    ),
                    "information_mcc": f"{information_mcc:.10f}",
                    "kmeans_mcc": f"{kmeans_mcc:.10f}",
                    "information_loss": f"{optimizer.loss:.10f}",
                    "optimizer_success": optimizer.success,
                    "optimizer_iterations": optimizer.n_iter,
                    "optimizer_evaluations": optimizer.n_function_evaluations,
                    "elapsed_seconds": f"{elapsed:.6f}",
                }
                if formulaicity_gap_weight > 0 and formulaicity_gap_bits > 0:
                    row.update(
                        {
                            "information_model_loss": (
                                f"{optimizer.model_loss:.10f}"
                            ),
                            "formulaicity_penalty": (
                                f"{optimizer.formulaicity_penalty:.10f}"
                            ),
                            "soft_self_information_gap_bits": (
                                f"{optimizer.soft_self_information_gap_bits:.10f}"
                            ),
                            "formulaicity_gap_weight": (
                                f"{formulaicity_gap_weight:.10f}"
                            ),
                            "formulaicity_gap_target_bits": (
                                f"{formulaicity_gap_bits:.10f}"
                            ),
                        }
                    )
                append_result(results_path, row)
                completed[combination] = {key: str(value) for key, value in row.items()}
                completed_this_run += 1
                print(
                    f"[{book}] {len(completed):3d}/260 {combination}: "
                    f"information={information_mcc:.2f}, "
                    f"kmeans={kmeans_mcc:.2f}, {elapsed:.2f}s",
                    flush=True,
                )

    return results_path


def interval_counts(values: np.ndarray) -> list[int]:
    return [
        int(np.sum((values >= lower) & (values <= upper)))
        for lower, upper in INTERVALS
    ]


def plot_book(
    book: str,
    results_path: Path,
    output_dir: Path,
    *,
    regularized: bool = False,
) -> tuple[Path, Path]:
    results = pd.read_csv(results_path)
    if len(results) != 260:
        raise RuntimeError(
            f"{book} has {len(results)} completed combinations; expected 260"
        )
    config = BOOKS[book]
    methods = (
        (
            "information_mcc",
            (
                "Cross-Entropy + Gap Penalty"
                if regularized
                else "Cross-Entropy"
            ),
        ),
        ("kmeans_mcc", "k-means"),
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )

    for column, (field, title) in enumerate(methods):
        ordered = results.sort_values(field, ascending=False).head(20)
        axes[0, column].bar(
            np.arange(20),
            ordered[field],
            width=0.56,
            color="#8f8fff",
            alpha=0.92,
        )
        axes[0, column].axhline(
            95, linewidth=1, linestyle="--", color="forestgreen"
        )
        axes[0, column].axhline(
            90, linewidth=1, linestyle="--", color="orange"
        )
        axes[0, column].axhline(
            85, linewidth=1, linestyle="--", color="red"
        )
        axes[0, column].set_ylim(50, 99.5)
        axes[0, column].set_yticks(np.arange(50, 100, 5))
        axes[0, column].set_ylabel("MCC [%]")
        axes[0, column].set_title(title)
        axes[0, column].set_xticks(np.arange(20))
        axes[0, column].set_xticklabels(
            [
                f"ℓ-{row.window} n-{row.ngram} f-{row.features}"
                for row in ordered.itertuples()
            ],
            rotation=70,
            ha="right",
            rotation_mode="anchor",
            fontsize=14,
        )
        axes[0, column].tick_params(axis="x", pad=1.5)
        axes[0, column].grid(False)

        counts = interval_counts(results[field].to_numpy(dtype=float))
        interval_labels = [f"{lower}-{upper}" for lower, upper in INTERVALS]
        bars = axes[1, column].bar(
            interval_labels,
            counts,
            width=0.8,
            color="#1f77b4",
        )
        axes[1, column].set_xlabel("MCC Intervals [%]")
        axes[1, column].set_ylabel("Count")
        axes[1, column].set_facecolor("white")
        axes[1, column].grid(False)
        upper_limit = max(100, int(np.ceil(max(counts) / 20.0) * 20))
        axes[1, column].set_ylim(0, upper_limit)
        axes[1, column].yaxis.set_major_locator(MultipleLocator(20))
        axes[1, column].yaxis.set_minor_locator(MultipleLocator(10))
        axes[1, column].tick_params(
            axis="y", which="major", length=6, width=0.8
        )
        axes[1, column].tick_params(
            axis="y", which="minor", length=3, width=0.8
        )
        for bar, count in zip(bars, counts):
            x_position = bar.get_x() + bar.get_width() / 2
            if count >= 50:
                axes[1, column].text(
                    x_position,
                    count - 2,
                    str(int(count)),
                    ha="center",
                    va="top",
                    color="white",
                    fontsize=14,
                )
            else:
                axes[1, column].text(
                    x_position,
                    count + 2,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=14,
                )

    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.95,
        wspace=0.22,
        hspace=0.62,
    )
    pdf_path = output_dir / config["pdf"]
    png_path = output_dir / f"figure_{config['figure']}_{book}.png"
    figure.savefig(pdf_path, dpi=300)
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", choices=tuple(BOOKS), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the annotation CSVs and TOTHT_corpus_full.",
    )
    parser.add_argument(
        "--score-model",
        choices=("binary", "binomial", "multinomial"),
        default="multinomial",
        help=(
            "Likelihood used by the information method. Raw textual counts "
            "require multinomial (the default)."
        ),
    )
    parser.add_argument("--optimizer-restarts", type=int, default=6)
    parser.add_argument("--optimizer-iterations", type=int, default=200)
    parser.add_argument("--kmeans-restarts", type=int, default=100)
    parser.add_argument("--limit-combinations", type=int)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--formulaicity-gap-weight", type=float, default=0.0)
    parser.add_argument("--formulaicity-gap-bits", type=float, default=0.0)
    parser.add_argument("--formulaicity-smoothing", type=float, default=0.5)
    parser.add_argument("--combination-start", type=int, default=0)
    parser.add_argument("--combination-stop", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    global DATA_DIR
    args = parse_args()
    DATA_DIR = args.data_dir.resolve()
    if not (
        0 <= args.combination_start < args.combination_stop <= 260
    ):
        raise ValueError(
            "combination range must satisfy "
            "0 <= start < stop <= 260"
        )
    output_dir = args.output_dir.resolve()
    results_path = output_dir / (
        f"figure_{BOOKS[args.book]['figure']}_{args.book}_results.csv"
    )
    if not args.plot_only:
        results_path = run_book(
            args.book,
            args.score_model,
            args.optimizer_restarts,
            args.optimizer_iterations,
            args.kmeans_restarts,
            args.limit_combinations,
            output_dir,
            args.formulaicity_gap_weight,
            args.formulaicity_gap_bits,
            args.formulaicity_smoothing,
            args.combination_start,
            args.combination_stop,
        )
    if results_path.exists() and len(pd.read_csv(results_path)) == 260:
        pdf_path, png_path = plot_book(
            args.book,
            results_path,
            output_dir,
            regularized=(
                args.formulaicity_gap_weight > 0
                and args.formulaicity_gap_bits > 0
            ),
        )
        print(f"[{args.book}] SAVED {pdf_path}", flush=True)
        print(f"[{args.book}] SAVED {png_path}", flush=True)


if __name__ == "__main__":
    main()
