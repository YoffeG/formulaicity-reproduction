"""Fast optimization of the published loss and semantic cluster orientation.

The optimizer in this module deliberately retains the probability estimates
and loss used by ``entropy_optim.py``.  It changes only the numerical method:
the loss and its analytic gradient are evaluated with vectorized operations,
memberships remain continuous during optimization, and the final solution is
thresholded once.

Cluster fitting and the later formulaic/non-formulaic designation are kept
separate by default.  An explicitly enabled regularizer can additionally
require a minimum leave-one-out self-information gap between the two clusters.
Its default weight is zero, so published-loss results remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


_LOG_EPSILON = 1e-10
_LOG_2 = np.log(2.0)


@dataclass(frozen=True)
class LegacyOptimizationResult:
    """Result of optimizing the published loss, optionally with a gap cost."""

    labels: np.ndarray
    membership: np.ndarray
    loss: float
    log_loss: float
    n_iter: int
    success: bool
    message: str
    score_model: str
    restart: int
    n_function_evaluations: int
    legacy_loss: float = np.nan
    formulaicity_penalty: float = 0.0
    soft_self_information_gap_bits: float = np.nan


@dataclass(frozen=True)
class ClusterFormulaicityStats:
    """Descriptive formulaicity measurements for one fitted cluster."""

    cluster: int
    n_samples: int
    total_tokens: float
    vocabulary_size: int
    entropy_bits: float
    normalized_entropy: float
    mean_self_information_bits: float
    std_self_information_bits: float
    mean_repetition_rate: float
    top_10_share: float
    top_100_share: float


@dataclass(frozen=True)
class FormulaicityResult:
    """Semantic orientation of an otherwise unlabeled two-cluster solution."""

    formulaic_cluster: int
    non_formulaic_cluster: int
    criterion: str
    diagnostics_agree: bool
    stats: tuple[ClusterFormulaicityStats, ClusterFormulaicityStats]

    def oriented_labels(self, labels: np.ndarray) -> np.ndarray:
        """Return labels with 1=formulaic and 0=non-formulaic."""

        labels = np.asarray(labels)
        return (labels == self.formulaic_cluster).astype(int)


def _as_feature_matrix(samples: np.ndarray) -> np.ndarray:
    if sparse.issparse(samples):
        matrix = sparse.csr_matrix(samples, dtype=float)
        finite_values = np.all(np.isfinite(matrix.data))
        has_negative_values = np.any(matrix.data < 0)
        row_totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        matrix = np.asarray(samples, dtype=float)
        finite_values = np.all(np.isfinite(matrix))
        has_negative_values = np.any(matrix < 0)
        row_totals = matrix.sum(axis=1) if matrix.ndim == 2 else np.array([])
    if matrix.ndim != 2:
        raise ValueError(
            f"samples must be a two-dimensional matrix; got {matrix.shape}"
        )
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("samples must contain at least two rows and one feature")
    if not finite_values:
        raise ValueError("samples contains NaN or infinite values")
    if has_negative_values:
        raise ValueError("samples must contain non-negative feature values")
    # An all-zero row is a valid observation for a Bernoulli feature model
    # (and occurs naturally in the sparse Figure 1 benchmark).  The textual
    # count pipeline filters empty windows before optimization, so accepting
    # such rows here does not alter Figures 3--6.
    return matrix


def _canonical_score_model(score_model: str) -> str:
    model = score_model.lower()
    if model in {"binary", "bernoulli"}:
        return "binary"
    if model in {"binomial", "multinomial"}:
        return model
    raise ValueError(
        "score_model must be one of: binary, bernoulli, binomial, multinomial"
    )


def _cluster_log_score_and_gradient(
    samples: np.ndarray,
    weights: np.ndarray,
    score_model: str,
) -> tuple[float, np.ndarray]:
    """Return the model-consistent weighted log score and its exact gradient.

    Bernoulli feature probabilities are weighted occurrence rates across
    samples and therefore use the total sample weight as their denominator.
    Count/Binomial and multinomial probabilities instead use the total weighted
    feature count.  Logarithms retain the original ``1e-10`` offset.  Terms in
    the binomial coefficient are omitted because their total is independent of
    the partition and therefore cannot affect the optimum.
    """

    row_totals = np.asarray(samples.sum(axis=1)).ravel()
    feature_sums = np.asarray(weights @ samples).ravel()

    if score_model == "binary":
        total_sample_weight = float(weights.sum())
        if total_sample_weight <= 0:
            return -np.inf, np.full(samples.shape[0], np.nan)

        probabilities = feature_sums / total_sample_weight
        log_p = np.log(probabilities + _LOG_EPSILON)
        complementary_sums = total_sample_weight - feature_sums
        log_not_p = np.log(1.0 - probabilities + _LOG_EPSILON)
        score = (
            feature_sums @ log_p
            + complementary_sums @ log_not_p
        ) / _LOG_2
        direct_gradient = (
            np.asarray(samples @ (log_p - log_not_p)).ravel()
            + log_not_p.sum()
        ) / _LOG_2
        probability_derivative_scale = feature_sums / (
            probabilities + _LOG_EPSILON
        ) - complementary_sums / (
            1.0 - probabilities + _LOG_EPSILON
        )

        # dp_j/dw_i = (x_ij - p_j) / total_sample_weight
        probability_gradient = (
            np.asarray(samples @ probability_derivative_scale).ravel()
            - probabilities @ probability_derivative_scale
        ) / total_sample_weight
        return float(score), direct_gradient + probability_gradient / _LOG_2

    all_weighted_features = float(feature_sums.sum())
    if all_weighted_features <= 0:
        return -np.inf, np.full(samples.shape[0], np.nan)

    probabilities = feature_sums / all_weighted_features
    log_p = np.log(probabilities + _LOG_EPSILON)
    probability_derivative_scale = feature_sums / (
        probabilities + _LOG_EPSILON
    )

    if score_model == "binomial":
        complementary_sums = all_weighted_features - feature_sums
        log_not_p = np.log(1.0 - probabilities + _LOG_EPSILON)
        score = (
            feature_sums @ log_p
            + complementary_sums @ log_not_p
        )
        direct_gradient = (
            np.asarray(samples @ (log_p - log_not_p)).ravel()
            + row_totals * log_not_p.sum()
        )
        probability_derivative_scale -= complementary_sums / (
            1.0 - probabilities + _LOG_EPSILON
        )
    else:
        score = feature_sums @ log_p
        direct_gradient = np.asarray(samples @ log_p).ravel()

    # dp_j/dw_i = (x_ij - p_j * row_total_i) / total_feature_weight
    probability_gradient = (
        np.asarray(samples @ probability_derivative_scale).ravel()
        - row_totals * (probabilities @ probability_derivative_scale)
    ) / all_weighted_features

    return float(score), direct_gradient + probability_gradient


def legacy_partition_loss_and_gradient(
    membership: np.ndarray,
    samples: np.ndarray,
    score_model: str = "binary",
) -> tuple[float, np.ndarray]:
    """Evaluate the unchanged published loss and its analytic gradient."""

    matrix = _as_feature_matrix(samples)
    model = _canonical_score_model(score_model)
    membership = np.asarray(membership, dtype=float)
    if membership.shape != (matrix.shape[0],):
        raise ValueError(
            f"membership must have shape ({matrix.shape[0]},), "
            f"got {membership.shape}"
        )
    if np.any(~np.isfinite(membership)):
        raise ValueError("membership contains NaN or infinite values")
    if np.any((membership <= 0.0) | (membership >= 1.0)):
        raise ValueError("membership values must lie strictly between 0 and 1")
    matrix_values = matrix.data if sparse.issparse(matrix) else matrix
    if model == "binary" and np.any(matrix_values > 1.0):
        raise ValueError(
            "The published binary loss requires a binary feature matrix"
        )

    cluster_1_score, cluster_1_gradient = _cluster_log_score_and_gradient(
        matrix, membership, model
    )
    cluster_0_score, cluster_0_gradient_wrt_complement = (
        _cluster_log_score_and_gradient(matrix, 1.0 - membership, model)
    )
    loss = -(cluster_1_score + cluster_0_score)
    gradient = (
        cluster_0_gradient_wrt_complement - cluster_1_gradient
    )
    return float(loss), gradient


def legacy_partition_loss(
    membership: np.ndarray,
    samples: np.ndarray,
    score_model: str = "binary",
) -> float:
    """Evaluate only the unchanged published loss."""

    return legacy_partition_loss_and_gradient(
        membership, samples, score_model
    )[0]


def _soft_leave_one_out_mean_self_information_and_gradient(
    samples: np.ndarray,
    weights: np.ndarray,
    smoothing: float,
) -> tuple[float, np.ndarray]:
    """Return soft leave-one-out mean self-information and its gradient.

    Each row is scored against cluster feature counts excluding its own
    weighted contribution.  The row scores are then averaged using the same
    soft memberships.  This makes the formulaicity term differentiable without
    letting a sample make itself artificially predictable.
    """

    row_totals = np.asarray(samples.sum(axis=1)).ravel()
    n_samples, vocabulary_size = samples.shape
    total_weight = float(weights.sum())
    feature_counts = np.asarray(weights @ samples).ravel()
    total_feature_count = float(feature_counts.sum())
    held_out_totals = (
        total_feature_count
        - weights * row_totals
        + smoothing * vocabulary_size
    )
    if total_weight <= 0 or np.any(held_out_totals <= 0):
        raise ValueError("soft cluster weights must have positive total mass")

    if sparse.issparse(samples):
        coordinate_matrix = sparse.coo_matrix(samples)
        row_indices = coordinate_matrix.row
        column_indices = coordinate_matrix.col
        values = coordinate_matrix.data
        held_out_counts = (
            feature_counts[column_indices]
            - weights[row_indices] * values
            + smoothing
        )
        log_probabilities = np.log(
            held_out_counts / held_out_totals[row_indices]
        )
        row_log_scores = np.bincount(
            row_indices,
            weights=values * log_probabilities,
            minlength=n_samples,
        )
        self_information = -row_log_scores / (row_totals * _LOG_2)

        dependency_values = (
            (weights[row_indices] / total_weight)
            * (values / row_totals[row_indices])
            / held_out_counts
        )
        feature_dependency = np.bincount(
            column_indices,
            weights=dependency_values,
            minlength=vocabulary_size,
        )
        all_rows_dependency = np.asarray(
            samples @ feature_dependency
        ).ravel()
        own_row_dependency = np.bincount(
            row_indices,
            weights=values * dependency_values,
            minlength=n_samples,
        )
    else:
        matrix = np.asarray(samples, dtype=float)
        held_out_counts = (
            feature_counts[None, :]
            - weights[:, None] * matrix
            + smoothing
        )
        log_probabilities = np.log(
            held_out_counts / held_out_totals[:, None]
        )
        self_information = -(
            (matrix * log_probabilities).sum(axis=1)
            / (row_totals * _LOG_2)
        )
        dependency_matrix = (
            (weights[:, None] / total_weight)
            * (matrix / row_totals[:, None])
            / held_out_counts
        )
        feature_dependency = dependency_matrix.sum(axis=0)
        all_rows_dependency = matrix @ feature_dependency
        own_row_dependency = (matrix * dependency_matrix).sum(axis=1)

    mean_self_information = float(
        weights @ self_information / total_weight
    )
    direct_gradient = (
        self_information - mean_self_information
    ) / total_weight
    inverse_total_average = float(
        np.sum(weights / (total_weight * held_out_totals))
    )
    dependency_gradient = (
        -(all_rows_dependency - own_row_dependency) / _LOG_2
        + row_totals
        * (
            inverse_total_average
            - weights / (total_weight * held_out_totals)
        )
        / _LOG_2
    )
    return mean_self_information, direct_gradient + dependency_gradient


def partition_loss_and_gradient(
    membership: np.ndarray,
    samples: np.ndarray,
    score_model: str = "binary",
    formulaicity_gap_weight: float = 0.0,
    formulaicity_gap_bits: float = 0.0,
    formulaicity_smoothing: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Evaluate the published loss plus an optional formulaicity-gap cost.

    The optional addition is

    ``legacy_loss * weight * max(0, target_gap - observed_gap) ** 2``,

    where ``observed_gap`` is the absolute difference between the clusters'
    soft leave-one-out mean self-information.  Multiplication by the legacy
    loss makes ``weight`` comparable across matrices whose raw loss scales
    differ.  A zero weight follows the unchanged published-loss path exactly.
    """

    if formulaicity_gap_weight < 0:
        raise ValueError("formulaicity_gap_weight must be non-negative")
    if formulaicity_gap_bits < 0:
        raise ValueError("formulaicity_gap_bits must be non-negative")
    if formulaicity_smoothing <= 0:
        raise ValueError("formulaicity_smoothing must be positive")

    legacy_loss, legacy_gradient = legacy_partition_loss_and_gradient(
        membership, samples, score_model
    )
    if formulaicity_gap_weight == 0.0 or formulaicity_gap_bits == 0.0:
        return legacy_loss, legacy_gradient

    cluster_1_mean, cluster_1_gradient = (
        _soft_leave_one_out_mean_self_information_and_gradient(
            samples,
            np.asarray(membership, dtype=float),
            formulaicity_smoothing,
        )
    )
    cluster_0_mean, cluster_0_weight_gradient = (
        _soft_leave_one_out_mean_self_information_and_gradient(
            samples,
            1.0 - np.asarray(membership, dtype=float),
            formulaicity_smoothing,
        )
    )
    signed_gap = cluster_1_mean - cluster_0_mean
    observed_gap = abs(signed_gap)
    shortfall = max(0.0, formulaicity_gap_bits - observed_gap)
    if shortfall == 0.0:
        return legacy_loss, legacy_gradient

    relative_penalty = formulaicity_gap_weight * shortfall**2
    total_loss = legacy_loss * (1.0 + relative_penalty)
    signed_gap_gradient = (
        cluster_1_gradient + cluster_0_weight_gradient
    )
    shortfall_squared_gradient = (
        -2.0
        * shortfall
        * np.sign(signed_gap)
        * signed_gap_gradient
    )
    total_gradient = (
        (1.0 + relative_penalty) * legacy_gradient
        + legacy_loss
        * formulaicity_gap_weight
        * shortfall_squared_gradient
    )
    return float(total_loss), total_gradient


def _initial_membership(
    samples: np.ndarray,
    restart: int,
    rng: np.random.Generator,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    n_samples, n_features = samples.shape
    if restart == 0:
        projection = np.asarray(samples.sum(axis=1)).ravel()
    else:
        direction = rng.choice((-1.0, 1.0), size=n_features)
        projection = np.asarray(samples @ direction).ravel()

    order = np.argsort(projection, kind="stable")
    membership = np.full(n_samples, 0.1)
    membership[order[n_samples // 2 :]] = 0.9
    return np.clip(membership, lower_bound, upper_bound)


def optimize_legacy_partition(
    samples: np.ndarray,
    score_model: str = "binary",
    *,
    n_init: int = 6,
    max_iter: int = 500,
    tolerance: float = 1e-8,
    min_component_fraction: float = 0.02,
    random_state: int | None = 0,
    logit_bound: float = 10.0,
    formulaicity_gap_weight: float = 0.0,
    formulaicity_gap_bits: float = 0.0,
    formulaicity_smoothing: float = 0.5,
) -> LegacyOptimizationResult:
    """Optimize the published loss with multi-start L-BFGS-B.

    Directly optimizing membership in the sigmoid range is equivalent to the
    old bounded-logit parameterization, but avoids applying ``sigmoid`` during
    every loss evaluation.  The analytic gradient makes each optimizer step
    require one vectorized pass rather than hundreds of finite-difference or
    Powell evaluations.

    ``formulaicity_gap_weight`` optionally activates a relative cost when the
    clusters' soft leave-one-out mean self-information gap is smaller than
    ``formulaicity_gap_bits``.  Its default of zero preserves the published
    objective exactly.  ``min_component_fraction`` supplies the corresponding
    minimum-size safeguard after memberships are thresholded.
    """

    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError(
            "optimize_legacy_partition requires scipy, which is already a "
            "dependency of entropy_optim.py"
        ) from error

    matrix = _as_feature_matrix(samples)
    model = _canonical_score_model(score_model)
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must both be positive")
    if not 0.0 <= min_component_fraction < 0.5:
        raise ValueError("min_component_fraction must be in [0, 0.5)")
    if formulaicity_gap_weight < 0:
        raise ValueError("formulaicity_gap_weight must be non-negative")
    if formulaicity_gap_bits < 0:
        raise ValueError("formulaicity_gap_bits must be non-negative")
    if formulaicity_smoothing <= 0:
        raise ValueError("formulaicity_smoothing must be positive")

    lower_bound = 1.0 / (1.0 + np.exp(logit_bound))
    upper_bound = 1.0 - lower_bound
    bounds = [(lower_bound, upper_bound)] * matrix.shape[0]
    rng = np.random.default_rng(random_state)
    best_result: LegacyOptimizationResult | None = None
    collapsed_sizes: list[tuple[int, int]] = []
    use_formulaicity_gap = (
        formulaicity_gap_weight > 0.0
        and formulaicity_gap_bits > 0.0
    )
    objective_function = (
        partition_loss_and_gradient
        if use_formulaicity_gap
        else legacy_partition_loss_and_gradient
    )
    objective_arguments = (
        (
            matrix,
            model,
            formulaicity_gap_weight,
            formulaicity_gap_bits,
            formulaicity_smoothing,
        )
        if use_formulaicity_gap
        else (matrix, model)
    )

    for restart in range(n_init):
        initial_membership = _initial_membership(
            matrix, restart, rng, lower_bound, upper_bound
        )
        scipy_result = minimize(
            fun=objective_function,
            x0=initial_membership,
            args=objective_arguments,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "ftol": tolerance,
                "gtol": tolerance,
                "maxls": 50,
            },
        )
        membership = np.asarray(scipy_result.x, dtype=float)
        labels = (membership >= 0.5).astype(int)
        sizes = np.bincount(labels, minlength=2)
        fractions = sizes / len(labels)
        if np.min(fractions) < min_component_fraction:
            collapsed_sizes.append((int(sizes[0]), int(sizes[1])))
            continue

        if use_formulaicity_gap:
            legacy_loss_value = legacy_partition_loss(
                membership, matrix, model
            )
            cluster_1_mean = (
                _soft_leave_one_out_mean_self_information_and_gradient(
                    matrix, membership, formulaicity_smoothing
                )[0]
            )
            cluster_0_mean = (
                _soft_leave_one_out_mean_self_information_and_gradient(
                    matrix, 1.0 - membership, formulaicity_smoothing
                )[0]
            )
            soft_gap = abs(cluster_1_mean - cluster_0_mean)
            formulaicity_penalty = max(
                0.0,
                float(scipy_result.fun) - legacy_loss_value,
            )
        else:
            legacy_loss_value = float(scipy_result.fun)
            soft_gap = np.nan
            formulaicity_penalty = 0.0

        candidate = LegacyOptimizationResult(
            labels=labels,
            membership=membership,
            loss=float(scipy_result.fun),
            log_loss=float(np.log(scipy_result.fun)),
            n_iter=int(scipy_result.nit),
            success=bool(scipy_result.success),
            message=str(scipy_result.message),
            score_model=model,
            restart=restart,
            n_function_evaluations=int(scipy_result.nfev),
            legacy_loss=float(legacy_loss_value),
            formulaicity_penalty=float(formulaicity_penalty),
            soft_self_information_gap_bits=float(soft_gap),
        )
        if best_result is None or candidate.loss < best_result.loss:
            best_result = candidate

    if best_result is None:
        raise RuntimeError(
            "Every optimization restart collapsed below "
            f"min_component_fraction={min_component_fraction:.3f}; "
            f"cluster sizes were {collapsed_sizes}"
        )
    return best_result


def _leave_one_out_self_information(
    samples: np.ndarray,
    labels: np.ndarray,
    cluster: int,
    smoothing: float,
) -> np.ndarray:
    selected_indices = np.flatnonzero(labels == cluster)
    cluster_samples = samples[selected_indices]
    cluster_feature_counts = np.asarray(
        cluster_samples.sum(axis=0)
    ).ravel()
    cluster_total = float(cluster_feature_counts.sum())
    vocabulary_size = samples.shape[1]
    values = np.empty(len(selected_indices), dtype=float)

    for output_index in range(cluster_samples.shape[0]):
        if sparse.issparse(cluster_samples):
            row_start = cluster_samples.indptr[output_index]
            row_stop = cluster_samples.indptr[output_index + 1]
            active = cluster_samples.indices[row_start:row_stop]
            row_counts = cluster_samples.data[row_start:row_stop]
        else:
            row = cluster_samples[output_index]
            active = np.flatnonzero(row)
            row_counts = row[active]
        held_in_counts = (
            cluster_feature_counts[active] - row_counts + smoothing
        )
        held_in_total = (
            cluster_total - float(row_counts.sum())
            + smoothing * vocabulary_size
        )
        probabilities = held_in_counts / held_in_total
        values[output_index] = -float(
            np.sum(row_counts * np.log2(probabilities))
            / np.sum(row_counts)
        )
    return values


def self_information_distributions(
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    smoothing: float = 0.5,
) -> dict[int, np.ndarray]:
    """Return standard leave-one-out token self-information by cluster."""

    matrix = _as_feature_matrix(samples)
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (matrix.shape[0],):
        raise ValueError("labels and samples have different row counts")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain exactly the two clusters 0 and 1")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    return {
        cluster: _leave_one_out_self_information(
            matrix, labels, cluster, smoothing
        )
        for cluster in (0, 1)
    }


def _cluster_formulaicity_stats(
    samples: np.ndarray,
    labels: np.ndarray,
    cluster: int,
    self_information: np.ndarray,
) -> ClusterFormulaicityStats:
    cluster_samples = samples[labels == cluster]
    feature_counts = np.asarray(cluster_samples.sum(axis=0)).ravel()
    positive_counts = feature_counts[feature_counts > 0]
    total_tokens = float(positive_counts.sum())
    probabilities = positive_counts / total_tokens
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    vocabulary_size = len(positive_counts)
    normalized_entropy = (
        entropy / np.log2(vocabulary_size) if vocabulary_size > 1 else 0.0
    )

    row_totals = np.asarray(cluster_samples.sum(axis=1)).ravel()
    if sparse.issparse(cluster_samples):
        row_types = np.diff(cluster_samples.indptr)
    else:
        row_types = np.count_nonzero(cluster_samples, axis=1)
    repetition_rates = 1.0 - (row_types / row_totals)
    sorted_counts = np.sort(positive_counts)[::-1]

    def top_share(k: int) -> float:
        return float(sorted_counts[:k].sum() / total_tokens)

    return ClusterFormulaicityStats(
        cluster=cluster,
        n_samples=cluster_samples.shape[0],
        total_tokens=total_tokens,
        vocabulary_size=vocabulary_size,
        entropy_bits=entropy,
        normalized_entropy=float(normalized_entropy),
        mean_self_information_bits=float(np.mean(self_information)),
        std_self_information_bits=float(np.std(self_information)),
        mean_repetition_rate=float(np.mean(repetition_rates)),
        top_10_share=top_share(10),
        top_100_share=top_share(100),
    )


def determine_formulaic_cluster(
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    smoothing: float = 0.5,
    tie_tolerance_bits: float = 1e-9,
) -> FormulaicityResult:
    """Designate the more predictable fitted cluster as formulaic.

    The primary criterion is lower mean leave-one-out self-information using
    the standard ``-log2(p)`` definition.  Higher within-sample repetition is
    only a numerical tie-breaker.  This post-processing step does not feed
    back into optimization.
    """

    matrix = _as_feature_matrix(samples)
    labels = np.asarray(labels, dtype=int)
    distributions = self_information_distributions(
        matrix, labels, smoothing=smoothing
    )
    stats = tuple(
        _cluster_formulaicity_stats(
            matrix, labels, cluster, distributions[cluster]
        )
        for cluster in (0, 1)
    )
    self_information_difference = (
        stats[0].mean_self_information_bits
        - stats[1].mean_self_information_bits
    )

    if abs(self_information_difference) > tie_tolerance_bits:
        formulaic_cluster = (
            0 if self_information_difference < 0 else 1
        )
        criterion = "lower leave-one-out mean self-information"
    else:
        repetition_difference = (
            stats[0].mean_repetition_rate
            - stats[1].mean_repetition_rate
        )
        formulaic_cluster = 0 if repetition_difference >= 0 else 1
        criterion = "higher mean within-sample repetition (tie-breaker)"

    non_formulaic_cluster = 1 - formulaic_cluster
    formulaic_stats = stats[formulaic_cluster]
    other_stats = stats[non_formulaic_cluster]
    diagnostics_agree = (
        formulaic_stats.mean_repetition_rate
        >= other_stats.mean_repetition_rate
        and formulaic_stats.normalized_entropy
        <= other_stats.normalized_entropy
    )
    return FormulaicityResult(
        formulaic_cluster=formulaic_cluster,
        non_formulaic_cluster=non_formulaic_cluster,
        criterion=criterion,
        diagnostics_agree=diagnostics_agree,
        stats=stats,
    )
