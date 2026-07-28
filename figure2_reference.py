"""Local reference implementation of the Figure 2 data and comparators.

Only the small, stable API consumed by ``run_figure_2_drive_code.py`` is
included here.  This removes the run-time dependency on the Google Drive copy
of ``entropy_optim_rewritten.py`` while preserving its Gaussian generator:
one class has diagonal variance 3, the other variance 10, and ``noise``
replaces the requested fraction of low-variance coordinates by variance 10.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import matthews_corrcoef
from sklearn.mixture import GaussianMixture


def norm_mcc(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Map label-invariant MCC to the 50--100 scale used in the paper."""

    return 50.0 + 50.0 * abs(
        float(matthews_corrcoef(true_labels, cluster_labels))
    )


def generate_data(
    dim: int,
    n_samples1: int,
    n_samples2: int,
    noise: float = 0.0,
    plot: bool = False,
):
    """Generate the two Gaussian classes used by the Figure 2 source code."""

    del plot  # Plotting the individual draws is outside Figure 2 reproduction.
    # These apparently redundant draws and decompositions are retained to
    # match the random-number consumption and algebra of the audited source.
    diagonal_1 = np.diag(np.random.uniform(3.0, 3.0, size=dim))
    diagonal_2 = np.diag(np.random.uniform(10.0, 10.0, size=dim))
    gaussian_matrix = np.random.randn(dim, dim)
    basis_1, _, _ = np.linalg.svd(gaussian_matrix)
    basis_2, _, _ = np.linalg.svd(gaussian_matrix)
    covariance_1 = basis_1 @ diagonal_1 @ basis_1.T
    covariance_2 = basis_2 @ diagonal_2 @ basis_2.T
    replaced = int(dim * noise)
    covariance_1[:, :replaced] = covariance_2[:, :replaced]
    covariance_1[:replaced, :] = covariance_2[:replaced, :]

    samples1 = np.random.multivariate_normal(
        np.random.uniform(0.0, 0.0, size=dim),
        covariance_1,
        n_samples1,
    )
    samples2 = np.random.multivariate_normal(
        np.random.uniform(0.0, 0.0, size=dim),
        covariance_2,
        n_samples2,
    )
    samples = np.vstack((samples1, samples2))
    labels = np.concatenate(
        (
            np.ones(n_samples1, dtype=int),
            np.zeros(n_samples2, dtype=int),
        )
    )
    return samples, labels, covariance_1, covariance_2


def gmm_clustering(samples: np.ndarray, true_labels: np.ndarray):
    """Fit the full-covariance two-component Gaussian-mixture comparator."""

    del true_labels
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
    )
    labels = model.fit_predict(samples)
    return labels, model.covariances_


def differential_entropic_clustering(
    samples: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """Retain the optional differential-entropy comparator from the source."""

    indices = np.random.choice(len(samples), k, replace=False)
    means = samples[indices]
    base_covariance = np.cov(samples.T) + np.eye(samples.shape[1]) * 1e-3
    covariances = [base_covariance.copy() for _ in range(k)]
    previous_means = means.copy()
    responsibilities = np.zeros((len(samples), k))

    for _ in range(max_iter):
        for cluster in range(k):
            distribution = multivariate_normal(
                mean=means[cluster],
                cov=covariances[cluster],
                allow_singular=True,
            )
            responsibilities[:, cluster] = distribution.pdf(samples)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        new_means = []
        new_covariances = []
        for cluster in range(k):
            response = responsibilities[:, cluster]
            effective_weight = response.sum()
            mean = (
                response[:, None] * samples
            ).sum(axis=0) / effective_weight
            centered = samples - mean
            covariance = (
                centered.T @ (response[:, None] * centered)
            ) / effective_weight
            covariance += np.eye(samples.shape[1]) * 1e-3
            new_means.append(mean)
            new_covariances.append(covariance)

        means = np.asarray(new_means)
        covariances = new_covariances
        if np.linalg.norm(means - previous_means) < tol:
            break
        previous_means = means.copy()

    return np.argmax(responsibilities, axis=1), covariances
