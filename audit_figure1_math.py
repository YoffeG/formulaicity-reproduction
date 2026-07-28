#!/usr/bin/env python3
"""Audit the Figure 1 implementation against manuscript Equations 7--10.

This is an independent, intentionally direct implementation of the Bernoulli
probability estimator and two-cluster negative weighted log-likelihood.  It
checks both the vectorized loss and its analytic gradient.  No archived plot
data are read.
"""

from __future__ import annotations

import numpy as np

from formulaicity_optimization import legacy_partition_loss_and_gradient


EPSILON = 1e-10


def direct_cluster_log_likelihood(
    samples: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return the direct Eq. 7--9 weighted log-likelihood and probabilities."""

    probabilities = (weights[:, None] * samples).sum(axis=0) / weights.sum()
    row_log_likelihoods = (
        samples * np.log2(probabilities + EPSILON)
        + (1.0 - samples) * np.log2(1.0 - probabilities + EPSILON)
    ).sum(axis=1)
    return float(weights @ row_log_likelihoods), probabilities


def direct_partition_loss(
    membership: np.ndarray,
    samples: np.ndarray,
) -> float:
    """Return the direct Eq. 10 negative weighted log-likelihood."""

    cluster_1, _ = direct_cluster_log_likelihood(samples, membership)
    cluster_0, _ = direct_cluster_log_likelihood(
        samples, 1.0 - membership
    )
    return -(cluster_1 + cluster_0)


def finite_difference_gradient(
    membership: np.ndarray,
    samples: np.ndarray,
    step: float = 1e-6,
) -> np.ndarray:
    """Evaluate the direct objective gradient with central differences."""

    gradient = np.empty_like(membership)
    for index in range(len(membership)):
        upper = membership.copy()
        lower = membership.copy()
        upper[index] += step
        lower[index] -= step
        gradient[index] = (
            direct_partition_loss(upper, samples)
            - direct_partition_loss(lower, samples)
        ) / (2.0 * step)
    return gradient


def main() -> None:
    rng = np.random.default_rng(20260728)
    samples = rng.binomial(
        1,
        rng.uniform(0.08, 0.72, size=17),
        size=(12, 17),
    ).astype(float)
    membership = rng.uniform(0.15, 0.85, size=len(samples))

    direct_loss = direct_partition_loss(membership, samples)
    vectorized_loss, analytic_gradient = (
        legacy_partition_loss_and_gradient(
            membership,
            samples,
            score_model="binary",
        )
    )
    numerical_gradient = finite_difference_gradient(membership, samples)
    loss_difference = abs(direct_loss - vectorized_loss)
    gradient_difference = float(
        np.max(np.abs(numerical_gradient - analytic_gradient))
    )

    _, probabilities = direct_cluster_log_likelihood(samples, membership)
    probability_sum = float(probabilities.sum())
    if np.isclose(probability_sum, 1.0):
        raise AssertionError(
            "Independent Bernoulli probabilities were incorrectly normalized "
            "across features"
        )
    if not np.isclose(
        direct_loss,
        vectorized_loss,
        rtol=1e-12,
        atol=1e-10,
    ):
        raise AssertionError(
            f"Equation 10 loss mismatch: absolute difference {loss_difference}"
        )
    if not np.allclose(
        numerical_gradient,
        analytic_gradient,
        rtol=1e-6,
        atol=2e-6,
    ):
        raise AssertionError(
            "Analytic gradient mismatch: maximum absolute difference "
            f"{gradient_difference}"
        )

    print("PASS Figure 1 Equations 7--10")
    print(f"loss absolute difference: {loss_difference:.3e}")
    print(f"gradient maximum absolute difference: {gradient_difference:.3e}")
    print(
        "sum of independent Bernoulli feature probabilities: "
        f"{probability_sum:.6f} (correctly not constrained to one)"
    )


if __name__ == "__main__":
    main()
