#!/usr/bin/env python3
"""Audit the Figure 2 weighted covariance against Equation 19."""

from __future__ import annotations

import numpy as np

from figure2_engine import weighted_gaussian_parameters


def main() -> None:
    rng = np.random.default_rng(20260729)
    samples = rng.normal(size=(11, 4))
    weights = rng.uniform(0.1, 1.0, size=len(samples))
    epsilon = 1e-2

    observed_mean, observed_covariance, _ = weighted_gaussian_parameters(
        samples,
        weights,
        epsilon,
    )
    weight_sum = weights.sum()
    expected_mean = np.sum(weights[:, None] * samples, axis=0) / weight_sum
    centered = samples - expected_mean
    expected_covariance = (
        centered.T @ (weights[:, None] * centered) / weight_sum
        + epsilon * np.eye(samples.shape[1])
    )

    normalized = weights / weight_sum
    unbiased_covariance = (
        centered.T @ (normalized[:, None] * centered)
        / (1.0 - np.sum(normalized**2))
        + epsilon * np.eye(samples.shape[1])
    )

    mean_difference = float(
        np.max(np.abs(np.asarray(observed_mean) - expected_mean))
    )
    covariance_difference = float(
        np.max(
            np.abs(
                np.asarray(observed_covariance) - expected_covariance
            )
        )
    )
    if not np.allclose(observed_mean, expected_mean, rtol=1e-13, atol=1e-13):
        raise AssertionError(
            f"Equation 19 weighted-mean mismatch: {mean_difference:.3e}"
        )
    if not np.allclose(
        observed_covariance,
        expected_covariance,
        rtol=1e-13,
        atol=1e-13,
    ):
        raise AssertionError(
            "Equation 19 weighted-covariance mismatch: "
            f"{covariance_difference:.3e}"
        )
    if np.allclose(observed_covariance, unbiased_covariance):
        raise AssertionError(
            "Figure 2 covariance unexpectedly uses the unbiased correction"
        )

    scaled_mean, scaled_covariance, _ = weighted_gaussian_parameters(
        samples,
        7.0 * weights,
        epsilon,
    )
    if not np.allclose(scaled_mean, observed_mean):
        raise AssertionError("Weighted mean is not invariant to weight scale")
    if not np.allclose(scaled_covariance, observed_covariance):
        raise AssertionError(
            "Weighted covariance is not invariant to weight scale"
        )

    print("PASS Figure 2 Equation 19")
    print(f"mean maximum absolute difference: {mean_difference:.3e}")
    print(
        "covariance maximum absolute difference: "
        f"{covariance_difference:.3e}"
    )


if __name__ == "__main__":
    main()
