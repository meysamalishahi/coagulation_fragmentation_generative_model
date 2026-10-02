#!/usr/bin/env python3
"""Pinned JetNet 0.2.5 FPD/KPD routines used by the reference evaluator.

These functions are the dependency-light subset of JetNet's MIT-licensed
``evaluation/gen_metrics.py`` needed by the ViT-CFM calorimeter evaluation.
Keeping the exact formulas locally avoids importing unrelated JetNet dataset
dependencies during a multi-day unattended run.
"""

from __future__ import annotations

import warnings

from numba import njit
import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit
from scipy.stats import iqr


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError("feature statistics have incompatible shapes")
    difference = mu1 - mu2
    covariance_mean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covariance_mean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    return (
        difference.dot(difference)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covariance_mean)
    )


def _normalise_features(first, second=None):
    maximum = np.max(np.abs(first), axis=0)
    maximum[maximum == 0] = 1
    if second is None:
        return first / maximum
    return first / maximum, second / maximum


def _linear(values, intercept, slope):
    return intercept + slope * values


def fpd(
    real_features,
    generated_features,
    min_samples=20_000,
    max_samples=50_000,
    num_batches=20,
    num_points=10,
    normalise=True,
    seed=42,
):
    """JetNet 0.2.5 Fréchet physics distance and extrapolation error."""
    if len(real_features) < 50_000 or len(generated_features) < 50_000:
        warnings.warn("Recommended number of samples for FPD is 50,000", RuntimeWarning)
    first = np.asarray(real_features)
    second = np.asarray(generated_features)
    if normalise:
        first, second = _normalise_features(first, second)
    batch_sizes = (
        1.0 / np.linspace(1.0 / min_samples, 1.0 / max_samples, num_points)
    ).astype("int32")
    generator = np.random.default_rng(seed)
    values = []
    for batch_size in batch_sizes:
        batch_values = []
        for _ in range(num_batches):
            sample_first = first[generator.choice(len(first), size=batch_size)]
            sample_second = second[generator.choice(len(second), size=batch_size)]
            batch_values.append(
                _calculate_frechet_distance(
                    np.mean(sample_first, axis=0),
                    np.cov(sample_first, rowvar=False),
                    np.mean(sample_second, axis=0),
                    np.cov(sample_second, rowvar=False),
                )
            )
        values.append(np.mean(batch_values))
    parameters, covariance = curve_fit(
        _linear,
        1.0 / batch_sizes,
        values,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
    )
    return float(parameters[0]), float(np.sqrt(np.diag(covariance)[0]))


@njit
def _poly_kernel_pairwise(first, second, degree):
    gamma = 1.0 / first.shape[-1]
    return (first @ second.T * gamma + 1.0) ** degree


@njit
def _mmd_quadratic_unbiased(first_first, second_second, first_second):
    first_count, second_count = first_first.shape[0], second_second.shape[0]
    return (
        (first_first.sum() - np.trace(first_first)) / (first_count * (first_count - 1))
        + (second_second.sum() - np.trace(second_second))
        / (second_count * (second_count - 1))
        - 2.0 * np.mean(first_second)
    )


@njit
def _mmd_poly_quadratic_unbiased(first, second, degree=4):
    return _mmd_quadratic_unbiased(
        _poly_kernel_pairwise(first, first, degree),
        _poly_kernel_pairwise(second, second, degree),
        _poly_kernel_pairwise(first, second, degree),
    )


def kpd(
    real_features,
    generated_features,
    num_batches=10,
    batch_size=5_000,
    normalise=True,
    seed=42,
):
    """JetNet 0.2.5 kernel physics distance and interquantile error."""
    first = np.asarray(real_features)
    second = np.asarray(generated_features)
    if normalise:
        first, second = _normalise_features(first, second)
    values = []
    for index in range(num_batches):
        generator = np.random.default_rng(seed + index * 1000)
        sample_first = first[generator.choice(len(first), size=batch_size)]
        sample_second = second[generator.choice(len(second), size=batch_size)]
        values.append(_mmd_poly_quadratic_unbiased(sample_first, sample_second))
    return float(np.median(values)), float(iqr(values, rng=(16.275, 83.725)) / 2.0)
