"""Small synthetic copula datasets with no external files or copula packages."""

import numpy as np

from PyGNN import pobs


def gaussian_copula(size, dim=2, seed=42, correlation=0.7, absolute=False):
    """Rank-transform equicorrelated Gaussian (or absolute Gaussian) samples."""
    if size < 2 or dim < 2:
        raise ValueError("Use at least two rows and two dimensions.")
    if not 0 <= correlation < 1:
        raise ValueError("correlation must be in [0, 1).")
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((size, 1))
    independent = rng.standard_normal((size, dim))
    values = np.sqrt(correlation) * common + np.sqrt(1 - correlation) * independent
    return pobs(np.abs(values) if absolute else values)


def clayton_copula(size, dim=2, seed=42, theta=2.0):
    """Draw a positive-parameter Clayton copula using its gamma frailty mixture."""
    if size < 2 or dim < 2:
        raise ValueError("Use at least two rows and two dimensions.")
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be positive and finite.")
    rng = np.random.default_rng(seed)
    frailty = rng.gamma(shape=1 / theta, scale=1.0, size=(size, 1))
    exponentials = rng.exponential(size=(size, dim))
    return (1 + exponentials / frailty) ** (-1 / theta)
