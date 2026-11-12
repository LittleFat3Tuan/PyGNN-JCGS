"""Rank transformations for copula training data."""

import numpy as np
from scipy.stats import rankdata


def pobs(data):
    """Return columnwise average ranks divided by the observed count plus one.

    Accept a nonempty vector or matrix and return a NumPy float64 array of the
    same shape. Ties receive average ranks. NaNs remain missing and are omitted
    from the rank denominators. Observed outputs lie strictly between 0 and 1.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim not in (1, 2) or values.size == 0:
        raise ValueError("data must be a nonempty vector or matrix.")
    if np.isinf(values).any():
        raise ValueError("data must not contain infinite values.")
    counts = np.sum(~np.isnan(values), axis=0)
    return rankdata(values, method="average", axis=0, nan_policy="omit") / (counts + 1)
