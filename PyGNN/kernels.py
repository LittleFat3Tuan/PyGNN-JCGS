"""Gaussian mixture kernels with pairwise missing-coordinate handling."""

import torch

DEFAULT_BANDWIDTHS = tuple(10 ** x for x in (-1.5, -1, -0.5, -0.25, -0.125, -0.0625))


def sample_pair(x, y):
    """Validate floating samples, interpreting vectors as one-dimensional data."""
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Samples must be torch tensors.")
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Samples must have shapes (n, d) and (m, d).")
    if min(*x.shape, *y.shape) < 1:
        raise ValueError("Samples must be nonempty.")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("Samples must use a floating-point dtype.")
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError("Samples must share a device and dtype.")
    if torch.isinf(x).any() or torch.isinf(y).any():
        raise ValueError("Samples must not contain infinite values.")
    return x, y


def squared_distances(x, y, normalized=False):
    """Return pairwise squared distances, omitting pairwise missing coordinates.

    ``normalized=True`` divides by the number of jointly observed coordinates.
    Pairs with no observed overlap have distance zero. Matrix products avoid
    allocating an (n, m, d) difference tensor. Small negative values caused by
    floating-point roundoff are clipped.
    """
    x, y = sample_pair(x, y)
    mx, my = (~torch.isnan(x)).to(x.dtype), (~torch.isnan(y)).to(y.dtype)
    # A common origin reduces cancellation when observations have large offsets.
    origin = torch.nan_to_num(x.detach()).mean(dim=0)
    xx, yy = torch.nan_to_num(x - origin), torch.nan_to_num(y - origin)
    distances = (xx.square() @ my.T + mx @ yy.square().T - 2 * (xx @ yy.T))
    distances = distances.clamp_min(0)
    if normalized:
        distances = distances / (mx @ my.T).clamp_min(1)
    return distances


def rbf_kernel(x, y, bandwidth=None, normalized=False, weight=None):
    """Return the weighted Gaussian mixture kernel matrix.

    Component k is exp(-distance_squared / (2 * bandwidth[k]**2)). Without
    ``weight``, components are averaged. Explicit nonnegative weights are used
    as supplied, without renormalization.
    """
    distances = squared_distances(x, y, normalized)
    bandwidth = torch.as_tensor(
        DEFAULT_BANDWIDTHS if bandwidth is None else bandwidth,
        device=x.device, dtype=x.dtype,
    )
    if bandwidth.ndim != 1 or bandwidth.numel() == 0:
        raise ValueError("bandwidth must be a nonempty vector.")
    if not torch.isfinite(bandwidth).all() or (bandwidth <= 0).any():
        raise ValueError("All bandwidths must be finite and strictly positive.")
    if weight is None:
        weights = torch.full_like(bandwidth, 1 / bandwidth.numel())
    else:
        weights = torch.as_tensor(weight, device=x.device, dtype=x.dtype)
        if weights.shape != bandwidth.shape:
            raise ValueError("weight and bandwidth must have the same shape.")
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Weights must be finite, nonnegative, and have positive sum.")
    result = torch.zeros_like(distances)
    for bw, w in zip(bandwidth, weights):
        result = result + w * torch.exp(-distances / (2 * bw.square()))
    return result
