"""Quantile schedules and bandwidth estimates for adaptive GMMN training."""

import math

import torch


def qvec_by_length(length, EXQ=False):
    """Return 0.95 * 2**(-linspace(0, 9, length)), in descending order.

    ``EXQ=True`` appends 0.99, 0.999, and 1, giving length+3 quantiles.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer.")
    quantiles = 0.95 * 2 ** -torch.linspace(0, 9, length)
    if EXQ:
        quantiles = torch.cat((quantiles, torch.tensor([0.99, 0.999, 1.])))
    return quantiles


def quantile_estimate(data, size, num_rep, q, device="cpu"):
    """Average pairwise-distance quantiles, divided by sqrt(2 * dimension).

    Each repetition takes a fresh subsample without replacement. ``size`` is
    capped at the data size. Data must be a finite floating matrix; for missing
    observations, use fixed bandwidths instead. Quantiles are computed on CPU
    because some PyTorch devices do not implement torch.quantile. Returned
    bandwidths reside on ``device`` and retain the data dtype.
    """
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or min(data.shape) < 1:
        raise ValueError("data must be a nonempty (n, d) tensor.")
    if not data.is_floating_point() or not torch.isfinite(data).all():
        raise ValueError("Bandwidth estimation requires finite floating data; use adaBW=False for missing data.")
    if not isinstance(size, int) or isinstance(size, bool) or size < 2 or len(data) < 2:
        raise ValueError("Bandwidth estimation requires at least two observations.")
    if not isinstance(num_rep, int) or isinstance(num_rep, bool) or num_rep < 1:
        raise ValueError("num_rep must be a positive integer.")
    reference = data.detach().cpu()
    quantiles = torch.as_tensor(q, dtype=reference.dtype, device="cpu")
    if quantiles.ndim != 1 or quantiles.numel() == 0:
        raise ValueError("q must be a nonempty quantile vector.")
    if not torch.isfinite(quantiles).all() or ((quantiles < 0) | (quantiles > 1)).any():
        raise ValueError("q must contain finite values in [0, 1].")
    size = min(size, len(reference))
    total = torch.zeros_like(quantiles)
    for _ in range(num_rep):
        indices = torch.randperm(len(reference))[:size]
        total += torch.quantile(torch.pdist(reference[indices]), quantiles)
    bandwidth = total / (num_rep * math.sqrt(2 * data.shape[1]))
    # Coincident observations can give zero quantiles, which are invalid widths.
    return bandwidth.clamp_min(torch.finfo(bandwidth.dtype).eps).to(device)


def extend_bw(bw, lend=0.01, rend=0.99):
    """Prepend/append a total of len(bw) linearly spaced bandwidths.

    The middle vector and its order are preserved. This optional heuristic
    extends the bandwidth coverage without sorting the vector.
    """
    bw = torch.as_tensor(bw)
    if bw.ndim != 1 or bw.numel() == 0 or not bw.is_floating_point():
        raise ValueError("bw must be a nonempty floating vector.")
    if not torch.isfinite(bw).all() or (bw <= 0).any():
        raise ValueError("bw must contain positive finite values.")
    if not 0 < lend < rend or not math.isfinite(rend):
        raise ValueError("Endpoints must satisfy 0 < lend < rend < infinity.")
    left = len(bw) // 2
    right = len(bw) - left
    return torch.cat((
        torch.linspace(lend, bw[0].item(), left + 1, device=bw.device, dtype=bw.dtype)[:-1],
        bw,
        torch.linspace(bw[-1].item(), rend, right + 1, device=bw.device, dtype=bw.dtype)[1:],
    ))
