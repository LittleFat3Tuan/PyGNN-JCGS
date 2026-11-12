"""Reproducible MMD validation without changing model state or training RNGs."""

from contextlib import contextmanager

import torch

from .losses import LossFun


@contextmanager
def evaluating(model):
    """Temporarily disable training behavior and autograd, restoring every mode."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def AvgMMD(model, data, val_size, B, bandwidth=None, seed=42, *, prior=None):
    """Average B normalized squared MMDs against subsamples of ``data``.

    Subsampling is without replacement within each replicate. ``val_size`` is
    capped at len(data). A local CPU generator controls all pseudo-random draws
    for N, U, and t priors without advancing the training generator. For prior
    P, provide a matrix of explicit prior samples. This is a training-data
    diagnostic unless a separate held-out matrix is passed as ``data``.
    """
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or len(data) == 0:
        raise ValueError("data must be a nonempty matrix.")
    if not isinstance(val_size, int) or isinstance(val_size, bool) or val_size < 1:
        raise ValueError("val_size must be a positive integer.")
    if not isinstance(B, int) or isinstance(B, bool) or B < 1:
        raise ValueError("B must be a positive integer.")
    if model.inDist == "P":
        if prior is None or prior.ndim != 2 or prior.shape[1] != model.dims[0] or len(prior) < 1:
            raise ValueError("Validation for prior P requires explicit (n, input_dim) prior samples.")
    size = min(val_size, len(data), len(prior) if prior is not None else len(data))
    generator = torch.Generator().manual_seed(seed)
    criterion = LossFun("MMD", normalized=True)
    parameter = next(model.parameters())
    total = 0.0
    with evaluating(model):
        for _ in range(B):
            indices = torch.randperm(len(data), generator=generator)[:size]
            target = data[indices.to(data.device)].to(parameter.device, parameter.dtype)
            if model.inDist == "P":
                indices = torch.randperm(len(prior), generator=generator)[:size]
                generated = model(prior[indices.to(prior.device)])
            else:
                generated = model.sample(size, generator=generator)
            total += criterion(generated, target, bandwidth=bandwidth).item()
    return total / B
