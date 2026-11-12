"""Pseudo-random and scrambled Sobol' draws for the network's input priors."""

from inspect import signature

import numpy as np
import torch
from scipy.stats import chi2, norm, qmc


def _check_shape(d: int, size: int) -> None:
    if not isinstance(d, int) or isinstance(d, bool) or d < 1:
        raise ValueError("d must be a positive integer.")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer.")


def _random_device(device, generator):
    # Drawing on the generator's device also supports CPU generators with MPS.
    return device if generator is None else generator.device


def drawStdNormal(d, size, device="cpu", generator=None):
    """Return a float32 ``(size, d)`` sample with independent N(0, 1) entries.

    If supplied, ``generator`` controls every draw. Samples are transferred to
    ``device`` after drawing when the generator resides on another device.
    """
    _check_shape(d, size)
    return torch.randn(
        size, d, device=_random_device(device, generator), generator=generator,
        dtype=torch.float32,
    ).to(device)


def drawStdUniform(d, size, device="cpu", generator=None):
    """Return independent U[0, 1) inputs, optionally using a local generator."""
    _check_shape(d, size)
    return torch.rand(
        size, d, device=_random_device(device, generator), generator=generator,
        dtype=torch.float32,
    ).to(device)


def drawStdT4(d, size, device="cpu", generator=None):
    """Draw a multivariate t distribution with 4 degrees of freedom.

    Its location is zero and scale matrix is the identity (covariance is 2I).
    Each row shares one chi-square mixing variable across all coordinates;
    the coordinates therefore are dependent even with an identity scale.
    """
    _check_shape(d, size)
    draws = torch.randn(
        size, d + 4, device=_random_device(device, generator),
        generator=generator, dtype=torch.float32,
    )
    chi_square = draws[:, d:].square().sum(dim=1, keepdim=True)
    chi_square = chi_square.clamp_min(torch.finfo(draws.dtype).tiny)
    return (draws[:, :d] * torch.sqrt(4 / chi_square)).to(device)


def _sobol(d, size, seed):
    """Draw one scrambled Sobol' point set, protecting inverse-CDF endpoints."""
    _check_shape(d, size)
    rng_key = "rng" if "rng" in signature(qmc.Sobol).parameters else "seed"
    engine = qmc.Sobol(d=d, scramble=True, **{rng_key: np.random.default_rng(seed)})
    points = engine.random(n=size)
    eps = np.finfo(np.float64).eps
    return np.clip(points, eps, 1 - eps)


def drawQStdNormal(d, size, seed=None, device="cpu"):
    """Transform a scrambled Sobol' sample into standard normal inputs.

    Use a power-of-two ``size`` to retain Sobol' balance properties. ``seed``
    controls the scramble; a fresh scramble is drawn when it is None.
    """
    return torch.as_tensor(norm.ppf(_sobol(d, size, seed)), dtype=torch.float32,
                           device=device)


def drawQStdUniform(d, size, seed=None, device="cpu"):
    """Return scrambled Sobol' uniforms as a float32 tensor on ``device``."""
    samples = torch.as_tensor(_sobol(d, size, seed), dtype=torch.float32, device=device)
    # Float64 Sobol' coordinates very close to one can round to one in float32.
    return samples.clamp_max(1 - torch.finfo(samples.dtype).eps)


def drawQStdT4(d, size, seed=None, device="cpu"):
    """Transform a (d+1)-dimensional Sobol' set into multivariate t4 inputs.

    The first d coordinates generate normals and the last generates a shared
    chi-square(4) scale. The ``seed`` argument reproduces the scramble.
    """
    _check_shape(d, size)
    points = _sobol(d + 1, size, seed)
    values = norm.ppf(points[:, :d]) * np.sqrt(4 / chi2.ppf(points[:, d:], df=4))
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def draw_prior(distribution, d, size, *, device="cpu", generator=None):
    """Dispatch a built-in prior identified by ``N``, ``U``, or ``t``."""
    samplers = {"N": drawStdNormal, "U": drawStdUniform, "t": drawStdT4}
    if distribution not in samplers:
        raise ValueError("Use forward() with explicit inputs for a custom prior (P).")
    return samplers[distribution](d, size, device=device, generator=generator)


def draw_quasi_prior(distribution, d, size, *, device="cpu", seed=None):
    """Dispatch a scrambled Sobol' prior identified by ``N``, ``U``, or ``t``."""
    samplers = {"N": drawQStdNormal, "U": drawQStdUniform, "t": drawQStdT4}
    if distribution not in samplers:
        raise ValueError("Use forward() with explicit inputs for a custom prior (P).")
    return samplers[distribution](d, size, device=device, seed=seed)
