"""Distribution and pointwise losses used by the feed-forward generator."""

import torch
import torch.nn.functional as functional

from .kernels import rbf_kernel, sample_pair


class LossFun:
    """Select ``MMD``, ``tMMD``, ``CvM``, ``MSE``, or ``BCE``.

    MMD returns the biased empirical *squared* discrepancy, including diagonal
    kernel terms. ``normalized`` controls distance normalization for MMD/tMMD.
    """

    NAMES = ("MMD", "tMMD", "CvM", "BCE", "MSE")

    def __init__(self, loss_fun, normalized=False):
        if loss_fun not in self.NAMES:
            raise ValueError(f"loss_fun must be one of {self.NAMES}.")
        self.name = loss_fun
        self.normalized = bool(normalized)

    def _kernel_matrices(self, x, y, bandwidth, weight):
        return tuple(
            rbf_kernel(a, b, bandwidth, self.normalized, weight)
            for a, b in ((x, x), (y, y), (x, y))
        )

    def MMD(self, x, y, bandwidth=None, weight=None):
        """Compute mean Kxx + mean Kyy - 2 mean Kxy, allowing unequal sample sizes."""
        kxx, kyy, kxy = self._kernel_matrices(x, y, bandwidth, weight)
        return kxx.mean() + kyy.mean() - 2 * kxy.mean()

    def tMMD(self, x, y, bandwidth=None, weight=None):
        """Compute a variance-standardized MMD statistic.

        This option requires equal sample sizes of at least four. Its
        numerator uses zero within-sample diagonals divided by n**2.
        Nonpositive variance estimates raise an error rather than producing a
        NaN. Ordinary MMD is recommended for training.
        """
        x, y = sample_pair(x, y)
        m = len(x)
        if m != len(y) or m < 4:
            raise ValueError("tMMD requires equal sample sizes of at least four.")
        kxx, kyy, kxy = self._kernel_matrices(x, y, bandwidth, weight)
        mask = 1 - torch.eye(m, device=x.device, dtype=x.dtype)
        xx, yy = kxx * mask, kyy * mask
        sx, sy = xx.sum(1), yy.sum(1)
        c0, c1 = kxy.sum(0), kxy.sum(1)
        tx, ty, tc = xx.sum(), yy.sum(), kxy.sum()
        vx, vy, vc = xx.square().sum(), yy.square().sum(), kxy.square().sum()
        # Estimate variance from kernel row sums and squared kernel entries.
        variance = (
            2 / (m * (m - 1) * (m - 2) * (m - 3))
            * (2 * sx.dot(sx) - vx + 2 * sy.dot(sy) - vy)
            - (4 * m - 6) / (m**2 * (m - 1)**2 * (m - 2) * (m - 3))
            * (tx**2 + ty**2)
            + 4 * (m**2 - m - 1) / (m**3 * (m - 1)**2)
            * (c1.dot(c1) + c0.dot(c0))
            - 4 * (m - 2) / (m**2 * (m - 1)**3) * vc
            - (8 * m - 12) / (m**3 * (m - 1)**3) * tc**2
            + 8 / (m**2 * (m - 1) * (m - 2))
            * ((tx + ty) * tc / m - sx.dot(c1) - sy.dot(c0))
        )
        if not torch.isfinite(variance) or variance <= 0:
            raise FloatingPointError("tMMD variance is nonpositive or nonfinite; use MMD.")
        return (xx.mean() + yy.mean() - 2 * kxy.mean()) / variance.sqrt()

    def CvM_similarity(self, x, y):
        """Sum prod_j(1 - max(x_j, y_j)) over all pairs of copula observations."""
        x, y = sample_pair(x, y)
        for sample in (x, y):
            if not torch.isfinite(sample).all() or ((sample < 0) | (sample > 1)).any():
                raise ValueError("CvM requires finite copula samples in [0, 1].")
        return (1 - torch.maximum(x[:, None, :], y[None, :, :])).prod(dim=2).sum()

    def CvM(self, x, y):
        """Return the two-sample Cramer-von Mises statistic on the unit cube."""
        x, y = sample_pair(x, y)
        n, m = len(x), len(y)
        return (
            self.CvM_similarity(x, x) / n**2
            - 2 * self.CvM_similarity(x, y) / (n * m)
            + self.CvM_similarity(y, y) / m**2
        ) / (1 / n + 1 / m)

    def _pointwise_pair(self, x, y):
        x, y = sample_pair(x, y)
        if x.shape != y.shape:
            raise ValueError("Pointwise losses require matching sample shapes.")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Pointwise losses require finite samples.")
        return x, y

    def MSE(self, x, y):
        """Return mean squared error between paired observations."""
        return functional.mse_loss(*self._pointwise_pair(x, y))

    def BCE(self, x, y):
        """Return binary cross entropy between paired probabilities in [0, 1]."""
        return functional.binary_cross_entropy(*self._pointwise_pair(x, y))

    def forward(self, x, y, bandwidth=None, weight=None):
        """Dispatch the selected loss; kernel options apply only to MMD/tMMD."""
        method = getattr(self, self.name)
        if self.name in ("MMD", "tMMD"):
            return method(x, y, bandwidth=bandwidth, weight=weight)
        return method(x, y)

    __call__ = forward
