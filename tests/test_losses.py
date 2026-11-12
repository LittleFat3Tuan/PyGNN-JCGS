"""Check distribution losses against independent scalar calculations."""

import unittest

import numpy as np
import torch

from PyGNN import LossFun
from PyGNN.kernels import rbf_kernel


class LossTests(unittest.TestCase):
    def test_mmd_and_cvm_match_scalar_definitions(self):
        x = np.array([[0.1, 0.3], [0.4, 0.8], [0.7, 0.2]])
        y = np.array([[0.2, 0.5], [0.6, 0.9]])
        bandwidth = [0.1, 0.3, 0.7]

        def kernel_average(a, b):
            return np.mean([
                np.mean([np.exp(-np.mean((u - v)**2) / (2 * h**2)) for h in bandwidth])
                for u in a for v in b
            ])

        def cvm_similarity(a, b):
            return sum(np.prod(1 - np.maximum(u, v)) for u in a for v in b)

        expected_mmd = kernel_average(x, x) + kernel_average(y, y) - 2 * kernel_average(x, y)
        n, m = len(x), len(y)
        expected_cvm = (cvm_similarity(x, x) / n**2 - 2 * cvm_similarity(x, y) / (n * m)
                        + cvm_similarity(y, y) / m**2) / (1 / n + 1 / m)
        loss = LossFun("MMD", normalized=True)
        x, y = torch.tensor(x), torch.tensor(y)
        self.assertAlmostEqual(loss.MMD(x, y, bandwidth=bandwidth).item(), expected_mmd, places=12)
        self.assertAlmostEqual(loss.CvM(x, y).item(), expected_cvm, places=12)

    def test_missing_data_kernel_matches_scalar_definition(self):
        x = np.array([[0.2, np.nan, 0.8], [0.5, 0.7, np.nan], [np.nan] * 3])
        y = np.array([[0.1, 0.3, 0.9], [np.nan, 0.2, 0.3]])
        bw, weights = [0.15, 0.5], [0.3, 0.7]
        for normalized in (False, True):
            expected = np.empty((len(x), len(y)))
            for i, row_x in enumerate(x):
                for j, row_y in enumerate(y):
                    observed = ~(np.isnan(row_x) | np.isnan(row_y))
                    distance = ((row_x[observed] - row_y[observed])**2).sum()
                    if normalized and observed.sum():
                        distance /= observed.sum()
                    expected[i, j] = sum(w * np.exp(-distance / (2 * h**2))
                                         for h, w in zip(bw, weights))
            actual = rbf_kernel(torch.tensor(x), torch.tensor(y), bw, normalized, weights)
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)

    def test_mmd_gradient_and_unequal_sizes(self):
        torch.manual_seed(8)
        x = torch.rand(5, 2, dtype=torch.float64, requires_grad=True)
        y = torch.rand(7, 2, dtype=torch.float64)
        loss = LossFun("MMD", normalized=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z: loss(z, y, bandwidth=[0.3, 0.7]), (x,)))
        self.assertAlmostEqual(loss(x, x).item(), 0.0, places=12)
        value = loss(x, y)
        value.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_missing_data_gradient_is_finite(self):
        x = torch.tensor([[0.1, 0.2], [0.3, 0.8]], requires_grad=True)
        y = torch.tensor([[0.2, float("nan")], [float("nan"), 0.9]])
        LossFun("MMD", True)(x, y).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_tmmd_backward_and_degenerate_variance(self):
        generator = torch.Generator().manual_seed(17)
        x = torch.rand(12, 3, generator=generator, requires_grad=True)
        y = torch.rand(12, 3, generator=generator) * 0.6
        LossFun("tMMD", True)(x, y, bandwidth=[0.1, 0.3, 0.7]).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):
            LossFun("tMMD")(x[:3], y[:3])
        with self.assertRaises(FloatingPointError):
            separated = torch.arange(4).float()[:, None]
            LossFun("tMMD")(separated, separated + 10, bandwidth=[1e-6])

    def test_kernel_is_stable_under_large_common_translation(self):
        x = torch.tensor([[0., 0.], [8., 16.]])
        y = torch.tensor([[8., 0.], [16., 8.]])
        expected = rbf_kernel(x, y, [10.])
        shifted = rbf_kernel(x + 1e6, y + 1e6, [10.])
        torch.testing.assert_close(shifted, expected)

    def test_pointwise_losses_and_kernel_errors(self):
        x, y = torch.tensor([[0.2], [0.8]]), torch.tensor([[0.0], [1.0]])
        self.assertAlmostEqual(LossFun("MSE")(x, y).item(), 0.04, places=6)
        self.assertAlmostEqual(LossFun("BCE")(x, y).item(), -np.log(0.8), places=6)
        for bandwidth in ([0], [-1], [float("nan")], []):
            with self.assertRaises(ValueError):
                rbf_kernel(x, y, bandwidth)
        with self.assertRaises(ValueError):
            rbf_kernel(x, y, [0.1, 0.2], weight=[1])


if __name__ == "__main__":
    unittest.main()
