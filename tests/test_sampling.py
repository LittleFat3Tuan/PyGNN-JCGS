"""Check prior distributions, rank transforms, and independent seeded validation."""

import unittest

import numpy as np
import torch

from PyGNN import AvgMMD, FNN, pobs
from PyGNN.bandwidths import extend_bw, quantile_estimate, qvec_by_length
from PyGNN.sampling import drawQStdNormal, drawQStdT4, drawStdT4


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_priors_are_seeded_without_changing_global_rng(self):
        for distribution in ("N", "U", "t"):
            model = FNN([3, 4, 2], ["ReLU", "Sigmoid"], input_dist=distribution).eval()
            for quasi in (False, True):
                rng = torch.random.get_rng_state().clone()
                first = model.sample(16, QRS=quasi, seed=13)
                second = model.sample(16, QRS=quasi, seed=13)
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                torch.testing.assert_close(torch.random.get_rng_state(), rng)
                self.assertTrue(torch.isfinite(first).all())

    def test_t_prior_has_shared_scale_and_correct_marginal_variance(self):
        draws = drawStdT4(2, 120000, generator=torch.Generator().manual_seed(5))
        np.testing.assert_allclose(draws.var(0).numpy(), [2, 2], rtol=0.1)
        self.assertGreater(np.corrcoef(draws.abs().numpy().T)[0, 1], 0.15)
        for sampler in (drawQStdNormal, drawQStdT4):
            sample = sampler(3, 1024, seed=42)
            self.assertEqual(sample.shape, (1024, 3))
            self.assertTrue(torch.isfinite(sample).all())

    def test_validation_preserves_buffers_modes_and_random_state(self):
        for distribution in ("N", "U", "t"):
            model = FNN([2, 8, 2], ["ReLU", "Sigmoid"], batch_norm=True,
                        drop_out=0.2, input_dist=distribution)
            model.model.BN1.eval()
            modes = [module.training for module in model.modules()]
            state = {key: value.clone() for key, value in model.state_dict().items()}
            data = torch.rand(12, 2)
            rng = torch.random.get_rng_state().clone()
            first = AvgMMD(model, data, 30, 2, seed=1)
            second = AvgMMD(model, data, 30, 2, seed=1)
            self.assertEqual(first, second)
            self.assertEqual(modes, [module.training for module in model.modules()])
            torch.testing.assert_close(torch.random.get_rng_state(), rng)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_rank_ties_and_missing_values(self):
        np.testing.assert_allclose(pobs([3, 1, 1]), [0.75, 0.375, 0.375])
        np.testing.assert_allclose(pobs([[1, 4], [2, 2]]), [[1/3, 2/3], [2/3, 1/3]])
        np.testing.assert_allclose(pobs([2, np.nan, 1]), [2/3, np.nan, 1/3], equal_nan=True)

    def test_custom_prior_validation_and_mode_restoration_on_error(self):
        model = FNN([1, 4, 2], ["ReLU", "Sigmoid"], input_dist="P")
        prior = torch.rand(10, 1)
        score = AvgMMD(model, torch.rand(12, 2), 8, 2, prior=prior)
        self.assertTrue(np.isfinite(score))
        with self.assertRaises(ValueError):
            AvgMMD(model, torch.rand(12, 3), 8, 2, prior=prior)
        self.assertTrue(model.training)

    def test_distance_quantiles_and_zero_width_protection(self):
        data = torch.tensor([[0.0], [1.0], [3.0]])
        estimate = quantile_estimate(data, 100, 2, [0, 0.5, 1])
        torch.testing.assert_close(estimate, torch.tensor([1, 2, 3]) / np.sqrt(2))
        self.assertEqual(len(qvec_by_length(6, EXQ=True)), 9)
        self.assertEqual(len(extend_bw(torch.tensor([0.2, 0.4, 0.6]))), 6)
        self.assertTrue((quantile_estimate(torch.zeros(5, 2), 10, 1, [0, 0.5]) > 0).all())


if __name__ == "__main__":
    unittest.main()
