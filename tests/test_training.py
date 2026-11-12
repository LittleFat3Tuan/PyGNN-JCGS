"""Exercise fitting, adaptive transitions, the public API, and persistence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from PyGNN import FNN, fit, load_model, save_model
from PyGNN.early_stopping import EarlyStopping


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_fixed_training_and_default_or_oversized_batch(self):
        data = torch.rand(17, 2)
        for batch in (None, 100):
            model = FNN([2, 8, 2], ["ReLU", "Sigmoid"])
            before = model.model.Dense1.weight.detach().clone()
            history = fit(model, data, batch_size=batch, nEpochs=3, adaBW=False, verbose=0)
            self.assertEqual(len(history), 3)
            self.assertFalse(torch.equal(before, model.model.Dense1.weight))
            self.assertTrue(all(torch.isfinite(torch.tensor(history))))

    def test_remainder_rows_and_batch_norm_singleton_are_retained(self):
        for size, batch_norm in ((10, False), (9, True)):
            data = torch.arange(2 * size).reshape(size, 2).float() / (2 * size)
            model = FNN([2, 8, 2], ["ReLU", "Sigmoid"], batch_norm=batch_norm)
            observed = []
            optimizer_step = model.step

            def recording_step(x, y, *args, **kwargs):
                observed.extend(y[:, 0].tolist())
                return optimizer_step(x, y, *args, **kwargs)

            model.step = recording_step
            fit(model, data, batch_size=4, nEpochs=1, adaBW=False, verbose=0)
            self.assertEqual(sorted(observed), sorted(data[:, 0].tolist()))

    def test_paired_custom_prior_learns_pointwise_target(self):
        prior = torch.rand(32, 1)
        model = FNN([1, 1], [None], input_dist="P", loss_fun="MSE", lrate=0.05)
        losses = fit(model, 0.6 * prior + 0.2, prior=prior, nEpochs=60,
                     batch_size=8, adaBW=False, verbose=0)
        self.assertLess(losses[-1], losses[0] / 100)

    def test_dropout_is_applied_to_hidden_stages_only(self):
        model = FNN([2, 8, 2], ["ReLU", "Sigmoid"], drop_out=0.5)
        samples = model.sample(64)
        self.assertTrue(((samples > 0) & (samples < 1)).all())
        self.assertEqual(samples.shape, (64, 2))
        self.assertEqual(model.output_dim, 2)

    def test_adaptive_stop_snapshots_and_force_continuation(self):
        data = torch.rand(16, 2)
        for force in (False, True):
            model = FNN([2, 5, 2], ["ReLU", "Sigmoid"])
            schedule = [2, 3, 4]
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with patch.object(model, "step", return_value=torch.tensor(1.0)), \
                     patch("PyGNN.training.AvgMMD", return_value=0.5):
                    losses = fit(
                        model, data, nEpochs=8, verbose=0, patience=1,
                        ref_size=8, ref_rep=1, val_size=8, val_rep=1,
                        bw_lengths=schedule, EXQ=False, adaptive_lr=True,
                        collect_val_every_epoch=True, force_nonstop=force,
                        chkpt_path=directory / "nested/checkpoint.pt",
                        log_path=directory / "nested/log.json",
                        val_hist_dir=directory / "validation",
                        take_snapshot=[3], snapshot_dir=directory / "snapshots",
                    )
                self.assertEqual(len(losses), 8 if force else 3)
                self.assertEqual(schedule, [2, 3, 4])
                self.assertEqual(model.training_log_["update"], [1])
                self.assertEqual(model.training_log_["stop"], [2])
                self.assertEqual(model.training_log_["stop_reason"], "validation_plateau")
                self.assertEqual(len(model.training_log_["val_losses"]), len(losses))
                self.assertAlmostEqual(model.optimizer.param_groups[0]["lr"], 0.0002)
                self.assertTrue((directory / "snapshots/E3.pt").is_file())
                log = json.loads((directory / "nested/log.json").read_text())
                self.assertEqual(log["forced_continuation"], force)

    def test_exhausted_schedule_stops_cleanly_and_defaults_can_be_reused(self):
        model = FNN([2, 4, 2], ["ReLU", "Sigmoid"])
        data = torch.rand(8, 2)
        with patch.object(model, "step", return_value=torch.tensor(1.0)):
            losses = fit(model, data, nEpochs=8, patience=1, bw_lengths=[2],
                         ref_size=8, ref_rep=1, verbose=0, chkpt_path=None,
                         val_hist_dir=None, log_path=None)
        self.assertEqual(len(losses), 2)
        self.assertEqual(model.training_log_["stop_reason"], "bandwidth_schedule_exhausted")
        for _ in range(2):
            fit(model, data, nEpochs=1, ref_size=8, ref_rep=1, verbose=0,
                chkpt_path=None, val_hist_dir=None, log_path=None)
            self.assertEqual(len(model.bandwidth_), 9)

    def test_early_stopping_smoothing_and_validation_baseline(self):
        model = FNN([1, 1], [None])
        stopper = EarlyStopping(patience=3, sleep=0, checkpoint_path=None,
                                val_hist_dir=None, alpha=0.25, delta_val=0.1)
        for value in (4, 2, 1):
            stopper(value, model)
        self.assertAlmostEqual(stopper.smooth_loss(), (4 * 0.75**2 + 2 * 0.75 + 1)
                               / (0.75**2 + 0.75 + 1))
        stopper.start_val_monitoring()
        stopper.update_val_loss(1.0)
        stopper.update_val_loss(0.95)
        stopper.update_val_loss(0.8)
        self.assertFalse(stopper.val_monitoring)
        self.assertFalse(stopper.stop_training())

    def test_model_artifact_and_state_dict_round_trip(self):
        model = FNN([2, 7, 2], ["ReLU", "Sigmoid"], batch_norm=True).double().eval()
        inputs = torch.rand(10, 2, dtype=torch.float64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested/model.pt"
            save_model(model, path)
            restored = load_model(path)
            self.assertFalse(restored.training)
            torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
            weights_model = FNN([2, 7, 2], ["ReLU", "Sigmoid"], batch_norm=True).double().eval()
            weights_model.load_state_dict(model.state_dict())
            torch.testing.assert_close(weights_model(inputs), model(inputs), rtol=0, atol=0)

    def test_public_api_and_input_validation(self):
        from PyGNN.models import FNN as model_class
        from PyGNN.training import fit as training_function
        from PyGNN.losses import LossFun
        from PyGNN.preprocessing import pobs
        import PyGNN
        self.assertIs(model_class, PyGNN.FNN)
        self.assertIs(training_function, PyGNN.fit)
        self.assertIs(LossFun, PyGNN.LossFun)
        self.assertIs(pobs, PyGNN.pobs)
        model = FNN([2, 4, 2], ["ReLU", "Sigmoid"])
        with self.assertRaises(ValueError):
            fit(model, torch.rand(3, 3), nEpochs=1, adaBW=False, verbose=0)
        with self.assertRaises(ValueError):
            fit(model, torch.rand(3, 2), batch_size=0, nEpochs=1, adaBW=False, verbose=0)
        with self.assertRaises(ValueError):
            FNN([2, 0, 2], ["ReLU", "Sigmoid"])
        with self.assertRaises(ValueError):
            FNN([2, 4, None], ["ReLU", "Sigmoid"])
        with self.assertRaises(ValueError):
            FNN([2, 4, 2], ["ReLU"])


if __name__ == "__main__":
    unittest.main()
