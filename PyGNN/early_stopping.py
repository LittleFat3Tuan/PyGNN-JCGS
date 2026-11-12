"""Training-plateau detection and post-update validation monitoring."""

import math
from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    """Track a bias-corrected exponential average and validation improvements.

    A training plateau requests a bandwidth update. After an update, the first
    validation loss becomes a baseline; any later relative improvement ends
    monitoring successfully. Otherwise a window of ``patience`` observations
    sets ``stop_flag``. Checkpoints contain model state dicts only.

    Args:
        patience: Consecutive non-improving training epochs before a request.
        delta_tr: Required relative decrease in smoothed training loss.
        delta_val: Required relative decrease from the validation baseline.
        sleep: Initial epochs to record without checking training plateaus.
        checkpoint_path: State-dict filename, or None to disable checkpoints.
        val_hist_dir: Directory for validation-window CSVs, or None to disable.
        alpha: Exponential smoothing weight in (0, 1].
    """

    def __init__(self, patience=5, delta_tr=0.0, delta_val=0.0, sleep=50,
                 checkpoint_path="checkpoint.pth", val_hist_dir=".", alpha=0.25):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not 0 <= delta_tr < 1 or not 0 <= delta_val < 1:
            raise ValueError("Relative improvement thresholds must be in [0, 1).")
        self.set_patience(patience)
        self.set_sleep(sleep)
        self.delta_tr, self.delta_val, self.alpha = delta_tr, delta_val, alpha
        self.checkpoint_path, self.val_hist_dir = checkpoint_path, val_hist_dir
        self.best_loss = math.inf
        self.wait = 0
        self.loss_history = []
        self._weighted_sum = self._weighted_count = 0.0
        self._processed = 0
        self.stop_flag = self.val_monitoring = False
        self.val_epochs_remaining = 0
        self.val_best_loss = math.inf
        self.val_history = []

    def smooth_loss(self):
        """Return the bias-corrected exponential average of recorded losses."""
        if not self.loss_history:
            raise ValueError("No training losses have been recorded.")
        for loss in self.loss_history[self._processed:]:
            self._weighted_sum = loss + (1 - self.alpha) * self._weighted_sum
            self._weighted_count = 1 + (1 - self.alpha) * self._weighted_count
        self._processed = len(self.loss_history)
        return self._weighted_sum / self._weighted_count

    def set_patience(self, new_patience):
        """Set a strictly positive integer patience window."""
        if not isinstance(new_patience, int) or isinstance(new_patience, bool) or new_patience < 1:
            raise ValueError("patience must be a positive integer.")
        self.patience = new_patience

    def update_patience(self, nepoch):
        """Set patience using the epoch-dependent exponential schedule."""
        self.set_patience(self.calculate_patience(nepoch))

    def calculate_patience(self, nepoch):
        """Return floor(exp(1 + min(3, nepoch / 33))) for a zero-based epoch."""
        return int(math.floor(math.exp(1 + min(3, nepoch / 33))))

    def set_sleep(self, new_sleep):
        """Defer training-plateau checks for this many upcoming epochs."""
        if not isinstance(new_sleep, int) or isinstance(new_sleep, bool) or new_sleep < 0:
            raise ValueError("sleep must be a nonnegative integer.")
        self.sleep = new_sleep

    def save_checkpoint(self, model):
        """Save current model weights, creating parent directories if needed."""
        if self.checkpoint_path is not None:
            path = Path(self.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)

    def stop_training(self):
        """Return whether validation monitoring has requested termination."""
        return self.stop_flag

    def start_val_monitoring(self):
        """Begin a fresh validation window after a bandwidth update."""
        if not self.stop_flag:
            self.val_monitoring = True
            self.val_epochs_remaining = self.patience
            self.val_best_loss = math.inf
            self.val_history = []

    def save_val_history(self):
        """Write this monitoring window under val_hist_dir, if enabled."""
        if self.val_hist_dir is not None:
            directory = Path(self.val_hist_dir)
            directory.mkdir(parents=True, exist_ok=True)
            end = len(self.loss_history)
            start = end - len(self.val_history)
            np.savetxt(directory / f"val_losses_S{start}E{end}.csv", self.val_history)

    def update_val_loss(self, val_loss):
        """Record a finite validation loss and update the stopping decision."""
        if not math.isfinite(val_loss):
            raise FloatingPointError("Validation loss is nonfinite.")
        if not self.val_monitoring:
            return
        self.val_history.append(float(val_loss))
        self.val_epochs_remaining -= 1
        if val_loss < self.val_best_loss * (1 - self.delta_val):
            if math.isfinite(self.val_best_loss):
                self.val_monitoring = False
                self.save_val_history()
                return
            self.val_best_loss = val_loss
        if self.val_epochs_remaining <= 0:
            self.val_monitoring = False
            self.stop_flag = True
            self.save_val_history()

    def __call__(self, current_loss, model):
        """Record one epoch; return True when a plateau or stop needs handling."""
        if not math.isfinite(current_loss):
            raise FloatingPointError("Training loss is nonfinite.")
        self.loss_history.append(float(current_loss))
        if self.stop_flag:
            return True
        if self.sleep:
            self.sleep -= 1
            return False
        smoothed = self.smooth_loss()
        if smoothed < self.best_loss * (1 - self.delta_tr):
            self.best_loss, self.wait = smoothed, 0
            self.save_checkpoint(model)
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.best_loss, self.wait = math.inf, 0
            return True
        return False
