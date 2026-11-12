"""Mini-batch fitting and coordination of adaptive bandwidth phases."""

from pathlib import Path

import torch
from tqdm.auto import tqdm

from .bandwidths import extend_bw, quantile_estimate, qvec_by_length
from .early_stopping import EarlyStopping
from .kernels import DEFAULT_BANDWIDTHS, rbf_kernel
from .persistence import write_json
from .sampling import draw_prior
from .validation import AvgMMD


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _batch_slices(size, batch_size, minimum):
    """Include every row, merging a too-small final batch into its predecessor."""
    starts = list(range(0, size, batch_size))
    if len(starts) > 1 and size - starts[-1] < minimum:
        starts.pop()
    return [slice(start, end) for start, end in zip(starts, starts[1:] + [size])]


class _BandwidthSchedule:
    """Consume an independent copy of the quantile-length schedule."""

    def __init__(self, data, lengths, size, repeats, extreme, extend, device):
        lengths = tuple(lengths)
        if not lengths:
            raise ValueError("bw_lengths must contain at least one length.")
        for length in lengths:
            _positive_int(length, "bw_lengths entries")
            if extend and length < 2:
                raise ValueError("ExtendBW requires bw_lengths entries of at least two.")
        self.lengths = iter(lengths)
        self.data, self.size, self.repeats = data, size, repeats
        self.extreme, self.extend, self.device = extreme, extend, device

    def next(self):
        length = next(self.lengths, None)
        if length is None:
            return None
        q = qvec_by_length(length // 2 if self.extend else length,
                           EXQ=False if self.extend else self.extreme)
        bandwidth = quantile_estimate(self.data, self.size, self.repeats, q, self.device)
        return extend_bw(bandwidth) if self.extend else bandwidth


def fit(model, data, batch_size=None, nEpochs=100, prior=None, bandwidth=None,
        weight=None, adaBW=True, ref_size=5000, ref_rep=100, val_size=3000,
        val_rep=50, val_bw=None, val_seed=42, delta_tr=0.0, delta_val=0.0,
        verbose=2, chkpt_path="checkpoint.pt", val_hist_dir=".", patience=3,
        snapshot_dir="./snapshots", take_snapshot=None, force_nonstop=False,
        bw_lengths=(6, 12, 24), adaptive_lr=False, EXQ=True, ExtendBW=False,
        log_path="train_log.json", collect_val_every_epoch=False):
    """Fit ``model`` in place and return a list of epoch-average scalar losses.

    Args:
        model: An FNN instance, with its own Adam optimizer and LossFun.
        data: Floating (N, output_dim) training tensor. NaNs are supported by
            fixed-bandwidth MMD/tMMD only. Infinite values are never accepted.
        batch_size: Rows per gradient step; None uses all N rows. A short final
            batch is included or merged to satisfy batch-norm/tMMD constraints.
        nEpochs: Maximum number of complete passes through the training sample.
        prior: Explicit (N, input_dim) samples, required only for input_dist=P.
            Rows remain paired with targets for MSE/BCE; distribution losses
            shuffle target and prior samples independently.
        bandwidth: Positive fixed kernel widths when adaBW=False.
        weight: Fixed kernel weights, supported only when adaBW=False.
        adaBW: Estimate bandwidths and update them on training plateaus.
        ref_size: Subsample size for bandwidth estimates, capped at N.
        ref_rep: Number of repeated distance-quantile estimates to average.
        val_size: Validation subsample size, capped at N.
        val_rep: Number of normalized MMD replicates per validation evaluation.
        val_bw: Fixed validation bandwidths, independent of training widths.
        val_seed: Local seed reused for comparable validation evaluations.
        delta_tr: Required relative improvement of smoothed training loss.
        delta_val: Required relative improvement within a validation window.
        verbose: 0=silent, 1=progress bar, 2=bar plus ten-epoch reports,
            3=one report per epoch.
        chkpt_path: State-dict path for smoothed training improvements, or None.
        val_hist_dir: Directory for validation-window CSVs, or None.
        patience: Initial training plateau patience; updated after each phase.
        snapshot_dir: Directory for state-dict snapshots named E<epoch>.pt.
        take_snapshot: Iterable of one-based epochs at which to save weights.
        force_nonstop: Continue with fixed final widths after a stop request.
        bw_lengths: Quantile counts for successive phases; never mutated.
        adaptive_lr: Divide learning rate by five on the first two updates.
        EXQ: Append 0.99, 0.999, and 1 quantiles in ordinary adaptive mode.
        ExtendBW: Use half-length quantiles and extend_bw instead of EXQ.
        log_path: JSON training log path, or None to disable file output.
        collect_val_every_epoch: Also evaluate validation outside monitoring
            windows, without changing the stopping rule.

    Training logs and final widths are also stored as ``model.training_log_``
    and ``model.bandwidth_``. Log events use zero-based epochs; snapshot names
    are one-based. Final weights are retained; checkpoints are not restored.
    """
    _positive_int(nEpochs, "nEpochs")
    if not isinstance(adaBW, bool):
        raise TypeError("adaBW must be a bool.")
    if verbose not in (0, 1, 2, 3):
        raise ValueError("verbose must be 0, 1, 2, or 3.")
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or len(data) < 1:
        raise ValueError("data must be a nonempty (N, output_dim) tensor.")
    if data.shape[1] != model.output_dim or not data.is_floating_point():
        raise ValueError("data must be floating and match model.output_dim.")
    if torch.isinf(data).any():
        raise ValueError("data must not contain infinite values.")
    kernel_loss = model.lossFun.name in ("MMD", "tMMD")
    if (adaBW or not kernel_loss) and not torch.isfinite(data).all():
        raise ValueError("Missing data require fixed-bandwidth MMD/tMMD (adaBW=False).")
    if adaBW and (not kernel_loss or bandwidth is not None or weight is not None):
        raise ValueError("Adaptive MMD/tMMD estimates its own widths; leave bandwidth and weight unset.")
    if adaptive_lr and not adaBW:
        raise ValueError("adaptive_lr requires adaBW=True.")
    if adaBW or collect_val_every_epoch:
        _positive_int(val_size, "val_size")
        _positive_int(val_rep, "val_rep")
    size = len(data)
    batch_size = size if batch_size is None else batch_size
    _positive_int(batch_size, "batch_size")
    batch_size = min(batch_size, size)
    minimum = 4 if model.lossFun.name == "tMMD" else (2 if model.batchNorm else 1)
    if batch_size < minimum:
        raise ValueError(f"This model/loss needs batches with at least {minimum} observations.")
    if model.inDist == "P":
        if not isinstance(prior, torch.Tensor) or prior.shape != (size, model.dims[0]):
            raise ValueError("Prior P requires an (N, input_dim) prior tensor.")
        if not torch.isfinite(prior).all():
            raise ValueError("prior must be finite.")
    elif prior is not None:
        raise ValueError("Explicit prior samples require input_dist='P'.")
    parameter = next(model.parameters())
    data = data.detach().to(parameter.device, parameter.dtype)
    if prior is not None:
        prior = prior.detach().to(parameter.device, parameter.dtype)
    snapshots = set(() if take_snapshot is None else take_snapshot)
    for epoch in snapshots:
        _positive_int(epoch, "snapshot epochs")
    if snapshots and snapshot_dir is None:
        raise ValueError("snapshot_dir is required when take_snapshot is nonempty.")

    log = {"update": [], "stop": [], "bandwidths": []}
    if collect_val_every_epoch:
        log["val_losses"] = []
    stopper = schedule = None
    active = adaBW
    lr_updates = 0
    if adaBW:
        # Adjust distance scaling and extreme quantiles for larger samples.
        if size > 6000:
            model.lossFun.normalized = True
        schedule = _BandwidthSchedule(data, bw_lengths, ref_size, ref_rep,
                                      EXQ and size <= 8000, ExtendBW, model.device)
        stopper = EarlyStopping(patience, delta_tr, delta_val, sleep=0,
                                checkpoint_path=chkpt_path, val_hist_dir=val_hist_dir)
        current_bw = schedule.next()
    else:
        current_bw = torch.as_tensor(DEFAULT_BANDWIDTHS if bandwidth is None else bandwidth,
                                     device=parameter.device, dtype=parameter.dtype)
    if kernel_loss:
        # Validate kernel settings before any optimizer step or file output.
        rbf_kernel(data[:1], data[:1], current_bw, model.lossFun.normalized, weight)
    if adaBW or collect_val_every_epoch:
        rbf_kernel(data[:1], data[:1], val_bw, normalized=True)
    log["bandwidths"].append({"epoch": 0, "values": current_bw.tolist()})
    batches = _batch_slices(size, batch_size, minimum)
    losses = []
    epochs = tqdm(range(nEpochs), disable=verbose not in (1, 2))
    model.train()
    for epoch in epochs:
        inputs = prior if prior is not None else draw_prior(
            model.inDist, model.dims[0], size, device=model.device)
        x_indices = torch.randperm(size, device=parameter.device)
        y_indices = (x_indices if model.lossFun.name in ("MSE", "BCE")
                     else torch.randperm(size, device=parameter.device))
        total_loss = 0.0
        for batch in batches:
            x, y = inputs[x_indices[batch]], data[y_indices[batch]]
            total_loss += model.step(x, y, bandwidth=current_bw, weight=weight).item() * len(y)
        losses.append(total_loss / size)
        stop_reason = None
        if active and stopper(losses[-1], model) and not stopper.val_monitoring:
            next_bw = schedule.next()
            if next_bw is None:
                stop_reason = "bandwidth_schedule_exhausted"
            else:
                current_bw = next_bw
                stopper.update_patience(epoch)
                stopper.set_sleep(5)
                stopper.start_val_monitoring()
                log["update"].append(epoch)
                log["bandwidths"].append({"epoch": epoch, "values": current_bw.tolist()})
                if adaptive_lr and lr_updates < 2:
                    for group in model.optimizer.param_groups:
                        group["lr"] /= 5
                    lr_updates += 1
                if verbose:
                    tqdm.write(f"Bandwidth phase updated at epoch {epoch + 1}.")
        if collect_val_every_epoch or (active and stopper.val_monitoring):
            validation = AvgMMD(model, data, val_size, val_rep, val_bw, val_seed, prior=prior)
            if collect_val_every_epoch:
                log["val_losses"].append(validation)
            if active and stopper.val_monitoring:
                stopper.update_val_loss(validation)
                if stopper.stop_training():
                    stop_reason = "validation_plateau"
        if epoch + 1 in snapshots:
            path = Path(snapshot_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path / f"E{epoch + 1}.pt")
        if verbose == 3 or (verbose == 2 and (epoch + 1) % 10 == 0):
            tqdm.write(f"Epoch {epoch + 1}: loss={losses[-1]:.6g}")
        if stop_reason is not None:
            log["stop"].append(epoch)
            log["stop_reason"] = stop_reason
            active = False
            if verbose:
                tqdm.write(f"Stop requested at epoch {epoch + 1}: {stop_reason}.")
            if not force_nonstop:
                break
    log["epochs_completed"] = len(losses)
    log["forced_continuation"] = bool(force_nonstop and log["stop"])
    log.setdefault("stop_reason", "max_epochs")
    model.training_log_ = log
    model.bandwidth_ = current_bw.detach().cpu().clone()
    if adaBW or collect_val_every_epoch:
        write_json(log_path, log)
    return losses
