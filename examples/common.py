"""Shared command-line, training, evaluation, and output handling for tutorials."""

import argparse
from pathlib import Path

import numpy as np
import torch

from PyGNN import FNN, LossFun, fit, save_model
from PyGNN.persistence import write_json
from PyGNN.validation import evaluating

from .data import clayton_copula, gaussian_copula


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser(dataset, adaptive):
    """Create the common example CLI, with small portable defaults."""
    parser = argparse.ArgumentParser(description=(
        f"Train {'an adaptive' if adaptive else 'a fixed-bandwidth'} GMMN on {dataset} data."
    ))
    parser.add_argument("--ntrain", type=_positive, default=512, help="training rows")
    parser.add_argument("--ntest", type=_positive, default=256, help="test and generated rows")
    parser.add_argument("--dim", type=_positive, default=2, help="copula dimension, at least 2")
    parser.add_argument("--latent-dim", type=_positive, default=2, help="prior dimension")
    parser.add_argument("--hidden", type=_positive, default=64, help="hidden layer width")
    parser.add_argument("--epochs", type=_positive, default=100 if adaptive else 40)
    parser.add_argument("--batch-size", type=_positive, default=128)
    parser.add_argument("--prior", choices=("N", "U", "t"), default="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--threads", type=_positive, default=1, help="PyTorch CPU threads")
    parser.add_argument("--output", type=Path, default=Path("outputs") / (
        "adaptive" if adaptive else dataset))
    parser.add_argument("--plot", action="store_true", help="save diagnostics.png (needs matplotlib)")
    if dataset == "gaussian":
        parser.add_argument("--absolute", action="store_true", help="rank absolute Gaussian data")
    if adaptive:
        parser.add_argument("--ref-size", type=_positive, default=128)
        parser.add_argument("--ref-rep", type=_positive, default=3)
        parser.add_argument("--val-size", type=_positive, default=128)
        parser.add_argument("--val-rep", type=_positive, default=3)
        parser.add_argument("--patience", type=_positive, default=3)
    return parser


def plot_diagnostics(training, generated, losses, path):
    """Save a loss curve and a comparison of the first two copula coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].plot(np.arange(1, len(losses) + 1), losses)
    axes[0].set(xlabel="Epoch", ylabel="Squared MMD", title="Training loss")
    for axis, values, title in zip(axes[1:], (training, generated),
                                   ("Training sample", "Generated sample")):
        axis.scatter(values[:, 0], values[:, 1], s=5, alpha=0.5)
        axis.set(xlabel="U1", ylabel="U2", title=title, xlim=(0, 1), ylim=(0, 1))
        axis.set_aspect("equal")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run(dataset, adaptive=False, argv=None):
    """Run a complete tutorial and save independently reusable model artifacts."""
    parser = build_parser(dataset, adaptive)
    args = parser.parse_args(argv)
    if min(args.ntrain, args.ntest, args.dim) < 2:
        parser.error("ntrain, ntest, and dim must each be at least 2")
    if adaptive and args.ref_size < 2:
        parser.error("ref-size must be at least 2")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    sampler = gaussian_copula if dataset == "gaussian" else clayton_copula
    data_options = {"absolute": args.absolute} if dataset == "gaussian" else {}
    training = sampler(args.ntrain, args.dim, args.seed, **data_options)
    test = sampler(args.ntest, args.dim, args.seed + 1, **data_options)
    target = torch.tensor(training, dtype=torch.float32)
    model = FNN([args.latent_dim, args.hidden, args.dim], ["ReLU", "Sigmoid"],
                input_dist=args.prior, MMD_normalized=True, device=args.device)
    criterion = LossFun("MMD", normalized=True)
    test_tensor = torch.tensor(test, device=args.device, dtype=torch.float32)
    with evaluating(model):
        initial = criterion(model.sample(args.ntest, seed=args.seed + 2), test_tensor).item()
    args.output.mkdir(parents=True, exist_ok=True)
    options = {}
    if adaptive:
        options = dict(
            ref_size=args.ref_size, ref_rep=args.ref_rep, val_size=args.val_size,
            val_rep=args.val_rep, patience=args.patience, delta_tr=0.01,
            delta_val=0.001, bw_lengths=(3, 6, 12), EXQ=False,
            collect_val_every_epoch=True,
            chkpt_path=args.output / "best_training_state.pt",
            val_hist_dir=args.output / "validation", log_path=args.output / "training_log.json",
        )
    losses = fit(model, target, batch_size=args.batch_size, nEpochs=args.epochs,
                 adaBW=adaptive, verbose=0, **options)
    model.eval()
    with torch.no_grad():
        generated = model.sample(args.ntest, seed=args.seed + 2)
        quasi = model.sample(args.ntest, QRS=True, seed=args.seed + 3)
        test_mmd = criterion(generated, test_tensor).item()
    generated, quasi = generated.cpu().numpy(), quasi.cpu().numpy()
    save_model(model, args.output / "model.pt")
    np.savetxt(args.output / "training.csv", training, delimiter=",")
    np.savetxt(args.output / "generated_prs.csv", generated, delimiter=",")
    np.savetxt(args.output / "generated_qrs.csv", quasi, delimiter=",")
    np.savetxt(args.output / "history.csv", losses, delimiter=",")
    metrics = {
        "dataset": dataset, "adaptive": adaptive, "seed": args.seed,
        "ntrain": args.ntrain, "dim": args.dim, "epochs": len(losses),
        "untrained_test_mmd_squared": initial, "test_mmd_squared": test_mmd,
        "stop_reason": model.training_log_["stop_reason"],
    }
    write_json(args.output / "metrics.json", metrics)
    if args.plot:
        plot_diagnostics(training, generated, losses, args.output / "diagnostics.png")
    print(f"Completed {len(losses)} epochs. Held-out squared MMD: {initial:.6f} -> {test_mmd:.6f}")
    print(f"Saved model, samples, losses, and metrics to {args.output.resolve()}")
    return metrics
