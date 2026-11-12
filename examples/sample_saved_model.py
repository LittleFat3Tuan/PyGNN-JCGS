"""Load a tutorial model and save a new pseudo-random or quasi-random sample."""

import argparse
from pathlib import Path

import numpy as np
import torch

from PyGNN import load_model, pobs


def main():
    """Read model metadata automatically and generate a CSV sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="model.pt created by an example")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quasi", action="store_true")
    parser.add_argument("--rank", action="store_true", help="convert generated margins to ranks/(n+1)")
    parser.add_argument("--output", type=Path, default=Path("outputs/new_samples.csv"))
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size must be positive")
    model = load_model(args.model, device=args.device)
    with torch.no_grad():
        samples = model.sample(args.size, QRS=args.quasi, seed=args.seed).cpu().numpy()
    if args.rank:
        samples = pobs(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, samples, delimiter=",")
    print(f"Saved {samples.shape[0]} observations to {args.output.resolve()}")


if __name__ == "__main__":
    main()
