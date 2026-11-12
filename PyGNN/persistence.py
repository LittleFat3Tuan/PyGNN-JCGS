"""Portable inference artifacts and structured training logs."""

import json
from pathlib import Path

import torch

from .models import FNN


def write_json(path, content):
    """Write JSON with parent-directory creation; None disables output."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_model(model, path):
    """Save constructor settings, dtype, and weights for inference.

    Adam state and training-controller state are not included. Use separate
    optimizer/state management when exact training resumption is required.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "configuration": model.configuration(),
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "state_dict": model.state_dict(),
    }, path)


def load_model(path, device="cpu"):
    """Load a save_model artifact onto ``device`` and return an evaluation model."""
    artifact = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("Expected a PyGNN save_model artifact with format_version=1.")
    configuration = dict(artifact["configuration"])
    configuration["device"] = device
    dtypes = {name: getattr(torch, name) for name in ("float16", "float32", "float64", "bfloat16")}
    if artifact["dtype"] not in dtypes:
        raise ValueError("Unsupported model dtype in artifact.")
    model = FNN(**configuration).to(dtype=dtypes[artifact["dtype"]])
    model.load_state_dict(artifact["state_dict"])
    return model.eval()
