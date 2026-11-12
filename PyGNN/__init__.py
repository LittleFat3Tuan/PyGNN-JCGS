"""Generative moment matching networks for learning multivariate distributions."""

from .models import FNN, NormalQuantile
from .losses import LossFun
from .persistence import load_model, save_model
from .preprocessing import pobs
from .training import fit
from .validation import AvgMMD

__version__ = "0.1.0"
__all__ = ["FNN", "NormalQuantile", "LossFun", "fit", "AvgMMD", "pobs",
           "save_model", "load_model"]
