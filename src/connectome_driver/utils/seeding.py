"""Seeding of every random number generator a run touches."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch. Environments are seeded through ``reset(seed=...)``.

    ``deterministic`` also forces deterministic cuDNN kernels; it is slower and still does not
    make sparse CUDA kernels bitwise reproducible, so it is off by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
