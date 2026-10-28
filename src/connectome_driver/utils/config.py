"""YAML configuration with command-line overrides (``key.subkey=value``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Load a YAML config and apply dot-list overrides, e.g. ``edges.min_syn_count=10``."""
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def compose_config(
    config: str | Path, model_files: Sequence[str | Path] = (), overrides: Sequence[str] = ()
) -> DictConfig:
    """The configuration of a training run: base config, then each model file merged into
    ``model`` in order, then dot-list overrides (what ``train.py --model ... key=value`` does)."""
    cfg = load_config(config)
    for path in model_files:
        cfg.model = OmegaConf.merge(cfg.model, OmegaConf.load(path))
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
