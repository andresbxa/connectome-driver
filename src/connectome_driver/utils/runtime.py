"""Explicit, recorded runtime optimizations; scientific defaults remain unchanged."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from omegaconf import DictConfig


def configure_runtime(cfg: DictConfig) -> dict:
    runtime = cfg.get("runtime", {})
    cache_name = runtime.get("tunableop_cache")
    backend = cfg.model.core.get("sparse_backend", "torch")
    if not cache_name and backend == "torch":
        return {}
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("The requested ROCm optimizations require a ROCm GPU")
    info = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(), "sparse_backend": backend,
        "tunableop_enabled": False,
    }
    if cache_name:
        cache = Path(cache_name).resolve()
        if not cache.is_file():
            raise FileNotFoundError(f"TunableOp cache not found: {cache}")
        # Replay only. A long run must not trigger unbounded tuning on unseen shapes.
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.set_filename(str(cache), insert_device_ordinal=False)
        if not torch.cuda.tunable.read_file(str(cache)):
            raise RuntimeError("TunableOp rejected the cache; regenerate it for this environment")
        torch.cuda.tunable.enable(True)
        # Persist the resolved machine-local path, not an environment interpolation.
        cfg.runtime.tunableop_cache = str(cache)
        info.update(tunableop_enabled=True, tunableop_cache=str(cache),
                    tunableop_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                    tunableop_tuning=False)
    return info
