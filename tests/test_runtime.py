from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from connectome_driver.utils.config import compose_config
from connectome_driver.utils.plan import load_plan
from connectome_driver.utils.runtime import configure_runtime


def test_default_runtime_never_touches_gpu(monkeypatch):
    available = Mock(side_effect=AssertionError("default must not initialize CUDA"))
    monkeypatch.setattr(torch.cuda, "is_available", available)
    assert configure_runtime(compose_config("configs/train.yaml")) == {}
    available.assert_not_called()


def test_requested_runtime_rejects_cpu(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    cfg = compose_config("configs/train.yaml", overrides=["runtime.tunableop_cache=missing.csv"])
    with pytest.raises(RuntimeError, match="require a ROCm GPU"):
        configure_runtime(cfg)


def test_cache_replay_is_validated_and_tuning_disabled(tmp_path, monkeypatch):
    cache = tmp_path / "tuning.csv"
    cache.write_text("cache contents")
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU")
    for name in ("tuning_enable", "set_filename", "read_file", "enable"):
        monkeypatch.setattr(torch.cuda.tunable, name, Mock(return_value=True))
    cfg = compose_config("configs/train.yaml", overrides=[f"runtime.tunableop_cache={cache.as_posix()}"])
    info = configure_runtime(cfg)
    torch.cuda.tunable.tuning_enable.assert_called_once_with(False)
    torch.cuda.tunable.read_file.assert_called_once_with(str(cache.resolve()))
    assert len(info["tunableop_cache_sha256"]) == 64
    torch.cuda.tunable.read_file.return_value = False
    with pytest.raises(RuntimeError, match="rejected"):
        configure_runtime(cfg)


def test_rocm_plan_preserves_scientific_configuration(monkeypatch):
    monkeypatch.setenv("CONNECTOME_TUNABLEOP_CACHE", "/local/tuning.csv")
    base = load_plan("configs/experiments/whole_brain.yaml")
    optimized = load_plan("configs/experiments/whole_brain_rocm.yaml")
    assert [(x.exp_name, x.seed) for x in base] == [(x.exp_name, x.seed) for x in optimized]
    for first, second in zip(base, optimized):
        old, new = first.config(), second.config()
        assert new.ppo.micro_batch_size == 8
        assert new.model.core.sparse_backend == "rocm_row"
        assert new.runtime.tunableop_cache == "/local/tuning.csv"
        del new.runtime
        del new.model.core.sparse_backend
        new.ppo.micro_batch_size = old.ppo.micro_batch_size
        assert OmegaConf.to_container(old, resolve=True) == OmegaConf.to_container(new, resolve=True)
