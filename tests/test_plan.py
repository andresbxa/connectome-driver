import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf

from connectome_driver.utils.config import compose_config
from connectome_driver.utils.plan import load_plan


def write_plan(tmp_path: Path) -> Path:
    path = tmp_path / "plan.yaml"
    OmegaConf.save(OmegaConf.create({
        "common": ["ppo.total_steps=1000"],
        "runs": {
            "a": {"model": ["configs/model/connectome.yaml"], "seeds": [1, 2, 3]},
            "b": {"model": ["configs/model/connectome.yaml", "configs/variant/rewired.yaml"],
                  "overrides": ["model.core.steps=4"], "seeds": [1]},
        },
    }), path)
    return path


def test_plan_is_ordered_seed_by_seed(tmp_path):
    runs = load_plan(write_plan(tmp_path))
    assert [(r.exp_name, r.seed) for r in runs] == [("a", 1), ("b", 1), ("a", 2), ("a", 3)]
    assert runs[1].overrides == ["ppo.total_steps=1000", "model.core.steps=4"]


def test_run_config_matches_train_composition(tmp_path):
    run = load_plan(write_plan(tmp_path))[1]
    cfg = run.config()
    assert cfg.model.core.type == "graph"
    assert cfg.model.graph.variant == "rewired"  # later model files win
    assert cfg.model.core.steps == 4  # overrides win over model files
    assert cfg.ppo.total_steps == 1000
    assert (cfg.exp_name, cfg.seed) == ("b", 1)
    assert cfg == compose_config("configs/train.yaml", run.model, [*run.overrides, "exp_name=b", "seed=1"])


def test_runner_status_uses_explicit_output_root(tmp_path, monkeypatch, capsys):
    module_spec = importlib.util.spec_from_file_location("experiment_runner", "scripts/run_experiments.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    output = tmp_path / "separate-output"
    done = output / "a" / "seed1"
    done.mkdir(parents=True)
    (done / "DONE").write_text("1000\n")
    (done / "eval.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["runner", str(write_plan(tmp_path)),
                                     "--run-dir", str(output), "--status"])
    module.main()
    assert "done     a seed1" in capsys.readouterr().out
