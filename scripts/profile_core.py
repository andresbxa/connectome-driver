"""Time and memory of the policy per core, graph size, T, dtype and checkpointing.

For every configuration it measures, on the GPU:
- rollout: one forward pass for the 8 parallel environments (no gradient);
- update: forward + backward of one PPO minibatch (256 samples), split into the largest
  micro-batch that fits in memory;
- peak GPU memory of that update.

From these it estimates the wall time of a training run:

    hours = iterations * (rollout_steps * (env_ms + forward_ms) + minibatches * epochs * update_ms)

with ``env_ms`` the time of one vectorized environment step, measured separately (CPU-bound and
the same for every core). The estimate is calibrated against a real run when ``--calibrate``
points to one (its measured wall time divided by the estimate for the same configuration).

Each configuration runs in its own subprocess, with the PyTorch allocator capped below the
card's memory. On Windows the driver otherwise spills to system RAM instead of failing (huge
slowdowns that look like valid measurements), and a cuSPARSE allocation failure leaves the CUDA
context unusable for the rest of the process.

With ``--plan``, it profiles the entries of an experiment plan instead, each configured exactly
as in training (used for the cost columns of the results table).

Usage:
    python scripts/profile_core.py [--calibrate runs/mlp_baseline/seed1] [--quick]
    python scripts/profile_core.py --plan configs/experiments/main.yaml --out results/profile_main.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from connectome_driver.data.graphs import build_graph
from connectome_driver.envs.car_racing import make_vector_env, observation_shape
from connectome_driver.models.policy import build_policy, needs_graph
from connectome_driver.utils.config import compose_config, load_config
from connectome_driver.utils.plan import load_plan
from connectome_driver.utils.runtime import configure_runtime

MICRO_BATCHES = (256, 128, 64, 32, 16)
MEMORY_FRACTION = 0.9  # leave room for cuSPARSE workspaces, which bypass the allocator


def cuda_ms(fn, reps: int) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / reps


def env_step_ms(cfg, steps: int = 200) -> float:
    envs = make_vector_env(cfg.env, cfg.ppo.num_envs)
    envs.reset(seed=0)
    actions = np.full(cfg.ppo.num_envs, 3)
    for _ in range(20):
        envs.step(actions)
    t = time.perf_counter()
    for _ in range(steps):
        envs.step(actions)
    ms = 1000 * (time.perf_counter() - t) / steps
    envs.close()
    return ms


def profile(cfg, graph, obs_shape, device, micro_batches=MICRO_BATCHES) -> dict:
    policy = build_policy(cfg.model, obs_shape, 5, graph).to(device)
    n_env, mb = cfg.ppo.num_envs, cfg.ppo.num_envs * cfg.ppo.rollout_steps // cfg.ppo.num_minibatches
    obs8 = torch.randint(0, 256, (n_env, *obs_shape), dtype=torch.uint8, device=device)
    with torch.no_grad():
        forward_ms = cuda_ms(lambda: policy.act(obs8), reps=20)

    row = {"forward_ms": forward_ms, "micro_batch": None, "update_ms": None, "peak_mem_mb": None}
    for micro in micro_batches:
        obs = torch.randint(0, 256, (micro, *obs_shape), dtype=torch.uint8, device=device)
        actions = torch.zeros(micro, dtype=torch.long, device=device)

        def update(obs=obs, actions=actions, micro=micro):
            for _ in range(mb // micro):
                _, logprob, entropy, value = policy.act(obs, actions)
                (logprob.sum() + value.sum() + entropy.sum()).backward()

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row["update_ms"] = cuda_ms(update, reps=2)
            row["micro_batch"] = micro
            row["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
            break
        except torch.cuda.OutOfMemoryError:
            policy.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    row["params"] = sum(policy.parameter_counts().values())
    return row


GRAPHS = {"k3": 3, "k4": 4, "full": None}


def profile_one(config: str, spec: dict) -> dict:
    """Profile a single configuration (runs inside a subprocess)."""
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    micro = [m for m in MICRO_BATCHES if m <= spec.get("max_micro", MICRO_BATCHES[0])]
    if "model" in spec:  # an entry of an experiment plan, configured exactly as in training
        cfg = compose_config(config, spec["model"], spec["overrides"])
        configure_runtime(cfg)
        graph = build_graph(cfg.model.graph, seed=1) if needs_graph(cfg.model) else None
        return profile(cfg, graph, observation_shape(cfg.env), torch.device("cuda"), micro)
    base = load_config(config)
    graph = build_graph(OmegaConf.merge(base.model.graph, {"subgraph_k": GRAPHS[spec["graph"]]}))
    if spec["core"] == "mlp":
        cfg = OmegaConf.merge(base, {"model": OmegaConf.load("configs/model/mlp.yaml")})
    else:
        core = {"type": "graph", "steps": spec["T"], "signed": True, "dtype": spec["dtype"],
                "checkpoint": spec["checkpoint"]}
        cfg = OmegaConf.merge(base, {"model": {"core": core}})
    return profile(cfg, graph, observation_shape(base.env), torch.device("cuda"), micro)


def run_isolated(config: str, spec: dict) -> dict:
    """Profile ``spec`` in a fresh process; after a CUDA failure retry with smaller micro-batches."""
    max_micro = MICRO_BATCHES[0]
    while True:
        payload = json.dumps({**spec, "max_micro": max_micro})
        proc = subprocess.run(
            [sys.executable, "-W", "ignore", __file__, "--config", config, "--one", payload],
            capture_output=True, text=True, check=False,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("ROW ")]
        if proc.returncode == 0 and lines:
            return {**spec, **json.loads(lines[-1][4:])}
        smaller = [m for m in MICRO_BATCHES if m < max_micro]
        if not smaller:
            return {**spec, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed"}
        max_micro = smaller[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--calibrate", type=Path, default=None, help="run dir of a finished MLP run")
    parser.add_argument("--quick", action="store_true", help="fewer configurations")
    parser.add_argument("--plan", type=Path, default=None,
                        help="profile the entries of an experiment plan instead of the core sweep")
    parser.add_argument("--only", nargs="*", default=None, help="plan entries to profile")
    parser.add_argument("--out", type=Path, default=Path("results/profile_core.json"))
    parser.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plan is not None and args.calibrate is not None:
        parser.error("--calibrate needs the core sweep (it uses its full-brain MLP row)")

    if args.one is not None:
        print("ROW " + json.dumps(profile_one(args.config, json.loads(args.one))), flush=True)
        return

    base = load_config(args.config)
    env_ms = env_step_ms(base)
    print(f"vectorized env step ({base.ppo.num_envs} envs): {env_ms:.1f} ms")

    steps = (4, 8) if args.quick else (4, 8, 12)
    specs = {name: build_graph(OmegaConf.merge(base.model.graph, {"subgraph_k": k}))
             for name, k in GRAPHS.items()}
    if args.plan is not None:
        entries = {r.exp_name: r for r in load_plan(args.plan) if args.only is None or r.exp_name in args.only}
        jobs = [{"exp": name, "model": r.model, "overrides": r.overrides} for name, r in entries.items()]
    else:
        jobs = [{"core": "mlp", "graph": name} for name in GRAPHS]
        jobs += [
            {"core": "graph", "graph": name, "T": T, "dtype": dtype, "checkpoint": ckpt}
            for name, T, dtype, ckpt in itertools.product(GRAPHS, steps, ("float32", "bfloat16"), (False, True))
        ]
    rows = []
    for job in jobs:
        rows.append(run_isolated(args.config, job))
        print(rows[-1], flush=True)

    ppo = base.ppo
    iterations = ppo.total_steps // (ppo.num_envs * ppo.rollout_steps)
    df = pd.DataFrame(rows)
    per_iter_s = (ppo.rollout_steps * (env_ms + df["forward_ms"])
                  + ppo.num_minibatches * ppo.update_epochs * df["update_ms"]) / 1000
    df["est_hours_per_run"] = iterations * per_iter_s / 3600

    calibration = None
    if args.calibrate is not None:
        ep = pd.read_csv(args.calibrate / "episodes.csv")
        real_h = ep["wall_time"].iloc[-1] / 3600
        est = df.loc[(df.core == "mlp") & (df.graph == "full"), "est_hours_per_run"].iloc[0]
        calibration = real_h / est
        df["est_hours_per_run"] *= calibration
        print(f"calibration: real {real_h:.2f} h / estimated {est:.2f} h = x{calibration:.2f}")

    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "env_step_ms": env_ms,
        "iterations_per_run": iterations,
        "total_steps_per_run": ppo.total_steps,
        "calibration": calibration,
        "graphs": {name: {"neurons": s.graph.num_nodes, "edges": s.graph.num_edges,
                          "inputs": len(s.input_index), "readouts": len(s.readout_index)}
                   for name, s in specs.items()},
        "rows": json.loads(df.to_json(orient="records")),
    }, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
