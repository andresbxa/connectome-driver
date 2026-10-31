"""Tune only the shared 8x8 graph-weight gradient GEMM, outside a training run.

Example for the complete connectome:
    python scripts/tune_rocm.py --nodes 139255 --micro-batch 8 --out .venv/rocm-gemm.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--node-dim", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nodes, args.micro_batch, args.node_dim) < 1:
        parser.error("nodes, micro-batch and node-dim must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("This tuner requires a ROCm GPU")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache_already_present = args.out.is_file()
    torch.manual_seed(42)
    x = torch.randn(args.nodes * args.micro_batch, args.node_dim,
                    device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    torch.cuda.tunable.enable(False)
    reference = (x.T @ grad).cpu().float()
    torch.cuda.tunable.set_filename(str(args.out.resolve()), insert_device_ordinal=False)
    torch.cuda.tunable.set_max_tuning_duration(5)
    torch.cuda.tunable.set_max_tuning_iterations(3)
    torch.cuda.tunable.enable(True)
    torch.cuda.tunable.tuning_enable(True)
    print("Tuning the shared-weight gradient; this can take several minutes.", flush=True)
    started = time.perf_counter()
    value = x.T @ grad
    torch.cuda.synchronize()
    tuning_seconds = time.perf_counter() - started
    torch.cuda.tunable.tuning_enable(False)
    actual = value.cpu().float()
    error = float((actual - reference).norm() / reference.norm().clamp_min(1e-20))
    if not torch.isfinite(actual).all() or error > 0.02:
        raise RuntimeError(f"The selected GEMM failed the relative L2 accuracy check: {error}")
    # Persist the public results explicitly; do not rely on interpreter shutdown.
    with args.out.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for name, version in torch.cuda.tunable.get_validators():
            writer.writerow(("Validator", name, version))
        writer.writerows(torch.cuda.tunable.get_results())
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "hip": torch.version.hip, "nodes": args.nodes, "micro_batch": args.micro_batch,
              "node_dim": args.node_dim, "tuning_seconds": tuning_seconds,
              "cache_already_present": cache_already_present,
              "relative_l2_difference_from_default": error,
              "cache_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
