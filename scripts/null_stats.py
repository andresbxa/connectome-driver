"""Report on the null models: cost, how much they rewire, and how far the readout ends up.

For the k=3 path subgraph (the main experiment) and the full brain, samples each null with
``seed=0`` and records its runtime, swap acceptance, fraction of rewired edges, and the
reachability of the descending neurons from the visual afferents within 2/3/4/8/12 hops.
Nulls can change how many hops the visual signal needs, so a null could look better or worse
than the connectome only because the signal arrives sooner or later; these numbers are needed
to read the results.

Usage:
    python scripts/null_stats.py [--graphs k3 full] [--out results/null_stats.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from connectome_driver.data import nulls
from connectome_driver.data.graphs import build_graph, spec_from_graph
from connectome_driver.data.stats import reachability
from connectome_driver.utils.config import load_config

HOP_BUDGETS = [2, 3, 4, 8, 12]
GRAPHS = {"k3": 3, "full": None}


def describe(spec, report: dict | None = None, seconds: float | None = None) -> dict:
    g = spec.graph
    out_deg = np.bincount(g.src, minlength=g.num_nodes)
    row = {
        "edges": g.num_edges,
        "max_out_degree": int(out_deg.max()),
        "reachability": reachability(g, spec.input_index, spec.readout_index, HOP_BUDGETS),
    }
    if seconds is not None:
        row["seconds"] = round(seconds, 2)
    return {**row, **(report or {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=list(GRAPHS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/null_stats.json"))
    args = parser.parse_args()

    gcfg = load_config(args.config).model.graph
    result = {"seed": args.seed, "swaps_per_edge": gcfg.swaps_per_edge, "blocks": gcfg.blocks,
              "hop_budgets": HOP_BUDGETS}
    for name in args.graphs:
        cfg = OmegaConf.merge(gcfg, {"subgraph_k": GRAPHS[name], "variant": "connectome"})
        real = build_graph(cfg)
        g = real.graph
        print(f"{name}: {g.num_nodes:,} neurons, {g.num_edges:,} edges", flush=True)
        rows = {"neurons": g.num_nodes, "connectome": describe(real)}
        blocks = nulls.neuron_blocks(g, by=gcfg.blocks)
        samplers = {
            "rewired": lambda r, g=g: nulls.rewire_degree_preserving(
                g, seed=args.seed, swaps_per_edge=gcfg.swaps_per_edge, report=r),
            "random": lambda r, g=g: nulls.random_graph(g, seed=args.seed, report=r),
            "block": lambda r, g=g, blocks=blocks: nulls.rewire_within_blocks(
                g, blocks, seed=args.seed, swaps_per_edge=gcfg.swaps_per_edge, report=r),
        }
        for variant, sample in samplers.items():
            report = {}
            start = time.perf_counter()
            null = sample(report)
            seconds = time.perf_counter() - start
            rows[variant] = describe(spec_from_graph(null), report, seconds)
            frac = rows[variant]["reachability"]["reachable_fraction"]
            print(f"  {variant:8} {seconds:7.1f} s  " + "  ".join(f"<= {b} hops {frac[b]:.0%}" for b in HOP_BUDGETS)
                  + f"  {json.dumps(report)}", flush=True)
        frac = rows["connectome"]["reachability"]["reachable_fraction"]
        print("  connectome          " + "  ".join(f"<= {b} hops {frac[b]:.0%}" for b in HOP_BUDGETS), flush=True)
        result[name] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
