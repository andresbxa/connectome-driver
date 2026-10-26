"""Statistics of the processed connectome: partition, degrees, signs, path lengths, subgraphs.

Prints a report and writes it to ``results/connectome_stats.json``.

Usage:
    python scripts/graph_stats.py [key=value ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from connectome_driver.data import stats
from connectome_driver.data.graph import load_connectome
from connectome_driver.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="results/connectome_stats.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    graph = load_connectome(cfg.processed_dir)
    neurons = graph.neurons
    sources = np.flatnonzero(neurons["is_visual_afferent"].to_numpy())
    targets = np.flatnonzero(neurons["is_readout"].to_numpy())

    roles = stats.role_counts(neurons)
    report = {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "total_synapses": int(graph.syn_count.sum(dtype=np.int64)),
        "syn_count_per_edge": stats._summary(graph.syn_count),
        "roles": roles,
        "roles_minus_flygm": {k: roles[k] - v for k, v in stats.FLYGM_COUNTS.items()},
        "visual_afferents": len(sources),
        "readout_neurons": len(targets),
        "nt_type_counts": neurons["nt_type"].fillna("<NA>").value_counts().to_dict(),
        "signs": stats.sign_stats(graph),
        "degrees": stats.degree_stats(graph),
        "visual_to_readout": stats.reachability(graph, sources, targets, cfg.stats.hop_budgets),
        "path_subgraphs": stats.subgraph_sizes(graph, sources, targets, cfg.stats.subgraph_k),
    }

    print("\nsuper_class -> role" + (" (vs Codex flow)" if "flow" in neurons else ""))
    print(stats.super_class_table(neurons).to_string(index=False))
    print("\nroles (this project vs FlyGM):")
    for name, count in roles.items():
        print(f"  {name:<10} {count:>8,}   FlyGM {stats.FLYGM_COUNTS[name]:>8,}")
    print("\nfull report:")
    print(json.dumps(report, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
