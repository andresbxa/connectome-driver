"""Raw Codex tables -> annotated neuron table + edge list in ``data/processed``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from connectome_driver.data import codex
from connectome_driver.data.graph import Connectome, build_connectome, save_connectome
from connectome_driver.data.partition import assign_roles, neuron_signs, select


def preprocess(cfg: DictConfig) -> tuple[Connectome, dict]:
    raw = Path(cfg.raw_dir)
    neurons = codex.read_neuron_table(raw / cfg.files.classification, raw / cfg.files.neurons)

    neurons["role"] = assign_roles(
        neurons["super_class"], cfg.partition.afferent, cfg.partition.efferent
    )
    neurons["sign"] = neuron_signs(neurons["nt_type"], cfg.signs)
    neurons["is_visual_afferent"] = select(neurons, cfg.visual_afferents)
    neurons["is_readout"] = select(neurons, cfg.readout)

    connections = codex.read_connections(raw / cfg.files.connections)
    graph, report = build_connectome(
        neurons, connections, cfg.edges.min_syn_count, cfg.edges.drop_autapses
    )
    report["neurons"] = graph.num_nodes
    report["visual_afferents"] = int(neurons["is_visual_afferent"].sum())
    report["readout"] = int(neurons["is_readout"].sum())
    if report["visual_afferents"] == 0 or report["readout"] == 0:
        raise ValueError(f"empty visual afferent or readout selection: {report}")
    return graph, report


def run(cfg: DictConfig) -> dict:
    graph, report = preprocess(cfg)
    meta = {
        "source": "FlyWire Codex, materialization 783",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "report": report,
        "edge_syn_count_total": int(np.sum(graph.syn_count, dtype=np.int64)),
    }
    save_connectome(graph, cfg.processed_dir, meta)
    return report
