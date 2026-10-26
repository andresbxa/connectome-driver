"""A ten-neuron toy brain written in the FlyWire Codex file format.

    visual 101 ─┐
                ├─> optic 201 ─> central 202 ─> descending 301 ─> motor 401
    visual 102 ─┘
    olfactory 103 ─> central 203 ─> descending 302
    ascending 501 ──────┘

Plus rows that preprocessing must drop: a pair below the synapse threshold (201 -> 203),
an autapse (202 -> 202) and a connection from a neuron missing in the classification (999).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from connectome_driver.utils.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
CONNECTIONS_FILE = "connections_princeton.csv.gz"

CLASSIFICATION = [
    # root_id, flow, super_class, class
    (101, "afferent", "sensory", "visual"),
    (102, "afferent", "sensory", "visual"),
    (103, "afferent", "sensory", "olfactory"),
    (201, "intrinsic", "optic", None),
    (202, "intrinsic", "central", None),
    (203, "intrinsic", "central", None),
    (301, "efferent", "descending", None),
    (302, "efferent", "descending", None),
    (401, "efferent", "motor", None),
    (501, "afferent", "ascending", None),
]

NEUROTRANSMITTERS = [
    (101, "ACH"), (102, "ACH"), (103, "ACH"), (201, "GABA"), (202, "GLUT"),
    (203, None), (301, "ACH"), (302, None), (401, "ACH"), (501, "DA"),
]  # 203 and 302 have no neuron-level prediction, as ~14% of v783 neurons

# Transmitter label on the connections, one per presynaptic neuron as in Codex. It fills in
# 203 (GABA); 302 has no outgoing connections and stays unknown.
EDGE_TRANSMITTER = {**dict(NEUROTRANSMITTERS), 203: "GABA", 999: "ACH"}

CONNECTIONS = [
    # pre, post, neuropil, syn_count
    (101, 201, "ME_R", 3),
    (101, 201, "LO_R", 4),  # pair total 7, dominant neuropil LO_R
    (102, 201, "ME_R", 6),
    (201, 202, "PLP_R", 10),
    (202, 301, "GNG", 5),
    (203, 302, "GNG", 8),
    (103, 203, "AL_R", 12),
    (501, 203, "GNG", 6),
    (301, 401, "GNG", 5),
    (201, 203, "SMP_R", 2),  # below threshold
    (202, 202, "SMP_R", 9),  # autapse
    (999, 202, "SMP_R", 20),  # unknown presynaptic neuron
]

EXPECTED_EDGES = {
    (101, 201): 7, (102, 201): 6, (201, 202): 10, (202, 301): 5,
    (203, 302): 8, (103, 203): 12, (501, 203): 6, (301, 401): 5,
}


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    cls = pd.DataFrame(CLASSIFICATION, columns=["root_id", "flow", "super_class", "class"])
    cls["sub_class"] = None
    cls["side"] = "right"
    cls.to_csv(d / "classification.csv.gz", index=False)
    nts = pd.DataFrame(NEUROTRANSMITTERS, columns=["root_id", "nt_type"])
    nts["nt_type_score"] = 0.9
    nts.to_csv(d / "neurons.csv.gz", index=False)
    conn = pd.DataFrame(CONNECTIONS, columns=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
    conn["nt_type"] = conn["pre_root_id"].map(EDGE_TRANSMITTER)
    conn.to_csv(d / CONNECTIONS_FILE, index=False)
    return d


@pytest.fixture
def cfg(raw_dir: Path, tmp_path: Path):
    return load_config(
        CONFIG, [f"raw_dir={raw_dir.as_posix()}", f"processed_dir={(tmp_path / 'processed').as_posix()}"]
    )
