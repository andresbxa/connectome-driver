"""Readers for the raw FlyWire Codex tables.

Only the columns the pipeline needs are loaded, and their presence is checked so that a
schema change in a new Codex export fails loudly instead of silently producing a wrong graph.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CONNECTION_COLUMNS = ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]
CLASSIFICATION_COLUMNS = ["root_id", "flow", "super_class", "class", "sub_class", "side"]
NEURON_COLUMNS = ["root_id", "nt_type"]

# Columns that may be absent without breaking the pipeline.
OPTIONAL_COLUMNS = {"flow", "sub_class", "side", "nt_type"}


def _read(path: Path, columns: list[str], dtype: dict) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from FlyWire Codex (see README, section Data)."
        )
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in columns if c not in header and c not in OPTIONAL_COLUMNS]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; found {list(header)}")
    usecols = [c for c in columns if c in header]
    return pd.read_csv(path, usecols=usecols, dtype={k: v for k, v in dtype.items() if k in usecols})


def read_connections(path: Path) -> pd.DataFrame:
    """One row per (pre, post, neuropil) with its synapse count and transmitter."""
    return _read(
        path,
        CONNECTION_COLUMNS,
        {
            "pre_root_id": "int64",
            "post_root_id": "int64",
            "neuropil": "category",
            "syn_count": "int32",
            "nt_type": "category",
        },
    )


def read_classification(path: Path) -> pd.DataFrame:
    """One row per neuron with its hierarchical annotations."""
    dtype = {c: "string" for c in CLASSIFICATION_COLUMNS if c != "root_id"}
    return _read(path, CLASSIFICATION_COLUMNS, {"root_id": "int64", **dtype})


def read_neurotransmitters(path: Path) -> pd.DataFrame:
    """One row per neuron with its predicted neurotransmitter."""
    return _read(path, NEURON_COLUMNS, {"root_id": "int64", "nt_type": "string"})


def read_neuron_table(classification_path: Path, neurons_path: Path) -> pd.DataFrame:
    """Annotated neurons: the classification table plus the predicted neurotransmitter.

    The classification table defines the set of neurons (the graph's nodes). Neurons without a
    neurotransmitter prediction keep ``nt_type = <NA>``.
    """
    cls = read_classification(classification_path)
    nts = read_neurotransmitters(neurons_path)
    if cls["root_id"].duplicated().any():
        raise ValueError("classification table has duplicated root_id values")
    nts = nts.drop_duplicates("root_id")
    table = cls.merge(nts, on="root_id", how="left", validate="one_to_one")
    return table.sort_values("root_id", ignore_index=True)
