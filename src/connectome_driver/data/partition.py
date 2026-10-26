"""Node roles (afferent / intrinsic / efferent), neuron selections and neurotransmitter signs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

AFFERENT, INTRINSIC, EFFERENT = 0, 1, 2
ROLE_NAMES = ("afferent", "intrinsic", "efferent")


def assign_roles(
    super_class: pd.Series, afferent: Sequence[str], efferent: Sequence[str]
) -> np.ndarray:
    """Role code per neuron from its super_class. Unlisted super_classes are intrinsic."""
    overlap = set(afferent) & set(efferent)
    if overlap:
        raise ValueError(f"super_classes listed as both afferent and efferent: {overlap}")
    sc = super_class.fillna("").str.lower()
    roles = np.full(len(sc), INTRINSIC, dtype=np.int8)
    roles[sc.isin([s.lower() for s in afferent]).to_numpy()] = AFFERENT
    roles[sc.isin([s.lower() for s in efferent]).to_numpy()] = EFFERENT
    return roles


def select(neurons: pd.DataFrame, rules: Mapping[str, Sequence[str]]) -> np.ndarray:
    """Boolean mask of neurons matching ``{column: [values]}``.

    Values are OR-ed within a column and columns are AND-ed. Matching ignores case.
    """
    mask = np.ones(len(neurons), dtype=bool)
    for column, values in rules.items():
        if column not in neurons.columns:
            raise KeyError(f"selection column {column!r} not in neuron table")
        col = neurons[column].fillna("").str.lower()
        mask &= col.isin([v.lower() for v in values]).to_numpy()
    return mask


def fill_transmitter_from_edges(
    neurons: pd.DataFrame, connections: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Neuron transmitter, falling back to the one on its outgoing connections.

    In v783, ~14% of neurons have no neuron-level prediction (``nt_type`` empty, score 0),
    but every connection carries a transmitter label, and the label is unique per presynaptic
    neuron. Where a neuron has a prediction the two always agree, so the connection label is
    used as a fallback. Neurons whose connections disagree (none in v783) stay unknown.

    Returns the filled ``nt_type`` and its source: ``neuron``, ``edges`` or ``none``.
    """
    nt = neurons["nt_type"].astype("string")
    source = pd.Series(np.where(nt.notna(), "neuron", "none"), index=neurons.index)
    if "nt_type" not in connections.columns:
        return nt, source

    per_pre = connections.groupby("pre_root_id", observed=True)["nt_type"].agg(["nunique", "first"])
    unique = per_pre.loc[per_pre["nunique"] == 1, "first"].astype("string")
    fallback = neurons["root_id"].map(unique)
    fill = nt.isna() & fallback.notna()
    nt = nt.where(~fill, fallback)
    source[fill] = "edges"
    return nt, source


def neuron_signs(nt_type: pd.Series, sign_map: Mapping[str, int]) -> np.ndarray:
    """Sign (+1 / -1) of each neuron's outgoing synapses from its predicted neurotransmitter.

    One sign per presynaptic neuron (Dale's law), so every outgoing edge of a neuron shares it.
    Missing or unlisted neurotransmitters get ``sign_map["unknown"]``.
    """
    lookup = {k.lower(): int(v) for k, v in sign_map.items()}
    if any(v not in (-1, 1) for v in lookup.values()):
        raise ValueError(f"signs must be +1 or -1, got {lookup}")
    unknown = lookup.get("unknown", 1)
    nt = nt_type.fillna("").str.lower()
    return nt.map(lambda t: lookup.get(t, unknown)).to_numpy(dtype=np.int8)
