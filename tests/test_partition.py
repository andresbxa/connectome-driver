import numpy as np
import pandas as pd
import pytest

from connectome_driver.data.partition import (
    AFFERENT,
    EFFERENT,
    INTRINSIC,
    assign_roles,
    fill_transmitter_from_edges,
    neuron_signs,
    select,
)


def test_roles_by_super_class():
    sc = pd.Series(["sensory", "Ascending", "optic", "descending", "motor", None], dtype="string")
    roles = assign_roles(sc, ["sensory", "ascending"], ["descending", "motor"])
    assert roles.tolist() == [AFFERENT, AFFERENT, INTRINSIC, EFFERENT, EFFERENT, INTRINSIC]


def test_roles_reject_overlap():
    with pytest.raises(ValueError):
        assign_roles(pd.Series(["x"], dtype="string"), ["a", "b"], ["b"])


def test_signs_follow_neurotransmitter():
    nt = pd.Series(["ACH", "gaba", "GLUT", "DA", None, "weird"], dtype="string")
    signs = neuron_signs(nt, {"ach": 1, "gaba": -1, "glut": -1, "da": 1, "unknown": 1})
    assert signs.tolist() == [1, -1, -1, 1, 1, 1]


def test_glutamate_sign_is_configurable():
    nt = pd.Series(["GLUT"], dtype="string")
    assert neuron_signs(nt, {"glut": 1})[0] == 1


def test_signs_must_be_unit():
    with pytest.raises(ValueError):
        neuron_signs(pd.Series(["ACH"], dtype="string"), {"ach": 2})


def test_transmitter_fallback_only_when_edges_agree():
    neurons = pd.DataFrame(
        {"root_id": [1, 2, 3, 4], "nt_type": ["ACH", None, None, None]}
    ).astype({"nt_type": "string"})
    conns = pd.DataFrame(
        {"pre_root_id": [1, 2, 2, 3, 3], "nt_type": ["GABA", "GLUT", "GLUT", "ACH", "GABA"]}
    ).astype({"nt_type": "category"})
    nt, source = fill_transmitter_from_edges(neurons, conns)
    # Neuron-level prediction wins; 3 has conflicting edges and 4 has none.
    assert nt.tolist() == ["ACH", "GLUT", pd.NA, pd.NA]
    assert source.tolist() == ["neuron", "edges", "none", "none"]


def test_select_ands_columns_and_ors_values():
    df = pd.DataFrame(
        {
            "super_class": ["sensory", "sensory", "optic", "sensory"],
            "class": ["visual", "olfactory", "visual", "Ocellar"],
        },
        dtype="string",
    )
    mask = select(df, {"super_class": ["sensory"], "class": ["visual", "ocellar"]})
    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_select_unknown_column():
    with pytest.raises(KeyError):
        select(pd.DataFrame({"a": ["x"]}), {"b": ["x"]})
