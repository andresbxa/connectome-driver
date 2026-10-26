import numpy as np
import pandas as pd
import pytest

from connectome_driver.data.partition import (
    AFFERENT,
    EFFERENT,
    INTRINSIC,
    assign_roles,
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
