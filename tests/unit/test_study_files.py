"""Study result files: every value keyed by name, read back bit for bit."""

import numpy as np
import pytest

from aquakin.study import (
    read_matrix,
    read_named_values,
    read_trajectory,
    write_matrix,
    write_named_values,
    write_trajectory,
)

AWKWARD = [0.1, 1 / 3, 1e-300, -2.5e17, np.nextafter(1.0, 2.0)]


def test_named_values_read_back_exactly(tmp_path):
    params = {"asm1.muH": 1 / 3, "asm1.KS": np.float64(10.0), "adm1.k_m_ac": np.asarray(8.0)}
    state = {"tank1": {"SNH": AWKWARD[1], "SO": 0.0}, "clarifier": {"layer1.TSS": AWKWARD[4]}}
    for values in (params, state):
        path = write_named_values(tmp_path / "v.yaml", values)
        back = read_named_values(path)
        assert back == {
            k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v))
            for k, v in values.items()
        }
        assert list(back) == list(values)  # order kept
    with pytest.raises(TypeError, match="keyed by name"):
        write_named_values(tmp_path / "bad.yaml", {0: 1.0})


def test_a_trajectory_reads_back_exactly(tmp_path):
    t = np.linspace(0.0, 1.0, 5)
    columns = {"effluent.SNH": np.asarray(AWKWARD), "effluent.Q": np.array([1, 2, np.inf, 4, 5.0])}
    path = write_trajectory(tmp_path / "s.csv", t, columns)
    assert path.read_text().splitlines()[0] == "t,effluent.SNH,effluent.Q"
    t2, back = read_trajectory(path)
    np.testing.assert_array_equal(t2, t)
    assert list(back) == list(columns)
    for name, values in columns.items():
        np.testing.assert_array_equal(back[name], values)
    with pytest.raises(ValueError, match="has 2 values for 5 times"):
        write_trajectory(tmp_path / "x.csv", t, {"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="'t' is the time column"):
        write_trajectory(tmp_path / "x.csv", t, {"t": t})


def test_a_matrix_keeps_its_row_and_column_names(tmp_path):
    names = ["asm1.muH", "asm1.KS"]
    F = np.array([[AWKWARD[0], AWKWARD[1]], [AWKWARD[1], AWKWARD[3]]])
    m, rows, cols = read_matrix(write_matrix(tmp_path / "F.csv", F, names))
    np.testing.assert_array_equal(m, F)
    assert rows == names and cols == names
    m, rows, cols = read_matrix(write_matrix(tmp_path / "J.csv", F[:, :1], names, ["only"]))
    assert m.shape == (2, 1) and cols == ["only"]
    with pytest.raises(ValueError, match="shape"):
        write_matrix(tmp_path / "bad.csv", F, ["a"])
