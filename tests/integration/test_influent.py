"""Tests for influent CSV / text parsing."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.influent import (
    _influent_from_text,
    load_bsm1_influent,
    load_bsm2_influent,
    read_influent_csv,
)


@pytest.fixture
def asm1():
    return aquakin.load_network("asm1")


def _csv_text(network):
    # One header row + two data rows, in the default BSM1 column order.
    from aquakin.plant.influent import _BSM1_COLUMN_ORDER

    header = ",".join(_BSM1_COLUMN_ORDER)
    zeros = {name: 0.0 for name in _BSM1_COLUMN_ORDER}
    rows = []
    for i, tval in enumerate((0.0, 0.25)):
        vals = dict(zeros, t=tval, Q=18000.0 + i, SNH=30.0, SS=70.0)
        rows.append(",".join(f"{vals[c]:g}" for c in _BSM1_COLUMN_ORDER))
    return header + "\n" + "\n".join(rows) + "\n"


def test_influent_from_text_parses_without_a_file(asm1):
    series = _influent_from_text(_csv_text(asm1), asm1)
    assert series.t.shape == (2,)
    assert series.C.shape == (2, asm1.n_species)
    assert float(series.Q[0]) == pytest.approx(18000.0)
    assert float(series.C[0, asm1.species_index["SNH"]]) == pytest.approx(30.0)


def test_read_influent_csv_matches_text(asm1, tmp_path):
    text = _csv_text(asm1)
    p = tmp_path / "inf.csv"
    p.write_text(text)
    from_file = read_influent_csv(p, asm1)
    from_text = _influent_from_text(text, asm1)
    assert jnp.allclose(from_file.C, from_text.C)
    assert jnp.allclose(from_file.Q, from_text.Q)


def test_read_influent_csv_missing_file(asm1, tmp_path):
    with pytest.raises(FileNotFoundError):
        read_influent_csv(tmp_path / "nope.csv", asm1)


def test_influent_from_text_captures_T_column(asm1):
    """A 'T' column is captured into InfluentSeries.T (in the file's units);
    its absence leaves T = None."""
    from aquakin.plant.influent import _BSM1_COLUMN_ORDER

    order = _BSM1_COLUMN_ORDER + ["T"]
    header = ",".join(order)
    zeros = {name: 0.0 for name in order}
    rows = []
    for tval, temp in ((0.0, 12.0), (0.25, 16.0)):
        vals = dict(zeros, t=tval, Q=18000.0, T=temp)
        rows.append(",".join(f"{vals[c]:g}" for c in order))
    text = header + "\n" + "\n".join(rows) + "\n"
    series = _influent_from_text(text, asm1, column_order=order)
    assert series.T is not None
    assert float(series.T[0]) == pytest.approx(12.0)
    assert float(series.T[1]) == pytest.approx(16.0)
    # No T column -> T is None (default, back-compatible).
    assert _influent_from_text(_csv_text(asm1), asm1).T is None


def test_load_bsm2_influent_carries_temperature_kelvin(asm1):
    """The BSM2 influent files carry a temperature column; load_bsm2_influent
    returns it in Kelvin (the degC file values + 273.15), seasonally varying."""
    for profile in ("dry", "rain", "storm"):
        inf = load_bsm2_influent(profile, asm1)
        assert inf.T is not None
        assert inf.T.shape == inf.t.shape
        # 11.5-18.5 degC window -> ~284.6-291.7 K.
        assert 283.0 < float(inf.T.min()) < float(inf.T.max()) < 293.0
        # at(t) interpolates the temperature onto the returned stream.
        s = inf.at(jnp.asarray(7.0))
        assert s.T is not None and 285.0 < float(s.T) < 291.0


def test_load_bsm1_influent_has_no_temperature(asm1):
    """BSM1 files have no temperature column, so T stays None (back-compatible)."""
    assert load_bsm1_influent("dry", asm1).T is None


@pytest.mark.parametrize("profile", ["dry", "rain", "storm"])
def test_load_bsm1_influent_no_tempfile(asm1, profile, monkeypatch):
    # The package data must be parsed directly from text; loading must not
    # create a temporary file (the old disk round-trip).
    import tempfile

    def _boom(*a, **k):
        raise AssertionError("load_bsm1_influent must not use a tempfile")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _boom)
    series = load_bsm1_influent(profile, asm1)
    assert series.t.shape[0] > 1
    assert series.C.shape[1] == asm1.n_species
