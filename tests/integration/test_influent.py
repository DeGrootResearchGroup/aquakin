"""Tests for influent CSV / text parsing."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.influent import (
    _influent_from_text,
    load_bsm1_influent,
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
