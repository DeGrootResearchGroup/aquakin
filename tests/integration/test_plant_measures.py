"""Measured channels of plant streams: flow, total COD / N / P, TSS and custom sums."""

import numpy as np
import pytest

import aquakin
from aquakin.plant import InfluentSeries, Measure
from aquakin.plant.bsm import build_bsm1
from aquakin.plant.calibrate import _normalize_observables, _resolve_observable_columns
from aquakin.plant.identifiability import _observed_channels
from aquakin.plant.measures import measure_columns
from aquakin.utils.composition import composition_table

FEED = dict(
    SI=30.0, SS=69.5, XI=51.2, XS=202.32, XB_H=28.17, SNH=31.56, SND=6.95, XND=10.59, SALK=7.0
)


def test_asm1_measures():
    m = aquakin.load_model("asm1")
    W, q = measure_columns(m, ("Q", "COD", "TN", "TSS", "SNH"))
    col = {
        name: dict(zip(m.species, W[:, i]))
        for i, name in enumerate(("Q", "COD", "TN", "TSS", "SNH"))
    }
    assert list(q) == [1.0, 0.0, 0.0, 0.0, 0.0] and not W[:, 0].any()
    # Measured COD counts organics, not the electron acceptors oxygen and nitrate.
    assert col["COD"]["SO"] == 0.0 and col["COD"]["SNO"] == 0.0 and col["COD"]["XB_H"] == 1.0
    assert col["TN"]["SNH"] == 1.0 and col["TN"]["XB_H"] == pytest.approx(0.086)
    assert col["TSS"] == {
        s: (0.75 if s in ("XI", "XS", "XB_H", "XB_A", "XP") else 0.0) for s in m.species
    }
    assert col["SNH"]["SNH"] == 1.0 and sum(col["SNH"].values()) == 1.0


def test_measures_follow_each_model():
    asm2d = aquakin.load_model("asm2d")
    W, _ = measure_columns(asm2d, ("TSS", "TP"))
    assert W[asm2d.species_index["XTSS"], 0] == 1.0 and W[:, 0].sum() == 1.0
    assert W[asm2d.species_index["SPO4"], 1] == 1.0
    adm1 = aquakin.load_model("adm1")
    with pytest.raises(KeyError, match="has no TSS species"):
        measure_columns(adm1, ("TSS",))
    with pytest.raises(KeyError, match="carries no P"):
        measure_columns(adm1, ("TP",))
    with pytest.raises(KeyError, match=r"Unknown observed channel 'SNHH'.*Did you mean: SNH"):
        measure_columns(aquakin.load_model("asm1"), ("SNHH",))
    with pytest.raises(KeyError, match="'SSS', which is not a species"):
        measure_columns(aquakin.load_model("asm1"), (Measure("x", {"SSS": 1.0}),))


def test_a_model_file_extending_asm1_has_its_composition():
    """The BSM2 project's 15 degC ASM1 extends the shipped ASM1 with its own
    composition parameters, which its TN follows."""
    from pathlib import Path

    path = (
        Path(__file__).parents[2] / "examples" / "projects" / "bsm2" / "models" / "asm1_bsm2.yaml"
    )
    m = aquakin.load_model_from_file(path)
    assert composition_table(m)["XB_H"]["N"] == pytest.approx(0.08)
    W, _ = measure_columns(m, ("TN",))
    assert W[m.species_index["XB_H"], 0] == pytest.approx(0.08)


def test_bsm1_effluent_measures_match_the_stream():
    m = aquakin.load_model("asm1")
    plant = build_bsm1(model=m)
    plant.add_influent(
        "feed", InfluentSeries.constant(m, Q=18446.0, **FEED), to=plant.influent_endpoint
    )
    params = plant.default_parameters()
    t = np.array([0.0, 0.5])
    sol = plant.solve(t_span=(0.0, 0.5), t_eval=t, params=params)
    channels = ("Q", "COD", "TN", "TSS", "SNH", Measure("SS/2", {"SS": 0.5}))
    resolved, n = _resolve_observable_columns(
        plant, _normalize_observables([("effluent", channels)], None, None)
    )
    got = np.asarray(_observed_channels(plant, resolved, sol.t, sol.state, params))
    eff = plant.stream(sol, "effluent", params)
    C, Q = np.asarray(eff.C), np.asarray(eff.Q)
    table = composition_table(m, electron_acceptor_cod=False)
    cod = sum(C[:, m.species_index[s]] * max(table[s].get("COD", 0.0), 0.0) for s in m.species)
    tn = sum(C[:, m.species_index[s]] * table[s].get("N", 0.0) for s in m.species)
    tss = 0.75 * sum(C[:, m.species_index[s]] for s in ("XI", "XS", "XB_H", "XB_A", "XP"))
    expected = np.stack(
        [Q, cod, tn, tss, C[:, m.species_index["SNH"]], 0.5 * C[:, m.species_index["SS"]]], axis=1
    )
    assert n == 6
    np.testing.assert_allclose(got, expected, rtol=1e-10)
