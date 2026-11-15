"""Measured channels of plant streams: flow, temperature, COD, TKN, NOx, TN, TP,
TSS and custom sums, across every shipped model family, following the run's
composition parameters."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import CSTRUnit, InfluentSeries, Measure, Plant
from aquakin.plant.bsm import build_bsm1
from aquakin.plant.calibrate import _normalize_observables, _resolve_observable_columns
from aquakin.plant.identifiability import (
    SensorSet,
    _observed_channels,
    plant_fisher_information,
    plant_rank_sensor_sets,
)
from aquakin.plant.measures import KEYWORDS, measure_map
from aquakin.utils.composition import composition_table

FEED = dict(
    SI=30.0, SS=69.5, XI=51.2, XS=202.32, XB_H=28.17, SNH=31.56, SND=6.95, XND=10.59, SALK=7.0
)


def _species(model, channel):
    """The species a channel weighs, as a set."""
    W = measure_map(model, (channel,)).W[:, 0]
    return {s for s, w in zip(model.species, W) if w}


# Per family: the species of some channels. A channel listed as None is one the
# model cannot give (no composition for it), and must raise. COD, TKN and TN are
# checked against the composition table for every family.
FAMILIES = {
    "asm1": {
        "COD": {"SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP"},
        "NOx": {"SNO"},
        "TP": None,
        "TSS": {"XI", "XS", "XB_H", "XB_A", "XP"},
    },
    "asm2d": {"NOx": {"SNO3"}, "TSS": {"XTSS"}},
    "asm3": {"NOx": {"SNOX"}, "TP": None, "TSS": {"XSS"}},
    "asm3_2step_n2o": {"NOx": {"SNO2", "SNO3"}, "TP": None},
    "asm3_biop": {"NOx": {"SNO"}},
    "adm1": {"NOx": None, "TP": None, "TSS": None},
}


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_channels_of_every_model_family(name):
    m = aquakin.load_model(name)
    table = composition_table(m)
    for channel, expected in FAMILIES[name].items():
        if expected is None:
            with pytest.raises(KeyError):
                measure_map(m, (channel,))
        else:
            assert _species(m, channel) == expected, channel
    species = set(m.species)
    gases = {s for s in species if s.startswith("S_gas_")}
    gases |= {"SN2", "SN2O"} & species
    if "SNO3" in species and "SNO" in species:
        gases.add("SNO")  # nitric oxide
    # COD: the positive-COD species, without electron acceptors or gases.
    cod = _species(m, "COD")
    assert not cod & gases and not cod & {"SO", "SO2", "S_h2", "S_ch4"}
    assert all(table[s].get("COD", 0.0) > 0 for s in cod)
    # TN = TKN + NOx, none of them seeing the dissolved gases.
    tn, tkn = _species(m, "TN"), _species(m, "TKN")
    nox = _species(m, "NOx") if FAMILIES[name].get("NOx", 1) is not None else set()
    assert tn == tkn | nox and not tkn & nox and not tn & gases
    assert tkn == {s for s in species if table[s].get("N", 0.0)} - nox - gases
    W = measure_map(m, ("TKN", "NOx", "TN") if nox else ("TKN", "TN")).W
    np.testing.assert_allclose(W[:, -1], W[:, :-1].sum(axis=1), atol=1e-15)


def test_composition_weights_follow_the_parameters():
    m = aquakin.load_model("asm1")
    mm = measure_map(m, ("TN", "COD"))
    assert mm.params == ("i_XB", "i_XP")
    p = np.asarray(m.default_parameters(), float).copy()
    p[m.param_index["i_XB"]] = 0.07
    moved = composition_table(m, params=p)
    W = mm.W + np.tensordot(p[[m.param_index[n] for n in mm.params]] - mm.theta0, mm.dW, axes=1)
    assert W[m.species_index["XB_H"], 0] == pytest.approx(moved["XB_H"]["N"], rel=1e-12)
    assert W[m.species_index["XB_H"], 0] == pytest.approx(0.07, rel=1e-12)
    np.testing.assert_array_equal(W[:, 1], mm.W[:, 1])  # COD does not depend on i_XB


def test_a_species_named_like_a_keyword_and_bad_channels():
    asm1 = aquakin.load_model("asm1")
    with pytest.raises(KeyError, match=r"Unknown observed channel 'SNHH'.*Did you mean: SNH"):
        measure_map(asm1, ("SNHH",))
    with pytest.raises(KeyError, match="'SSS', which is not a species"):
        measure_map(asm1, (Measure("x", {"SSS": 1.0}),))
    with pytest.raises(TypeError, match="a channel is"):
        measure_map(asm1, (3,))
    with pytest.raises(ValueError, match="at least one channel"):
        measure_map(asm1, ())
    assert set(KEYWORDS) >= {"Q", "T", "TKN", "NOx", "TN"}


def test_a_model_file_extending_a_shipped_model_has_its_composition():
    """The BSM2 project's 15 degC ASM1 extends the shipped ASM1 with its own
    composition parameters, which its TN follows."""
    path = (
        Path(__file__).parents[2] / "examples" / "projects" / "bsm2" / "models" / "asm1_bsm2.yaml"
    )
    m = aquakin.load_model_from_file(path)
    assert composition_table(m)["XB_H"]["N"] == pytest.approx(0.08)
    assert measure_map(m, ("TN",)).W[m.species_index["XB_H"], 0] == pytest.approx(0.08)


def test_bsm1_effluent_measures_match_the_stream():
    m = aquakin.load_model("asm1")
    plant = build_bsm1(model=m)
    plant.add_influent(
        "feed", InfluentSeries.constant(m, Q=18446.0, **FEED), to=plant.influent_endpoint
    )
    params = plant.default_parameters()
    t = np.array([0.0, 0.5])
    sol = plant.solve(t_span=(0.0, 0.5), t_eval=t, params=params)
    channels = ("Q", "COD", "TKN", "NOx", "TN", "TSS", "SNH", Measure("SS/2", {"SS": 0.5}))
    resolved, n = _resolve_observable_columns(
        plant, _normalize_observables([("effluent", channels)], None, None)
    )
    got = np.asarray(_observed_channels(plant, resolved, sol.t, sol.state, params))
    eff = plant.stream(sol, "effluent", params)
    C, Q = np.asarray(eff.C), np.asarray(eff.Q)
    table = composition_table(m, electron_acceptor_cod=False)

    def total(component, skip=()):
        return sum(
            C[:, m.species_index[s]] * max(table[s].get(component, 0.0), 0.0)
            for s in m.species
            if s not in skip
        )

    nox = C[:, m.species_index["SNO"]]
    tss = 0.75 * sum(C[:, m.species_index[s]] for s in ("XI", "XS", "XB_H", "XB_A", "XP"))
    expected = np.stack(
        [
            Q,
            total("COD"),
            total("N", skip=("SNO",)),
            nox,
            total("N"),
            tss,
            C[:, m.species_index["SNH"]],
            0.5 * C[:, m.species_index["SS"]],
        ],
        axis=1,
    )
    assert n == 8 and eff.T is None
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def _asm1_tank(T=None):
    m = aquakin.load_model("asm1")
    plant = Plant("tank")
    plant.add_unit(CSTRUnit("tank", m, 1000.0, ["in"], conditions={"T": 293.15}))
    plant.add_influent("feed", InfluentSeries.constant(m, Q=2000.0, T=T, **FEED), to="tank.in")
    return plant


def test_temperature_is_a_channel_of_a_stream_that_carries_one():
    plant = _asm1_tank(T=288.15)
    params = plant.default_parameters()
    sol = plant.solve(t_span=(0.0, 0.1), t_eval=np.array([0.0, 0.1]), params=params)
    stream = plant.stream(sol, "tank", params)
    np.testing.assert_allclose(np.asarray(stream.T), 288.15)
    resolved, _ = _resolve_observable_columns(
        plant, _normalize_observables([("tank", ["T", "Q"])], None, None)
    )
    got = np.asarray(_observed_channels(plant, resolved, sol.t, sol.state, params))
    np.testing.assert_allclose(got, [[288.15, 2000.0]] * 2)

    bare = _asm1_tank()
    sol = bare.solve(t_span=(0.0, 0.1), t_eval=np.array([0.0, 0.1]))
    assert bare.stream(sol, "tank").T is None
    resolved, _ = _resolve_observable_columns(
        bare, _normalize_observables([("tank", ["T"])], None, None)
    )
    with pytest.raises(ValueError, match="carries no temperature"):
        _observed_channels(bare, resolved, sol.t, sol.state, bare.default_parameters())


@pytest.mark.slow
def test_a_free_composition_parameter_is_seen_by_the_ranking():
    """Fitting ``i_XB``: total nitrogen depends on it through its weights as well
    as through the plant state, and the ranking's per-set scores equal each
    set's own Fisher information, which differentiates the whole forward model."""
    plant = _asm1_tank()
    t_obs = jnp.linspace(0.02, 0.2, 4)
    free = ["asm1.i_XB", "asm1.muH"]
    kw = dict(t_span=(0.0, 0.2))
    sets = [[("tank", ["TN", "SNH"])], [("tank", ["COD", "SNH"])]]
    ranked = plant_rank_sensor_sets(
        plant, [SensorSet(o, sigma=0.1) for o in sets], t_obs, free, **kw
    )
    by_index = {c.index: c.value for c in ranked}
    for i, obs in enumerate(sets):
        fim = plant_fisher_information(plant, obs, t_obs, free, sigma=0.1, **kw)
        assert by_index[i] == pytest.approx(aquakin.design_criterion(fim, "D"), rel=1e-8)
    # TN carries i_XB's information directly (through its weight on the biomass).
    tn = plant_fisher_information(plant, [("tank", ["TN"])], t_obs, ["asm1.i_XB"], sigma=0.1, **kw)
    assert float(tn.matrix[0, 0]) > 0.0
