"""Solids units that convert their g/m3 constants, and the automatic
unit-scaling translator.

Each test describes one physical system twice, with a model declaring its
species in g/m3 and a copy declaring some of them in kg/m3, and checks the two
agree once the units are accounted for.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import aquakin
from aquakin.plant import CSTRUnit, MixerUnit, Plant, UnitScalingTranslator, WiringError
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.separators import IdealThickener
from aquakin.plant.streams import Stream
from aquakin.plant.takacs import TakacsClarifier
from aquakin.plant.translators import IdentityTranslator, unit_scaling

KG_SPECIES = ("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP")


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


@pytest.fixture(scope="module")
def asm1_kg(asm1):
    """ASM1 with its COD species declared in kgCOD/m3."""
    units = {**asm1.species_units, **dict.fromkeys(KG_SPECIES, "kgCOD/m3")}
    return dataclasses.replace(asm1, name="asm1_kg", species_units=units)


def _to_kg(asm1, asm1_kg, C):
    """``C`` (in ASM1's units) expressed in ``asm1_kg``'s."""
    return UnitScalingTranslator(asm1, asm1_kg).translate(C)


# ----- the thickener ------------------------------------------------------------


def test_thickener_reaches_the_same_target_from_kg_per_m3(asm1, asm1_kg):
    C_g = asm1.concentrations({"XS": 2000.0, "XB_H": 3000.0, "XI": 1000.0, "SS": 10.0})
    C_k = _to_kg(asm1, asm1_kg, C_g)
    out_g = IdealThickener("t", asm1, target_tss_percent=7.0).compute_outputs(
        0.0, None, {"inlet": Stream(Q=100.0, C=C_g, model=asm1)}, None
    )
    out_k = IdealThickener("t", asm1_kg, target_tss_percent=7.0).compute_outputs(
        0.0, None, {"inlet": Stream(Q=100.0, C=C_k, model=asm1_kg)}, None
    )
    for port in ("underflow", "overflow"):
        np.testing.assert_allclose(out_k[port].Q, out_g[port].Q, rtol=1e-12)
        np.testing.assert_allclose(out_k[port].C, _to_kg(asm1, asm1_kg, out_g[port].C), rtol=1e-12)
    assert float(out_g["underflow"].Q) < 100.0  # it did thicken


def test_thickener_reports_solids_declared_in_moles(asm1):
    # Moles cannot be put on a g/m3 basis: the species is used as if in g/m3
    # and the plant's unit audit reports it.
    molar = dataclasses.replace(asm1, species_units={**asm1.species_units, "XS": "mol/m3"})
    th = IdealThickener("t", molar, target_tss_percent=7.0)
    assert th.assumed_units() == {"XS": "g/m3"}
    plant = Plant("th", time_unit="d")
    plant.add_unit(th)
    plant.add_influent("feed", molar.influent(Q=1.0), to="t")
    (err,) = plant.check_units(include_models=False).errors
    assert "'XS'" in err.detail and "mol/m3" in err.detail


def test_thickener_on_shipped_units_is_unchanged(asm1):
    th = IdealThickener("t", asm1, target_tss_percent=7.0)
    assert set(np.unique(np.asarray(th._tss_vec))) <= {0.0, 0.75}
    assert th.assumed_units() == {}


# ----- the Takács settler -------------------------------------------------------


def _settler_plant(model, C, mode):
    plant = Plant(f"settler_{model.name}", time_unit="d")
    plant.add_unit(
        TakacsClarifier(
            "settler",
            model,
            area=1500.0,
            height=4.0,
            underflow_Q=18000.0,
            composition_mode=mode,
        )
    )
    feed = InfluentSeries.constant(model, Q=36000.0)
    feed = dataclasses.replace(feed, C=np.tile(np.asarray(C), (2, 1)))
    plant.add_influent("feed", feed, to="settler")
    return plant


@pytest.mark.parametrize("mode", ["per_species", "lumped_tss"])
def test_settler_with_default_parameters_settles_kg_per_m3_alike(asm1, asm1_kg, mode):
    C_g = asm1.concentrations(
        {"XS": 300.0, "XB_H": 2500.0, "XB_A": 150.0, "XI": 1200.0, "XP": 450.0, "SS": 5.0}
    )
    C_k = _to_kg(asm1, asm1_kg, C_g)
    grams = _settler_plant(asm1, C_g, mode)
    kilos = _settler_plant(asm1_kg, C_k, mode)
    y_g = grams.initial_state()
    settler = grams.units["settler"]
    if mode == "per_species":
        # Each layer holds the particulates in the model's own units.
        part_scale = np.asarray(
            [0.001 if sp in KG_SPECIES else 1.0 for sp in settler.particulate_species]
        )
        scale = np.tile(part_scale, settler.n_layers)
    else:
        # Each layer holds the TSS, in g/m3 either way.
        scale = np.ones(settler.n_layers)
    y_k = np.asarray(y_g) * scale
    d_g = np.asarray(grams.derivative(y_g))
    d_k = np.asarray(kilos.derivative(y_k))
    assert np.any(d_g != 0.0)
    np.testing.assert_allclose(d_k, d_g * scale, rtol=1e-9, atol=1e-12)
    # The outlets carry the same solids, in each model's units.
    out_g = grams.outputs_at(0.0, y_g)
    out_k = kilos.outputs_at(0.0, y_k)
    for port in ("overflow", "underflow"):
        np.testing.assert_allclose(
            out_k[("settler", port)].C,
            _to_kg(asm1, asm1_kg, out_g[("settler", port)].C),
            rtol=1e-9,
            atol=1e-15,
        )


def test_settler_with_own_parameters_keeps_the_model_basis(asm1_kg):
    own = dict(v0=0.474, vmax=0.25, rh=0.576, rp=2.86, fns=2.28e-3, X_threshold=3.0)
    settler = TakacsClarifier(
        "s", asm1_kg, area=1500.0, height=4.0, underflow_Q=1.0e4, settling_params=own
    )
    default = TakacsClarifier("d", asm1_kg, area=1500.0, height=4.0, underflow_Q=1.0e4)
    factors = dict(zip(settler.particulate_species, settler._part_tss_factors, strict=True))
    converted = dict(zip(default.particulate_species, default._part_tss_factors, strict=True))
    assert factors["XS"] == pytest.approx(0.75)
    assert converted["XS"] == pytest.approx(750.0)


def test_settler_reports_solids_declared_in_moles(asm1):
    molar = dataclasses.replace(asm1, species_units={**asm1.species_units, "XB_H": "mol/m3"})
    settler = TakacsClarifier("s", molar, area=1500.0, height=4.0, underflow_Q=1.0e4)
    assert settler.assumed_units() == {"XB_H": "g/m3"}
    own = dict(v0=474.0, vmax=250.0, rh=5.7e-4, rp=2.86e-3, fns=2.28e-3, X_threshold=3000.0)
    mine = TakacsClarifier(
        "m", molar, area=1500.0, height=4.0, underflow_Q=1.0e4, settling_params=own
    )
    assert mine.assumed_units() == {}


# ----- the unit-scaling translator ------------------------------------------------


def test_unit_scaling_factors(asm1, asm1_kg):
    scale = np.asarray(unit_scaling(asm1_kg, asm1))
    for i, sp in enumerate(asm1.species):
        assert scale[i] == (1000.0 if sp in KG_SPECIES else 1.0)
    assert unit_scaling(asm1, asm1) is None
    # Spellings of one unit need no scaling.
    mg = dataclasses.replace(asm1, species_units={**asm1.species_units, "SS": "mg/L"})
    assert unit_scaling(mg, asm1) is None
    C = asm1.default_concentrations()
    assert UnitScalingTranslator(mg, asm1).translate(C) is C


def test_unit_scaling_refuses_what_it_cannot_match(asm1, tmp_path):
    molar = dataclasses.replace(asm1, species_units={**asm1.species_units, "SNH": "mol/m3"})
    with pytest.raises(ValueError, match="species 'SNH'"):
        unit_scaling(molar, asm1)
    blank = dataclasses.replace(asm1, species_units={**asm1.species_units, "SNH": ""})
    with pytest.raises(ValueError, match="cannot be read"):
        unit_scaling(blank, asm1)
    reordered = dataclasses.replace(asm1, species=list(reversed(asm1.species)))
    with pytest.raises(ValueError, match="same species in the same order"):
        unit_scaling(reordered, asm1)


def test_translator_coupling_is_diagonal(asm1, asm1_kg):
    pattern = UnitScalingTranslator(asm1_kg, asm1).coupling_pattern()
    assert np.array_equal(pattern, np.eye(asm1.n_species, dtype=bool))


# ----- the plant inserts it -----------------------------------------------------------


def _mixer(name, model):
    return MixerUnit(name=name, input_port_names=["a"], model=model)


def test_connect_converts_between_unit_bases(asm1, asm1_kg):
    plant = Plant("mixed_units", time_unit="d")
    plant.add_unit(_mixer("g_side", asm1))
    plant.add_unit(_mixer("kg_side", asm1_kg))
    plant.add_influent("feed", asm1.influent({"XS": 200.0, "SNH": 25.0}, Q=100.0), to="g_side")
    plant.connect("g_side", "kg_side")
    translator = plant.connections[-1].translator
    assert isinstance(translator, UnitScalingTranslator)
    out = plant.outputs_at(0.0, plant.initial_state())
    C_kg = np.asarray(out[("kg_side", "out")].C)
    assert C_kg[asm1.species_index["XS"]] == pytest.approx(0.2)
    assert C_kg[asm1.species_index["SNH"]] == pytest.approx(25.0)


def test_add_influent_converts_into_a_unit_on_another_basis(asm1, asm1_kg):
    plant = Plant("kg_feed", time_unit="d")
    plant.add_unit(_mixer("kg_side", asm1_kg))
    plant.add_influent("feed", asm1.influent({"XS": 200.0}, Q=100.0), to="kg_side")
    assert isinstance(plant.connections[-1].translator, UnitScalingTranslator)
    out = plant.outputs_at(0.0, plant.initial_state())
    assert float(out[("kg_side", "out")].C[asm1.species_index["XS"]]) == pytest.approx(0.2)


def test_connect_rejects_unconvertible_bases(asm1):
    molar = dataclasses.replace(
        asm1, name="asm1_molar", species_units={**asm1.species_units, "SNH": "mol/m3"}
    )
    plant = Plant("bad", time_unit="d")
    plant.add_unit(_mixer("g_side", asm1))
    plant.add_unit(_mixer("mol_side", molar))
    with pytest.raises(WiringError, match="cannot be converted"):
        plant.connect("g_side", "mol_side")


def test_identical_units_keep_the_instance_check(asm1):
    # Two instances of one model with identical units are still a mistake to
    # flag, not something to scale.
    twin = dataclasses.replace(asm1)
    plant = Plant("twins", time_unit="d")
    plant.add_unit(_mixer("a", asm1))
    plant.add_unit(_mixer("b", twin))
    with pytest.raises(ValueError, match="different \\*instances\\*"):
        plant.connect("a", "b")
    same = Plant("same", time_unit="d")
    same.add_unit(_mixer("a", asm1))
    same.add_unit(_mixer("b", asm1))
    same.connect("a", "b")
    assert isinstance(same.connections[-1].translator, IdentityTranslator)


def test_converted_connection_in_a_reactor_plant_solves(asm1, asm1_kg):
    # A g/m3 front end feeding a kg/m3 tank: the plant solves and the tank's
    # inflow is the front end's composition in kg.
    plant = Plant("front_end", time_unit="d")
    plant.add_unit(_mixer("front", asm1))
    plant.add_unit(
        CSTRUnit(
            "tank",
            asm1_kg,
            volume=10.0,
            input_port_names=["inlet"],
            conditions=asm1_kg.condition_defaults(),
        )
    )
    plant.add_influent("feed", asm1.influent({"SI": 30.0}, Q=1000.0), to="front")
    plant.connect("front", "tank")
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=np.array([1.0]))
    # SI is inert: after 100 residence times the tank holds the feed's 30 g/m3
    # as 0.03 kg/m3.
    assert sol.C_named("tank", "SI")[-1] == pytest.approx(0.03, rel=1e-6)


@pytest.mark.parametrize("recycle_tol", [1e-8, None])
def test_converted_connection_inside_a_recycle_loop(asm1, asm1_kg, recycle_tol):
    # A stateless loop (mixer in g/m3, splitter in kg/m3) is resolved exactly by
    # both recycle solvers: the loop's fixed point is the feed composition.
    from aquakin.plant import RatioSplitter

    plant = Plant("loop", time_unit="d", recycle_tol=recycle_tol)
    plant.add_unit(MixerUnit(name="mix", input_port_names=["feed", "rec"], model=asm1))
    plant.add_unit(
        RatioSplitter(name="split", model=asm1_kg, output_port_ratios={"out": 0.2, "rec": 0.8})
    )
    plant.add_influent("feed", asm1.influent({"SI": 30.0}, Q=100.0), to="mix.feed")
    plant.connect("mix", "split")
    plant.connect("split.rec", "mix.rec")
    out = plant.outputs_at(0.0, plant.initial_state())
    i = asm1.species_index["SI"]
    assert float(out[("mix", "out")].C[i]) == pytest.approx(30.0, rel=1e-10)
    assert float(out[("split", "out")].C[i]) == pytest.approx(0.03, rel=1e-10)
