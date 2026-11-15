"""Plant-level unit audit: Plant.check_units(), the solve-time time-unit guard,
the translator endpoint check in connect(), and declared influent units.

Everything here builds plants and inspects their declared units; nothing is
integrated, so the tests stay in the fast tier.
"""

import dataclasses

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import (
    ADM1toASM1,
    ASM1toADM1,
    CSTRUnit,
    InfluentSeries,
    Plant,
    PlantUnitsReport,
    UnitsError,
    UVUnit,
    WiringError,
)
from aquakin.plant.bsm import build_bsm1, build_bsm2
from aquakin.plant.influent import load_bsm1_influent, load_bsm2_influent
from aquakin.plant.takacs import TakacsClarifier


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


@pytest.fixture(scope="module")
def adm1():
    return aquakin.load_model("adm1")


@pytest.fixture(scope="module")
def ozone():
    return aquakin.load_model("ozone_bromate")


def _cstr(name, model, volume=1000.0):
    return CSTRUnit(
        name,
        model,
        volume=volume,
        input_port_names=["inlet"],
        conditions=model.condition_defaults(),
    )


def _with_units(model, **species_units):
    """A copy of ``model`` that declares different units for some species."""
    return dataclasses.replace(model, species_units={**model.species_units, **species_units})


def _errors(plant, **kw):
    return [str(i) for i in plant.check_units(**kw).errors]


# ----- the shipped builders are clean --------------------------------------


@pytest.mark.parametrize("builder", [build_bsm1, build_bsm2])
def test_bsm_builders_audit_clean(builder):
    report = builder().check_units()
    assert isinstance(report, PlantUnitsReport)
    assert report.ok and report.errors == []
    assert report.time_unit == "d"
    assert "OK" in report.summary()


def test_bsm2_with_shipped_influent_audits_clean():
    plant = build_bsm2()
    asm_model = plant.units[plant.influent_endpoint.split(".")[0]].model
    plant.add_influent("feed", load_bsm2_influent("dry", asm_model))
    report = plant.check_units()
    assert report.ok, report.summary()
    assert report.warnings == []


def test_bsm_influent_loaders_suit_a_subset_model(tmp_path):
    # A model using only some of the file's columns gets only its own entries.
    path = tmp_path / "sub.yaml"
    path.write_text(
        """
model:
  name: sub
  description: "two ASM1 species"
species:
  - {name: SS, units: g_COD/m3, default_concentration: 1.0}
  - {name: SNH, units: g_N/m3, default_concentration: 1.0}
conditions: []
reactions:
  - name: R
    rate: "k * [SS]"
    parameters: {k: {value: 0.1, units: "1/d"}}
    stoichiometry: {SS: -1}
"""
    )
    sub = aquakin.load_model_from_file(str(path))
    for loader, extra in ((load_bsm1_influent, set()), (load_bsm2_influent, {"T"})):
        series = loader("dry", sub)
        assert set(series.units) == {"t", "Q", "SS", "SNH"} | extra


def test_bsm_influent_loaders_declare_units(asm1):
    bsm1 = load_bsm1_influent("dry", asm1)
    assert bsm1.units["t"] == "d" and bsm1.units["Q"] == "m3/d"
    assert bsm1.units["SNH"] == "g_N/m3"
    bsm2 = load_bsm2_influent("dry", asm1)
    assert bsm2.units["T"] == "K"


# ----- time unit ---------------------------------------------------------------


def test_models_on_different_time_units_is_an_error(asm1, ozone):
    plant = Plant("mixed_clock")
    plant.add_unit(_cstr("a", asm1))
    plant.add_unit(_cstr("o", ozone, volume=10.0))
    plant.add_influent("fa", asm1.influent(Q=100.0), to="a")
    plant.add_influent("fo", ozone.influent(Q=1.0), to="o")
    report = plant.check_units()
    assert not report.ok and report.time_unit is None
    (err,) = report.errors
    assert "'asm1' in 1/d" in err.detail and "'ozone_bromate' in 1/s" in err.detail
    # check() carries it too, and raise_on_error raises the typed error.
    chk = plant.check()
    assert not chk.ok and chk.unit_errors
    assert "units (error): [plant]" in chk.summary()
    with pytest.raises(UnitsError):
        plant.check(raise_on_error=True)
    # Solving is refused outright: no result from a mixed clock is meaningful.
    with pytest.raises(UnitsError, match="disagree on the time unit"):
        plant.solve(t_span=(0.0, 1.0))


def test_unit_assuming_days_in_a_seconds_plant(ozone):
    plant = Plant("uv_seconds")
    plant.add_unit(_cstr("o", ozone, volume=10.0))
    plant.add_unit(UVUnit("uv", ozone, volume=1.0, intensity=10.0, d10=5.0))
    plant.add_influent("f", ozone.influent(Q=1.0), to="o")
    plant.connect("o", "uv")
    assert plant.check_units().time_unit == "s"
    assert any("unit 'uv'" in e and "'d'" in e for e in _errors(plant))
    with pytest.raises(UnitsError, match="assume time in 'd'"):
        plant.solve(t_span=(0.0, 1.0))


def test_time_guard_does_not_block_a_consistent_plant(make_cstr_plant):
    # The guard runs in every solve's setup; a consistent plant passes it.
    plant = make_cstr_plant()
    plant._build_state_layout()
    assert plant.check_units().ok


# ----- unit operations' fixed concentration basis ------------------------------


def test_settler_flags_particulates_in_kg_per_m3(asm1):
    kg = _with_units(asm1, XS="kgCOD/m3")
    plant = Plant("settler")
    plant.add_unit(_cstr("tank", kg))
    plant.add_unit(TakacsClarifier("settler", kg, area=1500.0, height=4.0, underflow_Q=1.0e4))
    plant.connect("tank", "settler")
    errs = _errors(plant, include_models=False)
    assert len(errs) == 1
    assert "unit 'settler'" in errs[0] and "'XS'" in errs[0] and "kgCOD/m3" in errs[0]


def _in_hours(model):
    """A copy of ``model`` whose rate constants are declared per hour."""
    import re

    per_hour = {
        k: re.sub(r"(?<![A-Za-z_])d(?![A-Za-z_0-9])", "h", u)
        for k, u in model.parameter_units.items()
    }
    return dataclasses.replace(model, parameter_units=per_hour)


def test_settler_with_own_parameters_assumes_nothing(asm1):
    # Settling parameters the user supplies are in the plant's units, so a
    # settler in an hourly plant is fine; only the BSM defaults fix m/d.
    hourly = _in_hours(asm1)
    assert hourly.time_unit == "h"
    custom = dict(v0=19.75, vmax=10.4, rh=5.76e-4, rp=2.86e-3, fns=2.28e-3, X_threshold=3000.0)
    plant = Plant("hourly")
    plant.add_unit(_cstr("tank", hourly))
    plant.add_unit(
        TakacsClarifier(
            "settler", hourly, area=1500.0, height=4.0, underflow_Q=400.0, settling_params=custom
        )
    )
    plant.connect("tank", "settler")
    assert plant.units["settler"].assumed_units() == {}
    assert plant.check_units(include_models=False).ok
    plant._build_state_layout()  # the solve-time guard lets it through

    defaults = Plant("hourly_defaults")
    defaults.add_unit(_cstr("tank", hourly))
    defaults.add_unit(TakacsClarifier("settler", hourly, area=1500.0, height=4.0, underflow_Q=400.0))
    defaults.connect("tank", "settler")
    with pytest.raises(UnitsError, match="assume time in 'd'"):
        defaults._build_state_layout()


def test_settler_ignores_species_outside_tss(asm1):
    # XND carries no TSS weight, so its units do not enter the settling
    # function and are not constrained.
    model = _with_units(asm1, XND="kmolN/m3")
    plant = Plant("settler_xnd")
    plant.add_unit(_cstr("tank", model))
    plant.add_unit(TakacsClarifier("settler", model, area=1500.0, height=4.0, underflow_Q=1.0e4))
    plant.connect("tank", "settler")
    assert "XND" not in plant.units["settler"].assumed_units()
    assert plant.check_units(include_models=False).ok


# ----- translators -----------------------------------------------------------------


def _adm1_in_grams(adm1):
    """ADM1 declared on a g/m3 and mol/m3 basis instead of kg/m3 and kmol/m3."""
    regrams = {
        sp: ("g_COD/m3" if "COD" in u else u.replace("kmol", "mol"))
        for sp, u in adm1.species_units.items()
    }
    return _with_units(adm1, **regrams)


def _asm1_in_kilograms(asm1):
    """ASM1 declared on a kg/m3 and kmol/m3 basis instead of g/m3 and mol/m3."""
    rekilos = {
        sp: (u.replace("g_", "kg_") if u.startswith("g_") else u.replace("mol", "kmol"))
        for sp, u in asm1.species_units.items()
    }
    return _with_units(asm1, **rekilos)


def test_translators_convert_the_asm1_side_too(asm1, adm1):
    asm1_kg = _asm1_in_kilograms(asm1)
    to_adm = ASM1toADM1(asm1_kg, adm1)
    C_g = asm1.default_concentrations()
    C_kg = C_g / to_adm._source_scale        # the same state, in kilograms
    assert jnp.allclose(
        to_adm.translate(C_kg, digester_pH=7.0),
        ASM1toADM1(asm1, adm1).translate(C_g, digester_pH=7.0),
        rtol=1e-12,
        atol=0.0,
    )
    to_asm = ADM1toASM1(adm1, asm1_kg)
    C_adm = adm1.default_concentrations()
    assert jnp.allclose(
        to_asm.translate(C_adm, digester_pH=7.0),
        ADM1toASM1(adm1, asm1).translate(C_adm, digester_pH=7.0) / 1000.0,
        rtol=1e-12,
        atol=1e-15,
    )


def test_translators_on_shipped_models_need_no_scaling(asm1, adm1):
    # Every shipped species unit is already the interface's working unit, so no
    # scale vector is built and translate runs the exact historic arithmetic.
    for tr in (ASM1toADM1(asm1, adm1), ADM1toASM1(adm1, asm1)):
        assert tr._source_scale is None and tr._target_scale is None


def test_asm1_to_adm1_converts_to_the_target_models_units(asm1, adm1):
    adm1_g = _adm1_in_grams(adm1)
    C = asm1.default_concentrations()
    kg = ASM1toADM1(asm1, adm1).translate(C, digester_pH=7.0)
    g = ASM1toADM1(asm1, adm1_g).translate(C, digester_pH=7.0)
    assert jnp.allclose(g, 1000.0 * kg, rtol=1e-12, atol=0.0)
    assert float(jnp.max(jnp.abs(kg))) > 0.0


def test_adm1_to_asm1_converts_from_the_source_models_units(asm1, adm1):
    adm1_g = _adm1_in_grams(adm1)
    tr_g = ADM1toASM1(adm1_g, asm1)
    C_kg = adm1.default_concentrations()
    C_g = C_kg / tr_g._source_scale          # the same state, in grams
    expected = ADM1toASM1(adm1, asm1).translate(C_kg, digester_pH=7.0)
    got = tr_g.translate(C_g, digester_pH=7.0)
    assert jnp.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_translator_rejects_a_unit_it_cannot_convert(asm1, adm1):
    # Inorganic N as a mass needs a molar mass, not a unit conversion.
    with pytest.raises(ValueError, match="'S_IN'.*cannot be converted"):
        ASM1toADM1(asm1, _with_units(adm1, S_IN="g_N/m3"))
    # A model left at the schema's default species units (mol/L) claims an
    # amount of substance where the ASM1 side works in g_COD/m3.
    with pytest.raises(ValueError, match="cannot be converted"):
        ASM1toADM1(_with_units(asm1, SS="mol/L"), adm1)


def test_translator_warns_and_assumes_working_unit_when_unreadable(asm1, adm1):
    with pytest.warns(UserWarning, match="cannot read the units 'furlongs'.*assuming 'g_N/m3'"):
        tr = ADM1toASM1(adm1, _with_units(asm1, SNH="furlongs"))
    assert tr._target_scale is None


def test_translated_plant_in_other_units_audits_clean(asm1, adm1):
    adm1_g = _adm1_in_grams(adm1)
    plant = Plant("interface")
    plant.add_unit(_cstr("a", asm1))
    plant.add_unit(_cstr("d", adm1_g))
    plant.connect("a", "d", translator=ASM1toADM1(asm1, adm1_g))
    assert plant.check_units(include_models=False).ok


def test_connect_rejects_translator_built_for_other_models(asm1, adm1):
    plant = Plant("wrong_translator")
    plant.add_unit(_cstr("a", asm1))
    plant.add_unit(_cstr("b", asm1))
    with pytest.raises(WiringError, match="translator's target model is 'adm1'"):
        plant.connect("a", "b", translator=ASM1toADM1(asm1, adm1))


def test_connect_rejects_translator_built_for_models_in_other_units(asm1, adm1):
    # The translator converts from the units of the models it was built with,
    # so wiring a kg/m3 translator to a g/m3 digester would feed it values 1000x
    # too small.
    plant = Plant("stale_translator")
    plant.add_unit(_cstr("a", asm1))
    plant.add_unit(_cstr("d", _adm1_in_grams(adm1)))
    with pytest.raises(WiringError, match="built for a target model declaring 'S_su'"):
        plant.connect("a", "d", translator=ASM1toADM1(asm1, adm1))


def test_connect_accepts_translator_on_same_species_instance(asm1, adm1):
    # A separately loaded copy of the same model has the same species list, so
    # the translator still indexes correctly and is accepted.
    plant = Plant("same_species")
    plant.add_unit(_cstr("a", asm1))
    plant.add_unit(_cstr("d", adm1))
    other_asm1 = dataclasses.replace(asm1)
    plant.connect("a", "d", translator=ASM1toADM1(other_asm1, adm1))


# ----- influent declarations -----------------------------------------------------


def _single_tank(asm1, influent):
    plant = Plant("tank")
    plant.add_unit(_cstr("tank", asm1))
    plant.add_influent("feed", influent, to="tank")
    return plant


def test_influent_declared_units_that_match_are_clean(asm1):
    feed = asm1.influent(
        {"SS": 60.0}, Q=18446.0, T=293.15, units={"t": "d", "Q": "m3/d", "T": "K", "SS": "mg/L"}
    )
    assert _single_tank(asm1, feed).check_units().ok


@pytest.mark.parametrize(
    "units, fragment",
    [
        ({"Q": "m3/h"}, "flow 'Q' in 'm3/h'"),
        ({"t": "h"}, "time 't' in 'h'"),
        ({"T": "degC"}, "Kelvin"),
        ({"T": "℃"}, "Kelvin"),
        ({"SS": "kgCOD/m3"}, "species 'SS' in 'kgCOD/m3'"),
        ({"SNH": "g_COD/m3"}, "species 'SNH'"),
    ],
)
def test_influent_declared_units_that_disagree(asm1, units, fragment):
    feed = asm1.influent({"SS": 60.0}, Q=18446.0, units=units)
    errs = _errors(_single_tank(asm1, feed), include_models=False)
    assert len(errs) == 1 and fragment in errs[0], errs


def test_influent_units_reject_unreadable_units(asm1):
    with pytest.raises(ValueError, match="cannot be read: t='hours'"):
        asm1.influent({"SS": 60.0}, Q=1.0, units={"t": "hours"})


def test_influent_units_reject_unknown_keys(asm1):
    with pytest.raises(ValueError, match="not 't', 'Q', 'T' or a species"):
        asm1.influent({"SS": 60.0}, Q=1.0, units={"flow": "m3/d"})
    with pytest.raises(ValueError, match="must be strings"):
        asm1.influent({"SS": 60.0}, Q=1.0, units={"Q": 24})


def test_read_influent_csv_declares_units(tmp_path, asm1):
    from aquakin.plant import read_influent_csv

    path = tmp_path / "feed.csv"
    path.write_text("t,Q,SS\n0,100,60\n1,100,60\n")
    series = read_influent_csv(
        path, asm1, column_map={"t": "t", "Q": "Q", "SS": "SS"}, units={"Q": "L/s"}
    )
    assert series.units == {"Q": "L/s"}
    errs = _errors(_single_tank(asm1, series), include_models=False)
    assert len(errs) == 1 and "L/s" in errs[0]


# ----- model roll-up ------------------------------------------------------------


def test_model_findings_are_warnings_not_errors(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(
        """
model:
  name: broken
  description: "a cross-currency sum"
species:
  - {name: A, units: g_COD/m3, default_concentration: 1.0}
  - {name: B, units: g_N/m3, default_concentration: 1.0}
conditions: []
reactions:
  - name: R
    rate: "k * ([A] + [B])"
    parameters: {k: {value: 0.1, units: "1/d"}}
    stoichiometry: {A: -1, B: 1}
"""
    )
    net = aquakin.load_model_from_file(str(path))
    plant = _single_tank(net, net.influent(Q=1.0))
    report = plant.check_units()
    assert report.ok
    assert any("model 'broken'" in str(w) for w in report.warnings)
    assert plant.check_units(include_models=False).warnings == []
    # check() lists the warnings without failing on them.
    chk = plant.check()
    assert chk.ok
    assert "units (warning): [model 'broken']" in chk.summary()


def test_model_without_a_time_unit_is_a_warning(tmp_path):
    # No rate constant declares units, so the time unit cannot be inferred and
    # the plant cannot confirm its clock; that is reported, not raised.
    path = tmp_path / "no_units.yaml"
    path.write_text(
        """
model:
  name: no_units
  description: "rate constants without units"
species:
  - {name: A, units: g_COD/m3, default_concentration: 1.0}
conditions: []
reactions:
  - name: R
    rate: "k * [A]"
    parameters: {k: {value: 0.1}}
    stoichiometry: {A: -1}
"""
    )
    net = aquakin.load_model_from_file(str(path))
    assert net.time_unit is None
    report = _single_tank(net, net.influent(Q=1.0)).check_units()
    assert report.ok and report.time_unit is None
    assert any("time unit cannot be inferred" in w.detail for w in report.warnings)
