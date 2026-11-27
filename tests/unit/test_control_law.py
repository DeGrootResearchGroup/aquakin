"""The expression controller: its law's value, its errors, and the cascade order.

Each value is checked against a hand calculation from the stream it reads; each
error branch by its message. The plant-level tests build plants (no solve) to
check the producer order and the build-time checks; the solves are in
``tests/integration/test_expression_controller.py``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import ExpressionController, PIController, Plant
from aquakin.plant.control_law import CHANNELS, ControlLawError, compile_law
from aquakin.plant.errors import WiringError
from aquakin.plant.signal_bus import order_signal_producers
from aquakin.plant.streams import Stream

ASM1_TSS = ("XS", "XI", "XB_H", "XB_A", "XP")


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


def _stream(model, **conc):
    C = model.concentrations(conc, base="zero")
    return {"measured": Stream(Q=jnp.zeros(()), C=C, model=model)}


def _value(ctrl, inputs=None, t=0.0, signals=None, params=None):
    model = ctrl.model
    p = model.default_parameters() if params is None and model is not None else params
    out = ctrl.signal_outputs(jnp.asarray(t), jnp.zeros(0), inputs or {}, p, signals)
    return float(out[ctrl.signal_name])


def _ctrl(law, model=None, constants=None, name="c", signal="u"):
    return ExpressionController(
        name, model=model, signal_name=signal, law=law, constants=constants or {}
    )


# --- the law's value ---------------------------------------------------------


def test_a_species_and_constants_give_the_hand_computed_value(asm1):
    ctrl = _ctrl("offset + Kp*([SNH] - sp)", asm1, {"Kp": 40.0, "sp": 1.0, "offset": 120.0})
    assert _value(ctrl, _stream(asm1, SNH=2.5)) == pytest.approx(120.0 + 40.0 * 1.5, rel=1e-14)


def test_clip_bounds_the_output(asm1):
    ctrl = _ctrl("clip(Kp*[SNH], 0, 240)", asm1, {"Kp": 40.0})
    assert _value(ctrl, _stream(asm1, SNH=3.0)) == pytest.approx(120.0, rel=1e-14)
    assert _value(ctrl, _stream(asm1, SNH=10.0)) == 240.0
    assert _value(ctrl, _stream(asm1, SNH=-1.0)) == 0.0


def test_tss_is_the_benchmark_sum_on_asm1(asm1):
    conc = {"XS": 50.0, "XI": 20.0, "XB_H": 100.0, "XB_A": 7.0, "XP": 3.0, "SS": 99.0}
    ctrl = _ctrl("TSS", asm1)
    expected = 0.75 * sum(conc[s] for s in ASM1_TSS)
    assert _value(ctrl, _stream(asm1, **conc)) == pytest.approx(expected, rel=1e-12)


def test_tn_follows_the_composition_parameter_of_the_run(asm1):
    """TN on ASM1 is SNH + SNO + SND + XND + i_XB (XB_H + XB_A) + i_XP (XP + XI);
    a changed ``i_XB`` in the parameters the plant hands the unit moves it."""
    conc = {"SNH": 2.0, "SNO": 5.0, "SND": 1.0, "XND": 3.0, "XB_H": 100.0, "XB_A": 10.0}
    conc |= {"XP": 30.0, "XI": 40.0}
    names = list(asm1.parameters)
    i_xb = next(n for n in names if n.endswith("i_XB"))
    i_xp = next(n for n in names if n.endswith("i_XP"))
    p = np.asarray(asm1.default_parameters(), float).copy()
    p[asm1.param_index[i_xb]] = 0.12
    ctrl = _ctrl("TN", asm1)

    def hand(params):
        ixb, ixp = params[asm1.param_index[i_xb]], params[asm1.param_index[i_xp]]
        return (
            conc["SNH"] + conc["SNO"] + conc["SND"] + conc["XND"]
            + ixb * (conc["XB_H"] + conc["XB_A"]) + ixp * (conc["XP"] + conc["XI"])
        )

    default = np.asarray(asm1.default_parameters(), float)
    assert _value(ctrl, _stream(asm1, **conc)) == pytest.approx(hand(default), rel=1e-10)
    moved = _value(ctrl, _stream(asm1, **conc), params=jnp.asarray(p))
    assert moved == pytest.approx(hand(p), rel=1e-10)
    assert moved != pytest.approx(hand(default), rel=1e-6)


def test_a_name_read_twice_is_one_read(asm1):
    """A channel or signal used twice is read once and counted once."""
    conc = {"XS": 40.0}
    tss = _ctrl("TSS - 0.5*TSS", asm1)
    assert tss._law.channels == ("TSS",)
    assert _value(tss, _stream(asm1, **conc)) == pytest.approx(0.5 * 0.75 * 40.0, rel=1e-14)
    sig = _ctrl("a*a + a")
    assert sig.required_signals == ("a",)
    assert _value(sig, signals={"a": 3.0}) == 12.0


def test_a_closure_language_must_say_what_a_name_is():
    from aquakin.core.time_expr import ClosureParser

    with pytest.raises(NotImplementedError, match="ClosureParser does not say what a name reads"):
        ClosureParser("x").parse()


def test_the_clock_pi_and_e():
    ctrl = _ctrl("2*t + pi - e")
    assert _value(ctrl, t=3.0) == pytest.approx(6.0 + np.pi - np.e, rel=1e-14)


def test_a_signal_is_read_from_the_bus_so_far():
    ctrl = _ctrl("0.5*kla_tank3 + 1")
    assert ctrl.required_signals == ("kla_tank3",)
    assert _value(ctrl, signals={"kla_tank3": 100.0}) == 51.0


def test_a_law_compiles_to_a_traceable_differentiable_function(asm1):
    """d/d[SNH] of clip(Kp*([SNH] - sp), 0, 240) is Kp inside the bounds, 0 outside."""
    law = compile_law("clip(Kp*([SNH] - sp), 0, 240)", {"Kp": 40.0, "sp": 1.0}, asm1, what="x")
    i = asm1.species_index["SNH"]

    def f(snh):
        C = jnp.zeros(asm1.n_species).at[i].set(snh)
        return law(0.0, C, {}, asm1.default_parameters())

    import jax

    assert float(jax.jit(jax.grad(f))(2.0)) == 40.0
    assert float(jax.grad(f)(20.0)) == 0.0


def test_the_pi_controller_ignores_the_bus(asm1):
    pi = PIController("pi", asm1, "SO", 2.0, 25.0, 0.002, 0.001, 120.0, 0.0, 360.0, "kla")
    inputs = _stream(asm1, SO=1.0)
    alone = pi.signal_outputs(0.0, pi.initial_state(), inputs, None)
    with_bus = pi.signal_outputs(0.0, pi.initial_state(), inputs, None, {"other": 5.0})
    assert float(alone["kla"]) == float(with_bus["kla"]) == 145.0


def test_the_unit_is_stateless_with_one_port_only_when_sensing(asm1):
    sensing = _ctrl("[SNH]", asm1)
    blind = _ctrl("1")
    assert (sensing.state_size, sensing.state_names(), sensing.output_ports) == (0, (), [])
    assert sensing.input_ports == ["measured"] and blind.input_ports == []
    assert sensing.compute_outputs(0.0, None, {}, None) == {}
    assert sensing.flow_outputs({}, None) == {}


# --- errors at construction ----------------------------------------------------


@pytest.mark.parametrize(
    ("law", "match"),
    [
        ("Kp*", "expected a number, a name, '\\[species\\]' or '\\('"),
        ("", "the expression is empty"),
        ("foo(1)", "'foo' at position 0 is not a function a control law can call"),
        ("clip(1, 2)", "clip\\(\\) takes 3 argument"),
        ("1 2", "expected an operator"),
    ],
)
def test_an_unreadable_law_names_the_controller(law, match):
    with pytest.raises(ControlLawError, match=f"ExpressionController 'c': {match}"):
        _ctrl(law)


@pytest.mark.parametrize("name", ["t", "pi", "e", "clip", "TSS"])
def test_a_constant_cannot_take_a_name_the_law_reserves(name):
    with pytest.raises(ControlLawError, match=f"the constant '{name}' takes a name"):
        _ctrl("1", constants={name: 1.0})


def test_a_species_not_in_the_model_gets_a_hint(asm1):
    with pytest.raises(ControlLawError, match=r"'\[SNHH\]' at position 0 is not a species.*SNH"):
        _ctrl("[SNHH]", asm1)


@pytest.mark.parametrize("law", ["[SNH] + 1", "2*TSS"])
def test_a_law_reading_the_stream_needs_a_model(law):
    with pytest.raises(ControlLawError, match=r"senses no stream\. Give it model="):
        _ctrl(law)


def test_a_channel_the_model_cannot_give_is_refused(simple_model):
    with pytest.raises(ControlLawError, match="has no TSS species"):
        _ctrl("TSS", simple_model)


def test_the_published_signal_cannot_be_a_channel_name():
    with pytest.raises(ControlLawError, match="signal_name 'TN' is the name of a measurement"):
        _ctrl("1", signal="TN")


def test_channels_exclude_flow_and_temperature():
    assert set(CHANNELS) == {"TSS", "COD", "TN", "TKN", "NOx", "TP"}


# --- errors when evaluated ------------------------------------------------------


def test_an_unconnected_measured_port_is_a_wiring_error(asm1):
    with pytest.raises(WiringError, match="nothing is connected to its 'measured' port"):
        _value(_ctrl("[SNH]", asm1, name="loop"))


def test_a_sensed_stream_in_another_species_order_is_refused(asm1):
    asm3 = aquakin.load_model("asm3")
    bad = {"measured": Stream(Q=jnp.zeros(()), C=asm3.default_concentrations(), model=asm3)}
    with pytest.raises(ValueError, match="would read the wrong species"):
        _value(_ctrl("[SNH]", asm1), bad)


# --- the names a law reads, checked against the plant ---------------------------


@pytest.mark.parametrize(
    ("law", "constants", "published", "match"),
    [
        ("Kpp*2", {"Kp": 1.0}, {}, "reads 'Kpp', which is not one of its constants.*Did you mean: Kp"),
        ("kla_tank*2", {}, {"kla_tank3": "a"}, "Did you mean: kla_tank3"),
        ("TSS", {}, {"TSS": "pi"}, "'TSS' is both a measurement channel of its law and a signal"),
        ("Kp", {"Kp": 1.0}, {"Kp": "pi"}, "'Kp' is both a constant of its law and a signal 'pi'"),
    ],
)
def test_check_signals_names_what_the_law_cannot_read(asm1, law, constants, published, match):
    with pytest.raises(ControlLawError, match=match):
        _ctrl(law, asm1, constants).check_signals(published)


def test_check_signals_accepts_a_published_signal():
    _ctrl("2*kla").check_signals({"kla": "pi"})  # the property: no error


def test_a_plant_build_reports_an_unknown_name_with_its_hint():
    plant = Plant("p", time_unit="d")
    plant.add_unit(_ctrl("2*speed", name="a", signal="sped"))
    with pytest.raises(ControlLawError, match=r"reads 'speed'.*Did you mean: sped"):
        plant.signals_at(0.0, jnp.zeros(0))


# --- the cascade order -----------------------------------------------------------


def test_a_consumer_listed_first_sees_its_producers_value():
    plant = Plant("cascade", time_unit="d")
    plant.add_unit(_ctrl("demand + 1", name="follow", signal="kla"))
    plant.add_unit(_ctrl("10*t", name="lead", signal="demand"))
    signals = plant.signals_at(2.0, jnp.zeros(0))
    assert float(signals["demand"]) == 20.0
    assert float(signals["kla"]) == 21.0
    assert plant._signal_order == ("lead", "follow")


def test_the_order_keeps_the_unit_order_where_the_reads_allow():
    units = {
        "a": _ctrl("c + 1", name="a", signal="a_out"),
        "b": _ctrl("1", name="b", signal="b_out"),
        "c": _ctrl("1", name="c", signal="c"),
    }
    assert order_signal_producers(units, ["a", "b", "c"]) == ("b", "c", "a")


def test_a_cycle_raises_naming_it():
    plant = Plant("loop", time_unit="d")
    plant.add_unit(_ctrl("y + 1", name="first", signal="x"))
    plant.add_unit(_ctrl("x + 1", name="second", signal="y"))
    with pytest.raises(
        ValueError,
        match="in a cycle: first -> second -> first \\('first' reads 'y' from 'second', "
        "'second' reads 'x' from 'first'\\)",
    ):
        plant.signals_at(0.0, jnp.zeros(0))


def test_a_law_reading_its_own_signal_is_a_cycle():
    with pytest.raises(ValueError, match="in a cycle: self -> self"):
        order_signal_producers({"self": _ctrl("u + 1", name="self", signal="u")}, ["self"])


# --- the units that read a signal they are told the name of ----------------------


def test_an_aeration_kla_signal_sets_the_kla_times_gain(asm1):
    """The clean-water kLa is signal * gain on oxygen and ratio * that on a
    co-stripped gas; the process-water kLa is that times ``alpha``."""
    from aquakin.plant import Aeration, CSTRUnit

    aer = Aeration(kla_signal="kla3", gain=0.5, alpha=0.4, stripped={"SNO": 0.8})
    tank = CSTRUnit("tank3", asm1, 1000.0, ["in"], aeration=aer, conditions={"T": 293.15})
    assert tank.required_signals == ("kla3",)
    kla = np.asarray(tank.kla_vector({"kla3": 100.0}))
    assert kla[asm1.species_index["SO"]] == 50.0
    assert kla[asm1.species_index["SNO"]] == pytest.approx(40.0, rel=1e-14)
    assert np.count_nonzero(kla) == 2
    process = np.asarray(tank.kla_vector({"kla3": 100.0}, process_water=True))
    np.testing.assert_allclose(process, 0.4 * kla, rtol=1e-15)


def test_an_ifas_unit_reads_its_kla_signal(asm1):
    from aquakin.plant.catalog import build_unit

    settings = {
        "model": "asm1",
        "volume": 1000.0,
        "conditions": {"T": 293.15},
        "specific_surface_area": 500.0,
        "fill_fraction": 0.5,
        "biofilm_thickness": 1e-4,
        "aeration": {"kla_signal": "kla_ifas"},
    }
    ifas = build_unit("ifas", "ifas", settings)
    assert ifas.required_signals == ("kla_ifas",)
    assert float(ifas.kla_vector({"kla_ifas": 90.0})[asm1.species_index["SO"]]) == 90.0


def test_an_ifas_unit_applies_alpha_to_its_kla_signal(asm1):
    """The IFAS bulk shares the CSTR aeration term: its oxygen transfer is
    ``alpha * signal * (do_sat - SO)``."""
    from aquakin.plant.catalog import build_unit

    settings = {
        "model": "asm1",
        "volume": 1000.0,
        "conditions": {"T": 293.15},
        "specific_surface_area": 500.0,
        "fill_fraction": 0.5,
        "biofilm_thickness": 1e-4,
        "aeration": {"kla_signal": "kla_ifas", "alpha": 0.5},
    }
    ifas = build_unit("ifas", "ifas", settings)
    C = asm1.default_concentrations()
    inputs = {"in": Stream(Q=jnp.asarray(100.0), C=C, model=asm1)}
    state = ifas.initial_state()
    oxygen, _stripped = ifas.oxygen_transfer(
        0.0, state, inputs, asm1.default_parameters(), {"kla_ifas": 90.0}
    )
    so = asm1.species_index["SO"]
    assert float(oxygen[so]) == pytest.approx(0.5 * 90.0 * (8.0 - float(state[so])), rel=1e-12)


@pytest.mark.parametrize(
    ("modes", "match"),
    [
        ({}, "exactly one of kla= \\(open loop\\), do_setpoint=.*none was given"),
        ({"kla": 1.0, "kla_signal": "k"}, "got kla and kla_signal"),
        ({"do_setpoint": 2.0, "kla_signal": "k"}, "got do_setpoint and kla_signal"),
    ],
)
def test_an_aeration_needs_exactly_one_mode(modes, match):
    from aquakin.plant import Aeration

    with pytest.raises(ValueError, match=match):
        Aeration(**modes)


@pytest.mark.parametrize(
    ("modes", "match"),
    [
        ({}, "exactly one of flow= \\(fixed\\).*none was given"),
        ({"flow": 1.0, "flow_signal": "q"}, "got flow and flow_signal"),
        ({"setpoint": 1.0, "flow_signal": "q"}, "got setpoint and flow_signal"),
    ],
)
def test_a_dosing_unit_needs_exactly_one_mode(asm1, modes, match):
    from aquakin.plant import DosingUnit, Reagent

    with pytest.raises(ValueError, match=match):
        DosingUnit("d", Reagent.from_species(asm1, SS=4e5), **modes)


def test_a_signal_driven_dose_reads_its_signal(asm1):
    from aquakin.plant import DosingUnit, Reagent

    dose = DosingUnit("d", Reagent.from_species(asm1, SS=4e5), flow_signal="q", gain=0.5)
    assert dose.required_signals == ("q",)
    feed = Stream(Q=jnp.asarray(98.0), C=asm1.concentrations({"SS": 10.0}, base="zero"), model=asm1)
    out = dose.compute_outputs(0.0, jnp.zeros(0), {"in": feed}, None, {"q": 4.0})["out"]
    assert float(out.Q) == 100.0
    assert float(out.C[asm1.species_index["SS"]]) == pytest.approx(
        (98.0 * 10.0 + 2.0 * 4e5) / 100.0, rel=1e-12
    )
    with pytest.raises(ValueError, match="reads its dose flow from a signal but no"):
        dose.compute_outputs(0.0, jnp.zeros(0), {"in": feed}, None)


@pytest.mark.parametrize(
    ("settings", "inputs"),
    [
        ({"model": "asm1", "signal_name": "u", "law": "[SNH]"}, ["measured"]),
        ({"signal_name": "u", "law": "2*t"}, []),
    ],
)
def test_the_catalogue_gives_the_measured_port_only_with_a_model(settings, inputs):
    from aquakin.plant.catalog import check_unit, unit_kind

    built = check_unit("expression_controller", "c", settings)
    assert built["ok"] and built["inputs"] == inputs
    assert unit_kind("expression_controller").ports_for(settings) == (inputs, [])
