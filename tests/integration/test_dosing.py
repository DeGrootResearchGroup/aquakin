"""Chemical dosing: the Reagent value object and the DosingUnit (issue #278).

Covers the reagent composition builder, the dosing-unit spec validation, the
fixed-flow mass balance, and feedback dosing -- the plant auto-wires a PI
controller (the aeration loop's controller, reused) that manipulates the dose
flow to hold a setpoint, with the dose-flow signal computed before the stream
sweep so the dosed output stream can read it.
"""

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import DosingUnit, Reagent
from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.plant import Plant
from aquakin.plant.streams import Stream


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


def _tank(asm1, name="tank"):
    return CSTRUnit(name=name, model=asm1, volume=1000.0,
                    input_port_names=["in"], conditions={"T": 293.15})


# --- Reagent ---------------------------------------------------------------

def test_reagent_from_species_is_base_zero(asm1):
    r = Reagent.from_species(asm1, SS=4e5, label="methanol")
    ss = asm1.species_index["SS"]
    assert r.label == "methanol"
    assert float(r.composition[ss]) == 4e5
    # everything not named is zero (a neat reagent, not the model defaults)
    assert float(jnp.sum(r.composition)) == 4e5


# --- spec validation -------------------------------------------------------

def test_requires_exactly_one_mode(asm1):
    r = Reagent.from_species(asm1, SS=4e5)
    with pytest.raises(ValueError, match="exactly one"):
        DosingUnit("d", r)                                   # neither
    with pytest.raises(ValueError, match="exactly one"):
        DosingUnit("d", r, flow=2.0, setpoint=1.0)           # both
    with pytest.raises(ValueError, match="flow must be"):
        DosingUnit("d", r, flow=-1.0)


def test_feedback_needs_sensor_and_species(asm1):
    r = Reagent.from_species(asm1, SS=4e5)
    with pytest.raises(ValueError, match="measured_species= and sensor="):
        DosingUnit("d", r, setpoint=1.0)                     # no sensor/species
    with pytest.raises(ValueError, match="not in the reagent's model"):
        DosingUnit("d", r, setpoint=1.0, measured_species="NOPE", sensor="tank")


def test_fixed_dose_requires_nothing_else(asm1):
    d = DosingUnit("d", Reagent.from_species(asm1, SS=4e5), flow=2.0)
    assert d.state_size == 0
    assert d.required_signals == ()
    assert not d.is_closed_loop


# --- fixed-flow behaviour --------------------------------------------------

def test_fixed_dose_mass_balance(asm1):
    """The dosed stream is the flow-weighted mix of the inlet and the reagent."""
    d = DosingUnit("d", Reagent.from_species(asm1, SS=4e5), flow=2.0)
    C_in = asm1.concentrations({"SS": 60.0}, base="zero")
    s_in = Stream(Q=jnp.asarray(1000.0), C=C_in, model=asm1, scalars={"T": jnp.asarray(291.0)})
    out = d.compute_outputs(jnp.asarray(0.0), jnp.zeros((0,)), {"in": s_in},
                            asm1.default_parameters())["out"]
    ss = asm1.species_index["SS"]
    assert float(out.Q) == pytest.approx(1002.0)
    assert float(out.C[ss]) == pytest.approx((1000 * 60 + 2 * 4e5) / 1002)
    assert float(out.scalars["T"]) == pytest.approx(291.0)   # keeps the line T


def test_fixed_dose_flow_outputs_is_inflow_plus_dose(asm1):
    d = DosingUnit("d", Reagent.from_species(asm1, SS=4e5), flow=2.0)
    out = d.flow_outputs({"in": jnp.asarray(1000.0)}, asm1.default_parameters())
    assert float(out["out"]) == pytest.approx(1002.0)


def test_fixed_dose_in_a_plant_adds_load(asm1):
    p = Plant("fixeddose", time_unit="d")
    p.add_unit(DosingUnit("carbon", Reagent.from_species(asm1, SS=4e5), flow=2.0))
    p.add_unit(_tank(asm1))
    p.add_influent("feed", InfluentSeries.constant(asm1, SS=60.0, SNH=20.0, Q=1000.0),
                   to="carbon.in")
    p.connect("carbon", "tank")
    outs = p.outputs_at(jnp.asarray(0.0), p.initial_state())
    ss = asm1.species_index["SS"]
    assert float(outs[("carbon", "out")].C[ss]) == pytest.approx(
        (1000 * 60 + 2 * 4e5) / 1002)


# --- feedback dosing -------------------------------------------------------

def _feedback_plant(asm1, **dose_kw):
    p = Plant("fbdose", time_unit="d")
    p.add_unit(DosingUnit(
        "carbon", Reagent.from_species(asm1, SS=4e5),
        setpoint=1.0, measured_species="SNO", sensor="tank", **dose_kw))
    p.add_unit(_tank(asm1))
    p.add_influent("feed",
                   InfluentSeries.constant(asm1, SS=60.0, SNH=20.0, SNO=5.0, Q=1000.0),
                   to="carbon.in")
    p.connect("carbon", "tank")
    return p


@pytest.fixture(scope="module")
def feedback_plant(asm1):
    """One saturating feedback-dosing plant, shared by the read-only solve tests
    below so a solve compiled by one is reused by the other (the compiled solve
    is cached on the plant instance)."""
    return _feedback_plant(asm1, flow_max=10.0)


def test_feedback_dose_auto_wires_controller(asm1):
    p = _feedback_plant(asm1, flow_max=10.0)
    p._build_state_layout()
    assert "carbon_dosing" in p.units                        # auto-wired controller
    ctrl = p.units["carbon_dosing"]
    assert ctrl.setpoint == 1.0 and ctrl.measured_species == "SNO"
    assert p.units["carbon"].required_signals == ("_dose_carbon_dosing_flow",)
    # the sensor tap was wired: tank -> controller.measured
    assert any(c.from_unit == "tank" and c.to_unit == "carbon_dosing"
               for c in p.connections)


def test_feedback_dose_solves_and_publishes_signal(feedback_plant):
    p = feedback_plant
    sol = p.solve(t_span=(0.0, 2.0), t_eval=jnp.array([2.0]))
    assert bool(jnp.all(jnp.isfinite(sol.state)))
    sig = p.signals_at(jnp.asarray(2.0), sol.state[-1])
    flow = float(sig["_dose_carbon_dosing_flow"])
    # SNO (5) is far above the setpoint (1), so the controller drives the carbon
    # dose to its upper bound to push denitrification.
    assert flow == pytest.approx(10.0, abs=1e-6)


def test_feedback_dosing_plant_extended_after_a_solve(asm1):
    """A unit added after a solve re-runs the controller set-up, which keeps the
    dosing controller it already derived (not a second one, which would collide
    by name) and solves exactly as a plant built with the unit from the start."""
    def extend(p):
        p.add_unit(_tank(asm1, "tank2"))
        p.connect("tank.out", "tank2.in")

    t_eval = jnp.array([0.5, 1.0])
    edited = _feedback_plant(asm1, flow_max=10.0)
    edited.solve(t_span=(0.0, 1.0), t_eval=t_eval)
    extend(edited)
    sol = edited.solve(t_span=(0.0, 1.0), t_eval=t_eval)

    fresh = _feedback_plant(asm1, flow_max=10.0)
    extend(fresh)
    ref = fresh.solve(t_span=(0.0, 1.0), t_eval=t_eval)
    # The edited plant derived its controller before tank2 was added, so the
    # add order (and with it the state layout) differs: compare by unit.
    assert sorted(edited.list_units()) == sorted(fresh.list_units())
    assert edited.list_units().count("carbon_dosing") == 1
    for name in fresh.list_units():
        assert jnp.allclose(sol.unit_state(name), ref.unit_state(name), rtol=1e-12, atol=0.0)


def test_grad_flows_through_feedback_dosing(feedback_plant):
    """The gradient through the saturating feedback-dosing solve matches a
    central finite difference (step 1e-3 in a common scale on every
    parameter). At the default solver tolerance the two agree to ~1e-3, so the
    check is held at 5e-3. The solve uses the same span and save grid as
    test_feedback_dose_solves_and_publishes_signal, so the difference's forward
    solves reuse that test's compiled solve."""
    p = feedback_plant
    base = p.default_parameters()

    def loss(scale):
        sol = p.solve(t_span=(0.0, 2.0), t_eval=jnp.array([2.0]),
                      params=base * scale)
        return jnp.sum(sol.state ** 2)

    g = float(jax.grad(loss)(1.0))
    h = 1e-3
    fd = (float(loss(1.0 + h)) - float(loss(1.0 - h))) / (2.0 * h)
    assert g == pytest.approx(fd, rel=5e-3)


def test_shared_dosing_controller_disagreement_raises(asm1):
    p = Plant("bad", time_unit="d")
    r = Reagent.from_species(asm1, SS=4e5)
    p.add_unit(DosingUnit("dA", r, setpoint=1.0, measured_species="SNO",
                          sensor="tank", controller="co"))
    p.add_unit(DosingUnit("dB", r, setpoint=2.0, measured_species="SNO",
                          sensor="tank", controller="co"))     # different setpoint
    p.add_unit(_tank(asm1))
    with pytest.raises(ValueError, match="must agree"):
        p._build_state_layout()


def test_feedback_dose_sensor_must_be_a_reactor(asm1):
    """A feedback sensor that is not a concentration-state reactor (here a
    stateless mixer) is rejected at setup with a clear error, rather than failing
    opaquely deep in the first solve (#353)."""
    from aquakin.plant import MixerUnit
    p = Plant("bad_sensor", time_unit="d")
    r = Reagent.from_species(asm1, SS=4e5)
    p.add_unit(DosingUnit("carbon", r, setpoint=1.0, measured_species="SNO",
                          sensor="mix", flow_max=10.0))
    p.add_unit(MixerUnit("mix", ["a"], asm1))            # stateless: no concentration state
    with pytest.raises(ValueError, match="not a concentration vector"):
        p._build_state_layout()


def test_controlled_dose_without_bus_raises(asm1):
    """A feedback dose's compute_outputs called without the signal bus raises,
    rather than silently dosing nothing (the plant always supplies the bus)."""
    p = _feedback_plant(asm1, flow_max=10.0)
    p._build_state_layout()
    d = p.units["carbon"]
    s_in = Stream(Q=jnp.asarray(1000.0), C=asm1.default_concentrations(),
                  model=asm1)
    with pytest.raises(ValueError, match="control-signal bus"):
        d.compute_outputs(jnp.asarray(0.0), jnp.zeros((0,)), {"in": s_in},
                          asm1.default_parameters())          # signals=None
