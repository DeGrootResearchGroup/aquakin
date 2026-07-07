"""Focused tests for the assembly / argument-validation ``raise`` statements in
``aquakin/plant/plant.py``.

Every test triggers its raise at assembly or argument-check time, before any
stiff plant integration -- so the whole module stays fast. Where a guard is a
defensive check on an internal value that the public ``solve``/``connect`` API
cannot actually produce (an internal gradient-string, the stable-adjoint entry
point), the private method is exercised directly with the offending value; that
is the only way to reach the guard and is noted in the test.

The tiny-plant idioms (a single CSTR + influent, a mixer/tank/splitter loop)
mirror ``tests/integration/test_plant_assembly.py``.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin import IntegratorConfig
from aquakin.plant import CSTRUnit, MixerUnit, Plant
from aquakin.plant.influent import InfluentSeries


@pytest.fixture
def simple_net():
    return aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")


def _constant_influent(net, *, Q=10.0, C=(1.0, 0.0), t_end=100.0):
    return InfluentSeries(
        t=jnp.asarray([0.0, t_end]),
        Q=jnp.asarray([Q, Q]),
        C=jnp.asarray([list(C), list(C)]),
        model=net,
    )


def _single_cstr_plant(net):
    plant = Plant("one")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(net), to="tank.inlet")
    return plant


# ----- constructor / assembly validation -----------------------------------


def test_recycle_passes_must_be_positive():
    """plant.py:204 -- Plant(recycle_passes=0) is rejected."""
    with pytest.raises(ValueError, match="recycle_passes must be >= 1"):
        Plant("bad", recycle_passes=0)


def test_add_unit_rejects_duplicate_name(simple_net):
    """plant.py:325 -- re-adding a unit under an existing name is rejected."""
    plant = Plant("dup")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    with pytest.raises(ValueError, match="already added"):
        plant.add_unit(
            CSTRUnit(
                name="tank",
                model=simple_net,
                volume=50.0,
                input_port_names=["inlet"],
                conditions={"T": 293.15},
            )
        )


def test_add_influent_rejects_duplicate_name(simple_net):
    """plant.py:365 -- re-registering an influent under an existing name is
    rejected."""
    plant = _single_cstr_plant(simple_net)  # already has "feed"
    with pytest.raises(ValueError, match="Influent 'feed' already added"):
        plant.add_influent("feed", _constant_influent(simple_net), to="tank.inlet")


def test_connect_unknown_unit(simple_net):
    """plant.py:597 -- an endpoint naming a unit the plant does not have."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(KeyError, match="Unknown unit 'ghost'"):
        plant.connect("ghost", "tank")


def test_unit_model_requires_model_attribute(simple_net):
    """plant.py:719 -- the translator-default helper needs the unit to carry a
    ``model``; a unit without one is a KeyError. No shipped unit lacks a model,
    so the guard is exercised directly with a stand-in object placed in the
    unit table (the only way to reach it)."""
    plant = _single_cstr_plant(simple_net)
    plant.units["modelless"] = object()  # has no .model attribute
    with pytest.raises(KeyError, match="has no 'model' attribute"):
        plant._unit_model("modelless")


# ----- aeration / dosing controller materialization ------------------------


def test_aeration_controller_senses_non_concentration_unit():
    """plant.py:794 -- a closed-loop DO controller whose ``sensor`` is a mixer
    (no concentration state) is rejected at topology setup."""
    from aquakin.plant.cstr import Aeration

    asm1 = aquakin.load_model("asm1")
    plant = Plant("bad_aer")
    plant.add_unit(MixerUnit(name="mix", input_port_names=["a"], model=asm1))
    plant.add_unit(
        CSTRUnit(
            name="reactor",
            model=asm1,
            volume=1000.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
            aeration=Aeration(do_setpoint=2.0, sensor="mix"),
        )
    )
    with pytest.raises(ValueError, match="state is not a concentration vector"):
        plant._build_state_layout()  # runs _materialize_aeration


def test_dosing_controller_senses_unknown_unit():
    """plant.py:887 -- a feedback DosingUnit whose ``sensor`` is not a unit of
    the plant is rejected at topology setup."""
    from aquakin.plant.dosing import DosingUnit, Reagent

    asm1 = aquakin.load_model("asm1")
    plant = Plant("bad_dose")
    plant.add_unit(
        CSTRUnit(
            name="reactor",
            model=asm1,
            volume=1000.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        DosingUnit(
            name="dose",
            reagent=Reagent.from_species(asm1, SS=4.0e5),
            setpoint=1.0,
            measured_species="SNO",
            sensor="ghost",  # not a unit of the plant
        )
    )
    with pytest.raises(ValueError, match="senses unit 'ghost', which"):
        plant._build_state_layout()  # runs _materialize_dosing


# ----- parameter coercion --------------------------------------------------


def test_coerce_params_wrong_shape(simple_net):
    """plant.py:1140 -- a parameter vector of the wrong length is rejected
    (reached through ``solve(params=...)`` before any integration)."""
    plant = _single_cstr_plant(simple_net)
    bad = jnp.zeros(999)
    with pytest.raises(ValueError, match="params has shape"):
        plant.solve(t_span=(0.0, 1.0), t_eval=jnp.asarray([0.0, 1.0]), params=bad)


# ----- solve argument checks (fire before any real solve) ------------------


def test_solve_y0_wrong_shape(simple_net):
    """plant.py:2431 -- a warm-start ``y0`` of the wrong length is rejected up
    front (the shape check fires before the integration)."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(ValueError, match="y0 has shape"):
        plant.solve(t_span=(0.0, 1.0), t_eval=jnp.asarray([0.0, 1.0]), y0=jnp.zeros(999))


def test_solve_tspan_end_must_exceed_start(simple_net):
    """plant.py:2444 -- a non-increasing ``t_span`` is rejected before solving."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(ValueError, match="t_span end must exceed start"):
        plant.solve(t_span=(5.0, 5.0), t_eval=jnp.asarray([5.0]))


def test_solve_rejects_bad_colored_jacobian(simple_net):
    """plant.py:2645 -- an out-of-range ``colored_jacobian`` is rejected."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(ValueError, match="colored_jacobian must be True, False"):
        plant.solve(
            t_span=(0.0, 1.0),
            t_eval=jnp.asarray([0.0, 1.0]),
            integrator=IntegratorConfig(colored_jacobian="bogus"),
        )


def test_solve_rejects_events_and_event_together(simple_net):
    """plant.py:2664 -- passing both the user-facing ``events=`` and the
    low-level ``event=`` is rejected (a non-empty ``events`` list and any
    ``event`` object suffice; the check fires before either is used)."""
    plant = _single_cstr_plant(simple_net)
    sentinel_event = object()
    with pytest.raises(ValueError, match=r"pass either events= .* or"):
        plant.solve(
            t_span=(0.0, 1.0),
            t_eval=jnp.asarray([0.0, 1.0]),
            events=[object()],
            event=sentinel_event,
        )


def test_validate_solve_args_rejects_unknown_gradient(simple_net):
    """plant.py:2641 -- a defensive guard on the internal gradient string. The
    public ``diff=`` decode only ever yields 'auto'/'jax_adjoint', so the guard
    is exercised directly with an out-of-range value."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(ValueError, match="gradient must be 'auto'"):
        plant._validate_solve_args(
            events=None,
            event=None,
            integrator=IntegratorConfig(),
            forward_fast=False,
            gradient="nonsense",
            params=plant.default_parameters(),
            y0=None,
        )


def test_validate_solve_args_forward_fast_rejects_stable_adjoint(simple_net):
    """plant.py:2694 -- forward_fast is incompatible with the stable adjoint.
    The ``gradient`` string reaching the validator is 'auto'/'jax_adjoint' from
    the public decode, so this guard on the literal 'stable_adjoint' is
    exercised directly."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(ValueError, match="forward_fast is a non-differentiable"):
        plant._validate_solve_args(
            events=None,
            event=None,
            integrator=IntegratorConfig(),
            forward_fast=True,
            gradient="stable_adjoint",
            params=plant.default_parameters(),
            y0=None,
        )


def _stable_adjoint_kwargs(**overrides):
    """The full keyword set ``_solve_stable_adjoint`` requires, with valid
    defaults; the guard under test fires before any of them is used."""
    kw = dict(
        rtol=1e-6,
        atol=1e-6,
        solver=None,
        factormax=None,
        colored_jacobian="auto",
        order=5,
        adjoint=None,
        dtmax=None,
        event=None,
        adjoint_max_steps=4096,
        adjoint_low_memory=False,
        time_factor=1.0,
        time_unit=None,
    )
    kw.update(overrides)
    return kw


def test_solve_stable_adjoint_rejects_adjoint_and_dtmax(simple_net):
    """plant.py:2812 -- the stable-adjoint entry point forms its own adjoint and
    controls its own steps, so an explicit adjoint=/dtmax= is rejected. Through
    ``solve`` a non-None dtmax pins the jax_adjoint path, so the guard is
    exercised directly on the private entry point."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(ValueError, match="do not also pass adjoint= or dtmax="):
        plant._solve_stable_adjoint(
            0.0,
            1.0,
            jnp.asarray([0.0, 1.0]),
            plant.default_parameters(),
            plant.initial_state(),
            **_stable_adjoint_kwargs(dtmax=0.5),  # not allowed on this path
        )


def test_solve_stable_adjoint_rejects_event(simple_net):
    """plant.py:2817 -- an ``event=`` is only supported on the forward
    jax_adjoint path, so the stable-adjoint entry point rejects it. Exercised
    directly (through ``solve`` an ``event`` pins jax_adjoint)."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(ValueError, match="only supported on the forward"):
        plant._solve_stable_adjoint(
            0.0,
            1.0,
            jnp.asarray([0.0, 1.0]),
            plant.default_parameters(),
            plant.initial_state(),
            **_stable_adjoint_kwargs(event=object()),
        )


# ----- operating-condition sensitivity spec validation ---------------------


def test_parse_operating_unknown_species(simple_net):
    """plant.py:4104 -- an ``influent_concentration`` operating spec naming a
    species not in the influent model is rejected. Pure spec validation (the
    sensitivity entry points call it before any solve), exercised directly."""
    plant = _single_cstr_plant(simple_net)
    with pytest.raises(KeyError, match="operating species 'ZZ' is not in"):
        plant._parse_operating(
            [{"kind": "influent_concentration", "port": "feed", "species": "ZZ"}]
        )
