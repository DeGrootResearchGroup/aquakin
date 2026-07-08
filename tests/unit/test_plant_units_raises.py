"""Constructor / argument-validation ``raise`` coverage for the plant units.

Each test triggers one previously-untested ``raise`` in ``aquakin.plant`` via the
most natural constructor or method call with a bad argument -- almost all are
construction-time ``ValueError`` guards, so none needs a stiff plant solve. The
sensitivity mode/shape/transform validations are reached on a tiny single-CSTR
toy-decay plant (or with ``state=`` / ``continuation=False`` so no steady-state
solve runs).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import CSTRUnit, Plant
from aquakin.plant.influent import InfluentSeries


# ---------------------------------------------------------------------------
# Shared fixtures: the toy A -> B decay model and a one-CSTR plant on it.
# ---------------------------------------------------------------------------
def _decay_model():
    return aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")


def _single_cstr_plant(net):
    plant = Plant("single_cstr")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    influent = InfluentSeries(
        t=jnp.asarray([0.0, 100.0]),
        Q=jnp.asarray([10.0, 10.0]),
        C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        model=net,
    )
    plant.add_influent("feed", influent, to="tank.inlet")
    return plant


@pytest.fixture
def decay_model():
    return _decay_model()


@pytest.fixture
def decay_plant(decay_model):
    return _single_cstr_plant(decay_model)


# ---------------------------------------------------------------------------
# control.py -- PIController Ti / Tt validation
# ---------------------------------------------------------------------------
def _pi_kwargs(net, **overrides):
    kw = dict(
        name="pi",
        model=net,
        measured_species="A",
        setpoint=1.0,
        Kp=1.0,
        Ti=100.0,
        Tt=100.0,
        offset=0.0,
        out_min=0.0,
        out_max=10.0,
        signal_name="kla_ctrl",
    )
    kw.update(overrides)
    return kw


def test_picontroller_ti_must_be_positive(decay_model):
    from aquakin.plant.control import PIController

    with pytest.raises(ValueError, match="Ti must be > 0"):
        PIController(**_pi_kwargs(decay_model, Ti=0.0))


def test_picontroller_tt_must_be_positive_with_antiwindup(decay_model):
    from aquakin.plant.control import PIController

    with pytest.raises(ValueError, match="Tt must be > 0 with anti-windup"):
        PIController(**_pi_kwargs(decay_model, Tt=0.0, use_antiwindup=True))


# ---------------------------------------------------------------------------
# cstr.py -- aeration species check + missing-condition check
# ---------------------------------------------------------------------------
def test_cstr_aeration_species_not_in_model(decay_model):
    from aquakin.plant.cstr import Aeration, build_aeration_vectors

    aer = Aeration(species="NOT_A_SPECIES", kla=10.0, do_sat=8.0)
    with pytest.raises(ValueError, match="aeration species 'NOT_A_SPECIES' is not in the model"):
        build_aeration_vectors(aer, decay_model, "tank")


def test_cstr_missing_required_condition(decay_model):
    # The toy model requires condition "T"; omit it.
    with pytest.raises(ValueError, match="missing required condition values"):
        CSTRUnit(
            name="tank",
            model=decay_model,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={},
        )


# ---------------------------------------------------------------------------
# separators.py -- IdealThickener tss_removal_percent + target_tss_percent
# ---------------------------------------------------------------------------
def test_thickener_tss_removal_percent_out_of_range(decay_model):
    from aquakin.plant.separators import IdealThickener

    with pytest.raises(ValueError, match=r"tss_removal_percent must be in \[0, 100\]"):
        IdealThickener(
            name="thick",
            model=decay_model,
            target_tss_percent=5.0,
            tss_removal_percent=150.0,
        )


def test_thickener_target_tss_percent_must_be_positive(decay_model):
    from aquakin.plant.separators import IdealThickener

    with pytest.raises(ValueError, match="target_tss_percent must be"):
        IdealThickener(
            name="thick",
            model=decay_model,
            target_tss_percent=0.0,
        )


# ---------------------------------------------------------------------------
# takacs.py -- composition_mode + feed_layer validation
# ---------------------------------------------------------------------------
def test_takacs_bad_composition_mode(decay_model):
    from aquakin.plant.takacs import TakacsClarifier

    with pytest.raises(ValueError, match="composition_mode must be"):
        TakacsClarifier(
            name="clar",
            model=decay_model,
            area=100.0,
            height=4.0,
            underflow_Q=100.0,
            composition_mode="bogus",
        )


def test_takacs_feed_layer_out_of_range(decay_model):
    from aquakin.plant.takacs import TakacsClarifier

    with pytest.raises(ValueError, match=r"feed_layer must be in \[0, 10\)"):
        TakacsClarifier(
            name="clar",
            model=decay_model,
            area=100.0,
            height=4.0,
            underflow_Q=100.0,
            feed_layer=20,
        )


# ---------------------------------------------------------------------------
# clarifier.py -- IdealClarifier capture_efficiency in [0, 1]
# ---------------------------------------------------------------------------
def test_ideal_clarifier_capture_efficiency_out_of_range(decay_model):
    from aquakin.plant.clarifier import IdealClarifier

    with pytest.raises(ValueError, match=r"capture_efficiency must be in \[0, 1\]"):
        IdealClarifier(
            name="clar",
            model=decay_model,
            underflow_Q=100.0,
            capture_efficiency=1.5,
        )


# ---------------------------------------------------------------------------
# coupling.py -- CouplingAware.coupling_pattern abstract NotImplementedError
# ---------------------------------------------------------------------------
def test_coupling_aware_abstract_method_not_implemented():
    from aquakin.plant.coupling import CouplingAware

    # A concrete subclass that calls super().coupling_pattern() reaches the
    # abstract body's ``raise NotImplementedError``.
    class Bare(CouplingAware):
        def coupling_pattern(self):
            return super().coupling_pattern()

    with pytest.raises(NotImplementedError):
        Bare().coupling_pattern()


# ---------------------------------------------------------------------------
# digester.py -- ADM1DigesterUnit missing-condition check
# ---------------------------------------------------------------------------
def test_digester_missing_condition():
    from aquakin.plant.digester import ADM1DigesterUnit

    # A stub model that requires a condition its defaults do not supply, so the
    # {defaults, conditions} merge still leaves a gap. (Every shipped model's
    # defaults cover its required conditions, so a stub is the only way to reach
    # this guard.)
    class _Fields:
        def __init__(self):
            self.fields = {"T": jnp.asarray([308.15])}

    class _StubModel:
        def __init__(self):
            self.n_species = 2
            self.conditions_required = ("T", "needs_this")
            self.species_index = {"a": 0, "b": 1}

        def default_conditions(self):
            return _Fields()

    with pytest.raises(ValueError, match="missing condition values"):
        ADM1DigesterUnit(
            name="dig",
            model=_StubModel(),
            volume=1000.0,
            conditions={},  # "needs_this" absent from defaults + conditions
        )


# ---------------------------------------------------------------------------
# primary_clarifier.py -- f_PS in (0, 1)
# ---------------------------------------------------------------------------
def test_primary_clarifier_f_ps_out_of_range(decay_model):
    from aquakin.plant.primary_clarifier import PrimaryClarifier

    with pytest.raises(ValueError, match=r"f_PS must be in \(0, 1\)"):
        PrimaryClarifier(
            name="prim",
            model=decay_model,
            volume=900.0,
            f_PS=1.5,
        )


# ---------------------------------------------------------------------------
# settling.py -- LayeredSettling n_layers >= 2
# ---------------------------------------------------------------------------
def test_layered_settling_needs_two_layers():
    from aquakin.plant.settling import LayeredSettling

    with pytest.raises(ValueError, match="needs n_layers >= 2"):
        LayeredSettling(n_layers=1)


# ---------------------------------------------------------------------------
# interfaces.py -- ASM1toADM1 / ADM1toASM1 fdegrade NotImplementedError
# ---------------------------------------------------------------------------
def test_asm1_to_adm1_fdegrade_not_implemented():
    from aquakin.plant.interfaces import ASM1toADM1

    asm1 = aquakin.load_model("asm1")
    adm1 = aquakin.load_model("adm1")
    with pytest.raises(NotImplementedError, match="fdegrade_adm = 0"):
        ASM1toADM1(source_model=asm1, target_model=adm1, fdegrade_adm=0.5)


def test_adm1_to_asm1_fdegrade_not_implemented():
    from aquakin.plant.interfaces import ADM1toASM1

    asm1 = aquakin.load_model("asm1")
    adm1 = aquakin.load_model("adm1")
    with pytest.raises(NotImplementedError, match="fdegrade_as = 0"):
        ADM1toASM1(source_model=adm1, target_model=asm1, fdegrade_as=0.5)


# ---------------------------------------------------------------------------
# bsm/bsm2.py -- wastage schedule length mismatch
# ---------------------------------------------------------------------------
def test_bsm2_wastage_schedule_length_mismatch():
    from aquakin.plant.bsm.bsm2 import bsm2_wastage_schedule

    # 4 fixed values but only 4 step times => needs len(steps)+1 == 5 values.
    with pytest.raises(ValueError, match=r"needs len\(steps\)\+1 values"):
        bsm2_wastage_schedule(steps=[10.0, 20.0, 30.0, 40.0])


# ---------------------------------------------------------------------------
# bsm/warmstart.py -- no activated-sludge reactors in the plant
# ---------------------------------------------------------------------------
def test_warmstart_no_as_reactors(decay_model):
    from aquakin.plant.bsm.warmstart import bsm1_warm_start
    from aquakin.plant.mixer import MixerUnit

    # A mixer-only plant: no aeration-carrying AS reactor at all (a plain CSTR
    # would be counted, since it carries the ``aeration`` attribute).
    plant = Plant("mixer_only")
    plant.add_unit(MixerUnit("mix", ["a", "b"], decay_model))
    with pytest.raises(ValueError, match="No activated-sludge reactors found"):
        bsm1_warm_start(plant)


# ---------------------------------------------------------------------------
# ifas.py -- fill_fraction / positive-dimension / biofilm mask-shape checks
# ---------------------------------------------------------------------------
def _ifas_kwargs(net, **overrides):
    kw = dict(
        name="ifas",
        model=net,
        volume=100.0,
        input_port_names=["inlet"],
        specific_surface_area=500.0,
        fill_fraction=0.4,
        biofilm_thickness=1e-4,
        conditions={"T": 293.15},
    )
    kw.update(overrides)
    return kw


def test_ifas_fill_fraction_out_of_range(decay_model):
    from aquakin.plant.ifas import IFASUnit

    with pytest.raises(ValueError, match=r"fill_fraction must be in \(0, 1\]"):
        IFASUnit(**_ifas_kwargs(decay_model, fill_fraction=1.5))


def test_ifas_dimensions_must_be_positive(decay_model):
    from aquakin.plant.ifas import IFASUnit

    with pytest.raises(ValueError, match="must be positive"):
        IFASUnit(**_ifas_kwargs(decay_model, specific_surface_area=-1.0))


def test_ifas_biofilm_fixed_mask_wrong_shape(decay_model):
    from aquakin.plant.ifas import IFASUnit

    # The model has 2 species; a length-3 mask has the wrong shape. This raise
    # is downstream of the biofilm construction, so pass a valid config.
    with pytest.raises(ValueError, match=r"biofilm_fixed_mask must have shape"):
        IFASUnit(
            **_ifas_kwargs(
                decay_model,
                biofilm_fixed_mask=np.zeros((3,), dtype=bool),
            )
        )


# ---------------------------------------------------------------------------
# sensitivity.py -- mode / shape / transform validation (no stiff solve).
# The mode checks fire after the branch dispatch (invalid mode skips every
# branch); the ranges-shape checks fire before any solve; steady_state_dgsm's
# transform check runs with continuation=False so no steady-state solve happens.
# ---------------------------------------------------------------------------
def _out_A(state_or_sol):
    # A trivial scalar output usable both on a state vector (steady) and on a
    # PlantSolution (dynamic). For the steady path we pass state= directly, so
    # output_fn only ever sees the state vector there.
    return jnp.atleast_1d(jnp.asarray(state_or_sol)[..., 0])


def test_steady_state_sensitivity_bad_mode(decay_plant):
    # state= is supplied so no steady-state solve runs; the invalid mode falls
    # through the forward/reverse branches to the ValueError.
    y0 = decay_plant.initial_state()
    with pytest.raises(ValueError, match="mode must be 'auto', 'forward', or 'reverse'"):
        decay_plant.steady_state_sensitivity(
            state=y0,
            output_fn=_out_A,
            mode="bogus",
        )


def test_steady_state_dgsm_ranges_wrong_shape(decay_plant):
    # The ranges-shape guard fires before any steady-state solve.
    with pytest.raises(ValueError, match=r"ranges must have shape"):
        decay_plant.steady_state_dgsm(
            ranges=[[0.05, 0.2], [0.05, 0.2]],  # 2 rows, but 1 screened param
            output_fn=_out_A,
            wrt=["simple_decay.A_to_B.k"],
        )


def test_steady_state_dgsm_bad_input_dist(decay_plant):
    with pytest.raises(ValueError, match="input_dist must be 'uniform' or 'normal'"):
        decay_plant.steady_state_dgsm(
            ranges=[[0.05, 0.2]],
            output_fn=_out_A,
            wrt=["simple_decay.A_to_B.k"],
            input_dist="bogus",
            continuation=False,
        )


def test_steady_state_dgsm_normal_requires_transforms(decay_plant):
    # continuation=False skips the nominal steady-state solve; the normal-input
    # transform check fires first.
    with pytest.raises(ValueError, match="input_dist='normal' requires input_transforms"):
        decay_plant.steady_state_dgsm(
            ranges=[[0.05, 0.2]],
            output_fn=_out_A,
            wrt=["simple_decay.A_to_B.k"],
            input_dist="normal",
            input_transforms=None,
            continuation=False,
        )


def test_dynamic_adjoint_kwargs_bad_mode():
    from aquakin.plant.sensitivity import _dynamic_adjoint_kwargs

    with pytest.raises(ValueError, match="mode must be 'reverse', 'forward', or 'auto'"):
        _dynamic_adjoint_kwargs("bogus")


def test_dynamic_value_jac_bad_mode(decay_plant):
    # Invalid mode skips both the forward and reverse branches of
    # _dynamic_value_jac and hits its trailing ValueError -- no solve runs.
    from aquakin.plant.sensitivity import _dynamic_value_jac

    params = decay_plant.default_parameters()
    with pytest.raises(ValueError, match="mode must be 'reverse', 'forward', or 'auto'"):
        _dynamic_value_jac(
            decay_plant,
            params,
            [0],
            params[:1],
            output_fn=lambda sol: jnp.zeros((1,)),
            t_span=(0.0, 1.0),
            t_eval=None,
            y0=decay_plant.initial_state(),
            mode="bogus",
            solve_kwargs={},
        )


def test_dynamic_value_jac_forward_rejects_extra_kwargs(decay_plant):
    from aquakin.plant.sensitivity import _dynamic_value_jac

    params = decay_plant.default_parameters()
    with pytest.raises(TypeError, match="forward-mode dynamic sensitivity"):
        _dynamic_value_jac(
            decay_plant,
            params,
            [0],
            params[:1],
            output_fn=lambda sol: jnp.zeros((1,)),
            t_span=(0.0, 1.0),
            t_eval=None,
            y0=decay_plant.initial_state(),
            mode="forward",
            solve_kwargs={"not_a_valid_kwarg": 1},
        )


def test_dynamic_sensitivity_operating_requires_forward(decay_plant):
    # An operating-condition sensitivity is forward-mode only; reverse raises
    # before any solve.
    with pytest.raises(ValueError, match="operating-condition sensitivity is available only"):
        decay_plant.dynamic_sensitivity(
            output_fn=lambda sol: jnp.zeros((1,)),
            t_span=(0.0, 1.0),
            operating=[{"kind": "influent_concentration", "port": "feed", "species": "A"}],
            mode="reverse",
        )


def test_dynamic_dgsm_ranges_wrong_shape(decay_plant):
    with pytest.raises(ValueError, match=r"ranges must have shape"):
        decay_plant.dynamic_dgsm(
            ranges=[[0.05, 0.2], [0.05, 0.2]],  # 2 rows, 1 screened param
            output_fn=lambda sol: jnp.zeros((1,)),
            t_span=(0.0, 1.0),
            wrt=["simple_decay.A_to_B.k"],
        )
