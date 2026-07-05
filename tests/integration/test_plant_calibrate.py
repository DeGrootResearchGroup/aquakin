"""Plant calibration through the forward-model seam (``plant.calibrate``).

The reactor-calibration machinery is reused unchanged for a plant: only the
forward solve (``plant.solve`` + stream reconstruction) and the by-name plant
parameter surface differ. These tests pin the plant end on a small, non-stiff
single-CSTR plant -- a synthetic-data parameter recovery (with the reverse
stable-adjoint gradient flowing through the whole plant solve) and the
plant-specific validation errors.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import FreeICConfig, OptimizerConfig
from aquakin.plant import CSTRUnit, Plant
from aquakin.plant.influent import InfluentSeries


def _single_cstr_plant(net):
    """A one-unit plant: a CSTR on the toy A -> B decay model, constant feed."""
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


def _two_cstr_plant(net):
    """A two-unit chain (tank1 -> tank2) on the same decay model, constant feed.

    Both tanks reference the same model, so the plant has one shared parameter
    block (``"simple_decay.A_to_B.k"``) and two output streams (``"tank1"`` /
    ``"tank2"``) -- a small vehicle for a multi-stream fit."""
    plant = Plant("two_cstr")
    for name in ("tank1", "tank2"):
        plant.add_unit(
            CSTRUnit(
                name=name,
                model=net,
                volume=100.0,
                input_port_names=["inlet"],
                conditions={"T": 293.15},
            )
        )
    plant.connect("tank1", "tank2.inlet")
    influent = InfluentSeries(
        t=jnp.asarray([0.0, 100.0]),
        Q=jnp.asarray([10.0, 10.0]),
        C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        model=net,
    )
    plant.add_influent("feed", influent, to="tank1.inlet")
    return plant


@pytest.fixture
def decay_plant():
    net = aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")
    return net, _single_cstr_plant(net)


@pytest.fixture
def two_tank_plant():
    net = aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")
    return net, _two_cstr_plant(net)


PARAM = "simple_decay.A_to_B.k"


@pytest.mark.slow  # a plant solve inside an optimizer loop
def test_plant_calibrate_recovers_synthetic_parameter(decay_plant):
    """Generate synthetic effluent B(t) at a known rate constant, then recover it
    from a perturbed start by fitting the plant parameter against the stream."""
    net, plant = decay_plant
    base = plant.default_parameters()
    gidx = plant.parameter_index(PARAM)
    true_k = float(base[gidx])

    t_obs = jnp.linspace(1.0, 40.0, 8)
    truth_params = base.at[gidx].set(true_k)
    sol = plant.solve(t_span=(0.0, 40.0), t_eval=t_obs, params=truth_params)
    obs_B = plant.stream(sol, "tank", truth_params).C_named("B")  # (n_t,)

    # Start well away from the truth; recover it.
    start = base.at[gidx].set(true_k * 0.3)
    result = plant.calibrate(
        obs_B,
        t_obs,
        [PARAM],
        target="tank",
        observed_channels=["B"],
        params=start,
        transforms={PARAM: "positive_log"},
        optimizer=OptimizerConfig(max_iter=60),
    )
    assert result.converged
    # Recovery is to a few percent: the synthetic data uses the concrete forward
    # solve while the fit's reverse gradient runs the stable-adjoint ESDIRK, so the
    # loss minimum sits within the solver-path difference of true_k -- the fit
    # finding it confirms the gradient flows correctly through the whole plant.
    assert result.params_named[PARAM] == pytest.approx(true_k, rel=5e-2)
    # The fitted full vector carries the recovered value at the right slot.
    assert float(result.params[gidx]) == pytest.approx(true_k, rel=5e-2)


def test_plant_calibrate_gradient_is_finite_through_the_plant(decay_plant):
    """check_finite=True must pass: the reverse stable-adjoint gradient of the
    plant-stream loss is finite (the whole point of routing through plant.solve's
    discrete adjoint)."""
    net, plant = decay_plant
    base = plant.default_parameters()
    t_obs = jnp.linspace(1.0, 20.0, 5)
    sol = plant.solve(t_span=(0.0, 20.0), t_eval=t_obs, params=base)
    obs_B = plant.stream(sol, "tank", base).C_named("B")
    # A single L-BFGS-B step from the truth; check_finite would raise on a NaN.
    result = plant.calibrate(
        obs_B,
        t_obs,
        [PARAM],
        target="tank",
        observed_channels=["B"],
        params=base,
        transforms={PARAM: "positive_log"},
        optimizer=OptimizerConfig(max_iter=1),
        check_finite=True,
    )
    assert np.isfinite(result.loss)


@pytest.mark.slow  # a plant solve inside an optimizer loop
def test_plant_calibrate_multi_stream_recovers(two_tank_plant):
    """A single shared rate constant recovered from TWO output streams at once
    (the tank1 and tank2 effluents), fit through `observables=`."""
    net, plant = two_tank_plant
    base = plant.default_parameters()
    gidx = plant.parameter_index(PARAM)
    true_k = float(base[gidx])

    t_obs = jnp.linspace(1.0, 40.0, 8)
    truth = base.at[gidx].set(true_k)
    sol = plant.solve(t_span=(0.0, 40.0), t_eval=t_obs, params=truth)
    b1 = plant.stream(sol, "tank1", truth).C_named("B")
    b2 = plant.stream(sol, "tank2", truth).C_named("B")
    obs = jnp.stack([b1, b2], axis=1)  # (n_t, 2): one column per stream

    start = base.at[gidx].set(true_k * 0.3)
    result = plant.calibrate(
        obs,
        t_obs,
        [PARAM],
        observables=[
            aquakin.plant.PlantObservable("tank1", ["B"]),
            aquakin.plant.PlantObservable("tank2", ["B"]),
        ],
        params=start,
        transforms={PARAM: "positive_log"},
        optimizer=OptimizerConfig(max_iter=60),
    )
    assert result.converged
    assert result.params_named[PARAM] == pytest.approx(true_k, rel=5e-2)


def test_plant_observable_normalizer():
    """The observable spec accepts a PlantObservable, a bare stream name, a dict,
    and a (stream, channels) pair; a bad entry is rejected."""
    from aquakin.plant.calibrate import PlantObservable, _normalize_observables

    # Single-target fallback when observables is None.
    assert _normalize_observables(None, "effluent", ["SNH"]) == [
        PlantObservable("effluent", ["SNH"])
    ]
    # Mixed forms all coerce to PlantObservable.
    out = _normalize_observables(
        [
            PlantObservable("a", ("X",)),
            "b",
            {"stream": "c", "channels": ["Y"]},
            ("d", ["Z"]),
        ],
        "unused",
        None,
    )
    assert [o.stream for o in out] == ["a", "b", "c", "d"]
    assert out[1].channels is None  # bare name -> all channels
    with pytest.raises(TypeError, match="each observable must be"):
        _normalize_observables([123], "x", None)


def test_plant_calibrate_multi_stream_shape_mismatch(two_tank_plant):
    """Two observables expect two columns; a 1-column observations array errors
    (fast, before any solve)."""
    net, plant = two_tank_plant
    with pytest.raises(ValueError, match="across the observable"):
        plant.calibrate(
            jnp.zeros((3, 1)),
            jnp.array([1.0, 2.0, 3.0]),
            [PARAM],
            observables=[("tank1", ["B"]), ("tank2", ["B"])],
        )


@pytest.mark.slow  # a full BSM1 (multi-unit, recycled, stiff) fit
def test_plant_calibrate_recovers_bsm1_muH():
    """The real test: recover a kinetic parameter (heterotroph max growth rate
    muH) of the full BSM1 plant -- five aerated reactors, a settler, and the
    internal + RAS recycles -- from its effluent nitrogen, with the reverse
    gradient flowing through the whole stiff monolithic solve."""
    from aquakin import load_model
    from aquakin.plant.bsm import bsm1_warm_start, build_bsm1, load_bsm1_influent
    from aquakin.plant.influent import InfluentSeries

    asm1 = load_model("asm1")
    plant = build_bsm1(asm1)
    # A constant influent (the dry-weather first sample) gives a clean transient.
    dry = load_bsm1_influent("dry", asm1)
    plant.add_influent(
        "influent",
        InfluentSeries(
            t=jnp.asarray([0.0, 200.0]),
            Q=jnp.asarray([float(dry.Q[0]), float(dry.Q[0])]),
            C=jnp.tile(dry.C[0], (2, 1)),
            model=asm1,
        ),
    )
    y0 = bsm1_warm_start(plant)
    gidx = plant.parameter_index("asm1.muH")
    base = plant.default_parameters()
    true_muH = float(base[gidx])

    t_obs = jnp.linspace(0.5, 8.0, 6)
    truth = base.at[gidx].set(true_muH)
    sol = plant.solve(t_span=(0.0, 8.0), t_eval=t_obs, params=truth, y0=y0)
    eff = plant.stream(sol, "effluent", truth)
    obs = jnp.stack([eff.C_named("SNH"), eff.C_named("SNO")], axis=1)

    result = plant.calibrate(
        obs,
        t_obs,
        ["asm1.muH"],
        target="effluent",
        observed_channels=["SNH", "SNO"],
        params=base.at[gidx].set(true_muH * 0.7),
        y0=y0,
        transforms={"asm1.muH": "positive_log"},
        optimizer=OptimizerConfig(max_iter=25),
    )
    assert result.converged
    assert result.params_named["asm1.muH"] == pytest.approx(true_muH, rel=2e-2)


@pytest.mark.slow  # a plant solve inside an optimizer loop
def test_plant_calibrate_free_ic_recovers(decay_plant):
    """Jointly recover a rate constant AND an assembled-state initial condition
    (the tank's initial A) from the early transient, via `free_ic`."""
    net, plant = decay_plant
    plant._build_state_layout()
    a_idx = plant._state_layout["tank"][0] + net.species_index["A"]
    base = plant.default_parameters()
    true_A0 = 5.0
    true_y0 = plant.initial_state().at[a_idx].set(true_A0)

    # Observe the early transient, where the initial condition still matters
    # (integrate from t=0 so the fitted IC is the t=0 value).
    t_obs = jnp.linspace(0.2, 6.0, 8)
    sol = plant.solve(t_span=(0.0, 6.0), t_eval=t_obs, y0=true_y0)
    obs_A = plant.stream(sol, "tank", None).C_named("A")

    result = plant.calibrate(
        obs_A,
        t_obs,
        [PARAM],
        target="tank",
        observed_channels=["A"],
        params=base,
        y0=plant.initial_state(),  # start from the default (A0 = 1)
        t_span=(0.0, 6.0),
        free_ic=FreeICConfig(["tank.A"]),
        transforms={PARAM: "positive_log"},
        optimizer=OptimizerConfig(max_iter=80),
    )
    assert result.converged
    assert result.params_named[PARAM] == pytest.approx(
        float(base[plant.parameter_index(PARAM)]), rel=5e-2
    )
    assert result.ic_named[0]["tank.A"] == pytest.approx(true_A0, rel=5e-2)
    # The fitted full state carries the recovered IC at its flat slot.
    assert float(result.C0_fitted[0][a_idx]) == pytest.approx(true_A0, rel=5e-2)


def test_plant_free_ic_normalizer():
    """The free-IC spec accepts 'unit.species' strings and (unit, species) pairs;
    a malformed entry is rejected."""
    from aquakin.plant.calibrate import _normalize_free_ic

    assert _normalize_free_ic(["tank.A", ("tank2", "B")]) == [("tank", "A"), ("tank2", "B")]
    assert _normalize_free_ic(None) == []
    with pytest.raises(ValueError, match="must be 'unit"):
        _normalize_free_ic(["no_dot"])
    with pytest.raises(TypeError, match="free_ic entry must be"):
        _normalize_free_ic([123])


def test_plant_calibrate_free_ic_unknown_species(decay_plant):
    net, plant = decay_plant
    with pytest.raises(KeyError, match="Unknown free_ic species"):
        plant.calibrate(
            jnp.zeros((3, 1)),
            jnp.array([0.2, 0.4, 0.6]),
            [PARAM],
            target="tank",
            observed_channels=["A"],
            free_ic=FreeICConfig(["tank.Z"]),
        )


def test_plant_calibrate_free_ic_bad_bounds(decay_plant):
    net, plant = decay_plant
    with pytest.raises(ValueError, match="ic_bounds must satisfy"):
        plant.calibrate(
            jnp.zeros((3, 1)),
            jnp.array([0.2, 0.4, 0.6]),
            [PARAM],
            target="tank",
            observed_channels=["A"],
            free_ic=FreeICConfig(["tank.A"], bounds=(5.0, 1.0)),
        )


@pytest.mark.slow  # two plant solves per objective evaluation
def test_plant_calibrate_multibatch_recovers(decay_plant):
    """Joint multi-batch fit: two runs of the same plant from different initial
    states share one rate constant, and their summed data terms recover it."""
    net, plant = decay_plant
    plant._build_state_layout()
    a_idx = plant._state_layout["tank"][0] + net.species_index["A"]
    base = plant.default_parameters()
    true_k = float(base[plant.parameter_index(PARAM)])
    t_obs = jnp.linspace(0.5, 8.0, 6)

    def _run(a0):
        y0 = plant.initial_state().at[a_idx].set(a0)
        obs = plant.stream(
            plant.solve(t_span=(0.0, 8.0), t_eval=t_obs, y0=y0), "tank", None
        ).C_named("B")
        return obs, y0

    o1, y1 = _run(3.0)
    o2, y2 = _run(8.0)  # a genuinely different transient

    result = plant.calibrate(
        [o1, o2],
        [t_obs, t_obs],
        [PARAM],
        target="tank",
        observed_channels=["B"],
        params=base.at[plant.parameter_index(PARAM)].set(true_k * 0.4),
        y0=[y1, y2],
        t_span=[(0.0, 8.0), (0.0, 8.0)],
        transforms={PARAM: "positive_log"},
        optimizer=OptimizerConfig(max_iter=60),
    )
    assert result.converged
    assert result.params_named[PARAM] == pytest.approx(true_k, rel=5e-2)


def test_plant_calibrate_multibatch_requires_y0_list(decay_plant):
    """A multi-batch fit (list of observation arrays) needs one y0 per dataset."""
    net, plant = decay_plant
    obs = [jnp.zeros((3, 1)), jnp.zeros((3, 1))]
    t = [jnp.array([0.2, 0.4, 0.6]), jnp.array([0.2, 0.4, 0.6])]
    with pytest.raises(ValueError, match="needs y0 as a list"):
        plant.calibrate(obs, t, [PARAM], target="tank", observed_channels=["B"], y0=None)


def test_plant_calibrate_multibatch_rejects_free_ic(decay_plant):
    net, plant = decay_plant
    obs = [jnp.zeros((3, 1)), jnp.zeros((3, 1))]
    t = [jnp.array([0.2, 0.4, 0.6]), jnp.array([0.2, 0.4, 0.6])]
    with pytest.raises(ValueError, match="free_ic is not yet supported"):
        plant.calibrate(
            obs,
            t,
            [PARAM],
            target="tank",
            observed_channels=["B"],
            y0=[plant.initial_state(), plant.initial_state()],
            free_ic=FreeICConfig(["tank.A"]),
        )


def test_plant_calibrate_multibatch_length_mismatch(decay_plant):
    net, plant = decay_plant
    with pytest.raises(ValueError, match="lists of equal length"):
        plant.calibrate(
            [jnp.zeros((3, 1)), jnp.zeros((3, 1))],
            [jnp.array([0.2, 0.4, 0.6])],
            [PARAM],
            target="tank",
            observed_channels=["B"],
            y0=[plant.initial_state(), plant.initial_state()],
        )


# ---------- validation (fast, no solve) ----------


def test_plant_calibrate_rejects_unknown_parameter(decay_plant):
    net, plant = decay_plant
    with pytest.raises(KeyError, match="Unknown plant parameter"):
        plant.calibrate(jnp.zeros(3), jnp.array([1.0, 2.0, 3.0]), ["nope.k"], target="tank")


def test_plant_calibrate_rejects_unknown_target(decay_plant):
    net, plant = decay_plant
    with pytest.raises(KeyError, match="Unknown calibration target"):
        plant.calibrate(jnp.zeros(3), jnp.array([1.0, 2.0, 3.0]), [PARAM], target="not_a_stream")


def test_plant_calibrate_rejects_unknown_channel(decay_plant):
    net, plant = decay_plant
    with pytest.raises(KeyError, match="Unknown observed channel"):
        plant.calibrate(
            jnp.zeros(3),
            jnp.array([1.0, 2.0, 3.0]),
            [PARAM],
            target="tank",
            observed_channels=["Z"],
        )


def test_plant_calibrate_rejects_empty_free_params(decay_plant):
    net, plant = decay_plant
    with pytest.raises(ValueError, match="free_params must be non-empty"):
        plant.calibrate(jnp.zeros(3), jnp.array([1.0, 2.0, 3.0]), [], target="tank")


def test_plant_calibrate_rejects_shape_mismatch(decay_plant):
    net, plant = decay_plant
    with pytest.raises(ValueError, match="rows but t_obs"):
        plant.calibrate(
            jnp.zeros(5),
            jnp.array([1.0, 2.0, 3.0]),
            [PARAM],
            target="tank",
            observed_channels=["B"],
        )
