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


@pytest.fixture
def decay_plant():
    net = aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")
    return net, _single_cstr_plant(net)


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
        max_iter=60,
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
        max_iter=1,
        check_finite=True,
    )
    assert np.isfinite(result.loss)


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
