"""Calibration observations in declared units (``observation_units=``).

The helpers are tested directly; the reactor fit is checked by fitting the same
data twice, once in the model's units and once in another unit declared with
``observation_units``, and the plant entry point by capturing what it hands on
to the problem resolution (no plant solve).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate.calibration.loss import _convert_observations, _observation_unit_factors

# ----- the factors ----------------------------------------------------------------


def test_factors_per_column():
    cols = [("SNH", "g_N/m3"), ("XS", "g_COD/m3"), ("SO", "g_O2/m3")]
    f = _observation_unit_factors(cols, {"XS": "kgCOD/m3", "SNH": "mg/L"})
    np.testing.assert_allclose(f, [1.0, 1000.0, 1.0])
    # Nothing declared, or only spellings of the model's units: no conversion.
    assert _observation_unit_factors(cols, None) is None
    assert _observation_unit_factors(cols, {}) is None
    assert _observation_unit_factors(cols, {"SNH": "mg/L"}) is None


def test_a_name_covers_every_column_of_that_name():
    cols = [("B", "mol/L"), ("B", "mol/L")]
    np.testing.assert_allclose(_observation_unit_factors(cols, {"B": "mmol/L"}), [1e-3, 1e-3])


@pytest.mark.parametrize(
    "units, match",
    [
        ({"XX": "mg/L"}, "not observed columns"),
        ({"SNH": 5}, "must be a string"),
        ({"SNH": "mol/m3"}, r"observation_units\['SNH'\]: cannot convert"),
        ({"SNH": "gibberish/L"}, "cannot be read"),
        ({"BLANK": "mg/L"}, "cannot be read"),
    ],
)
def test_factors_refuse_what_they_cannot_convert(units, match):
    cols = [("SNH", "g_N/m3"), ("BLANK", "")]
    with pytest.raises(ValueError, match=match):
        _observation_unit_factors(cols, units)


# ----- applying them ------------------------------------------------------------------


def test_conversion_of_observations_and_sigma():
    f = np.array([1.0, 1000.0])
    obs = jnp.ones((4, 2))
    out, sig = _convert_observations(obs, jnp.asarray(0.5), f)
    np.testing.assert_allclose(out, [[1.0, 1000.0]] * 4)
    np.testing.assert_allclose(sig, [0.5, 500.0])  # a scalar becomes per column
    _, sig = _convert_observations(obs, jnp.asarray([0.1, 0.2]), f)
    np.testing.assert_allclose(sig, [0.1, 200.0])
    _, sig = _convert_observations(obs, jnp.full((4, 2), 0.1), f)
    np.testing.assert_allclose(sig[0], [0.1, 100.0])
    # A plain list is per column for one dataset.
    _, sig = _convert_observations(obs, [0.1, 0.2], f)
    np.testing.assert_allclose(sig, [0.1, 200.0])
    for passthrough in (None, "estimate"):
        assert _convert_observations(obs, passthrough, f)[1] == passthrough


def test_conversion_of_multi_batch_and_single_column_data():
    f = np.array([1e-3])
    out, sig = _convert_observations([jnp.ones(3), jnp.full(5, 2.0)], [1.0, 2.0], f)
    np.testing.assert_allclose(out[0], [1e-3] * 3)
    np.testing.assert_allclose(out[1], [2e-3] * 5)
    np.testing.assert_allclose(sig, [1e-3, 2e-3])
    same, sig = _convert_observations(jnp.ones(3), jnp.asarray(1.0), None)
    np.testing.assert_allclose(same, 1.0)


# ----- the reactor fit ----------------------------------------------------------------


@pytest.fixture(scope="module")
def decay(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0 = jnp.asarray([1.0, 0.0])
    params = simple_model.default_parameters().at[0].set(0.25)
    t_obs = jnp.linspace(0.5, 10.0, 12)
    obs = reactor.solve(C0, params=params, t_span=(0.0, 10.0), t_eval=t_obs).C_named("B")
    return reactor, C0, t_obs, obs


def _fit(reactor, C0, t_obs, obs, **kw):
    return aquakin.calibrate(
        reactor,
        C0,
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="nll",
        laplace=True,
        **kw,
    )


def test_calibrate_fits_data_in_declared_units(decay):
    # The model's species are in mol/L; the data in mmol/L with a 1 mmol/L
    # noise level give the same fit and posterior as mol/L data at 1e-3 mol/L.
    reactor, C0, t_obs, obs = decay
    noisy = obs + 2e-3 * jnp.sin(7.0 * t_obs)
    native = _fit(reactor, C0, t_obs, noisy, sigma=1e-3)
    declared = _fit(
        reactor, C0, t_obs, 1000.0 * noisy, sigma=1.0, observation_units={"B": "mmol/L"}
    )
    k_native = native.params_named["A_to_B.k"]
    assert declared.params_named["A_to_B.k"] == pytest.approx(k_native, rel=1e-8)
    np.testing.assert_allclose(declared.posterior_cov, native.posterior_cov, rtol=1e-6)


def test_calibrate_refuses_units_with_an_observation_fn(decay):
    reactor, C0, t_obs, obs = decay
    with pytest.raises(ValueError, match="observation_fn"):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observation_fn=lambda C, p: C[1],
            observation_units={"B": "mmol/L"},
        )


def test_profile_likelihood_forwards_the_units(decay, monkeypatch):
    import aquakin.integrate.profile as profile_mod

    seen = {}

    class _Stop(Exception):
        pass

    def _stub(*args, **kw):
        seen.update(kw)
        raise _Stop

    reactor, C0, t_obs, obs = decay
    monkeypatch.setattr(profile_mod, "calibrate", _stub)
    # Each inner fit fails at the stub and is recorded as a gap; only the
    # arguments it received matter here.
    with pytest.warns(UserWarning, match="inner fit failed"):
        aquakin.profile_likelihood(
            reactor,
            C0,
            1000.0 * obs,
            t_obs,
            ["A_to_B.k"],
            grid=[0.2, 0.25, 0.3],
            profile_ic="A",
            observed_species=["B"],
            observation_units={"B": "mmol/L"},
            sigma=1.0,
        )
    assert seen["observation_units"] == {"B": "mmol/L"}


# ----- the plant ----------------------------------------------------------------------


def test_plant_calibrate_converts_before_resolving(cstr_plant, monkeypatch):
    import aquakin.plant.calibrate as plant_cal

    plant = cstr_plant
    seen = {}

    class _Stop(Exception):
        pass

    def _stub(plant, observations, t_obs, free_params, **kw):
        seen.update(observations=observations, sigma=kw["sigma"])
        raise _Stop

    monkeypatch.setattr(plant_cal, "_resolve_plant_problem", _stub)
    with pytest.raises(_Stop):
        plant.calibrate(
            jnp.full((3, 2), 500.0),
            jnp.array([1.0, 2.0, 3.0]),
            ["simple_decay.A_to_B.k"],
            target="tank",
            observed_channels=["A", "B"],
            observation_units={"B": "mmol/L"},
            loss="nll",
            sigma=jnp.asarray([1.0, 5.0]),
        )
    np.testing.assert_allclose(seen["observations"][0], [500.0, 0.5])
    np.testing.assert_allclose(seen["sigma"], [1.0, 5e-3])
    # A name that is not an observed channel is refused.
    with pytest.raises(ValueError, match="not observed columns"):
        plant.calibrate(
            jnp.ones((3, 1)),
            jnp.array([1.0, 2.0, 3.0]),
            ["simple_decay.A_to_B.k"],
            target="tank",
            observed_channels=["B"],
            observation_units={"A": "mmol/L"},
        )
    # Every channel observed (channels=None) is resolved from the stream model.
    with pytest.raises(_Stop):
        plant.calibrate(
            jnp.full((3, 2), 1.0),
            jnp.array([1.0, 2.0, 3.0]),
            ["simple_decay.A_to_B.k"],
            target="tank",
            observation_units={"A": "mmol/L"},
        )
    np.testing.assert_allclose(seen["observations"][0], [1e-3, 1.0])
