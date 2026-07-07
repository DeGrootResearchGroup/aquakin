"""Unit tests for the argument-validation ``raise`` paths in the sensitivity /
profile / forward-sensitivity / DGSM entry points.

These pin the guard clauses that fire *before* any expensive solve -- bad
``output_fn`` / ``grid`` / ``delta`` / ``ranges`` / name-length arguments and the
result-object name accessors -- so a refactor that drops or reworks one of them
fails here. No stiff or full-plant solves are used: the reactor-based cases hit
the guard before integrating, and the DGSM cases pass a trivial pure ``fn``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate.forward_sensitivity import ForwardSensitivityResult
from aquakin.integrate.global_sensitivity import _validate_dgsm_ranges, dgsm
from aquakin.integrate.profile import profile_likelihood
from aquakin.integrate.sensitivity import sensitivity


def _batch_reactor(model):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    return aquakin.BatchReactor(model, conditions)


# ---------- sensitivity(): output_fn is required ----------


def test_sensitivity_requires_output_fn(simple_model):
    reactor = _batch_reactor(simple_model)
    C0 = jnp.array([1.0, 0.0])
    with pytest.raises(ValueError, match="output_fn is required"):
        sensitivity(reactor, C0)


# ---------- profile_likelihood(): grid / delta guards ----------


def _profile_args(model):
    C0 = jnp.array([1.0, 0.0])
    t_obs = jnp.array([0.0, 1.0, 2.0])
    obs = jnp.zeros((3, model.n_species))
    return C0, obs, t_obs


def test_profile_rejects_empty_grid(simple_model):
    reactor = _batch_reactor(simple_model)
    C0, obs, t_obs = _profile_args(simple_model)
    with pytest.raises(ValueError, match="grid must be a non-empty 1-D array"):
        profile_likelihood(
            reactor,
            C0,
            obs,
            t_obs,
            [simple_model.parameters[0]],
            grid=np.array([]),
            profile_param=simple_model.parameters[0],
        )


def test_profile_rejects_2d_grid(simple_model):
    reactor = _batch_reactor(simple_model)
    C0, obs, t_obs = _profile_args(simple_model)
    with pytest.raises(ValueError, match="grid must be a non-empty 1-D array"):
        profile_likelihood(
            reactor,
            C0,
            obs,
            t_obs,
            [simple_model.parameters[0]],
            grid=np.zeros((2, 2)),
            profile_param=simple_model.parameters[0],
        )


def test_profile_rejects_nonpositive_delta(simple_model):
    reactor = _batch_reactor(simple_model)
    C0, obs, t_obs = _profile_args(simple_model)
    with pytest.raises(ValueError, match="delta must be > 0"):
        profile_likelihood(
            reactor,
            C0,
            obs,
            t_obs,
            [simple_model.parameters[0]],
            grid=np.array([1.0, 2.0]),
            profile_param=simple_model.parameters[0],
            delta=0.0,
        )


# ---------- ForwardSensitivityResult: name accessors ----------


def _forward_sens_result(model):
    # A tiny fabricated result: one time point, all species, one sens parameter.
    S = jnp.zeros((1, model.n_species, 1))
    return ForwardSensitivityResult(
        solution=None,
        S=S,
        sens_params=[model.parameters[0]],
        model=model,
    )


def test_forward_sensitivity_S_named_rejects_unknown_species(simple_model):
    result = _forward_sens_result(simple_model)
    with pytest.raises(KeyError, match="Unknown species 'not_a_species'"):
        result.S_named("not_a_species")


def test_forward_sensitivity_dC_dparam_rejects_unknown_param(simple_model):
    result = _forward_sens_result(simple_model)
    with pytest.raises(KeyError, match="is not a sensitivity parameter"):
        result.dC_dparam(simple_model.species[0], "not_a_param")


# ---------- _validate_dgsm_ranges: shape / bound / name-length guards ----------


def test_validate_dgsm_ranges_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"ranges must have shape \(d, 2\)"):
        _validate_dgsm_ranges(np.zeros((3, 3)), None)


def test_validate_dgsm_ranges_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="each range must satisfy upper > lower"):
        _validate_dgsm_ranges([(1.0, 0.0)], None)


def test_validate_dgsm_ranges_rejects_input_name_length_mismatch():
    with pytest.raises(ValueError, match="input_names has 1 entries but ranges has d=2"):
        _validate_dgsm_ranges([(0.0, 1.0), (0.0, 1.0)], ["only_one"])


# ---------- dgsm(): output_names length / all-non-finite guards ----------


def test_dgsm_rejects_output_name_length_mismatch():
    # A vector-output fn (m=2) with a mismatched output_names length.
    with pytest.raises(ValueError, match="output_names has 1 entries but fn returns m=2"):
        dgsm(
            lambda z: jnp.array([z[0], z[1]]),
            [(0.0, 1.0), (0.0, 1.0)],
            n_samples=8,
            output_names=["only_one"],
        )


def test_dgsm_rejects_all_nonfinite_output():
    # Every sample is non-finite, so no output has >= 2 finite samples.
    with pytest.raises(RuntimeError, match="DGSM needs >= 2 finite samples"):
        dgsm(
            lambda z: jnp.nan * z[0],
            [(0.0, 1.0), (0.0, 1.0)],
            n_samples=8,
        )
