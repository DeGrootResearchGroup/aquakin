"""Unit tests for RTD analytics."""

import jax.numpy as jnp
import numpy as np
import pytest

from aquakin.utils import rtd


def test_cstr_moments():
    """For an ideal CSTR, E(t) = (1/tau) * exp(-t/tau).

    Mean should equal tau, variance tau^2, Morrill ~= 21.85.
    """
    tau = 100.0
    t = jnp.linspace(0.0, 20.0 * tau, 20_000)
    C = jnp.exp(-t / tau)  # un-normalised tracer; E_curve normalises

    mean = float(rtd.mean_residence_time(t, C))
    var = float(rtd.variance(t, C))
    morrill = float(rtd.morrill_index(t, C))

    assert mean == pytest.approx(tau, rel=5e-3)
    assert var == pytest.approx(tau**2, rel=2e-2)
    assert morrill == pytest.approx(np.log(0.1) / np.log(0.9), rel=2e-3)


def test_pfr_morrill_near_one():
    """A narrow Gaussian impulse approximates plug flow; Morrill -> 1."""
    tau = 100.0
    sigma = 1.0
    t = jnp.linspace(0.0, 200.0, 10_000)
    C = jnp.exp(-0.5 * ((t - tau) / sigma) ** 2)
    morrill = float(rtd.morrill_index(t, C))
    assert morrill == pytest.approx(1.0, abs=0.05)


def test_F_curve_bounds_and_monotonic():
    tau = 50.0
    t = jnp.linspace(0.0, 500.0, 1000)
    C = jnp.exp(-t / tau)
    F = rtd.F_curve(t, C)
    assert float(F[0]) == 0.0
    assert float(F[-1]) == pytest.approx(1.0, abs=1e-3)
    assert jnp.all(jnp.diff(F) >= -1e-12)


def test_E_curve_integrates_to_one():
    t = jnp.linspace(0.0, 200.0, 5000)
    C = jnp.exp(-(t - 50.0) ** 2 / (2 * 10.0**2))
    E = rtd.E_curve(t, C)
    integral = float(jnp.trapezoid(E, t))
    assert integral == pytest.approx(1.0, rel=1e-6)


def test_percentile_time_invalid_q():
    t = jnp.linspace(0.0, 10.0, 11)
    C = jnp.ones_like(t)
    with pytest.raises(ValueError):
        rtd.percentile_time(t, C, q=1.5)


def test_mismatched_lengths_rejected():
    t = jnp.linspace(0.0, 10.0, 11)
    C = jnp.ones(10)
    with pytest.raises(ValueError):
        rtd.E_curve(t, C)
