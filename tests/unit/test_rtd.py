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


def test_percentile_time_truncated_tail_warns():
    """A tracer impulse cut off before it washes out must warn: E_curve
    normalises the partial area to 1 and biases the percentile times (and
    Morrill) low, so the truncation is flagged rather than passing silently."""
    tau = 100.0
    # Window ends at t=120 (~1.2 tau): C[-1] = exp(-1.2) ~= 0.30 of the peak,
    # i.e. clearly not washed out.
    t = jnp.linspace(0.0, 120.0, 2000)
    C = jnp.exp(-t / tau)
    with pytest.warns(UserWarning, match="truncated"):
        rtd.percentile_time(t, C, q=0.9)
    # The Morrill index (which uses t90) therefore also warns.
    with pytest.warns(UserWarning, match="truncated"):
        rtd.morrill_index(t, C)


def test_morrill_index_warns_only_once():
    """morrill_index calls percentile_time twice (t90, t10); a truncated
    response must warn exactly once, not once per inner call."""
    t = jnp.linspace(0.0, 120.0, 2000)
    C = jnp.exp(-t / 100.0)  # not washed out at 1.2 tau
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        rtd.morrill_index(t, C)
    truncation = [w for w in caught if "truncated" in str(w.message)]
    assert len(truncation) == 1


def test_percentile_time_full_washout_no_warning():
    """A fully washed-out response (tail at baseline) resolves percentiles with
    no truncation warning."""
    tau = 50.0
    t = jnp.linspace(0.0, 1000.0, 5000)   # ~20 tau: C[-1] ~= exp(-20) ~ 0
    C = jnp.exp(-t / tau)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")          # any warning fails the test
        assert float(rtd.percentile_time(t, C, q=0.9)) > 0.0
        assert float(rtd.percentile_time(t, C, q=1.0)) == pytest.approx(
            float(t[-1]), rel=1e-6
        )
