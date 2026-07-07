"""Unit tests for the single public ``metrics.time_average`` helper.

``time_average(values, t, axis=0)`` is the one trapezoidal time-average kernel
the plant metric / design / aeration / ghg / evaluation code shares (issue #476
removed the per-module wrappers, two of which had an *inverted* ``(t, values)``
signature). These fast tests pin the value, the canonical argument order, the
one-point steady-state convention, the ``axis`` reduction, and the public export.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.metrics import time_average


def test_trapezoidal_average_matches_numpy():
    # Time-average = (1/(t1-t0)) * integral, on a non-uniform grid so a t<->values
    # swap would give a different number.
    t = jnp.array([0.0, 1.0, 3.0, 4.0])
    v = jnp.array([2.0, 4.0, 4.0, 0.0])
    expect = np.trapezoid(np.asarray(v), np.asarray(t)) / (4.0 - 0.0)
    assert float(time_average(v, t)) == pytest.approx(float(expect), rel=1e-12)


def test_argument_order_is_values_then_times():
    # The regression guard for the footgun this issue fixes: the signature is
    # (values, t), NOT (t, values). On this asymmetric data the two orders give
    # different results, so a future re-flip fails here loudly.
    t = jnp.array([0.0, 1.0, 3.0])
    v = jnp.array([0.0, 10.0, 10.0])
    correct = float(time_average(v, t))               # values first
    swapped = float(time_average(t, v))               # the inverted call
    assert correct == pytest.approx(25.0 / 3.0, rel=1e-12)  # analytic average
    assert not np.isclose(correct, swapped)           # the orders are distinguishable


def test_single_point_returns_the_sample():
    # A one-point solution (Plant.run_to_steady_state returns only the terminal
    # state) has a zero-width window; the average of a constant is that sample,
    # not a divide-by-zero.
    assert float(time_average(jnp.array([7.5]), jnp.array([2.0]))) == pytest.approx(7.5)


def test_axis_reduces_the_time_axis_only():
    # A (n_t, n_channel) trajectory reduces along axis=0 to a per-channel average.
    t = jnp.array([0.0, 2.0])
    v = jnp.array([[1.0, 10.0], [3.0, 30.0]])          # (2 times, 2 channels)
    out = time_average(v, t, axis=0)
    assert out.shape == (2,)
    np.testing.assert_allclose(np.asarray(out), [2.0, 20.0], rtol=1e-12)


def test_public_export_is_the_same_object():
    # Promoted to the public API (aquakin.time_average and aquakin.plant.time_average)
    # so callers reuse the one kernel instead of re-wrapping it.
    assert aquakin.time_average is time_average
    assert aquakin.plant.time_average is time_average
