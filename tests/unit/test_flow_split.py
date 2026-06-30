"""Controlled overflow/underflow split helpers shared by the clarifiers.

The clarifiers (``IdealClarifier``, ``TakacsClarifier``) split one inlet into a
controlled (fixed-setpoint) flow and a free remainder via this shared logic. It
had no direct unit coverage -- only end-to-end exercise through the plant solves
-- so the affine (``clamp=False``) flow rule and the construction-time validation
were untested in isolation.
"""

import jax.numpy as jnp
import pytest

from aquakin.plant._flow_split import (
    _min_setpoint,
    split_controlled_flows,
    validate_controlled_split,
)


class _Schedule:
    """Minimal stand-in for a time schedule (any object exposing ``min_value``)."""

    def __init__(self, min_value):
        self.min_value = min_value


# ----- validate_controlled_split ------------------------------------------

def test_validate_requires_exactly_one_setpoint():
    with pytest.raises(ValueError, match="exactly one"):
        validate_controlled_split("clar", overflow_Q=10.0, underflow_Q=5.0)
    with pytest.raises(ValueError, match="exactly one"):
        validate_controlled_split("clar", overflow_Q=None, underflow_Q=None)


def test_validate_rejects_negative_setpoint():
    with pytest.raises(ValueError, match="overflow_Q must be non-negative"):
        validate_controlled_split("clar", overflow_Q=-1.0, underflow_Q=None)
    with pytest.raises(ValueError, match="underflow_Q must be non-negative"):
        validate_controlled_split("clar", overflow_Q=None, underflow_Q=-1.0)


def test_validate_accepts_single_nonnegative_setpoint():
    # Neither call raises.
    validate_controlled_split("clar", overflow_Q=None, underflow_Q=0.0)
    validate_controlled_split("clar", overflow_Q=12.5, underflow_Q=None)


def test_validate_uses_schedule_minimum():
    """A schedule setpoint is validated by its minimum: a non-negative minimum is
    accepted, a negative one is rejected."""
    validate_controlled_split("clar", overflow_Q=None, underflow_Q=_Schedule(0.0))
    with pytest.raises(ValueError, match="underflow_Q must be non-negative"):
        validate_controlled_split("clar", overflow_Q=None,
                                  underflow_Q=_Schedule(-3.0))


# ----- _min_setpoint -------------------------------------------------------

def test_min_setpoint_float_and_schedule():
    assert _min_setpoint(4.0) == 4.0
    assert _min_setpoint(_Schedule(2.5)) == 2.5
    assert _min_setpoint(_Schedule(0.0)) == 0.0


# ----- split_controlled_flows ---------------------------------------------

def test_underflow_controlled_unclamped_is_affine():
    """With ``clamp=False`` the underflow-controlled split is the affine BSM rule
    ``Q_e = Q_f - Q_u`` -- exactly so ``Plant._resolve_flows`` stays linear."""
    Q_over, Q_under = split_controlled_flows(
        overflow_Q=None, underflow_Q=20.0, Q_in=jnp.asarray(100.0), clamp=False)
    assert float(Q_over) == pytest.approx(80.0)
    assert float(Q_under) == pytest.approx(20.0)


def test_underflow_controlled_unclamped_allows_negative_overflow():
    """A setpoint above the feed makes the affine overflow go negative. This is
    deliberate (it keeps the flow map exactly affine); a future 'fix' that clamps
    here would silently break the linear recycle solve, so pin it."""
    Q_over, Q_under = split_controlled_flows(
        overflow_Q=None, underflow_Q=120.0, Q_in=jnp.asarray(100.0), clamp=False)
    assert float(Q_over) == pytest.approx(-20.0)
    # Q_under = Q_in - Q_over closes the balance regardless.
    assert float(Q_over) + float(Q_under) == pytest.approx(100.0)


def test_clamp_keeps_flows_physical():
    """The concentration/settling stage uses ``clamp=True`` so a feed below the
    setpoint never produces a negative-flow stream."""
    Q_over, Q_under = split_controlled_flows(
        overflow_Q=None, underflow_Q=120.0, Q_in=jnp.asarray(100.0), clamp=True)
    assert float(Q_over) == pytest.approx(0.0)        # clipped up to 0
    assert float(Q_under) == pytest.approx(100.0)
    # And the upper clip: a zero underflow setpoint cannot make overflow exceed Q_in.
    Q_over2, Q_under2 = split_controlled_flows(
        overflow_Q=None, underflow_Q=0.0, Q_in=jnp.asarray(100.0), clamp=True)
    assert float(Q_over2) == pytest.approx(100.0)
    assert float(Q_under2) == pytest.approx(0.0)


def test_overflow_controlled_remainder_tracks_feed():
    """When the *overflow* is the setpoint, the underflow is the remainder."""
    Q_over, Q_under = split_controlled_flows(
        overflow_Q=30.0, underflow_Q=None, Q_in=jnp.asarray(100.0), clamp=False)
    assert float(Q_over) == pytest.approx(30.0)
    assert float(Q_under) == pytest.approx(70.0)
