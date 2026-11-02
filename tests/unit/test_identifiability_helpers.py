"""The profile sweep and the observation-noise resolution shared by the reactor
and plant identifiability tools, driven with stub fits (no solves)."""

import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from aquakin.integrate.fisher import _observation_noise
from aquakin.integrate.profile import _guarded, _profile_sweep

GRID = np.arange(5.0)


def _recording_fit(loss_of):
    """A stub ``fit_point`` recording ``(value, warm value)`` per call; its fit's
    ``.loss`` is ``loss_of(value, warm)`` and ``.value`` the grid value."""
    calls = []

    def fit(value, warm):
        calls.append((float(value), None if warm is None else warm.value))
        loss = loss_of(float(value), warm)
        return None if loss is None else SimpleNamespace(value=float(value), loss=loss)

    return fit, calls


def test_sweep_continues_from_the_anchor_and_bridges_a_gap():
    """With warm_start the anchor is fitted cold, then the sweep goes up and
    down, each point warm-started from the last good fit -- a failed point is
    skipped over, not used as a start."""
    fit, calls = _recording_fit(lambda v, w: None if v == 3.0 else (v - 2.0) ** 2)
    fits = _profile_sweep(
        GRID, fit, warm_start=True, anchor=2.0, polish=False, polish_passes=0, polish_tol=0.0
    )
    assert calls == [(2.0, None), (3.0, 2.0), (4.0, 2.0), (1.0, 2.0), (0.0, 1.0)]
    assert fits[3] is None and [f.value for f in fits if f] == [0.0, 1.0, 2.0, 4.0]


def test_sweep_without_warm_start_fits_every_point_cold():
    fit, calls = _recording_fit(lambda v, w: v)
    _profile_sweep(
        GRID, fit, warm_start=False, anchor=None, polish=False, polish_passes=0, polish_tol=0.0
    )
    assert calls == [(v, None) for v in GRID]


def test_polish_refits_a_point_its_neighbour_beats():
    """A point stranded in a worse minimum (loss 10 at x=1 cold) is re-fitted
    from the better neighbour, and the improved fit replaces it; a re-fit that
    does not improve is discarded."""

    def loss(v, w):
        if v == 1.0:
            return 0.5 if (w is not None and w.value == 0.0) else 10.0
        return 0.0

    fit, calls = _recording_fit(loss)
    fits = _profile_sweep(
        GRID[:3],
        fit,
        warm_start=False,
        anchor=None,
        polish=True,
        polish_passes=2,
        polish_tol=0.05,
    )
    assert fits[1].loss == 0.5
    # Pass 1: x=1 is re-fitted from x=0 (accepted, 10 -> 0.5), then, still
    # beaten by x=2, from x=2 (worse, discarded). Pass 2 repeats both re-fits,
    # neither improves, and the polish stops.
    polish_calls = calls[3:]
    assert polish_calls == [(1.0, 0.0), (1.0, 2.0), (1.0, 0.0), (1.0, 2.0)]


def test_guarded_turns_an_exception_into_a_warned_gap():
    def boom(value, warm):
        raise RuntimeError("solver blew up")

    with pytest.warns(UserWarning, match=r"k=0.5 \(RuntimeError: solver blew up\)"):
        assert _guarded(boom, "k")(0.5, None) is None


def test_observation_noise_shapes_mask_and_errors():
    sig, keep, div = _observation_noise("run 0", [0.1, 0.2], [[True, False], [True, True]], 2, 2)
    np.testing.assert_array_equal(np.asarray(sig), [[0.1, np.inf], [0.1, 0.2]])
    np.testing.assert_array_equal(np.asarray(div), [[0.1, 1.0], [0.1, 0.2]])
    assert np.asarray(keep).tolist() == [[True, False], [True, True]]
    # Without a mask nothing is infinite and the divisor is sigma itself.
    sig, keep, div = _observation_noise("run 0", 0.3, None, 2, 2)
    assert keep is None and np.all(np.asarray(sig) == 0.3) and div is sig
    # A zero sigma at an unmeasured entry is fine; at a measured one it is not.
    _observation_noise("run 0", [[0.1, 0.0]], [[True, False]], 1, 2)
    with pytest.raises(ValueError, match="run 0: sigma must be positive"):
        _observation_noise("run 0", [[0.1, 0.0]], None, 1, 2)
    with pytest.raises(ValueError, match="run 0: sigma of shape \\(3,\\) does not broadcast"):
        _observation_noise("run 0", [0.1, 0.2, 0.3], None, 4, 2)
    with pytest.raises(ValueError, match="run 0: mask of shape \\(3,\\) does not broadcast"):
        _observation_noise("run 0", 0.1, [True, False, True], 4, 2)
    with pytest.raises(ValueError, match="mask leaves no observation"):
        _observation_noise("run 0", 0.1, jnp.zeros((4, 2), dtype=bool), 4, 2)


def test_observation_noise_skips_the_positivity_check_under_tracing():
    """Inside a trace sigma has no concrete value to check; it must not raise."""
    import jax

    def scaled(s):
        return _observation_noise("run 0", s, None, 2, 1)[2].sum()

    assert float(jax.jit(scaled)(jnp.asarray(0.5))) == 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jax.grad(scaled)(jnp.asarray(0.5))
