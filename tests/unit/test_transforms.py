"""Unit tests for the shared parameter-transform algebra.

``aquakin.integrate._transforms`` is the single source of truth for the
``{none, positive_log, logit}`` maps used by both the reactor/plant calibration
path (``jax.numpy`` backend, differentiable) and the host-side DGSM screen
(``numpy`` backend). These tests pin the round-trip, the analytic derivative
against autodiff, and cross-backend agreement, plus the thin delegating wrappers
that used to carry their own copies of the algebra.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquakin.integrate._transforms import (
    VALID_TRANSFORMS,
    dphysical_dunconstrained,
    from_unconstrained,
    to_unconstrained,
)

# Physical sample points valid for every transform: positive and inside (0, 1).
_PHYS = np.array([0.05, 0.2, 0.5, 0.8, 0.95])


@pytest.mark.parametrize("kind", VALID_TRANSFORMS)
def test_round_trip_physical_unconstrained(kind):
    u = to_unconstrained(_PHYS, kind, xp=np)
    back = from_unconstrained(u, kind, xp=np)
    np.testing.assert_allclose(back, _PHYS, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", VALID_TRANSFORMS)
def test_numpy_matches_jax_backend(kind):
    u_np = to_unconstrained(_PHYS, kind, xp=np)
    u_jx = to_unconstrained(jnp.asarray(_PHYS), kind, xp=jnp)
    np.testing.assert_allclose(np.asarray(u_jx), u_np, rtol=1e-12, atol=1e-12)
    p_np = from_unconstrained(u_np, kind, xp=np)
    p_jx = from_unconstrained(jnp.asarray(u_np), kind, xp=jnp)
    np.testing.assert_allclose(np.asarray(p_jx), p_np, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", VALID_TRANSFORMS)
def test_derivative_matches_autodiff(kind):
    # dphysical_dunconstrained(physical) must equal d/du from_unconstrained(u).
    for phys in _PHYS:
        u = float(to_unconstrained(np.array(phys), kind, xp=np))
        ad = float(jax.grad(lambda z: from_unconstrained(z, kind))(u))
        analytic = float(dphysical_dunconstrained(phys, kind, xp=np))
        assert analytic == pytest.approx(ad, rel=1e-9, abs=1e-12)


def test_unknown_transform_raises():
    for fn in (to_unconstrained, from_unconstrained, dphysical_dunconstrained):
        with pytest.raises(ValueError, match="Unknown transform"):
            fn(0.5, "softplus")


def test_calibrate_wrappers_delegate_unchanged():
    # The calibrate.py wrappers must reproduce their historical values exactly.
    from aquakin.integrate.calibrate import (
        _from_unconstrained,
        _jacobian_physical_wrt_theta,
        _to_unconstrained,
    )

    for kind in VALID_TRANSFORMS:
        theta = jnp.asarray(0.3)  # unconstrained value
        phys = _from_unconstrained(theta, kind)
        # positive_log: exp; logit: sigmoid; none: identity
        expected = {
            "none": 0.3,
            "positive_log": float(np.exp(0.3)),
            "logit": float(jax.nn.sigmoid(0.3)),
        }[kind]
        assert float(phys) == pytest.approx(expected, rel=1e-12)
        # round-trip through the wrappers
        assert float(_to_unconstrained(phys, kind)) == pytest.approx(0.3, rel=1e-9)
        # delta-method Jacobian dp/dtheta == d/dtheta from_unconstrained
        jac = float(_jacobian_physical_wrt_theta(theta, kind))
        ad = float(jax.grad(lambda z: _from_unconstrained(z, kind))(theta))
        assert jac == pytest.approx(ad, rel=1e-9, abs=1e-12)


def test_plant_wrappers_delegate_unchanged():
    from aquakin.plant.sensitivity import _dtheta_dz, _from_z, _to_z

    for kind in VALID_TRANSFORMS:
        phys = 0.3
        z = _to_z(phys, kind)
        np.testing.assert_allclose(_from_z(z, kind), phys, rtol=1e-9)
        # chain-rule factor is expressed via the physical value
        expected = {"none": 1.0, "positive_log": phys, "logit": phys * (1.0 - phys)}[kind]
        np.testing.assert_allclose(float(_dtheta_dz(phys, kind)), expected, rtol=1e-12)
