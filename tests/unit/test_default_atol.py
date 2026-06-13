"""Per-component absolute-tolerance builder (``default_atol``).

The solver uses a vector ``atol`` scaled to each state's typical magnitude
(SUNDIALS "vector atol" / Hairer "atol proportional to typical value"). Every
component of the returned floor must stay strictly positive -- a zero ``atol_i``
removes the noise floor on that component, which the solver literature warns
against.
"""

import jax.numpy as jnp
import pytest

from aquakin.integrate._common import default_atol


def test_positive_for_normal_scales():
    atol = default_atol(jnp.asarray([1.0, 100.0, 5000.0]))
    assert jnp.all(atol > 0.0)


def test_all_zero_scale_falls_back_to_unit_floor():
    """When every magnitude is zero there is no scale to floor against, so the
    relative floor ``floor_frac*char`` would itself be 0. The unit-scale
    fallback keeps every component strictly positive."""
    atol = default_atol(jnp.zeros(5))
    assert jnp.all(atol > 0.0)
    # floor_frac (1e-6) * unit char (1.0) * atol_factor (1e-6) = 1e-12.
    assert jnp.allclose(atol, 1e-12)


def test_identity_for_nonzero_scale_input():
    """The fallback only fires when the characteristic scale is zero, so an
    input with any nonzero magnitude is unaffected by it."""
    scale = jnp.asarray([0.0, 2.0, 0.0])
    atol = default_atol(scale)
    # char = max = 2.0; floor = floor_frac*char = 2e-6; typ = max(|scale|, 2e-6)
    # = [2e-6, 2.0, 2e-6]; atol = atol_factor * typ.
    expected = 1e-6 * jnp.asarray([2e-6, 2.0, 2e-6])
    assert jnp.allclose(atol, expected)


def test_reference_raises_the_floor():
    """A per-species reference magnitude lifts the floor where the operating
    scale alone would be too small."""
    atol = default_atol(jnp.zeros(2), reference=jnp.asarray([10.0, 0.0]))
    # char = 10; species 0 floored at 10, species 1 at floor_frac*char = 1e-5.
    assert atol[0] == pytest.approx(1e-6 * 10.0)
    assert atol[1] == pytest.approx(1e-6 * 1e-5)
