"""AD-mode helpers that hide the diffrax adjoint plumbing."""

import diffrax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.integrate._common import (
    check_finite_gradient,
    forward_adjoint,
    with_adjoint,
)


def test_forward_adjoint_is_direct_adjoint():
    assert isinstance(forward_adjoint(), diffrax.DirectAdjoint)
    # Also exported at the package level.
    assert isinstance(aquakin.forward_adjoint(), diffrax.DirectAdjoint)


def test_with_adjoint_swaps_adjoint_without_mutating_original(simple_network):
    base = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(T=293.15)
    )
    assert base.adjoint is None
    fwd = with_adjoint(base, forward_adjoint())
    assert isinstance(fwd.adjoint, diffrax.DirectAdjoint)
    assert base.adjoint is None                  # original untouched
    assert fwd.network is base.network           # shallow copy shares the network


def test_check_finite_gradient_passes_and_raises():
    check_finite_gradient(jnp.array([1.0, 2.0]), what="grad", remedy="do X")
    with pytest.raises(RuntimeError, match="non-finite.*do X"):
        check_finite_gradient(jnp.array([1.0, jnp.nan]), what="grad", remedy="do X")
    with pytest.raises(RuntimeError):
        check_finite_gradient(jnp.array([jnp.inf]), what="grad", remedy="do X")
