"""AD-mode helpers that hide the diffrax adjoint plumbing."""

import diffrax
import jax.numpy as jnp
import pytest

import jax

import aquakin
from aquakin.integrate._common import (
    GradientCheckMixin,
    check_finite_gradient,
    forward_adjoint,
    with_adjoint,
)
from aquakin.integrate.batch import BatchReactor
from aquakin.integrate.biofilm import BiofilmReactor
from aquakin.integrate.particle import ParticleTrackReactor
from aquakin.integrate.pfr import PlugFlowReactor


def test_forward_adjoint_is_direct_adjoint():
    assert isinstance(forward_adjoint(), diffrax.DirectAdjoint)
    # Also exported at the package level.
    assert isinstance(aquakin.forward_adjoint(), diffrax.DirectAdjoint)


def test_with_adjoint_swaps_adjoint_without_mutating_original(simple_model):
    base = aquakin.BatchReactor(
        simple_model, aquakin.SpatialConditions.uniform(T=293.15)
    )
    assert base.adjoint is None
    fwd = with_adjoint(base, forward_adjoint())
    assert isinstance(fwd.adjoint, diffrax.DirectAdjoint)
    assert base.adjoint is None                  # original untouched
    assert fwd.model is base.model           # shallow copy shares the model


def test_check_finite_gradient_passes_and_raises():
    check_finite_gradient(jnp.array([1.0, 2.0]), what="grad", remedy="do X")
    with pytest.raises(RuntimeError, match="non-finite.*do X"):
        check_finite_gradient(jnp.array([1.0, jnp.nan]), what="grad", remedy="do X")
    with pytest.raises(RuntimeError):
        check_finite_gradient(jnp.array([jnp.inf]), what="grad", remedy="do X")


def test_check_finite_gradient_is_public():
    """The DIY checker is exported at the package level so a user with their own
    loss + optimizer can guard a gradient without reaching into a submodule."""
    assert aquakin.check_finite_gradient is check_finite_gradient


def test_every_reactor_has_gradient_check():
    """All reactor types inherit the finiteness-check mixin -- the raw-jax.grad
    silent-non-finite footgun is guardable from any reactor."""
    for cls in (BatchReactor, PlugFlowReactor, BiofilmReactor,
                ParticleTrackReactor):
        assert issubclass(cls, GradientCheckMixin)


def test_check_gradient_finite_returns_finite_and_raises_on_nan(simple_model):
    reactor = aquakin.BatchReactor(
        simple_model, aquakin.SpatialConditions.uniform(T=293.15)
    )
    finite = jnp.array([1.0, 2.0, 3.0])
    # Returns the value unchanged so it composes inline.
    assert reactor.check_gradient_finite(finite) is finite
    with pytest.raises(RuntimeError, match="non-finite"):
        reactor.check_gradient_finite(jnp.array([1.0, jnp.nan]))


def test_check_gradient_finite_remedy_depends_on_dtmax(simple_model):
    """An uncapped reactor's remedy points at the dtmax cap / forward mode; a
    capped reactor's remedy notes the cap is already set."""
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    nan = jnp.array([jnp.nan])

    uncapped = aquakin.BatchReactor(simple_model, cond)          # dtmax=None
    with pytest.raises(RuntimeError, match="dtmax cap"):
        uncapped.check_gradient_finite(nan)

    capped = aquakin.BatchReactor(
        simple_model, cond,
        integrator=aquakin.IntegratorConfig(dtmax=1e-3))
    with pytest.raises(RuntimeError, match="already caps dtmax"):
        capped.check_gradient_finite(nan)


def test_check_gradient_finite_guards_a_real_gradient(simple_model):
    """End-to-end: a real reverse-mode gradient through a (finite) solve passes
    the guard, mirroring how a user wraps their own jax.grad."""
    reactor = aquakin.BatchReactor(
        simple_model, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    C0 = jnp.asarray([1.0, 0.0])

    def loss(p):
        return reactor.solve(C0, params=p, t_span=(0.0, 10.0)).C[-1, 1]

    g = reactor.check_gradient_finite(
        jax.grad(loss)(simple_model.default_parameters()), what="my gradient"
    )
    assert jnp.all(jnp.isfinite(g))
