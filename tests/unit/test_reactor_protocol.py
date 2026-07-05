"""The Reactor / ConditionedReactor structural-type contracts.

These are documentation-as-types: the tightened protocols declare what the
``sensitivity`` / ``fit`` / ``calibrate`` consumers actually rely on (the
model, the five solver settings, and ``solve`` / ``conditions``), so the
runtime ``isinstance`` checks below pin the intended taxonomy.
"""

import jax.numpy as jnp

import aquakin
from aquakin.integrate._common import ConditionedReactor, Reactor
from aquakin.integrate.particle import ParticleTrackReactor, Track


def _conds():
    return aquakin.SpatialConditions.uniform(T=293.15)


def test_batch_is_a_conditioned_reactor(simple_model):
    r = aquakin.BatchReactor(simple_model, _conds())
    assert isinstance(r, Reactor)
    assert isinstance(r, ConditionedReactor)


def test_pfr_is_a_conditioned_reactor(simple_model):
    r = aquakin.PlugFlowReactor(simple_model, _conds(), n_points=3,
                                length=1.0, velocity=1.0)
    assert isinstance(r, ConditionedReactor)


def test_biofilm_is_a_conditioned_reactor(simple_model):
    r = aquakin.BiofilmReactor(
        simple_model, _conds(), n_layers=2, thickness=1e-3,
        area_per_volume=50.0, diffusivity=1e-4, boundary_layer=1e-4,
    )
    assert isinstance(r, ConditionedReactor)


def test_particle_reactor_is_a_reactor_but_not_conditioned(simple_model):
    """A particle reactor carries a track, not conditions: a Reactor (so it can
    be fit), but not a ConditionedReactor (no condition-field gradients)."""
    track = Track(t=jnp.linspace(0.0, 1.0, 4),
                  fields={"T": jnp.full(4, 293.15)})
    r = ParticleTrackReactor(simple_model, track)
    assert isinstance(r, Reactor)
    assert not isinstance(r, ConditionedReactor)


def test_cfd_reactor_is_not_a_reactor(simple_model):
    """CFDReactor exposes step(), not solve(), so it is deliberately not a
    Reactor -- it is not consumable by sensitivity / fit / calibrate."""
    r = aquakin.CFDReactor(simple_model)
    assert not isinstance(r, Reactor)
    assert not isinstance(r, ConditionedReactor)
