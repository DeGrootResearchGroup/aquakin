"""Compiled solves are keyed on the adjoint's type and configuration, not its ``id``.

``DifferentiationConfig(mode="forward", method="through_solve")`` gives a solve
a fresh ``diffrax.DirectAdjoint`` on every call (and every reactor its own), so
a key that named the adjoint by ``id`` missed every time and a forward-mode
solve compiled again on each call. These tests count XLA compilations around a
repeat forward-mode solve, check the reused program's result bit for bit
against a fresh object's, and check that adjoints of different type or
configuration never share a program.
"""

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

from aquakin import BatchReactor, DifferentiationConfig, SolverSettings, SpatialConditions
from aquakin.integrate._cache import _SOLVER_CACHE, reactor_settings_key, settings_cache_key
from aquakin.integrate._config import with_adjoint

FORWARD = DifferentiationConfig(mode="forward", method="through_solve")
REVERSE = DifferentiationConfig(mode="reverse", method="through_solve")
T_EVAL = jnp.linspace(0.0, 10.0, 6)


def _key(adjoint):
    """The settings key of a solve that differs only in ``adjoint``."""
    return settings_cache_key(1e-6, 1e-8, adjoint, None, 1000)


def _plant_solve(plant, diff):
    return plant.solve((0.0, 10.0), T_EVAL, rtol=1e-8, atol=1e-10, diff=diff)


def test_repeat_forward_mode_plant_solve_compiles_nothing(make_cstr_plant, xla_compiles):
    """A second forward-mode solve of one plant compiles nothing, and returns a
    fresh plant's result bit for bit. The tank follows the closed form
    ``A = A* + (A0 - A*) exp(-(Q/V + k) t)``, ``A* = (Q/V) / (Q/V + k)``."""
    plant = make_cstr_plant()
    _plant_solve(plant, FORWARD)
    xla_compiles.clear()
    again = _plant_solve(plant, FORWARD)
    assert xla_compiles == []

    fresh = _plant_solve(make_cstr_plant(), FORWARD)
    np.testing.assert_array_equal(np.asarray(again.state), np.asarray(fresh.state))
    k = float(plant.default_parameters()[0])
    a0 = float(plant.initial_state()[0])
    a_star = 0.1 / (0.1 + k)
    expected = a_star + (a0 - a_star) * np.exp(-(0.1 + k) * np.asarray(T_EVAL))
    # An adaptive solve at rtol = 1e-8: two decades above it.
    np.testing.assert_allclose(np.asarray(again.state[:, 0]), expected, rtol=1e-6, atol=1e-9)


def test_forward_and_reverse_plant_solves_do_not_share(make_cstr_plant, xla_compiles):
    """The reverse-mode adjoint (the default ``RecursiveCheckpointAdjoint``)
    and the forward-mode ``DirectAdjoint`` compile different programs, so a
    plant that solved in one mode compiles again for the other."""
    plant = make_cstr_plant()
    _plant_solve(plant, REVERSE)
    xla_compiles.clear()
    _plant_solve(plant, FORWARD)
    assert xla_compiles != []
    assert len(plant._jit_cache) == 2


def test_fresh_forward_mode_reactors_share_one_compiled_solve(simple_model, xla_compiles):
    """Two forward-mode reactors of one model and settings each carry their own
    ``DirectAdjoint``; the second reuses the first's compiled solve and matches
    the closed form ``A = A0 exp(-k t)``."""
    conditions = SpatialConditions.uniform(T=293.15)
    C0 = simple_model.concentrations({"A": 1.0})
    params = simple_model.default_parameters()

    def solve():
        reactor = BatchReactor(simple_model, conditions, settings=SolverSettings(
            rtol=1e-9,
            atol=1e-12,
            diff=FORWARD,
        ))
        return reactor.solve(C0, (0.0, 3.0), T_EVAL / 10.0 * 3.0, params=params)

    solve()
    xla_compiles.clear()
    sol = solve()
    assert xla_compiles == []
    expected = np.exp(-float(params[0]) * np.asarray(T_EVAL) / 10.0 * 3.0)
    # An adaptive solve at rtol = 1e-9: two decades above it.
    np.testing.assert_allclose(np.asarray(sol.C[:, 0]), expected, rtol=1e-7)


def test_adjoints_of_different_configuration_key_apart(simple_model):
    """A ``RecursiveCheckpointAdjoint`` keys by its ``checkpoints``: equal
    configurations share a key, a different one (or a different adjoint type,
    or the default) does not. A reactor carrying each builds its own entry."""
    rc = diffrax.RecursiveCheckpointAdjoint
    assert _key(rc(checkpoints=4)) == _key(rc(checkpoints=4))
    assert _key(diffrax.DirectAdjoint()) == _key(diffrax.DirectAdjoint())
    distinct = [None, rc(), rc(checkpoints=4), rc(checkpoints=8), diffrax.DirectAdjoint()]
    assert len({_key(a) for a in distinct}) == len(distinct)

    reactor = BatchReactor(simple_model, SpatialConditions.uniform(T=293.15))
    keys = {reactor_settings_key(with_adjoint(reactor, a)) for a in distinct}
    assert len(keys) == len(distinct)

    C0 = simple_model.concentrations({"A": 1.0})
    before = len(_SOLVER_CACHE)
    four = with_adjoint(reactor, rc(checkpoints=4)).solve(C0, (0.0, 1.0), T_EVAL / 10.0)
    eight = with_adjoint(reactor, rc(checkpoints=8)).solve(C0, (0.0, 1.0), T_EVAL / 10.0)
    assert len(_SOLVER_CACHE) == before + 2
    # The checkpoint count changes only the reverse pass, not the values.
    np.testing.assert_array_equal(np.asarray(four.C), np.asarray(eight.C))


class _OpaqueAdjoint:
    """An adjoint-like object that is not a dataclass: its settings are not
    readable as fields."""


@pytest.mark.parametrize("make", [diffrax.BacksolveAdjoint, _OpaqueAdjoint])
def test_an_adjoint_with_unreadable_settings_is_keyed_by_identity(make):
    """An adjoint whose settings are not plain scalar fields
    (``BacksolveAdjoint`` holds a ``kwargs`` dict; a non-dataclass has no
    fields to read) keys by identity: two such objects never share a key,
    while one object always gives the same key."""
    a, b = make(), make()
    assert _key(a) == _key(a)
    assert _key(a) != _key(b)
