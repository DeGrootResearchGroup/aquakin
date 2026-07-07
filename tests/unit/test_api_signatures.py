"""Public-API signature contract + calling-convention smoke (fast gate).

These guard the failure class behind the incident fixed in
``tests/integration/test_asm_family.py`` (a reactor ``solve`` signature changed
so ``params`` became keyword-only; the only tests exercising the old positional
convention were ``slow``-marked, so the break passed the PR fast gate and
surfaced only after merge). The tests here run in the **fast gate** and:

1. pin the public calling contract via :func:`inspect.signature` -- so any
   signature change to a hot method lights up the PR immediately, forcing the
   author to reconcile every caller (including ``slow``/``validation``-only
   ones) in the same change; and
2. execute the documented calling conventions on the tiny A->B model, so a
   break in how keyword arguments thread through is caught fast, not at merge.

They are deliberately cheap (the 2-species toy model), so they belong in the
fast gate where signature drift must be caught.
"""

import inspect

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.integrate.batch import BatchReactor
from aquakin.integrate.biofilm import BiofilmReactor
from aquakin.integrate.particle import ParticleTrackReactor
from aquakin.integrate.pfr import PlugFlowReactor
from aquakin.plant.plant import Plant

# Every reactor exposes solve(); these are the four shipped reactor types.
_REACTORS = [BatchReactor, PlugFlowReactor, BiofilmReactor, ParticleTrackReactor]
# solve_sensitivity is implemented on the continuous reactors only (not the
# Lagrangian particle-track reactor).
_SENS_REACTORS = [BatchReactor, PlugFlowReactor, BiofilmReactor]


@pytest.mark.parametrize("cls", _REACTORS)
def test_reactor_solve_keeps_params_keyword_only(cls):
    """The cross-reactor solve contract: ``C0`` is the leading positional and
    ``params`` is KEYWORD_ONLY. This is exactly the invariant whose violation
    caused the post-merge break -- a parameter vector passed positionally would
    otherwise land in a ``t_span``/``t_eval`` slot. Pinning it here makes any
    future reorder fail on the PR, not after merge."""
    params = inspect.signature(cls.solve).parameters
    assert "C0" in params and "params" in params, (
        f"{cls.__name__}.solve lost C0/params"
    )
    positional = [
        n for n, p in params.items()
        if n != "self"
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional and positional[0] == "C0", (
        f"{cls.__name__}.solve must take C0 first positionally; got {positional}"
    )
    assert params["params"].kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{cls.__name__}.solve 'params' must stay KEYWORD_ONLY so it can never be "
        "filled by a stray positional argument"
    )


@pytest.mark.parametrize("cls", _SENS_REACTORS)
def test_reactor_solve_sensitivity_contract(cls):
    """solve_sensitivity keeps ``sens_params`` keyword-only and still takes
    ``C0``/``params``. ``params`` is now KEYWORD_ONLY too, harmonized with
    ``solve`` so it can never be filled by a stray positional argument (the same
    invariant, and the same failure class, as ``solve``)."""
    params = inspect.signature(cls.solve_sensitivity).parameters
    assert "C0" in params and "params" in params
    assert "sens_params" in params, f"{cls.__name__}.solve_sensitivity lost sens_params"
    assert params["sens_params"].kind is inspect.Parameter.KEYWORD_ONLY
    positional = [
        n for n, p in params.items()
        if n != "self"
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional and positional[0] == "C0", (
        f"{cls.__name__}.solve_sensitivity must take C0 first positionally; got {positional}"
    )
    assert params["params"].kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{cls.__name__}.solve_sensitivity 'params' must stay KEYWORD_ONLY so it can "
        "never be filled by a stray positional argument (harmonized with solve)"
    )


def test_plant_solve_contract():
    """Plant.solve keeps the surface a calibration / dynamic run depends on:
    ``t_span`` leads the positionals, ``params`` is accepted, and the
    operationally-important ``y0`` (warm start), ``integrator`` (step config) and
    ``diff`` (autodiff config) are KEYWORD_ONLY so they can't be shifted by a
    positional argument."""
    params = inspect.signature(Plant.solve).parameters
    for name in ("params", "t_span", "t_eval", "y0", "integrator", "diff"):
        assert name in params, f"Plant.solve lost '{name}'"
    positional = [
        n for n, p in params.items()
        if n != "self"
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional[0] == "t_span", (
        f"Plant.solve must take t_span first positionally; got {positional}"
    )
    for name in ("y0", "integrator", "diff"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"Plant.solve '{name}' must stay KEYWORD_ONLY"
        )


def test_every_reactor_has_check_gradient_finite():
    """All reactors expose the reverse-gradient finiteness guard (the
    GradientCheckMixin surface), so the DIY-loss footgun is guardable anywhere."""
    for cls in _REACTORS:
        assert hasattr(cls, "check_gradient_finite")


def test_batch_solve_documented_call_forms_execute(simple_model):
    """The documented BatchReactor.solve call forms actually run on the tiny
    A->B model -- a fast behavioural smoke that catches a kwarg no longer
    threading through (which a pure signature check would miss)."""
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    r = BatchReactor(simple_model, cond)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_model.default_parameters()
    t_eval = jnp.linspace(0.0, 1.0, 3)
    # keyword params + keyword t_span/t_eval (the README/quickstart form)
    s1 = r.solve(C0, params=p, t_span=(0.0, 1.0), t_eval=t_eval)
    # positional t_span and t_eval, default params
    s2 = r.solve(C0, (0.0, 1.0), t_eval)
    assert jnp.all(jnp.isfinite(s1.C)) and jnp.all(jnp.isfinite(s2.C))
    # passing params positionally must be rejected (it is keyword-only) -- this is
    # precisely the call shape the post-merge incident used.
    with pytest.raises(TypeError):
        r.solve(C0, p, (0.0, 1.0), t_eval)


def test_batch_solve_sensitivity_executes(simple_model):
    """solve_sensitivity runs with its documented (C0, t_span, t_eval,
    params=..., sens_params=...) form and returns a finite sensitivity of the
    right shape."""
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    r = BatchReactor(simple_model, cond)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_model.default_parameters()
    sol, S = r.solve_sensitivity(
        C0, (0.0, 1.0), jnp.linspace(0.0, 1.0, 3), params=p, sens_params=[0],
    )
    assert S.shape == (3, simple_model.n_species, 1)
    assert jnp.all(jnp.isfinite(S))
    # params is keyword-only: passing it positionally now lands in the t_span
    # slot and must be rejected (the same guard as solve).
    with pytest.raises(TypeError):
        r.solve_sensitivity(C0, p, (0.0, 1.0), sens_params=[0])


def test_check_gradient_finite_guards_a_reverse_gradient(simple_model):
    """The reverse-gradient guard composes with a real jax.grad through solve
    (keyword params) -- the recommended DIY-loss pattern, fast-gate-covered."""
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    r = BatchReactor(simple_model, cond)
    C0 = jnp.asarray([1.0, 0.0])

    def loss(p):
        return r.solve(C0, params=p, t_span=(0.0, 1.0)).C[-1, 1]

    g = r.check_gradient_finite(jax.grad(loss)(simple_model.default_parameters()))
    assert jnp.all(jnp.isfinite(g))
