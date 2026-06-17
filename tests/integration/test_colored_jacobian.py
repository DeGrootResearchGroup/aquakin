"""Colored-AD Jacobian materialization for the implicit-stage solve.

The plant Jacobian is sparse (dense per-unit kinetic blocks + sparse inter-unit
coupling), so the per-step implicit operator can be formed by column compression
-- ``C`` colored Jacobian-vector products instead of ``n`` -- giving a matrix
identical to the dense one when the sparsity pattern is a superset of the real
nonzeros. These tests pin (1) the coloring/reconstruction math, (2) that the
positive-state probe yields a superset pattern, (3) that the colored solve
reproduces the default trajectory and gradient to tolerance with no step
explosion, and (4) that the setup guard catches a bad pattern and falls back.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate import colored_jacobian as cj
from aquakin.integrate.colored_jacobian import (
    ColoredVeryChord,
    build_colored_root_finder,
    colored_jacobian_max_error,
    greedy_color,
    jacobian_sparsity_pattern,
)


# --------------------------------------------------------------------------
# Synthetic sparse RHS (fast, no plant) -- the coloring/reconstruction math.
# --------------------------------------------------------------------------

def _synthetic_rhs(n=12, seed=0):
    """A nonlinear RHS whose Jacobian has a known block-sparse pattern."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for b in range(0, n - n % 4, 4):
        A[b:b + 4, b:b + 4] = rng.standard_normal((4, 4))   # dense 4-blocks
    A[0, n - 4] = 1.3
    A[n - 3, 2] = -0.7                                       # off-diagonal links
    Aj = jnp.asarray(A)

    def rhs(y):
        return jnp.tanh(Aj @ y) + 0.1 * y                   # Jacobian sparsity == A | I

    truth = (np.abs(A) > 0) | np.eye(n, dtype=bool)
    return rhs, truth


def test_greedy_color_is_structurally_orthogonal():
    _, truth = _synthetic_rhs()
    color = greedy_color(truth)
    for c in range(color.max() + 1):
        cols = np.where(color == c)[0]
        # no row may carry two columns of the same color
        assert np.all(truth[:, cols].sum(axis=1) <= 1)


def test_pattern_is_superset_of_truth():
    rhs, truth = _synthetic_rhs()
    P = jacobian_sparsity_pattern(rhs, jnp.ones(truth.shape[0]), n_probe=16, seed=1)
    assert np.all(P[truth])                       # superset: covers every true nonzero


def test_colored_reconstruction_is_exact_synthetic():
    rhs, _ = _synthetic_rhs(n=12)
    rf, n_colors = build_colored_root_finder(
        rhs, jnp.ones(12), rtol=1e-3, atol=1e-3, n_probe=16)
    assert n_colors <= 5                          # ~ widest dense block
    rng = np.random.default_rng(3)
    for _ in range(6):
        y = jnp.asarray(rng.standard_normal(12))
        assert colored_jacobian_max_error(rhs, y, rf) < 1e-10


# --------------------------------------------------------------------------
# Plant integration (BSM1) -- correctness of the wired solve.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bsm1():
    """A constant-influent BSM1 plant + warm start (built fresh per call via the
    factory so each test gets an independent instance/cache)."""
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
    from aquakin.plant.influent import InfluentSeries

    def make():
        asm1 = aquakin.load_network("asm1")
        p = build_bsm1(network=asm1)
        C0 = asm1.concentrations({
            "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
            "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0})
        inf = InfluentSeries(t=jnp.array([0.0, 100.0]),
                             Q=jnp.full((2,), BSM1_Q_AVG),
                             C=jnp.tile(C0, (2, 1)), network=asm1)
        p.add_influent("feed", inf, to="inlet_mix.fresh")
        y0 = bsm1_warm_start(p)
        return p, y0

    return make


def _rhs_at(plant, y0, params):
    """The plant RHS as y -> dy/dt at t=0 with the cached recycle map (the exact
    function the solver's chord Jacobian linearises)."""
    plant._build_state_layout()
    plant._build_parameter_layout()
    t0 = jnp.asarray(0.0)
    plant._check_recycle_map_constant(t0, y0, params)
    states0 = plant._split_state(y0)
    rmap = plant._maybe_recycle_map(t0, states0, params)
    return lambda y: plant._rhs(t0, y, params, recycle_map=rmap)


@pytest.mark.slow
def test_plant_pattern_superset_over_trajectory(bsm1):
    """The positive-probe pattern covers the J nonzeros at states the solve
    actually visits (not just the probe states)."""
    plant, y0 = bsm1()
    params = plant.default_parameters()
    rhs = _rhs_at(plant, y0, params)
    P = jacobian_sparsity_pattern(rhs, y0, n_probe=24)
    # gather real trajectory states and check each state's J nonzeros are covered
    sol = plant.solve(t_span=(0.0, 6.0), t_eval=jnp.linspace(0.0, 6.0, 60),
                      params=params, y0=y0, rtol=1e-5, atol=1e-3,
                      max_steps=2_000_000)
    fj = jax.jit(jax.jacfwd(rhs))
    for row in np.asarray(sol.state):
        J = np.asarray(fj(jnp.maximum(jnp.asarray(row), 0.0)))
        true = np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-300)
        assert np.all(P[true]), "trajectory J has a nonzero outside the pattern"


@pytest.mark.slow
def test_plant_colored_matches_dense_jacobian(bsm1):
    plant, y0 = bsm1()
    params = plant.default_parameters()
    rhs = _rhs_at(plant, y0, params)
    rf, _ = build_colored_root_finder(rhs, y0, rtol=1e-4, atol=1e-3)
    rng = np.random.default_rng(0)
    base = np.abs(np.asarray(y0)) + 1.0
    for _ in range(6):
        y = jnp.asarray(base * np.exp(rng.normal(0.0, 1.0, y0.shape[0])))
        Jscale = float(jnp.max(jnp.abs(jax.jacfwd(rhs)(y)))) + 1e-300
        assert colored_jacobian_max_error(rhs, y, rf) < 1e-8 * Jscale


@pytest.mark.slow
def test_plant_colored_solve_matches_default(bsm1):
    plant, y0 = bsm1()
    params = plant.default_parameters()
    te = jnp.linspace(0.0, 8.0, 41)
    kw = dict(t_span=(0.0, 8.0), t_eval=te, params=params, y0=y0,
              rtol=1e-5, atol=1e-3, max_steps=2_000_000)
    s_def = plant.solve(**kw)
    s_col = plant.solve(**kw, colored_jacobian=True)
    assert plant._colored_root_finder[2] is True          # guard passed
    a, b = np.asarray(s_def.state), np.asarray(s_col.state)
    assert np.all(np.isfinite(b))
    rel = np.max(np.abs(a - b) / (np.abs(a) + 1e-6))
    assert rel < 1e-3            # within-tolerance drift (exact J, slightly diff LU path)


@pytest.mark.slow
def test_plant_colored_gradient_matches_default(bsm1):
    plant, y0 = bsm1()
    base = plant.default_parameters()
    # build the colored RF concretely first (the pattern needs concrete arrays)
    plant.solve(t_span=(0.0, 0.1), t_eval=jnp.array([0.1]), params=base, y0=y0,
                rtol=1e-4, atol=1e-3, max_steps=1_000_000, colored_jacobian=True)
    assert plant._colored_root_finder[2] is True

    def loss(scale, colored):
        s = plant.solve(t_span=(0.0, 3.0), t_eval=jnp.array([3.0]),
                        params=base * scale, y0=y0, gradient="jax_adjoint",
                        rtol=1e-5, atol=1e-3, max_steps=2_000_000,
                        colored_jacobian=colored)
        return jnp.sum(s.state[-1] ** 2)

    g_def = jax.grad(lambda s: loss(s, False))(1.0)
    g_col = jax.grad(lambda s: loss(s, True))(1.0)
    assert np.isfinite(g_def) and np.isfinite(g_col)
    assert abs(g_col - g_def) / (abs(g_def) + 1e-12) < 1e-5


@pytest.mark.slow
def test_guard_falls_back_on_truncated_pattern(bsm1, monkeypatch):
    """A pattern that misses real nonzeros must be caught by the start-state
    guard, which warns and falls back to the dense solver (still correct)."""
    plant, y0 = bsm1()
    params = plant.default_parameters()

    # Force a diagonal-only (truncated) pattern -> colored J misses every
    # off-diagonal -> the guard's colored-vs-dense error is large.
    def _diag_only(rhs, y0_, **kw):
        n = jnp.asarray(y0_).shape[0]
        return np.eye(n, dtype=bool)
    monkeypatch.setattr(cj, "jacobian_sparsity_pattern", _diag_only)

    te = jnp.linspace(0.0, 5.0, 26)
    kw = dict(t_span=(0.0, 5.0), t_eval=te, params=params, y0=y0,
              rtol=1e-5, atol=1e-3, max_steps=2_000_000)
    s_def = plant.solve(**kw)
    with pytest.warns(RuntimeWarning, match="falling back to the dense solver"):
        s_col = plant.solve(**kw, colored_jacobian=True)
    assert plant._colored_root_finder[2] is False         # guard failed
    # fallback path == dense default, so trajectories match tightly
    rel = np.max(np.abs(np.asarray(s_def.state) - np.asarray(s_col.state))
                 / (np.abs(np.asarray(s_def.state)) + 1e-6))
    assert rel < 1e-10


def test_colored_rejected_with_events_and_stable_adjoint(bsm1):
    plant, y0 = bsm1()
    params = plant.default_parameters()
    ev = [aquakin.Event(at_times=[1.0])]
    with pytest.raises(ValueError, match="colored_jacobian"):
        plant.solve(t_span=(0.0, 2.0), t_eval=jnp.array([2.0]), params=params,
                    y0=y0, events=ev, colored_jacobian=True)
    with pytest.raises(ValueError, match="colored_jacobian"):
        plant.solve(t_span=(0.0, 2.0), t_eval=jnp.array([2.0]), params=params,
                    y0=y0, gradient="stable_adjoint", colored_jacobian=True)


# --------------------------------------------------------------------------
# BSM2 (slow) -- the large stiff plant the optimization targets.
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_colored_bsm2_matches_default():
    from aquakin.plant.bsm import bsm2_warm_start
    from aquakin.plant.bsm.bsm2 import (
        build_bsm2, bsm2_asm1_network, bsm2_parameters)
    from aquakin.plant.influent import load_bsm2_influent

    def make():
        asm1 = bsm2_asm1_network(); adm1 = aquakin.load_network("adm1")
        params = bsm2_parameters(asm1, adm1)
        p = build_bsm2(asm1_network=asm1, adm1_network=adm1)
        p.add_influent("feed", load_bsm2_influent("dry", asm1))
        return p, params

    p, params = make()
    y0 = bsm2_warm_start(p)
    te = jnp.linspace(0.0, 6.0, 25)
    kw = dict(t_span=(0.0, 6.0), t_eval=te, params=params, y0=y0,
              rtol=1e-4, atol=1e-3, max_steps=8_000_000)
    s_def = p.solve(**kw)
    s_col = p.solve(**kw, colored_jacobian=True)
    assert p._colored_root_finder[2] is True
    a, b = np.asarray(s_def.state), np.asarray(s_col.state)
    assert np.all(np.isfinite(b))
    rel = np.max(np.abs(a - b) / (np.abs(a) + 1e-3))
    assert rel < 2e-2            # within-tolerance drift over the dynamic run
