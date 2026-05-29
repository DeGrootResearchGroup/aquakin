"""Unit tests for the differentiable charge-balance pH solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin  # noqa: F401  (enables x64)
from aquakin.core.ph_solver import (
    charge_balance_residual,
    equilibrium_constants,
    solve_ph,
)


def _bisection_reference(totals, T_kelvin=293.15, n=200):
    """Independent NumPy bisection root of the charge-balance residual.

    Uses the same residual function as the solver but a derivative-free root
    finder, so agreement is a genuine cross-check of the Newton solver.
    """
    K = equilibrium_constants(jnp.asarray(T_kelvin))

    def f(h):
        return float(charge_balance_residual(jnp.asarray(h), K=K, **totals))

    lo, hi = 1e-14, 1.0  # f is strictly decreasing in h
    flo, fhi = f(lo), f(hi)
    assert flo > 0 > fhi, "root not bracketed"
    for _ in range(n):
        mid = np.sqrt(lo * hi)  # geometric bisection (log space)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    h = np.sqrt(lo * hi)
    return -np.log10(h)


CASES = [
    # Carbonate buffer with matching cation alkalinity -> mildly alkaline.
    dict(tot_carbonate=1.0e-3, z_cation_eq=1.0e-3),
    # Ammonia + carbonate, with a strong-anion deficit.
    dict(tot_carbonate=2.0e-3, tot_ammonia=3.0e-3, z_cation_eq=2.0e-3),
    # Sulfide + phosphate + acetate, representative sewer water.
    dict(
        tot_carbonate=1.5e-3,
        tot_acetate=5.0e-4,
        tot_ammonia=2.0e-3,
        tot_phosphate=3.0e-4,
        tot_sulfide=3.0e-4,
        strong_anion_eq=5.0e-4,
        z_cation_eq=3.28e-3,
    ),
    # Acidic case: strong-anion excess.
    dict(tot_carbonate=1.0e-3, strong_anion_eq=2.0e-3, z_cation_eq=5.0e-4),
]


@pytest.mark.parametrize("totals", CASES)
def test_matches_bisection(totals):
    pH = float(solve_ph(**totals))
    pH_ref = _bisection_reference(totals)
    assert pH == pytest.approx(pH_ref, abs=1e-6)


@pytest.mark.parametrize("totals", CASES)
def test_residual_is_zero_at_solution(totals):
    K = equilibrium_constants(jnp.asarray(293.15))
    pH = solve_ph(**totals)
    h = 10.0 ** (-pH)
    res = float(charge_balance_residual(h, K=K, **totals))
    assert abs(res) < 1e-12


def test_temperature_shifts_pH():
    totals = dict(tot_carbonate=2.0e-3, z_cation_eq=2.0e-3)
    pH_20 = float(solve_ph(**totals, T_kelvin=293.15))
    pH_10 = float(solve_ph(**totals, T_kelvin=283.15))
    # Equilibrium constants are temperature dependent, so pH must move.
    assert abs(pH_20 - pH_10) > 1e-3


def test_gradient_is_finite_and_matches_fd():
    # d(pH)/d(total ammonia): adding base (NH3 buffer) should be smooth.
    def f(nh):
        return solve_ph(tot_carbonate=2.0e-3, tot_ammonia=nh, z_cation_eq=2.0e-3)

    nh0 = 3.0e-3
    g = float(jax.grad(f)(nh0))
    assert np.isfinite(g)
    eps = 1e-7
    fd = (float(f(nh0 + eps)) - float(f(nh0 - eps))) / (2 * eps)
    assert g == pytest.approx(fd, rel=1e-4, abs=1e-4)


def test_vmap_over_states():
    carb = jnp.array([1.0e-3, 2.0e-3, 3.0e-3])
    zcat = jnp.array([1.0e-3, 2.0e-3, 3.0e-3])
    pH = jax.vmap(lambda c, z: solve_ph(tot_carbonate=c, z_cation_eq=z))(carb, zcat)
    assert pH.shape == (3,)
    assert jnp.all(jnp.isfinite(pH))
