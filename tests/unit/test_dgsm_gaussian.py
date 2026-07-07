"""Gaussian-input DGSM: transform-space sampling + the Poincare-constant bound.

A DGSM screen can use the parameter's actual (prior) input distribution, not only
a uniform box: the Sobol total-index bound generalizes to ``S_j^tot <=
poincare_j * nu_j / Var(g)`` with ``poincare_j`` the Poincare constant of the
input measure -- ``(b_j-a_j)^2/pi^2`` for a uniform input, ``sigma_j^2`` for a
normal one (Sobol & Kucherenko 2010, Sec. 8; Lamboni et al. 2013, Thm 3.1). For a
positive (rate) or bounded (fraction) parameter the "normal input" lives in the
calibration-transform space, so sampling and the chain rule are taken there.
"""
import numpy as np
import pytest

from aquakin.integrate._qmc import _sobol_normal_sample
from aquakin.plant.sensitivity import _dgsm_aggregate, _to_z, _from_z, _dtheta_dz


def test_sobol_normal_sample_recovers_mean_std():
    m = np.array([np.log(5.0), 0.0])
    s = np.array([0.3, 1.0])
    Z, n = _sobol_normal_sample(m, s, 2, 4096, 0)
    assert n == 4096
    assert np.allclose(Z.mean(axis=0), m, atol=3e-3)
    assert np.allclose(Z.std(axis=0), s, rtol=3e-2)


@pytest.mark.parametrize("kind,theta",
                         [("positive_log", 1.7), ("logit", 0.3), ("none", 2.0)])
def test_transform_roundtrip(kind, theta):
    assert _from_z(_to_z(theta, kind), kind) == pytest.approx(theta, abs=1e-12)


def test_poincare_reproduces_uniform_when_constant_is_uniform():
    # _dgsm_aggregate with poincare = (b-a)^2/pi^2 is identical to the uniform
    # path -- the Poincare argument is a strict generalization, not a new formula.
    rng = np.random.default_rng(0)
    N, m, k = 200, 2, 3
    grad_sq = rng.random((N, m, k))
    outputs = rng.random((N, m))
    rng2 = np.array([1.0, 4.0, 0.25])
    b_unif, *_ = _dgsm_aggregate(grad_sq, outputs, rng2)
    b_poin, *_ = _dgsm_aggregate(grad_sq, outputs, rng2, poincare=rng2 / np.pi ** 2)
    assert np.allclose(b_unif, b_poin, equal_nan=True)


def test_gaussian_dgsm_single_lognormal_input_bound_near_one():
    # One log-normal input, g = c*theta: the total-index bound is
    # exp(s^2) s^2 / (exp(s^2)-1) ~ 1 (S_tot=1 for the sole input), validating the
    # transform-space chain rule and the sigma^2 Poincare constant together.
    s = 0.3
    Z, n = _sobol_normal_sample(np.array([np.log(5.0)]), np.array([s]), 1, 8192, 0)
    theta = _from_z(Z[:, 0], "positive_log")
    c = 2.0
    g = c * theta
    dg_dz = np.full(n, c) * _dtheta_dz(theta, "positive_log")     # dg/dtheta * dtheta/dz
    grad_sq = (dg_dz ** 2).reshape(n, 1, 1)
    bound, *_ = _dgsm_aggregate(grad_sq, g.reshape(n, 1), rng2=None,
                                poincare=np.array([s ** 2]))
    analytic = np.exp(s ** 2) * s ** 2 / (np.exp(s ** 2) - 1.0)
    assert float(bound[0, 0]) == pytest.approx(analytic, rel=0.02)
    assert float(bound[0, 0]) >= 1.0           # a valid Sobol total-index upper bound
