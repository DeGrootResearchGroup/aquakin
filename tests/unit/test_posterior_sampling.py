"""Unit tests of the posterior sampler's pieces that need no fit: the R-hat and
effective-sample-size diagnostics on synthetic chains with known answers, the
sampler settings, the proposal covariance, the transform log-Jacobian and the
density wrapper's handling of failed solves and of the support."""

from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import lfilter

import aquakin
from aquakin.integrate.calibration.sampling import (
    _BatchDensity,
    _ess_bulk,
    _log_jacobian,
    _proposal_factor,
    _rhat_basic,
    _split_rhat,
    _start_points,
)


def _ar1_chains(phi, n_chains=4, n_draws=5000, seed=0):
    """Stationary AR(1) chains ``x_t = phi x_{t-1} + e_t`` with unit-variance
    innovations, started from the stationary distribution."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_chains, n_draws))
    e[:, 0] /= np.sqrt(1.0 - phi**2)
    return lfilter([1.0], [1.0, -phi], e, axis=1)


# --- Diagnostics -------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_ess_bulk_matches_the_ar1_closed_form(phi):
    """The bulk ESS of stationary AR(1) chains is ``N (1 - phi) / (1 + phi)``
    (the integrated autocorrelation time of AR(1) is ``(1 + phi) / (1 -
    phi)``). One estimate from 4 x 20 000 draws has a sampling error of about
    5 % at ``phi = 0.9`` (less below), so the ratio is averaged over five
    independent replications (about 2.5 %) and checked to 10 %."""
    ratios = []
    for seed in range(5):
        x = _ar1_chains(phi, n_draws=20000, seed=seed)
        ratios.append(_ess_bulk(x) / (x.size * (1.0 - phi) / (1.0 + phi)))
    assert np.mean(ratios) == pytest.approx(1.0, rel=0.1)


def test_ess_bulk_of_chains_that_have_not_mixed_is_small():
    """Four chains of independent draws, each around its own centre (0, 3, 6,
    9 standard deviations), carry almost no information about the common
    distribution: the between-chain variance dominates every autocorrelation,
    so the ESS is a few draws per chain, not the 8000 a within-chain view
    reports."""
    x = np.random.default_rng(2).standard_normal((4, 2000)) + 3.0 * np.arange(4)[:, None]
    assert _ess_bulk(x) < 0.01 * x.size


def test_rhat_is_one_for_chains_from_one_distribution():
    """Independent chains of one distribution give R-hat within 0.01 of 1 (its
    sampling spread at 4 x 5000 draws is a few thousandths)."""
    assert _split_rhat(_ar1_chains(0.5)) == pytest.approx(1.0, abs=0.01)


def test_rhat_flags_a_shifted_chain_and_a_rescaled_chain():
    """A chain displaced by one standard deviation, or a chain three times as
    wide (same centre: only the folded R-hat sees it), raises R-hat above the
    1.01 convergence threshold."""
    x = np.random.default_rng(1).standard_normal((4, 2000))
    shifted = x.copy()
    shifted[0] += 1.0
    wide = x.copy()
    wide[0] *= 3.0
    assert _split_rhat(shifted) > 1.05
    assert _split_rhat(wide) > 1.05


def test_rhat_basic_against_a_hand_computation():
    """``sqrt(((n-1)/n W + var(chain means)) / W)`` on two 4-draw chains:
    each has within-chain variance 1/3 (ddof=1 of 0, 1, 0, 1), the means are
    0.5 and 2.5 (variance 2), so R-hat = sqrt((3/4 * 1/3 + 2) / (1/3))."""
    x = np.array([[0.0, 1.0, 0.0, 1.0], [2.0, 3.0, 2.0, 3.0]])
    expected = np.sqrt((0.75 / 3.0 + 2.0) / (1.0 / 3.0))
    assert _rhat_basic(x) == pytest.approx(expected, rel=1e-12)


def test_rhat_basic_is_nan_for_constant_chains():
    """Chains that never move have no within-chain variance; R-hat is nan
    rather than a division warning."""
    assert np.isnan(_rhat_basic(np.ones((2, 5))))


# --- Settings --------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_chains": 1}, "n_chains must be at least 2"),
        ({"n_draws": 3}, "n_draws must be at least 4"),
        ({"n_warmup": -1}, "n_warmup must be >= 0"),
    ],
)
def test_sampler_config_rejects_bad_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        aquakin.SamplerConfig(**kwargs)


# --- The proposal and the density ------------------------------------------


def test_proposal_factor_reproduces_a_full_rank_covariance():
    """``L L^T`` is the Laplace covariance itself when it is positive definite."""
    cov = np.array([[2.0, 0.6], [0.6, 0.5]])
    L = _proposal_factor(cov)
    np.testing.assert_allclose(L @ L.T, cov, rtol=1e-12)


def test_proposal_factor_gives_a_null_direction_the_largest_variance():
    """A direction with zero Laplace variance (null, or held on a bound) gets
    the largest Laplace variance, so the chains still move along it."""
    L = _proposal_factor(np.diag([4.0, 0.0]))
    np.testing.assert_allclose(L @ L.T, np.diag([4.0, 4.0]), rtol=1e-12)


@pytest.mark.parametrize(
    "transform, expected",
    [
        ("positive_log", lambda u: u),  # p = e^u, dp/du = e^u
        ("logit", lambda u: -np.log1p(np.exp(-u)) - np.log1p(np.exp(u))),  # s (1 - s)
        ("none", lambda u: 0.0),
    ],
)
def test_log_jacobian_matches_each_transform_closed_form(transform, expected):
    """``log |dp/du|`` of one transform, summed over two coordinates."""
    problem = SimpleNamespace(transforms=[transform, transform])
    u = jnp.array([-0.7, 1.3])
    value = float(_log_jacobian(problem, u))
    assert value == pytest.approx(float(expected(-0.7) + expected(1.3)), rel=1e-12, abs=1e-15)


def _failing_log_p(theta):
    """A log density whose 'solve' fails (through ``equinox.error_if``, as a
    diverging stiff solve does) for ``theta[0] > 0``."""
    value = -0.5 * jnp.sum(theta**2)
    return eqx.error_if(value, theta[0] > 0.0, "A linear solver returned non-finite output.")


def test_density_rejects_failed_solves_and_points_outside_the_support():
    """A point whose solve fails, and a point outside ``[lb, ub]``, get
    ``-inf``; the others get their density."""
    density = _BatchDensity(_failing_log_p, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
    thetas = np.array([[-1.0, 2.0], [1.0, 0.0], [-1.0, 6.0]])
    np.testing.assert_array_equal(density(thetas), [-2.5, -np.inf, -np.inf])


def test_density_lets_an_error_that_is_not_a_solve_failure_propagate():
    """Only a solve failure is a rejection; a programming error is raised."""

    def broken(theta):
        raise KeyError("not a solve failure")

    density = _BatchDensity(broken, np.array([-1.0]), np.array([1.0]))
    with pytest.raises(KeyError, match="not a solve failure"):
        density(np.array([[0.0]]))


def test_start_points_fall_back_to_the_map_where_the_density_is_not_finite():
    """A chain whose starting draw lands where the density is not finite starts
    from the MAP instead; the others keep their draw."""
    density = _BatchDensity(
        lambda th: jnp.where(th[0] > 0.0, jnp.nan, -0.5 * jnp.sum(th**2)),
        np.array([-50.0]),
        np.array([50.0]),
    )
    theta_map = np.array([-1.0])
    config = aquakin.SamplerConfig(n_chains=8)
    starts, log_p = _start_points(
        density, theta_map, np.array([[3.0]]), config, np.random.default_rng(0)
    )
    moved = starts[:, 0] != theta_map[0]
    assert 0 < moved.sum() < config.n_chains  # both kinds of start occur
    assert np.all(starts[moved, 0] <= 0.0)
    np.testing.assert_allclose(log_p, -0.5 * starts[:, 0] ** 2, rtol=1e-12)
