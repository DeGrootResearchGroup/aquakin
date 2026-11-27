"""Integration tests of ``aquakin.sample_posterior`` and ``Plant.sample_posterior``:
the draws against closed-form posteriors (linear-Gaussian with a prior, a
transformed parameter in each ``prior_space``, a truncated one, the
Jeffreys-marginal noise level), against the Laplace posterior where it is
accurate (one rate, a rate with its initial pool, a log-linear rate, a plant),
against a quadrature reference on a banana-shaped posterior where Laplace is
not, against the profile likelihood and an importance-sampling reference on the
ozone-demand design, and the errors.

Monte Carlo tolerances: a posterior mean is checked to ``4 * sd /
sqrt(ess)`` and a standard deviation to ``4 / sqrt(2 * ess)`` relative (the
standard error of a sample standard deviation of a near-Gaussian quantity is
``sd / sqrt(2 n)``), with ``ess`` the sampler's own bulk effective sample size,
which the diagnostics tests pin against closed forms. Four standard errors keep
a fixed seed comfortably clear of chance while still failing when a density
term is wrong (each test states the size of the error it must detect). An
interval endpoint (a quantile ``q``) is checked by :func:`_quantile_band`: its
Monte Carlo error in probability is ``sqrt(q (1 - q) / ess_q)``, with ``ess_q``
the effective sample size of the indicator ``x <= x_q`` (the tail ESS of
Vehtari et al. 2021), and the reference must lie between the draws' quantiles
four such errors either side of ``q``.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_t, norm

import aquakin
from aquakin.integrate.calibration.result import CalibrationResult
from aquakin.integrate.calibration.sampling import _ess, _split_chains
from tests.conftest import EXAMPLES, FIXTURES

K = "A_to_B.k"
# A short observation grid: the linear-Gaussian tests observe a parameter
# through observation_fn, so the solve only has to be cheap.
T_SHORT = jnp.linspace(0.1, 0.5, 5)
CONFIG = aquakin.SamplerConfig(n_chains=4, n_draws=1000, n_warmup=300, seed=0)
# Fewer draws where each costs a stiff solve over the whole decay (about 2.5 ms):
# the agreement checks need an ESS of a few hundred, not a thousand.
DECAY_CONFIG = aquakin.SamplerConfig(n_chains=4, n_draws=500, n_warmup=150, seed=0)


@pytest.fixture(scope="module")
def reactor(simple_model):
    return aquakin.BatchReactor(simple_model, simple_model.default_conditions())


def _observe_k(model):
    """An observation of the rate constant itself, ``y = k``: a linear-Gaussian
    model of ``k`` with a closed-form posterior."""
    idx = model.param_index[K]
    return lambda C, p: p[idx]


def _mean_tol(sd, samples):
    return 4.0 * sd / np.sqrt(samples.ess_bulk[K])


def _std_tol(samples, name=K):
    return 4.0 / np.sqrt(2.0 * samples.ess_bulk[name])


def _quantile_band(chains, q, n_se=4.0):
    """The range a reference ``q``-quantile may take and still agree with the
    draws ``chains``, ``(n_chains, n_draws)``: the draws' quantiles at ``q -
    n_se * e`` and ``q + n_se * e``, with ``e = sqrt(q (1 - q) / ess_q)`` the
    Monte Carlo standard error of the estimated probability below the quantile
    and ``ess_q`` the effective sample size of the indicator chains ``x <=
    x_q``. Free of any density estimate, so it holds for a skewed posterior."""
    x_q = np.quantile(chains, q)
    ess_q = _ess(_split_chains((chains <= x_q).astype(float)))
    e = n_se * np.sqrt(q * (1.0 - q) / ess_q)
    return float(np.quantile(chains, max(q - e, 0.0))), float(np.quantile(chains, min(q + e, 1.0)))


def _fit_k(reactor, y, sigma, **kwargs):
    model = reactor.model
    return aquakin.calibrate(
        reactor,
        model.default_concentrations(),
        jnp.asarray(y),
        T_SHORT[: len(y)],
        [K],
        observation_fn=_observe_k(model),
        loss="nll",
        sigma=sigma,
        laplace=aquakin.LaplaceConfig(method="gauss_newton"),
        **kwargs,
    )


# --- Closed-form posteriors ---------------------------------------------------


def test_linear_gaussian_with_prior_matches_the_conjugate_posterior(reactor):
    """``y_i ~ N(k, sigma)`` with the prior ``k ~ N(m0, s0)`` has the posterior
    ``N(mu, tau)``, ``1/tau^2 = 1/s0^2 + n/sigma^2``, ``mu = tau^2 (m0/s0^2 +
    sum(y)/sigma^2)``. The draws match it, are laid out as full parameter
    vectors, and the chains have converged."""
    y = np.array([1.12, 0.94, 1.05, 1.21, 0.88])
    sigma, m0, s0 = 0.3, 0.8, 0.2
    result = _fit_k(reactor, y, sigma, transforms={K: "none"}, priors={K: (m0, s0)})
    samples = aquakin.sample_posterior(result, reactor, config=CONFIG)

    tau = (1.0 / s0**2 + len(y) / sigma**2) ** -0.5
    mu = tau**2 * (m0 / s0**2 + y.sum() / sigma**2)
    k = samples.params_named[K]
    assert abs(k.mean() - mu) < _mean_tol(tau, samples)
    assert k.std() / tau == pytest.approx(1.0, abs=_std_tol(samples))
    # Layout of posterior_draws: the free column holds the draws, the fixed
    # columns the calibrated values.
    idx = reactor.model.param_index[K]
    assert samples.draws.shape == (CONFIG.n_chains * CONFIG.n_draws, reactor.model.n_params)
    np.testing.assert_array_equal(samples.draws[:, idx], k)
    assert samples.r_hat[K] < 1.01
    assert samples.ess_bulk[K] > 400
    assert np.all((samples.acceptance_rate > 0.3) & (samples.acceptance_rate < 0.6))


def test_transform_jacobian_keeps_the_physical_posterior(reactor):
    """With ``positive_log``, no prior and ``prior_space="physical"``, the
    posterior of ``k`` itself is the likelihood ``N(mean(y), sigma / sqrt(n))``
    truncated to the model's bounds ``[1e-3, 10]``. The sampler works in ``log
    k``, so it is right only with the transform's Jacobian ``e^u`` in its
    density: without it the density of ``k`` gains a factor ``1/k``. Both
    means are computed by quadrature on the support; the one without the
    Jacobian is more than four tolerances away."""
    y = np.array([0.4, 1.6, 0.7, 1.3])
    sigma = 1.0
    result = _fit_k(
        reactor,
        y,
        sigma,
        transforms={K: "positive_log"},
        initial_params=reactor.model.default_parameters().at[0].set(1.0),
    )
    config = aquakin.SamplerConfig(n_chains=4, n_draws=2000, n_warmup=300, seed=0)
    samples = aquakin.sample_posterior(result, reactor, config=config, prior_space="physical")

    tau = sigma / np.sqrt(len(y))
    grid = np.linspace(1e-3, 10.0, 200001)
    likelihood = norm.pdf(grid, y.mean(), tau)
    expected = np.sum(grid * likelihood) / np.sum(likelihood)
    without_jacobian = np.sum(likelihood) / np.sum(likelihood / grid)
    k = samples.params_named[K]
    tol = _mean_tol(k.std(), samples)
    assert abs(without_jacobian - expected) > 4 * tol  # the error the test must detect
    assert abs(k.mean() - expected) < tol


def test_transformed_prior_space_samples_the_density_in_log_k(reactor):
    """With ``positive_log``, no prior and the default ``prior_space``, the
    posterior is the likelihood read as a density in ``u = log k``: ``exp(-n
    (e^u - mean(y))^2 / (2 sigma^2))`` on ``[log 1e-3, log 10]``, with no
    Jacobian. The draws of ``log k`` match its mean and standard deviation by
    quadrature. The physical-space density (the same times ``e^u``) has a mean
    of ``u`` more than four tolerances away, so a Jacobian left in is caught."""
    y = np.array([0.4, 1.6, 0.7, 1.3])
    sigma = 1.0
    result = _fit_k(
        reactor,
        y,
        sigma,
        transforms={K: "positive_log"},
        initial_params=reactor.model.default_parameters().at[0].set(1.0),
    )
    config = aquakin.SamplerConfig(n_chains=4, n_draws=2000, n_warmup=300, seed=0)
    samples = aquakin.sample_posterior(result, reactor, config=config)

    u = np.linspace(np.log(1e-3), np.log(10.0), 200001)
    w = np.exp(-0.5 * len(y) * (np.exp(u) - y.mean()) ** 2 / sigma**2)
    mean_u = np.sum(u * w) / np.sum(w)
    sd_u = np.sqrt(np.sum((u - mean_u) ** 2 * w) / np.sum(w))
    physical_mean_u = np.sum(u * w * np.exp(u)) / np.sum(w * np.exp(u))
    log_k = np.log(samples.params_named[K])
    tol = _mean_tol(sd_u, samples)
    assert abs(physical_mean_u - mean_u) > 4 * tol  # the error the test must detect
    assert abs(log_k.mean() - mean_u) < tol
    assert log_k.std() / sd_u == pytest.approx(1.0, abs=_std_tol(samples))


def test_transformed_prior_space_interval_is_the_laplace_interval(reactor):
    """Observing ``log k`` itself, ``y_i ~ N(log k, sigma)``, with
    ``positive_log`` makes the posterior of ``u = log k`` exactly Gaussian in
    the default prior space, ``N(mean(y), sigma / sqrt(n))``, which is the
    Laplace posterior. The draws' 95 % interval of ``log k`` has the Laplace
    endpoints, ``u_MAP -/+ 1.96 sd``, within its Monte Carlo error. In the
    physical prior space the posterior of ``u`` is shifted by ``sd^2`` = 0.36,
    outside that error at both ends."""
    y = np.array([-2.9, -2.0, -2.5, -1.8])  # about log 0.1; [1e-3, 10] is 7 sd away
    sigma = 1.2
    model = reactor.model
    idx = model.param_index[K]
    result = aquakin.calibrate(
        reactor,
        model.default_concentrations(),
        jnp.asarray(y),
        T_SHORT[: len(y)],
        [K],
        observation_fn=lambda C, p: jnp.log(p[idx]),
        transforms={K: "positive_log"},
        loss="nll",
        sigma=sigma,
        laplace=aquakin.LaplaceConfig(method="gauss_newton"),
    )
    config = aquakin.SamplerConfig(n_chains=4, n_draws=2000, n_warmup=300, seed=0)
    samples = aquakin.sample_posterior(result, reactor, config=config)

    sd = float(np.sqrt(result.posterior_cov[0, 0]))
    assert sd == pytest.approx(sigma / np.sqrt(len(y)), rel=1e-6)  # Laplace is exact here
    u_map = np.log(result.params_named[K])
    chains = samples.chains[:, :, 0]
    for q, z in ((0.025, -1.959964), (0.975, 1.959964)):
        lo, hi = _quantile_band(chains, q)
        assert lo < u_map + z * sd < hi
        assert not lo < u_map + z * sd + sd**2 < hi  # the physical-space endpoint
    assert samples.r_hat[K] < 1.01


def test_draws_stay_inside_the_fit_bounds_as_a_truncated_posterior(reactor):
    """An absolute bound of the fit is the support of the posterior: with
    ``k >= 0.8`` and the likelihood ``N(1, 0.3)`` the posterior is that normal
    truncated at 0.8, of mean ``1 + 0.3 phi(a) / (1 - Phi(a))``, ``a = -2/3``.
    Untruncated draws would average 1, 0.13 below it."""
    y = np.array([0.7, 1.3, 1.15, 0.85])
    sigma, lower = 0.6, 0.8
    result = _fit_k(
        reactor,
        y,
        sigma,
        transforms={K: "none"},
        initial_params=reactor.model.default_parameters().at[0].set(1.0),
        optimizer=aquakin.OptimizerConfig(bounds={K: (lower, 10.0)}),
    )
    samples = aquakin.sample_posterior(result, reactor, config=CONFIG)

    tau = sigma / np.sqrt(len(y))
    a = (lower - y.mean()) / tau
    expected = y.mean() + tau * norm.pdf(a) / norm.sf(a)
    k = samples.params_named[K]
    assert k.min() >= lower
    assert abs(k.mean() - expected) < _mean_tol(tau, samples)


def test_estimated_sigma_samples_the_jeffreys_marginal(reactor):
    """With ``sigma="estimate"`` the density is the concentrated likelihood,
    which is the posterior with the noise level integrated out under ``p(sigma)
    ∝ 1/sigma``: a Student t with ``n - 1`` degrees of freedom around
    ``mean(y)``, of variance ``SSE / (n (n - 3))``. The Laplace variance at
    the estimated noise, ``SSE / n^2``, is 26 % narrower in standard deviation
    for ``n = 8``, which the draws must not follow. The t has excess kurtosis
    2, which doubles the variance of a sample variance; the standard-deviation
    tolerance is ``4 / sqrt(ess)`` accordingly."""
    y = np.array([1.1, 0.9, 1.25, 0.8, 1.0, 1.15, 0.95, 0.85])
    n = len(y)
    t_obs = jnp.linspace(0.1, 0.8, n)
    model = reactor.model
    result = aquakin.calibrate(
        reactor,
        model.default_concentrations(),
        jnp.asarray(y),
        t_obs,
        [K],
        transforms={K: "none"},
        observation_fn=_observe_k(model),
        loss="nll",
        sigma="estimate",
        laplace=aquakin.LaplaceConfig(method="gauss_newton"),
    )
    config = aquakin.SamplerConfig(n_chains=4, n_draws=2000, n_warmup=300, seed=0)
    samples = aquakin.sample_posterior(result, reactor, config=config)

    sse = float(np.sum((y - y.mean()) ** 2))
    sd = np.sqrt(sse / (n * (n - 3)))
    k = samples.params_named[K]
    tol = 4.0 / np.sqrt(samples.ess_bulk[K])
    assert abs(np.sqrt(sse) / n / sd - 1.0) > 2 * tol  # Laplace is distinguishable
    assert abs(k.mean() - y.mean()) < _mean_tol(sd, samples)
    assert k.std() / sd == pytest.approx(1.0, abs=tol)


# --- Agreement with Laplace where it holds ------------------------------------


@pytest.fixture(scope="module")
def decay_data(reactor):
    """Ten noisy observations of A in the decay ``A -> B`` at ``k = 0.1``."""
    model = reactor.model
    C0 = model.default_concentrations()
    t_obs = jnp.linspace(1.0, 20.0, 10)
    truth = reactor.solve(C0, t_span=(0.0, 20.0), t_eval=t_obs).C_named("A")
    sigma = 0.02
    obs = truth + sigma * np.random.default_rng(0).standard_normal(truth.shape)
    return C0, t_obs, obs, sigma


def test_agrees_with_laplace_on_a_well_identified_rate(reactor, decay_data):
    """Ten precise observations of a first-order decay determine ``log k`` to
    about 2 %, where the posterior is close to Gaussian: the draws of ``log k``
    match the Laplace mean (the MAP) and standard deviation within their Monte
    Carlo error."""
    C0, t_obs, obs, sigma = decay_data
    result = aquakin.calibrate(
        reactor,
        C0,
        obs,
        t_obs,
        [K],
        observed_species=["A"],
        transforms={K: "positive_log"},
        loss="nll",
        sigma=sigma,
    )
    samples = aquakin.sample_posterior(result, reactor, config=DECAY_CONFIG)

    log_k = np.log(samples.params_named[K])
    sd = float(np.sqrt(result.posterior_cov[0, 0]))
    assert abs(log_k.mean() - np.log(result.params_named[K])) < _mean_tol(sd, samples)
    assert log_k.std() / sd == pytest.approx(1.0, abs=_std_tol(samples))


@pytest.fixture(scope="module")
def pool_fit(reactor, decay_data):
    """The decay fitted for ``k`` and the initial pool of A jointly."""
    C0, t_obs, obs, sigma = decay_data
    return aquakin.calibrate(
        reactor,
        C0.at[0].set(0.8),  # a starting guess for the pool, not its value
        obs,
        t_obs,
        [K],
        observed_species=["A"],
        transforms={K: "positive_log"},
        loss="nll",
        sigma=sigma,
        free_ic=aquakin.FreeICConfig(species=["A"]),
        laplace=aquakin.LaplaceConfig(include_ic=True),
    )


def test_samples_a_free_initial_pool_jointly_with_the_rate(reactor, pool_fit):
    """The pool is sampled in its log coordinate alongside ``log k``: both
    coordinates' means and standard deviations match the joint Laplace
    posterior (near Gaussian at this precision), and ``ic_draws`` are the
    pools of the sampled coordinates."""
    samples = aquakin.sample_posterior(pool_fit, reactor, config=DECAY_CONFIG)

    assert samples.coordinate_names == pool_fit.joint_names == [K, "ic[0].A"]
    theta_map = [np.log(pool_fit.params_named[K]), np.log(pool_fit.ic_named[0]["A"])]
    flat = samples.chains.reshape(-1, 2)
    for j, name in enumerate(samples.coordinate_names):
        sd = float(np.sqrt(pool_fit.joint_cov[j, j]))
        tol = 4.0 * sd / np.sqrt(samples.ess_bulk[name])
        assert abs(flat[:, j].mean() - theta_map[j]) < tol
        assert flat[:, j].std() / sd == pytest.approx(1.0, abs=_std_tol(samples, name))
    np.testing.assert_allclose(samples.ic_draws[0]["A"], np.exp(flat[:, 1]), rtol=1e-12)


# --- Where Laplace is wrong -----------------------------------------------------


@pytest.fixture(scope="module")
def two_rate_model():
    return aquakin.load_model_from_file(FIXTURES / "two_rate_decay.yaml")


def test_banana_posterior_departs_from_laplace_as_the_quadrature_does(two_rate_model):
    """Observing ``y = a + b^2`` (``a``, ``b`` the two rates, unconstrained),
    with priors ``a ~ N(0, 2)`` and ``b ~ N(0, 0.7)``, bends the posterior
    along the parabola ``a = 1 - b^2``. Laplace sees only the curvature at the
    MAP ``b = 0``: it puts ``a`` near 0.94 with a standard deviation of 0.49.
    The posterior of ``a`` lies lower (its mean about 0.5) and is wider (about
    0.74), as a 2-D quadrature of the same density gives; the draws match the
    quadrature, and the gap to Laplace is several times their tolerance."""
    model = two_rate_model
    names = ["A_to_B.k1", "A_to_B.k2"]
    ia, ib = (model.param_index[n] for n in names)
    reactor = aquakin.BatchReactor(model, model.default_conditions())
    y = np.ones(4)
    sigma, sd_b = 1.0, 0.7
    result = aquakin.calibrate(
        reactor,
        model.default_concentrations(),
        jnp.asarray(y),
        jnp.linspace(1e-3, 4e-3, 4),
        names,
        observation_fn=lambda C, p: p[ia] + p[ib] ** 2,
        loss="nll",
        sigma=sigma,
        priors={names[0]: (0.0, 2.0), names[1]: (0.0, sd_b)},
        optimizer=aquakin.OptimizerConfig(starts=[{names[0]: 0.9, names[1]: 0.1}]),
    )
    config = aquakin.SamplerConfig(n_chains=4, n_draws=5000, n_warmup=500, seed=0)
    samples = aquakin.sample_posterior(result, reactor, config=config)

    # Quadrature of the same density on a grid fine enough for its moments.
    a, b = np.meshgrid(np.linspace(-12.0, 4.0, 1601), np.linspace(-4.0, 4.0, 801), indexing="ij")
    log_p = -0.5 * len(y) * (a + b**2 - 1.0) ** 2 / sigma**2 - a**2 / 8.0 - 0.5 * (b / sd_b) ** 2
    w = np.exp(log_p - log_p.max())
    w /= w.sum()
    mean_a = float(np.sum(w * a))
    sd_a = float(np.sqrt(np.sum(w * (a - mean_a) ** 2)))

    a_draws = samples.params_named[names[0]]
    mean_tol = 4.0 * sd_a / np.sqrt(samples.ess_bulk[names[0]])
    sd_tol = _std_tol(samples, names[0])
    laplace_a = result.params_named[names[0]]
    laplace_sd = result.params_named_std[names[0]]
    # The expected departure from Laplace, well beyond the tolerances: the
    # posterior of ``a`` is lower and wider.
    assert mean_a < laplace_a - 3.0 * mean_tol
    assert sd_a > laplace_sd * (1.0 + 3.0 * sd_tol)
    # The draws follow the quadrature, not Laplace.
    assert abs(a_draws.mean() - mean_a) < mean_tol
    assert a_draws.std() / sd_a == pytest.approx(1.0, abs=sd_tol)
    assert max(samples.r_hat.values()) < 1.01


# --- Reproducibility and predictive band ----------------------------------------


@pytest.fixture(scope="module")
def k_fit(reactor):
    y = np.array([1.12, 0.94, 1.05, 1.21, 0.88])
    return _fit_k(reactor, y, 0.3, transforms={K: "positive_log"})


def test_same_seed_gives_the_same_draws(reactor, k_fit):
    """Draws are a function of the seed: the same seed reproduces them exactly,
    another seed gives other draws."""
    config = aquakin.SamplerConfig(n_chains=2, n_draws=50, n_warmup=20, seed=3)
    first = aquakin.sample_posterior(k_fit, reactor, config=config)
    again = aquakin.sample_posterior(k_fit, reactor, config=config)
    other = aquakin.sample_posterior(k_fit, reactor, config=replace(config, seed=4))
    np.testing.assert_array_equal(first.chains, again.chains)
    np.testing.assert_array_equal(first.acceptance_rate, again.acceptance_rate)
    assert not np.array_equal(first.chains, other.chains)


def test_predictive_band_propagates_given_draws(reactor, k_fit):
    """``predictive_band(draws=)`` propagates exactly the given parameter
    vectors: its percentiles are those of the closed-form decay ``A(t) =
    A0 exp(-k t)`` over the same draws, to the solver's tolerance."""
    config = aquakin.SamplerConfig(n_chains=2, n_draws=60, n_warmup=20, seed=0)
    samples = aquakin.sample_posterior(k_fit, reactor, config=config)
    C0 = reactor.model.default_concentrations()
    t_eval = jnp.linspace(0.5, 3.0, 6)
    band = k_fit.predictive_band(
        reactor, C0, t_eval, draws=samples.draws, observed_species=["A"]
    )

    curves = np.exp(-np.outer(samples.params_named[K], np.asarray(t_eval)))  # A0 = 1
    assert band.n_valid == samples.draws.shape[0]
    np.testing.assert_allclose(band.lo[:, 0], np.percentile(curves, 2.5, axis=0), rtol=1e-4)
    np.testing.assert_allclose(band.hi[:, 0], np.percentile(curves, 97.5, axis=0), rtol=1e-4)
    np.testing.assert_allclose(band.median[:, 0], np.median(curves, axis=0), rtol=1e-4)


def test_predictive_band_rejects_draws_of_the_wrong_width(reactor, k_fit):
    with pytest.raises(ValueError, match="draws has 3 columns"):
        k_fit.predictive_band(
            reactor, reactor.model.default_concentrations(), T_SHORT, draws=np.ones((5, 3))
        )


# --- Errors -----------------------------------------------------------------------


def test_refuses_a_loss_that_is_not_a_likelihood(reactor):
    model = reactor.model
    result = aquakin.calibrate(
        reactor,
        model.default_concentrations(),
        jnp.array([1.0, 1.1, 0.9]),
        T_SHORT[:3],
        [K],
        observation_fn=_observe_k(model),
        loss="mse",
    )
    with pytest.raises(ValueError, match="needs loss='nll'"):
        aquakin.sample_posterior(result, reactor, config=CONFIG)


def test_refuses_a_fit_without_a_laplace_posterior(reactor, k_fit):
    no_laplace = replace(k_fit, posterior_cov=None)
    with pytest.raises(ValueError, match="calibrate with laplace=True"):
        aquakin.sample_posterior(no_laplace, reactor, config=CONFIG)


def test_refuses_free_pools_without_their_joint_covariance(reactor, pool_fit):
    """A fit of free pools needs ``include_ic=True``; ``pool_fit``, fitted that
    way as the message advises, is sampled in
    ``test_samples_a_free_initial_pool_jointly_with_the_rate``."""
    without_joint = replace(pool_fit, joint_cov=None)
    with pytest.raises(ValueError, match=r"LaplaceConfig\(include_ic=True\)"):
        aquakin.sample_posterior(without_joint, reactor, config=CONFIG)


def test_refuses_a_result_that_did_not_come_from_a_fit(reactor):
    result = CalibrationResult(
        params=reactor.model.default_parameters(),
        params_named={K: 0.1},
        loss=0.0,
        converged=True,
        message="",
        n_iter=0,
        parameter_names=[K],
        transforms=["positive_log"],
    )
    with pytest.raises(ValueError, match="carries no fit"):
        aquakin.sample_posterior(result, reactor, config=CONFIG)


def test_refuses_a_reactor_on_another_model(k_fit, two_rate_model):
    other = aquakin.BatchReactor(two_rate_model, two_rate_model.default_conditions())
    with pytest.raises(ValueError, match="different model"):
        aquakin.sample_posterior(k_fit, other, config=CONFIG)


def _as_plant_result(result, plant):
    """``result`` with its forward model replaced by a plant's, as if it came
    from ``plant.calibrate``: what the entry points dispatch on."""
    from aquakin.plant.calibrate import _PlantForwardModel

    fm = _PlantForwardModel(
        plant=plant,
        observables=(),
        y0=None,
        integrator=None,
        diff=aquakin.DifferentiationConfig(),
        time_unit=None,
    )
    return replace(result, _fit_record=result._fit_record._replace(forward_model=fm))


def test_each_entry_point_refuses_the_other_kind_of_fit(reactor, k_fit, make_cstr_plant):
    """A reactor fit goes to ``aquakin.sample_posterior``, a plant fit to
    ``plant.sample_posterior`` on its own plant; each refusal names the right
    call."""
    plant, other_plant = make_cstr_plant(), make_cstr_plant()
    with pytest.raises(ValueError, match=r"aquakin.sample_posterior\(result, reactor\)"):
        plant.sample_posterior(k_fit, config=CONFIG)
    plant_result = _as_plant_result(k_fit, plant)
    with pytest.raises(ValueError, match=r"plant.sample_posterior\(result\)"):
        aquakin.sample_posterior(plant_result, reactor, config=CONFIG)
    with pytest.raises(ValueError, match="calibrated on a different plant"):
        other_plant.sample_posterior(plant_result, config=CONFIG)


def test_each_entry_point_refuses_an_unknown_prior_space(reactor, k_fit, make_cstr_plant):
    """``prior_space`` is checked by the shared sampler, so the reactor and the
    plant entry points give the same error."""
    plant = make_cstr_plant()
    with pytest.raises(ValueError, match=r"prior_space must be 'transformed'.*got 'log'"):
        aquakin.sample_posterior(k_fit, reactor, config=CONFIG, prior_space="log")
    with pytest.raises(ValueError, match=r"prior_space must be 'transformed'.*got 'log'"):
        plant.sample_posterior(_as_plant_result(k_fit, plant), config=CONFIG, prior_space="log")


# --- Plant ----------------------------------------------------------------------


@pytest.mark.slow  # a plant fit (~10 s, mostly compiling its adjoint) plus 600 plant solves
def test_plant_posterior_agrees_with_laplace_on_the_single_cstr(cstr_plant):
    """On the single-CSTR plant, eight precise observations of B in the tank
    determine ``log k`` to about 1.5 %, where the posterior is close to
    Gaussian: the plant draws of ``log k`` match the Laplace mean and standard
    deviation within their Monte Carlo error."""
    plant = cstr_plant
    name = plant.parameter_names()[0]
    t_obs = jnp.linspace(2.0, 40.0, 8)
    sol = plant.solve(t_span=(0.0, 40.0), t_eval=t_obs)
    sigma = 0.01
    obs = plant.stream(sol, "tank").C_named("B")
    obs = obs + sigma * np.random.default_rng(0).standard_normal(obs.shape)
    result = plant.calibrate(
        obs,
        t_obs,
        [name],
        target="tank",
        observed_channels=["B"],
        loss="nll",
        sigma=sigma,
        transforms={name: "positive_log"},
        laplace=aquakin.LaplaceConfig(method="gauss_newton"),
    )
    config = aquakin.SamplerConfig(n_chains=2, n_draws=200, n_warmup=100, seed=0)
    samples = plant.sample_posterior(result, config=config)

    log_k = np.log(samples.params_named[name])
    sd = float(np.sqrt(result.posterior_cov[0, 0]))
    assert abs(log_k.mean() - np.log(result.params_named[name])) < 4.0 * sd / np.sqrt(
        samples.ess_bulk[name]
    )
    assert log_k.std() / sd == pytest.approx(1.0, abs=_std_tol(samples, name))
    assert samples.draws.shape == (400, plant.default_parameters().shape[0])


# --- Against the profile likelihood ------------------------------------------------


def _weighted_quantile(x, w, q):
    """The ``q``-quantile of the values ``x`` under the normalized weights ``w``."""
    order = np.argsort(x)
    return float(x[order][np.searchsorted(np.cumsum(w[order]), q)])


def _importance_quantile(x, log_w, q, n_boot=200, seed=0):
    """The ``q``-quantile of ``x`` under self-normalized importance weights
    ``exp(log_w)``, and its standard error from ``n_boot`` bootstrap resamples
    of the (draw, weight) pairs."""
    w = np.exp(log_w - log_w.max())
    estimate = _weighted_quantile(x, w / w.sum(), q)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, x.size, x.size)
        boots.append(_weighted_quantile(x[i], w[i] / w[i].sum(), q))
    return estimate, float(np.std(boots))


@pytest.mark.slow  # a four-coordinate fit, two nine-point profiles and 54 000 solves
def test_ozone_demand_intervals_against_the_profile_and_the_marginal():
    """On the ozone-demand design with only O3 observed (15 samples, ``sigma``
    = 1 umol/L), ``k1``, ``k2`` and the pools D and M fitted, and a log-space
    prior of 0.3 on each pool centred off the truth (D = 12, M = 800), the
    95 % intervals with the default ``prior_space`` (measured: R-hat <= 1.006,
    bulk ESS about 1300 on four seeds) are:

    - ``k2``: MCMC 1.40e-6 to 4.64e-6, profile 1.41e-6 to 4.62e-6 (Laplace
      1.40e-6 to 4.67e-6). With the default prior space ``k2`` agrees with the
      profile, as Laplace does: both endpoints within the Monte Carlo band of
      :func:`_quantile_band`. The physical prior space moves it up by about
      the log-space variance ``sd^2`` (about 0.1; measured 1.53e-6 to
      5.02e-6), which puts the profile's lower endpoint outside that band.
    - ``k1``: the lower endpoint agrees with the profile (7.37e-3) within the
      band. The upper endpoint of the marginal is about 2.5 % above the
      profile's (1.687e-2 by importance sampling, 1.645e-2 profile): the
      posterior is curved in a way the profile, which maximizes over the
      other coordinates instead of integrating them out, and Laplace cannot
      show. The draws' upper endpoint is checked against an independent
      importance-sampling estimate of the marginal, to four combined Monte
      Carlo standard errors (the draws' quantile error and the bootstrap
      error of the importance estimate). The draws alone (an error of about
      0.012 in ``log k1``) cannot separate the two; the importance estimate
      (an error of about 0.004, against a gap of 0.025) puts the profile
      endpoint outside its own four-error band, which pins the finding.

    The importance sample is 30 000 draws of a Student t (5 degrees of
    freedom, 1.5 times the Laplace covariance) around the MAP, weighted by the
    posterior, written out here from ``reactor.solve`` (the data term plus the
    pool priors, no Jacobian), so it shares neither the sampler nor its
    density code. Its random numbers come from NumPy alone, so it does not
    move when the sampler's chains do. The reactor runs at ``rtol = atol =
    1e-6``: its O3 differs from a solve at ``1e-8`` by a few 1e-6 umol/L
    against the noise of 1, and it is four times as fast."""
    model = aquakin.load_model_from_file(EXAMPLES / "ozone_demand.yaml")
    reactor = aquakin.BatchReactor(model, model.default_conditions(), rtol=1e-6, atol=1e-6)
    C_true = model.concentrations({"O3": 50.0, "D": 15.0, "M": 1000.0})
    t_obs = jnp.asarray(
        np.r_[np.linspace(2.0, 20.0, 5), np.linspace(60.0, 600.0, 5), np.linspace(700.0, 1800.0, 5)]
    )
    truth = reactor.solve(C_true, t_span=(0.0, 1800.0), t_eval=t_obs).C_named("O3")
    obs = truth + np.random.default_rng(0).standard_normal(truth.shape)
    C_guess = model.concentrations({"O3": 50.0, "D": 12.0, "M": 800.0})  # the pool priors' centres
    rates, pools, pool_std = ["fast.k1", "slow.k2"], ["D", "M"], 0.3
    common = dict(
        observed_species=["O3"],
        loss="nll",
        sigma=1.0,
        free_ic=aquakin.FreeICConfig(pools, bounds=(1e-2, 1e5), prior_log_std=pool_std),
    )
    result = aquakin.calibrate(
        reactor,
        C_guess,
        obs,
        t_obs,
        rates,
        laplace=aquakin.LaplaceConfig(method="gauss_newton", include_ic=True),
        **common,
    )
    assert result.at_bound == {}
    assert result.joint_names == [*rates, "ic[0].D", "ic[0].M"]
    config = aquakin.SamplerConfig(n_chains=4, n_draws=5000, n_warmup=1000, seed=0)
    samples = aquakin.sample_posterior(result, reactor, config=config)
    assert max(samples.r_hat.values()) <= 1.01

    # Grid points 1.65 to 2.6 Laplace sd either side of the estimate, where the
    # profile crosses its threshold, so the linear interpolation between them
    # places the crossing to about 0.01 sd.
    offsets = np.array([-2.6, -2.25, -1.95, -1.65, 0.0, 1.65, 1.95, 2.25, 2.6])
    sd = np.sqrt(np.diag(result.joint_cov))
    profile_ci = {}
    for j, name in enumerate(rates):
        estimate = result.params_named[name]
        profile = aquakin.profile_likelihood(
            reactor,
            C_guess,
            obs,
            t_obs,
            rates,
            grid=estimate * np.exp(offsets * sd[j]),
            profile_param=name,
            anchor=estimate,
            initial_params=result.params,
            optimizer=aquakin.OptimizerConfig(n_starts=1),
            **common,
        )
        profile_ci[name] = np.log(profile.ci)

    # k2, both endpoints, and k1's lower endpoint: the profile.
    k1, k2 = samples.chains[:, :, 0], samples.chains[:, :, 1]
    for chains, q, endpoint in (
        (k2, 0.025, profile_ci["slow.k2"][0]),
        (k2, 0.975, profile_ci["slow.k2"][1]),
        (k1, 0.025, profile_ci["fast.k1"][0]),
    ):
        lo, hi = _quantile_band(chains, q)
        assert lo < endpoint < hi
    lo, hi = _quantile_band(k2, 0.025)
    assert not lo < profile_ci["slow.k2"][0] + sd[1] ** 2 < hi  # the physical-space shift

    # k1's upper endpoint: the marginal, by importance sampling.
    i_k = [model.param_index[n] for n in rates]
    i_c = [model.species_index[s] for s in pools]
    log_centre = jnp.log(C_guess[jnp.asarray(i_c)])

    @jax.jit
    def log_posterior(theta):
        p = result.params.at[jnp.asarray(i_k)].set(jnp.exp(theta[:2]))
        C0 = C_guess.at[jnp.asarray(i_c)].set(jnp.exp(theta[2:]))
        pred = reactor.solve(C0, t_span=(0.0, 1800.0), t_eval=t_obs, params=p).C_named("O3")
        return -0.5 * jnp.sum((obs - pred) ** 2) - 0.5 * jnp.sum(
            ((theta[2:] - log_centre) / pool_std) ** 2
        )

    theta_map = np.log([*(result.params_named[n] for n in rates), *result.ic_named[0].values()])
    proposal = multivariate_t(loc=theta_map, shape=1.5 * result.joint_cov, df=5, seed=1)
    x = proposal.rvs(size=30000)
    log_w = np.array([float(log_posterior(jnp.asarray(xi))) for xi in x]) - proposal.logpdf(x)
    is_upper, is_se = _importance_quantile(x[:, 0], log_w, 0.975)
    lo, hi = _quantile_band(k1, 0.975, n_se=1.0)
    mcmc_se = 0.5 * (hi - lo)  # the draws' quantile error, in log k1
    assert abs(np.quantile(k1, 0.975) - is_upper) < 4.0 * np.hypot(mcmc_se, is_se)
    # The marginal departs from the profile there, beyond the importance error.
    assert abs(profile_ci["fast.k1"][1] - is_upper) > 4.0 * is_se
