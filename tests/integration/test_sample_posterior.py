"""Integration tests of ``aquakin.sample_posterior`` and ``Plant.sample_posterior``:
the draws against closed-form posteriors (linear-Gaussian with a prior, a
transformed parameter, a truncated one, the Jeffreys-marginal noise level),
against the Laplace posterior where it is accurate (one rate, a rate with its
initial pool, a plant), against a quadrature reference on a banana-shaped
posterior where Laplace is not, and the errors.

Monte Carlo tolerances: a posterior mean is checked to ``4 * sd /
sqrt(ess)`` and a standard deviation to ``4 / sqrt(2 * ess)`` relative (the
standard error of a sample standard deviation of a near-Gaussian quantity is
``sd / sqrt(2 n)``), with ``ess`` the sampler's own bulk effective sample size,
which the diagnostics tests pin against closed forms. Four standard errors keep
a fixed seed comfortably clear of chance while still failing when a density
term is wrong (each test states the size of the error it must detect).
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

import aquakin
from aquakin.integrate.calibration.result import CalibrationResult
from tests.conftest import FIXTURES

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
    """With ``positive_log`` and no prior, the posterior of ``k`` itself is the
    likelihood ``N(mean(y), sigma / sqrt(n))`` truncated to the model's bounds
    ``[1e-3, 10]``. The sampler works in ``log k``, so it is right only with the
    transform's Jacobian ``e^u`` in its density: without it the density of
    ``k`` gains a factor ``1/k``. Both means are computed by quadrature on the
    support; the one without the Jacobian is more than four tolerances away."""
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

    tau = sigma / np.sqrt(len(y))
    grid = np.linspace(1e-3, 10.0, 200001)
    likelihood = norm.pdf(grid, y.mean(), tau)
    expected = np.sum(grid * likelihood) / np.sum(likelihood)
    without_jacobian = np.sum(likelihood) / np.sum(likelihood / grid)
    k = samples.params_named[K]
    tol = _mean_tol(k.std(), samples)
    assert abs(without_jacobian - expected) > 4 * tol  # the error the test must detect
    assert abs(k.mean() - expected) < tol


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
