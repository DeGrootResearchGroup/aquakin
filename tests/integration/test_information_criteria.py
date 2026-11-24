"""Information criteria of calibrated fits (``aquakin.aic`` / ``aicc`` / ``bic``,
``CalibrationResult.log_likelihood`` and ``aquakin.compare_models``).

The first-order decay ``A -> B`` has the closed-form trajectory
``A = A0 exp(-k t)``, ``B = A0 (1 - exp(-k t))``, so the Gaussian
log-likelihood of a fit can be written out by hand at the fitted parameters and
compared with what the fit reports. The single-CSTR plant has a closed form too
(``conftest.make_cstr_plant``).

Tolerances: the fits solve at the default ``rtol = 1e-6``, so a prediction
(of order 1) differs from the closed form by at most about ``PRED_ERR``. The
log-likelihood moves by ``r / s**2`` per unit change of a prediction, so it can
differ from the hand value by up to ``PRED_ERR * sum(|r| / s**2)``, which is
what :func:`_gauss_loglik` returns as its tolerance.
"""

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import FreeICConfig, OptimizerConfig
from tests.conftest import FIXTURES

K_TRUE = 0.1
SIGMA = 0.02
T_OBS = np.linspace(1.0, 30.0, 10)
K = "A_to_B.k"
PRED_ERR = 1e-6  # the solves' rtol, on predictions of order 1


def _decay(k, t, a0=1.0):
    """``(n_t, 2)`` closed-form ``[A, B]`` of the decay at rate ``k``."""
    a = a0 * np.exp(-k * np.asarray(t))
    return np.stack([a, a0 - a], axis=1)


def _gauss_loglik(obs, pred, sigma):
    """Hand-written Gaussian log-likelihood over the non-NaN entries, and its
    tolerance (see the module docstring)."""
    obs = np.asarray(obs, dtype=float)
    seen = ~np.isnan(obs)
    r = (obs - pred)[seen]
    s = np.broadcast_to(sigma, obs.shape)[seen]
    value = float(np.sum(-0.5 * np.log(2.0 * np.pi * s**2) - r**2 / (2.0 * s**2)))
    return value, PRED_ERR * float(np.sum(np.abs(r) / s**2))


def _concentrated_loglik(obs, pred):
    """Hand-written log-likelihood at each column's MLE noise ``SSE_j / n_j``,
    and its tolerance (the derivative in a prediction is ``r / s2_j`` here too)."""
    value = tol = 0.0
    for j in range(obs.shape[1]):
        r = obs[:, j] - pred[:, j]
        s2 = float(np.sum(r**2)) / r.size
        value += -0.5 * r.size * (math.log(2.0 * math.pi * s2) + 1.0)
        tol += PRED_ERR * float(np.sum(np.abs(r))) / s2
    return value, tol


@pytest.fixture(scope="module")
def reactor(simple_model):
    return aquakin.BatchReactor(simple_model, simple_model.default_conditions())


@pytest.fixture(scope="module")
def data():
    """Noisy observations of A and B at ``K_TRUE`` (seeded)."""
    rng = np.random.default_rng(1)
    return _decay(K_TRUE, T_OBS) + SIGMA * rng.standard_normal((T_OBS.size, 2))


def _fit(reactor, obs, **kw):
    kw.setdefault("loss", "nll")
    kw.setdefault("laplace", False)
    kw.setdefault("use_priors", False)
    C0 = kw.pop("C0", reactor.model.default_concentrations())
    t_obs = kw.pop("t_obs", jnp.asarray(T_OBS))
    return aquakin.calibrate(
        reactor,
        C0,
        obs,
        t_obs,
        kw.pop("free_params", [K]),
        observed_species=kw.pop("observed_species", ["A", "B"]),
        transforms=kw.pop("transforms", {K: "positive_log"}),
        **kw,
    )


@pytest.fixture(scope="module")
def known_fit(reactor, data):
    return _fit(reactor, jnp.asarray(data), sigma=jnp.asarray(SIGMA))


def test_known_sigma_log_likelihood_is_the_hand_gaussian_at_the_mle(known_fit, data):
    """With a known sigma, ``log_likelihood`` is the complete Gaussian
    log-likelihood (with its ``-(N/2) log(2 pi)``) at the fitted rate, that rate
    maximizes it, and AIC / AICc / BIC are the textbook formulas with k = 1,
    n = 20."""
    k_hat = known_fit.params_named[K]
    expected, tol = _gauss_loglik(data, _decay(k_hat, T_OBS), SIGMA)
    assert known_fit.log_likelihood == pytest.approx(expected, abs=tol)
    # The fitted rate is the maximum of the hand-written likelihood.
    for step in (0.99, 1.01):
        assert _gauss_loglik(data, _decay(step * k_hat, T_OBS), SIGMA)[0] < expected
    assert (known_fit.n_estimated, known_fit.n_observations) == (1, 20)
    assert aquakin.aic(known_fit) == pytest.approx(2.0 - 2.0 * expected, abs=2 * tol)
    assert aquakin.bic(known_fit) == pytest.approx(math.log(20) - 2.0 * expected, abs=2 * tol)
    assert aquakin.aicc(known_fit) == pytest.approx(
        2.0 - 2.0 * expected + 2.0 * 1 * 2 / (20 - 1 - 1), abs=2 * tol
    )


def test_loss_is_not_the_negative_log_likelihood(known_fit):
    """The reported loss omits ``(N/2) log(2 pi)``: with no prior, ``-loss`` and
    ``log_likelihood`` differ by exactly that constant (N = 20)."""
    gap = -known_fit.loss - known_fit.log_likelihood
    assert gap == pytest.approx(10.0 * math.log(2.0 * math.pi), rel=1e-12)


def test_sigma_estimate_counts_one_noise_level_per_output(reactor, data, known_fit):
    """With ``sigma="estimate"`` the log-likelihood is at each output's MLE
    noise level, and k counts the rate plus one noise level per observed output
    (1 + 2 here, against 1 with a known sigma)."""
    fit = _fit(reactor, jnp.asarray(data), sigma="estimate")
    expected, tol = _concentrated_loglik(data, _decay(fit.params_named[K], T_OBS))
    assert fit.log_likelihood == pytest.approx(expected, abs=tol)
    assert fit.n_estimated == known_fit.n_estimated + 2 == 3
    assert aquakin.aic(fit) == pytest.approx(6.0 - 2.0 * expected, abs=2 * tol)


def test_missing_observations_are_not_counted(reactor, data):
    """NaN entries are left out of n and of the log-likelihood."""
    obs = np.array(data)
    obs[[0, 3, 7], [1, 0, 1]] = np.nan
    fit = _fit(reactor, jnp.asarray(obs), sigma=jnp.asarray(SIGMA))
    assert fit.n_observations == 17
    expected, tol = _gauss_loglik(obs, _decay(fit.params_named[K], T_OBS), SIGMA)
    assert fit.log_likelihood == pytest.approx(expected, abs=tol)
    assert aquakin.bic(fit) == pytest.approx(math.log(17) - 2.0 * expected, abs=2 * tol)


def test_prior_is_excluded_from_the_log_likelihood(reactor, data, known_fit):
    """A strong prior moves the fit off the MLE; ``log_likelihood`` is still the
    data likelihood at the fitted rate, the loss carries the prior penalty on
    top, and the criteria warn. Refitting as the warning advises gives the
    prior-free fit, whose likelihood is higher and which does not warn."""
    prior = (0.2, 0.003)  # the data alone determine k to about 1e-3
    fit = _fit(reactor, jnp.asarray(data), sigma=jnp.asarray(SIGMA), priors={K: prior})
    k_map = fit.params_named[K]
    assert k_map > 1.1 * known_fit.params_named[K]  # pulled towards the prior
    expected, tol = _gauss_loglik(data, _decay(k_map, T_OBS), SIGMA)
    assert fit.log_likelihood == pytest.approx(expected, abs=tol)
    penalty = 0.5 * ((k_map - prior[0]) / prior[1]) ** 2
    assert fit.loss == pytest.approx(-expected - 10.0 * math.log(2.0 * math.pi) + penalty, abs=tol)
    with pytest.warns(UserWarning, match="objective included a prior"):
        aquakin.aic(fit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert known_fit.log_likelihood > fit.log_likelihood
        aquakin.aic(known_fit)


def test_free_initial_pools_are_counted_per_dataset(reactor):
    """A fitted initial pool adds one estimated quantity per dataset; the
    likelihood is at the fitted pools (``B = A0 (1 - exp(-k t))``)."""
    a0 = [1.0, 0.6]
    rng = np.random.default_rng(2)
    obs = [_decay(K_TRUE, T_OBS, a) + SIGMA * rng.standard_normal((10, 2)) for a in a0]
    base = reactor.model.default_concentrations()
    fit = _fit(
        reactor,
        [jnp.asarray(o) for o in obs],
        C0=[base, base.at[0].set(0.8)],
        t_obs=[jnp.asarray(T_OBS)] * 2,
        sigma=jnp.asarray(SIGMA),
        free_ic=FreeICConfig(["A"], bounds=(0.1, 10.0)),
    )
    assert (fit.n_estimated, fit.n_observations) == (3, 40)
    k_hat = fit.params_named[K]
    parts = [
        _gauss_loglik(o, _decay(k_hat, T_OBS, pools["A"]), SIGMA)
        for o, pools in zip(obs, fit.ic_named)
    ]
    expected, tol = sum(p[0] for p in parts), sum(p[1] for p in parts)
    assert fit.log_likelihood == pytest.approx(expected, abs=tol)


@pytest.fixture(scope="module")
def two_path_model(tmp_path_factory):
    """The decay with a second-order path added, ``k2 [A]^2``: nests the
    first-order model at ``k2 = 0``."""
    text = (FIXTURES / "simple_model.yaml").read_text()
    text = text.replace("name: simple_decay", "name: two_path_decay")
    text += (
        "  - name: A_sq\n"
        '    description: "Second-order decay."\n'
        '    rate: "k2 * [A] * [A]"\n'
        "    parameters:\n"
        "      k2:\n"
        "        value: 0.0\n"
        '        units: "L mol-1 s-1"\n'
        "    stoichiometry:\n"
        "      A: -1\n"
        "      B: +1\n"
    )
    path = tmp_path_factory.mktemp("models") / "two_path.yaml"
    path.write_text(text)
    model = aquakin.load_model_from_file(path)
    return aquakin.BatchReactor(model, model.default_conditions())


# Two fits per case, one of a model compiled only here: about 12 s per case.
@pytest.mark.slow
@pytest.mark.parametrize("k2_true, preferred", [(0.0, "first_order"), (0.3, "two_path")])
def test_bic_prefers_the_true_structure(reactor, two_path_model, k2_true, preferred):
    """Fits of the first-order model and of the two-path model that nests it:
    on data from the first-order model BIC prefers it (the extra rate buys too
    little likelihood), on data from the two-path model it prefers that. The
    nested model, started from the simpler one's fit, reaches at least its
    likelihood; the table's deltas and weights follow from the BICs."""
    t = np.linspace(1.0, 30.0, 20)
    sol = two_path_model.solve(
        two_path_model.model.default_concentrations(),
        t_span=(0.0, 30.0),
        t_eval=jnp.asarray(t),
        params=two_path_model.model.parameter_values({"A_sq.k2": k2_true}),
    )
    rng = np.random.default_rng(3)
    obs = np.asarray(sol.C) + SIGMA * rng.standard_normal((t.size, 2))
    common = {"t_obs": jnp.asarray(t), "sigma": jnp.asarray(SIGMA)}
    simple = _fit(reactor, jnp.asarray(obs), **common)
    k_simple = simple.params_named[K]
    with warnings.catch_warnings():
        # On first-order data the best k2 is at or below 0, so it may end on its
        # lower bound; the information criteria do not use the Laplace std.
        warnings.filterwarnings("ignore", "calibrate: .* optimiser bound", RuntimeWarning)
        nested = _fit(
            two_path_model,
            jnp.asarray(obs),
            free_params=[K, "A_sq.k2"],
            transforms={K: "positive_log", "A_sq.k2": "none"},
            initial_params=two_path_model.model.parameter_values({K: k_simple}),
            # k2 may go a little negative, but not below -k_min = -1e-3 (the
            # declared lower bound of k), where A could grow and blow up.
            optimizer=OptimizerConfig(bounds={"A_sq.k2": (-5e-4, 5.0)}),
            **common,
        )
    _value, tol = _gauss_loglik(obs, _decay(k_simple, t), SIGMA)
    assert nested.log_likelihood >= simple.log_likelihood - 2 * tol
    fits = {"first_order": simple, "two_path": nested}
    table = aquakin.compare_models(fits, criterion="bic")
    assert table.best == preferred
    other = "two_path" if preferred == "first_order" else "first_order"
    assert table.delta[other] == pytest.approx(
        aquakin.bic(fits[other]) - aquakin.bic(fits[preferred]), rel=1e-12
    )
    w = math.exp(-0.5 * table.delta[other])
    assert table.weights[preferred] == pytest.approx(1.0 / (1.0 + w), rel=1e-12)
    assert table.n_estimated == {"first_order": 1, "two_path": 2}


def test_compare_models_aic_table(known_fit, reactor, data):
    """The AIC table: values, names best first, Akaike weights from the deltas."""
    est = _fit(reactor, jnp.asarray(data), sigma="estimate")
    table = aquakin.compare_models({"known": known_fit, "est": est})
    values = {"known": aquakin.aic(known_fit), "est": aquakin.aic(est)}
    best = min(values, key=values.get)
    assert table.names[0] == best == table.best
    assert table.values == values
    assert table.n_observations == 20
    raw = {n: math.exp(-0.5 * (v - values[best])) for n, v in values.items()}
    for name in values:
        assert table.delta[name] == pytest.approx(values[name] - values[best], abs=1e-12)
        assert table.weights[name] == pytest.approx(raw[name] / sum(raw.values()), rel=1e-12)
    assert table.log_likelihood["known"] == known_fit.log_likelihood


def test_mse_fit_is_refused_and_the_advice_works(reactor, data):
    """An "mse" fit has no likelihood; refitting with sigma="estimate", as the
    error advises, gives the least-squares AIC,
    ``2k + sum_j n_j (log(2 pi SSE_j / n_j) + 1)`` with k = 1 + 2."""
    fit = _fit(reactor, jnp.asarray(data), loss="mse")
    assert fit.log_likelihood is None
    with pytest.raises(ValueError, match="no log-likelihood: information criteria need"):
        aquakin.aic(fit)
    refit = _fit(reactor, jnp.asarray(data), sigma="estimate")
    pred = _decay(refit.params_named[K], T_OBS)
    sse = np.sum((np.asarray(data) - pred) ** 2, axis=0)
    expected_aic = 2 * 3 + float(np.sum(10 * (np.log(2 * np.pi * sse / 10) + 1)))
    _value, tol = _concentrated_loglik(data, pred)
    assert aquakin.aic(refit) == pytest.approx(expected_aic, abs=2 * tol)


def test_aicc_needs_more_observations_than_parameters(reactor):
    """Two observations of one output with an estimated sigma: n = k = 2."""
    obs = jnp.asarray(_decay(K_TRUE, [5.0, 10.0])[:, 1])
    fit = _fit(
        reactor, obs, t_obs=jnp.asarray([5.0, 10.0]), observed_species=["B"], sigma="estimate"
    )
    with pytest.raises(ValueError, match="AICc is undefined for 2 observations"):
        aquakin.aicc(fit)


def test_compare_models_refuses_different_data(known_fit, reactor, data):
    """Fits of different observations, an unknown criterion, a single fit and a
    result that carries no record of its data are each refused."""
    other = _fit(reactor, jnp.asarray(data + 1e-3), sigma=jnp.asarray(SIGMA))
    with pytest.raises(ValueError, match=r"fits \['b'\] are not of the same observations"):
        aquakin.compare_models({"a": known_fit, "b": other})
    with pytest.raises(ValueError, match="criterion must be one of"):
        aquakin.compare_models({"a": known_fit, "b": known_fit}, criterion="dic")
    with pytest.raises(ValueError, match="needs at least two fits; got 1"):
        aquakin.compare_models({"a": known_fit})
    bare = aquakin.CalibrationResult(
        params=known_fit.params,
        params_named=known_fit.params_named,
        loss=known_fit.loss,
        converged=True,
        message="",
        n_iter=0,
        parameter_names=known_fit.parameter_names,
        transforms=known_fit.transforms,
        log_likelihood=known_fit.log_likelihood,
        n_observations=20,
        n_estimated=1,
    )
    with pytest.raises(ValueError, match=r"fits \['bare'\] carry no record"):
        aquakin.compare_models({"a": known_fit, "bare": bare})


def test_plant_fit_log_likelihood_is_the_hand_gaussian(cstr_plant):
    """The same statistics from ``Plant.calibrate`` on the single-CSTR plant,
    whose effluent is ``B = 1 - A``, ``A = A_ss + (1 - A_ss) exp(-(0.1 + k) t)``
    with ``A_ss = 0.1 / (0.1 + k)``. The likelihood is checked at whatever rate
    the (short) fit returns, so it does not depend on convergence."""
    param = "simple_decay.A_to_B.k"
    t = np.linspace(5.0, 50.0, 8)

    def effluent_b(k):
        a_ss = 0.1 / (0.1 + k)
        return 1.0 - (a_ss + (1.0 - a_ss) * np.exp(-(0.1 + k) * t))

    rng = np.random.default_rng(4)
    obs = effluent_b(K_TRUE) + SIGMA * rng.standard_normal(t.size)
    fit = cstr_plant.calibrate(
        jnp.asarray(obs),
        jnp.asarray(t),
        [param],
        target="tank",
        observed_channels=["B"],
        t_span=(0.0, 50.0),
        transforms={param: "positive_log"},
        loss="nll",
        sigma=jnp.asarray(SIGMA),
        laplace=False,
        optimizer=OptimizerConfig(max_iter=2),
    )
    pred = effluent_b(fit.params_named[param])
    expected, tol = _gauss_loglik(obs[:, None], pred[:, None], SIGMA)
    assert fit.log_likelihood == pytest.approx(expected, abs=tol)
    assert (fit.n_estimated, fit.n_observations) == (1, 8)
    assert aquakin.bic(fit) == pytest.approx(math.log(8) - 2.0 * expected, abs=2 * tol)
