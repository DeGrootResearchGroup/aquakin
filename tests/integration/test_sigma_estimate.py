"""Tests for the concentrated likelihood, ``calibrate(loss="nll",
sigma="estimate")``: the noise level of each output is estimated from the
residuals instead of being supplied."""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import LaplaceConfig, OptimizerConfig

T_OBS = jnp.linspace(0.1, 10.0, 200)
TRUE_K = 0.25
SIGMA_A, SIGMA_B = 0.02, 0.05


@pytest.fixture(scope="module")
def noisy(simple_model_module):
    model = simple_model_module
    reactor = aquakin.BatchReactor(model, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0 = jnp.asarray([1.0, 0.0])
    p = model.default_parameters().at[0].set(TRUE_K)
    sol = reactor.solve(C0, params=p, t_span=(0.0, 10.0), t_eval=T_OBS)
    rng = np.random.default_rng(42)
    obs_b = np.asarray(sol.C_named("B")) + SIGMA_A * rng.standard_normal(T_OBS.shape)
    obs_ab = np.stack(
        [
            np.asarray(sol.C_named("A")) + SIGMA_A * rng.standard_normal(T_OBS.shape),
            np.asarray(sol.C_named("B")) + SIGMA_B * rng.standard_normal(T_OBS.shape),
        ],
        axis=1,
    )
    return reactor, C0, jnp.asarray(obs_b), jnp.asarray(obs_ab)


@pytest.fixture(scope="module")
def simple_model_module():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "simple_model.yaml"
    return aquakin.load_model_from_file(path)


_KW = dict(free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"})


@pytest.mark.parametrize("method", ["lbfgsb", "gauss_newton"])
def test_estimated_sigma_and_std_match_known_noise(noisy, method):
    """One output with known noise 0.02: the estimate is the residual RMS at
    the fit, close to 0.02; the optimum is the least-squares one; and the
    Laplace std equals the fixed-sigma nll std at the estimated sigma, which is
    close to the std at the true sigma."""
    reactor, C0, obs_b, _ = noisy
    est = aquakin.calibrate(
        reactor,
        C0,
        obs_b,
        T_OBS,
        observed_species=["B"],
        loss="nll",
        sigma="estimate",
        optimizer=OptimizerConfig(method=method),
        laplace=LaplaceConfig(method="gauss_newton"),
        **_KW,
    )
    mse = aquakin.calibrate(
        reactor, C0, obs_b, T_OBS, observed_species=["B"], loss="mse", laplace=False, **_KW
    )
    k_hat = est.params_named["A_to_B.k"]
    assert k_hat == pytest.approx(mse.params_named["A_to_B.k"], rel=1e-5)
    assert k_hat == pytest.approx(TRUE_K, rel=0.02)

    # sigma_hat is the RMS residual at the MAP.
    p = reactor.model.default_parameters().at[0].set(k_hat)
    pred = reactor.solve(C0, params=p, t_span=(0.0, 10.0), t_eval=T_OBS).C_named("B")
    rms = float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(obs_b)) ** 2)))
    assert est.sigma_estimate.shape == (1,)
    assert float(est.sigma_estimate[0]) == pytest.approx(rms, rel=1e-5)
    assert float(est.sigma_estimate[0]) == pytest.approx(SIGMA_A, rel=0.15)

    # The std is the plug-in one, and close to the true-sigma std.
    fixed = aquakin.calibrate(
        reactor,
        C0,
        obs_b,
        T_OBS,
        observed_species=["B"],
        loss="nll",
        sigma=float(est.sigma_estimate[0]),
        laplace=LaplaceConfig(method="gauss_newton"),
        **_KW,
    )
    truth = aquakin.calibrate(
        reactor,
        C0,
        obs_b,
        T_OBS,
        observed_species=["B"],
        loss="nll",
        sigma=SIGMA_A,
        laplace=LaplaceConfig(method="gauss_newton"),
        **_KW,
    )
    sd = est.params_named_std["A_to_B.k"]
    assert sd == pytest.approx(fixed.params_named_std["A_to_B.k"], rel=1e-4)
    assert sd == pytest.approx(truth.params_named_std["A_to_B.k"], rel=0.15)
    # The reported loss is the nll at the estimated sigma.
    n = T_OBS.shape[0]
    assert est.loss == pytest.approx(n * np.log(est.sigma_estimate[0]) + 0.5 * n, rel=1e-6)


def test_estimated_sigma_per_output_and_optimizers_agree(noisy):
    """Two outputs with noise 0.02 and 0.05 get separate estimates; the
    Gauss-Newton re-estimation reaches the L-BFGS-B optimum of the concentrated
    objective."""
    reactor, C0, _, obs_ab = noisy
    common = dict(observed_species=["A", "B"], loss="nll", sigma="estimate", laplace=False)
    lb = aquakin.calibrate(reactor, C0, obs_ab, T_OBS, **common, **_KW)
    gn = aquakin.calibrate(
        reactor,
        C0,
        obs_ab,
        T_OBS,
        optimizer=OptimizerConfig(method="gauss_newton"),
        **common,
        **_KW,
    )
    assert lb.sigma_estimate[0] == pytest.approx(SIGMA_A, rel=0.15)
    assert lb.sigma_estimate[1] == pytest.approx(SIGMA_B, rel=0.15)
    assert gn.converged
    assert gn.params_named["A_to_B.k"] == pytest.approx(lb.params_named["A_to_B.k"], rel=1e-5)
    np.testing.assert_allclose(gn.sigma_estimate, lb.sigma_estimate, rtol=1e-5)
    assert gn.loss == pytest.approx(lb.loss, rel=1e-8, abs=1e-8)


def test_fd_laplace_of_concentrated_objective_is_close_to_plug_in(noisy):
    """The FD Hessian of the concentrated objective agrees with the plug-in
    Fisher (Gauss-Newton) to within the small curvature from estimating sigma."""
    reactor, C0, obs_b, _ = noisy
    kw = dict(observed_species=["B"], loss="nll", sigma="estimate", **_KW)
    gn = aquakin.calibrate(reactor, C0, obs_b, T_OBS, laplace=LaplaceConfig("gauss_newton"), **kw)
    fd = aquakin.calibrate(reactor, C0, obs_b, T_OBS, laplace=LaplaceConfig("fd"), **kw)
    assert fd.params_named_std["A_to_B.k"] == pytest.approx(
        gn.params_named_std["A_to_B.k"], rel=0.05
    )


def test_sigma_estimate_validation(noisy):
    reactor, C0, obs_b, _ = noisy
    kw = dict(observed_species=["B"], laplace=False, **_KW)
    with pytest.raises(ValueError, match='requires loss="nll"'):
        aquakin.calibrate(reactor, C0, obs_b, T_OBS, loss="mse", sigma="estimate", **kw)
    with pytest.raises(ValueError, match="estimate"):
        aquakin.calibrate(reactor, C0, obs_b, T_OBS, loss="nll", sigma="guess", **kw)


def test_profile_likelihood_on_concentrated_objective(noisy):
    """The profile of the concentrated nll rises by (N/2) log(SSE / SSE_min)
    with one output, read against the usual 1.92 threshold; its 95 % interval
    for A0 brackets the true value 1.0."""
    reactor, C0, obs_b, _ = noisy
    prof = aquakin.profile_likelihood(
        reactor,
        C0,
        obs_b,
        T_OBS,
        ["A_to_B.k"],
        grid=np.linspace(0.97, 1.03, 13),
        profile_ic="A",
        observed_species=["B"],
        loss="nll",
        sigma="estimate",
        optimizer=OptimizerConfig(n_starts=1),
        polish=False,
    )
    n = T_OBS.shape[0]
    i_min = int(np.nanargmin(prof.loss))
    sse_min = n * float(prof.fits[i_min].sigma_estimate[0]) ** 2
    for f, dl in zip(prof.fits, prof.delta_loss):
        sse = n * float(f.sigma_estimate[0]) ** 2
        assert dl == pytest.approx(0.5 * n * np.log(sse / sse_min), abs=1e-6)
    lo, hi = prof.ci
    assert lo is not None and hi is not None
    assert lo < 1.0 < hi
