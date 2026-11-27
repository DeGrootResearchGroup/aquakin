"""Integration tests for model-discrimination design.

The rivals are first-order decay ``A -> B`` (``simple_model``,
``A = A0 exp(-k1 t)``) and second-order decay (``second_order_decay.yaml``,
``A = A0 / (1 + k2 A0 t)``), both fitted to the same early-time data of a
first-order decay. They agree over the fitted window and diverge later, so
every quantity of the criterion has a closed form: the predictions, their
sensitivities to ``k`` (``-t A0 exp(-k1 t)`` and ``-A0^2 t / (1 + k2 A0 t)^2``)
and so ``Vi = Covi Ji Ji^T``.
"""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import aquakin
from aquakin import DesignConfig, Experiment
from aquakin.integrate.discrimination import (
    _discrimination_program,
    _fit_noise,
    _fitted,
    _prepare,
    _resolve_pair,
    _setups,
)
from aquakin.integrate.experiment_design import _start_design
from tests.conftest import FIXTURES
from tests.finite_difference import central_fd

A0, K_TRUE, SIGMA = 1.0, 0.3, 0.01
T_FIT = np.linspace(0.2, 2.0, 8)
C0 = jnp.asarray([A0, 0.0])
OBS = A0 * np.exp(-K_TRUE * T_FIT) + 0.005 * np.array([1, -1, 0.5, -0.5, 1, -1, 0.3, -0.3])


@pytest.fixture(scope="module")
def second_order():
    return aquakin.load_model_from_file(FIXTURES / "second_order_decay.yaml")


def _reactor(model):
    # rtol 1e-10: the values are compared with closed forms at rel 1e-7.
    return aquakin.BatchReactor(model, aquakin.SpatialConditions.uniform(1, T=293.15), rtol=1e-10)


@pytest.fixture(scope="module")
def reactors(simple_model, second_order):
    return _reactor(simple_model), _reactor(second_order)


def _fit(reactor, sigma=SIGMA, transform=None, **kw):
    transforms = None if transform is None else {"A_to_B.k": transform}
    kw.setdefault("observed_species", ["A"])
    return aquakin.calibrate(
        reactor,
        C0,
        OBS,
        T_FIT,
        ["A_to_B.k"],
        loss="nll",
        sigma=sigma,
        transforms=transforms,
        **kw,
    )


@pytest.fixture(scope="module")
def fits(reactors):
    """Both models fitted with the known noise, ``k`` untransformed."""
    return tuple(_fit(r) for r in reactors)


@pytest.fixture(scope="module")
def models(reactors, fits):
    return tuple(zip(reactors, fits))


def _exp(times, **kw):
    return Experiment(C0, jnp.asarray(times, dtype=float), sigma=1.0, **kw)


# ---------- closed forms ----------


def _closed_form(fit1, fit2, t, noise, criterion="buzzi_ferraris", a0=A0):
    """The criterion at sampling times ``t`` from the closed-form predictions
    and sensitivities, with the sensitivity taken in each fit's coordinates."""
    t = np.asarray(t, dtype=float)
    k1, k2 = fit1.params_named["A_to_B.k"], fit2.params_named["A_to_B.k"]
    y1 = a0 * np.exp(-k1 * t)
    y2 = a0 / (1.0 + k2 * a0 * t)
    d = y1 - y2
    if criterion == "hunter_reiner":
        return float(d @ d / (2.0 * noise**2))
    J1 = -t * a0 * np.exp(-k1 * t)
    J2 = -(a0**2) * t / (1.0 + k2 * a0 * t) ** 2
    # d y / d theta = (d y / d k) (d k / d theta): 1 for "none", k for log.
    J1 = J1 * (k1 if fit1.transforms[0] == "positive_log" else 1.0)
    J2 = J2 * (k2 if fit2.transforms[0] == "positive_log" else 1.0)
    c1, c2 = float(fit1.posterior_cov[0, 0]), float(fit2.posterior_cov[0, 0])
    A = 2.0 * noise**2 * np.eye(t.size) + c1 * np.outer(J1, J1) + c2 * np.outer(J2, J2)
    return float(d @ np.linalg.solve(A, d))


def _ranked_value(models, times, **kw):
    [score] = aquakin.rank_discrimination(*models, [_exp(times)], **kw)
    return score.value


# ---------- the criterion's value ----------


@pytest.mark.parametrize(
    "transform, sigma",
    [(None, SIGMA), ("positive_log", "estimate")],
    ids=["untransformed-known-noise", "log-estimated-noise"],
)
def test_buzzi_ferraris_matches_the_closed_form(reactors, transform, sigma):
    """T = d^T (2 Sigma + V1 + V2)^{-1} d with the closed-form d and Vi, the
    sensitivities in the fit's own coordinates (``posterior_cov`` is in log k
    under positive_log), and Sigma the mean of the fits' noise variances
    (their ``sigma_estimate`` when estimated)."""
    f1, f2 = (_fit(r, sigma=sigma, transform=transform) for r in reactors)
    if sigma == "estimate":
        noise = np.sqrt(0.5 * (f1.sigma_estimate[0] ** 2 + f2.sigma_estimate[0] ** 2))
        assert f1.sigma_estimate[0] != pytest.approx(f2.sigma_estimate[0], rel=1e-3)
    else:
        noise = SIGMA
    assert list(f1.transforms) == [transform or "none"]
    times = [2.0, 5.0, 10.0]
    value = _ranked_value(tuple(zip(reactors, (f1, f2))), times)
    assert value == pytest.approx(_closed_form(f1, f2, times, noise), rel=1e-7)


def test_hunter_reiner_is_buzzi_ferraris_without_covariance(reactors, fits, models):
    """Hunter-Reiner is d^T (2 Sigma)^{-1} d: Buzzi-Ferraris with zero
    posterior covariances, and the closed form without V."""
    times = [1.0, 4.0, 8.0]
    zero = tuple(replace(f, posterior_cov=jnp.zeros((1, 1))) for f in fits)
    hr = _ranked_value(models, times, criterion="hunter_reiner")
    bf0 = _ranked_value(tuple(zip(reactors, zero)), times)
    assert hr == pytest.approx(bf0, rel=1e-12)
    assert hr == pytest.approx(_closed_form(*fits, times, SIGMA, "hunter_reiner"), rel=1e-7)
    # The covariances matter: with them the criterion is smaller.
    assert _ranked_value(models, times) < 0.99 * hr


def test_explicit_sigma_overrides_the_fits_noise(models, fits):
    """sigma= replaces the fits' noise level: Hunter-Reiner scales as 1/sigma^2."""
    times = [3.0, 9.0]
    value = _ranked_value(models, times, criterion="hunter_reiner", sigma=0.05)
    assert value == pytest.approx(_closed_form(*fits, times, 0.05, "hunter_reiner"), rel=1e-7)


def test_rank_orders_candidates_best_first(models, fits):
    """Late samples, where the models diverge, rank above early ones, and each
    score is its candidate's closed-form value."""
    early, late = [0.5, 1.0], [8.0, 10.0]
    ranked = aquakin.rank_discrimination(*models, [_exp(early), _exp(late), _exp([1.0, 3.0, 5.0])])
    assert [s.index for s in ranked] == [1, 2, 0]
    for s in ranked:
        times = (early, late, [1.0, 3.0, 5.0])[s.index]
        assert s.value == pytest.approx(_closed_form(*fits, times, SIGMA), rel=1e-7)
        assert s.improvement is None


# ---------- the optimized design ----------


@pytest.mark.parametrize("criterion", ["buzzi_ferraris", "hunter_reiner"])
def test_optimal_sample_moves_to_where_the_models_diverge(models, fits, criterion):
    """The issue's acceptance case. The fits agree over the fitted window
    (to 2) and diverge later; one sample started at the end of the window
    moves to the maximizer of the closed-form criterion, well past it."""
    res = aquakin.optimize_discrimination(
        *models, _exp([2.0]), criterion=criterion, design=DesignConfig(time_bounds=(0.1, 30.0))
    )
    best = minimize_scalar(
        lambda t: -_closed_form(*fits, [t], SIGMA, criterion),
        bounds=(2.0, 30.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    t_opt = float(res.experiments[0].t_obs[0])
    assert t_opt == pytest.approx(best.x, rel=1e-4)
    assert t_opt > 5.0
    assert res.value == pytest.approx(-best.fun, rel=1e-7)
    assert res.initial_value == pytest.approx(
        _closed_form(*fits, [2.0], SIGMA, criterion), rel=1e-7
    )
    assert res.success


def test_optimized_doses_and_spacing(models, fits):
    """With the initial concentration free as well, the design keeps its
    samples min_spacing apart, inside the bounds, and improves the criterion;
    its value is the closed form at the returned design."""
    res = aquakin.optimize_discrimination(
        *models,
        _exp([1.0, 2.0, 3.0]),
        design=DesignConfig(
            time_bounds=(0.5, 20.0), min_spacing=1.0, initial_bounds={"A": (0.5, 1.0)}
        ),
    )
    t = np.asarray(res.experiments[0].t_obs)
    assert np.all(np.diff(t) >= 1.0 - 1e-6) and t[0] >= 0.5 and t[-1] <= 20.0
    a0 = float(res.experiments[0].C0[0])
    assert 0.5 <= a0 <= 1.0
    assert res.value > 10.0 * res.initial_value
    assert res.value == pytest.approx(_closed_form(*fits, t, SIGMA, a0=a0), rel=1e-7)
    # An interior optimum in the dose: a nearby dose is worse.
    assert 0.55 < a0 < 0.95
    for other in (a0 - 0.02, a0 + 0.02):
        assert _closed_form(*fits, t, SIGMA, a0=other) < res.value


def test_criterion_gradient_matches_finite_difference(models):
    """The design gradient the optimizer uses, with respect to the scaled
    sampling times, matches a central finite difference of the criterion."""
    pair = _resolve_pair(*models, "buzzi_ferraris", None)
    exps = [_exp([3.0, 7.0])]
    setups = _setups(pair, [_prepare(exps[0], 0, pair)], [20.0])
    x0, layout, conc_lo, conc_hi = _start_design(
        exps, {}, models[0][0].model, 0, True, (0.0, 20.0), 0.0
    )
    program = _discrimination_program(setups, layout, "buzzi_ferraris")
    args = (_fitted(pair), setups[0].tobs, setups[0].C0, 0.0, 20.0, conc_lo, conc_hi)
    x0 = jnp.asarray(x0)
    _, grad = program(x0, *args)
    assert np.all(np.abs(np.asarray(grad)) > 1.0)
    for i in range(x0.size):
        # Relative step 1e-4: truncation O(h^2) ~ 1e-8 against solve error
        # 1e-10 / h ~ 1e-6 relative.
        fd = central_fd(lambda x: program(x, *args)[0], x0, i)
        assert float(grad[i]) == pytest.approx(fd, rel=1e-5)


def test_repeat_optimize_compiles_nothing(models, xla_compiles):
    """A second call of the same structure reuses the compiled criterion and
    gradient, from the same or another starting design."""
    design = DesignConfig(time_bounds=(0.1, 30.0))
    first = aquakin.optimize_discrimination(*models, _exp([2.0]), design=design)
    xla_compiles.clear()
    again = aquakin.optimize_discrimination(*models, _exp([2.0]), design=design)
    assert xla_compiles == []
    np.testing.assert_array_equal(again.experiments[0].t_obs, first.experiments[0].t_obs)
    moved = aquakin.optimize_discrimination(*models, _exp([3.0]), design=design)
    assert xla_compiles == []
    assert float(moved.experiments[0].t_obs[0]) == pytest.approx(
        float(first.experiments[0].t_obs[0]), rel=1e-4
    )


def test_repeat_rank_compiles_nothing(models, xla_compiles):
    """Ranking again with candidates of the same structure compiles nothing:
    they differ only in sampling times and share the latest one."""
    aquakin.rank_discrimination(*models, [_exp([1.0, 5.0])])
    xla_compiles.clear()
    # The same number of samples and the same latest time (the solve's end).
    aquakin.rank_discrimination(*models, [_exp([2.0, 5.0]), _exp([3.0, 4.0])])
    assert xla_compiles == []


# ---------- errors ----------


def test_unknown_criterion(models):
    with pytest.raises(ValueError, match="Unknown criterion 'T'"):
        aquakin.rank_discrimination(*models, [_exp([1.0])], criterion="T")


def test_model_must_be_a_pair(models):
    with pytest.raises(ValueError, match=r"model 2 must be a \(reactor, CalibrationResult\) pair"):
        aquakin.rank_discrimination(models[0], models[1][0], [_exp([1.0])])


def test_models_must_share_species(models):
    other = aquakin.load_model("ozone_bromate")
    reactor = aquakin.BatchReactor(other, other.default_conditions())
    with pytest.raises(ValueError, match=r"must share one species list.*model 2 has \['O3'"):
        aquakin.rank_discrimination(models[0], (reactor, models[1][1]), [_exp([1.0])])


def test_fit_must_come_from_calibrate(models, fits):
    """A result built by hand carries no fit; the fit calibrate returned works."""
    bare = replace(fits[1], _fit_record=None)
    with pytest.raises(
        ValueError, match="model 2: discrimination design needs the CalibrationResult"
    ):
        aquakin.rank_discrimination(models[0], (models[1][0], bare), [_exp([1.0])])


def test_reactor_must_be_on_the_fits_model(models, fits):
    with pytest.raises(ValueError, match="model 1: the reactor is on a different model"):
        aquakin.rank_discrimination((models[1][0], fits[0]), models[1], [_exp([1.0])])


def test_buzzi_ferraris_needs_a_laplace_posterior(reactors, fits, models):
    """Without a Laplace posterior Buzzi-Ferraris refuses; following the advice
    (Hunter-Reiner) gives the same value as with the posterior-carrying fit."""
    no_laplace = _fit(reactors[1], laplace=False)
    pair = (reactors[1], no_laplace)
    with pytest.raises(
        ValueError, match="model 2: criterion='buzzi_ferraris' needs the fit's Laplace"
    ):
        aquakin.rank_discrimination(models[0], pair, [_exp([4.0])])
    hr = _ranked_value((models[0], pair), [4.0], criterion="hunter_reiner")
    assert hr == pytest.approx(_closed_form(fits[0], no_laplace, [4.0], SIGMA, "hunter_reiner"))


def test_models_must_observe_the_same_quantities(reactors, models):
    obs_ab = np.stack([OBS, A0 - OBS], axis=1)
    both = aquakin.calibrate(
        reactors[1],
        C0,
        obs_ab,
        T_FIT,
        ["A_to_B.k"],
        observed_species=["A", "B"],
        loss="nll",
        sigma=SIGMA,
    )
    with pytest.raises(
        ValueError, match=r"model 1 observes species \['A'\], model 2 observes species \['A', 'B'\]"
    ):
        aquakin.rank_discrimination(models[0], (reactors[1], both), [_exp([1.0])])


def test_observation_functions_must_be_the_same(reactors, models, fits):
    """Fits through two different observation functions are refused, even
    with the same outputs; one shared function gives the value of observing
    the species it reads, and an experiment may name it."""

    def a_only(C, p):
        return C[0]

    def a_again(C, p):
        return 1.0 * C[0]

    def fit(reactor, fn):
        return _fit(reactor, observed_species=None, observation_fn=fn)

    f1 = fit(reactors[0], a_only)
    with pytest.raises(
        ValueError,
        match=r"observes observation_fn .*a_only, model 2 observes observation_fn .*a_again",
    ):
        aquakin.rank_discrimination(
            (reactors[0], f1), (reactors[1], fit(reactors[1], a_again)), [_exp([1.0])]
        )
    pair = ((reactors[0], f1), (reactors[1], fit(reactors[1], a_only)))
    times = [4.0, 8.0]
    expected = _ranked_value(models, times)
    assert _ranked_value(pair, times) == pytest.approx(expected, rel=1e-9)
    [named] = aquakin.rank_discrimination(*pair, [_exp(times, observation_fn=a_only)])
    assert named.value == pytest.approx(expected, rel=1e-9)


def test_fit_without_a_noise_level_needs_sigma(reactors, models, fits):
    """A fit with loss="mse" and no sigma has no noise level; passing sigma=
    as the message says gives the closed-form value at that noise."""
    mse = aquakin.calibrate(reactors[1], C0, OBS, T_FIT, ["A_to_B.k"], observed_species=["A"])
    pair = (reactors[1], mse)
    with pytest.raises(
        ValueError, match=r"no noise level of one value per observed output.*pass sigma="
    ):
        aquakin.rank_discrimination(models[0], pair, [_exp([4.0])], criterion="hunter_reiner")
    value = _ranked_value((models[0], pair), [4.0], criterion="hunter_reiner", sigma=SIGMA)
    assert value == pytest.approx(
        _closed_form(fits[0], mse, [4.0], SIGMA, "hunter_reiner"), rel=1e-7
    )


def test_differing_per_dataset_noise_is_not_one_level(reactors, models, fits):
    """Two datasets fitted at different sigmas give no single noise level per
    output; passing sigma= as the message says gives the closed-form value."""
    two = aquakin.calibrate(
        reactors[1],
        [C0, C0],
        [OBS, OBS],
        [T_FIT, T_FIT],
        ["A_to_B.k"],
        observed_species=["A"],
        loss="nll",
        sigma=[SIGMA, 2 * SIGMA],
    )
    pair = (models[0], (reactors[1], two))
    with pytest.raises(ValueError, match="no noise level of one value per observed output"):
        aquakin.rank_discrimination(*pair, [_exp([4.0])])
    value = _ranked_value(pair, [4.0], criterion="hunter_reiner", sigma=SIGMA)
    assert value == pytest.approx(
        _closed_form(fits[0], two, [4.0], SIGMA, "hunter_reiner"), rel=1e-7
    )


@pytest.mark.parametrize(
    "given, expected",
    [
        (0.02, [0.02, 0.02]),
        (np.array([0.01, 0.03]), [0.01, 0.03]),
        ([0.02, np.array([0.02, 0.02])], [0.02, 0.02]),
        ([0.02, 0.03], None),
        (np.full((5, 2), 0.02), None),
        (np.array([0.01, 0.02, 0.03]), None),
        (None, None),
    ],
    ids=[
        "scalar",
        "per-output",
        "equal-datasets",
        "differing-datasets",
        "per-observation",
        "wrong-length",
        "none",
    ],
)
def test_fit_noise_reads_one_level_per_output(given, expected):
    """The fit's noise level: its sigma broadcast to one value per output when
    it is a scalar or per-output (the same for every dataset), else none."""
    fit = SimpleNamespace(sigma_estimate=None)
    problem = SimpleNamespace(fit_sigma=given, n_observed=2)
    noise = _fit_noise(fit, problem)
    if expected is None:
        assert noise is None
    else:
        np.testing.assert_array_equal(noise, expected)


def test_experiment_must_observe_what_the_fits_observed(models):
    """An experiment that observes something else is refused; one naming the
    fits' species is the same design as one leaving it unset."""
    with pytest.raises(
        ValueError, match="experiment 1: leave observed_species and observation_fn unset"
    ):
        aquakin.rank_discrimination(*models, [_exp([1.0]), _exp([1.0], observed_species=["B"])])
    named = _ranked_value(models, [4.0, 8.0])
    [score] = aquakin.rank_discrimination(*models, [_exp([4.0, 8.0], observed_species=("A",))])
    assert score.value == named


def test_experiment_must_not_carry_a_reactor(models):
    with pytest.raises(ValueError, match="experiment 0: each model is solved on its own reactor"):
        aquakin.optimize_discrimination(*models, _exp([1.0], reactor=models[0][0]))


def test_empty_designs_are_refused(models):
    with pytest.raises(ValueError, match="experiments must contain at least one Experiment"):
        aquakin.optimize_discrimination(*models, [])
    with pytest.raises(ValueError, match="candidates must contain at least one Experiment"):
        aquakin.rank_discrimination(*models, [])


def test_design_controls_are_checked(models):
    """The controls go through optimize_experiment's checks."""
    with pytest.raises(ValueError, match="Nothing to optimise"):
        aquakin.optimize_discrimination(
            *models, _exp([1.0]), design=DesignConfig(optimize_times=False)
        )
    with pytest.raises(ValueError, match="t_obs lies outside time_bounds"):
        aquakin.optimize_discrimination(
            *models, _exp([1.0]), design=DesignConfig(time_bounds=(2.0, 5.0))
        )
