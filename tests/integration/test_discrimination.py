"""Integration tests for model-discrimination design.

The rivals are first-order decay ``A -> B`` (``simple_model``,
``A = A0 exp(-k1 t)``) and second-order decay (``second_order_decay.yaml``,
``A = A0 / (1 + k2 A0 t)``), both fitted to the same early-time data of a
first-order decay. They agree over the fitted window and diverge later, so
every quantity of the criterion has a closed form: the predictions, their
sensitivities to ``k`` (``-t A0 exp(-k1 t)`` and ``-A0^2 t / (1 + k2 A0 t)^2``)
and so ``Vi = Covi Ji Ji^T``.

Compiling the nested forward-mode solves and the fits dominates the cost, so
the fits and the reactors are module fixtures every test shares. The fits run
without a Laplace posterior and are given a fixed ``posterior_cov``: the
criterion uses whatever covariance the fit carries, and the closed form is
evaluated with the same one. Only the tests that compare with a closed form at
rel 1e-7 solve on the tight (``rtol=1e-10``) reactors; the others use the
default ``rtol=1e-6``.
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
# Posterior variances of k for the two fits, about their Laplace values: large
# enough that the prediction covariance changes the criterion visibly.
COV = (2.0e-5, 4.6e-5)
TIMES = [2.0, 5.0, 10.0]


def _a_only(C, p):
    return C[0]


@pytest.fixture(scope="module")
def second_order():
    return aquakin.load_model_from_file(FIXTURES / "second_order_decay.yaml")


def _reactor(model, rtol=1e-6):
    return aquakin.BatchReactor(model, aquakin.SpatialConditions.uniform(1, T=293.15), settings=aquakin.SolverSettings(
        rtol=rtol,
    ))


@pytest.fixture(scope="module")
def reactors(simple_model, second_order):
    return _reactor(simple_model), _reactor(second_order)


@pytest.fixture(scope="module")
def tight(simple_model, second_order):
    """Reactors for the closed-form comparisons at rel 1e-7."""
    return _reactor(simple_model, 1e-10), _reactor(second_order, 1e-10)


def _fit(reactor, **kw):
    kw = {"observed_species": ["A"], "loss": "nll", "sigma": SIGMA, **kw}
    return aquakin.calibrate(reactor, C0, OBS, T_FIT, ["A_to_B.k"], laplace=False, **kw)


@pytest.fixture(scope="module")
def raw_fits(reactors):
    """Both models fitted with the known noise, without a Laplace posterior."""
    return tuple(_fit(r) for r in reactors)


@pytest.fixture(scope="module")
def fits(raw_fits):
    return tuple(replace(f, posterior_cov=jnp.asarray([[c]])) for f, c in zip(raw_fits, COV))


@pytest.fixture(scope="module")
def models(reactors, fits):
    return tuple(zip(reactors, fits))


@pytest.fixture(scope="module")
def tight_models(tight, fits):
    return tuple(zip(tight, fits))


@pytest.fixture(scope="module")
def fn_fit(reactors):
    """Model 2 fitted through ``observation_fn`` by ``loss="mse"`` without a
    sigma, so with no noise level."""
    return _fit(reactors[1], loss="mse", sigma=None, observed_species=None, observation_fn=_a_only)


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


@pytest.mark.parametrize("variant", ["untransformed-known-noise", "log-estimated-noise"])
def test_buzzi_ferraris_matches_the_closed_form(tight, fits, variant):
    """T = d^T (2 Sigma + V1 + V2)^{-1} d with the closed-form d and Vi. In the
    second case model 2's fit ran in log k (its ``posterior_cov`` is in log k,
    the same variance as in k after the delta method) and estimated its noise,
    so the sensitivities must be taken in log k and Sigma is the mean of the
    two fits' noise variances."""
    f1, f2 = fits
    noise = SIGMA
    if variant == "log-estimated-noise":
        k2 = f2.params_named["A_to_B.k"]
        f2 = replace(
            f2,
            transforms=["positive_log"],
            posterior_cov=jnp.asarray([[COV[1] / k2**2]]),
            sigma_estimate=np.array([2 * SIGMA]),
        )
        noise = np.sqrt(0.5 * (SIGMA**2 + (2 * SIGMA) ** 2))
    value = _ranked_value(tuple(zip(tight, (f1, f2))), TIMES)
    assert value == pytest.approx(_closed_form(f1, f2, TIMES, noise), rel=1e-7)


def test_hunter_reiner_is_buzzi_ferraris_without_covariance(tight, fits, tight_models):
    """Hunter-Reiner is d^T (2 Sigma)^{-1} d: Buzzi-Ferraris with zero
    posterior covariances, and the closed form without V."""
    zero = tuple(replace(f, posterior_cov=jnp.zeros((1, 1))) for f in fits)
    hr = _ranked_value(tight_models, TIMES, criterion="hunter_reiner")
    bf0 = _ranked_value(tuple(zip(tight, zero)), TIMES)
    assert hr == pytest.approx(bf0, rel=1e-12)
    assert hr == pytest.approx(_closed_form(*fits, TIMES, SIGMA, "hunter_reiner"), rel=1e-7)
    # The covariances matter: with them the criterion is smaller.
    assert _ranked_value(tight_models, TIMES) < 0.99 * hr


def test_explicit_sigma_overrides_the_fits_noise(models):
    """sigma= replaces the fits' noise level: Hunter-Reiner scales as 1/sigma^2
    (the same solves, so exactly)."""
    base = _ranked_value(models, [4.0], criterion="hunter_reiner")
    value = _ranked_value(models, [4.0], criterion="hunter_reiner", sigma=5 * SIGMA)
    assert value == pytest.approx(base / 25.0, rel=1e-12)


def test_rank_orders_candidates_best_first(models, fits):
    """Late samples, where the models diverge, rank above early ones, and each
    score is its candidate's closed-form value (rel 1e-4: these solves run at
    rtol 1e-6)."""
    candidates = ([0.5, 1.0], [8.0, 10.0], [1.0, 5.0])
    kw = dict(criterion="hunter_reiner")
    ranked = aquakin.rank_discrimination(*models, [_exp(t) for t in candidates], **kw)
    assert [s.index for s in ranked] == [1, 2, 0]
    for s in ranked:
        expected = _closed_form(*fits, candidates[s.index], SIGMA, "hunter_reiner")
        assert s.value == pytest.approx(expected, rel=1e-4)
        assert s.improvement is None


# ---------- the optimized design ----------


def _best_single_sample(fits, criterion):
    return minimize_scalar(
        lambda t: -_closed_form(*fits, [t], SIGMA, criterion),
        bounds=(2.0, 30.0),
        method="bounded",
        options={"xatol": 1e-8},
    )


def _check_acceptance(tight_models, fits, criterion):
    """The issue's acceptance case. The fits agree over the fitted window
    (to 2) and diverge later; one sample started at the end of the window
    moves to the maximizer of the closed-form criterion (about t = 9)."""
    res = aquakin.optimize_discrimination(
        *tight_models,
        _exp([2.0]),
        criterion=criterion,
        design=DesignConfig(time_bounds=(0.1, 30.0)),
    )
    best = _best_single_sample(fits, criterion)
    t_opt = float(res.experiments[0].t_obs[0])
    assert t_opt == pytest.approx(best.x, rel=1e-4)
    assert t_opt > 5.0
    assert res.value == pytest.approx(-best.fun, rel=1e-7)
    assert res.initial_value == pytest.approx(
        _closed_form(*fits, [2.0], SIGMA, criterion), rel=1e-7
    )
    assert res.success


def test_hunter_reiner_sample_moves_to_where_the_models_diverge(tight_models, fits):
    _check_acceptance(tight_models, fits, "hunter_reiner")


def test_buzzi_ferraris_sample_moves_and_its_gradient_is_right(tight_models, fits):
    """The acceptance case for Buzzi-Ferraris, and the design gradient the
    optimizer uses, with respect to the scaled sampling time, against a central
    finite difference of the criterion. Over 10 s here: it compiles the
    Buzzi-Ferraris program (a Jacobian inside a design derivative) once and
    checks both against it, which is cheaper than two tests compiling it."""
    _check_acceptance(tight_models, fits, "buzzi_ferraris")
    pair = _resolve_pair(*tight_models, "buzzi_ferraris", None)
    exps = [_exp([2.0])]
    window = (0.1, 30.0)
    setups = _setups(pair, [_prepare(exps[0], 0, pair)], [30.0])
    x0, layout, conc_lo, conc_hi = _start_design(
        exps, {}, tight_models[0][0].model, 0, True, window, 0.0
    )
    # The program optimize_discrimination compiled, from the cache.
    program = _discrimination_program(setups, layout, "buzzi_ferraris")
    args = (_fitted(pair), setups[0].tobs, setups[0].C0, 0.1, 29.9, conc_lo, conc_hi)
    for x in (0.1, 0.2):  # t = 3.1 and 6.1, on the rising side of the optimum
        x = jnp.asarray([x])
        _, grad = program(x, *args)
        assert abs(float(grad[0])) > 1.0
        # Relative step 1e-4: truncation O(h^2) ~ 1e-8 against solve error
        # 1e-10 / h ~ 1e-6 relative.
        fd = central_fd(lambda x: program(x, *args)[0], x, 0)
        assert float(grad[0]) == pytest.approx(fd, rel=1e-5)


@pytest.mark.slow
def test_optimized_doses_and_spacing(tight_models, fits):
    """With the initial concentration free as well, the design keeps its
    samples min_spacing apart, inside the bounds, and improves the criterion;
    its value is the closed form at the returned design, with an interior
    optimum in the dose. Slow (a three-sample, dose-and-time program on the
    tight reactors); the dose layout is shared with optimize_experiment, whose
    dose tests run in the fast tier, and the discrimination program with the
    acceptance tests."""
    res = aquakin.optimize_discrimination(
        *tight_models,
        _exp([1.0, 2.0, 3.0]),
        design=DesignConfig(
            time_bounds=(0.5, 20.0), min_spacing=1.0, initial_bounds={"A": (0.5, 1.0)}
        ),
    )
    t = np.asarray(res.experiments[0].t_obs)
    assert np.all(np.diff(t) >= 1.0 - 1e-6) and t[0] >= 0.5 and t[-1] <= 20.0
    a0 = float(res.experiments[0].C0[0])
    assert res.value > 10.0 * res.initial_value
    assert res.value == pytest.approx(_closed_form(*fits, t, SIGMA, a0=a0), rel=1e-7)
    assert 0.55 < a0 < 0.95
    for other in (a0 - 0.02, a0 + 0.02):
        assert _closed_form(*fits, t, SIGMA, a0=other) < res.value


def test_repeat_optimize_compiles_nothing(models, xla_compiles):
    """A second call of the same structure reuses the compiled criterion and
    gradient, from the same or another starting design."""
    kw = dict(criterion="hunter_reiner", design=DesignConfig(time_bounds=(0.1, 30.0)))
    first = aquakin.optimize_discrimination(*models, _exp([2.0]), **kw)
    xla_compiles.clear()
    again = aquakin.optimize_discrimination(*models, _exp([2.0]), **kw)
    assert xla_compiles == []
    np.testing.assert_array_equal(again.experiments[0].t_obs, first.experiments[0].t_obs)
    moved = aquakin.optimize_discrimination(*models, _exp([3.0]), **kw)
    assert xla_compiles == []
    assert float(moved.experiments[0].t_obs[0]) == pytest.approx(
        float(first.experiments[0].t_obs[0]), rel=1e-4
    )


def test_repeat_rank_compiles_nothing(models, xla_compiles):
    """Ranking again with candidates of the same structure compiles nothing:
    they differ only in sampling times and share the latest one."""
    # Two samples ending at 10, the structure test_rank_orders_candidates_best_first uses.
    aquakin.rank_discrimination(*models, [_exp([1.0, 10.0])], criterion="hunter_reiner")
    xla_compiles.clear()
    candidates = [_exp([2.0, 10.0]), _exp([3.0, 4.0])]
    aquakin.rank_discrimination(*models, candidates, criterion="hunter_reiner")
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
    bare = replace(fits[1], _fit_record=None)
    with pytest.raises(
        ValueError, match="model 2: discrimination design needs the CalibrationResult"
    ):
        aquakin.rank_discrimination(models[0], (models[1][0], bare), [_exp([1.0])])


def test_reactor_must_be_on_the_fits_model(models, fits):
    with pytest.raises(ValueError, match="model 1: the reactor is on a different model"):
        aquakin.rank_discrimination((models[1][0], fits[0]), models[1], [_exp([1.0])])


def test_buzzi_ferraris_needs_a_laplace_posterior(reactors, raw_fits, models):
    """A fit without a Laplace posterior is refused by Buzzi-Ferraris; following
    the advice (Hunter-Reiner) gives the value Hunter-Reiner gives with the
    posterior, which it does not use."""
    pair = (reactors[1], raw_fits[1])
    with pytest.raises(
        ValueError, match="model 2: criterion='buzzi_ferraris' needs the fit's Laplace"
    ):
        aquakin.rank_discrimination(models[0], pair, [_exp([4.0])])
    hr = _ranked_value((models[0], pair), [4.0], criterion="hunter_reiner")
    assert hr == _ranked_value(models, [4.0], criterion="hunter_reiner")


def test_models_must_observe_the_same_quantities(reactors, models, fn_fit):
    with pytest.raises(
        ValueError,
        match=r"model 1 observes species \['A'\], model 2 observes observation_fn .*_a_only",
    ):
        aquakin.rank_discrimination(
            models[0], (reactors[1], fn_fit), [_exp([1.0])], criterion="hunter_reiner"
        )


def test_fit_without_a_noise_level_needs_sigma(reactors, fn_fit):
    """A fit with loss="mse" and no sigma has no noise level; passing sigma=
    as the message says gives a value: zero for a model against itself, which
    no experiment can tell apart, also when the experiment names the
    observation function."""
    pair = ((reactors[1], fn_fit), (reactors[1], fn_fit))
    with pytest.raises(
        ValueError, match=r"no noise level of one value per observed output.*pass sigma="
    ):
        aquakin.rank_discrimination(*pair, [_exp([4.0])], criterion="hunter_reiner")
    kw = dict(criterion="hunter_reiner", sigma=SIGMA)
    # Round-off only: the scaled gap is ~1e-15 where a real one is ~1 (the
    # criterion of the two rival models at t = 4 is about 50).
    assert _ranked_value(pair, [4.0], **kw) == pytest.approx(0.0, abs=1e-20)
    [named] = aquakin.rank_discrimination(*pair, [_exp([4.0], observation_fn=_a_only)], **kw)
    assert named.value == pytest.approx(0.0, abs=1e-20)


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
    kw = dict(criterion="hunter_reiner")
    with pytest.raises(
        ValueError, match="experiment 1: leave observed_species and observation_fn unset"
    ):
        aquakin.rank_discrimination(
            *models, [_exp([1.0]), _exp([1.0], observed_species=["B"])], **kw
        )
    [score] = aquakin.rank_discrimination(*models, [_exp([4.0], observed_species=("A",))], **kw)
    assert score.value == _ranked_value(models, [4.0], **kw)


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
