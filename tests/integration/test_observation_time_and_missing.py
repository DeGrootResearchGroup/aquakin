"""Tests for a time-aware ``observation_fn(t, C, params)`` and for missing
(``NaN``) observations, in ``calibrate`` and ``fisher_information``.

Each new formulation is checked against the per-entry-sigma formulation it
replaces, which gives the same optimum and Laplace covariance (the nll differs
only by a constant).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import Experiment, LaplaceConfig, OptimizerConfig
from aquakin.integrate.calibrate import observation_size, observation_takes_time, observe
from aquakin.integrate.experiment_design import _structure_key
from aquakin.integrate.fisher import _FisherSetup

K, A0, SIGMA, DRIFT = 0.25, 1.5, 0.01, 0.05
T_OBS = jnp.linspace(0.0, 12.0, 25)
C0 = jnp.asarray([A0, 0.0])
FREE = ["A_to_B.k"]
LOG = {"A_to_B.k": "positive_log"}


@pytest.fixture
def reactor(simple_model):
    return aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))


@pytest.fixture
def truth(reactor, simple_model):
    p = simple_model.default_parameters().at[0].set(K)
    C = reactor.solve(C0, params=p, t_span=(0.0, 12.0), t_eval=T_OBS).C
    rng = np.random.default_rng(3)
    return C + SIGMA * jnp.asarray(rng.standard_normal(C.shape))


def _fit(reactor, observations, **kw):
    return aquakin.calibrate(
        reactor,
        C0,
        observations,
        T_OBS,
        FREE,
        transforms=LOG,
        loss="nll",
        laplace=LaplaceConfig(method="gauss_newton"),
        optimizer=OptimizerConfig(method="gauss_newton"),
        **kw,
    )


def _same_fit(a, b):
    assert a.params_named["A_to_B.k"] == pytest.approx(b.params_named["A_to_B.k"], rel=1e-7)
    assert a.params_named_std["A_to_B.k"] == pytest.approx(b.params_named_std["A_to_B.k"], rel=1e-6)


def test_signature_detection():
    """Three required positional arguments: (t, C, params); callable with two:
    (C, params), as before time-aware functions existed."""

    def f(C, p=None):
        return C[0]

    assert not observation_takes_time(lambda C, p: C[0])
    assert not observation_takes_time(f)
    assert not observation_takes_time(lambda C, p, scale=2.0: scale * C[0])
    assert observation_takes_time(lambda t, C, p: C[0])
    assert observation_takes_time(lambda t, C, p, scale=2.0: scale * C[0])
    assert observation_takes_time(lambda t, C, p, s=1.0: s * C[0])
    with pytest.raises(TypeError, match=r"\(t, C, params\)"):
        observation_takes_time(lambda C: C[0])
    with pytest.raises(TypeError, match=r"\(t, C, params\)"):
        observation_takes_time(lambda a, b, c, d: a)
    with pytest.raises(TypeError, match="must be a callable"):
        observation_takes_time(3.0)

    class _Opaque:
        """A callable whose signature cannot be inspected."""

        def __call__(self, C, p):
            return C[0]

        @property
        def __signature__(self):
            raise ValueError("no signature")

    assert not observation_takes_time(_Opaque())
    assert not observation_takes_time(lambda *args: args[0][0])  # *args: the (C, params) form


def test_time_aware_observe_passes_the_sample_time():
    """observe / observation_size hand a time-aware function each sample time
    (no solve: a stand-in trajectory)."""
    ys = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    fn = lambda t, C, p: C[1] * (1.0 + t)
    y = observe(ys, None, jnp.arange(2), fn, t=jnp.asarray([0.0, 2.0]))
    np.testing.assert_allclose(np.asarray(y), [[2.0], [12.0]])
    assert observation_size(lambda t, C, p: jnp.stack([C[0], t]), ys[0], jnp.ones(1)) == 2


def test_fisher_setup_mask_gives_unmeasured_entries_infinite_sigma(reactor):
    """Experiment(mask=...) sets an infinite sigma (zero information) on the
    unmeasured entries and validates the mask, all before any solve."""
    mask = np.ones((T_OBS.shape[0], 2), dtype=bool)
    mask[::2, 1] = False
    setup = _FisherSetup(reactor, [Experiment(C0, T_OBS, sigma=SIGMA, mask=mask)], FREE)
    sig = np.asarray(setup.sigma[0])
    assert np.all(np.isinf(sig[~mask]))
    np.testing.assert_array_equal(sig[mask], SIGMA)
    with pytest.raises(ValueError, match="does not broadcast"):
        _FisherSetup(reactor, [Experiment(C0, T_OBS, sigma=SIGMA, mask=np.ones(3, bool))], FREE)
    with pytest.raises(ValueError, match="no observation"):
        _FisherSetup(reactor, [Experiment(C0, T_OBS, sigma=SIGMA, mask=False)], FREE)


class _StubReactor:
    """Stands in for a reactor in a :class:`_FisherSetup` (no solve): every
    sampled state is ``[k * t, 0]`` with ``k`` the first parameter."""

    def solve(self, C0, params, t_span, t_eval):
        A = params[0] * jnp.asarray(t_eval)
        return type("Sol", (), {"C": jnp.stack([A, jnp.zeros_like(A)], axis=1)})()


def test_fisher_time_aware_fn_sees_t_obs_exactly(reactor):
    """With a unit conversion whose round trip is inexact (x * 60 / 60 != x),
    a time-aware observation_fn still receives the entries of t_obs exactly:
    the indicator t == t_obs[i] is 1 at every sample, so every Jacobian row is
    the same nonzero value."""
    t_user = jnp.asarray([0.33585575305464355, 7.294965609839984])
    assert np.any(np.asarray(t_user) * 60.0 / 60.0 != np.asarray(t_user))
    hit = lambda t, C, p: C[0] / t * jnp.any(t == t_user)
    ex = Experiment(C0, t_user, sigma=SIGMA, observation_fn=hit, time_unit="min")
    setup = _FisherSetup(
        reactor, [ex], FREE, params=reactor.model.default_parameters().at[0].set(K)
    )
    np.testing.assert_array_equal(setup.tobs[0], t_user)
    setup.reactors = [_StubReactor()]
    J = np.asarray(setup.jacobian(setup.p0, setup.tobs, setup.C0))
    assert np.all(J != 0.0) and np.all(J[0] == J)


def test_fisher_mask_drops_a_non_finite_prediction(reactor):
    """A prediction that is -inf (log 0) at a masked entry, with a NaN tangent,
    adds an exact zero row to the Jacobian; the other rows are unchanged."""
    t = jnp.asarray([0.0, 1.0, 2.0])
    mask = np.array([False, True, True])
    ex = Experiment(
        C0, t, sigma=SIGMA, observation_fn=lambda C, p: jnp.log(C[0]), mask=mask[:, None]
    )
    setup = _FisherSetup(
        reactor, [ex], FREE, params=reactor.model.default_parameters().at[0].set(K)
    )
    setup.reactors = [_StubReactor()]
    J = np.asarray(setup.jacobian(setup.p0, setup.tobs, setup.C0))
    assert np.all(np.isfinite(J))
    assert J[0, 0] == 0.0 and J[1, 0] != 0.0 and J[1, 0] == J[2, 0]


def test_time_aware_observe_needs_sample_times():
    ys = jnp.ones((3, 2))
    with pytest.raises(ValueError, match="needs the sample times"):
        observe(ys, None, jnp.arange(2), lambda t, C, p: C[0])


@pytest.mark.slow  # full fits / Fisher solves; cheap fast-tier coverage in tests/unit
def test_time_dependent_observation_matches_per_entry_sigma(reactor, truth):
    """A drifting sensor y = B (1 + a t) through a time-aware observation_fn
    equals observing B on the de-drifted data y / (1 + a t) with the per-entry
    noise sigma / (1 + a t)."""
    gain = 1.0 + DRIFT * np.asarray(T_OBS)
    y = truth[:, 1] * jnp.asarray(gain)
    timed = _fit(reactor, y, observation_fn=lambda t, C, p: C[1] * (1.0 + DRIFT * t), sigma=SIGMA)
    ref = _fit(
        reactor,
        y / jnp.asarray(gain),
        observed_species=["B"],
        sigma=jnp.asarray(SIGMA / gain)[:, None],
    )
    _same_fit(timed, ref)


@pytest.mark.slow  # full fits / Fisher solves; cheap fast-tier coverage in tests/unit
def test_time_is_in_the_unit_of_t_obs(reactor, truth):
    """With t_obs in minutes on a seconds model, the time-aware function sees
    minutes: a drift of DRIFT per minute fitted on minute times equals a drift
    of DRIFT / 60 per second fitted on second times."""
    gain = 1.0 + DRIFT * np.asarray(T_OBS)  # T_OBS read as minutes
    y = truth[:, 1] * jnp.asarray(gain)
    per_min = _fit(
        reactor,
        y,
        observation_fn=lambda t, C, p: C[1] * (1.0 + DRIFT * t),
        sigma=SIGMA,
        time_unit="min",
    )
    per_s = aquakin.calibrate(
        reactor,
        C0,
        y,
        T_OBS * 60.0,
        FREE,
        transforms=LOG,
        loss="nll",
        sigma=SIGMA,
        observation_fn=lambda t, C, p: C[1] * (1.0 + DRIFT / 60.0 * t),
        laplace=LaplaceConfig(method="gauss_newton"),
        optimizer=OptimizerConfig(method="gauss_newton"),
    )
    _same_fit(per_min, per_s)


@pytest.mark.slow  # full fits / Fisher solves; cheap fast-tier coverage in tests/unit
def test_nan_masked_observations_match_per_entry_sigma(reactor, truth):
    """An output measured only at t = 0 (an initial total A + B), the other
    entries NaN, gives the same fit as the huge-sigma trick it replaces, and
    the loss differs only by the constant sum(log sigma) of the dummy entries."""
    total = truth[:, 0] + truth[:, 1]
    obs = jnp.stack([truth[:, 1], jnp.where(T_OBS == 0.0, total, jnp.nan)], axis=1)
    fn = lambda C, p: jnp.stack([C[1], C[0] + C[1]])
    masked = _fit(reactor, obs, observation_fn=fn, sigma=SIGMA)

    big = 1e8
    sig = np.full(obs.shape, SIGMA)
    sig[1:, 1] = big
    trick = _fit(
        reactor,
        jnp.nan_to_num(obs, nan=0.0),
        observation_fn=fn,
        sigma=jnp.asarray(sig),
    )
    _same_fit(masked, trick)
    n_dummy = obs.shape[0] - 1
    assert trick.loss - masked.loss == pytest.approx(n_dummy * np.log(big), rel=1e-9)


@pytest.mark.slow  # full fits / Fisher solves; cheap fast-tier coverage in tests/unit
def test_nan_masked_mse_averages_over_observed_entries(reactor, truth):
    """For mse a missing entry drops out of the mean; with all of one output's
    entries missing the fit equals observing the other output alone."""
    obs = jnp.stack([truth[:, 0], jnp.full(T_OBS.shape, jnp.nan)], axis=1)
    masked = aquakin.calibrate(reactor, C0, obs, T_OBS, FREE, loss="mse", laplace=False)
    only_a = aquakin.calibrate(
        reactor, C0, truth[:, 0], T_OBS, FREE, observed_species=["A"], loss="mse", laplace=False
    )
    assert masked.params_named["A_to_B.k"] == pytest.approx(
        only_a.params_named["A_to_B.k"], rel=1e-7
    )
    assert masked.loss == pytest.approx(only_a.loss, rel=1e-7)


def test_missing_observation_validation(reactor, truth):
    with pytest.raises(ValueError, match="inf"):
        aquakin.calibrate(reactor, C0, truth.at[3, 0].set(jnp.inf), T_OBS, FREE, laplace=False)
    with pytest.raises(ValueError, match="every observation is missing"):
        aquakin.calibrate(reactor, C0, jnp.full(truth.shape, jnp.nan), T_OBS, FREE, laplace=False)


@pytest.mark.slow  # full fits / Fisher solves; cheap fast-tier coverage in tests/unit
def test_fisher_mask_matches_per_entry_sigma(reactor, simple_model):
    """Experiment(mask=...) equals the same design with huge sigma on the
    unmeasured entries, and the time-aware observation_fn equals the
    de-drifted per-entry-sigma design."""
    p = simple_model.default_parameters().at[0].set(K)
    mask = np.ones((T_OBS.shape[0], 2), dtype=bool)
    mask[::2, 1] = False
    fim = aquakin.fisher_information(
        reactor, Experiment(C0, T_OBS, sigma=SIGMA, mask=mask), FREE, params=p, transforms=LOG
    )
    sig = np.where(mask, SIGMA, 1e12)
    ref = aquakin.fisher_information(
        reactor, Experiment(C0, T_OBS, sigma=sig), FREE, params=p, transforms=LOG
    )
    np.testing.assert_allclose(fim.matrix, ref.matrix, rtol=1e-12)
    full = aquakin.fisher_information(
        reactor, Experiment(C0, T_OBS, sigma=SIGMA), FREE, params=p, transforms=LOG
    )
    assert fim.matrix[0, 0] < full.matrix[0, 0]

    gain = 1.0 + DRIFT * np.asarray(T_OBS)
    timed = aquakin.fisher_information(
        reactor,
        Experiment(C0, T_OBS, sigma=SIGMA, observation_fn=lambda t, C, q: C[1] * (1 + DRIFT * t)),
        FREE,
        params=p,
        transforms=LOG,
    )
    dedrift = aquakin.fisher_information(
        reactor,
        Experiment(C0, T_OBS, sigma=(SIGMA / gain)[:, None], observed_species=["B"]),
        FREE,
        params=p,
        transforms=LOG,
    )
    np.testing.assert_allclose(timed.matrix, dedrift.matrix, rtol=1e-10)

    with pytest.raises(ValueError, match="does not broadcast"):
        aquakin.fisher_information(
            reactor, Experiment(C0, T_OBS, sigma=SIGMA, mask=np.ones(3, bool)), FREE, params=p
        )
    with pytest.raises(ValueError, match="no observation"):
        aquakin.fisher_information(
            reactor, Experiment(C0, T_OBS, sigma=SIGMA, mask=False), FREE, params=p
        )


def test_design_structure_key_distinguishes_masks():
    """Experiment design groups candidates by structure and shares one compiled
    evaluation per group; the mask is part of that structure."""
    a = Experiment(C0, T_OBS, sigma=SIGMA, mask=np.array([True, False]))
    b = Experiment(C0, T_OBS, sigma=SIGMA, mask=np.array([True, True]))
    c = Experiment(C0, T_OBS, sigma=SIGMA)
    assert len({_structure_key(a), _structure_key(b), _structure_key(c)}) == 3
