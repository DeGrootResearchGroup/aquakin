"""Saving a fit to a folder and reading it back (``CalibrationResult.save`` /
``aquakin.load_calibration``).

The fits are of the first-order decay ``A -> B`` (``simple_model``): a plain
fit, a two-dataset fit of the initial split (``FreeICConfig(fixed_total=True)``
with the joint Laplace posterior), a fit with ``sigma="estimate"``, one with a
prior, and a free-IC fit of the single-CSTR plant. A round trip must give back
every field of the result exactly, so the comparison is of ``repr``.
"""

import dataclasses
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import aquakin
from aquakin import CalibrationResult, FreeICConfig, LaplaceConfig, OptimizerConfig
from aquakin.integrate.calibration import saved
from aquakin.schema.loader import model_fingerprint
from aquakin.study import read_matrix, read_named_values
from tests.conftest import FIXTURES

K = "A_to_B.k"
SIGMA = 0.02
T_OBS = np.linspace(1.0, 30.0, 10)
# Fields a loaded result does not carry over from the fit: the live forward
# model, and (on a fit) the record a loaded result keeps of its model.
NOT_SAVED = {"_fit_record", "_model_record"}


def _decay(k, t, a0=1.0, b0=0.0):
    a = a0 * np.exp(-k * np.asarray(t))
    return np.stack([a, a0 + b0 - a], axis=1)


def _noisy(seed, **kw):
    rng = np.random.default_rng(seed)
    return jnp.asarray(_decay(0.1, T_OBS, **kw) + SIGMA * rng.standard_normal((T_OBS.size, 2)))


@pytest.fixture(scope="module")
def reactor(simple_model):
    return aquakin.BatchReactor(simple_model, simple_model.default_conditions())


def _fit(reactor, obs, C0=None, t_obs=None, **kw):
    kw.setdefault("sigma", jnp.asarray(SIGMA))
    return aquakin.calibrate(
        reactor,
        reactor.model.default_concentrations() if C0 is None else C0,
        obs,
        jnp.asarray(T_OBS) if t_obs is None else t_obs,
        [K],
        observed_species=["A", "B"],
        transforms={K: "positive_log"},
        loss="nll",
        use_priors=False,
        **kw,
    )


@pytest.fixture(scope="module")
def fits(reactor):
    """One fit of each kind the format must carry. Compiling the five fits
    takes about 25 s, paid once and shared by every test in the module: the
    round trip of each kind is what is under test, so none can be dropped."""
    split = [(0.7, 0.3), (0.5, 0.5)]
    return {
        "plain": _fit(reactor, _noisy(1), laplace=False),
        "free_ic": _fit(
            reactor,
            [_noisy(2, a0=a, b0=b) for a, b in split],
            C0=[jnp.asarray([0.6, 0.4]), jnp.asarray([0.6, 0.4])],
            t_obs=[jnp.asarray(T_OBS)] * 2,
            free_ic=FreeICConfig(["A", "B"], fixed_total=True),
            laplace=LaplaceConfig(include_ic=True),
        ),
        "estimate": _fit(reactor, _noisy(1), sigma="estimate", laplace=True),
        "prior": _fit(reactor, _noisy(1), priors={K: (0.12, 0.01)}, laplace=True),
        "bound": _bound_fit(reactor),
    }


def _bound_fit(reactor):
    """A fit held on a lower bound above the true rate: ``at_bound`` and a
    ``nan`` std."""
    with pytest.warns(RuntimeWarning, match="ended on an optimiser bound"):
        return _fit(
            reactor,
            _noisy(1),
            laplace=True,
            initial_params=jnp.asarray([0.3]),
            optimizer=OptimizerConfig(bounds={K: (0.15, 1.0)}),
        )


def _plain(value):
    """A field as plain Python values, so ``repr`` compares every bit."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    if hasattr(value, "shape"):  # a NumPy or JAX array
        return ("array", str(np.asarray(value).dtype), np.asarray(value).tolist())
    return value


def _fields(result) -> dict:
    return {
        f.name: _plain(getattr(result, f.name))
        for f in dataclasses.fields(CalibrationResult)
        if f.name not in NOT_SAVED
    }


def _round_trip(result, folder):
    result.save(folder)
    return aquakin.load_calibration(folder)


@pytest.mark.parametrize("kind", ["plain", "free_ic", "estimate", "prior", "bound"])
def test_every_field_survives_a_round_trip_exactly(fits, kind, tmp_path):
    """Each field of the loaded result has the repr of the fit's, arrays with
    their dtype and every float bit for bit."""
    loaded = _round_trip(fits[kind], tmp_path / kind)
    assert repr(_fields(loaded)) == repr(_fields(fits[kind]))
    assert loaded._fit_record is None


def test_the_fits_exercise_every_field(fits):
    """Every saved field is set (not ``None``, empty or ``False``) in at least
    one of the round-tripped fits, so none passes the round trip by being
    absent from all of them."""
    unset = (None, {}, [], False)
    empty = [
        name
        for name in _fields(fits["plain"])
        if all(_fields(fit)[name] in unset for fit in fits.values())
    ]
    assert empty == []


def test_the_files_hold_the_values_by_name(fits, reactor, tmp_path):
    """The folder is the study convention: values by name, matrices with named
    rows and columns, the model's fingerprint, the format and the version."""
    fit = fits["free_ic"]
    folder = fit.save(tmp_path / "fit")
    summary = yaml.safe_load((folder / saved.FIT_FILE).read_text())
    assert summary["format"] == 1 and summary["aquakin"] == aquakin.__version__
    text = (FIXTURES / "simple_model.yaml").read_text()
    assert summary["models"] == {"simple_decay": model_fingerprint(text, "simple_model")}
    assert summary["fitted"] == {K: fit.params_named[K]}
    assert summary["initial_pools"]["fitted"] == fit.ic_named
    params = read_named_values(folder / saved.PARAMETERS_FILE)
    assert params == {K: float(fit.params[reactor.model.param_index[K]])}
    c0, rows, columns = read_matrix(folder / saved.INITIAL_STATES_FILE)
    assert (rows, columns) == (["dataset 0", "dataset 1"], ["A", "B"])
    np.testing.assert_array_equal(c0, np.stack(fit.C0_fitted))
    cov, rows, columns = read_matrix(folder / saved.JOINT_COVARIANCE_FILE)
    assert rows == columns == fit.joint_names
    np.testing.assert_array_equal(cov, np.asarray(fit.joint_cov))
    # A fit without a posterior writes no matrix.
    plain = fits["plain"].save(tmp_path / "plain")
    assert sorted(p.name for p in plain.iterdir()) == ["fit.yaml", "parameters.yaml"]
    # The noise estimate is keyed by the observed species.
    estimate = fits["estimate"].save(tmp_path / "estimate")
    sigma = yaml.safe_load((estimate / saved.FIT_FILE).read_text())["sigma_estimate"]
    assert list(sigma) == ["A", "B"]


def test_saving_a_loaded_fit_writes_the_same_files(fits, tmp_path):
    """A loaded result keeps the names and fingerprints it was saved with, so
    saving it again needs no model and writes the same bytes."""
    first = fits["free_ic"].save(tmp_path / "first")
    second = aquakin.load_calibration(first).save(tmp_path / "second")
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    for name in names:
        assert (second / name).read_bytes() == (first / name).read_bytes()


def test_information_criteria_survive_the_round_trip(fits, reactor, tmp_path):
    """AIC and BIC of a loaded fit equal the fit's, and the loaded fit still
    carries its observations' fingerprint: compare_models accepts it beside a
    fit of the same data and refuses one of other data."""
    fit = fits["estimate"]
    loaded = _round_trip(fit, tmp_path / "fit")
    assert aquakin.aic(loaded) == aquakin.aic(fit)
    assert aquakin.bic(loaded) == aquakin.bic(fit)
    same_data = fits["plain"]  # the same observations, a known sigma
    table = aquakin.compare_models({"loaded": loaded, "known": same_data})
    expected = aquakin.compare_models({"loaded": fit, "known": same_data})
    assert table.values == expected.values
    other = _round_trip(_fit(reactor, _noisy(3), laplace=False), tmp_path / "other")
    with pytest.raises(ValueError, match=r"fits \['other'\] are not of the same observations"):
        aquakin.compare_models({"loaded": loaded, "other": other})


def test_methods_that_take_the_model_work_on_a_loaded_fit(fits, reactor, tmp_path):
    """posterior_draws and predictive_band take the model or reactor, so a
    loaded fit gives exactly the fit's draws and band."""
    fit = fits["prior"]
    loaded = _round_trip(fit, tmp_path / "fit")
    model = reactor.model
    np.testing.assert_array_equal(
        loaded.posterior_draws(model, 20, seed=3), fit.posterior_draws(model, 20, seed=3)
    )
    C0 = model.default_concentrations()
    t = jnp.asarray([1.0, 5.0])
    band = loaded.predictive_band(reactor, C0, t, n_draw=8)
    np.testing.assert_array_equal(band.median, fit.predictive_band(reactor, C0, t, n_draw=8).median)


def test_sampling_a_loaded_fit_asks_for_the_fit(fits, reactor, tmp_path):
    """The forward model is not saved, so the sampler refuses a loaded fit and
    says it needs the result a fit returns."""
    loaded = _round_trip(fits["prior"], tmp_path / "fit")
    with pytest.raises(ValueError, match="needs the CalibrationResult a fit returned"):
        aquakin.sample_posterior(loaded, reactor)


def test_a_plant_fit_saves_its_state_and_models_by_name(cstr_plant, tmp_path):
    """A plant fit names its parameters ``<model>.<param>``, its fitted
    initial state ``<unit>.<state>``, its models' fingerprints, and its noise
    outputs by position (the plant fit has no channel names to give); the
    round trip is exact."""
    param = "simple_decay.A_to_B.k"
    t = jnp.linspace(1.0, 20.0, 6)
    sol = cstr_plant.solve(t_span=(0.0, 20.0), t_eval=t)
    obs = cstr_plant.stream(sol, "tank", None).C_named("B")
    fit = cstr_plant.calibrate(
        obs,
        t,
        [param],
        target="tank",
        observed_channels=["B"],
        t_span=(0.0, 20.0),
        y0=cstr_plant.initial_state(),
        free_ic=FreeICConfig(["tank.A"]),
        transforms={param: "positive_log"},
        loss="nll",
        sigma="estimate",
        laplace=False,
        optimizer=OptimizerConfig(max_iter=3),
    )
    folder = fit.save(tmp_path / "plant")
    summary = yaml.safe_load((folder / saved.FIT_FILE).read_text())
    text = (FIXTURES / "simple_model.yaml").read_text()
    assert summary["models"] == {"simple_decay": model_fingerprint(text, "simple_model")}
    assert list(summary["sigma_estimate"]) == ["output[0]"]
    assert list(read_named_values(folder / saved.PARAMETERS_FILE)) == cstr_plant.parameter_names()
    c0, _, columns = read_matrix(folder / saved.INITIAL_STATES_FILE)
    named = cstr_plant.named_state(fit.C0_fitted[0])
    assert dict(zip(columns, c0[0])) == {
        f"{u}.{n}": v for u, values in named.items() for n, v in values.items()
    }
    loaded = aquakin.load_calibration(folder)
    assert repr(_fields(loaded)) == repr(_fields(fit))


def test_a_result_built_by_hand_needs_its_model(simple_model, tmp_path):
    """A result that came from no fit cannot name its parameter vector; given
    the model it saves, with a fingerprint per model (none for a model not read
    from a file) and its noise outputs named by position."""
    result = CalibrationResult(
        params=jnp.asarray([0.2]),
        params_named={K: 0.2},
        loss=1.5,
        converged=True,
        message="by hand",
        n_iter=0,
        parameter_names=[K],
        transforms=["positive_log"],
        sigma_estimate=np.asarray([0.1, 0.3]),
    )
    with pytest.raises(ValueError, match=r"pass the calibrated model: result.save\(folder"):
        result.save(tmp_path / "a")
    result.save(tmp_path / "b", model=simple_model)
    loaded = aquakin.load_calibration(tmp_path / "b")
    assert repr(_fields(loaded)) == repr(_fields(result))
    assert loaded.sigma_estimate.tolist() == [0.1, 0.3]
    summary = yaml.safe_load((tmp_path / "b" / saved.FIT_FILE).read_text())
    assert summary["sigma_estimate"] == {"output[0]": 0.1, "output[1]": 0.3}

    bare = SimpleNamespace(param_index={K: 0}, species_index={}, models=[SimpleNamespace(name="m")])
    result.save(tmp_path / "c", model=bare)
    assert yaml.safe_load((tmp_path / "c" / saved.FIT_FILE).read_text())["models"] == {"m": None}
    wider = SimpleNamespace(param_index={K: 0, "x": 1}, species_index={}, models=[])
    with pytest.raises(ValueError, match="the result has 1 parameters; the model names 2"):
        result.save(tmp_path / "d", model=wider)
    gap = SimpleNamespace(param_index={K: 0, "x": 2}, species_index={}, models=[])
    with pytest.raises(ValueError, match="parameter names do not cover every entry"):
        result.save(tmp_path / "e", model=gap)
    with pytest.raises(ValueError, match="sigma_estimate has 2 entries for 1 output names"):
        saved.fit_summary(result, ["only"])


def test_save_never_overwrites_a_saved_fit(fits, tmp_path):
    folder = fits["plain"].save(tmp_path / "fit")
    before = (folder / saved.FIT_FILE).read_bytes()
    with pytest.raises(FileExistsError, match=r"already holds fit.yaml, parameters.yaml"):
        fits["prior"].save(folder)
    assert (folder / saved.FIT_FILE).read_bytes() == before


@pytest.fixture
def saved_fit(fits, tmp_path):
    return fits["free_ic"].save(tmp_path / "fit")


def _edit_summary(folder, change):
    path = folder / saved.FIT_FILE
    summary = yaml.safe_load(path.read_text())
    change(summary)
    path.write_text(yaml.safe_dump(summary, sort_keys=False))


def test_a_missing_folder_or_fit_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="the folder does not exist"):
        aquakin.load_calibration(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match=r"holds no fit\.yaml"):
        aquakin.load_calibration(tmp_path)


@pytest.mark.parametrize(
    ("version", "message"),
    [(99, "in saved-fit format 99; this aquakin"), ("one", "format 'one'"), (None, "format None")],
)
def test_an_unknown_or_newer_format_is_refused(saved_fit, version, message):
    _edit_summary(saved_fit, lambda s: s.update(format=version))
    with pytest.raises(ValueError, match=message):
        aquakin.load_calibration(saved_fit)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("format: [1\n", r"fit.yaml is damaged: while parsing"),
        ("- 1\n", r"fit.yaml is damaged: it holds no summary of a fit"),
    ],
)
def test_a_damaged_fit_file(saved_fit, damage, message):
    (saved_fit / saved.FIT_FILE).write_text(damage)
    with pytest.raises(ValueError, match=message):
        aquakin.load_calibration(saved_fit)


def test_a_summary_missing_a_value(saved_fit):
    _edit_summary(saved_fit, lambda s: s.pop("loss"))
    with pytest.raises(ValueError, match=r"fit.yaml is damaged: it has no 'loss'"):
        aquakin.load_calibration(saved_fit)


def test_a_summary_with_a_wrong_value(saved_fit):
    _edit_summary(saved_fit, lambda s: s.update(loss="low"))
    with pytest.raises(ValueError, match=r"is damaged: could not convert string to float"):
        aquakin.load_calibration(saved_fit)


@pytest.mark.parametrize(
    "file", [saved.PARAMETERS_FILE, saved.JOINT_HESSIAN_FILE, saved.INITIAL_STATES_FILE]
)
def test_a_missing_file(saved_fit, file):
    (saved_fit / file).unlink()
    with pytest.raises(FileNotFoundError, match=f"{file} is missing: it is part of the fit"):
        aquakin.load_calibration(saved_fit)


def test_a_damaged_matrix(saved_fit):
    path = saved_fit / saved.COVARIANCE_FILE
    good = path.read_text()
    path.write_text(good.replace(K, "renamed"))
    with pytest.raises(ValueError, match=r"covariance\.csv is damaged: its rows or columns"):
        aquakin.load_calibration(saved_fit)
    path.write_text(good.replace(",", ",x", 2))
    with pytest.raises(ValueError, match="is damaged: could not convert string to float: 'x"):
        aquakin.load_calibration(saved_fit)


def test_the_saved_file_names_are_the_stored_ones():
    """Saved fits keep reading: the constants spell the names on disk."""
    assert (
        saved.FIT_FILE,
        saved.PARAMETERS_FILE,
        saved.COVARIANCE_FILE,
        saved.HESSIAN_FILE,
        saved.JOINT_COVARIANCE_FILE,
        saved.JOINT_HESSIAN_FILE,
        saved.INITIAL_STATES_FILE,
        saved.FORMAT,
    ) == (
        "fit.yaml",
        "parameters.yaml",
        "covariance.csv",
        "hessian.csv",
        "joint_covariance.csv",
        "joint_hessian.csv",
        "initial_states.csv",
        1,
    )
