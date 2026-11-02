"""Plant Fisher information and profile likelihood.

A single CSTR fed a constant influent, on a three-reaction model:
``A -> B`` (``k1``), ``B ->`` (``k2``) and ``D -> E`` (``k3``). The effluent
``A`` has a closed form, ``A(t) = A* + (A0 - A*) exp(-(D + k1) t)`` with
``A* = D A_in / (D + k1)`` and dilution ``D = Q / V``, so the Fisher
information it carries about ``k1`` can be checked exactly. ``k3`` moves only
``D`` and ``E``, which are never observed: the data cannot see it.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import LaplaceConfig, OptimizerConfig
from aquakin.plant import (
    CSTRUnit,
    Plant,
    PlantObservable,
    SensorSet,
    plant_fisher_information,
    plant_profile_likelihood,
    plant_rank_sensor_sets,
)
from aquakin.plant.influent import InfluentSeries

_MODEL = """\
model:
  name: chain
  version: "1.0"
  description: "A -> B -> (sink), and an unobserved D -> E."
  references: []
species:
  - {name: A, units: mol/L, default_concentration: 1.0}
  - {name: B, units: mol/L, default_concentration: 0.0}
  - {name: D, units: mol/L, default_concentration: 1.0}
  - {name: E, units: mol/L, default_concentration: 0.0}
conditions:
  - {name: T, default: 293.15}
reactions:
  - name: r1
    rate: "k1 * [A]"
    parameters:
      k1: {value: 0.2, units: "1/d", bounds: [1.0e-3, 10.0], transform: positive_log}
    stoichiometry: {A: -1, B: +1}
  - name: r2
    rate: "k2 * [B]"
    parameters:
      k2: {value: 0.3, units: "1/d", bounds: [1.0e-3, 10.0], transform: positive_log}
    stoichiometry: {B: -1}
  - name: r3
    rate: "k3 * [D]"
    parameters:
      k3: {value: 0.5, units: "1/d", bounds: [1.0e-3, 10.0], transform: positive_log}
    stoichiometry: {D: -1, E: +1}
"""

K1, K2, K3 = "chain.r1.k1", "chain.r2.k2", "chain.r3.k3"
Q, V = 10.0, 100.0
DIL = Q / V
T_OBS = jnp.linspace(0.5, 20.0, 12)
SIGMA = 0.01


def _plant(net, a_in=1.0):
    plant = Plant("chain_cstr")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=net,
            volume=V,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    feed = jnp.asarray([a_in, 0.0, 1.0, 0.0])
    plant.add_influent(
        "feed",
        InfluentSeries(
            t=jnp.asarray([0.0, 100.0]), Q=jnp.asarray([Q, Q]), C=jnp.stack([feed, feed]), model=net
        ),
        to="tank.inlet",
    )
    return plant


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    path = tmp_path_factory.mktemp("chain") / "chain.yaml"
    path.write_text(_MODEL)
    return aquakin.load_model_from_file(path)


@pytest.fixture(scope="module")
def decay_net():
    return aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")


@pytest.fixture(scope="module")
def plant(net):
    return _plant(net)


def _effluent_A(k1, t, a0=1.0, a_in=1.0):
    """Closed-form effluent A of the CSTR (it starts at ``a0``)."""
    a_star = DIL * a_in / (DIL + k1)
    return a_star + (a0 - a_star) * jnp.exp(-(DIL + k1) * t)


def _observations(plant, params, observables, t_obs=T_OBS):
    sol = plant.solve(t_span=(0.0, float(t_obs[-1])), t_eval=t_obs, params=params)
    cols = [plant.stream(sol, o.stream, params).C_named(c) for o in observables for c in o.channels]
    return jnp.stack(cols, axis=1)


# --- Fisher information ------------------------------------------------------


@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_matches_closed_form(plant, mode):
    """F for k1 from effluent A equals sum (dA/dk1)^2 / sigma^2, with dA/dk1
    from the closed form -- in log k1 and, with no transform, in k1 itself."""
    k1 = 0.2
    t = np.asarray(T_OBS)
    dA = np.asarray(jax.vmap(jax.grad(_effluent_A), in_axes=(None, 0))(k1, T_OBS))
    expected = np.sum(dA**2) / SIGMA**2
    kw = dict(sigma=SIGMA, t_span=(0.0, float(t[-1])), mode=mode)
    obs = [("tank", ["A"])]
    phys = plant_fisher_information(plant, obs, T_OBS, [K1], transforms={K1: "none"}, **kw)
    logk = plant_fisher_information(plant, obs, T_OBS, [K1], **kw)
    np.testing.assert_allclose(phys.matrix, [[expected]], rtol=1e-4)
    np.testing.assert_allclose(logk.matrix, [[expected * k1**2]], rtol=1e-4)
    assert logk.transforms == ["positive_log"]
    assert logk.values == pytest.approx({K1: k1})
    assert logk.std()[K1] == pytest.approx(expected**-0.5, rel=1e-4)


def test_forward_matches_jacobian_built_from_solve_sensitivity(plant):
    """The forward mode is the hand-built J = S[obs rows] * theta / sigma (the
    tank's outlet is its state), with a per-channel sigma and a mask."""
    params = plant.default_parameters()
    sigma = jnp.asarray([0.01, 0.02])
    mask = np.ones((T_OBS.shape[0], 2), dtype=bool)
    mask[::3, 1] = False
    fim = plant_fisher_information(
        plant,
        [PlantObservable("tank", ("A", "B"))],
        T_OBS,
        [K1, K2, K3],
        sigma=sigma,
        mask=mask,
        t_span=(0.0, 20.0),
    )
    _, _, S = plant.solve_sensitivity(params, [K1, K2, K3], t_span=(0.0, 20.0), t_eval=T_OBS)
    theta = np.asarray([params[plant.parameter_index(n)] for n in (K1, K2, K3)])
    J = np.asarray(S)[:, :2, :] * theta / np.asarray(sigma)[None, :, None]
    J = np.where(mask[..., None], J, 0.0).reshape(-1, 3)
    np.testing.assert_allclose(fim.jacobian, J, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fim.matrix, J.T @ J, rtol=1e-10)
    # k3 moves only the unobserved D and E: no information about it.
    assert np.all(fim.matrix[2] == 0.0)


def test_forward_and_reverse_agree(plant):
    """The variational solve and the stable discrete adjoint give the same
    information for a multi-channel, multi-parameter design."""
    kw = dict(sigma=[0.01, 0.02], t_span=(0.0, 20.0))
    obs = [("tank", ["A", "B"])]
    fwd = plant_fisher_information(plant, obs, T_OBS, [K1, K2], mode="forward", **kw)
    rev = plant_fisher_information(plant, obs, T_OBS, [K1, K2], mode="reverse", **kw)
    np.testing.assert_allclose(fwd.matrix, rev.matrix, rtol=1e-3)
    assert abs(fwd.matrix[0, 1]) > 1e-2 * np.sqrt(fwd.matrix[0, 0] * fwd.matrix[1, 1])


def test_equals_gauss_newton_laplace_hessian(plant):
    """At a Plant.calibrate point (loss='nll', Gauss-Newton Laplace, no priors)
    the reverse-mode information is the Laplace Hessian: the two differentiate
    the same solve in the same coordinates."""
    obs = [PlantObservable("tank", ("A", "B"))]
    truth = plant.default_parameters()
    data = _observations(plant, truth, obs)
    sigma = jnp.asarray([0.01, 0.02])
    start = truth.at[plant.parameter_index(K1)].set(0.25)
    result = plant.calibrate(
        data,
        T_OBS,
        [K1, K2],
        observables=obs,
        t_span=(0.0, 20.0),
        params=start,
        loss="nll",
        sigma=sigma,
        use_priors=False,
        optimizer=OptimizerConfig(max_iter=3),
        laplace=LaplaceConfig(method="gauss_newton"),
    )
    fim = plant_fisher_information(
        plant,
        obs,
        T_OBS,
        [K1, K2],
        sigma=sigma,
        t_span=(0.0, 20.0),
        params=result.params,
        mode="reverse",
    )
    np.testing.assert_allclose(fim.matrix, np.asarray(result.hessian_unconstrained), rtol=1e-8)


def test_runs_add_and_relative_sigma(net, plant):
    """Two runs (one with its own plant, at another influent) sum their
    information; a relative sigma is sigma * |y| at the assumed parameters."""
    other = _plant(net, a_in=2.0)
    obs = [("tank", ["A"])]
    t2 = jnp.linspace(1.0, 10.0, 5)
    kw = dict(free_params=[K1], sigma=SIGMA)
    one = plant_fisher_information(plant, obs, T_OBS, t_span=(0.0, 20.0), **kw)
    two = plant_fisher_information(other, obs, t2, t_span=(0.0, 10.0), **kw)
    both = plant_fisher_information(
        [plant, other], obs, [T_OBS, t2], t_span=[(0.0, 20.0), (0.0, 10.0)], **kw
    )
    np.testing.assert_allclose(both.matrix, one.matrix + two.matrix, rtol=1e-10)
    solo = plant_fisher_information([other], obs, t2, t_span=(0.0, 10.0), **kw)
    np.testing.assert_allclose(solo.matrix, two.matrix, rtol=1e-12)

    rel = plant_fisher_information(
        other, obs, t2, [K1], sigma=0.05, relative_sigma=True, t_span=(0.0, 10.0)
    )
    absolute = plant_fisher_information(
        other,
        obs,
        t2,
        [K1],
        sigma=0.05 * np.abs(np.asarray(_effluent_A(0.2, t2, a_in=2.0)))[:, None],
        t_span=(0.0, 10.0),
    )
    np.testing.assert_allclose(rel.matrix, absolute.matrix, rtol=1e-6)


def test_priors_add_precision(plant):
    """use_priors adds (d physical / d theta / std)^2 on the diagonal."""
    kw = dict(sigma=SIGMA, t_span=(0.0, 20.0))
    obs = [("tank", ["A"])]
    bare = plant_fisher_information(plant, obs, T_OBS, [K1, K3], **kw)
    with_prior = plant_fisher_information(
        plant, obs, T_OBS, [K1, K3], priors={K3: (0.5, 0.1)}, use_priors=True, **kw
    )
    added = with_prior.matrix - bare.matrix
    np.testing.assert_allclose(added, [[0.0, 0.0], [0.0, (0.5 / 0.1) ** 2]], atol=1e-9)


def test_fisher_input_validation(plant):
    obs = [("tank", ["A"])]
    with pytest.raises(ValueError, match="mode must be one of"):
        plant_fisher_information(plant, obs, T_OBS, [K1], sigma=SIGMA, mode="sideways")
    with pytest.raises(KeyError, match="Unknown plant parameter"):
        plant_fisher_information(plant, obs, T_OBS, ["chain.r1.kx"], sigma=SIGMA)
    with pytest.raises(KeyError, match="Unknown observed channel"):
        plant_fisher_information(plant, [("tank", ["Z"])], T_OBS, [K1], sigma=SIGMA)
    with pytest.raises(ValueError, match="strictly ascending"):
        plant_fisher_information(plant, obs, T_OBS[::-1], [K1], sigma=SIGMA)
    with pytest.raises(ValueError, match="y0 has 3 entries but there are 2 runs"):
        plant_fisher_information(plant, obs, [T_OBS, T_OBS], [K1], sigma=SIGMA, y0=[None] * 3)
    with pytest.raises(ValueError, match="plant has 2 entries but there are 1 runs"):
        plant_fisher_information([plant, plant], obs, T_OBS, [K1], sigma=SIGMA)
    with pytest.raises(ValueError, match="sigma must be positive"):
        plant_fisher_information(plant, obs, T_OBS, [K1], sigma=0.0)
    with pytest.raises(ValueError, match="mask leaves no observation"):
        plant_fisher_information(plant, obs, T_OBS, [K1], sigma=SIGMA, mask=False)
    with pytest.raises(ValueError, match="free_params must be non-empty"):
        plant_fisher_information(plant, obs, T_OBS, [], sigma=SIGMA)
    with pytest.raises(ValueError, match="sigma_floor must be >= 0"):
        plant_fisher_information(
            plant, obs, T_OBS, [K1], sigma=SIGMA, relative_sigma=True, sigma_floor=-1.0
        )
    with pytest.raises(ValueError, match="run 0: t_obs must be a non-empty 1-D array"):
        plant_fisher_information(plant, obs, jnp.ones((2, 2)), [K1], sigma=SIGMA)
    zero = plant.default_parameters().at[plant.parameter_index(K1)].set(0.0)
    with pytest.raises(ValueError, match="'chain.r1.k1' has transform 'positive_log'"):
        plant_fisher_information(plant, obs, T_OBS, [K1], sigma=SIGMA, params=zero)
    with pytest.raises(ValueError, match="'chain.r1.k1' has transform 'logit'"):
        plant_fisher_information(
            plant, obs, T_OBS, [K1], sigma=SIGMA, params=zero, transforms={K1: "logit"}
        )


def test_reverse_mode_solves_with_the_given_integrator(plant):
    """``integrator`` reaches the reverse-mode plant solve: a step budget too
    small for the run fails it."""
    with pytest.raises(RuntimeError, match="step budget \\(max_steps=2\\)"):
        plant_fisher_information(
            plant,
            [("tank", ["A"])],
            T_OBS,
            [K1],
            sigma=SIGMA,
            mode="reverse",
            integrator=aquakin.IntegratorConfig(max_steps=2),
        )


def test_runs_must_share_the_parameter_names(net, plant, decay_net):
    """A per-run plant built on another model has other parameters: rejected
    before any solve, naming the run."""
    other = Plant("decay_tank")
    other.add_unit(
        CSTRUnit(
            name="tank",
            model=decay_net,
            volume=V,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    with pytest.raises(ValueError, match="run 1: its plant has different parameters"):
        plant_fisher_information(
            [plant, other], [("tank", ["A"])], [T_OBS, T_OBS], [K1], sigma=SIGMA
        )


def test_whole_stream_observable_is_every_channel(plant):
    """An observable with no channels observes every species of the stream,
    in model order -- the same information as naming them all."""
    kw = dict(sigma=SIGMA, t_span=(0.0, 20.0))
    whole = plant_fisher_information(plant, ["tank"], T_OBS, [K1, K3], **kw)
    named = plant_fisher_information(plant, [("tank", ["A", "B", "D", "E"])], T_OBS, [K1, K3], **kw)
    np.testing.assert_allclose(whole.jacobian, named.jacobian, rtol=1e-12)
    # D and E see k3, so the whole stream identifies it where A and B alone cannot.
    assert whole.matrix[1, 1] > 0.0


# --- Profile likelihood ------------------------------------------------------


@pytest.mark.slow  # a Plant.calibrate fit per grid point
def test_profile_of_identified_parameter_matches_laplace(plant):
    """k1 is identified by effluent A and B: its likelihood-ratio interval
    brackets the truth and agrees with the Laplace interval (the problem is
    near-linear at this noise level)."""
    obs = [PlantObservable("tank", ("A", "B"))]
    truth = plant.default_parameters()
    data = _observations(plant, truth, obs)
    sigma = jnp.asarray([0.01, 0.01])
    kw = dict(observables=obs, t_span=(0.0, 20.0), loss="nll", sigma=sigma, use_priors=False)
    fim = plant_fisher_information(plant, obs, T_OBS, [K1, K2], sigma=sigma, t_span=(0.0, 20.0))
    sd_log = float(np.sqrt(fim.covariance()[0, 0]))  # marginal std of log k1
    grid = 0.2 * np.exp(np.linspace(-3.0, 3.0, 9) * sd_log)
    prof = plant_profile_likelihood(plant, data, T_OBS, [K1, K2], grid=grid, profile_param=K1, **kw)
    lo, hi = prof.ci
    assert lo is not None and hi is not None
    assert lo < 0.2 < hi
    # Laplace 95% interval in log k1: exp(+/- 1.96 sd).
    assert np.log(lo) == pytest.approx(np.log(0.2) - 1.96 * sd_log, abs=0.1 * sd_log)
    assert np.log(hi) == pytest.approx(np.log(0.2) + 1.96 * sd_log, abs=0.1 * sd_log)
    assert prof.mle == pytest.approx(0.2, rel=0.5 * sd_log)


@pytest.mark.slow  # a Plant.calibrate fit per grid point
def test_profile_of_unobserved_parameter_is_flat(plant):
    """k3 moves nothing observed: the profile is flat and the interval open on
    both sides."""
    obs = [PlantObservable("tank", ("A",))]
    truth = plant.default_parameters()
    data = _observations(plant, truth, obs)
    prof = plant_profile_likelihood(
        plant,
        data,
        T_OBS,
        [K1, K3],
        grid=[0.05, 0.2, 0.5, 2.0, 5.0],
        profile_param=K3,
        observables=obs,
        t_span=(0.0, 20.0),
        sigma=jnp.asarray([SIGMA]),
        use_priors=False,
    )
    assert prof.ci == (None, None)
    assert np.nanmax(prof.delta_loss) < 1e-6
    assert all(f is not None for f in prof.fits)


def test_profile_wires_each_grid_point_into_plant_calibrate(plant, monkeypatch):
    """With Plant.calibrate stubbed, the sweep pins the profiled value at its
    slot, drops it from the free set, warm-starts from the neighbour (one start,
    the neighbour's parameters and fitted state), shares one compiled-objective
    cache, forces laplace=False and forwards the remaining keywords."""
    import aquakin.plant.identifiability as ident

    idx = plant.parameter_index(K1)
    idx2 = plant.parameter_index(K2)
    calls = []

    def fake_calibrate(plant_, observations, t_obs, free, **kw):
        k1 = float(kw["params"][idx])
        calls.append(dict(kw, free=free, k1=k1))
        return SimpleNamespace(
            loss=100.0 * (k1 - 0.3) ** 2,
            params=kw["params"].at[idx2].add(1.0),  # mark the fit's re-fitted k2
            C0_fitted=[jnp.full(4, k1)],
        )

    monkeypatch.setattr(ident, "calibrate_plant", fake_calibrate)
    grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    prof = plant_profile_likelihood(
        plant,
        jnp.zeros((T_OBS.shape[0], 1)),
        T_OBS,
        [K1, K2],
        grid=grid,
        profile_param=K1,
        free_ic=aquakin.FreeICConfig(["tank.A"]),
        optimizer=OptimizerConfig(n_starts=3),
        observables=[("tank", ["A"])],
        t_span=(0.0, 20.0),
        polish=False,
    )
    assert [c["k1"] for c in calls] == pytest.approx([0.3, 0.4, 0.5, 0.2, 0.1])
    assert all(c["free"] == [K2] and c["laplace"] is False for c in calls)
    assert all(c["observables"] == [("tank", ["A"])] and c["t_span"] == (0.0, 20.0) for c in calls)
    assert len({id(c["_compiled_cache"]) for c in calls}) == 1
    cold, warm = calls[0], calls[1]
    assert cold["optimizer"].n_starts == 3 and cold["_ic_start"] is None
    assert warm["optimizer"].n_starts == 1
    # Warm from the anchor's fit: its re-fitted k2 (marked +1) and its state.
    assert float(warm["params"][idx2]) == pytest.approx(float(cold["params"][idx2]) + 1.0)
    assert float(warm["_ic_start"][0][0]) == pytest.approx(0.3)
    # The profile is the stub's parabola: minimum at 0.3, delta = 100 dk^2.
    assert prof.mle == 0.3 and prof.profiled == K1
    np.testing.assert_allclose(prof.delta_loss, [4.0, 1.0, 0.0, 1.0, 4.0])
    lo, hi = prof.ci
    assert lo == pytest.approx(0.2 - 0.1 * (1.92 - 1.0) / 3.0)
    assert hi == pytest.approx(0.4 + 0.1 * (1.92 - 1.0) / 3.0)


def test_profile_without_free_ic_passes_no_pool_start(plant, monkeypatch):
    """Without free_ic a warm point passes no ``_ic_start`` (the fit has no
    pools to seed), and the default parameters seed the cold fit."""
    import aquakin.plant.identifiability as ident

    calls = []

    def fake_calibrate(plant_, observations, t_obs, free, **kw):
        calls.append(kw)
        return SimpleNamespace(loss=0.0, params=kw["params"], C0_fitted=[jnp.zeros(4)])

    monkeypatch.setattr(ident, "calibrate_plant", fake_calibrate)
    plant_profile_likelihood(
        plant,
        jnp.zeros((T_OBS.shape[0], 1)),
        T_OBS,
        [K1, K2],
        grid=[0.1, 0.2],
        profile_param=K1,
        observables=[("tank", ["A"])],
        polish=False,
    )
    assert all(c["_ic_start"] is None for c in calls)
    default_k2 = float(plant.default_parameters()[plant.parameter_index(K2)])
    assert float(calls[0]["params"][plant.parameter_index(K2)]) == default_k2


def test_profile_input_validation(plant):
    obs = [("tank", ["A"])]
    kw = dict(observables=obs, sigma=jnp.asarray([SIGMA]))
    data = jnp.zeros((T_OBS.shape[0], 1))
    with pytest.raises(KeyError, match="Unknown profile_param"):
        plant_profile_likelihood(plant, data, T_OBS, [K1], grid=[0.1], profile_param="x.y", **kw)
    with pytest.raises(ValueError, match="free_params is empty"):
        plant_profile_likelihood(plant, data, T_OBS, [K1], grid=[0.1], profile_param=K1, **kw)
    with pytest.raises(ValueError, match="grid must be a non-empty"):
        plant_profile_likelihood(plant, data, T_OBS, [K1, K2], grid=[], profile_param=K1, **kw)
    with pytest.raises(TypeError, match="laplace=False"):
        plant_profile_likelihood(
            plant, data, T_OBS, [K1, K2], grid=[0.1], profile_param=K1, laplace=True, **kw
        )


@pytest.mark.slow  # a BSM1 variational solve
def test_bsm1_forward_matches_jacobian_built_from_solve_sensitivity():
    """On BSM1 (one day from the warm start) the forward-mode information of
    tank-5 SNH and SNO -- observed through the unit, whose outlet is its state
    -- is the J^T J built by hand from Plant.solve_sensitivity."""
    from aquakin.plant.bsm import bsm1_warm_start, build_bsm1

    asm1 = aquakin.load_model("asm1")
    feed = {"SS": 60.0, "SNH": 25.0, "XS": 200.0, "XB_H": 50.0, "SI": 30.0}
    feed |= {"XI": 25.0, "SND": 6.0, "XND": 10.0, "SALK": 7.0}
    plant = build_bsm1()
    plant.add_influent("feed", asm1.influent(feed, Q=18446.0))
    y0 = bsm1_warm_start(plant)
    wrt = ["asm1.muH", "asm1.muA", "asm1.KNH_A"]
    t_obs = jnp.linspace(0.25, 1.0, 4)
    sigma = jnp.asarray([0.5, 1.0])
    options = dict(max_steps=8_000)
    fim = plant_fisher_information(
        plant,
        [("tank5", ["SNH", "SNO"])],
        t_obs,
        wrt,
        sigma=sigma,
        t_span=(0.0, 1.0),
        y0=y0,
        sensitivity_options=options,
    )
    params = plant.default_parameters()
    _, _, S = plant.solve_sensitivity(
        params, wrt, t_span=(0.0, 1.0), t_eval=t_obs, y0=y0, **options
    )
    plant._build_state_layout()
    start = plant._state_layout["tank5"][0]
    rows = [start + asm1.species_index[s] for s in ("SNH", "SNO")]
    theta = np.asarray([params[plant.parameter_index(n)] for n in wrt])
    J = (np.asarray(S)[:, rows, :] * theta / np.asarray(sigma)[None, :, None]).reshape(-1, 3)
    assert np.all(np.isfinite(J)) and np.all(np.abs(J).max(axis=0) > 0.0)
    np.testing.assert_allclose(fim.matrix, J.T @ J, rtol=1e-8)


# --- Ranking sensor sets -----------------------------------------------------


def _count_sensitivity_solves(monkeypatch, plant):
    """Wrap ``plant.solve_sensitivity`` to count its calls."""
    calls = []
    original = plant.solve_sensitivity

    def counted(*args, **kw):
        calls.append(1)
        return original(*args, **kw)

    monkeypatch.setattr(plant, "solve_sensitivity", counted)
    return calls


def _criterion_of(plant, observables, sigma, free, criterion="D", **kw):
    """The criterion of one sensor set's information, the independent way."""
    fim = plant_fisher_information(plant, observables, T_OBS, free, sigma=sigma, **kw)
    return aquakin.design_criterion(fim, criterion)


def test_rank_scores_each_set_from_one_shared_solve(plant, monkeypatch):
    """Three candidates are scored from ONE sensitivity solve, and each score
    is the criterion of that set's own plant_fisher_information -- including
    the base's information and a set that names the port two ways."""
    calls = _count_sensitivity_solves(monkeypatch, plant)
    free = [K1, K2]
    base_obs, base_sigma = [("tank", ["A", "B"])], 0.05
    cands = [
        SensorSet([("tank", ["B"])], sigma=0.01),
        SensorSet([("tank.out", ["E", "B"])], sigma=[0.05, 0.03]),
        SensorSet([("tank", ["D"])], sigma=0.01),  # sees only k3: adds nothing
    ]
    ranked = plant_rank_sensor_sets(
        plant,
        cands,
        T_OBS,
        free,
        base=SensorSet(base_obs, sigma=base_sigma),
        t_span=(0.0, 20.0),
    )
    assert len(calls) == 1
    kw = dict(t_span=(0.0, 20.0))
    with_base = [
        ([("tank", ["B"])], [0.01]),
        ([("tank", ["E", "B"])], [0.05, 0.03]),
        ([("tank", ["D"])], [0.01]),
    ]
    expected = [
        _criterion_of(plant, base_obs + obs, [base_sigma] * 2 + sig, free, **kw)
        for obs, sig in with_base
    ]
    base_value = _criterion_of(plant, base_obs, base_sigma, free, **kw)
    by_index = {c.index: c for c in ranked}
    for i, v in enumerate(expected):
        assert by_index[i].value == pytest.approx(v, rel=1e-9)
        assert by_index[i].improvement == pytest.approx(v - base_value, rel=1e-6, abs=1e-9)
    # B (sigma 0.01) beats B (sigma 0.03) beats D, which adds nothing.
    assert [c.index for c in ranked] == [0, 1, 2]
    assert by_index[0].improvement > by_index[1].improvement > 0.0
    assert by_index[2].improvement == pytest.approx(0.0, abs=1e-9)


def test_rank_without_base_and_with_a_smaller_is_better_criterion(plant):
    """Without a base there is no improvement; "A" ranks the smallest summed
    variance first. Alone, a set seeing only A cannot separate k2: singular,
    so it ranks last with an infinite A."""
    cands = [
        SensorSet([("tank", ["A"])], sigma=0.01),
        SensorSet([("tank", ["A", "B"])], sigma=0.01),
    ]
    ranked = plant_rank_sensor_sets(
        plant, cands, T_OBS, [K1, K2], criterion="A", t_span=(0.0, 20.0)
    )
    assert [c.index for c in ranked] == [1, 0]
    assert ranked[1].value == np.inf and all(c.improvement is None for c in ranked)
    expected = _criterion_of(plant, [("tank", ["A", "B"])], 0.01, [K1, K2], "A", t_span=(0.0, 20.0))
    assert ranked[0].value == pytest.approx(expected, rel=1e-9)


def test_rank_averages_over_parameter_draws(plant, monkeypatch):
    """With param_draws the score is the mean of the per-draw criteria, one
    sensitivity solve per draw (shared by every candidate)."""
    calls = _count_sensitivity_solves(monkeypatch, plant)
    base = plant.default_parameters()
    idx = plant.parameter_index(K1)
    draws = jnp.stack([base.at[idx].set(0.1), base.at[idx].set(0.4)])
    cands = [SensorSet([("tank", ["A", "B"])], sigma=0.01), SensorSet(["tank"], sigma=0.01)]
    ranked = plant_rank_sensor_sets(
        plant, cands, T_OBS, [K1, K2], param_draws=draws, t_span=(0.0, 20.0)
    )
    assert len(calls) == 2
    expected = np.mean(
        [
            _criterion_of(
                plant, [("tank", ["A", "B"])], 0.01, [K1, K2], params=d, t_span=(0.0, 20.0)
            )
            for d in draws
        ]
    )
    assert {c.index: c.value for c in ranked}[0] == pytest.approx(expected, rel=1e-9)


def test_rank_relative_sigma_mask_subset_and_runs(net, plant):
    """A set's relative sigma, mask and the subset criterion reach the score
    exactly as plant_fisher_information applies them, over a two-run design
    with one plant per run."""
    other = _plant(net, a_in=2.0)
    t2 = jnp.linspace(1.0, 10.0, 5)
    mask = np.ones((T_OBS.shape[0], 1), dtype=bool)
    mask[::2] = False
    ss = SensorSet([("tank", ["B"])], sigma=0.1, relative_sigma=True, sigma_floor=1e-3)
    masked = SensorSet([("tank", ["A"])], sigma=0.01, mask=mask)
    runs = dict(t_span=[(0.0, 20.0), (0.0, 10.0)])
    ranked = plant_rank_sensor_sets(
        [plant, other], [ss], [T_OBS, t2], [K1, K2], subset=[K1], **runs
    )
    fim = plant_fisher_information(
        [plant, other],
        [("tank", ["B"])],
        [T_OBS, t2],
        [K1, K2],
        sigma=0.1,
        relative_sigma=True,
        sigma_floor=1e-3,
        **runs,
    )
    assert ranked[0].value == pytest.approx(
        aquakin.design_criterion(fim, "D", subset=[K1]), rel=1e-9
    )
    # A mask that does not suit the second run's sampling times is rejected.
    with pytest.raises(ValueError, match="sensor set 0, run 1: mask of shape"):
        plant_rank_sensor_sets([plant, other], [masked], [T_OBS, t2], [K1, K2], **runs)


def test_rank_forward_and_reverse_agree(plant):
    cands = [SensorSet([("tank", ["A"])], sigma=0.01), SensorSet([("tank", ["B"])], sigma=0.01)]
    kw = dict(t_span=(0.0, 20.0), base=SensorSet([("tank", ["B"])], sigma=0.05))
    fwd = plant_rank_sensor_sets(plant, cands, T_OBS, [K1, K2], mode="forward", **kw)
    rev = plant_rank_sensor_sets(plant, cands, T_OBS, [K1, K2], mode="reverse", **kw)
    assert [c.index for c in fwd] == [c.index for c in rev]
    for f, r in zip(fwd, rev):
        assert f.value == pytest.approx(r.value, rel=1e-3)


def test_rank_maps_each_set_to_its_own_stream(net):
    """On a two-tank chain, sets reading different streams (and one reading
    both, in reverse order) each score as their own plant_fisher_information:
    every stream's channels sit at the right offset in the shared solve."""
    chain = Plant("two_tanks")
    for name in ("tank1", "tank2"):
        chain.add_unit(
            CSTRUnit(
                name=name,
                model=net,
                volume=V,
                input_port_names=["inlet"],
                conditions={"T": 293.15},
            )
        )
    chain.connect("tank1", "tank2.inlet")
    feed = jnp.asarray([1.0, 0.0, 1.0, 0.0])
    chain.add_influent(
        "feed",
        InfluentSeries(
            t=jnp.asarray([0.0, 100.0]), Q=jnp.asarray([Q, Q]), C=jnp.stack([feed, feed]), model=net
        ),
        to="tank1.inlet",
    )
    obs_sets = [
        [("tank1", ["A", "B"])],
        [("tank2", ["B", "A"])],
        [("tank2", ["D"]), ("tank1", ["B"]), ("tank2", ["A"])],
    ]
    free = [K1, K2]
    sigmas = [[0.01, 0.02], [0.01, 0.02], [0.01, 0.02, 0.03]]
    cands = [SensorSet(o, sigma=sig) for o, sig in zip(obs_sets, sigmas)]
    kw = dict(t_span=(0.0, 20.0))
    ranked = plant_rank_sensor_sets(chain, cands, T_OBS, free, **kw)
    by_index = {c.index: c.value for c in ranked}
    for i, (o, c) in enumerate(zip(obs_sets, cands)):
        fim = plant_fisher_information(chain, o, T_OBS, free, sigma=c.sigma, **kw)
        assert np.isfinite(by_index[i])
        assert by_index[i] == pytest.approx(aquakin.design_criterion(fim, "D"), rel=1e-9)
    # The two tanks carry different information, so a column mix-up would show.
    assert abs(by_index[0] - by_index[1]) > 1e-3


def test_rank_input_validation(plant):
    ok = [SensorSet([("tank", ["A"])], sigma=0.01)]
    with pytest.raises(ValueError, match="at least one SensorSet"):
        plant_rank_sensor_sets(plant, [], T_OBS, [K1])
    with pytest.raises(ValueError, match="Unknown criterion"):
        plant_rank_sensor_sets(plant, ok, T_OBS, [K1], criterion="Z")
    with pytest.raises(KeyError, match="Unknown subset name"):
        plant_rank_sensor_sets(plant, ok, T_OBS, [K1], subset=[K3])
    with pytest.raises(ValueError, match="param_draws has 2 columns"):
        plant_rank_sensor_sets(plant, ok, T_OBS, [K1], param_draws=jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="sigma_floor must be >= 0"):
        bad = SensorSet([("tank", ["A"])], sigma=0.01, relative_sigma=True, sigma_floor=-1.0)
        plant_rank_sensor_sets(plant, [bad], T_OBS, [K1])
    with pytest.raises(KeyError, match="Unknown observed channel"):
        plant_rank_sensor_sets(plant, [SensorSet([("tank", ["Z"])], sigma=1.0)], T_OBS, [K1])
