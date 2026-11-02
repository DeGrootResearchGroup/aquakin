"""Dynamic (transient) plant sensitivity wrappers: ``plant.dynamic_sensitivity``
and ``plant.dynamic_dgsm``.

These differentiate a time-window output through the stiff dynamic solve -- the
dynamic counterpart of the steady-state IFT sensitivity. The wrapper's job is to
pick the adjoint that matches the AD direction (reverse -> the cap-free stable
adjoint; forward -> a forward-capable adjoint), which is the easy thing to get
wrong by hand. The solves are stiff plant integrations, so these are slow.
"""
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

# These are whole-plant stable-adjoint gradient tests: each builds a BSM1 plant and
# differentiates the stiff solve, compiling a multi-GB whole-plant program. Run
# serially in one process, a shard's worth of such tests accumulates compiled XLA
# programs faster than the allocator returns the freed memory to the OS, so the
# process RSS climbs test-by-test until the runner OOM-kills the worker -- the same
# memory wall that excludes the ``heavy`` whole-plant stable-adjoint validation
# tests from CI. So these carry the ``heavy`` marker too and run LOCALLY
# (``pytest -m heavy``), not on the shared CI runner. (Reducing ``max_steps`` does
# not change this: it shrinks only the trajectory buffer, not the compiled program,
# which is fixed by plant size.)
#
# ``max_steps`` is still kept tight -- a 2-day BSM1 solve takes a few hundred steps,
# so the cap below has a wide margin across the perturbed dgsm samples while keeping
# the per-solve saved-trajectory buffer small for fast local runs.
_MAX_STEPS = 8_000


def _bsm1():
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    asm1 = aquakin.load_model("asm1")
    infl = asm1.influent({"SS": 60., "SNH": 25., "XS": 200., "XB_H": 50.,
                          "SI": 30., "XI": 25., "SND": 6., "XND": 10.,
                          "SALK": 7.}, Q=18446.0)
    p = build_bsm1()
    p.add_influent("feed", infl)
    return p, asm1, bsm1_warm_start(p)


@pytest.mark.slow
@pytest.mark.heavy
def test_dynamic_sensitivity_modes_match_grad():
    """Reverse and forward dynamic sensitivity agree, match a manual stable-adjoint
    gradient (to machine precision) and finite differences. The wrapper selects the
    adjoint for each direction -- the footgun it exists to remove."""
    p, asm1, y0 = _bsm1()
    base = p.default_parameters()
    t_eval = jnp.linspace(0.0, 2.0, 5)

    def out_fn(sol):                                  # final effluent NO3 and NH4
        return jnp.array([sol.C_named("tank5", "SNO")[-1],
                          sol.C_named("tank5", "SNH")[-1]])

    kw = dict(output_fn=out_fn, t_span=(0.0, 2.0), t_eval=t_eval,
              wrt=["asm1.muA", "asm1.muH"], y0=y0, max_steps=_MAX_STEPS)
    Sr = np.asarray(p.dynamic_sensitivity(base, mode="reverse", **kw))
    Sf = np.asarray(p.dynamic_sensitivity(base, mode="forward", **kw))
    assert Sr.shape == (2, 2)
    assert np.allclose(Sf, Sr, rtol=1e-5, atol=1e-10)   # forward == reverse

    i = p.parameter_index("asm1.muA")
    th = float(base[i])

    def scalar(theta):
        pp = base.at[i].set(theta)
        sol = p.solve((0.0, 2.0), t_eval=t_eval, params=pp, y0=y0,
                      diff=aquakin.DifferentiationConfig(method="stable"),
                      integrator=aquakin.IntegratorConfig(max_steps=_MAX_STEPS))
        return sol.C_named("tank5", "SNO")[-1]

    g = float(jax.grad(scalar)(th))
    h = th * 1e-4
    fd = (scalar(th + h) - scalar(th - h)) / (2.0 * h)
    assert float(Sr[0, 0]) == pytest.approx(g, rel=1e-6)
    assert float(Sr[0, 0]) == pytest.approx(float(fd), rel=1e-4)


@pytest.mark.slow
@pytest.mark.heavy
def test_solve_sensitivity_matches_jacfwd():
    """Plant.solve_sensitivity -- the stable forward [y; S] variational solve
    (default staggered corrector: Kvaerno3 on the lean loop, colored Jacobian,
    cached recycle map) -- is finite and matches forward-mode jacfwd through the
    solve where both are finite. (The controller's error norm bounds S, so the
    same solve stays finite over long horizons where jacfwd through the stiff
    plant goes non-finite.)"""
    p, asm1, y0 = _bsm1()
    base = p.default_parameters()
    wrt = ["asm1.muA", "asm1.muH"]
    T = 2.0
    te = jnp.linspace(0.0, T, 5)
    p._build_state_layout()
    s0, _ = p._state_layout["tank5"]
    snh = s0 + asm1.species_index["SNH"]

    ts, ys, S = p.solve_sensitivity(base, wrt, t_span=(0.0, T), t_eval=te, y0=y0,
                                    max_steps=_MAX_STEPS)
    S = np.asarray(S)
    assert np.all(np.isfinite(S))
    assert ts.shape[0] == 5 and ys.shape == (5, y0.shape[0]) and S.shape[2] == 2
    S_aug = S[-1, snh, :]                       # d(tank5 SNH @T)/d[muA, muH]

    idx = [p.parameter_index(w) for w in wrt]

    def out(theta):
        pp = base.at[jnp.asarray(idx)].set(theta)
        sol = p.solve((0.0, T), t_eval=te, params=pp, y0=y0,
                      diff=aquakin.DifferentiationConfig(
                          mode="forward", method="through_solve"),
                      integrator=aquakin.IntegratorConfig(max_steps=_MAX_STEPS))
        return sol.C_named("tank5", "SNH")[-1]

    S_jf = np.asarray(jax.jacfwd(out)(base[jnp.asarray(idx)]))
    assert np.allclose(S_aug, S_jf, rtol=1e-4, atol=1e-10)


def _assert_correctors_agree(plant, params, wrt, *, rtol_S, rtol_y=1e-4, **kw):
    """The staggered and simultaneous correctors give the same trajectory and
    sensitivity, each sensitivity column compared relative to its largest entry."""
    _, ys_st, S_st = plant.solve_sensitivity(params, wrt, **kw)
    _, ys_si, S_si = plant.solve_sensitivity(params, wrt, corrector="simultaneous", **kw)
    S_st, S_si = np.asarray(S_st), np.asarray(S_si)
    assert np.all(np.isfinite(S_st)) and np.all(np.isfinite(S_si))
    scale = np.abs(S_si).max(axis=(0, 1), keepdims=True)
    assert np.max(np.abs(S_st - S_si) / scale) < rtol_S
    assert np.allclose(np.asarray(ys_st), np.asarray(ys_si), rtol=rtol_y, atol=1e-3)


@pytest.mark.slow
@pytest.mark.heavy
def test_solve_sensitivity_correctors_agree_bsm1():
    """On BSM1 over two days the staggered corrector (default) gives the
    simultaneous corrector's sensitivity to the solver tolerance -- the two
    solve the same augmented stages, one by the block-arrow Newton on the whole
    [y; S] system, the other state-first then sensitivity with the same
    factor."""
    p, asm1, y0 = _bsm1()
    _assert_correctors_agree(
        p,
        p.default_parameters(),
        ["asm1.muA", "asm1.muH", "asm1.bH", "asm1.KS"],
        rtol_S=1e-4,
        t_span=(0.0, 2.0),
        t_eval=jnp.linspace(0.0, 2.0, 5),
        y0=y0,
        rtol=1e-6,
        max_steps=_MAX_STEPS,
    )


@pytest.mark.slow
@pytest.mark.heavy
def test_solve_sensitivity_correctors_agree_bsm2_window():
    """The same agreement on a short window of the two-model BSM2 plant under
    the dynamic (dry-weather) influent, across the ASM1 water line and the ADM1
    digester: one water-line and one digester parameter. Both correctors bound
    an RMS error norm, so one component may move several times ``rtol``: the
    largest differences (measured 8e-4 of a sensitivity column's peak, 1.3e-4 of
    a state, at ``rtol=1e-5``) sit in the Takacs settler's blanket layers, whose
    threshold flux switches make them the most step-sequence sensitive. A wrong
    sensitivity stage gives errors of order one."""
    from aquakin.plant.bsm import bsm2_warm_start
    from aquakin.plant.bsm.bsm2 import bsm2_asm1_model, bsm2_parameters, build_bsm2
    from aquakin.plant.influent import load_bsm2_influent

    asm1, adm1 = bsm2_asm1_model(), aquakin.load_model("adm1")
    p = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    p.add_influent("feed", load_bsm2_influent("dry", asm1))
    _assert_correctors_agree(
        p,
        bsm2_parameters(asm1, adm1),
        ["asm1.muA", "adm1.k_m_ac"],
        rtol_S=3e-3,
        rtol_y=5e-4,
        t_span=(0.0, 0.5),
        t_eval=jnp.linspace(0.0, 0.5, 3),
        y0=bsm2_warm_start(p),
        rtol=1e-5,
        atol=1e-3,
        max_steps=_MAX_STEPS,
    )


@pytest.mark.slow
@pytest.mark.heavy
def test_solve_sensitivity_operating_influent_matches_fd():
    """solve_sensitivity differentiates OPERATING parameters -- a multiplicative
    scale on the influent flow and on a species' load -- through the same augmented
    [y;S] solve, matching finite differences. These are operating conditions, not
    kinetic parameters, so they reach the sensitivity only through the influent
    design override; the variational solve gains them over a parameter-only screen.
    """
    from aquakin.plant.bsm import build_bsm1
    from aquakin.plant.influent import InfluentSeries
    p, asm1, y0 = _bsm1()
    base = p.default_parameters()
    T = 0.5
    te = jnp.array([T])
    op = [{"kind": "influent_concentration", "port": "feed", "species": "SNH"},
          {"kind": "influent_flow", "port": "feed"}]
    ts, ys, S = p.solve_sensitivity(base, [], operating=op, t_span=(0.0, T),
                                    t_eval=te, y0=y0, max_steps=_MAX_STEPS)
    S = np.asarray(S)
    assert S.shape == (1, y0.shape[0], 2) and np.all(np.isfinite(S))
    p._build_state_layout()
    s0, _ = p._state_layout["tank5"]
    out = s0 + asm1.species_index["SNO"]
    sv = S[-1, out, :]                        # d(tank5 SNO @T)/d[nh_scale, q_scale]
    yb = float(np.asarray(ys)[-1, out])       # unscaled output = the variational primal

    infl = p.influents["feed"]
    eps = 1e-3

    def fd(snh=1.0, q=1.0):                    # re-solve at a scaled influent
        c = np.asarray(infl.C).copy()
        c[:, asm1.species_index["SNH"]] *= snh
        i2 = InfluentSeries(t=infl.t, Q=infl.Q * q, C=jnp.asarray(c),
                            model=asm1, T=infl.T)
        p2 = build_bsm1()
        p2.add_influent("feed", i2)
        return float(p2.solve(
            (0.0, T), t_eval=te, params=base, y0=y0,
            integrator=aquakin.IntegratorConfig(max_steps=_MAX_STEPS),
        ).C_named("tank5", "SNO")[-1])

    assert sv[0] == pytest.approx((fd(snh=1 + eps) - yb) / eps, rel=5e-3)
    assert sv[1] == pytest.approx((fd(q=1 + eps) - yb) / eps, rel=5e-3)


@pytest.mark.slow
@pytest.mark.heavy
@pytest.mark.xheavy
def test_dynamic_dgsm_matches_dgsm():
    """plant.dynamic_dgsm screens a transient output globally by reusing
    dynamic_sensitivity per sample. With the same Sobol seed it gives the same
    Sobol total-index bounds as the generic aquakin.dgsm over the same dynamic
    solve -- the wrapper just packages the fn + adjoint and jits the per-sample."""
    p, asm1, y0 = _bsm1()
    base = p.default_parameters()
    screen = ["asm1.muA", "asm1.muH", "asm1.bH"]
    idx = [p.parameter_index(s) for s in screen]
    val = np.array([float(base[i]) for i in idx])
    ranges = np.array([[v * 0.9, v * 1.1] for v in val])
    t_eval = jnp.linspace(0.0, 2.0, 5)

    def out_fn(sol):
        return jnp.array([sol.C_named("tank5", "SNO")[-1]])

    res = p.dynamic_dgsm(ranges, output_fn=out_fn, t_span=(0.0, 2.0), t_eval=t_eval,
                         wrt=screen, n_samples=4, seed=0, y0=y0, mode="reverse",
                         max_steps=_MAX_STEPS)
    assert res.sobol_total_bound.shape == (1, 3)

    def fn(x):
        pp = base.at[jnp.asarray(idx)].set(jnp.asarray(x))
        sol = p.solve((0.0, 2.0), t_eval=t_eval, params=pp, y0=y0,
                      diff=aquakin.DifferentiationConfig(method="stable"),
                      integrator=aquakin.IntegratorConfig(max_steps=_MAX_STEPS))
        return sol.C_named("tank5", "SNO")[-1]

    d = aquakin.dgsm(fn, ranges, input_names=screen, n_samples=4, seed=0,
                     diff=aquakin.DifferentiationConfig(mode="reverse"))
    mine, theirs = dict(res.ranked(0)), dict(d.ranked())
    for s in screen:
        assert mine[s] == pytest.approx(theirs[s], rel=1e-5, abs=1e-12)

    # convergence runs from the retained per-sample data
    counts, bound, se = res.convergence()
    assert bound.shape == (len(counts), 1, 3)
