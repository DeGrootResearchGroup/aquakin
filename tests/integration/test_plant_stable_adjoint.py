"""Cap-free reverse-mode gradients through the plant solve (``gradient="stable_adjoint"``).

A stiff plant (the BSM2 water line plus the ADM1 digester) cannot be
differentiated by the default reverse adjoint over a useful horizon: the
through-the-solve adjoint of the stiff implicit method returns non-finite values
unless the integrator step is capped, and capping fails the whole-plant solve.
``plant.solve(gradient="stable_adjoint")`` forms the gradient instead with the
hand-written discrete adjoint (the forward is a robust adaptive ESDIRK solve, the
reverse a per-step transposed solve over the saved trajectory), which is finite
at any step size.

The headline check is a gradient that flows from a *water-line* observation back
through the digester, the activated-sludge to anaerobic-digestion interface, and
the recycle to an ADM1 (digester) parameter -- a cross-network gradient -- and
matches a central finite difference. The cheap API-guard tests do not integrate.
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.bsm.bsm2 import (
    build_bsm2,
    bsm2_constant_influent,
    bsm2_parameters,
)

# Near-uniform activated-sludge warm start (the slow inerts are ~uniform across
# reactors at the published steady state; the fast variables relax within hours),
# so a short solve starts near steady state instead of from a stiff clean start.
_WARM = {"SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
         "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
         "SND": 0.7, "XND": 3.0, "SALK": 5.0}
_TANKS = ("tank1", "tank2", "tank3", "tank4", "tank5")


def _bsm2_plant():
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    warm = asm1.concentrations(_WARM)
    y0 = plant.initial_state(overrides={tk: warm for tk in _TANKS})
    return asm1, adm1, plant, y0


# --- cheap API guards (no integration) -------------------------------------

def test_invalid_gradient_raises():
    _asm1, _adm1, plant, y0 = _bsm2_plant()
    with pytest.raises(ValueError, match="jax_adjoint.*stable_adjoint"):
        plant.solve(t_span=(0.0, 1.0), y0=y0, gradient="not_a_mode")


def test_stable_adjoint_rejects_adjoint_and_dtmax():
    """``stable_adjoint`` controls its own adjoint and steps; passing the
    diffrax adjoint or a dtmax cap alongside it is a usage error."""
    _asm1, _adm1, plant, y0 = _bsm2_plant()
    with pytest.raises(ValueError, match="do not also pass"):
        plant.solve(t_span=(0.0, 1.0), y0=y0, gradient="stable_adjoint", dtmax=1e-2)
    with pytest.raises(ValueError, match="do not also pass"):
        plant.solve(t_span=(0.0, 1.0), y0=y0, gradient="stable_adjoint",
                    adjoint=diffrax.DirectAdjoint())


# --- the cross-interface gradient (slow: integrates the whole plant) -------

def _solve_kwargs():
    # The warm-started forward takes ~205 adaptive steps over a few days, so a
    # small max_steps suffices. Under gradient="stable_adjoint" max_steps also
    # sizes the backward scan's trajectory buffer, so keeping it tight is what
    # keeps the reverse pass cheap.
    return dict(rtol=1e-5, atol=1e-3, max_steps=2_000)


@pytest.mark.validation
def test_stable_adjoint_forward_matches_jax_adjoint():
    """The stable-adjoint forward primal equals the standard (jax_adjoint) solve;
    both integrate the same RHS with Kvaerno5, so they agree closely."""
    _asm1, _adm1, plant, y0 = _bsm2_plant()
    T = 3.0
    teval = jnp.array([T])
    a = plant.solve(t_span=(0.0, T), t_eval=teval, y0=y0, **_solve_kwargs())
    b = plant.solve(t_span=(0.0, T), t_eval=teval, y0=y0, gradient="stable_adjoint",
                    **_solve_kwargs())
    for unit, sp in (("tank1", "SNO"), ("tank5", "SNH"), ("digester", "S_gas_ch4")):
        assert float(b.C_named(unit, sp)[-1]) == pytest.approx(
            float(a.C_named(unit, sp)[-1]), rel=1e-3)


@pytest.mark.validation
def test_stable_adjoint_cross_interface_gradient_matches_fd():
    """Gradient of a water-line output with respect to an ADM1 (digester) rate,
    through the interface and the recycle, is finite and matches central FD."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    gidx = asm1.n_params + adm1.param_index["k_m_ac"]   # acetate-uptake max rate
    theta0 = float(base[gidx])
    T = 3.0

    def g(theta):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient="stable_adjoint", **_solve_kwargs())
        return sol.C_named("tank1", "SNO")[-1]   # water-line nitrate

    grad = float(jax.grad(g)(theta0))
    assert np.isfinite(grad)
    # A digester rate genuinely moves the water line through the reject recycle.
    assert grad != 0.0

    h = theta0 * 1e-3
    fd = (float(g(theta0 + h)) - float(g(theta0 - h))) / (2.0 * h)
    # The discrete adjoint is the exact gradient of the forward solve; it agrees
    # with the central difference to the finite-difference truncation/solver floor.
    assert grad == pytest.approx(fd, rel=2e-3)


@pytest.mark.validation
def test_auto_gradient_defaults_to_stable_adjoint():
    """With the default ``gradient="auto"`` (nothing passed), a stiff plant
    gradient is finite and matches the explicit stable-adjoint gradient -- the
    auto router sends the differentiated solve down the cap-free path, while a
    plain forward solve still uses the fast jax_adjoint path."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    gidx = asm1.n_params + adm1.param_index["k_m_ac"]
    theta0 = float(base[gidx])
    T = 3.0

    def g(theta, gradient):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient=gradient, **_solve_kwargs())
        return sol.C_named("tank1", "SNO")[-1]

    auto = float(jax.grad(lambda th: g(th, "auto"))(theta0))
    explicit = float(jax.grad(lambda th: g(th, "stable_adjoint"))(theta0))
    assert np.isfinite(auto)
    assert auto == pytest.approx(explicit, rel=1e-6)

    # A concrete forward solve under the same default is unchanged (fast path).
    fwd_auto = float(g(theta0, "auto"))
    fwd_jax = float(g(theta0, "jax_adjoint"))
    assert fwd_auto == pytest.approx(fwd_jax, rel=1e-6)


@pytest.mark.validation
def test_stable_adjoint_transient_influent_gradient_matches_fd():
    """Under a time-varying (diurnal-flow) influent the cross-interface gradient
    is still finite and matches central finite differences. The discrete adjoint
    carries the integration time in the state, so it is exact for the
    non-autonomous plant right-hand side, not only for a constant influent."""
    from aquakin.plant.bsm.bsm2 import BSM2_Q_REF
    from aquakin.plant.influent import InfluentSeries

    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    # A diurnal flow modulation makes the plant RHS explicitly time-dependent.
    c_const = bsm2_constant_influent(asm1).C[0]
    n = 120
    t_inf = jnp.linspace(0.0, 4.0, n)
    q_inf = BSM2_Q_REF * (1.0 + 0.3 * jnp.sin(2.0 * jnp.pi * t_inf))
    plant.add_influent(
        "feed",
        InfluentSeries(t=t_inf, Q=q_inf, C=jnp.tile(c_const, (n, 1)), network=asm1),
        to="front_mix.fresh",
    )
    warm = asm1.concentrations(_WARM)
    y0 = plant.initial_state(overrides={tk: warm for tk in _TANKS})
    base = bsm2_parameters(asm1, adm1)
    gidx = asm1.n_params + adm1.param_index["k_m_ac"]
    theta0 = float(base[gidx])
    T = 3.0

    # The diurnal forcing makes the adaptive solve take ~2000 steps -- right at
    # the constant-influent shared cap. The exact count drifts a few percent
    # across CPU architectures, so a 2000 cap that passes on one platform trips
    # "maximum solver steps reached" on another. Give the transient solve headroom
    # (max_steps also sizes the stable-adjoint backward-scan buffer).
    kw = {**_solve_kwargs(), "max_steps": 5_000}

    def g(theta):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient="stable_adjoint", **kw)
        return sol.C_named("tank1", "SNO")[-1]

    grad = float(jax.grad(g)(theta0))
    assert np.isfinite(grad)
    assert grad != 0.0
    # The stable-adjoint gradient is the *exact* gradient of the discrete solve
    # and is platform-stable (it agrees across machines to ~5 significant
    # figures). The finite-difference reference is the noisy side: each g(theta+-h)
    # re-runs an adaptive stiff solve whose step sequence shifts discretely with
    # theta (and across CPU/XLA builds), so the central difference carries a
    # roughly theta-independent absolute noise, i.e. a relative error ~ noise/(2h*grad).
    # The earlier h = theta*1e-3 made that signal ~2e-4 -- near the atol=1e-3 floor --
    # so FD landed within 0.2% locally but ~8% off on the CI runner. A 10x larger
    # step lifts the signal an order of magnitude above the noise (FD error scales
    # as 1/h here, the sensitivity being near-linear); rel=2e-2 then covers the
    # residual platform spread while still catching a genuinely wrong gradient
    # (sign, magnitude). dtmax cannot pin the grid: gradient="stable_adjoint"
    # controls its own steps and rejects it.
    h = theta0 * 1e-2
    fd = (float(g(theta0 + h)) - float(g(theta0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=2e-2)


@pytest.mark.validation
def test_stable_adjoint_gradient_finite_through_full_param_vector():
    """A full-parameter reverse gradient (the calibration case) is finite, where
    the default through-the-solve adjoint is not without a step cap."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    T = 2.0

    def loss(p):
        sol = plant.solve(t_span=(0.0, T), t_eval=jnp.array([T]), params=p, y0=y0,
                          gradient="stable_adjoint", **_solve_kwargs())
        return jnp.sum(sol.state[-1] ** 2)

    g = jax.grad(loss)(base)
    assert g.shape == base.shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)


@pytest.mark.validation
def test_stable_adjoint_solve_is_jittable():
    """The stable-adjoint plant solve can be wrapped in ``jax.jit``: its ``atol``
    no longer forces concretization. The jitted value and gradient match the
    eager ones. Jitting the calibration loss is what amortizes the (large) solve
    compile across optimizer iterations."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    gidx = asm1.n_params + adm1.param_index["k_m_ac"]
    T = 3.0
    teval = jnp.array([T])

    def g(theta):
        p = base.at[gidx].set(theta)
        sol = plant.solve(t_span=(0.0, T), t_eval=teval, params=p, y0=y0,
                          gradient="stable_adjoint", **_solve_kwargs())
        return sol.C_named("tank1", "SNO")[-1]

    theta0 = float(base[gidx])
    eager = float(g(theta0))
    jitted = float(jax.jit(g)(theta0))
    assert np.isfinite(jitted)
    assert jitted == pytest.approx(eager, rel=1e-6)

    ge = float(jax.grad(g)(theta0))
    gj = float(jax.jit(jax.grad(g))(theta0))
    assert np.isfinite(gj)
    assert gj == pytest.approx(ge, rel=1e-6)


@pytest.mark.validation
def test_stable_adjoint_forward_solve_is_cached():
    """A repeat *forward* stable-adjoint solve reuses the compiled closure (the
    parameter-sweep case), while a traced call (a gradient through the solve)
    bypasses the cache: the discrete adjoint's ``custom_vjp`` must be traced
    directly into the outer computation, not routed through an inner ``jax.jit``
    under an outer reverse-mode pass."""
    asm1, adm1, plant, y0 = _bsm2_plant()
    base = bsm2_parameters(asm1, adm1)
    T = 3.0
    kw = dict(t_span=(0.0, T), t_eval=jnp.array([T]), y0=y0,
              gradient="stable_adjoint", **_solve_kwargs())

    def _sa_keys():
        return [k for k in plant._jit_cache if k[0] == "stable_adjoint"]

    a = plant.solve(params=base, **kw)
    assert len(_sa_keys()) == 1                       # one compiled stable solve
    cached = plant._jit_cache[_sa_keys()[0]]

    plant.solve(params=base * 1.01, **kw)             # different params, same sig
    assert plant._jit_cache[_sa_keys()[0]] is cached  # reused, not rebuilt
    c = plant.solve(params=base, **kw)                # same params -> same result
    assert float(c.C_named("tank1", "SNO")[-1]) == pytest.approx(
        float(a.C_named("tank1", "SNO")[-1]), rel=1e-10)

    # A traced (gradient) call adds no stable-adjoint cache entry.
    n_before = len(_sa_keys())
    jax.grad(lambda th: plant.solve(
        params=base.at[0].set(th), **kw).C_named("tank1", "SNO")[-1]
    )(float(base[0]))
    assert len(_sa_keys()) == n_before
