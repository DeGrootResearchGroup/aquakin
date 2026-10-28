"""The Takács settler's initial state and a plant solve trace under jit / grad.

``Plant.solve`` forms its default ``atol`` from ``initial_state()``, which runs
inside the caller's trace. The settler's seed used to convert staged JAX values
to Python floats there, and the colored-Jacobian setup of the differentiable
solve then read the traced ``atol`` as a number, so ``jax.jit`` of a solve raised
a ``ConcretizationTypeError`` for every Takács configuration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.takacs import TakacsClarifier


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


@pytest.mark.parametrize("composition_mode", ["lumped_tss", "per_species"])
@pytest.mark.parametrize("soluble_holdup", [False, True])
@pytest.mark.parametrize("init_underflow_Q", [None, 18446.0])
def test_initial_state_traces_and_matches_eager(
    asm1, composition_mode, soluble_holdup, init_underflow_Q
):
    """Every seed path (uniform / settled blanket, both composition modes, with
    and without the soluble holdup) evaluates under ``jax.jit`` and equals the
    eager seed exactly."""
    clar = TakacsClarifier(
        name="c",
        model=asm1,
        area=1500.0,
        height=4.0,
        underflow_Q=18446.0,
        init_underflow_Q=init_underflow_Q,
        composition_mode=composition_mode,
        soluble_holdup=soluble_holdup,
    )
    eager = np.asarray(clar.initial_state())
    traced = np.asarray(jax.jit(lambda: clar.initial_state())())
    assert eager.shape == (clar.state_size,)
    np.testing.assert_array_equal(traced, eager)


def test_initial_state_seed_values(asm1):
    """The seed values themselves: the uniform lumped seed is the default
    composition's TSS in every layer, and the blanket seed is the
    flat-then-spike profile (clear above the feed, thickened bottom layer)."""
    defaults = np.asarray(asm1.default_concentrations())
    uni = TakacsClarifier(
        name="u", model=asm1, area=1500.0, height=4.0, underflow_Q=18446.0,
        composition_mode="lumped_tss",
    )
    factors = np.asarray(uni._part_tss_factors)
    X_f = float(np.sum(defaults[uni._part_indices] * factors))
    np.testing.assert_allclose(np.asarray(uni.initial_state()), X_f, rtol=1e-15)

    blanket = TakacsClarifier(
        name="b", model=asm1, area=1500.0, height=4.0, overflow_Q=18446.0,
        init_underflow_Q=18446.0, composition_mode="lumped_tss",
    )
    tss = np.asarray(blanket.initial_state())
    assert tss[0] == pytest.approx(2.0 * X_f, rel=1e-15)  # (Q_over + Q_u) / Q_u = 2
    np.testing.assert_allclose(tss[1 : blanket.feed_layer + 1], X_f, rtol=1e-15)
    X_clar = max(blanket._fns * X_f, 1.0)
    np.testing.assert_allclose(tss[blanket.feed_layer + 1 :], X_clar, rtol=1e-15)


def test_default_solve_atol_is_concrete_under_jit(asm1):
    """With a concrete ``y0`` the solve's default ``atol`` is computed outside
    the trace, so it is a concrete array (equal to the eager one) even inside
    ``jax.jit``; the colored-Jacobian setup reads it as a Python scalar."""
    from aquakin.plant import Plant
    from aquakin.plant.influent import InfluentSeries

    plant = Plant("settler")
    plant.add_unit(
        TakacsClarifier(
            name="clar", model=asm1, area=1500.0, height=4.0, underflow_Q=18446.0,
            init_underflow_Q=18446.0, composition_mode="lumped_tss",
            soluble_holdup=True,
        )
    )
    C = asm1.default_concentrations()
    plant.add_influent(
        "feed",
        InfluentSeries(t=jnp.asarray([0.0, 1.0]), Q=jnp.asarray([2e4, 2e4]),
                       C=jnp.stack([C, C]), model=asm1),
        to="clar.inlet",
    )
    y0 = plant.initial_state()
    eager = np.asarray(plant._default_solve_atol(y0))
    seen = {}

    @jax.jit
    def f(x):
        atol = plant._default_solve_atol(y0)
        seen["concrete"] = not isinstance(atol, jax.core.Tracer)
        seen["value"] = np.asarray(atol) if seen["concrete"] else None
        return x

    f(1.0)
    assert seen["concrete"]
    np.testing.assert_array_equal(seen["value"], eager)


@pytest.mark.slow
@pytest.mark.parametrize(
    "composition_mode, soluble_holdup",
    [("lumped_tss", True), ("per_species", False)],
)
def test_bsm1_takacs_solve_jit_and_grad(asm1, composition_mode, soluble_holdup):
    """``jax.jit`` and ``jax.grad`` of a short BSM1 solve with the Takács settler
    (the default ``atol`` path), including the validated lumped-TSS
    configuration. The jitted value matches the eager one to integration
    tolerance and the gradient is finite."""
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start, load_bsm1_influent

    plant = build_bsm1(
        model=asm1,
        use_takacs=True,
        settler_soluble_holdup=soluble_holdup,
        settler_composition_mode=composition_mode,
    )
    plant.add_influent("feed", load_bsm1_influent("dry", asm1))
    y0 = bsm1_warm_start(plant)
    p0 = plant.parameter_values({})

    def loss(p):
        s = plant.solve((0.0, 1.0), t_eval=jnp.asarray([1.0]), params=p, y0=y0)
        return jnp.sum(s.final_state**2)

    eager = float(loss(p0))
    jitted = float(jax.jit(loss)(p0))
    # Under jit the parameters are tracers, so gradient="auto" runs the
    # stable_adjoint integrator rather than the eager forward one: the two agree
    # to integration tolerance (~3e-7 here), not to round-off.
    assert jitted == pytest.approx(eager, rel=1e-5)
    g = jax.grad(loss)(p0)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0
