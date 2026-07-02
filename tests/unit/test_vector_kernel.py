"""Vectorized rate kernel: bit-identicality, op-count reduction, AD, fallback.

The kernel (``aquakin/core/vector_kernel.py``) evaluates all reaction rates in
batched ops with a much smaller traced jaxpr than the scalar per-reaction stack.
Its contract is that it is **bit-identical** to that scalar path while cutting
the traced op count (which dominates stiff-solve compile time). These tests pin
both, plus the graceful fallback for an unsupported node type.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.core.vector_kernel import (
    UnsupportedNode,
    build_vectorized_rates,
)


# Models spanning the node set: inline-division Monods (asm1), pH-switch /
# pH-inhibit / safe_div / max (adm1), Arrhenius / state-derived pH (wats),
# the SUMO bio-P models, and a precipitation model (derived SI_/R_ fields).
_MODELS = [
    "ozone_bromate", "uv_h2o2", "asm1", "asm2d", "asm2d_tud", "asm3",
    "asm3_2step", "adm1", "wats_sewer", "wats_sewer_extended",
    "precipitation_struvite_calcite",
]


def _count_eqns(jaxpr):
    j = getattr(jaxpr, "jaxpr", jaxpr)
    n = 0
    for eq in j.eqns:
        n += 1
        for sub in eq.params.values():
            if hasattr(sub, "eqns"):
                n += _count_eqns(sub)
            elif isinstance(sub, (list, tuple)):
                for s in sub:
                    if hasattr(s, "eqns"):
                        n += _count_eqns(s)
    return n


def _scalar_rates(net, C, params, condition_arrays, loc_idx):
    """The scalar per-reaction path, bypassing the kernel (for comparison)."""
    return jnp.stack(
        [f(C, params, condition_arrays, loc_idx) for f in net.rate_callables]
    )


def _prepared_inputs(net, C, loc_idx=0):
    """Apply the same clip / derived-condition / temperature preprocessing
    ``rates`` does, so kernel and scalar are compared on identical inputs."""
    p = net.default_parameters()
    ca = net.default_conditions().fields
    Cc = jnp.maximum(C, 0.0) if net.clip_negative_states else C
    if net.derived_condition_fn is not None:
        ca = net._augment_conditions(Cc, p, ca, loc_idx)
    if net.temperature_corrections:
        p = net._apply_temperature(p, ca, loc_idx)
    return Cc, p, ca


@pytest.mark.parametrize("name", _MODELS)
def test_kernel_is_bit_identical_to_scalar(name):
    """The kernel reproduces the scalar rate vector exactly (every byte), at a
    randomized feasible state -- not merely close. Bit-identicality is the whole
    safety guarantee (every validated steady state is preserved)."""
    net = aquakin.load_model(name)
    assert net._rate_kernel is not None, f"{name}: kernel should be built"
    rng = np.random.default_rng(12345)
    for _ in range(8):
        C = jnp.asarray(rng.uniform(0.0, 5.0, net.n_species))
        Cc, p, ca = _prepared_inputs(net, C)
        r_scalar = _scalar_rates(net, Cc, p, ca, 0)
        r_kernel = net._rate_kernel(Cc, p, ca, 0)
        # Exact equality (bit-for-bit), not allclose.
        assert jnp.array_equal(r_scalar, r_kernel), name


@pytest.mark.parametrize("name", _MODELS)
def test_kernel_reduces_op_count(name):
    """The kernel's traced jaxpr is no larger than the scalar stack's -- the
    compile-time payoff. (For the big stiff models it is several-fold smaller;
    here we only assert it never regresses.)"""
    net = aquakin.load_model(name)
    C = net.default_concentrations()
    Cc, p, ca = _prepared_inputs(net, C)
    scalar_eqns = _count_eqns(
        jax.make_jaxpr(lambda C, p: _scalar_rates(net, C, p, ca, 0))(Cc, p)
    )
    kernel_eqns = _count_eqns(
        jax.make_jaxpr(lambda C, p: net._rate_kernel(C, p, ca, 0))(Cc, p)
    )
    assert kernel_eqns <= scalar_eqns, (
        f"{name}: kernel {kernel_eqns} eqns > scalar {scalar_eqns}"
    )


def test_kernel_op_count_reduction_is_large_on_a_big_model():
    """A concrete floor: the kernel cuts the WATS rate jaxpr by >= 3x, the
    headline result of issue #373."""
    net = aquakin.load_model("wats_sewer_extended")
    C = net.default_concentrations()
    Cc, p, ca = _prepared_inputs(net, C)
    scalar = _count_eqns(
        jax.make_jaxpr(lambda C, p: _scalar_rates(net, C, p, ca, 0))(Cc, p))
    kernel = _count_eqns(
        jax.make_jaxpr(lambda C, p: net._rate_kernel(C, p, ca, 0))(Cc, p))
    assert scalar / kernel >= 3.0


@pytest.mark.parametrize("name", _MODELS)
def test_kernel_jacobian_is_finite_and_matches_scalar(name):
    """``dr/dC`` through the kernel is finite and matches the scalar Jacobian on
    every model. The forward value being bit-identical does NOT imply the
    derivative is sound (a constant-exponent pow can NaN the derivative while the
    value is exact), so this gradient-level check is the necessary companion to
    the forward bit-identicality test."""
    net = aquakin.load_model(name)
    rng = np.random.default_rng(7)
    C = jnp.asarray(rng.uniform(0.0, 5.0, net.n_species))
    Cc, p, ca = _prepared_inputs(net, C)
    Jk = jax.jacfwd(lambda CC: net._rate_kernel(CC, p, ca, 0))(Cc)
    Js = jax.jacfwd(lambda CC: _scalar_rates(net, CC, p, ca, 0))(Cc)
    assert bool(jnp.all(jnp.isfinite(Jk))), f"{name}: non-finite kernel Jacobian"
    assert jnp.allclose(Jk, Js, rtol=1e-9, atol=1e-12), name


def test_grad_flows_through_kernel():
    """jax.grad flows through a kernel-backed batch solve and is finite."""
    net = aquakin.load_model("asm1")
    reactor = aquakin.BatchReactor(
        net, net.default_conditions(),
        integrator=aquakin.IntegratorConfig(dtmax=1e-2))
    C0 = net.default_concentrations()
    p = net.default_parameters()
    t_eval = jnp.array([5.0])

    def loss(pp):
        return jnp.sum(reactor.solve(C0, t_span=(0.0, 5.0), t_eval=t_eval,
                                     params=pp).C[-1])

    g = jax.grad(loss)(p)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_kernel_matches_scalar_gradient():
    """The kernel's gradient w.r.t. params matches the scalar path's gradient.

    The *forward* rates are bit-identical, but the reverse-mode adjoint flows
    through different ops (gather / concatenate vs slice / stack), so cotangents
    accumulate in a different summation order -- the gradients agree to
    machine precision, not bit-for-bit. A tight tolerance pins AD parity."""
    net = aquakin.load_model("asm3")
    C = net.default_concentrations()
    ca = net.default_conditions().fields

    def k_sum(p):
        return jnp.sum(net._rate_kernel(C, p, ca, 0))

    def s_sum(p):
        return jnp.sum(_scalar_rates(net, C, p, ca, 0))

    p = net.default_parameters()
    gk = jax.grad(k_sum)(p)
    gs = jax.grad(s_sum)(p)
    assert jnp.allclose(gk, gs, rtol=1e-10, atol=1e-12)


def test_constant_exponent_keeps_derivative_finite():
    """A rate with a constant-exponent power whose base can reach zero -- e.g.
    ``(pH - pH_opt)**2`` -- must keep a finite derivative through the kernel.

    Regression for the subtle failure where the *forward* value is bit-identical
    but the *derivative* is NaN: a constant exponent gathered as a traced pool
    value activates the generic pow JVP's ``base**exp * log(base)`` term
    (``0 * log(0) = NaN`` at ``base == 0``). The kernel keeps constant exponents
    static (the ``powc`` kind) so it matches the scalar path's finite gradient.
    The whole-model forward bit-identicality test does NOT catch this -- only a
    derivative check does."""
    net = aquakin.load_model("wats_sewer_khalil_paper_balanced")
    C = net.default_concentrations()
    ca = net.default_conditions(1).fields

    # dr/dC through the kernel must be finite everywhere (the state-derived pH
    # makes the bad term contaminate every species column if it NaNs).
    Cc, p, ca2 = _prepared_inputs(net, C)
    J = jax.jacfwd(lambda CC: net._rate_kernel(CC, p, ca2, 0))(Cc)
    assert bool(jnp.all(jnp.isfinite(J)))

    # And it matches the scalar Jacobian. The *forward* rates are bit-identical,
    # but the derivative flows through different ops (gather / concatenate vs
    # slice / stack), so it agrees to machine precision, not bit-for-bit (whether
    # it lands bit-exact is platform-dependent -- a tight tolerance is correct).
    Js = jax.jacfwd(lambda CC: _scalar_rates(net, CC, p, ca2, 0))(Cc)
    assert jnp.allclose(J, Js, rtol=1e-9, atol=1e-12)


def test_unsupported_node_falls_back_to_scalar():
    """A model with an AST node type the kernel does not handle leaves
    ``_rate_kernel`` as ``None`` (scalar fallback), rather than raising at load.
    """
    from aquakin.core import nodes
    from aquakin.core.nodes import ASTNode

    class _UnknownNode(ASTNode):
        def compile(self, ctx):  # pragma: no cover - not evaluated here
            raise NotImplementedError

    net = aquakin.load_model("asm1")
    with pytest.raises(UnsupportedNode):
        build_vectorized_rates(
            [_UnknownNode()], ["r"], net.species_index, net.param_index
        )
