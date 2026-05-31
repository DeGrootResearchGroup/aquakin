"""Smoke / integration tests for the paper-faithful wats_sewer_khalil network.

These check that the compact, anoxic/anaerobic re-implementation of the
published Khalil et al. (2025) sewer nitrate-dosing model compiles with the
expected shape, that it carries no pH coupling (unlike the larger wats_sewer
network), that its chemistry RHS is finite and differentiable end-to-end, and
that its structural variants compile and remain AD-differentiable -- including
the half-order variants, whose square-root kinetics require a tighter
integrator-step cap for the reverse-mode adjoint to stay finite.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

_NDIR = os.path.join(os.path.dirname(aquakin.__file__), "networks")
_VARIANTS = [
    "wats_sewer_khalil_halforder",
    "wats_sewer_khalil_directsulfate",
    "wats_sewer_khalil_srbsubstrate",
    "wats_sewer_khalil_combined",
]


@pytest.fixture(scope="module")
def net():
    return aquakin.load_network("wats_sewer_khalil")


@pytest.fixture
def cond():
    return aquakin.SpatialConditions.uniform(1, A_V=56.7, X_BF=10.0)


def test_compiles_with_expected_shape(net):
    assert net.n_species == 11
    assert net.n_reactions == 13
    # Anoxic/anaerobic model: no pH coupling and no temperature condition.
    assert set(net.conditions_required) == {"A_V", "X_BF"}
    assert not net.derived_fields
    assert "pH" not in net.conditions_required
    assert net.positivity_threshold == pytest.approx(1.0e-3)


@pytest.mark.parametrize("variant", _VARIANTS)
def test_structural_variants_compile(variant):
    v = aquakin.load_network_from_file(os.path.join(_NDIR, variant + ".yaml"))
    assert v.name == variant
    assert v.n_species == 11
    assert v.n_reactions == 13
    cond = aquakin.SpatialConditions.uniform(1, A_V=56.7, X_BF=10.0)
    r = v.rates(v.default_concentrations(), v.default_parameters(), cond.fields, 0)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_parameter_priors_loaded(net):
    """The directly-measured sulfur-oxidation rates load as Gaussian priors."""
    pr = net.parameter_priors
    assert pr["k_sII_anox_f"] == pytest.approx((17.1, 2.3))
    assert pr["k_s0_anox_f"] == pytest.approx((2.2, 0.4))
    # the single heterotroph yield is fixed, not priored
    assert "y_h" not in pr


def test_rates_finite(net, cond):
    r = net.rates(net.default_concentrations(), net.default_parameters(), cond.fields, 0)
    assert r.shape == (13,)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_rhs_jacobian_wrt_params_is_finite(net, cond):
    C0 = net.default_concentrations()

    def rhs(p):
        return net.dCdt(C0, p, cond.fields, 0, stoich=net.compute_stoich(p))

    J = jax.jacobian(rhs)(net.default_parameters())
    assert J.shape == (11, len(net.parameters))
    assert np.all(np.isfinite(np.asarray(J)))


def test_batch_integrates_and_stays_nonnegative(net, cond):
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9, dtmax=5.0e-4)
    C0 = net.default_concentrations()
    sol = reactor.solve(C0, net.default_parameters(), t_span=(0.0, 5.0 / 24.0))
    assert bool(jnp.all(jnp.isfinite(sol.C)))
    assert float(jnp.min(sol.C)) >= -1e-6


def test_nitrate_dosing_lowers_sulfide(net, cond):
    """The model's purpose: nitrate availability suppresses net sulfide."""
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9, dtmax=5.0e-4)
    p = net.default_parameters()
    C_dosed = net.default_concentrations()  # default has nitrate dosed (S_NO=30)
    C_no = C_dosed.at[net.species_index["S_NO"]].set(0.0)
    sumS_dosed = float(reactor.solve(C_dosed, p, t_span=(0.0, 5.0 / 24.0)).C_named("sumS")[-1])
    sumS_no = float(reactor.solve(C_no, p, t_span=(0.0, 5.0 / 24.0)).C_named("sumS")[-1])
    assert sumS_dosed < sumS_no


def test_dtmax_enables_finite_gradient_through_stiff_solve(net, cond):
    """Reverse-mode gradient through the stiff solve is finite with a step cap
    and matches a finite difference of the same capped solve."""
    C0 = net.default_concentrations()
    p = net.default_parameters()
    ia = net.param_index["k_sII_anox_f"]
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9, dtmax=5.0e-4)

    def final_sumS(pp):
        return reactor.solve(C0, pp, t_span=(0.0, 0.1)).C_named("sumS")[-1]

    g = jax.grad(final_sumS)(p)
    assert np.all(np.isfinite(np.asarray(g)))
    d = float(p[ia]) * 1.0e-3
    fd = (float(final_sumS(p.at[ia].add(d))) - float(final_sumS(p.at[ia].add(-d)))) / (2.0 * d)
    assert float(g[ia]) == pytest.approx(fd, rel=0.05, abs=1e-7)


def test_halforder_variant_is_ad_differentiable_with_tighter_cap():
    """The square-root kinetics of the half-order variant need a tighter
    integrator-step cap for the reverse-mode adjoint to stay finite."""
    v = aquakin.load_network_from_file(
        os.path.join(_NDIR, "wats_sewer_khalil_halforder.yaml"))
    cond = aquakin.SpatialConditions.uniform(1, A_V=56.7, X_BF=10.0)
    C0 = v.default_concentrations()
    p = v.default_parameters()
    reactor = aquakin.BatchReactor(v, cond, rtol=1e-6, atol=1e-9, dtmax=1.0e-4)

    def final_so4(pp):
        return reactor.solve(C0, pp, t_span=(0.0, 0.1)).C_named("S_SO4")[-1]

    g = jax.grad(final_so4)(p)
    assert np.all(np.isfinite(np.asarray(g)))
