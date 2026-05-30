"""Smoke tests for the wats_sewer network and its state-derived pH.

These check that the (under-construction) WATS sewer network compiles, that pH
is solved from the state rather than supplied, and that the chemistry RHS is
finite and differentiable end-to-end -- the property the AD-based sensitivity
and calibration workflows depend on. Full stiff time-integration to steady
state requires the CSTR-cascade plant build and numerical hardening and is
covered separately.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture(scope="module")
def net():
    return aquakin.load_network("wats_sewer")


@pytest.fixture
def cond():
    return aquakin.SpatialConditions.uniform(1, T=20.0, A_V=56.7, X_BF=10.0)


def test_compiles_with_expected_shape(net):
    assert net.n_species == 18
    assert net.n_reactions == 46
    # pH is produced, not required from the user.
    assert "pH" not in net.conditions_required
    assert net.derived_fields == ["pH"]
    assert set(net.conditions_required) == {"T", "A_V", "X_BF"}
    # positivity limiter is enabled (matches the reference scheme).
    assert net.positivity_threshold == pytest.approx(1.0e-3)


@pytest.mark.parametrize(
    "variant",
    [
        "wats_sewer_v0",
        "wats_sewer_halforder",
        "wats_sewer_directsulfate",
        "wats_sewer_srbsubstrate",
        "wats_sewer_combined",
    ],
)
def test_structural_variants_compile(variant):
    """The model-structure-study variant networks load and compile."""
    import os
    ndir = os.path.join(os.path.dirname(aquakin.__file__), "networks")
    v = aquakin.load_network_from_file(os.path.join(ndir, variant + ".yaml"))
    assert v.name == variant
    assert v.n_species == 18
    assert v.n_reactions == 46
    cond = aquakin.SpatialConditions.uniform(1, T=20.0, A_V=56.7, X_BF=10.0)
    r = v.rates(v.default_concentrations(), v.default_parameters(), cond.fields, 0)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_parameter_priors_loaded(net):
    """Literature ranges / measured uncertainties load as Gaussian priors."""
    pr = net.parameter_priors
    # range [4, 8] -> N(midpoint, (hi-lo)/4)
    assert pr["mu_h"] == pytest.approx((6.0, 1.0))
    assert pr["k_no"] == pytest.approx((0.75, 0.125))
    # measured value +- reported std, used directly
    assert pr["k_sII_anox_f"] == pytest.approx((17.1, 2.3))
    assert pr["k_s0_anox_f"] == pytest.approx((2.2, 0.4))
    assert pr["k_s0_acid"] == pytest.approx((0.1, 0.01))
    # yields are stoichiometric / fixed -> no prior
    assert "y_h_anox" not in pr


def test_pH_is_state_derived_and_physical(net, cond):
    C0 = net.default_concentrations()
    derived = net.derived_condition_fn(C0, net.default_parameters(), cond.fields, 0)
    pH = float(derived["pH"])
    assert 4.0 < pH < 11.0


def test_rates_finite(net, cond):
    r = net.rates(net.default_concentrations(), net.default_parameters(), cond.fields, 0)
    assert r.shape == (46,)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_rhs_jacobian_wrt_params_is_finite(net, cond):
    C0 = net.default_concentrations()

    def rhs(p):
        return net.dCdt(C0, p, cond.fields, 0, stoich=net.compute_stoich(p))

    J = jax.jacobian(rhs)(net.default_parameters())
    assert J.shape == (18, len(net.parameters))
    assert np.all(np.isfinite(np.asarray(J)))


def test_batch_integrates_and_stays_nonnegative(net, cond):
    # With the positivity limiter, a closed-batch anaerobic solve must remain
    # finite and non-negative, and reduce sulfate into sulfide.
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9)
    C0 = net.default_concentrations()
    sol = reactor.solve(C0, net.default_parameters(), t_span=(0.0, 2.0))
    assert bool(jnp.all(jnp.isfinite(sol.C)))
    assert float(jnp.min(sol.C)) >= -1e-6
    # sulfate reduced, sulfide produced
    assert float(sol.C_named("S_SO4")[-1]) < float(C0[net.species_index["S_SO4"]])
    assert float(sol.C_named("sumS")[-1]) > float(C0[net.species_index["sumS"]])


def test_nitrate_dosing_lowers_sulfide(net, cond):
    # The paper's central result: nitrate availability suppresses net sulfide.
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9)
    p = net.default_parameters()
    C_no = net.default_concentrations()
    C_dosed = C_no.at[net.species_index["S_NO"]].set(20.0)
    sumS_no = float(reactor.solve(C_no, p, t_span=(0.0, 1.0)).C_named("sumS")[-1])
    sumS_dosed = float(reactor.solve(C_dosed, p, t_span=(0.0, 1.0)).C_named("sumS")[-1])
    assert sumS_dosed < sumS_no


def test_nitrate_enables_sulfide_oxidation_sensitivity(net, cond):
    # With nitrate present, final-step sulfide RHS must respond to the
    # nitrate-driven oxidation rate constant (the paper's key control lever).
    C0 = net.default_concentrations().at[net.species_index["S_NO"]].set(10.0)

    def sumS_rhs(p):
        return net.dCdt(C0, p, cond.fields, 0, stoich=net.compute_stoich(p))[
            net.species_index["sumS"]
        ]

    g = jax.grad(sumS_rhs)(net.default_parameters())
    assert float(g[net.param_index["k_sII_anox_f"]]) < 0.0  # oxidation removes sulfide


def test_dtmax_enables_finite_gradient_through_stiff_solve(net, cond):
    # Differentiating the full stiff solve with nitrate active requires capping
    # the step size: the L-stable solver otherwise steps over the fast
    # nitrate-sulfur reactions and the sensitivity of those modes goes
    # non-finite (in both AD directions). With dtmax set, the gradient is
    # finite and matches a finite-difference of the same (capped) solve.
    import diffrax

    C0 = net.default_concentrations().at[net.species_index["S_NO"]].set(20.0)
    p = net.default_parameters()
    ia = net.param_index["k_sII_anox_f"]
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9, dtmax=5.0e-4)

    def final_sumS(pp):
        return reactor.solve(C0, pp, t_span=(0.0, 0.1)).C_named("sumS")[-1]

    g = jax.grad(final_sumS)(p)
    assert np.all(np.isfinite(np.asarray(g)))

    # central finite difference of the same capped solve
    d = float(p[ia]) * 1.0e-3
    fd = (float(final_sumS(p.at[ia].add(d))) - float(final_sumS(p.at[ia].add(-d)))) / (2.0 * d)
    assert float(g[ia]) == pytest.approx(fd, rel=0.05, abs=1e-7)

    # forward mode must agree with reverse mode at the same cap
    fwd_reactor = aquakin.BatchReactor(
        net, cond, rtol=1e-6, atol=1e-9, dtmax=5.0e-4, adjoint=diffrax.ForwardMode()
    )
    tangent = jnp.zeros_like(p).at[ia].set(1.0)
    _, jvp = jax.jvp(
        lambda pp: fwd_reactor.solve(C0, pp, t_span=(0.0, 0.1)).C_named("sumS")[-1],
        (p,),
        (tangent,),
    )
    assert float(jvp) == pytest.approx(float(g[ia]), rel=1e-4, abs=1e-9)
