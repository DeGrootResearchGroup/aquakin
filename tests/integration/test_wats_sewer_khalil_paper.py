"""Smoke / integration tests for the paper-faithful wats_sewer_khalil_paper network.

These check that the re-implementation of the published Khalil et al. (2025)
sewer nitrate-dosing model -- the full WATS model (carbon backbone plus the
complete sulfur cycle, including the dormant aerobic chemical/biological sulfide
oxidation) with the paper's additions and modifications -- compiles with the
expected shape, that it uses a fixed operating pH, that its O2-gated aerobic
backbone and aerobic sulfur-oxidation pathway are dormant under the air-sealed
anaerobic batch, that its chemistry RHS is finite and differentiable end-to-end,
and that its structural variants compile and remain AD-differentiable --
including the half-order variants, whose square-root kinetics require a tighter
integrator-step cap for the reverse-mode adjoint to stay finite.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

# Slow module: stiff WATS/Khalil sewer-network solves. Excluded from the fast
# PR gate; runs in the merge-to-main suite (see the ``slow`` marker).
pytestmark = pytest.mark.slow

_NDIR = os.path.join(os.path.dirname(aquakin.__file__), "networks")
_VARIANTS = [
    "wats_sewer_khalil_paper_halforder",
    "wats_sewer_khalil_paper_directsulfate",
    "wats_sewer_khalil_paper_srbsubstrate",
    "wats_sewer_khalil_paper_combined",
]


@pytest.fixture(scope="module")
def net():
    return aquakin.load_network("wats_sewer_khalil_paper")


@pytest.fixture
def cond():
    return aquakin.SpatialConditions.uniform(A_V=56.7, X_BF=10.0, pH=7.5)


def test_compiles_with_expected_shape(net):
    # Faithful published model: 18-species state vector, 27 reactions (23 active +
    # 4 dormant aerobic sulfur-oxidation), fixed operating pH, no temperature.
    # FeS precipitation and N tracking are extensions in the _balanced model only.
    assert net.n_species == 18
    assert net.n_reactions == 27
    assert set(net.conditions_required) == {"A_V", "X_BF", "pH"}
    assert not net.derived_fields
    assert net.positivity_threshold == pytest.approx(1.0e-3)


@pytest.mark.parametrize("variant", _VARIANTS)
def test_structural_variants_compile(variant):
    v = aquakin.load_network_from_file(os.path.join(_NDIR, variant + ".yaml"))
    assert v.name == variant
    assert v.n_species == 18
    assert v.n_reactions == 27
    cond = aquakin.SpatialConditions.uniform(A_V=56.7, X_BF=10.0, pH=7.5)
    r = v.rates(v.default_concentrations(), v.default_parameters(), cond.fields, 0)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_aerobic_backbone_dormant_in_anaerobic_batch(net, cond):
    """The WATS aerobic carbon backbone and the aerobic sulfur-oxidation pathway
    are O2-gated, so with no oxygen (S_O = 0) those reactions carry zero rate and
    S_O never goes negative -- the batch is anaerobic/anoxic and only the
    anoxic/anaerobic processes drive it. The four trailing reactions are the
    dormant aerobic chemical/biological sulfide and elemental-S oxidations."""
    r = net.rates(net.default_concentrations(), net.default_parameters(), cond.fields, 0)
    assert float(jnp.max(jnp.abs(r[-4:]))) == pytest.approx(0.0, abs=1e-30)
    reactor = aquakin.BatchReactor(net, cond, dtmax=5.0e-4)
    sol = reactor.solve(net.default_concentrations(),
                        t_span=(0.0, 0.2), t_eval=jnp.linspace(0.0, 0.2, 5),
                        params=net.default_parameters())
    assert float(jnp.max(jnp.abs(sol.C_named("S_O")))) == pytest.approx(0.0, abs=1e-12)
    assert bool(jnp.all(jnp.isfinite(sol.C)))


def test_parameter_priors_loaded(net):
    """The directly-measured sulfur-oxidation rates load as Gaussian priors."""
    pr = net.parameter_priors
    assert pr["k_sII_anox_f"] == pytest.approx((17.1, 2.3))
    assert pr["k_s0_anox_f"] == pytest.approx((2.2, 0.4))
    # the single heterotroph yield is fixed, not priored
    assert "y_h" not in pr


# Parameters whose literature priors are declared on the full-WATS parent
# (wats_sewer_extended) and must be propagated onto this derived model by
# _make_khalil_paper.py so that calibrating the carbon backbone stays physically
# bounded. Each is checked for presence; a few are pinned to their exact Gaussian
# (a range [lo, hi] loads as mean (lo+hi)/2, std (hi-lo)/4).
_PROPAGATED_PRIORS = {
    "k_h1": (5.5, 4.0),      # mean/std, widened to allow the thesis value (12)
    "mu_h": (6.0, 1.0),      # range [4, 8]
    "k_h2": (1.15, 3.0),     # explicit mean/std (widened so the fit may drift)
    "q_ferm": (2.0, 1.2),
    "k_ch4_acid": (4.44, 5.0),
}
_PROPAGATED_PRESENT = ("k_h1", "k_h2", "q_ferm", "k_ch4_acid",
                       "mu_h", "k_sw", "k_o", "eps", "q_m")
# Paper values deliberately outside the parent's literature range: the bulk
# nitrate saturation (K_NO = 2.0 vs parent 0.5-1.0) and the as-printed elemental-
# sulfur reduction rate (k_S2-,S0 = 15.5 vs parent mean 0.1). The value-guard
# must leave these UNCONSTRAINED rather than impose the parent constraint.
_GUARDED_UNCONSTRAINED = ("k_no", "k_s0_acid")


def test_literature_priors_propagated_from_parent(net):
    """Shared-parameter literature priors from the full-WATS parent are copied
    onto the paper model, and the copy is value-guarded so deliberately different
    paper values stay free."""
    pr = net.parameter_priors
    for name in _PROPAGATED_PRESENT:
        assert name in pr, f"expected propagated prior on {name!r}"
    for name, expected in _PROPAGATED_PRIORS.items():
        assert pr[name] == pytest.approx(expected), name
    for name in _GUARDED_UNCONSTRAINED:
        assert name not in pr, f"{name!r} should stay unconstrained (value-guard)"


@pytest.mark.parametrize("variant", _VARIANTS)
def test_structural_variants_inherit_propagated_priors(variant):
    """The structural variants are built from the paper model, so they inherit
    the same propagated priors and the same value-guarded exclusions."""
    v = aquakin.load_network_from_file(os.path.join(_NDIR, variant + ".yaml"))
    pr = v.parameter_priors
    for name in _PROPAGATED_PRESENT:
        assert name in pr, f"{variant}: missing propagated prior on {name!r}"
    for name in _GUARDED_UNCONSTRAINED:
        assert name not in pr, f"{variant}: {name!r} should stay unconstrained"


def test_rates_finite(net, cond):
    r = net.rates(net.default_concentrations(), net.default_parameters(), cond.fields, 0)
    assert r.shape == (27,)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_rhs_jacobian_wrt_params_is_finite(net, cond):
    C0 = net.default_concentrations()

    def rhs(p):
        return net.dCdt(C0, p, cond.fields, 0, stoich=net.compute_stoich(p))

    J = jax.jacobian(rhs)(net.default_parameters())
    assert J.shape == (18, len(net.parameters))
    assert np.all(np.isfinite(np.asarray(J)))


def test_batch_integrates_and_stays_nonnegative(net, cond):
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9, dtmax=5.0e-4)
    C0 = net.default_concentrations()
    sol = reactor.solve(C0, params=net.default_parameters(), t_span=(0.0, 5.0 / 24.0))
    assert bool(jnp.all(jnp.isfinite(sol.C)))
    assert float(jnp.min(sol.C)) >= -1e-6


def test_nitrate_dosing_lowers_sulfide(net, cond):
    """The model's purpose: nitrate availability suppresses net sulfide."""
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9, dtmax=5.0e-4)
    p = net.default_parameters()
    C_dosed = net.default_concentrations()  # default has nitrate dosed (S_NO=30)
    C_no = net.concentrations({"S_NO": 0.0})
    sumS_dosed = float(reactor.solve(C_dosed, params=p, t_span=(0.0, 5.0 / 24.0)).C_named("sumS")[-1])
    sumS_no = float(reactor.solve(C_no, params=p, t_span=(0.0, 5.0 / 24.0)).C_named("sumS")[-1])
    assert sumS_dosed < sumS_no


def test_dtmax_enables_finite_gradient_through_stiff_solve(net, cond):
    """Reverse-mode gradient through the stiff solve is finite with a step cap
    and matches a finite difference of the same capped solve."""
    C0 = net.default_concentrations()
    p = net.default_parameters()
    ia = net.param_index["k_sII_anox_f"]
    reactor = aquakin.BatchReactor(net, cond, rtol=1e-6, atol=1e-9, dtmax=5.0e-4)

    def final_sumS(pp):
        return reactor.solve(C0, params=pp, t_span=(0.0, 0.1)).C_named("sumS")[-1]

    g = jax.grad(final_sumS)(p)
    assert np.all(np.isfinite(np.asarray(g)))
    d = float(p[ia]) * 1.0e-3
    fd = (float(final_sumS(p.at[ia].add(d))) - float(final_sumS(p.at[ia].add(-d)))) / (2.0 * d)
    assert float(g[ia]) == pytest.approx(fd, rel=0.05, abs=1e-7)


def test_halforder_variant_is_ad_differentiable_with_tighter_cap():
    """The square-root kinetics of the half-order variant need a tighter
    integrator-step cap for the reverse-mode adjoint to stay finite."""
    v = aquakin.load_network_from_file(
        os.path.join(_NDIR, "wats_sewer_khalil_paper_halforder.yaml"))
    cond = aquakin.SpatialConditions.uniform(A_V=56.7, X_BF=10.0, pH=7.5)
    C0 = v.default_concentrations()
    p = v.default_parameters()
    reactor = aquakin.BatchReactor(v, cond, rtol=1e-6, atol=1e-9, dtmax=1.0e-4)

    def final_so4(pp):
        return reactor.solve(C0, params=pp, t_span=(0.0, 0.1)).C_named("S_SO4")[-1]

    g = jax.grad(final_so4)(p)
    assert np.all(np.isfinite(np.asarray(g)))
