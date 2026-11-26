"""Smoke / integration tests for the original (reference-book) wats_sewer model.

This is the base WATS process matrix (Hvitved-Jacobsen, Vollertsen & Nielsen
2013, Tables 9.1-9.4): aerobic/anoxic/anaerobic heterotrophic carbon turnover
plus the sulfur cycle (sulfate reduction + chemical/biological sulfide oxidation
to sulfate), with state-derived (charge-balance) pH. It carries no
nitrate-dosing, methane or elemental-sulfur extensions.
These tests check it compiles with the expected shape, solves pH from the state,
and is finite and differentiable end-to-end.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.core.ph_solver import solve_ph


@pytest.fixture(scope="module")
def net():
    return aquakin.load_model("wats_sewer")


@pytest.fixture
def cond():
    return aquakin.SpatialConditions.uniform(1, T=20.0, A_V=56.7, X_BF=10.0)


def test_compiles_with_expected_shape(net):
    assert net.name == "wats_sewer"
    assert net.n_reactions == 34
    assert net.n_species == 15
    # State-derived pH (the WATS acid/base chemistry), not user-supplied.
    assert net.derived_fields == ["pH"]
    assert "pH" not in net.conditions_required
    assert set(net.conditions_required) == {"T", "A_V", "X_BF"}


def test_is_base_wats_without_extensions(net):
    """No nitrate-dosing / methane / elemental-sulfur extension species."""
    for absent in ("X_S0", "S_CH4", "X_BA"):
        assert absent not in net.species_index


def test_biofilm_growth_is_half_order(net, cond):
    """The biofilm heterotrophic growth follows the book's 1/2-order kinetics in
    the electron acceptor (WATS Eqs 5.3 aerobic, 5.30 anoxic): the aerobic biofilm
    growth rate scales as S_O^0.5, i.e. quadrupling DO doubles the rate."""
    p = net.default_parameters()
    C0 = net.default_concentrations()
    io = net.species_index["S_O"]

    def rates(so):
        return np.asarray(net.rates(C0.at[io].set(so), p, cond.fields, 0))

    r1, r4 = rates(1.0), rates(4.0)
    ratio = np.divide(r4, r1, out=np.zeros_like(r4), where=np.abs(r1) > 1e-9)
    half_order = np.sum((np.abs(ratio - 2.0) < 0.02) & (np.abs(r1) > 1e-6))
    # the two aerobic biofilm growth reactions + biofilm sulfide oxidation
    assert half_order >= 3


def test_rates_finite(net, cond):
    r = net.rates(net.default_concentrations(), net.default_parameters(), cond.fields, 0)
    assert r.shape == (34,)
    assert bool(jnp.all(jnp.isfinite(r)))


def test_ph_is_solved_from_state(net, cond):
    """The derived pH is the charge-balance solve of the state's own totals
    (the ``speciation:`` block's species and molar masses), done by hand."""
    C0 = net.default_concentrations()
    si = net.species_index
    derived = net.derived_condition_fn(C0, net.default_parameters(), cond.fields, 0)
    expected = solve_ph(
        tot_carbonate=C0[si["S_CO2"]] / 12000.0,
        tot_acetate=C0[si["S_VFA"]] / 64000.0,
        tot_ammonia=C0[si["S_NH"]] / 14000.0,
        tot_phosphate=C0[si["S_PO4"]] / 31000.0,
        tot_sulfide=C0[si["sumS"]] / 32000.0,
        strong_anion_eq=2.0 * C0[si["S_SO4"]] / 32000.0 + C0[si["S_NO"]] / 14000.0,
        z_cation_eq=3.28e-3,
        T_kelvin=293.15,
    )
    assert float(derived["pH"]) == pytest.approx(float(expected), rel=1e-12)


def test_rhs_parameter_derivative_matches_fd(net, cond):
    """The chemistry RHS's derivative w.r.t. the parameters (a JVP along one
    random direction, scaled by the parameters so every component moves)
    matches a central difference of it (step 1e-5, error ~1e-10 relative).
    A non-finite column would poison the JVP, so this also covers finiteness."""
    model, conds = net, cond
    C0 = model.default_concentrations()
    p = model.default_parameters()

    def rhs(pp):
        return model.dCdt(C0, pp, conds.fields, 0, stoich=model.compute_stoich(pp))

    v = p * jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, p.shape))
    _, jv = jax.jvp(rhs, (p,), (v,))
    h = 1e-5
    fd = (rhs(p + h * v) - rhs(p - h * v)) / (2.0 * h)
    scale = float(jnp.max(jnp.abs(jv)))
    assert scale > 0.0
    np.testing.assert_allclose(np.asarray(jv), np.asarray(fd), rtol=1e-7, atol=1e-8 * scale)
