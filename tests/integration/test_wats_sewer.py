"""Smoke / integration tests for the original (reference-book) wats_sewer model.

This is the base WATS process matrix (Hvitved-Jacobsen, Vollertsen & Nielsen
2013, Tables 9.1-9.4): aerobic/anoxic/anaerobic heterotrophic carbon turnover
plus the sulfur cycle (sulfate reduction + chemical/biological sulfide oxidation
to sulfate), with state-derived (charge-balance) pH. It carries none of the
nitrate-dosing / methane / elemental-sulfur extensions of wats_sewer_extended.
These tests check it compiles with the expected shape, solves pH from the state,
and is finite and differentiable end-to-end.
"""

import os

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
    C0 = net.default_concentrations()
    derived = net.derived_condition_fn(C0, net.default_parameters(), cond.fields, 0)
    assert "pH" in derived
    assert bool(jnp.isfinite(derived["pH"]))


def test_rhs_jacobian_wrt_params_is_finite(net, cond):
    C0 = net.default_concentrations()

    def rhs(p):
        return net.dCdt(C0, p, cond.fields, 0, stoich=net.compute_stoich(p))

    J = jax.jacobian(rhs)(net.default_parameters())
    assert J.shape == (15, len(net.parameters))
    assert np.all(np.isfinite(np.asarray(J)))
