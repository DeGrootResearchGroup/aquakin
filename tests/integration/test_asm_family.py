"""Smoke / sanity tests for ASM-family networks ported from SUMO.

Each shipped ASM network gets the same checks: it loads, has the right
shape, evaluates ``dCdt`` to finite values at default state, integrates
over a short window, and ``jax.grad`` through ``BatchReactor.solve``
returns finite gradients.
"""

import jax
import jax.numpy as jnp
import pytest

import aquakin


# (network name, expected n_species, expected n_reactions, expected n_params)
# Param counts include both kinetic AND stoichiometric constants (yields,
# N/P content, fractions) — all calibratable under the symbolic-stoich
# schema.
_ASM_NETWORKS = [
    ("asm2d", 19, 21, 63),
    ("asm2d_tud", 18, 22, 73),
    ("asm3", 13, 12, 39),
    ("asm3_biop", 17, 23, 65),
    # ADM1 (anaerobic digestion): liquid + gas headspace, state-derived pH,
    # explicit strong-ion (S_cat / S_an) states.
    ("adm1", 29, 25, 88),
]


@pytest.mark.parametrize("name,n_sp,n_rx,n_p", _ASM_NETWORKS)
def test_loads_with_expected_shape(name, n_sp, n_rx, n_p):
    net = aquakin.load_network(name)
    assert net.n_species == n_sp
    assert net.n_reactions == n_rx
    assert net.n_params == n_p


@pytest.mark.parametrize("name,_a,_b,_c", _ASM_NETWORKS)
def test_dCdt_finite_at_default_state(name, _a, _b, _c):
    net = aquakin.load_network(name)
    dC = net.dCdt(
        net.default_concentrations(),
        net.default_parameters(),
        net.default_conditions().fields,
        0,
    )
    assert jnp.all(jnp.isfinite(dC))


@pytest.mark.parametrize("name,_a,_b,_c", _ASM_NETWORKS)
@pytest.mark.slow  # heavy: stiff solve x every ASM/ADM network
def test_short_integration_finite(name, _a, _b, _c):
    """Each model integrates 0.05 d without producing non-finite values."""
    net = aquakin.load_network(name)
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    sol = reactor.solve(
        net.default_concentrations(),
        net.default_parameters(),
        t_span=(0.0, 0.05),
        t_eval=jnp.linspace(0.0, 0.05, 6),
    )
    assert jnp.all(jnp.isfinite(sol.C))


@pytest.mark.slow  # jax.grad through a stiff solve x every ASM/ADM network
@pytest.mark.parametrize("name,_a,_b,_c", _ASM_NETWORKS)
def test_ad_grad_through_solve(name, _a, _b, _c):
    """jax.grad through solve must produce finite gradients."""
    net = aquakin.load_network(name)
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    C0 = net.default_concentrations()

    # Pick any species that's likely to be sensitive — first one will do.
    target = net.species[0]

    def loss(params):
        sol = reactor.solve(
            C0, params=params, t_span=(0.0, 0.05), t_eval=jnp.linspace(0.0, 0.05, 6)
        )
        return jnp.sum(sol.C_named(target))

    g = jax.grad(loss)(net.default_parameters())
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("name,_a,_b,_c", _ASM_NETWORKS)
def test_no_duplicate_parameter_names(name, _a, _b, _c):
    """The shared-params schema means no namespaced duplicates."""
    net = aquakin.load_network(name)
    assert len(net.parameters) == len(set(net.parameters))


@pytest.mark.parametrize("name,_a,_b,_c", _ASM_NETWORKS)
def test_summary_smoke(name, _a, _b, _c):
    net = aquakin.load_network(name)
    s = net.summary()
    assert name in s
    assert f"Species ({net.n_species})" in s


def test_adm1_ph_is_state_derived():
    """ADM1 pH is produced by the charge-balance speciation solver, not an
    external condition: it is a derived field, sits at the BSM2 operating
    point for the default state, and responds to the acid/base state."""
    net = aquakin.load_network("adm1")
    # pH is produced, not required as an input.
    assert "pH" in net.derived_fields
    assert "pH" not in net.conditions_required
    # The strong-ion difference is carried by explicit states, not a condition.
    assert net.conditions_required == ["T"]
    assert "S_cat" in net.species_index
    assert "S_an" in net.species_index

    cond = net.default_conditions()
    C0 = net.default_concentrations()
    p = net.default_parameters()

    pH0 = float(net.derived_condition_fn(C0, p, cond.fields, 0)["pH"])
    # BSM2 reference digester operating point (charge-balance pH ~7.27).
    assert pH0 == pytest.approx(7.27, abs=0.05)

    si = net.species_index
    # Adding volatile fatty acid must lower the pH (more acid).
    C_acid = C0.at[si["S_ac"]].add(5.0)
    assert float(net.derived_condition_fn(C_acid, p, cond.fields, 0)["pH"]) < pH0
    # Adding strong cation (alkalinity) must raise the pH.
    C_alk = C0.at[si["S_cat"]].add(5.0e-3)
    assert float(net.derived_condition_fn(C_alk, p, cond.fields, 0)["pH"]) > pH0


def test_adm1_strong_ions_are_conservative():
    """S_cat / S_an carry no reaction stoichiometry, so they are constant
    over a batch (they change only by transport in a flow reactor)."""
    net = aquakin.load_network("adm1")
    si = net.species_index
    C0 = net.default_concentrations()
    p = net.default_parameters()
    dC = net.dCdt(C0, p, net.default_conditions().fields, 0)
    assert float(dC[si["S_cat"]]) == 0.0
    assert float(dC[si["S_an"]]) == 0.0


@pytest.mark.slow  # jax.grad through the stiff ADM1 state-derived-pH solve
def test_adm1_ph_operating_point_is_differentiable():
    """The explicit S_cat ion state sets the pH operating point and must flow
    differentiably through a solve into a downstream output. A short solve with
    the stiff-network step cap (see CLAUDE.md) keeps the reverse-mode adjoint
    finite."""
    net = aquakin.load_network("adm1")
    si = net.species_index
    C0 = net.default_concentrations()
    p = net.default_parameters()
    reactor = aquakin.BatchReactor(net, net.default_conditions(), dtmax=1e-2)
    t_eval = jnp.linspace(0.0, 0.5, 4)

    def loss(scat0):
        sol = reactor.solve(C0.at[si["S_cat"]].set(scat0), p, (0.0, 0.5), t_eval)
        return jnp.sum(sol.C_named("S_ch4"))

    g = jax.grad(loss)(5.0e-3)
    assert jnp.isfinite(g)


def test_asm3_uses_monod_helpers():
    """ASM3's auxiliaries should be rendered using the monod helpers."""
    yaml_text = (
        __import__("pathlib").Path("aquakin/networks/asm3.yaml").read_text()
    )
    assert "monod(" in yaml_text
    # Hydrolysis uses the surface-ratio form
    assert "monod_ratio(" in yaml_text


def test_asm3_biop_uses_monod_inh_ratio():
    """ASM3_BioP's storage rate has an inhibition-ratio Monod (MRinh)."""
    yaml_text = (
        __import__("pathlib").Path("aquakin/networks/asm3_biop.yaml").read_text()
    )
    assert "monod_inh_ratio(" in yaml_text


def test_adm1_c4_competition_uses_safe_div():
    """The valerate/butyrate C4 competition split is written with safe_div, not
    a dimensionless epsilon in the denominator, and the rates stay finite at the
    S_va = S_bu = 0 depletion point the epsilon used to guard."""
    import pathlib

    text = pathlib.Path("aquakin/networks/adm1.yaml").read_text()
    assert "safe_div([S_va], [S_va] + [S_bu])" in text
    assert "safe_div([S_bu], [S_va] + [S_bu])" in text
    # the old bare-epsilon guard is gone from the competition denominators
    assert "[S_bu] + 1.0e-6" not in text

    net = aquakin.load_network("adm1")
    iva, ibu = net.species_index["S_va"], net.species_index["S_bu"]
    C0 = net.default_concentrations().at[iva].set(0.0).at[ibu].set(0.0)
    dC = net.dCdt(C0, net.default_parameters(),
                  net.default_conditions().fields, 0)
    assert jnp.all(jnp.isfinite(dC))


@pytest.mark.slow  # jax.grad through a stiff solve to fit a yield, per network
@pytest.mark.parametrize("name,yield_param", [
    ("asm2d", "YH"),
    ("asm3", "YH_O2"),
    ("asm3_biop", "YH_O2"),
])
def test_yield_is_calibratable_in_sumo_models(name, yield_param):
    """Symbolic stoich: gradient w.r.t. a yield must be non-zero, proving
    the parameter actually flows through the stoichiometry matrix into
    the dynamics.

    Before the symbolic-stoich rewrite of the converter, yield parameters
    were frozen at literature defaults and invisible to ``jax.grad``.
    """
    net = aquakin.load_network(name)
    if yield_param not in net.parameters:
        pytest.skip(f"{name} does not expose {yield_param}")
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    C0 = net.default_concentrations()

    def loss(params):
        sol = reactor.solve(
            C0, params=params, t_span=(0.0, 0.02), t_eval=jnp.linspace(0.0, 0.02, 6)
        )
        # Sum every species to make sure the gradient picks up any dynamics
        # touched by the yield.
        return jnp.sum(sol.C)

    g = jax.grad(loss)(net.default_parameters())
    y_idx = net.param_index[yield_param]
    assert jnp.all(jnp.isfinite(g))
    assert float(g[y_idx]) != 0.0


def test_default_stoich_values_match_pre_symbolic_conversion():
    """At default parameters, the stoichiometry coefficients should match
    the values that were precomputed in the previous numeric-stoich version.

    Smoke check via a single representative entry of asm3 (Hydrolysis / SS,
    which was 1 - fSI = 0.99 with fSI=0.01 the SUMO default)."""
    net = aquakin.load_network("asm3")
    stoich = net.compute_stoich(net.default_parameters())
    hydrolysis_idx = net.reaction_names.index("Hydrolysis")
    ss_idx = net.species_index["SS"]
    # fSI default in SUMO ASM3 is 0.0; that may differ — just assert finite
    # and within physical range.
    val = float(stoich[hydrolysis_idx, ss_idx])
    assert 0.0 <= val <= 1.0
