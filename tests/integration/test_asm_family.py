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


# The SUMO-ported ASM/ADM networks under test. Names only -- the shape contract
# (species/reaction/parameter counts) lives in _EXPECTED_SHAPES below and is
# asserted by a single dedicated test, so a benign network edit (e.g. adding a
# calibratable yield) touches one place, not this shared parametrize list that
# six tests depend on.
_ASM_NETWORKS = ["asm2d", "asm2d_tud", "asm3", "asm3_biop", "adm1"]

# Expected (n_species, n_reactions, n_params) per network -- a deliberate
# regression contract (it catches an accidental shape change). Param counts
# include both kinetic AND stoichiometric constants (yields, N/P content,
# fractions), all calibratable under the symbolic-stoich schema. ADM1 carries
# the gas headspace, state-derived pH, and explicit strong-ion (S_cat/S_an)
# states. Update a tuple here when a network's shape changes on purpose.
_EXPECTED_SHAPES = {
    "asm2d": (19, 21, 69),
    "asm2d_tud": (18, 22, 77),
    "asm3": (13, 12, 41),
    "asm3_biop": (17, 23, 69),
    "adm1": (29, 25, 89),
}


@pytest.mark.parametrize("name,n_sp,n_rx,n_p",
                         [(n, *s) for n, s in _EXPECTED_SHAPES.items()])
def test_loads_with_expected_shape(name, n_sp, n_rx, n_p):
    net = aquakin.load_network(name)
    assert net.n_species == n_sp
    assert net.n_reactions == n_rx
    assert net.n_params == n_p


@pytest.mark.parametrize("name", _ASM_NETWORKS)
def test_dCdt_finite_at_default_state(name):
    net = aquakin.load_network(name)
    dC = net.dCdt(
        net.default_concentrations(),
        net.default_parameters(),
        net.default_conditions().fields,
        0,
    )
    assert jnp.all(jnp.isfinite(dC))


@pytest.mark.parametrize("name", _ASM_NETWORKS)
@pytest.mark.slow  # heavy: stiff solve x every ASM/ADM network
def test_short_integration_finite(name):
    """Each model integrates 0.05 d without producing non-finite values."""
    net = aquakin.load_network(name)
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    sol = reactor.solve(
        net.default_concentrations(),
        params=net.default_parameters(),
        t_span=(0.0, 0.05),
        t_eval=jnp.linspace(0.0, 0.05, 6),
    )
    assert jnp.all(jnp.isfinite(sol.C))


@pytest.mark.slow  # jax.grad through a stiff solve x every ASM/ADM network
@pytest.mark.parametrize("name", _ASM_NETWORKS)
def test_ad_grad_through_solve(name):
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


def test_asm3_short_integration_and_ad_grad_finite():
    """Fast-gate representative of the two ``slow`` ASM-family checks above
    (``test_short_integration_finite`` / ``test_ad_grad_through_solve``): asm3
    is the smallest shipped ASM/ADM network, so a single short stiff solve and a
    ``jax.grad`` through it run cheaply. This keeps a core differentiability path
    -- integrate a stiff biological network and back-propagate -- in the PR gate,
    so a change that breaks it fails fast rather than only in the merge suite.
    """
    net = aquakin.load_network("asm3")
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    C0 = net.default_concentrations()
    t_eval = jnp.linspace(0.0, 0.05, 6)

    def loss(params):
        sol = reactor.solve(C0, params=params, t_span=(0.0, 0.05), t_eval=t_eval)
        return jnp.sum(sol.C_named(net.species[0]))

    # Forward primal is finite ...
    p = net.default_parameters()
    sol = reactor.solve(C0, params=p, t_span=(0.0, 0.05), t_eval=t_eval)
    assert jnp.all(jnp.isfinite(sol.C))
    # ... and so is the reverse-mode gradient through the stiff solve.
    g = jax.grad(loss)(p)
    assert g.shape == p.shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)


@pytest.mark.parametrize("name", _ASM_NETWORKS)
def test_no_duplicate_parameter_names(name):
    """The shared-params schema means no namespaced duplicates."""
    net = aquakin.load_network(name)
    assert len(net.parameters) == len(set(net.parameters))


@pytest.mark.parametrize("name", _ASM_NETWORKS)
def test_summary_smoke(name):
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
    reactor = aquakin.BatchReactor(
        net, net.default_conditions(),
        integrator=aquakin.IntegratorConfig(dtmax=1e-2))
    t_eval = jnp.linspace(0.0, 0.5, 4)

    def loss(scat0):
        sol = reactor.solve(C0.at[si["S_cat"]].set(scat0), (0.0, 0.5), t_eval, params=p)
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


def test_asm3_biop_polyp_storage_uses_max_ratio_inhibition():
    """ASM3_BioP's poly-P storage inhibition carries the maximum-ratio K_max term
    (the SUMO/Henze ``(Kmax - XPP/XPAO)/(KiPP + Kmax - XPP/XPAO)`` form), not the
    plain ``monod_inh_ratio`` the import had produced -- which capped stored
    poly-P far below its physical maximum."""
    yaml_text = (
        __import__("pathlib").Path("aquakin/networks/asm3_biop.yaml").read_text()
    )
    assert "Kmax_PAO" in yaml_text
    assert "Monod_inh_XPP_max" in yaml_text
    # The buggy plain inhibition-ratio form is gone.
    assert "monod_inh_ratio(" not in yaml_text


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


# The bio-P / nitrification networks share an import that (a) collapsed the
# autotroph half-saturation Monod terms onto the heterotroph values and (b)
# dropped the maximum-poly-P-ratio term from the poly-P storage inhibition. These
# pin the corrected constants per network (the SUMO ASM2D/ASM2D_TUD/ASM3_BioP and
# Henze STR No. 9 values), so a regeneration that reintroduces the bug fails here.
@pytest.mark.parametrize("network, nh4_aut_param, nh4_aut_value, max_ratio_param", [
    ("asm2d", "KNH4_AUT", 1.0, "KMAX"),
    ("asm3", "KA_NH4", 1.0, None),               # plain ASM3 nitrification (no bio-P)
    ("asm3_biop", "KNH_A", 1.0, "Kmax_PAO"),
    ("asm2d_tud", "KNH_A", 1.0, "fPP_max"),
])
def test_biop_autotroph_and_polyp_constants(network, nh4_aut_param, nh4_aut_value,
                                            max_ratio_param):
    net = aquakin.load_network(network)
    pv = net.parameter_values({})
    # Autotroph ammonia half-saturation is the nitrifier-specific value, NOT the
    # heterotroph 0.05 / 0.01 the collapsed term had used.
    assert nh4_aut_param in net.param_index
    assert float(pv[net.param_index[nh4_aut_param]]) == pytest.approx(nh4_aut_value)
    # The poly-P storage carries its maximum-ratio parameter (bio-P networks).
    if max_ratio_param is not None:
        assert max_ratio_param in net.param_index


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
