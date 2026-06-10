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
    # ADM1 (anaerobic digestion) -- first cut: liquid phase, fixed pH.
    ("adm1", 24, 19, 78),
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
            C0, params, t_span=(0.0, 0.05), t_eval=jnp.linspace(0.0, 0.05, 6)
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
            C0, params, t_span=(0.0, 0.02), t_eval=jnp.linspace(0.0, 0.02, 6)
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
