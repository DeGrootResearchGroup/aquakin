"""Shipped rate expressions stay finite when a ratio's denominator reaches 0.

Several shipped models carry a ratio whose denominator is a sum of species that
can all deplete to zero together: the ASM2d / TUD substrate-competition fraction
``SF / (SF + SA)``, the ASM1 hydrolysis ``XS * XB_H / (KX * XB_H + XS)``, the
ASM2d poly-P / PHA lysis ``XPP / XPAO * XPAO`` and the WATS maintenance split
``S_B / (S_B + S_VFA)``. A bare ``0 / 0`` makes every rate that uses the term
NaN, which then poisons ``dC/dt`` for every species the reaction touches. These
are written with ``safe_div`` so the rate takes its physical limit 0 there (and
the value is unchanged anywhere the denominator is non-zero).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.core.nodes import SafeDivideNode, SpeciesNode

# (model, species that deplete together), each set to the values in
# ``_DEPLETED_VALUES``.
_CASES = [
    ("asm1", ["XS", "XB_H"]),
    ("asm1_ammonia_limitation", ["XS", "XB_H"]),
    ("asm2d", ["SF", "SA"]),
    ("asm2d", ["XPAO"]),
    ("asm2d_chemp", ["SF", "SA"]),
    ("asm2d_tud", ["SF", "SA"]),
    ("wats_sewer", ["S_B", "S_VFA"]),
    ("wats_sewer_extended", ["S_B", "S_VFA"]),
    ("wats_sewer_khalil_thesis", ["S_B", "S_VFA"]),
]


#: The depleted value: exactly 0, and a tiny negative value just past
#: depletion. Both are evaluated in one compiled, vmapped call.
_DEPLETED_VALUES = jnp.asarray([0.0, -1e-9])


def _evaluate_depleted(net, depleted):
    """Rates, ``dC/dt`` and ``grad(sum(dC/dt))`` at each depleted value.

    Each array carries a leading axis over ``_DEPLETED_VALUES``.
    """
    p = net.default_parameters()
    cf = net.default_conditions().fields
    C = net.default_concentrations()
    idx = jnp.asarray([net.species_index[sp] for sp in depleted])

    def evaluate(value):
        c = C.at[idx].set(value)
        grad = jax.grad(lambda c: jnp.sum(net.dCdt(c, p, cf, 0)))(c)
        return net.rates(c, p, cf, 0), net.dCdt(c, p, cf, 0), grad

    return tuple(np.asarray(a) for a in jax.jit(jax.vmap(evaluate))(_DEPLETED_VALUES))


def _walk(node):
    yield node
    for child in node.children():
        yield from _walk(child)


def _guarded_by(net, depleted):
    """Reactions with a ``safe_div`` whose denominator holds only depleted species.

    That denominator is exactly 0 once they deplete, so the guard is what sets
    the rate there.
    """
    guarded = []
    for reaction, tree in zip(net.reaction_names, net.rate_asts, strict=True):
        for node in _walk(tree):
            if isinstance(node, SafeDivideNode):
                species = {n.name for n in _walk(node.denom) if isinstance(n, SpeciesNode)}
                if species and species <= set(depleted):
                    guarded.append(reaction)
                    break
    return guarded


@pytest.mark.parametrize(("name", "depleted"), _CASES)
def test_depleted_competition_terms_give_finite_rates(name, depleted):
    """Every rate, ``dC/dt`` and its gradient stay finite at depletion, and each
    reaction whose ratio is guarded takes the promised limit there: exactly 0."""
    net = aquakin.load_model(name)
    rates, dcdt, grad = _evaluate_depleted(net, depleted)
    assert np.all(np.isfinite(rates))
    assert np.all(np.isfinite(dcdt))
    # The gradient through the guarded ratio is finite too.
    assert np.all(np.isfinite(grad))

    guarded = _guarded_by(net, depleted)
    assert guarded, f"{name}: no safe_div over {depleted} -- the case is stale"
    cols = [net.reaction_names.index(r) for r in guarded]
    # Row 0 is the exact depletion, where the guarded denominator is 0. (At
    # -1e-9 the denominator is not 0 unless the model clips the state first,
    # which not every one does inside ``rates``.)
    np.testing.assert_array_equal(rates[0, cols], 0.0)


def test_safe_div_rewrite_leaves_nonzero_values_unchanged():
    # Away from the singularity the guarded expression equals the bare ratio.
    net = aquakin.load_model("wats_sewer")
    p = net.default_parameters()
    cf = net.default_conditions().fields
    C = net.default_concentrations() + 1.0  # every species present (incl. oxygen)
    i = net.reaction_names.index("maint_aer_SB_substrate")
    r = float(net.rates(C, p, cf, 0)[i])
    sb, svfa = float(C[net.species_index["S_B"]]), float(C[net.species_index["S_VFA"]])
    # The S_B and VFA maintenance reactions split one rate in that ratio.
    j = net.reaction_names.index("maint_aer_VFA_substrate")
    r_vfa = float(net.rates(C, p, cf, 0)[j])
    assert r / (r + r_vfa) == pytest.approx(sb / (sb + svfa), rel=1e-12)
