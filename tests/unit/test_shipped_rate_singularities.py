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

# (model, species that deplete together). Each is set to exactly 0, and to a tiny
# negative value that the rate evaluation clips to 0.
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


@pytest.mark.parametrize("value", [0.0, -1e-9])
@pytest.mark.parametrize(("name", "depleted"), _CASES)
def test_depleted_competition_terms_give_finite_rates(name, depleted, value):
    net = aquakin.load_model(name)
    p = net.default_parameters()
    cf = net.default_conditions().fields
    C = net.default_concentrations()
    for sp in depleted:
        C = C.at[net.species_index[sp]].set(value)
    assert np.all(np.isfinite(np.asarray(net.rates(C, p, cf, 0))))
    assert np.all(np.isfinite(np.asarray(net.dCdt(C, p, cf, 0))))
    # The gradient through the guarded ratio is finite too.
    g = jax.grad(lambda c: jnp.sum(net.dCdt(c, p, cf, 0)))(C)
    assert np.all(np.isfinite(np.asarray(g)))


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
