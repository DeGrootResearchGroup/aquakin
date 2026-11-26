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
from aquakin.plant.catalog import shipped_models

# (model, species that deplete together), each set to the values in
# ``_DEPLETED_VALUES``.
_CASES = [
    ("asm1", ["XS", "XB_H"]),
    ("asm1_ammonia_limitation", ["XS", "XB_H"]),
    ("asm2d", ["SF", "SA"]),
    ("asm2d", ["XPAO"]),
    ("asm2d_chemp", ["SF", "SA"]),
    ("asm2d_tud", ["SF", "SA"]),
    ("asm2d_tud", ["XPP", "XPAO"]),
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
    # Row 0 is the exact depletion, where the guarded denominator is 0. Row 1
    # is just past it (-1e-9, as a stiff solver can produce): the rate is
    # exactly 0 there too, not a small negative growth rate, because each model
    # either clips the state inside ``rates`` (``clip_negative_states``) or its
    # rate is otherwise zero at that state.
    np.testing.assert_array_equal(rates[:, cols], 0.0)


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


#: Every shipped model, by name.
_SHIPPED = [m["name"] for m in shipped_models()]


@pytest.mark.parametrize("name", _SHIPPED)
def test_rates_at_a_slightly_negative_state_equal_the_depleted_rates(name):
    """A species a stiff solver steps to -1e-9 gives exactly the rates of 0.

    Each species in turn is set to 0 and to -1e-9, with every other species
    present (default + 1, so the kinetics are active). Every rate, including any
    that depends on the species through a state-derived pH, must be identical
    at the two states: a Monod or ratio term evaluated at the negative value
    instead runs the reaction backwards (a small negative rate) or, near a
    pole, blows it up. Every shipped model sets ``clip_negative_states``,
    which makes the two states identical inside ``rates``.
    """
    net = aquakin.load_model(name)
    p = net.default_parameters()
    cf = net.default_conditions().fields
    base = net.default_concentrations() + 1.0
    n = base.shape[0]

    def rates_with(value):
        states = jnp.tile(base, (n, 1)).at[jnp.arange(n), jnp.arange(n)].set(value)
        return jax.vmap(lambda c: net.rates(c, p, cf, 0))(states)

    at_zero, below_zero = (np.asarray(a) for a in jax.jit(jax.vmap(rates_with))(_DEPLETED_VALUES))
    # Equal, with a NaN matching a NaN: a rate singular at the depleted state
    # itself is a property of its rate law, not of the clip, and is not what
    # this test is about.
    differs = (at_zero != below_zero) & ~(np.isnan(at_zero) & np.isnan(below_zero))
    species = list(net.species_index)
    moved = {
        species[i]: [net.reaction_names[j] for j in np.flatnonzero(differs[i])]
        for i in range(n)
        if np.any(differs[i])
    }
    assert not moved, f"{name}: rates move at -1e-9 (species -> reactions): {moved}"


def _aerobic_state(net):
    """Every species at 1.0: oxygen and substrate balanced against a small
    biomass, so the aerobic storage processes still run after a day. The
    model's default mixed liquor uses its oxygen up within minutes."""
    return jnp.ones(len(net.species))


def test_asm2d_tud_glycogen_formation_is_finite_with_no_stored_glycogen():
    """``asm2d_tud`` glycogen formation takes its finite limit at XGLY = 0.

    The published drive ``XPHA / XGLY * (f_GLY / (KfGLY + f_GLY)) * XPAO``,
    f_GLY = XGLY / XPAO, tends to ``XPHA / KfGLY`` as XGLY -> 0; written as that
    product it is ``inf * 0 = NaN`` there. Every other pool is at 1.0.
    """
    net = aquakin.load_model("asm2d_tud")
    p = net.default_parameters()
    cf = net.default_conditions().fields
    c = _aerobic_state(net).at[net.species_index["XGLY"]].set(0.0)

    rates = np.asarray(net.rates(c, p, cf, 0))
    assert np.all(np.isfinite(rates))
    grad = jax.grad(lambda c: jnp.sum(net.dCdt(c, p, cf, 0)))(c)
    assert np.all(np.isfinite(np.asarray(grad)))

    # The aerobic rate at the limit, from the closed form above.
    def value(name):
        return float(c[net.species_index[name]])

    def param(name):
        return float(p[net.param_index[name]])

    def monod(x, k):
        return x / (k + x)

    expected = (
        param("qGLY")
        * value("XPHA") / param("KfGLY")
        * monod(value("SO"), param("KO_HYD"))
        * monod(value("XPHA"), param("KPHA"))
    )
    i = net.reaction_names.index("Aerobic_glycogen_formation")
    assert rates[i] == pytest.approx(expected, rel=1e-12)
    assert rates[i] > 0.0

    # A batch solve started there stays finite.
    reactor = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(1, T=293.15))
    sol = reactor.solve(c, params=p, t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 5))
    assert np.all(np.isfinite(np.asarray(sol.C)))
    assert float(sol.C_named("XGLY")[-1]) > 0.0


#: The floor on the stored poly-P fraction f_PP = XPP / XPAO in the asm2d_tud
#: storage rates (g P / g COD).
_F_MIN = 1e-3

#: The PAO biomass of the poly-P storage checks (g COD/m3), so XPP = f * 200.
_XPAO = 200.0


def _polyp_state(net, f):
    """The default state + 1, with XPAO = 200 and XPP = f * XPAO."""
    c = net.default_concentrations() + 1.0
    c = c.at[net.species_index["XPAO"]].set(_XPAO)
    return c.at[net.species_index["XPP"]].set(f * _XPAO)


def _polyp_storage_by_hand(net, c, p, divisor):
    """The two poly-P storage rates in the published form, written out by hand.

    ``qPP [* etaNO_PAO] * XPAO / divisor * ... * Monod_XPP_max * XPAO``, with the
    factors in the YAML's order so the floating-point operations are the same.
    ``divisor`` is XPP for the published rate, or the floored XPP to check the
    cap; the max-ratio term always sees the true f_PP.
    """

    def value(name):
        return float(c[net.species_index[name]])

    def param(name):
        return float(p[net.param_index[name]])

    def monod(x, k):
        return x / (k + x)

    xpp, xpao = value("XPP"), value("XPAO")
    f = xpp / xpao
    fmax, kfpp = param("fPP_max"), param("KfPP")
    max_ratio = (fmax - f) / (kfpp + fmax - f)
    so, sno = value("SO"), value("SNO")
    anoxic = (
        param("qPP") * param("etaNO_PAO") * xpao / divisor
        * (param("KO_PAO") / (param("KO_PAO") + so))
        * monod(sno, param("KNO_HYD"))
        * monod(value("SPO"), param("KP_H"))
        * monod(value("XPHA"), param("KPHA"))
        * max_ratio
        * xpao
    )
    aerobic = (
        param("qPP") * xpao / divisor
        * monod(so, param("KO_HYD"))
        * monod(value("SPO"), param("KP_H"))
        * monod(value("XPHA"), param("KPHA"))
        * max_ratio
        * xpao
    )
    return {"Anoxic_storage_of_XPP": anoxic, "Aerobic_storage_of_XPP": aerobic}


def _storage_rates(net, c, p):
    rates = np.asarray(net.rates(c, p, net.default_conditions().fields, 0))
    return {r: float(rates[net.reaction_names.index(r)])
            for r in ("Anoxic_storage_of_XPP", "Aerobic_storage_of_XPP")}


@pytest.mark.parametrize("f", [1.0001e-3, 2e-3, 0.05, 0.3])
def test_asm2d_tud_polyp_storage_is_the_published_rate_above_the_floor(f):
    """Above f_min the floor is inactive: each poly-P storage rate is the
    published ``1/f_PP`` form, bit for bit (``==``, no tolerance)."""
    net = aquakin.load_model("asm2d_tud")
    p = net.default_parameters()
    c = _polyp_state(net, f)
    expected = _polyp_storage_by_hand(net, c, p, float(c[net.species_index["XPP"]]))
    got = _storage_rates(net, c, p)
    assert got == expected
    assert all(v > 0.0 for v in got.values())


@pytest.mark.parametrize("f", [0.0, 1e-6, 5e-4, 0.9999e-3])
def test_asm2d_tud_polyp_storage_is_capped_below_the_floor(f):
    """Below f_min the ``1/f_PP`` factor is ``1/f_min``: XPP divides as
    ``f_min * XPAO`` while the max-ratio term keeps the true f_PP. At XPP = 0
    the published rate is infinite; here it is that finite cap."""
    net = aquakin.load_model("asm2d_tud")
    p = net.default_parameters()
    c = _polyp_state(net, f)
    expected = _polyp_storage_by_hand(net, c, p, _F_MIN * _XPAO)
    got = _storage_rates(net, c, p)
    # The floored divisor is formed as XPP + XPAO * (f_min - XPP / XPAO), a few
    # roundings away from f_min * XPAO.
    for name, rate in got.items():
        assert rate == pytest.approx(expected[name], rel=1e-12, abs=0), name
        assert rate > 0.0


def test_asm2d_tud_polyp_storage_gradient_below_the_floor_matches_fd():
    """Below the floor the storage rate depends on XPP only through the
    max-ratio term; its AD derivative matches a central difference there, and
    every rate gradient is finite at XPP = 0 itself."""
    net = aquakin.load_model("asm2d_tud")
    p = net.default_parameters()
    cf = net.default_conditions().fields
    i = net.reaction_names.index("Aerobic_storage_of_XPP")
    j = net.species_index["XPP"]

    def rate(xpp):
        c = _polyp_state(net, 0.0).at[j].set(xpp)
        return net.rates(c, p, cf, 0)[i]

    x0, h = 0.5 * _F_MIN * _XPAO, 1e-4
    ad = float(jax.grad(rate)(x0))
    fd = (float(rate(x0 + h)) - float(rate(x0 - h))) / (2 * h)
    # The rate is smooth in XPP away from the floor: a central difference with
    # h = 1e-4 on XPP = 0.1 is accurate to ~1e-8 relative.
    assert ad == pytest.approx(fd, rel=1e-6)
    assert ad < 0.0

    c0 = _polyp_state(net, 0.0)
    jac = np.asarray(jax.jacfwd(lambda c: net.rates(c, p, cf, 0))(c0))
    assert np.all(np.isfinite(jac))
    grad = np.asarray(jax.grad(lambda c: jnp.sum(net.dCdt(c, p, cf, 0)))(c0))
    assert np.all(np.isfinite(grad))


def test_asm2d_tud_batch_solve_from_no_stored_polyp_stays_finite():
    """A batch solve started at XPP = 0 is finite and stores poly-P: XPP rises
    from 0 under the aerobic storage rate, which the floor keeps finite."""
    net = aquakin.load_model("asm2d_tud")
    p = net.default_parameters()
    c = _aerobic_state(net).at[net.species_index["XPP"]].set(0.0)
    reactor = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(1, T=293.15))
    sol = reactor.solve(c, params=p, t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 5))
    assert np.all(np.isfinite(np.asarray(sol.C)))
    xpp = np.asarray(sol.C_named("XPP")).ravel()
    assert xpp[0] == 0.0
    assert np.all(np.diff(xpp) > 0.0)
