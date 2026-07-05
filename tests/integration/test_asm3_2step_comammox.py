"""The asm3_2step_comammox model: asm3_2step plus a complete-ammonia-oxidising
(comammox) organism, parameterised from Kits et al. (2017).

Checks the structure, the complete-nitrification behaviour (NH4 -> NO3 in one
organism), and the headline ecological result: comammox's high ammonia affinity
lets it out-compete the canonical AOB at LOW ammonium, while the AOB win at HIGH
ammonium (niche differentiation). COD/N/charge closure is in test_asm_continuity.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture
def net():
    return aquakin.load_model("asm3_2step_comammox")


def _rate(net, name, **C):
    p = net.default_parameters()
    Cv = net.concentrations(C)
    r = net.rates(Cv, p, {"T": jnp.array([303.15])}, 0)
    return float(r[net.reaction_names.index(name)])


def test_structure(net):
    assert net.n_species == 16          # asm3_2step (15) + XCMX
    assert net.n_reactions == 22        # asm3_2step (19) + 3 comammox processes
    assert "XCMX" in net.species
    assert "Complete_nitrification_by_XCMX" in net.reaction_names
    # The inherited two-step AOB/NOB are still present (comammox competes with them).
    assert "Aerobic_growth_of_XAOB_nitritation" in net.reaction_names
    assert "Aerobic_growth_of_XNOB_nitratation" in net.reaction_names


def test_complete_nitrification_nh4_to_no3(net):
    # Comammox alone (no AOB/NOB), aerobic: ammonium is taken straight to nitrate
    # with no nitrite accumulation (complete nitrification in one organism).
    cond = aquakin.SpatialConditions.uniform(T=303.15)
    r = aquakin.BatchReactor(net, cond)
    C0 = net.concentrations({"SO2": 300.0, "SNH4": 30.0, "SNO2": 0.0, "SNO3": 0.0,
                             "XCMX": 100.0, "XAOB": 0.0, "XNOB": 0.0, "XH": 0.0,
                             "XSTO": 0.0, "SS": 0.0, "SALK": 0.05})
    sol = r.solve(C0, params=net.default_parameters(), t_span=(0.0, 2.0),
                  t_eval=jnp.linspace(0.0, 2.0, 11))
    nh4 = np.asarray(sol.C_named("SNH4"))
    no2 = np.asarray(sol.C_named("SNO2"))
    no3 = np.asarray(sol.C_named("SNO3"))
    assert nh4[-1] < 1.0                          # ammonium oxidised
    assert no3[-1] > 25.0                          # to nitrate
    assert no2.max() < 1e-6                         # no nitrite (one-organism complete)


def test_comammox_wins_at_low_ammonium(net):
    # Oligotrophic niche (Kits 2017): comammox's high ammonia affinity makes its
    # per-biomass rate exceed the AOB's at low ammonium.
    rc = _rate(net, "Complete_nitrification_by_XCMX",
               SO2=5.0, SNH4=0.02, XCMX=1.0, XAOB=1.0, SALK=0.05)
    ra = _rate(net, "Aerobic_growth_of_XAOB_nitritation",
               SO2=5.0, SNH4=0.02, XCMX=1.0, XAOB=1.0, SALK=0.05)
    assert rc > ra


def test_aob_win_at_high_ammonium(net):
    # ... while the canonical AOB's higher maximum rate wins when ammonium is
    # plentiful (the substrate-affinity trade-off that separates the niches).
    rc = _rate(net, "Complete_nitrification_by_XCMX",
               SO2=5.0, SNH4=5.0, XCMX=1.0, XAOB=1.0, SALK=0.05)
    ra = _rate(net, "Aerobic_growth_of_XAOB_nitritation",
               SO2=5.0, SNH4=5.0, XCMX=1.0, XAOB=1.0, SALK=0.05)
    assert ra > rc


def test_grad_through_solve_is_finite(net):
    cond = aquakin.SpatialConditions.uniform(T=303.15)
    r = aquakin.BatchReactor(net, cond)
    C0 = net.concentrations({"SO2": 300.0, "SNH4": 30.0, "XCMX": 100.0})

    def loss(p):
        sol = r.solve(C0, params=p, t_span=(0.0, 0.5),
                      t_eval=jnp.linspace(0.0, 0.5, 4))
        return jnp.sum(sol.C_named("SNO3"))

    g = jax.grad(loss)(net.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    assert float(g[net.param_index["mu_CMX"]]) != 0.0
