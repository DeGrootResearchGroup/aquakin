"""The asm3_2step_anammox model: asm3_2step plus anammox (anaerobic ammonium
oxidation), after Strous et al. (1998, 1999).

Checks the anammox stoichiometry (the canonical Strous NH4:NO2:NO3 ~ 1:1.32:0.26
ratio), anaerobic deammonification (NH4 + NO2 -> N2), oxygen inhibition, the
temperature dependency, and that gradients flow. COD/N/charge closure is covered
by test_asm_continuity.py.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture
def net():
    return aquakin.load_model("asm3_2step_anammox")


def test_structure(net):
    # asm3_2step (15 species, 19 reactions) + XAMX biomass and its 3 processes.
    assert net.n_species == 16
    assert net.n_reactions == 22
    assert "XAMX" in net.species
    assert "Anammox_growth" in net.reaction_names
    # Inherited two-step nitrification survives.
    assert "Aerobic_growth_of_XAOB_nitritation" in net.reaction_names
    assert "Aerobic_growth_of_XNOB_nitratation" in net.reaction_names


def test_strous_stoichiometry_ratios(net):
    # The anammox growth row must reproduce the canonical Strous (1998) ratios
    # NH4 : NO2 : NO3 ~ 1 : 1.32 : 0.26, with ~1.02 N2 produced per NH4.
    p = net.default_parameters()
    S = np.asarray(net.compute_stoich(p))
    j = net.reaction_names.index("Anammox_growth")
    idx = net.species_index
    nh4 = S[j, idx["SNH4"]]
    no2 = S[j, idx["SNO2"]]
    no3 = S[j, idx["SNO3"]]
    n2 = S[j, idx["SN2"]]
    assert no2 / nh4 == pytest.approx(1.32, abs=0.03)        # NO2 : NH4
    assert -no3 / nh4 == pytest.approx(0.26, abs=0.02)        # NO3 : NH4
    assert -0.5 * n2 / nh4 == pytest.approx(1.02, abs=0.03)   # N2 (mol) per NH4


def test_anaerobic_deammonification(net):
    # No oxygen, ammonium + nitrite, anammox seeded: NH4 and NO2 are consumed and
    # N2 is produced (with a little nitrate). Nitrite is the limiting acceptor
    # (1.32 per NH4), so it runs out first.
    cond = aquakin.SpatialConditions.uniform(T=303.15)
    r = aquakin.BatchReactor(net, cond)
    C0 = net.concentrations({"SO2": 0.0, "SNH4": 30.0, "SNO2": 30.0, "SNO3": 0.0,
                             "SN2": 0.0, "XAMX": 200.0, "XAOB": 0.0, "XNOB": 0.0,
                             "XH": 0.0, "XSTO": 0.0, "SS": 0.0, "SALK": 0.05})
    sol = r.solve(C0, params=net.default_parameters(), t_span=(0.0, 2.0),
                  t_eval=jnp.linspace(0.0, 2.0, 11))
    nh4 = np.asarray(sol.C_named("SNH4"))
    no2 = np.asarray(sol.C_named("SNO2"))
    no3 = np.asarray(sol.C_named("SNO3"))
    n2 = np.asarray(sol.C_named("SN2"))
    assert no2[-1] < 0.5                       # nitrite (limiting) consumed
    assert nh4[-1] < nh4[0]                     # ammonium consumed
    assert no3[-1] > no3[0]                     # a little nitrate produced
    assert n2[-1] > 20.0                        # dinitrogen produced
    # Total N conserved (all dissolved): NH4 + NO2 + NO3 + N2.
    tot = nh4 + no2 + no3 + n2
    assert tot[-1] == pytest.approx(tot[0], abs=0.3)


def test_oxygen_inhibits_anammox(net):
    # Anammox carries a monod_inh O2 term: its rate collapses in the presence of
    # oxygen (the basis for needing low DO / anoxic conditions).
    p = net.default_parameters()
    C = net.concentrations({"SNH4": 30.0, "SNO2": 30.0, "XAMX": 200.0})
    j = net.reaction_names.index("Anammox_growth")
    anox = float(net.rates(C.at[net.species_index["SO2"]].set(0.0), p,
                           {"T": jnp.array([303.15])}, 0)[j])
    aer = float(net.rates(C.at[net.species_index["SO2"]].set(1.0), p,
                          {"T": jnp.array([303.15])}, 0)[j])
    assert anox > 0.0
    assert aer < 0.1 * anox                     # >90% inhibited at 1 gO2/m3


def test_temperature_speeds_anammox(net):
    # Anammox growth carries the source activation-energy temperature dependency
    # (Ea ~ 70 kJ/mol), so it is markedly faster warm (sidestream ~30 C) than at
    # 20 C.
    p = net.default_parameters()
    C = net.concentrations({"SO2": 0.0, "SNH4": 30.0, "SNO2": 30.0, "XAMX": 200.0})
    j = net.reaction_names.index("Anammox_growth")
    r20 = float(net.rates(C, p, {"T": jnp.array([293.15])}, 0)[j])
    r30 = float(net.rates(C, p, {"T": jnp.array([303.15])}, 0)[j])
    assert r30 > 2.0 * r20


def test_grad_through_solve_is_finite(net):
    cond = aquakin.SpatialConditions.uniform(T=303.15)
    r = aquakin.BatchReactor(net, cond)
    C0 = net.concentrations({"SO2": 0.0, "SNH4": 30.0, "SNO2": 30.0, "XAMX": 200.0})

    def loss(p):
        sol = r.solve(C0, params=p, t_span=(0.0, 0.5),
                      t_eval=jnp.linspace(0.0, 0.5, 4))
        return jnp.sum(sol.C_named("SN2"))

    g = jax.grad(loss)(net.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    assert float(g[net.param_index["mu_AMX"]]) != 0.0
