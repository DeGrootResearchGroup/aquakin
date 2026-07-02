"""The asm3_2step_n2o model: asm3_2step with the two-pathway AOB N2O model of
Pocquet et al. (2016).

Checks that the AOB nitritation step is replaced by the electron-pathway
metabolism (NH4 -> NH2OH -> NO -> NO2) with N2O production, and that the model
reproduces Pocquet's headline trends: N2O rises with nitrite (free nitrous acid
stimulates the ND pathway) and peaks at intermediate dissolved oxygen (the
Haldane DO term). COD/N/charge closure is covered by test_asm_continuity.py.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture
def net():
    return aquakin.load_model("asm3_2step_n2o")


def _final_n2o(net, DO, no2, pH=7.5, t_end=0.3):
    cond = aquakin.SpatialConditions.uniform(T=293.15, pH=pH)
    r = aquakin.BatchReactor(net, cond)
    C0 = net.concentrations({"SO2": DO, "SNH4": 30.0, "SNO2": no2, "XAOB": 100.0,
                             "XNOB": 0.0, "XH": 0.0, "XSTO": 0.0, "SS": 0.0,
                             "SALK": 0.05})
    sol = r.solve(C0, params=net.default_parameters(), t_span=(0.0, t_end),
                  t_eval=jnp.linspace(0.0, t_end, 11))
    return np.asarray(sol.C_named("SN2O"))


def test_structure(net):
    # asm3_2step (15 sp, 19 rx) + 3 dissolved N-oxides, nitritation replaced by
    # the 5-process AOB chain: 18 species, 23 reactions; needs a pH condition.
    assert net.n_species == 18
    assert net.n_reactions == 23
    for s in ("SNH2OH", "SNO", "SN2O"):
        assert s in net.species, s
    assert "Aerobic_growth_of_XAOB_nitritation" not in net.reaction_names
    assert "pH" in net.conditions_required
    # Inherited NOB nitratation and heterotroph processes survive.
    assert "Aerobic_growth_of_XNOB_nitratation" in net.reaction_names
    assert "Hydrolysis" in net.reaction_names


def test_n2o_is_produced_during_nitritation(net):
    n2o = _final_n2o(net, DO=200.0, no2=20.0)
    assert n2o[-1] > 0.0
    assert np.all(np.isfinite(n2o))


def test_n2o_rises_with_nitrite(net):
    # Pocquet Fig 3 / Table 1: more nitrite -> more free nitrous acid -> more
    # N2O via the ND pathway (DO held high so the comparison is clean).
    low = _final_n2o(net, DO=200.0, no2=5.0)[-1]
    mid = _final_n2o(net, DO=200.0, no2=20.0)[-1]
    high = _final_n2o(net, DO=200.0, no2=80.0)[-1]
    assert low < mid < high


def test_n2o_peaks_at_intermediate_do(net):
    # The ND-pathway Haldane DO term: N2O production rises as DO falls to a
    # maximum, then decreases toward zero DO -- so intermediate DO yields more
    # N2O than either a high or a very low DO.
    high_do = _final_n2o(net, DO=300.0, no2=20.0)[-1]
    mid_do = _final_n2o(net, DO=50.0, no2=20.0)[-1]
    low_do = _final_n2o(net, DO=8.0, no2=20.0)[-1]
    assert mid_do > high_do
    assert mid_do > low_do


def test_free_nitrous_acid_depends_on_pH(net):
    # Free nitrous acid is the acid fraction of nitrite, so a lower pH raises it
    # and drives more ND-pathway N2O at the same nitrite level.
    n2o_low_pH = _final_n2o(net, DO=200.0, no2=20.0, pH=6.5)[-1]
    n2o_high_pH = _final_n2o(net, DO=200.0, no2=20.0, pH=8.0)[-1]
    assert n2o_low_pH > n2o_high_pH


def test_grad_through_solve_is_finite(net):
    cond = aquakin.SpatialConditions.uniform(T=293.15, pH=7.5)
    r = aquakin.BatchReactor(net, cond)
    C0 = net.concentrations({"SO2": 100.0, "SNH4": 30.0, "SNO2": 20.0,
                             "XAOB": 100.0})

    def loss(p):
        sol = r.solve(C0, params=p, t_span=(0.0, 0.1),
                      t_eval=jnp.linspace(0.0, 0.1, 4))
        return jnp.sum(sol.C_named("SN2O"))

    g = jax.grad(loss)(net.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    assert float(g[net.param_index["q_AOB_N2O_ND"]]) != 0.0
