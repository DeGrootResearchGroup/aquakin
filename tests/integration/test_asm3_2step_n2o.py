"""The asm3_2step_n2o model: asm3_2step with the two-pathway AOB N2O model of
Pocquet et al. (2016).

Checks that the AOB nitritation step is replaced by the electron-pathway
metabolism (NH4 -> NH2OH -> NO -> NO2) with N2O production, and that the model
reproduces Pocquet's headline trends: N2O rises with nitrite (free nitrous acid
stimulates the ND pathway) and peaks at intermediate dissolved oxygen (the
Haldane DO term, maximum 1 at sqrt(K * KI)), and that AMO is Monod in free
ammonia. COD/N/charge closure is covered by test_asm_continuity.py.
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


def _rate(net, name, conc, pH=7.5, T=293.15):
    """One reaction rate at a single state (no integration)."""
    cond = aquakin.SpatialConditions.uniform(T=T, pH=pH)
    C = net.concentrations(conc)
    r = net.rates(C, net.default_parameters(), cond.fields, 0)
    return float(r[net.reaction_names.index(name)])


def _param(net, name):
    return float(net.default_parameters()[net.param_index[name]])


_ND = "AOB_N2O_production_ND_pathway"
_AMO = "AOB_NH3_oxidation_to_NH2OH"


def _f_do_nd(net, DO):
    # Recover the ND-pathway DO factor from the rate by dividing out the
    # hydroxylamine, free-nitrous-acid and biomass factors.
    conc = {"SO2": DO, "SNH4": 30.0, "SNO2": 20.0, "SNH2OH": 1.0, "XAOB": 100.0}
    pH, pKa = 7.5, _param(net, "pKa_HNO2")
    fna = 20.0 / (1.0 + 10.0 ** (pH - pKa))
    K_fna = _param(net, "K_HNO2_AOB")
    K_oh = _param(net, "K_NH2OH_AOB")
    pre = (_param(net, "q_AOB_N2O_ND") * 1.0 / (1.0 + K_oh) * fna / (fna + K_fna)
           * 100.0)
    return _rate(net, _ND, conc, pH=pH) / pre


def test_nd_do_term_peaks_at_one(net):
    # Pocquet (2016) Table 3: the Haldane-type ND DO factor reaches exactly 1 at
    # DO = sqrt(K_O2_AOB_ND * KI_O2_AOB), and is smaller on either side.
    do_star = np.sqrt(_param(net, "K_O2_AOB_ND") * _param(net, "KI_O2_AOB"))
    f_star = _f_do_nd(net, do_star)
    np.testing.assert_allclose(f_star, 1.0, rtol=1e-10)
    for DO in (0.05, 0.3, 1.0, 2.0, 8.0):
        assert _f_do_nd(net, DO) < f_star


def test_nd_n2o_peaks_at_intermediate_do(net):
    # N2O production via ND rises as DO falls to the peak, then decreases
    # toward zero DO.
    high = _f_do_nd(net, 5.0)
    mid = _f_do_nd(net, 0.6)
    low = _f_do_nd(net, 0.02)
    assert mid > high
    assert mid > low


def test_amo_is_monod_in_free_ammonia(net):
    # AMO uses free ammonia (NH3), not total ammonium: at fixed total SNH4 a
    # higher pH raises NH3 and the AMO rate, and the rate matches the
    # free-ammonia Monod term with pKa = 6344 / T.
    conc = {"SO2": 8.0, "SNH4": 5.0, "XAOB": 100.0}
    T = 293.15
    r_low = _rate(net, _AMO, conc, pH=7.0, T=T)
    r_high = _rate(net, _AMO, conc, pH=8.0, T=T)
    assert r_high > 2.0 * r_low
    pKa = 6344.0 / (T * np.log(10.0))
    for pH, r in ((7.0, r_low), (8.0, r_high)):
        s_nh3 = 5.0 / (1.0 + 10.0 ** (pKa - pH))
        K = _param(net, "K_NH3_AOB")
        expected = (_param(net, "q_AOB_AMO") * 8.0 / (8.0 + _param(net, "K_O2_AOB_1"))
                    * s_nh3 / (s_nh3 + K) * 100.0)
        np.testing.assert_allclose(r, expected, rtol=1e-10)


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
