"""ASM2d + saturation-index-driven chemical-P precipitation (`asm2d_chemp`).

Verifies the composition of the generalised precipitation engine with the full
ASM2d biology: ferric dosing precipitates phosphate by the aqueous saturation
index (computed from the free-ion activities at the operating pH) with the
bounded, differentiable supersaturation driver, replacing ASM2d's simple
empirical metal-hydroxide precipitation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture(scope="module")
def net():
    return aquakin.load_network("asm2d_chemp")


def test_inherits_asm2d_and_swaps_the_precipitation_model(net):
    # The simple ASM2d metal model is removed; the saturation-driven one added.
    assert "XMeOH" not in net.species and "XMeP" not in net.species
    assert "S_Fe3" in net.species and "X_FePO4" in net.species
    assert "Precipitation" not in net.reaction_names           # the simple kPRE law
    assert "FePO4_precipitation" in net.reaction_names
    # The biological core is inherited.
    assert "Aerobic_storage_of_XPP" in net.reaction_names
    # The saturation engine exposes its derived SI / driver fields.
    for f in ("SI_FePO4", "R_FePO4", "SI_FeOH3", "R_FeOH3"):
        assert f in net.derived_fields


def test_saturation_index_tracks_the_dose_and_pH(net):
    params = net.default_parameters()

    def fields(pH, fe):
        C = net.concentrations({"SPO4": 15.0, "S_Fe3": fe, "X_FePO4": 1e-4,
                                "X_FeOH3": 1e-4, "SALK": 7.0}, base="zero")
        cond = aquakin.OperatingConditions(T=293.15, pH=pH)
        d = net.derived_condition_fn(C, params, cond.fields, 0)
        return {k: float(v) for k, v in d.items()}

    # No metal -> massively undersaturated, driver pushes toward dissolution.
    f0 = fields(7.0, 0.0)
    assert f0["SI_FePO4"] < 0 and f0["R_FePO4"] < 0
    # Dosed metal -> supersaturated, driver near +1 (precipitation).
    fd = fields(7.0, 0.6)
    assert fd["SI_FePO4"] > 0 and fd["R_FePO4"] > 0.9
    # pH floor: as pH rises the hydroxide saturation grows faster than the
    # phosphate, diverting metal to Fe(OH)3 (worse chemical-P at higher pH).
    lo, hi = fields(6.5, 0.6), fields(8.0, 0.6)
    assert (hi["SI_FeOH3"] - hi["SI_FePO4"]) > (lo["SI_FeOH3"] - lo["SI_FePO4"])


# A realistic aerated mixed liquor (a near-zero state hits the inherited
# heterotroph-growth 0/0 in [SF]/([SA]+[SF])).
_MIXED_LIQUOR = {
    "SO2": 2.0, "SF": 2.0, "SA": 2.0, "SNH4": 5.0, "SNO3": 3.0, "SPO4": 15.0,
    "SI": 30.0, "SALK": 7.0, "XI": 800.0, "XS": 40.0, "XH": 1200.0,
    "XPAO": 300.0, "XPP": 80.0, "XPHA": 20.0, "XAUT": 80.0, "XTSS": 2800.0,
    "X_FePO4": 1e-4, "X_FeOH3": 1e-4,
}


@pytest.mark.slow
def test_ferric_precipitates_phosphate_conserving_P(net):
    cond = aquakin.OperatingConditions(T=293.15, pH=7.0)

    def final(fe):
        C0 = net.concentrations({**_MIXED_LIQUOR, "S_Fe3": fe}, base="zero")
        r = aquakin.BatchReactor(
            net, cond, integrator=aquakin.IntegratorConfig(dtmax=1e-3))
        sol = r.solve(C0, t_span=(0.0, 0.3), t_eval=jnp.array([0.3]))
        return (float(sol.C_named("SPO4")[-1]),
                float(sol.C_named("X_FePO4")[-1]),
                float(sol.C_named("S_Fe3")[-1]))

    p0, fp0, _ = final(0.0)
    pd, fpd, fe_left = final(0.6)
    # Ferric drives precipitation: FePO4 solid forms and the dosed metal is used.
    assert fpd > 0.5 and fe_left < 0.05
    # Chemical-P removes a substantial amount of phosphate from solution.
    assert pd < p0 - 10.0
    # The phosphate removed vs the no-dose case is accounted for by the P locked
    # into the FePO4 solid (P_MW = 31 g P / mol). The ~few-% residual is the
    # biology handling phosphorus differently at the lower SPO4 the dose creates
    # (the bio-P uptake responds to the dissolved phosphate), so this is a
    # roughly-conserving cross-run comparison, not an exact single-run balance.
    p_removed = p0 - pd
    p_in_solid = (fpd - fp0) * 31.0
    assert p_in_solid == pytest.approx(p_removed, rel=0.15)


def test_bounded_driver_is_differentiable(net):
    """The bounded supersaturation driver R = tanh(SI/(2v)*ln10) is smooth, so
    jax.grad of the precipitation driver w.r.t. the ferric dose is finite (the
    reason a dynamic solve of this network is differentiable -- the power-law
    driver of the ultra-insoluble metal phosphate would not be)."""
    params = net.default_parameters()
    cond = aquakin.OperatingConditions(T=293.15, pH=7.0)
    fe_idx = net.species_index["S_Fe3"]

    def driver(fe):
        C = net.concentrations({"SPO4": 15.0, "X_FePO4": 1e-4, "X_FeOH3": 1e-4,
                                "SALK": 7.0}, base="zero").at[fe_idx].set(fe)
        return net.derived_condition_fn(C, params, cond.fields, 0)["R_FePO4"]

    g = float(jax.grad(driver)(0.3))
    assert np.isfinite(g)
