"""COD / nitrogen conservation tests for the ASM1<->ADM1 interfaces."""

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1


@pytest.fixture
def nets():
    return aquakin.load_network("asm1"), aquakin.load_network("adm1")


def _asm_vec(asm1, **over):
    C = jnp.zeros((asm1.n_species,))
    for sp, v in over.items():
        C = C.at[asm1.species_index[sp]].set(float(v))
    return C


# COD- and N-carrying coefficients (gN/gCOD) consistent with the interface.
_FNAA = 0.098
_FXNI = 0.06
_FNBAC = 0.08
_FNXC = 0.0376
_CODEQUIV = 40.0 / 14.0


def _asm_cod(asm1, C):
    return float(sum(C[asm1.species_index[s]]
                     for s in ("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP")))


def _asm_n(asm1, C):
    g = lambda s: float(C[asm1.species_index[s]])
    return (g("SNH") + g("SND") + g("XND")
            + _FNBAC * (g("XB_H") + g("XB_A"))
            + _FXNI * (g("XI") + g("XP")))


def _adm_cod(adm1, y):
    return float(sum(y[adm1.species_index[s]] for s in
                     ("S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac",
                      "S_I", "X_c", "X_ch", "X_pr", "X_li", "X_I"))) * 1000.0


def _adm_n(adm1, y):
    gi = adm1.species_index
    S_IN = float(y[gi["S_IN"]]) * 14000.0
    n_aa = _FNAA * float(y[gi["S_aa"]] + y[gi["X_pr"]]) * 1000.0
    n_in = _FXNI * float(y[gi["S_I"]] + y[gi["X_I"]]) * 1000.0
    n_xc = _FNXC * float(y[gi["X_c"]]) * 1000.0
    return S_IN + n_aa + n_in + n_xc


def test_cod_conserved_anaerobic_feed(nets):
    """For an anaerobic feed (SO = SNO = 0) all COD is conserved across the
    interface (no electron-acceptor demand)."""
    asm1, adm1 = nets
    trans = ASM1toADM1(source_network=asm1, target_network=adm1)
    C = _asm_vec(asm1, SI=28.0, SS=3.0, XI=95.0, XS=360.0, XB_H=50.0, XB_A=0.1,
                 XP=0.7, SNH=35.0, SND=5.0, XND=16.0, SALK=7.0, SO=0.0, SNO=0.0)
    y = trans.translate(C)
    assert _adm_cod(adm1, y) == pytest.approx(_asm_cod(asm1, C), rel=1e-6)


def test_nitrogen_conserved(nets):
    asm1, adm1 = nets
    trans = ASM1toADM1(source_network=asm1, target_network=adm1)
    C = _asm_vec(asm1, SI=28.0, SS=3.0, XI=95.0, XS=360.0, XB_H=50.0, XB_A=0.1,
                 XP=0.7, SNH=35.0, SND=5.0, XND=16.0, SALK=7.0, SO=0.0, SNO=0.0)
    y = trans.translate(C)
    assert _adm_n(adm1, y) == pytest.approx(_asm_n(asm1, C), rel=1e-6)


def test_electron_acceptor_demand_removes_cod(nets):
    """With O2/NO3 present, exactly the electron-acceptor COD demand is removed."""
    asm1, adm1 = nets
    trans = ASM1toADM1(source_network=asm1, target_network=adm1)
    C = _asm_vec(asm1, SI=28.0, SS=40.0, XI=95.0, XS=360.0, XB_H=200.0, XB_A=10.0,
                 XP=0.7, SNH=35.0, SND=5.0, XND=16.0, SALK=7.0, SO=5.0, SNO=8.0)
    demand = 5.0 + _CODEQUIV * 8.0
    y = trans.translate(C)
    assert _adm_cod(adm1, y) == pytest.approx(_asm_cod(asm1, C) - demand, rel=1e-6)
    # Nitrogen is still conserved (biomass-N freed by the demand goes to S_IN).
    assert _adm_n(adm1, y) == pytest.approx(_asm_n(asm1, C), rel=1e-6)


def test_outputs_nonnegative_and_finite(nets):
    asm1, adm1 = nets
    trans = ASM1toADM1(source_network=asm1, target_network=adm1)
    C = _asm_vec(asm1, SI=28.0, SS=3.0, XI=95.0, XS=360.0, XB_H=50.0, XB_A=0.1,
                 XP=0.7, SNH=35.0, SND=5.0, XND=16.0, SALK=7.0)
    y = trans.translate(C)
    assert jnp.all(jnp.isfinite(y))
    # COD/IN/inert states must be non-negative.
    for sp in ("S_su", "S_aa", "S_I", "X_ch", "X_pr", "X_li", "X_I", "S_IN", "S_IC"):
        assert float(y[adm1.species_index[sp]]) >= -1e-12


def test_grad_through_interface(nets):
    asm1, adm1 = nets
    trans = ASM1toADM1(source_network=asm1, target_network=adm1)
    C = _asm_vec(asm1, SS=40.0, XS=360.0, XB_H=200.0, SNH=35.0, SND=5.0, XND=16.0,
                 SALK=7.0)
    g = jax.grad(lambda c: jnp.sum(trans.translate(c)))(C)
    assert jnp.all(jnp.isfinite(g))


# --- ADM1 -> ASM1 (adm2asm) -------------------------------------------------

def _adm_cod_full(adm1, C):
    """Total ADM1 COD including the soluble gases S_h2 / S_ch4."""
    return float(sum(C[adm1.species_index[s]] for s in (
        "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2",
        "S_ch4", "S_I", "X_c", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa",
        "X_c4", "X_pro", "X_ac", "X_h2", "X_I"))) * 1000.0


def _adm_n_full(adm1, C):
    gi = adm1.species_index
    g = lambda s: float(C[gi[s]])
    biomass = 1000.0 * (g("X_su") + g("X_aa") + g("X_fa") + g("X_c4")
                        + g("X_pro") + g("X_ac") + g("X_h2"))
    return (g("S_IN") * 14000.0 + _FNBAC * biomass
            + _FNAA * (g("S_aa") + g("X_pr")) * 1000.0
            + _FXNI * (g("S_I") + g("X_I")) * 1000.0
            + _FNXC * g("X_c") * 1000.0)


def _asm_n_out(asm1, y):
    g = lambda s: float(y[asm1.species_index[s]])
    return g("SNH") + g("SND") + g("XND") + _FXNI * (g("XI") + g("XP"))


def test_adm2asm_nitrogen_conserved(nets):
    asm1, adm1 = nets
    trans = ADM1toASM1(source_network=adm1, target_network=asm1)
    C = adm1.default_concentrations()  # realistic digester steady state
    y = trans.translate(C)
    assert _asm_n_out(asm1, y) == pytest.approx(_adm_n_full(adm1, C), rel=1e-6)


def test_adm2asm_cod_conserved_minus_stripped_gas(nets):
    """COD is conserved except for S_h2 + S_ch4, which strip to gas."""
    asm1, adm1 = nets
    trans = ADM1toASM1(source_network=adm1, target_network=asm1)
    C = adm1.default_concentrations()
    stripped = float(C[adm1.species_index["S_h2"]]
                     + C[adm1.species_index["S_ch4"]]) * 1000.0
    y = trans.translate(C)
    asm_cod = float(sum(y[asm1.species_index[s]]
                        for s in ("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP")))
    assert asm_cod == pytest.approx(_adm_cod_full(adm1, C) - stripped, rel=1e-6)


def test_adm2asm_outputs_finite_and_grad(nets):
    asm1, adm1 = nets
    trans = ADM1toASM1(source_network=adm1, target_network=asm1)
    C = adm1.default_concentrations()
    y = trans.translate(C)
    assert jnp.all(jnp.isfinite(y))
    # No nitrate/oxygen/biomass produced by the digester effluent.
    for sp in ("SO", "SNO", "XB_H", "XB_A"):
        assert float(y[asm1.species_index[sp]]) == pytest.approx(0.0, abs=1e-12)
    g = jax.grad(lambda c: jnp.sum(trans.translate(c)))(C)
    assert jnp.all(jnp.isfinite(g))
