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


def test_excess_acceptor_demand_over_conserves_cod_by_default(nets):
    """When the electron-acceptor (O2+NO3) COD demand exceeds the degradable COD,
    each cascade draw clamps at zero, so only the available COD is removed and the
    surplus demand is silently dropped -- COD is over-conserved (the documented
    pathological regime). The default interface does NOT raise."""
    asm1, adm1 = nets
    trans = ASM1toADM1(source_network=asm1, target_network=adm1)
    # demand = CODequiv*SNO = 285.7 gCOD, far above the 35 gCOD of degradable pools.
    C = _asm_vec(asm1, SI=28.0, XI=95.0, XP=0.7, SS=10.0, XS=20.0, XB_H=5.0,
                 SNH=35.0, SND=5.0, XND=16.0, SALK=7.0, SNO=100.0)
    demand = _CODEQUIV * 100.0
    degradable = 10.0 + 20.0 + 5.0                  # SS + XS + XB_H (XB_A = 0)
    y = trans.translate(C)
    # Only the available COD is removed -- not the full (larger) demand...
    assert _adm_cod(adm1, y) == pytest.approx(_asm_cod(asm1, C) - degradable, rel=1e-6)
    # ...so COD is over-conserved by exactly the dropped surplus demand.
    surplus = demand - degradable
    assert surplus > 0
    assert (_adm_cod(adm1, y) - (_asm_cod(asm1, C) - demand)
            == pytest.approx(surplus, rel=1e-6))


def test_strict_raises_on_excess_acceptor_demand(nets):
    """``ASM1toADM1(strict=True)`` raises when the electron-acceptor demand is not
    fully absorbed by the degradable COD, instead of silently over-conserving. A
    normal anoxic feed (demand <= degradable COD) still passes."""
    asm1, adm1 = nets
    strict = ASM1toADM1(source_network=asm1, target_network=adm1, strict=True)
    # In-regime anoxic feed: the demand is absorbed, so strict does not fire.
    C_ok = _asm_vec(asm1, SI=28.0, SS=40.0, XI=95.0, XS=360.0, XB_H=200.0, XB_A=10.0,
                    SNH=35.0, SND=5.0, XND=16.0, SALK=7.0, SO=5.0, SNO=8.0)
    assert jnp.all(jnp.isfinite(strict.translate(C_ok)))
    # Nitrate far exceeding the degradable COD: the surplus is dropped -> raise.
    C_bad = _asm_vec(asm1, SI=28.0, XI=95.0, XP=0.7, SS=10.0, XS=20.0, XB_H=5.0,
                     SNH=35.0, SND=5.0, XND=16.0, SALK=7.0, SNO=100.0)
    with pytest.raises(Exception, match="over-conserved"):
        strict.translate(C_bad)


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


# --- digester-pH feedback ---------------------------------------------------

def test_asm2adm_uses_digester_pH(nets):
    """asm2adm evaluates its inorganic-carbon charge balance at the supplied
    digester pH; the fixed pH_adm is only the standalone fallback."""
    asm1, adm1 = nets
    trans = ASM1toADM1(source_network=asm1, target_network=adm1)  # pH_adm = 7.0
    C = _asm_vec(asm1, SI=28.0, SS=3.0, XI=95.0, XS=360.0, XB_H=50.0, XB_A=0.1,
                 XP=0.7, SNH=35.0, SND=5.0, XND=16.0, SALK=7.0, SO=0.0, SNO=0.0)
    sic = adm1.species_index["S_IC"]
    # No pH supplied == evaluating at the fixed pH_adm.
    assert float(trans.translate(C)[sic]) == pytest.approx(
        float(trans.translate(C, digester_pH=7.0)[sic]), rel=1e-12)
    # A higher digester pH shifts S_IC (more bicarbonate -> lower total IC).
    sic_70 = float(trans.translate(C, digester_pH=7.0)[sic])
    sic_73 = float(trans.translate(C, digester_pH=7.3)[sic])
    assert sic_73 < sic_70
    assert abs(sic_73 - sic_70) / sic_70 > 0.02
    # COD and N are pH-independent, so still conserved at the shifted pH.
    y = trans.translate(C, digester_pH=7.3)
    assert _adm_cod(adm1, y) == pytest.approx(_asm_cod(asm1, C), rel=1e-6)
    assert _adm_n(adm1, y) == pytest.approx(_asm_n(asm1, C), rel=1e-6)


def test_adm2asm_uses_digester_pH(nets):
    """adm2asm evaluates its alkalinity charge balance at the supplied digester
    pH; conservation is unaffected and the gradient stays finite."""
    asm1, adm1 = nets
    trans = ADM1toASM1(source_network=adm1, target_network=asm1)
    C = adm1.default_concentrations()
    salk = asm1.species_index["SALK"]
    assert float(trans.translate(C)[salk]) == pytest.approx(
        float(trans.translate(C, digester_pH=7.0)[salk]), rel=1e-12)
    assert float(trans.translate(C, digester_pH=7.3)[salk]) != pytest.approx(
        float(trans.translate(C, digester_pH=7.0)[salk]), rel=1e-6)
    y = trans.translate(C, digester_pH=7.3)
    assert _asm_n_out(asm1, y) == pytest.approx(_adm_n_full(adm1, C), rel=1e-6)
    g = jax.grad(lambda c: jnp.sum(trans.translate(c, digester_pH=7.3)))(C)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.slow
def test_plant_feeds_digester_pH_to_interface():
    """In the assembled BSM2 plant the ASM->ADM interface is fed the digester's
    state-derived pH, so the digester inlet the plant resolves uses that pH
    (~7.3), not the interface's fixed pH_adm of 7.0."""
    from aquakin.plant.bsm import bsm2_warm_start
    from aquakin.plant.bsm.bsm2 import (
        BSM2_CONSTANT_INFLUENT_T, build_bsm2, bsm2_asm1_network,
        bsm2_constant_influent, bsm2_parameters)

    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed",
                       bsm2_constant_influent(asm1, T=BSM2_CONSTANT_INFLUENT_T))
    params = bsm2_parameters(asm1, adm1)
    y0 = bsm2_warm_start(plant)
    plant.derivative(y0, params=params)   # build the state / parameter layouts
    states = plant.states_by_unit(y0)
    di = adm1.species_index

    dig_pH = float(plant.units["digester"].operating_pH(
        states["digester"], plant._params_for_unit("digester", params)))
    assert 7.0 < dig_pH < 7.6     # the digester operates near pH 7.3, not 7.0

    # The digester inlet the plant actually resolves.
    all_outputs, streams = plant._resolve_streams(0.0, states, params)
    inlet = plant._collect_inputs("digester", all_outputs, streams, states, params)
    inlet_C = inlet[plant.units["digester"].input_ports[0]].C
    sludge = all_outputs[("sludge_mix", "out")]
    iface = ASM1toADM1(source_network=asm1, target_network=adm1)
    sic_plant = float(inlet_C[di["S_IC"]])
    sic_fb = float(iface.translate(sludge.C, digester_pH=dig_pH)[di["S_IC"]])
    sic_fixed = float(iface.translate(sludge.C)[di["S_IC"]])
    # The plant feeds the digester pH (matches the fed-back translation) and so
    # differs materially from the fixed-pH_adm result.
    assert sic_plant == pytest.approx(sic_fb, rel=1e-9)
    assert abs(sic_plant - sic_fixed) / sic_fixed > 0.03


def test_needs_src_ph_influent_edge_warns():
    """Defensive: a ``needs_src_pH`` translator wired to an EXTERNAL influent has
    no source unit (so no state-derived pH), and the pH feedback would silently
    fall back to the fixed ``pH_adm``. The connection-index build warns rather
    than failing silently. Not reachable in shipped plants (the digester is
    always a real source unit), so this exercises the defensive guard directly."""
    from aquakin.plant.plant import Plant, Connection

    class _SrcPHTranslator:
        needs_src_pH = True

        def translate(self, C, digester_pH=None):
            return C

    plant = Plant("defensive")
    plant._unit_order = ["sink"]
    plant.connections = [
        Connection(from_unit=None, from_port="feed", to_unit="sink",
                   to_port="in", translator=_SrcPHTranslator()),
    ]
    with pytest.warns(UserWarning, match="needs_src_pH"):
        plant._build_connection_index()
