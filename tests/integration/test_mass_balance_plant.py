"""Shipped composition tables + results-level plant mass balance (#150)."""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.balance import ComponentBalance, MassBalance
from aquakin.plant.plant import Plant
from aquakin.plant.cstr import Aeration, CSTRUnit
from aquakin.utils.balance import check_conservation
from aquakin.utils.composition import canonical_content, composition_table


# --- shipped composition tables ---------------------------------------------

def test_composition_table_reproduces_asm_cod_continuity():
    """The shipped table closes the Gujer COD balance on every ASM model
    (excluding ASM1 denitrification, whose electrons leave as untracked N2)."""
    for model in ("asm1", "asm2d", "asm2d_tud", "asm3", "asm3_biop"):
        net = aquakin.load_model(model)
        excl = {"anoxic_growth_heterotrophs"} if model == "asm1" else set()
        viol = [r for r, _, _ in check_conservation(
            net, composition_table(net), tol=1e-2, quantities=["COD"])
            if r not in excl]
        assert not viol, f"{model} COD: {viol}"


def test_adm1_composition_conserves_cod_and_nitrogen():
    """ADM1 conserves COD (bar the biogas gas-transfer/outflow reactions) and N
    under the shipped table -- the regression guard for the disintegration/decay
    inorganic-nitrogen terms (verified against the official BSM2 adm1_ODE)."""
    adm1 = aquakin.load_model("adm1")
    comp = composition_table(adm1)
    gas = {"transfer_h2", "transfer_ch4", "gas_outflow_h2", "gas_outflow_ch4"}
    cod = [r for r, _, _ in check_conservation(adm1, comp, tol=1e-3,
                                               quantities=["COD"]) if r not in gas]
    assert not cod, f"ADM1 COD imbalance: {cod}"
    # Nitrogen closes on EVERY reaction (no untracked N gas in ADM1).
    n = check_conservation(adm1, comp, tol=1e-4, quantities=["N"])
    assert not n, f"ADM1 N imbalance: {[(r, round(v, 5)) for r, _, v in n]}"


def test_canonical_content_unit_factors():
    """canonical_content converts to a single g-of-component basis: ASM g/m³
    (factor 1), ADM kg COD (×1000) and kmol N (×14000)."""
    asm1 = aquakin.load_model("asm1")
    cod = canonical_content(asm1, "COD")
    assert cod[asm1.species_index["SS"]] == pytest.approx(1.0)
    assert cod[asm1.species_index["SO"]] == pytest.approx(-1.0)   # acceptor
    adm1 = aquakin.load_model("adm1")
    assert canonical_content(adm1, "COD")[adm1.species_index["S_su"]] == pytest.approx(1000.0)
    assert canonical_content(adm1, "N")[adm1.species_index["S_IN"]] == pytest.approx(14000.0)


def test_lab_vs_electron_cod_convention():
    """electron_acceptor_cod=False (lab COD) zeroes nitrate/N2 COD; the default
    (electron equivalents) gives nitrate its -4.571 NH4-referenced COD."""
    asm1 = aquakin.load_model("asm1")
    j = asm1.species_index["SNO"]
    assert canonical_content(asm1, "COD")[j] == pytest.approx(-32.0 / 7.0, rel=1e-3)
    assert canonical_content(asm1, "COD", electron_acceptor_cod=False)[j] == 0.0


def test_composition_table_unknown_model_hints():
    class _Shim:               # a model with no shipped table
        name = "asmX"
    with pytest.raises(KeyError, match="No shipped composition table"):
        composition_table(_Shim())


# --- results-level balance ---------------------------------------------------

def _aerated_cstr_plant(net, *, Q=5000.0):
    """A single aerated ASM1 CSTR fed a constant influent (a minimal plant that
    exercises the inflow / outflow / aeration-O2 / inventory terms)."""
    plant = Plant("one")
    plant.add_unit(CSTRUnit(name="tank", model=net, volume=2000.0,
                            input_port_names=["inlet"], conditions={"T": 293.15},
                            aeration=Aeration(kla=240.0, do_sat=8.0)))
    plant.influent_endpoint = "tank.inlet"
    plant.effluent_endpoint = "tank.out"
    feed = net.influent({"SS": 200.0, "XS": 150.0, "SNH": 30.0, "SO": 0.0,
                         "SALK": 7.0}, Q=Q)
    plant.add_influent("feed", feed, to="tank.inlet")
    return plant


def test_mass_balance_closes_on_aerated_cstr():
    """At steady state a single aerated CSTR closes COD (oxygen removes it via
    aeration) and N (no gas at neutral, fully-aerobic conditions)."""
    net = aquakin.load_model("asm1")
    plant = _aerated_cstr_plant(net)
    ss = plant.run_to_steady_state(max_time=200.0)
    assert ss.converged
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 5), y0=ss.state)
    mb = plant.mass_balance(sol, components=("COD", "N"))

    assert isinstance(mb, MassBalance)
    assert isinstance(mb["COD"], ComponentBalance)
    assert mb.closed(rtol=1e-3)
    for q in ("COD", "N"):
        assert abs(mb[q].relative_imbalance) < 1e-3
    # The aeration oxygen transfer is the COD sink (positive g O2 removed).
    assert mb.gas_detail["aeration_O2"] > 0.0
    assert "closed" in mb.summary()


def test_mass_balance_default_ports_are_influents_and_dangling_outputs():
    net = aquakin.load_model("asm1")
    plant = _aerated_cstr_plant(net)
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.array([0.0, 1.0]))
    mb = plant.mass_balance(sol, components=("COD",))
    assert mb.influent_ports == ["feed"]
    assert mb.effluent_ports == plant.check().dangling_outputs   # ["tank.out"]


@pytest.mark.slow
def test_mass_balance_closes_on_bsm1_steady_state():
    """The full BSM1 plant closes COD and N to ~machine precision at steady
    state (single-model water line: no digester gas phase)."""
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start

    net = aquakin.load_model("asm1")
    plant = build_bsm1(model=net)
    feed = net.influent({"SS": 69.5, "XS": 202.32, "XB_H": 28.17, "SNH": 31.56,
                         "SND": 6.95, "XND": 10.59, "SI": 30.0, "XI": 51.2,
                         "SALK": 7.0}, Q=18446.0)
    plant.add_influent("feed", feed, to="inlet_mix.fresh")
    ss = plant.run_to_steady_state(y0=jnp.asarray(bsm1_warm_start(plant)),
                                   max_time=200.0)
    sol = plant.solve(t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 9),
                      y0=ss.state)
    mb = plant.mass_balance(sol, components=("COD", "N"))
    assert abs(mb["COD"].relative_imbalance) < 1e-4
    assert abs(mb["N"].relative_imbalance) < 1e-4


@pytest.mark.slow
def test_mass_balance_closes_on_bsm1_takacs_lumped_settler():
    """The BSM1 plant with the Takács reference settler closes COD and N at
    steady state. ``build_bsm1(use_takacs=True)`` defaults to the reference
    ``settler1dv4`` config (``composition_mode="lumped_tss"``, soluble holdup),
    whose head block is one TSS value per layer rather than per species -- the
    regression guard for the lumped-settler inventory in ``_unit_inventory``
    (which would otherwise fail to reshape the TSS head block)."""
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start

    net = aquakin.load_model("asm1")
    plant = build_bsm1(model=net, use_takacs=True)
    cl = plant.units["clarifier"]
    assert cl.composition_mode == "lumped_tss" and cl.soluble_holdup is True
    feed = net.influent({"SS": 69.5, "XS": 202.32, "XB_H": 28.17, "SNH": 31.56,
                         "SND": 6.95, "XND": 10.59, "SI": 30.0, "XI": 51.2,
                         "SALK": 7.0}, Q=18446.0)
    plant.add_influent("feed", feed, to="inlet_mix.fresh")
    ss = plant.run_to_steady_state(y0=jnp.asarray(bsm1_warm_start(plant)),
                                   max_time=200.0)
    sol = plant.solve(t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 9),
                      y0=ss.state)
    mb = plant.mass_balance(sol, components=("COD", "N"))
    assert abs(mb["COD"].relative_imbalance) < 1e-4
    assert abs(mb["N"].relative_imbalance) < 1e-4


@pytest.mark.slow
def test_mass_balance_closes_on_bsm2_steady_state():
    """The two-model BSM2 plant (ASM1 water line + ADM1 digester, biogas, the
    reject recycle) closes COD and N to <0.5% at steady state -- the digester
    biogas falls out of the reaction integral and the cross-model inventories
    sum on the canonical g basis."""
    from aquakin.plant.bsm import (build_bsm2, bsm2_warm_start,
                                   bsm2_constant_influent, bsm2_parameters)

    asm1 = aquakin.load_model("asm1")
    adm1 = aquakin.load_model("adm1")
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    params = bsm2_parameters(asm1, adm1)
    ss = plant.run_to_steady_state(params=params,
                                   y0=jnp.asarray(bsm2_warm_start(plant)),
                                   max_time=400.0)
    sol = plant.solve(t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 9),
                      params=params, y0=ss.state)
    mb = plant.mass_balance(sol, components=("COD", "N"), params=params)
    assert abs(mb["COD"].relative_imbalance) < 5e-3
    assert abs(mb["N"].relative_imbalance) < 5e-3
