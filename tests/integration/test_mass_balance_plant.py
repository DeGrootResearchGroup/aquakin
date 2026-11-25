"""Shipped composition tables + results-level plant mass balance (#150)."""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.balance import ComponentBalance, MassBalance
from aquakin.plant.cstr import Aeration, CSTRUnit
from aquakin.plant.plant import Plant
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

def _aerated_cstr_plant(net, *, Q=5000.0, T=293.15, aeration=None):
    """A single aerated ASM1 CSTR fed a constant influent (a minimal plant that
    exercises the inflow / outflow / aeration-O2 / inventory terms)."""
    plant = Plant("one", time_unit="d")
    plant.add_unit(CSTRUnit(name="tank", model=net, volume=2000.0,
                            input_port_names=["inlet"], conditions={"T": T},
                            aeration=aeration or Aeration(kla=240.0, do_sat=8.0)))
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


def test_mass_balance_counts_the_temperature_corrected_aeration():
    """With the aeration temperature correction on, away from its reference
    temperature, the balance counts the oxygen the solve transferred (the
    corrected saturation and kLa), so COD still closes at steady state."""
    net = aquakin.load_model("asm1")
    aeration = Aeration(kla=240.0, do_sat=8.0, temperature_correction=True)
    plant = _aerated_cstr_plant(net, T=283.15, aeration=aeration)
    ss = plant.run_to_steady_state(max_time=200.0)
    assert ss.converged
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 5), y0=ss.state)
    mb = plant.mass_balance(sol, components=("COD", "N"))
    assert abs(mb["COD"].relative_imbalance) < 1e-3, mb.summary()


def test_mass_balance_closes_with_time_unit_override():
    """solve(time_unit=...) reports t in hours, but the flows are per day: the
    balance still closes and its gram totals match the native-unit solve."""
    net = aquakin.load_model("asm1")
    plant = _aerated_cstr_plant(net)
    ss = plant.run_to_steady_state(max_time=200.0)
    assert ss.converged
    # Perturb the start so the accumulation term is non-trivial.
    y0 = ss.state * 1.2
    sol_d = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 25), y0=y0)
    sol_h = plant.solve(t_span=(0.0, 24.0), t_eval=jnp.linspace(0.0, 24.0, 25), y0=y0,
                        time_unit="h")
    mb_d = plant.mass_balance(sol_d, components=("COD", "N"))
    mb_h = plant.mass_balance(sol_h, components=("COD", "N"))
    assert mb_h.window == pytest.approx((0.0, 24.0), rel=1e-6, abs=0)
    assert mb_h.closed(rtol=1e-2)
    for q in ("COD", "N"):
        assert mb_h[q].inflow == pytest.approx(mb_d[q].inflow, rel=1e-6)
        assert mb_h[q].outflow == pytest.approx(mb_d[q].outflow, rel=1e-5)
        assert mb_h[q].gas == pytest.approx(mb_d[q].gas, rel=1e-5)


def test_stateful_units_own_their_component_inventory():
    """The #505 inversion: a unit whose state layout is non-trivial (the layered
    Takács settler, the ADM1 digester with its gas headspace) declares its own
    ``component_inventory`` contract, so the mass balance dispatches on a method
    instead of reaching into private layout attributes. A plain
    concentration-vector unit does *not* implement the contract -- it falls
    through to the generic ``volume·C`` path in ``_unit_inventory``.
    """
    from aquakin.plant.digester import ADM1DigesterUnit
    from aquakin.plant.takacs import TakacsClarifier

    # The two special-layout units expose the contract; a plain CSTR does not.
    assert hasattr(TakacsClarifier, "component_inventory")
    assert hasattr(ADM1DigesterUnit, "component_inventory")
    assert hasattr(ADM1DigesterUnit, "_state_volume_vector")
    assert not hasattr(CSTRUnit, "component_inventory")

    asm1 = aquakin.load_model("asm1")
    tak = TakacsClarifier(name="t", model=asm1, area=1000.0, height=4.0, overflow_Q=1.0)
    # per_species blanket: unit concentration of the first particulate species in
    # every layer, unit content on that species -> inventory = layer_vol·n_layers.
    layer_vol = tak.area * tak.height / tak.n_layers
    content_vec = np.zeros(asm1.n_species)
    content_vec[tak._part_indices[0]] = 1.0
    prof = np.zeros((tak.n_layers, tak._n_part))
    prof[:, 0] = 1.0
    state = np.zeros(tak.state_size)
    state[: tak._part_block_size] = prof.reshape(-1)
    inv = tak.component_inventory(state, {"X": content_vec}, asm1.default_parameters())
    assert inv["X"] == pytest.approx(layer_vol * tak.n_layers)

    # Digester: the three gas-headspace states live in V_gas, the liquid states
    # in the liquid volume -- the layout the balance no longer hard-codes.
    adm1 = aquakin.load_model("adm1")
    dig = ADM1DigesterUnit(name="d", model=adm1, volume=3400.0)
    params = adm1.default_parameters()
    vol = dig._state_volume_vector(params)
    v_gas = float(params[adm1.param_index["V_gas"]])
    for sp in ("S_gas_h2", "S_gas_ch4", "S_gas_co2"):
        assert vol[adm1.species_index[sp]] == pytest.approx(v_gas)
    liquid_states = [i for i in range(adm1.n_species) if vol[i] != v_gas]
    assert liquid_states  # there are liquid states
    assert all(vol[i] == pytest.approx(3400.0) for i in liquid_states)


def _digester(adm1):
    from aquakin.plant.digester import ADM1DigesterUnit

    return ADM1DigesterUnit(name="digester", model=adm1, volume=3400.0,
                            conditions={"T": 308.15}, input_port_names=["feed"])


def test_digester_reacts_for_the_balance_with_cod_gas_only():
    """The digester declares ReactionBalanceUnit (not OxygenTransferUnit) with
    COD as its only gas component, and its reaction volumes put the three gas
    states at ``V_gas`` and the liquid states at the liquid volume."""
    from aquakin.plant.units import OxygenTransferUnit, ReactionBalanceUnit

    adm1 = aquakin.load_model("adm1")
    dig = _digester(adm1)
    assert isinstance(dig, ReactionBalanceUnit)
    assert not isinstance(dig, OxygenTransferUnit)
    assert dig.reaction_gas_components == frozenset({"COD"})
    params = adm1.default_parameters()
    vols = dig.reaction_volumes(dig.initial_state(), params)
    v_gas = float(params[adm1.param_index["V_gas"]])
    for sp in ("S_gas_h2", "S_gas_ch4", "S_gas_co2"):
        assert vols[adm1.species_index[sp]] == v_gas
    assert vols[adm1.species_index["S_su"]] == 3400.0


@pytest.mark.slow  # the ADM1 compile and the 21-row balance take ~20 s
def test_digester_exports_biogas_cod_and_no_nitrogen_gas():
    """In the mass balance the digester's reaction-integral COD export equals
    the biogas COD computed independently from the headspace
    (``digester_gas``), and its N gas is exactly zero."""
    adm1 = aquakin.load_model("adm1")
    dig = _digester(adm1)
    p = Plant("dig", time_unit="d")
    p.add_unit(dig)
    p.add_influent("feed", adm1.influent(Q=170.0), to="digester.feed")
    sol = p.solve(t_span=(0.0, 0.5), t_eval=np.linspace(0.0, 0.5, 21))
    mb = p.mass_balance(sol)
    assert mb["N"].gas == 0.0
    assert mb.gas_detail["aeration_O2"] == 0.0
    # Two routes to the same export: they differ only by the trapezoid sampling.
    assert mb.gas_detail["reaction_COD"] == pytest.approx(
        mb.gas_detail["biogas_COD"], rel=1e-4
    )


def test_unit_capabilities_for_the_gas_term():
    """Every kinetic unit declares ReactionBalanceUnit and every aerating one
    OxygenTransferUnit (the SBR included); a non-reactive unit declares neither,
    so the balance needs no attribute probe."""
    from aquakin.plant import IFASUnit, InterfaceSettling, MBRUnit, SBRPhase, SBRUnit
    from aquakin.plant.mixer import MixerUnit
    from aquakin.plant.units import OxygenTransferUnit, ReactionBalanceUnit

    asm1 = aquakin.load_model("asm1")
    T = {"T": 293.15}
    units = [
        CSTRUnit("c", asm1, volume=100.0, input_port_names=["in"], conditions=T),
        MBRUnit("m", asm1, volume=100.0, conditions=T),
        IFASUnit(name="i", model=asm1, volume=100.0, input_port_names=["in"],
                 specific_surface_area=500.0, fill_fraction=0.4,
                 biofilm_thickness=5e-4, n_layers=2, conditions=T),
        SBRUnit("s", asm1, [SBRPhase("react", 1.0, kla=10.0)], full_volume=100.0,
                feed_flow=None, decant_flow=0.0, feed="continuous",
                settling=InterfaceSettling(v_settle=1.0, area=1.0), conditions=T),
    ]
    for u in units:
        assert isinstance(u, ReactionBalanceUnit), u.name
        assert isinstance(u, OxygenTransferUnit), u.name
        assert u.reaction_gas_components == frozenset({"COD", "N"})
    mixer = MixerUnit("x", ["a", "b"], asm1)
    assert not isinstance(mixer, ReactionBalanceUnit)
    assert not isinstance(mixer, OxygenTransferUnit)


def test_mass_balance_default_ports_are_influents_and_dangling_outputs():
    net = aquakin.load_model("asm1")
    plant = _aerated_cstr_plant(net)
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.array([0.0, 1.0]))
    mb = plant.mass_balance(sol, components=("COD",))
    assert mb.influent_ports == ["feed"]
    assert mb.effluent_ports == plant.check().dangling_outputs   # ["tank.out"]



def _anoxic_cstr_plant(T_feed, temperature_model=None):
    """A single anoxic BSM2-ASM1 CSTR (static ``T`` 293.15 K) fed nitrate and
    biomass at ``T_feed``, so denitrification makes N2 at a temperature-
    dependent rate (the BSM2 ASM1 carries Arrhenius corrections)."""
    from aquakin.plant.bsm.bsm2 import bsm2_asm1_model
    from aquakin.plant.influent import InfluentSeries

    net = bsm2_asm1_model()
    plant = Plant("anoxic", time_unit="d")
    plant.add_unit(CSTRUnit(name="tank", model=net, volume=1000.0,
                            input_port_names=["inlet"], conditions={"T": 293.15},
                            aeration=Aeration(kla=0.0)))
    feed = InfluentSeries.constant(net, SS=200.0, XS=100.0, XB_H=1500.0, SNO=20.0,
                                   SNH=25.0, SALK=7.0, Q=1000.0, T=T_feed)
    plant.add_influent("feed", feed, to="tank.inlet")
    if temperature_model is not None:
        plant.set_temperature_model(temperature_model)
    return plant


@pytest.mark.parametrize("heat_balance", [False, True], ids=["algebraic", "heat_balance"])
def test_mass_balance_reacts_at_the_operating_temperature(heat_balance):
    """The balance's reaction integral (the denitrification N2 and the COD it
    oxidizes) must run at the temperature the solve ran the kinetics at: the
    flow-weighted inlet temperature under the algebraic model, the tank's own
    lagged temperature state under a heat balance -- not the unit's static
    ``T``. With the feed 10 K below the static ``T`` the static evaluation
    misses N by ~14% (algebraic) and ~1.4% (heat balance, whose tank cools
    only ~4 K over the window) (#724)."""
    tm = aquakin.HeatBalanceTemperature() if heat_balance else None
    plant = _anoxic_cstr_plant(283.15, tm)
    sol = plant.solve(t_span=(0.0, 0.5), t_eval=jnp.linspace(0.0, 0.5, 401))
    mb = plant.mass_balance(sol, components=("COD", "N"))
    # Denitrification really happens, so the N gas term is being tested.
    assert mb.gas_detail["denitrification_N2"] > 0.1 * mb["N"].inflow
    # The gas integrals are trapezoids over the saved points across the
    # start-up transient: with the feed at the static T the residual is ~2e-3
    # on 201 points and ~1e-4 on 801, so 401 points closes well inside 2e-3.
    for q in ("COD", "N"):
        assert abs(mb[q].relative_imbalance) < 2e-3, (q, mb[q].relative_imbalance)


def test_mass_balance_of_a_plant_without_reactions_has_no_gas():
    """A storage tank reacts nothing and aerates nothing: the gas terms are
    exactly zero and the inflow splits into outflow plus accumulation."""
    from aquakin.plant.storage import StorageTank

    net = aquakin.load_model("asm1")
    plant = Plant("store", time_unit="d")
    plant.add_unit(StorageTank(name="store", model=net, volume=160.0, output_flow=50.0))
    plant.add_influent("feed", net.influent({"SS": 60.0, "SNH": 20.0}, Q=100.0), to="store.in")
    sol = plant.solve(t_span=(0.0, 0.5), t_eval=jnp.linspace(0.0, 0.5, 101))
    mb = plant.mass_balance(sol, components=("COD", "N"))
    for q in ("COD", "N"):
        assert mb[q].gas == 0.0
        # The default start inventory dilutes out: accumulation is exercised.
        assert abs(mb[q].accumulation) > 0.1 * mb[q].inflow
        # Trapezoid flux integrals over the dilution transient on 101 points.
        assert abs(mb[q].relative_imbalance) < 1e-4, (q, mb[q].relative_imbalance)


# The whole-plant balances below start from the shipped warm start rather than
# a steady state: the balance carries the accumulation term, so it closes over
# any window. The window opens one day in, after the fast transients away from
# the warm start (hours long) have passed: the gas and flux integrals are
# trapezoids over the saved points, accurate only where the trajectory is
# smooth on the save grid (from t = 0 on the same grids BSM1 misses COD by
# ~50%, and still ~2% on 201 points). Over the window the plant
# inventory is still moving (BSM1: accumulation ~28% of the COD inflow), so
# the accumulation bookkeeping is exercised, not just in = out + gas.
_BSM_WINDOW = (1.0, 3.0)


@pytest.mark.slow
def test_mass_balance_closes_on_bsm1_from_warm_start():
    """The full BSM1 plant closes COD and N to 1e-4 over a window in which its
    inventory is still changing (single-model water line: no digester gas
    phase)."""
    from aquakin.plant.bsm import bsm1_warm_start, build_bsm1

    net = aquakin.load_model("asm1")
    plant = build_bsm1(model=net)
    feed = net.influent({"SS": 69.5, "XS": 202.32, "XB_H": 28.17, "SNH": 31.56,
                         "SND": 6.95, "XND": 10.59, "SI": 30.0, "XI": 51.2,
                         "SALK": 7.0}, Q=18446.0)
    plant.add_influent("feed", feed, to="inlet_mix.fresh")
    sol = plant.solve(t_span=(0.0, _BSM_WINDOW[1]), t_eval=jnp.linspace(*_BSM_WINDOW, 33),
                      y0=jnp.asarray(bsm1_warm_start(plant)))
    mb = plant.mass_balance(sol, components=("COD", "N"))
    assert abs(mb["COD"].relative_imbalance) < 1e-4
    assert abs(mb["N"].relative_imbalance) < 1e-4


@pytest.mark.slow
def test_mass_balance_closes_on_bsm1_takacs_lumped_settler():
    """The BSM1 plant with the layered Takács settler closes COD and N at
    steady state. ``build_bsm1(use_takacs=True)`` defaults to the reference
    ``settler1dv4``'s form (``composition_mode="lumped_tss"``, soluble holdup),
    whose head block is one TSS value per layer rather than per species -- the
    regression guard for the lumped-settler branch of the settler's
    ``component_inventory`` (which would otherwise fail to reshape the TSS head
    block). This one stays at steady state: the lumped blanket's N inventory
    assumes the model-default solids composition, so it is exact only where
    the blanket does not accumulate."""
    from aquakin.plant.bsm import bsm1_warm_start, build_bsm1

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
def test_mass_balance_closes_on_bsm2_from_warm_start():
    """The two-model BSM2 plant (ASM1 water line + ADM1 digester, biogas, the
    reject recycle) closes COD and N to <0.5% over a window from the warm start
    -- the digester biogas falls out of the reaction integral and the
    cross-model inventories sum on the canonical g basis."""
    from aquakin.plant.bsm import (
        bsm2_constant_influent,
        bsm2_parameters,
        bsm2_warm_start,
        build_bsm2,
    )

    asm1 = aquakin.load_model("asm1")
    adm1 = aquakin.load_model("adm1")
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    params = bsm2_parameters(asm1, adm1)
    sol = plant.solve(t_span=(0.0, _BSM_WINDOW[1]), t_eval=jnp.linspace(*_BSM_WINDOW, 9),
                      params=params, y0=jnp.asarray(bsm2_warm_start(plant)))
    mb = plant.mass_balance(sol, components=("COD", "N"), params=params)
    assert abs(mb["COD"].relative_imbalance) < 5e-3
    assert abs(mb["N"].relative_imbalance) < 5e-3
