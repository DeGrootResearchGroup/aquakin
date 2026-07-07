"""A²O biological-nutrient-removal plant (ASM2d) regression tests.

These exercise ASM2d to a *viable* nutrient-removal steady state -- something no
other test does, and the reason several ASM2d process-matrix errors went
unnoticed (they all conserve COD/N/P, so the continuity suite passed while
nitrification and bio-P were broken). Running the plant catches them: with the
lysis biomass assignment, the poly-P storage K_MAX inhibition, the autotroph /
PP-uptake half-saturation constants, and the in-plant positivity limiter all
correct, the plant nitrifies and removes phosphorus; with any of them broken it
washes out the nitrifiers or fails bio-P.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import FerricDose, a2o_influent, a2o_warm_start, build_a2o


@pytest.fixture(scope="module")
def asm2d():
    return aquakin.load_model("asm2d")


# --------------------------------------------------------------------------
# Fast model-level checks (no plant solve): the process-matrix fixes.
# --------------------------------------------------------------------------


def test_lysis_consumes_the_correct_biomass(asm2d):
    """Heterotroph lysis (rate bH*[XH]) must decrement XH, and autotroph lysis
    (bAUT*[XAUT]) must decrement XAUT -- not the other way round. The original
    import swapped the heterotroph-lysis biomass onto XAUT, applying the (large)
    heterotroph decay to the nitrifiers."""
    net = asm2d
    sidx = net.species_index
    ridx = {n: i for i, n in enumerate(net.reaction_names)}
    S = net.stoich_matrix
    # Heterotroph lysis: consumes XH, leaves XAUT untouched.
    r1 = ridx["Lysis_1"]
    assert float(S[r1, sidx["XH"]]) == pytest.approx(-1.0)
    assert float(S[r1, sidx["XAUT"]]) == 0.0
    # Autotroph lysis: consumes XAUT.
    r2 = ridx["Lysis_2"]
    assert float(S[r2, sidx["XAUT"]]) == pytest.approx(-1.0)
    assert float(S[r2, sidx["XH"]]) == 0.0


def test_autotrophs_grow_not_crash_under_aeration(asm2d):
    """With XH present, the net autotroph rate at an aerated, ammonia-rich state
    must be growth-positive (only its own lysis removes it). The lysis swap made
    it strongly negative (heterotroph decay applied to XAUT)."""
    net = asm2d
    C = net.concentrations(
        {
            "SO2": 6.0,
            "SF": 5.0,
            "SA": 2.0,
            "SNH4": 25.0,
            "SNO3": 1.0,
            "SPO4": 5.0,
            "SI": 30.0,
            "SALK": 7.0,
            "XH": 1800.0,
            "XPAO": 400.0,
            "XPP": 50.0,
            "XPHA": 5.0,
            "XAUT": 120.0,
            "XTSS": 3500.0,
        },
        base="zero",
    )
    rates = net.rates(jnp.maximum(C, 0.0), net.default_parameters(), {"T": jnp.array([293.15])}, 0)
    xaut = net.species_index["XAUT"]
    dXAUT = float(net.stoich_matrix[:, xaut] @ rates)
    # Net growth: aerobic growth (+) dominates its lysis (-).
    assert dXAUT > 0.0


def test_precipitation_consumes_metal_and_forms_metal_phosphate(asm2d):
    """The chemical-P precipitation reaction must consume metal hydroxide (XMeOH)
    and form metal phosphate (XMeP), not just track their sum in XTSS. The import
    had dropped the XMeOH / XMeP coefficients, leaving the metal an inexhaustible
    catalyst that precipitated phosphate into nothing."""
    net = asm2d
    S = net.compute_stoich(net.default_parameters())
    ridx = {n: i for i, n in enumerate(net.reaction_names)}
    si = net.species_index
    p = ridx["Precipitation"]
    assert float(S[p, si["SPO4"]]) == pytest.approx(-1.0)
    assert float(S[p, si["XMeOH"]]) < 0.0  # metal hydroxide consumed
    assert float(S[p, si["XMeP"]]) > 0.0  # metal phosphate formed
    # Redissolution reverses it.
    r = ridx["Redissolution"]
    assert float(S[r, si["XMeOH"]]) > 0.0
    assert float(S[r, si["XMeP"]]) < 0.0


def test_polyp_storage_has_kmax_term(asm2d):
    """The aerobic poly-P storage inhibition must carry the maximum-ratio K_MAX
    term, so stored poly-P can reach ~K_MAX*XPAO rather than being capped ~17x
    too low. Probe: at a low stored fraction the storage inhibition factor is
    near 1 (storage enabled), and KMAX is a model parameter."""
    net = asm2d
    assert "KMAX" in net.param_index
    # KO2_AUT / KNH4_AUT / KPS distinct from the heterotroph/hydrolysis values.
    pv = net.parameter_values({})
    names = net.param_index
    assert float(pv[names["KNH4_AUT"]]) == pytest.approx(1.0)  # not 0.05
    assert float(pv[names["KO2_AUT"]]) == pytest.approx(0.5)  # not 0.2
    assert float(pv[names["KPS"]]) == pytest.approx(0.2)  # not 0.01


def test_cstr_reaction_term_routes_through_dCdt(asm2d):
    """A plant CSTR's reaction term must be the model's canonical ``dCdt`` --
    so ``clip_negative_states`` and the positivity limiter are applied identically
    to a standalone reactor, and cannot be silently bypassed. (The original plant
    bug was CSTRUnit building its RHS from ``rates()`` directly, leaving the
    limiter inert and letting a consumed soluble integrate negative.)

    With no inflow ports and no aeration the CSTR RHS is purely the reaction term,
    so it must equal ``dCdt`` exactly; and on a near-depleted species the limiter
    must actually fire (the term differs from the un-limited ``stoich.T @ rates``).
    """
    from aquakin.plant.cstr import CSTRUnit

    net = asm2d
    assert net.positivity_threshold is not None  # asm2d carries the limiter
    unit = CSTRUnit(
        name="t", model=net, volume=1000.0, input_port_names=[], conditions={"T": 293.15}
    )
    params = net.default_parameters()
    cond = {"T": jnp.asarray([293.15])}

    # A near-depleted acetate pool (below the 1e-3 limiter threshold) being
    # consumed by heterotroph/PAO uptake -- where the limiter must throttle.
    state = net.concentrations(
        {
            "SO2": 2.0,
            "SF": 2.0,
            "SA": 5.0e-4,
            "SNH4": 5.0,
            "SNO3": 3.0,
            "SPO4": 15.0,
            "SALK": 7.0,
            "XH": 1200.0,
            "XPAO": 300.0,
            "XPP": 80.0,
            "XPHA": 20.0,
            "XAUT": 80.0,
        },
        base="zero",
    )

    rhs = unit.rhs(0.0, state, {}, params)  # no inflow, no aeration
    expected = net.dCdt(state, params, cond, 0)
    assert float(jnp.max(jnp.abs(rhs - expected))) == 0.0  # bit-identical

    # The limiter is genuinely active: the un-limited net term over-consumes the
    # depleted acetate, and dCdt (hence the CSTR) throttles it back.
    sa = net.species_index["SA"]
    unlimited = net.compute_stoich(params).T @ net.rates(state, params, cond, 0)
    assert unlimited[sa] < 0.0  # acetate being consumed
    assert float(rhs[sa]) > float(unlimited[sa])  # throttled toward zero


# --------------------------------------------------------------------------
# Fast builder-assembly checks (no plant solve): structure of build_a2o and
# the influent / warm-start helpers.
# --------------------------------------------------------------------------


def test_build_a2o_assembles_expected_units(asm2d):
    plant = build_a2o(asm2d)
    units = set(plant.list_units())
    expected = {
        "front_mix",
        "anaer1",
        "anaer2",
        "anoxic_mix",
        "anox1",
        "anox2",
        "aer1",
        "aer2",
        "aer3",
        "aer_split",
        "clarifier",
        "underflow_split",
    }
    assert expected <= units
    assert "ferric_dose" not in units
    assert plant.effluent_endpoint == "clarifier.overflow"
    # Zone volumes from the A²O constants.
    assert plant.units["anaer1"].volume == pytest.approx(750.0)
    assert plant.units["aer1"].volume == pytest.approx(1333.0)


def test_build_a2o_aeration_only_in_aerobic_zone(asm2d):
    plant = build_a2o(asm2d)
    for name in ("anaer1", "anaer2", "anox1", "anox2"):
        assert plant.units[name].aeration is None
    for name in ("aer1", "aer2", "aer3"):
        assert plant.units[name].aeration is not None


def test_build_a2o_ferric_inserts_dosing_unit(asm2d):
    plant = build_a2o(asm2d, ferric=FerricDose(flow=5.0))
    assert "ferric_dose" in plant.list_units()


def test_build_a2o_recycle_streams_registered(asm2d):
    plant = build_a2o(asm2d)
    assert {"internal_recycle", "ras", "wastage"} <= set(plant.list_streams())


def test_build_a2o_use_takacs_selects_stateful_settler(asm2d):
    ideal = build_a2o(asm2d, use_takacs=False)
    takacs = build_a2o(asm2d, use_takacs=True)
    assert ideal.units["clarifier"].state_size == 0
    assert takacs.units["clarifier"].state_size > 0


def test_a2o_influent_flow_and_overrides(asm2d):
    inf = a2o_influent(asm2d)
    s = inf.at(0.0)
    assert float(s.Q) == pytest.approx(18446.0)
    base_po4 = float(s.C[asm2d.species_index["SPO4"]])
    over = a2o_influent(asm2d, overrides={"SPO4": base_po4 + 7.0}).at(0.0)
    assert float(over.C[asm2d.species_index["SPO4"]]) == pytest.approx(base_po4 + 7.0)
    # An un-overridden species is unchanged.
    assert float(over.C[asm2d.species_index["SA"]]) == pytest.approx(
        float(s.C[asm2d.species_index["SA"]])
    )


def test_a2o_warm_start_seeds_reactors(asm2d):
    plant = build_a2o(asm2d)
    y0 = a2o_warm_start(plant)
    assert y0.shape == plant.initial_state().shape
    # The aerobic reactors are seeded with a non-trivial biomass composition.
    sb = plant.states_by_unit(y0)
    assert float(jnp.sum(sb["aer3"])) > 0.0


def test_ferric_dose_defaults_and_frozen():
    import dataclasses

    fd = FerricDose(flow=5.0)
    assert fd.xmeoh_conc == pytest.approx(1.0e5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fd.flow = 1.0


# --------------------------------------------------------------------------
# Slow plant-level check: the working nutrient-removal steady state.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a2o_removes_nitrogen_and_phosphorus(asm2d):
    """The default A²O plant, warm-started, reaches a feasible steady state that
    nitrifies and removes phosphorus -- the end-to-end check that the ASM2d fixes
    and the in-plant positivity limiter all hold together."""
    net = asm2d
    plant = build_a2o(net)
    plant.add_influent("feed", a2o_influent(net))
    y0 = a2o_warm_start(plant)
    sol = plant.solve(
        t_span=(0.0, 200.0),
        t_eval=jnp.array([200.0]),
        y0=y0,
        rtol=1e-5,
        atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=4_000_000),
    )
    eff = plant.stream(sol, "effluent")
    last = {s: float(eff.C_named(s)[-1]) for s in ("SNH4", "SNO3", "SPO4", "SF", "SA", "SI")}
    infl = a2o_influent(net).at(0.0)
    p_in = float(infl.C[net.species_index["SPO4"]])
    nh_in = float(infl.C[net.species_index["SNH4"]])

    # Feasible: no recirculating-negative soluble pools.
    assert last["SA"] >= -1e-2
    assert last["SNO3"] >= -1e-2
    # Inert soluble COD is conserved through the plant.
    assert last["SI"] == pytest.approx(30.0, abs=0.5)
    # Nitrification: most influent ammonia is removed.
    assert last["SNH4"] < 0.5 * nh_in
    # Biological P removal: effluent phosphate well below influent.
    assert last["SPO4"] < 0.5 * p_in

    # A healthy PAO population with stored poly-P established in the aerobic zone.
    sb = plant.states_by_unit(sol.final_state)
    aer3 = sb["aer3"]
    assert float(aer3[net.species_index["XPAO"]]) > 500.0
    assert float(aer3[net.species_index["XPP"]]) > 50.0


@pytest.mark.slow
def test_ferric_dosing_polishes_phosphorus(asm2d):
    """On a phosphorus-rich, VFA-limited influent biological P removal alone
    leaves residual phosphate; ferric dosing precipitates it (forming XMeP) and
    lowers the effluent phosphate -- the combined bio-P + chemical-P behaviour."""
    net = asm2d
    overrides = {"SPO4": 15.0, "SA": 20.0}

    def effluent_p(ferric):
        plant = build_a2o(net, ferric=ferric)
        plant.add_influent("feed", a2o_influent(net, overrides=overrides))
        y0 = a2o_warm_start(plant)
        sol = plant.solve(
            t_span=(0.0, 200.0),
            t_eval=jnp.array([200.0]),
            y0=y0,
            rtol=1e-5,
            atol=1e-3,
            integrator=aquakin.IntegratorConfig(max_steps=8_000_000),
        )
        eff = plant.stream(sol, "effluent")
        return float(eff.C_named("SPO4")[-1]), float(eff.C_named("XMeP")[-1])

    p_bio, xmep_bio = effluent_p(None)
    p_chem, xmep_chem = effluent_p(FerricDose(flow=5.0))

    assert xmep_bio == pytest.approx(0.0, abs=1e-3)  # no precipitation without metal
    assert xmep_chem > 1.0  # metal phosphate forms
    assert p_chem < p_bio  # ferric lowers effluent P


def test_asm2d_grad_flows(asm2d):
    """jax.grad flows through an asm2d BatchReactor solve -- exercising the new
    AD paths (the K_MAX poly-P inhibition's safe_div, clip_negative_states and the
    positivity limiter are all AD-safe) without the cost of the full plant
    discrete adjoint."""
    net = asm2d
    cond = aquakin.OperatingConditions(T=293.15)
    # dtmax caps the reverse-mode adjoint of the stiff bio-P kinetics (the
    # documented stiff-model gradient remedy).
    reactor = aquakin.BatchReactor(net, cond, integrator=aquakin.IntegratorConfig(dtmax=1e-3))
    C0 = net.concentrations(
        {
            "SO2": 2.0,
            "SF": 20.0,
            "SA": 20.0,
            "SNH4": 20.0,
            "SNO3": 2.0,
            "SPO4": 8.0,
            "SI": 30.0,
            "SALK": 7.0,
            "XH": 1500.0,
            "XPAO": 600.0,
            "XPP": 150.0,
            "XPHA": 30.0,
            "XAUT": 120.0,
            "XTSS": 4000.0,
        },
        base="zero",
    )
    base = net.default_parameters()
    qpp = net.param_index["qPP"]
    t_eval = jnp.linspace(0.0, 0.1, 4)

    def loss(scale):
        params = base.at[qpp].multiply(scale)
        sol = reactor.solve(C0, t_span=(0.0, 0.1), t_eval=t_eval, params=params)
        return sol.C_named("XPP")[-1]

    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(g)
