"""Tests for the Takács 1-D secondary clarifier."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import Plant
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.takacs import TakacsClarifier


@pytest.fixture
def asm1():
    return aquakin.load_model("asm1")


def _clarifier_plant(asm1, *, Q_inlet=56000.0, overflow_Q=18446.0, C_inlet=None):
    plant = Plant("clarifier")
    plant.add_unit(
        TakacsClarifier(
            name="clar",
            model=asm1,
            area=1500.0,
            height=4.0,
            overflow_Q=overflow_Q,
        )
    )
    if C_inlet is None:
        C_inlet = asm1.default_concentrations()
    inf = InfluentSeries(
        t=jnp.asarray([0.0, 10.0]),
        Q=jnp.asarray([Q_inlet, Q_inlet]),
        C=jnp.stack([C_inlet, C_inlet]),
        model=asm1,
    )
    plant.add_influent("feed", inf, to="clar.inlet")
    return plant


def test_solubles_pass_through(asm1):
    """Soluble species (SS, SNH, ...) must appear at the same concentration
    in both overflow and underflow because they don't settle."""
    # Make soluble SS distinctive in the inlet.
    C_inlet = asm1.concentrations(SS=50.0)
    plant = _clarifier_plant(asm1, C_inlet=C_inlet)
    sol = plant.solve(
        t_span=(0.0, 2.0), t_eval=jnp.asarray([0.0, 2.0]),
        rtol=1e-4, atol=1e-3,
    )
    # Reconstruct streams at t_end to inspect overflow / underflow.
    clar = plant.units["clar"]
    state = sol.state[-1]
    inf = plant.influents["feed"].at(jnp.asarray(2.0))
    streams = clar.compute_outputs(
        jnp.asarray(2.0), state, {"inlet": inf}, plant.default_parameters()
    )
    SS_idx = asm1.species_index["SS"]
    assert float(streams["overflow"].C[SS_idx]) == pytest.approx(50.0, rel=1e-6)
    assert float(streams["underflow"].C[SS_idx]) == pytest.approx(50.0, rel=1e-6)


def test_particulates_concentrate_in_underflow(asm1):
    """Particulates settle: at quasi-steady state the underflow X_BH
    concentration must significantly exceed the inlet, and the overflow
    must be much lower."""
    plant = _clarifier_plant(asm1)
    sol = plant.solve(
        t_span=(0.0, 2.0), t_eval=jnp.asarray([0.0, 2.0]),
        rtol=1e-4, atol=1e-3,
    )
    state = sol.state[-1]
    clar = plant.units["clar"]
    inf = plant.influents["feed"].at(jnp.asarray(2.0))
    streams = clar.compute_outputs(
        jnp.asarray(2.0), state, {"inlet": inf}, plant.default_parameters()
    )
    X_BH_in = float(inf.C[asm1.species_index["X_BH"] if "X_BH" in asm1.species_index else asm1.species_index["XB_H"]])
    XBH_idx = asm1.species_index["XB_H"]
    overflow_XBH = float(streams["overflow"].C[XBH_idx])
    underflow_XBH = float(streams["underflow"].C[XBH_idx])
    assert overflow_XBH < float(inf.C[XBH_idx]), "particulates should clarify"
    assert underflow_XBH > float(inf.C[XBH_idx]), "particulates should thicken"


def test_overall_mass_balance(asm1):
    """Mass in (Q_in × C_in) ≈ mass out (Q_over × C_over + Q_under × C_under)
    at quasi-steady state for every particulate species."""
    plant = _clarifier_plant(asm1)
    sol = plant.solve(
        t_span=(0.0, 5.0), t_eval=jnp.asarray([0.0, 5.0]),
        rtol=1e-4, atol=1e-3,
    )
    state = sol.state[-1]
    clar = plant.units["clar"]
    inf = plant.influents["feed"].at(jnp.asarray(5.0))
    streams = clar.compute_outputs(
        jnp.asarray(5.0), state, {"inlet": inf}, plant.default_parameters()
    )
    Q_in = float(inf.Q)
    Q_over = float(streams["overflow"].Q)
    Q_under = float(streams["underflow"].Q)
    assert Q_over + Q_under == pytest.approx(Q_in, rel=1e-6)
    # Particulate mass balance (within solver tolerance).
    XBH_idx = asm1.species_index["XB_H"]
    mass_in = Q_in * float(inf.C[XBH_idx])
    mass_out = (
        Q_over * float(streams["overflow"].C[XBH_idx])
        + Q_under * float(streams["underflow"].C[XBH_idx])
    )
    # Allow ~10% drift since we're not at exact steady state.
    assert mass_out == pytest.approx(mass_in, rel=0.15)


def test_takacs_blanket_profile_and_clarification(asm1):
    """At a realistic (BSM1) solids loading the corrected Takács model must
    build a monotone sludge blanket (dense bottom, clear top), clarify the
    effluent strongly, thicken the underflow, and conserve solids tightly.

    This exercises the clarification-zone flux limiting (without it the effluent
    is poorly clarified) and the per-species flux apportioning (settling moves
    each species at the bulk velocity, conserving total settleable solids)."""
    from aquakin.plant.metrics import derived_TSS
    from aquakin.plant.streams import Stream

    C = asm1.concentrations({"XB_H": 2200.0, "XB_A": 120.0, "XS": 80.0,
                             "XI": 1100.0, "XP": 600.0, "XND": 5.0})
    Q_in, overflow = 36892.0, 18061.0   # ~BSM1 clarifier loading

    plant = Plant("clar_load")
    clar = TakacsClarifier(name="clar", model=asm1, area=1500.0,
                           height=4.0, overflow_Q=overflow)
    plant.add_unit(clar)
    plant.add_influent("feed", InfluentSeries(
        t=jnp.asarray([0.0, 50.0]), Q=jnp.full((2,), Q_in),
        C=jnp.stack([C, C]), model=asm1), to="clar.inlet")
    sol = plant.solve(t_span=(0.0, 5.0), t_eval=jnp.asarray([0.0, 5.0]),
                      rtol=1e-5, atol=1e-3,
                      integrator=aquakin.IntegratorConfig(max_steps=200_000))
    assert jnp.all(jnp.isfinite(sol.state))

    st0, sz = plant._state_layout["clar"]
    lay = sol.state[-1, st0:st0 + sz].reshape((clar.n_layers, clar._n_part))
    tss = jnp.sum(lay * jnp.asarray(clar._part_tss_factors), axis=1)  # bottom->top
    # A settled blanket: bottom much denser than top, monotone non-increasing up.
    assert float(tss[0]) > 10.0 * float(tss[-1])
    assert bool(jnp.all(jnp.diff(tss) <= 1e-6))

    out = clar.compute_outputs(
        jnp.asarray(5.0), sol.state[-1, st0:st0 + sz],
        {"inlet": Stream(Q=jnp.asarray(Q_in), C=C, model=asm1)},
        plant.default_parameters())
    feed_tss = float(derived_TSS(C, asm1))
    eff_tss = float(derived_TSS(out["overflow"].C, asm1))
    und_tss = float(derived_TSS(out["underflow"].C, asm1))
    assert eff_tss < 0.05 * feed_tss       # strongly clarified effluent
    assert und_tss > 1.5 * feed_tss        # thickened underflow
    # Tight solids mass balance (corrected apportioning conserves TSS).
    mass_in = Q_in * feed_tss
    mass_out = overflow * eff_tss + (Q_in - overflow) * und_tss
    assert mass_out == pytest.approx(mass_in, rel=0.03)


def test_no_inflow_no_dynamics(asm1):
    """Initially the clarifier is at uniform default concentrations. With
    zero inflow (Q=0) and zero overflow, the clarifier should reach a
    quiescent state where particulates simply settle to the bottom."""
    plant = Plant("quiescent_clar")
    plant.add_unit(
        TakacsClarifier(
            name="clar", model=asm1, area=1500.0, height=4.0,
            overflow_Q=0.0,
        )
    )
    inf = InfluentSeries(
        t=jnp.asarray([0.0, 10.0]),
        Q=jnp.asarray([0.0, 0.0]),
        C=jnp.stack([
            asm1.default_concentrations(),
            asm1.default_concentrations(),
        ]),
        model=asm1,
    )
    plant.add_influent("feed", inf, to="clar.inlet")
    sol = plant.solve(
        t_span=(0.0, 0.5), t_eval=jnp.asarray([0.0, 0.5]),
        rtol=1e-4, atol=1e-3,
    )
    # All particulate state must remain finite.
    assert jnp.all(jnp.isfinite(sol.state))


def test_shared_asm1_constants_are_single_source(asm1):
    """Clarifier/Takács settling sets and the metrics TSS set come from the one
    shared constants module, so they cannot drift apart."""
    from aquakin.plant._constants import (
        ASM1_SETTLING_SPECIES,
        ASM1_TSS_FACTOR,
        ASM1_TSS_SPECIES,
    )
    from aquakin.plant import metrics
    from aquakin.plant import takacs as takacs_mod
    from aquakin.plant.clarifier import IdealClarifier

    ideal = IdealClarifier(name="c", model=asm1, overflow_Q=1.0)
    tak = TakacsClarifier(name="t", model=asm1, area=1.0, height=4.0,
                          overflow_Q=1.0)
    assert tuple(ideal.particulate_species) == ASM1_SETTLING_SPECIES
    assert tuple(tak.particulate_species) == ASM1_SETTLING_SPECIES
    # Takács TSS factors: the TSS species at the shared factor, XND at zero.
    assert takacs_mod._DEFAULT_TSS_FACTORS == {
        **{s: ASM1_TSS_FACTOR for s in ASM1_TSS_SPECIES}, "XND": 0.0}
    # Metrics reuses the same TSS set / factor object.
    assert metrics._TSS_SPECIES is ASM1_TSS_SPECIES
    assert metrics._TSS_FACTOR == ASM1_TSS_FACTOR


def test_composition_mode_validation(asm1):
    """An unrecognised composition_mode is rejected at construction."""
    with pytest.raises(ValueError, match="composition_mode"):
        TakacsClarifier(name="c", model=asm1, area=1.0, height=4.0,
                        overflow_Q=1.0, composition_mode="bogus")


def test_lumped_tss_state_size_and_initial_state(asm1):
    """lumped_tss carries one TSS value per layer; the uniform seed is the
    default composition summed to TSS, identical in every layer."""
    lm = TakacsClarifier(name="lm", model=asm1, area=1500.0, height=4.0,
                         overflow_Q=18446.0, composition_mode="lumped_tss")
    assert lm.state_size == lm.n_layers
    init = lm.initial_state()
    assert init.shape == (lm.n_layers,)
    part_def = jnp.asarray([float(asm1.default_concentrations()[i])
                            for i in lm._part_indices])
    expected = float(jnp.sum(part_def * jnp.asarray(lm._part_tss_factors)))
    assert bool(jnp.allclose(init, expected))


def test_lumped_tss_blanket_seed_matches_per_species(asm1):
    """With init_underflow_Q the settled-blanket TSS profile is shared: the
    lumped seed equals the per-species seed summed to TSS layer by layer."""
    kw = dict(model=asm1, area=1500.0, height=4.0, underflow_Q=18446.0,
              init_underflow_Q=18446.0)
    ps = TakacsClarifier(name="ps", composition_mode="per_species", **kw)
    lm = TakacsClarifier(name="lm", composition_mode="lumped_tss", **kw)
    ps_tss = jnp.sum(
        ps.initial_state().reshape((ps.n_layers, ps._n_part))
        * jnp.asarray(ps._part_tss_factors)[None, :], axis=1)
    assert bool(jnp.allclose(lm.initial_state(), ps_tss))


def test_lumped_tss_rhs_matches_per_species_aggregated(asm1):
    """The lumped per-layer dTSS/dt equals the per-species dTSS/dt aggregated
    over species for the SAME total-TSS profile and inlet -- both apply the
    identical Takács flux/convection math to the total solids."""
    from aquakin.plant.streams import Stream

    C = asm1.concentrations({"XB_H": 2200.0, "XB_A": 120.0, "XS": 80.0,
                             "XI": 1100.0, "XP": 600.0, "XND": 5.0})
    Q_in, overflow = 36892.0, 18061.0
    kw = dict(model=asm1, area=1500.0, height=4.0, overflow_Q=overflow)
    ps = TakacsClarifier(name="ps", composition_mode="per_species", **kw)
    lm = TakacsClarifier(name="lm", composition_mode="lumped_tss", **kw)

    part_def = jnp.asarray([float(C[i]) for i in ps._part_indices])
    feed_tss = float(jnp.sum(part_def * jnp.asarray(ps._part_tss_factors)))
    # A non-uniform settled profile (dense bottom, clear top), bottom -> top.
    scales = jnp.asarray([3.0, 1.5, 1.2, 1.0, 1.0, 1.0, 0.4, 0.2, 0.05, 0.02])
    ps_state = (scales[:, None] * part_def[None, :]).reshape(-1)
    lm_state = scales * feed_tss

    params = asm1.default_parameters()
    inlet = Stream(Q=jnp.asarray(Q_in), C=C, model=asm1)
    d_lm = lm.rhs(jnp.asarray(0.0), lm_state, {"inlet": inlet}, params)
    d_ps = ps.rhs(jnp.asarray(0.0), ps_state, {"inlet": inlet}, params)
    d_ps_tss = jnp.sum(
        d_ps.reshape((ps.n_layers, ps._n_part))
        * jnp.asarray(ps._part_tss_factors)[None, :], axis=1)
    assert d_lm.shape == (lm.n_layers,)
    assert bool(jnp.all(jnp.isfinite(d_lm)))
    # Equal up to the per-species apportioning's 1e-12 denominator guard,
    # relative to the (large) flux magnitude.
    assert bool(jnp.allclose(d_lm, d_ps_tss, rtol=1e-9, atol=1e-6))


def test_lumped_tss_outputs_clarify_thicken_passthrough(asm1):
    """lumped_tss compute_outputs: particulates scale by the boundary-layer /
    feed TSS ratio (clarified overflow, thickened underflow), solubles pass
    through, and a zero feed is guarded."""
    from aquakin.plant.metrics import derived_TSS
    from aquakin.plant.streams import Stream

    lm = TakacsClarifier(name="lm", model=asm1, area=1500.0, height=4.0,
                         overflow_Q=18061.0, composition_mode="lumped_tss")
    C = asm1.concentrations({"XB_H": 2200.0, "XB_A": 120.0, "XS": 80.0,
                             "XI": 1100.0, "XP": 600.0, "XND": 5.0,
                             "SS": 42.0, "SNH": 17.0})
    feed_tss = float(derived_TSS(C, asm1))
    # Bottom (index 0) thickened, top (index n-1) clarified.
    lm_state = jnp.asarray([3.0, 1.5, 1.2, 1.0, 1.0, 1.0, 0.4, 0.2, 0.05, 0.02]) * feed_tss

    params = asm1.default_parameters()
    out = lm.compute_outputs(
        jnp.asarray(0.0), lm_state,
        {"inlet": Stream(Q=jnp.asarray(36892.0), C=C, model=asm1)}, params)
    eff_tss = float(derived_TSS(out["overflow"].C, asm1))
    und_tss = float(derived_TSS(out["underflow"].C, asm1))
    assert eff_tss < feed_tss          # clarified
    assert und_tss > feed_tss          # thickened
    # Particulate scaling is the boundary-layer / feed TSS ratio.
    assert eff_tss == pytest.approx(0.02 * feed_tss, rel=1e-6)
    assert und_tss == pytest.approx(3.0 * feed_tss, rel=1e-6)
    # Solubles pass through unchanged in both outlets.
    for sp, val in (("SS", 42.0), ("SNH", 17.0)):
        i = asm1.species_index[sp]
        assert float(out["overflow"].C[i]) == pytest.approx(val, rel=1e-9)
        assert float(out["underflow"].C[i]) == pytest.approx(val, rel=1e-9)

    # Zero-feed guard: no NaN/Inf, particulates fall back to feed passthrough.
    out0 = lm.compute_outputs(
        jnp.asarray(0.0), lm_state,
        {"inlet": Stream(Q=jnp.asarray(0.0), C=jnp.zeros((asm1.n_species,)),
                         model=asm1)}, params)
    assert bool(jnp.all(jnp.isfinite(out0["overflow"].C)))
    assert bool(jnp.all(jnp.isfinite(out0["underflow"].C)))


def test_lumped_tss_solids_mass(asm1):
    """In lumped mode solids_mass sums the per-layer TSS over the layer volumes,
    matching the per-species mass for the same total-TSS profile."""
    kw = dict(model=asm1, area=1500.0, height=4.0, overflow_Q=18061.0)
    ps = TakacsClarifier(name="ps", composition_mode="per_species", **kw)
    lm = TakacsClarifier(name="lm", composition_mode="lumped_tss", **kw)
    part_def = jnp.asarray([float(asm1.default_concentrations()[i])
                            for i in ps._part_indices])
    feed_tss = float(jnp.sum(part_def * jnp.asarray(ps._part_tss_factors)))
    scales = jnp.asarray([3.0, 1.5, 1.2, 1.0, 1.0, 1.0, 0.4, 0.2, 0.05, 0.02])
    lm_state = scales * feed_tss
    ps_state = (scales[:, None] * part_def[None, :]).reshape(-1)
    layer_volume = lm.area * lm.height / lm.n_layers
    assert float(lm.solids_mass(lm_state)) == pytest.approx(
        float(jnp.sum(lm_state)) * layer_volume, rel=1e-12)
    assert float(lm.solids_mass(lm_state)) == pytest.approx(
        float(ps.solids_mass(ps_state)), rel=1e-9)


def test_lumped_tss_grad_through_rhs(asm1):
    """jax.grad flows through the lumped-mode rhs without NaNs (AD-clean)."""
    import jax
    from aquakin.plant.streams import Stream

    lm = TakacsClarifier(name="lm", model=asm1, area=1500.0, height=4.0,
                         overflow_Q=18061.0, composition_mode="lumped_tss")
    C = asm1.concentrations({"XB_H": 2200.0, "XI": 1100.0})
    inlet = Stream(Q=jnp.asarray(36892.0), C=C, model=asm1)
    state = lm.initial_state()

    def loss(p):
        d = lm.rhs(jnp.asarray(0.0), state, {"inlet": inlet}, p)
        return jnp.sum(d ** 2)

    g = jax.grad(loss)(asm1.default_parameters())
    assert bool(jnp.all(jnp.isfinite(g)))


def test_lumped_tss_plant_solve(asm1):
    """A lumped_tss clarifier integrates in a plant to a finite, clarifying
    steady-ish state (clarified effluent, thickened underflow)."""
    from aquakin.plant.metrics import derived_TSS
    from aquakin.plant.streams import Stream

    C = asm1.concentrations({"XB_H": 2200.0, "XB_A": 120.0, "XS": 80.0,
                             "XI": 1100.0, "XP": 600.0, "XND": 5.0})
    Q_in, overflow = 36892.0, 18061.0
    plant = Plant("clar_lumped")
    clar = TakacsClarifier(name="clar", model=asm1, area=1500.0, height=4.0,
                           overflow_Q=overflow, composition_mode="lumped_tss")
    plant.add_unit(clar)
    plant.add_influent("feed", InfluentSeries(
        t=jnp.asarray([0.0, 50.0]), Q=jnp.full((2,), Q_in),
        C=jnp.stack([C, C]), model=asm1), to="clar.inlet")
    sol = plant.solve(t_span=(0.0, 5.0), t_eval=jnp.asarray([0.0, 5.0]),
                      rtol=1e-5, atol=1e-3,
                      integrator=aquakin.IntegratorConfig(max_steps=200_000))
    assert jnp.all(jnp.isfinite(sol.state))

    st0, sz = plant._state_layout["clar"]
    assert sz == clar.n_layers
    tss = sol.state[-1, st0:st0 + sz]                 # bottom -> top
    assert float(tss[0]) > 10.0 * float(tss[-1])      # dense bottom, clear top
    assert bool(jnp.all(jnp.diff(tss) <= 1e-6))       # monotone non-increasing up

    out = clar.compute_outputs(
        jnp.asarray(5.0), tss,
        {"inlet": Stream(Q=jnp.asarray(Q_in), C=C, model=asm1)},
        plant.default_parameters())
    feed_tss = float(derived_TSS(C, asm1))
    assert float(derived_TSS(out["overflow"].C, asm1)) < 0.05 * feed_tss
    assert float(derived_TSS(out["underflow"].C, asm1)) > 1.5 * feed_tss


def test_soluble_holdup_state_size_and_initial_state(asm1):
    """Opt-in soluble holdup appends an (n_layers, n_soluble) tail block, leaving
    the particulate head block (and the no-holdup state_size) unchanged."""
    kw = dict(model=asm1, area=1500.0, height=4.0, underflow_Q=18446.0)
    base = TakacsClarifier(name="b", **kw)
    held = TakacsClarifier(name="h", soluble_holdup=True, **kw)
    assert held.state_size == base.state_size + held.n_layers * held._n_sol
    y0 = held.initial_state()
    assert y0.shape[0] == held.state_size
    # Head block is byte-identical to the no-holdup initial state.
    assert bool(jnp.allclose(y0[: base.state_size], base.initial_state()))


def test_soluble_holdup_steady_state_invariance(asm1):
    """A non-reacting soluble at the uniform = feed profile is a fixed point
    (rhs ~ 0) and both outlets carry the feed concentration -- so soluble holdup
    leaves every steady state unchanged."""
    from aquakin.plant.streams import Stream

    held = TakacsClarifier(name="h", model=asm1, area=1500.0, height=4.0,
                           underflow_Q=18446.0, soluble_holdup=True)
    C = asm1.concentrations({"SS": 50.0, "SNH": 25.0})
    inlet = Stream(Q=jnp.asarray(20000.0), C=C, model=asm1)
    params = asm1.default_parameters()
    sol_idx = held._sol_idx_arr
    # State: particulate blanket seed + every soluble layer at the feed value.
    part0 = held._particulate_initial_state()
    sol_feed = jnp.tile(C[sol_idx], held.n_layers)
    y = jnp.concatenate([part0, sol_feed])
    _, sol_d = held._unpack(held.rhs(jnp.asarray(0.0), y, {"inlet": inlet}, params))
    assert float(jnp.max(jnp.abs(sol_d))) < 1e-9      # soluble block is at rest
    outs = held.compute_outputs(jnp.asarray(0.0), y, {"inlet": inlet}, params)
    assert bool(jnp.allclose(outs["overflow"].C[sol_idx], C[sol_idx]))
    assert bool(jnp.allclose(outs["underflow"].C[sol_idx], C[sol_idx]))


def test_soluble_holdup_damps_vs_passthrough(asm1):
    """With holdup the overflow soluble is the (lagged) top layer, not the
    instantaneous feed: perturbing a held layer shifts the outlet, and a plant
    solve shows the overflow soluble lagging a step change that the no-holdup
    clarifier passes through instantly."""
    from aquakin.plant.streams import Stream

    held = TakacsClarifier(name="h", model=asm1, area=1500.0, height=4.0,
                           underflow_Q=18446.0, soluble_holdup=True)
    C = asm1.concentrations({"SS": 50.0})
    inlet = Stream(Q=jnp.asarray(20000.0), C=C, model=asm1)
    params = asm1.default_parameters()
    sol_idx = held._sol_idx_arr
    part0 = held._particulate_initial_state()
    y = jnp.concatenate([part0, jnp.tile(C[sol_idx], held.n_layers)])
    base_over = held.compute_outputs(jnp.asarray(0.0), y, {"inlet": inlet}, params)[
        "overflow"].C[sol_idx]
    # Bump the first soluble in the TOP layer (overflow boundary) by +5.
    top0 = held._part_block_size + (held.n_layers - 1) * held._n_sol
    y_pert = y.at[top0].add(5.0)
    pert_over = held.compute_outputs(jnp.asarray(0.0), y_pert, {"inlet": inlet}, params)[
        "overflow"].C[sol_idx]
    assert float(pert_over[0] - base_over[0]) == pytest.approx(5.0, rel=1e-9)

    # Plant solve: step the influent SS up; the held overflow must lag the feed
    # at a short time (whereas a no-holdup clarifier equals the feed instantly).
    Q_in, overflow = 36892.0, 18446.0
    C_lo = asm1.concentrations({"XB_H": 2200.0, "XI": 1100.0, "SS": 10.0})
    C_hi = asm1.concentrations({"XB_H": 2200.0, "XI": 1100.0, "SS": 200.0})
    plant = Plant("clar_hold")
    clar = TakacsClarifier(name="clar", model=asm1, area=1500.0, height=4.0,
                           underflow_Q=overflow, soluble_holdup=True)
    plant.add_unit(clar)
    plant.add_influent("feed", InfluentSeries(
        t=jnp.asarray([0.0, 0.0, 50.0]), Q=jnp.full((3,), Q_in),
        C=jnp.stack([C_lo, C_hi, C_hi]), model=asm1), to="clar.inlet")
    sol = plant.solve(t_span=(0.0, 0.05), t_eval=jnp.asarray([0.0, 0.05]),
                      rtol=1e-5, atol=1e-4,
                      integrator=aquakin.IntegratorConfig(max_steps=200_000))
    assert jnp.all(jnp.isfinite(sol.state))
    SS_idx = asm1.species_index["SS"]
    out = clar.compute_outputs(
        jnp.asarray(0.05), sol.state[-1],
        {"inlet": Stream(Q=jnp.asarray(Q_in), C=C_hi, model=asm1)},
        plant.default_parameters())
    # Overflow SS still well below the 200 feed shortly after the step: holdup.
    assert float(out["overflow"].C[SS_idx]) < 150.0


def test_soluble_holdup_grad_through_rhs(asm1):
    """jax.grad flows through the soluble-holdup rhs without NaNs."""
    import jax
    from aquakin.plant.streams import Stream

    held = TakacsClarifier(name="h", model=asm1, area=1500.0, height=4.0,
                           overflow_Q=18061.0, soluble_holdup=True)
    C = asm1.concentrations({"XB_H": 2200.0, "XI": 1100.0, "SS": 40.0, "SNH": 25.0})
    inlet = Stream(Q=jnp.asarray(36892.0), C=C, model=asm1)
    state = held.initial_state()

    def loss(p):
        d = held.rhs(jnp.asarray(0.0), state, {"inlet": inlet}, p)
        return jnp.sum(d ** 2)

    g = jax.grad(loss)(asm1.default_parameters())
    assert bool(jnp.all(jnp.isfinite(g)))


def test_controlled_split_helper():
    """Shared overflow/underflow split helper used by both clarifiers."""
    from aquakin.plant._flow_split import (
        split_controlled_flows,
        validate_controlled_split,
    )

    # Exactly one of overflow/underflow must be set.
    with pytest.raises(ValueError, match="exactly one"):
        validate_controlled_split("X", None, None)
    with pytest.raises(ValueError, match="exactly one"):
        validate_controlled_split("X", 1.0, 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        validate_controlled_split("X", None, -1.0)
    validate_controlled_split("X", None, 5.0)  # ok

    Q_in = jnp.asarray(100.0)
    # Fixed underflow -> overflow is the remainder.
    o, u = split_controlled_flows(None, 30.0, Q_in, clamp=True)
    assert float(o) == pytest.approx(70.0) and float(u) == pytest.approx(30.0)
    # Fixed overflow -> underflow is the remainder.
    o, u = split_controlled_flows(40.0, None, Q_in, clamp=True)
    assert float(o) == pytest.approx(40.0) and float(u) == pytest.approx(60.0)
    # clamp keeps both in [0, Q_in] when the feed dips below the setpoint.
    o, u = split_controlled_flows(None, 30.0, jnp.asarray(20.0), clamp=True)
    assert 0.0 <= float(o) <= 20.0 and 0.0 <= float(u) <= 20.0
    # clamp=False leaves the split affine (overflow may exceed Q_in).
    o, u = split_controlled_flows(None, 30.0, jnp.asarray(20.0), clamp=False)
    assert float(o) == pytest.approx(-10.0)  # 20 - 30, affine


def test_build_bsm1_takacs_defaults_to_reference_settler(asm1):
    """build_bsm1(use_takacs=True) reproduces the BSM1 reference secondary
    settler settler1dv4 (MODELTYPE=0): the solubles are held through the layers
    and the particulate phase is lumped TSS with the feed-scaled outlet
    composition. These are dynamic-only properties (the steady state is
    invariant to them); ``per_species`` matches the BSM2 settler1dv5 instead."""
    from aquakin.plant.bsm import build_bsm1

    cl = build_bsm1(model=asm1, use_takacs=True).units["clarifier"]
    assert isinstance(cl, TakacsClarifier)
    assert cl.composition_mode == "lumped_tss"
    assert cl.soluble_holdup is True

    # The settings remain overridable for the BSM2-style settler.
    cl2 = build_bsm1(model=asm1, use_takacs=True,
                     settler_composition_mode="per_species",
                     settler_soluble_holdup=False).units["clarifier"]
    assert cl2.composition_mode == "per_species"
    assert cl2.soluble_holdup is False
