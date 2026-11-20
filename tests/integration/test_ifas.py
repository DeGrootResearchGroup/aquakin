"""IFAS / MBBR plant unit: a CSTR bulk coupled to a depth-resolved biofilm.

The unit wires the existing :class:`~aquakin.integrate.biofilm.BiofilmReactor`
into the flowsheet: its state is the bulk concentration plus the biofilm depth
profile, and its RHS is the biofilm diffusion--reaction core with the plant's
bulk convection + aeration added on the bulk compartment. The headline check is
that an IFAS tank removes more substrate (and nitrifies more) than an
equivalent CSTR -- the point of the intensification retrofit.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import Aeration, CSTRUnit, IFASUnit, MBBRUnit, Plant
from aquakin.plant.ifas import _default_biofilm_fixed_mask
from aquakin.plant.streams import Stream


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


def _ifas(asm1, **kw):
    base = dict(
        name="r",
        model=asm1,
        volume=1000.0,
        input_port_names=["in"],
        specific_surface_area=500.0,
        fill_fraction=0.4,
        biofilm_thickness=5e-4,
        n_layers=4,
        conditions={"T": 293.15},
    )
    base.update(kw)
    return IFASUnit(**base)


# --- unit-protocol surface (fast) -------------------------------------------


def test_state_size_and_ports(asm1):
    u = _ifas(asm1, n_layers=4)
    assert u.state_size == (4 + 1) * asm1.n_species
    assert u.input_ports == ["in"]
    assert u.output_ports == ["out"]
    assert u.initial_state().shape == (u.state_size,)


def test_mbbr_is_an_alias():
    assert MBBRUnit is IFASUnit


def test_validation_errors(asm1):
    with pytest.raises(ValueError, match="fill_fraction"):
        _ifas(asm1, fill_fraction=1.5)
    with pytest.raises(ValueError, match="missing required condition"):
        IFASUnit(
            "r",
            asm1,
            1000.0,
            ["in"],
            specific_surface_area=500.0,
            fill_fraction=0.4,
            biofilm_thickness=5e-4,
            conditions={},
        )


def test_default_fixed_mask_freezes_biomass_not_substrate(asm1):
    """The mature-biofilm default holds the biomass + inert structure (XB_H,
    XB_A, XP, XI) fixed but leaves the hydrolysis substrates (XS, XND) dynamic --
    freezing XS would make it a spurious SS source."""
    from aquakin.integrate.biofilm import _default_soluble_mask

    mask = _default_biofilm_fixed_mask(asm1, _default_soluble_mask(asm1))
    frozen = {s for s, f in zip(asm1.species, list(map(bool, mask))) if f}
    assert frozen == {"XI", "XB_H", "XB_A", "XP"}
    assert "XS" not in frozen and "XND" not in frozen


def test_initial_state_seeds_bulk_and_layers(asm1):
    u = _ifas(asm1, biofilm_initial=asm1.concentrations({"XB_H": 3000.0}))
    y = u.initial_state().reshape(u._n_comp, asm1.n_species)
    xbh = asm1.species_index["XB_H"]
    assert float(y[0, xbh]) == float(asm1.default_concentrations()[xbh])  # bulk
    assert float(y[1, xbh]) == 3000.0  # layers


def test_compute_outputs_is_the_bulk(asm1):
    u = _ifas(asm1, aeration=Aeration(kla=240.0))
    y0 = u.initial_state()
    s = Stream(Q=jnp.asarray(500.0), C=asm1.default_concentrations(), model=asm1)
    out = u.compute_outputs(jnp.asarray(0.0), y0, {"in": s}, asm1.default_parameters())
    bulk = y0.reshape(u._n_comp, asm1.n_species)[0]
    assert jnp.allclose(out["out"].C, bulk)
    assert float(out["out"].Q) == 500.0


def test_rhs_is_finite(asm1):
    u = _ifas(
        asm1, aeration=Aeration(kla=240.0), biofilm_initial=asm1.concentrations({"XB_H": 3000.0})
    )
    s = Stream(
        Q=jnp.asarray(500.0),
        C=asm1.concentrations({"SS": 60.0, "SNH": 25.0, "SO": 2.0}),
        model=asm1,
    )
    d = u.rhs(jnp.asarray(0.0), u.initial_state(), {"in": s}, asm1.default_parameters())
    assert d.shape == (u.state_size,)
    assert jnp.all(jnp.isfinite(d))


def test_closed_loop_aeration_requires_signal(asm1):
    """A DO-setpoint IFAS reads a control signal (the plant auto-wires the
    controller from the Aeration spec, exactly as for a CSTR)."""
    u = _ifas(asm1, aeration=Aeration(do_setpoint=2.0))
    assert u.required_signals == ("_aer_r_kla",)


# --- the headline integration check (slow: real plant solves) ---------------


def _plant(unit, asm1):
    p = Plant("t", time_unit="d")
    p.add_unit(unit)
    p.add_influent(
        "feed",
        asm1.influent({"SS": 60.0, "SNH": 25.0, "XB_H": 50.0, "SO": 2.0}, Q=500.0),
        to="r.in",
    )
    return p


@pytest.mark.slow
def test_ifas_removes_more_than_equivalent_cstr(asm1):
    """An IFAS tank (same volume + aeration) removes more soluble COD and
    nitrifies more than a plain CSTR -- the attached biomass adds capacity."""
    cstr = CSTRUnit(
        "r", asm1, 1000.0, ["in"], conditions={"T": 293.15}, aeration=Aeration(kla=600.0)
    )
    ifas = _ifas(
        asm1,
        aeration=Aeration(kla=600.0),
        biofilm_initial=asm1.concentrations({"XB_H": 3000.0, "XB_A": 150.0}),
    )

    sc = _plant(cstr, asm1).run_to_steady_state()
    pi = _plant(ifas, asm1)
    si = pi.run_to_steady_state()
    assert sc.converged and si.converged

    ec = _plant(cstr, asm1).stream(sc.solution, "r.out")
    ei = pi.stream(si.solution, "r.out")
    ss_c, ss_i = float(ec.C_named("SS")[-1]), float(ei.C_named("SS")[-1])
    nh_c, nh_i = float(ec.C_named("SNH")[-1]), float(ei.C_named("SNH")[-1])
    assert ss_i < ss_c  # more COD removal
    assert nh_i < nh_c  # more nitrification
    assert np.isfinite(ss_i) and ss_i >= 0.0


def test_ifas_plant_evaluates_kla_history(asm1):
    """The BSM evaluator must score a plant containing an IFAS tank.

    ``_as_reactors`` collects the IFAS unit (it carries an ``aeration`` spec) and
    ``_kla_history`` then reads its ``_controlled_kla`` / ``_kla_vec`` -- which the
    shared ``AerationUnit`` mixin supplies, so the previously-missing accessors no
    longer raise ``AttributeError`` mid-evaluation. Open-loop, so the constant-kLa
    tiling path is exercised (and ``_controlled_kla`` is read for the controlled
    check)."""
    from aquakin.plant.bsm.evaluation import _as_reactors, _kla_history

    p = _plant(_ifas(asm1, aeration=Aeration(kla=600.0)), asm1)
    sol = p.solve(t_span=(0.0, 0.5), t_eval=jnp.array([0.25, 0.5]), y0=p.initial_state())
    reactors = _as_reactors(p)
    assert "r" in reactors  # the IFAS tank is a reactor
    kla = _kla_history(p, sol, p.default_parameters(), reactors)
    assert kla.shape == (2, 1)
    assert np.all(np.isfinite(np.asarray(kla))) and float(kla[0, 0]) == 600.0


@pytest.mark.slow
def test_grad_through_ifas_plant_is_finite(asm1):
    """jax.grad flows through a plant containing an IFAS unit (the biofilm
    diffusion--reaction core is differentiable end to end)."""
    ifas = _ifas(
        asm1, aeration=Aeration(kla=600.0), biofilm_initial=asm1.concentrations({"XB_H": 3000.0})
    )
    p = _plant(ifas, asm1)
    y0 = p.run_to_steady_state().state
    muH = asm1.param_index["muH"]

    def loss(mu):
        params = asm1.default_parameters().at[muH].set(mu)
        sol = p.solve(t_span=(0.0, 1.0), y0=y0, params=params)
        return jnp.sum(sol.state)

    g = jax.grad(loss)(4.0)
    assert jnp.isfinite(g)


# --- layer resolution and the steady-state criterion -------------------------


def test_layer_resolution_reports_the_worst_single_layer_drop(asm1):
    """``layer_resolution`` measures how much of a soluble's profile falls across
    one layer -- the quantity the first-order discretization error tracks.

    Checked on hand-built profiles so the metric is pinned independently of any
    solve: a flat profile is perfectly resolved, and one that collapses inside a
    single layer is not.
    """
    u = _ifas(asm1, n_layers=4)
    ss, so = asm1.species_index["SS"], asm1.species_index["SO"]

    flat = jnp.zeros((5, asm1.n_species)).at[:, ss].set(10.0).at[:, so].set(2.0)
    assert u.layer_resolution(flat)["SS"] == pytest.approx(0.0)

    # All of SS consumed between the bulk and the first layer.
    collapsed = jnp.zeros((5, asm1.n_species)).at[0, ss].set(10.0)
    assert u.layer_resolution(collapsed)["SS"] == pytest.approx(1.0)

    # Gradual decay is well resolved; the flat species is unaffected either way.
    gradual = jnp.zeros((5, asm1.n_species))
    gradual = gradual.at[:, ss].set(jnp.asarray([10.0, 9.0, 8.0, 7.0, 6.0]))
    gradual = gradual.at[:, so].set(2.0)
    report = u.layer_resolution(gradual)
    assert report["SS"] == pytest.approx(0.1)
    assert report["SO"] == pytest.approx(0.0)
    # Particulates are not reported -- they never diffuse.
    assert "XB_H" not in report


def test_layer_resolution_accepts_a_flat_state(asm1):
    """The report reads a flat unit state, as ``states_by_unit`` hands it back."""
    u = _ifas(asm1, n_layers=4)
    flat = u.initial_state()
    assert set(u.layer_resolution(flat)) == {s for s in asm1.species if not s.startswith("X")}


def test_norm_steady_state_warning_threshold_and_message(asm1):
    """The guard fires on a misleading verdict and names the offending state.

    Exercised directly on the helper so the threshold and the message are pinned
    without a multi-minute solve. ``NORM_RESIDUAL_WARN_FACTOR * ss_rtol`` is the
    line: a healthy plant sits orders of magnitude inside it.
    """
    from aquakin.plant.plant import NORM_RESIDUAL_WARN_FACTOR

    plant = _plant(_ifas(asm1, n_layers=4, aeration=Aeration(kla=600.0)), asm1)
    plant._build_state_layout()
    state = plant.initial_state()
    floor = jnp.maximum(jnp.abs(state), 1e-6)
    ss_rtol = 1e-3

    quiet = NORM_RESIDUAL_WARN_FACTOR * ss_rtol / 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plant._warn_if_norm_test_overstates(state, jnp.zeros_like(state), floor, quiet, ss_rtol)

    # One badly unsteady entry is enough, however large the rest of the state.
    worst = int(jnp.argmax(jnp.abs(state)))
    F = jnp.zeros_like(state).at[worst].set(5.0 * jnp.abs(state[worst]))
    loud = NORM_RESIDUAL_WARN_FACTOR * ss_rtol * 10.0
    with pytest.warns(RuntimeWarning, match="reported convergence") as caught:
        plant._warn_if_norm_test_overstates(state, F, floor, loud, ss_rtol)
    message = str(caught[0].message)
    assert plant._describe_state_index(worst) in message
    assert "scaled_residual" in message


@pytest.mark.slow
def test_norm_steady_state_warns_when_frozen_biomass_hides_a_moving_bulk():
    """A held-fixed biofilm inventory defeats the norm steady-state test.

    The attached biomass contributes ``n_layers * density`` to ``||y||`` but
    nothing to ``||dy/dt||``, so it inflates the tolerance the *bulk* is held to
    and the solve stops while the bulk is still moving. The solver cannot refuse
    to return -- but it must not call that converged without saying so.

    The configuration is a carrier-dominated tank whose bulk substrate is driven
    low by the biofilm: a small residual in absolute terms is then a large one
    relative to the bulk's own concentration, which is exactly the gap the norm
    test cannot see.
    """
    import os

    model = aquakin.load_model_from_file(
        os.path.join(os.path.dirname(__file__), "..", "fixtures", "bm1_biofilm.yaml")
    )
    unit = IFASUnit(
        "reactor",
        model,
        volume=1.25e-3,
        input_port_names=["in"],
        specific_surface_area=80.0,
        fill_fraction=1.0,
        biofilm_thickness=5e-4,
        n_layers=32,
        diffusivity=jnp.array([1e-4, 2e-4, 0.0]),
        boundary_layer=1e-9,
        conditions={"T": 293.15},
        aeration=Aeration(kla=1.0e6, do_sat=10.0, species="SO2"),
        biofilm_initial=model.concentrations({"XH": 1e4, "SS": 30.0, "SO2": 10.0}),
    )
    plant = Plant("t", time_unit="d")
    plant.add_unit(unit)
    plant.add_influent("feed", model.influent({"SS": 30.0, "SO2": 10.0}, Q=0.02), to="reactor.in")

    with pytest.warns(RuntimeWarning, match="reported convergence"):
        result = plant.run_to_steady_state()
    assert result.converged  # the verdict the warning is about
    assert result.residual > 0.1  # ... and why it is misleading

    # The componentwise criterion is the documented remedy and does not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        tight = plant.run_to_steady_state(
            criterion="scaled_residual", tol=1e-7, scale_floor=1e-3, max_time=200.0
        )
    assert tight.residual < 0.1


# --- a biofilm that grows (moving boundary) ---------------------------------


def _bm1_model():
    import os

    return aquakin.load_model_from_file(
        os.path.join(os.path.dirname(__file__), "..", "fixtures", "bm1_biofilm.yaml")
    )


def _growing(model, **kw):
    base = dict(
        name="r",
        model=model,
        volume=1.0,
        input_port_names=["in"],
        specific_surface_area=10.0,
        fill_fraction=1.0,
        biofilm_thickness=3.0e-5,
        n_layers=24,
        diffusivity=jnp.array([1e-4, 2e-4, 0.0]),
        boundary_layer=1e-9,
        conditions={"T": 293.15},
        aeration=Aeration(kla=1.0e6, do_sat=10.0, species="SO2"),
        dynamic_thickness=True,
        biofilm_density=1.0e4,
        detachment_rate=2.0e5,
        detachment_law="quadratic",
        biofilm_initial=model.concentrations({"XH": 1.0e4, "SS": 30.0, "SO2": 10.0}),
    )
    base.update(kw)
    return IFASUnit(**base)


def test_dynamic_thickness_adds_one_state_entry():
    """``L_f`` becomes a state: the last entry, named and seeded."""
    model = _bm1_model()
    u = _growing(model, n_layers=8)
    assert u.state_size == (8 + 1) * model.n_species + 1
    assert u.state_names()[-1] == "L_f"
    assert len(u.state_names()) == u.state_size
    y0 = u.initial_state()
    assert y0.shape == (u.state_size,)
    assert float(y0[-1]) == pytest.approx(3.0e-5)
    # The fixed-thickness unit is untouched: no extra entry, no L_f.
    fixed = _growing(
        model, n_layers=8, dynamic_thickness=False, biofilm_density=None, detachment_rate=0.0
    )
    assert fixed.state_size == (8 + 1) * model.n_species
    assert "L_f" not in fixed.state_names()


def test_dynamic_thickness_requires_a_coherent_configuration():
    model = _bm1_model()
    with pytest.raises(ValueError, match="biofilm_density"):
        _growing(model, biofilm_density=None)
    with pytest.raises(ValueError, match="detachment_law"):
        _growing(model, detachment_law="exponential")
    with pytest.raises(ValueError, match="detachment_rate"):
        _growing(model, detachment_rate=-1.0)
    # Freezing the attached biomass contradicts growing it.
    frozen = jnp.asarray([False, False, True])
    with pytest.raises(ValueError, match="cannot combine dynamic_thickness"):
        _growing(model, biofilm_fixed_mask=frozen)


def test_dynamic_thickness_warns_on_a_seed_off_the_stated_density():
    model = _bm1_model()
    with pytest.warns(RuntimeWarning, match="incompressible"):
        _growing(model, biofilm_initial=model.concentrations({"XH": 50.0}))


@pytest.mark.slow
@pytest.mark.parametrize("detach_rate", [1.0e5, 2.0e5, 4.0e5])
def test_growing_biofilm_settles_at_the_thickness_the_closure_predicts(detach_rate):
    """A fully penetrated film balances growth against quadratic detachment.

    With the substrate abundant everywhere inside, the net specific production
    is uniform, so ``dL_f/dt = mu_net*L_f - lambda*L_f**2`` and the steady
    thickness is ``mu_net/lambda``. A flow-dominated tank holds the bulk at the
    influent composition, which fixes ``mu_net`` without a second unknown.
    """
    model = _bm1_model()
    unit = _growing(model, detachment_rate=detach_rate)
    plant = Plant("g", time_unit="d")
    plant.add_unit(unit)
    plant.add_influent("feed", model.influent({"SS": 30.0, "SO2": 10.0}, Q=5000.0), to="r.in")

    sol = plant.solve(t_span=(0.0, 30.0), y0=plant.initial_state(), t_eval=jnp.asarray([30.0]))
    rows, thickness = unit._split(jnp.asarray(np.asarray(sol.state)[-1]))
    rows = np.asarray(rows)

    ss = rows[0, model.species_index["SS"]]
    o2 = rows[0, model.species_index["SO2"]]
    mu_net = 6.0 * ss / (4.0 + ss) * o2 / (0.2 + o2) - 0.4
    assert float(thickness) == pytest.approx(mu_net / detach_rate, rel=0.01)


@pytest.mark.slow
def test_growing_biofilm_stays_incompressible():
    """The film holds its stated density at every depth, which is the premise.

    The displacement velocity is derived from incompressibility, so if the
    advection were wrong the total particulate concentration would drift away
    from ``biofilm_density`` -- the sharpest check on the transport.
    """
    model = _bm1_model()
    unit = _growing(model)
    plant = Plant("g", time_unit="d")
    plant.add_unit(unit)
    plant.add_influent("feed", model.influent({"SS": 30.0, "SO2": 10.0}, Q=5000.0), to="r.in")

    sol = plant.solve(
        t_span=(0.0, 20.0), y0=plant.initial_state(), t_eval=jnp.linspace(1.0, 20.0, 5)
    )
    for row in np.asarray(sol.state):
        rows, thickness = unit._split(jnp.asarray(row))
        layers = np.asarray(rows)[1:, model.species_index["XH"]]
        np.testing.assert_allclose(layers, 1.0e4, rtol=1e-6)
        assert float(thickness) > 0.0


@pytest.mark.slow
def test_a_starved_biofilm_thins_instead_of_holding_its_thickness():
    """Decay outrunning growth must shrink the film -- the case a fixed
    thickness cannot represent at all."""
    model = _bm1_model()
    unit = _growing(model, biofilm_thickness=1.0e-4)
    plant = Plant("g", time_unit="d")
    plant.add_unit(unit)
    # No substrate in the feed: lysis runs, growth does not.
    plant.add_influent("feed", model.influent({"SS": 0.0, "SO2": 10.0}, Q=5000.0), to="r.in")

    sol = plant.solve(t_span=(0.0, 10.0), y0=plant.initial_state(), t_eval=jnp.asarray([0.0, 10.0]))
    start, end = np.asarray(sol.state)[0, -1], np.asarray(sol.state)[-1, -1]
    assert end < 0.5 * start
    assert end > 0.0


def test_the_tank_declares_its_inventory_to_the_mass_balance(asm1):
    """The plant mass balance must see the tank's liquid *and* its biofilm.

    An IFAS state is a depth profile, not a single concentration vector, so
    without an inventory contract the balance skips the unit entirely -- the
    bulk liquid as well as the attached biomass -- and the shortfall appears as
    an imbalance the size of the whole tank.
    """
    from aquakin.plant.units import ComponentInventoryUnit

    u = _ifas(asm1, n_layers=4, biofilm_initial=asm1.concentrations({"XB_H": 3000.0}))
    assert isinstance(u, ComponentInventoryUnit)

    table = aquakin.composition_table(asm1)
    cod = np.asarray([table[s].get("COD", 0.0) for s in asm1.species])
    got = u.component_inventory(u.initial_state(), {"COD": cod}, asm1.default_parameters())["COD"]

    rows = np.asarray(u.initial_state()).reshape(5, asm1.n_species)
    film_volume = u._area_per_volume * u.biofilm_thickness / u.n_layers
    expected = u.volume * (rows[0] @ cod + film_volume * sum(rows[k] @ cod for k in range(1, 5)))
    assert got == pytest.approx(expected, rel=1e-12)
    assert got > 0.0


def test_a_growing_biofilm_reports_the_inventory_of_its_current_thickness():
    """The attached inventory follows ``L_f``, not the configured seed value."""
    model = _bm1_model()
    u = _growing(model, n_layers=8)
    table = {s: {"COD": 1.0} for s in model.species}
    cod = np.asarray([table[s]["COD"] for s in model.species])

    thin = u.initial_state()
    thick = thin.at[-1].set(float(thin[-1]) * 3.0)
    params = model.default_parameters()
    inv_thin = u.component_inventory(thin, {"COD": cod}, params)["COD"]
    inv_thick = u.component_inventory(thick, {"COD": cod}, params)["COD"]

    # Only the biofilm term scales; the bulk contribution is unchanged.
    bulk_only = u.volume * float(np.asarray(thin).reshape(-1)[: model.n_species] @ cod)
    assert (inv_thick - bulk_only) == pytest.approx(3.0 * (inv_thin - bulk_only), rel=1e-12)
