"""IFAS / MBBR plant unit: a CSTR bulk coupled to a depth-resolved biofilm.

The unit wires the existing :class:`~aquakin.integrate.biofilm.BiofilmReactor`
into the flowsheet: its state is the bulk concentration plus the biofilm depth
profile, and its RHS is the biofilm diffusion--reaction core with the plant's
bulk convection + aeration added on the bulk compartment. The headline check is
that an IFAS tank removes more substrate (and nitrifies more) than an
equivalent CSTR -- the point of the intensification retrofit.
"""

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
    base = dict(name="r", model=asm1, volume=1000.0, input_port_names=["in"],
                specific_surface_area=500.0, fill_fraction=0.4,
                biofilm_thickness=5e-4, n_layers=4, conditions={"T": 293.15})
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
        IFASUnit("r", asm1, 1000.0, ["in"], specific_surface_area=500.0,
                 fill_fraction=0.4, biofilm_thickness=5e-4, conditions={})


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
    assert float(y[0, xbh]) == float(asm1.default_concentrations()[xbh])   # bulk
    assert float(y[1, xbh]) == 3000.0                                      # layers


def test_compute_outputs_is_the_bulk(asm1):
    u = _ifas(asm1, aeration=Aeration(kla=240.0))
    y0 = u.initial_state()
    s = Stream(Q=jnp.asarray(500.0), C=asm1.default_concentrations(), model=asm1)
    out = u.compute_outputs(jnp.asarray(0.0), y0, {"in": s}, asm1.default_parameters())
    bulk = y0.reshape(u._n_comp, asm1.n_species)[0]
    assert jnp.allclose(out["out"].C, bulk)
    assert float(out["out"].Q) == 500.0


def test_rhs_is_finite(asm1):
    u = _ifas(asm1, aeration=Aeration(kla=240.0),
              biofilm_initial=asm1.concentrations({"XB_H": 3000.0}))
    s = Stream(Q=jnp.asarray(500.0),
               C=asm1.concentrations({"SS": 60.0, "SNH": 25.0, "SO": 2.0}),
               model=asm1)
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
    p = Plant("t")
    p.add_unit(unit)
    p.add_influent(
        "feed",
        asm1.influent({"SS": 60.0, "SNH": 25.0, "XB_H": 50.0, "SO": 2.0}, Q=500.0),
        to="r.in")
    return p


@pytest.mark.slow
def test_ifas_removes_more_than_equivalent_cstr(asm1):
    """An IFAS tank (same volume + aeration) removes more soluble COD and
    nitrifies more than a plain CSTR -- the attached biomass adds capacity."""
    cstr = CSTRUnit("r", asm1, 1000.0, ["in"], conditions={"T": 293.15},
                    aeration=Aeration(kla=600.0))
    ifas = _ifas(asm1, aeration=Aeration(kla=600.0),
                 biofilm_initial=asm1.concentrations({"XB_H": 3000.0, "XB_A": 150.0}))

    sc = _plant(cstr, asm1).run_to_steady_state()
    pi = _plant(ifas, asm1)
    si = pi.run_to_steady_state()
    assert sc.converged and si.converged

    ec = _plant(cstr, asm1).stream(sc.solution, "r.out")
    ei = pi.stream(si.solution, "r.out")
    ss_c, ss_i = float(ec.C_named("SS")[-1]), float(ei.C_named("SS")[-1])
    nh_c, nh_i = float(ec.C_named("SNH")[-1]), float(ei.C_named("SNH")[-1])
    assert ss_i < ss_c        # more COD removal
    assert nh_i < nh_c        # more nitrification
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
    sol = p.solve(t_span=(0.0, 0.5), t_eval=jnp.array([0.25, 0.5]),
                  y0=p.initial_state())
    reactors = _as_reactors(p)
    assert "r" in reactors                       # the IFAS tank is a reactor
    kla = _kla_history(p, sol, p.default_parameters(), reactors)
    assert kla.shape == (2, 1)
    assert np.all(np.isfinite(np.asarray(kla))) and float(kla[0, 0]) == 600.0


@pytest.mark.slow
def test_grad_through_ifas_plant_is_finite(asm1):
    """jax.grad flows through a plant containing an IFAS unit (the biofilm
    diffusion--reaction core is differentiable end to end)."""
    ifas = _ifas(asm1, aeration=Aeration(kla=600.0),
                 biofilm_initial=asm1.concentrations({"XB_H": 3000.0}))
    p = _plant(ifas, asm1)
    y0 = p.run_to_steady_state().state
    muH = asm1.param_index["muH"]

    def loss(mu):
        params = asm1.default_parameters().at[muH].set(mu)
        sol = p.solve(t_span=(0.0, 1.0), y0=y0, params=params)
        return jnp.sum(sol.state)

    g = jax.grad(loss)(4.0)
    assert jnp.isfinite(g)
