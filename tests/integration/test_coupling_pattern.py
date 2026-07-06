"""Per-component structural coupling contract (issue #388).

Every stateful unit emits its structural Jacobian sparsity via
``coupling_pattern()`` (the :class:`CouplingAware` ABC), so the plant assembles
the colored-Jacobian pattern from the equations rather than a single-operating-
point probe -- which goes stale on the nonlinear couplings (Monod kinetics, the
Takacs settling velocity, the ASM<->ADM interface branches) that switch on only
as the influent drives the plant off that point. These tests pin the contract
(shapes, ABC enforcement, the superset property) and -- slow -- that the assembled
plant pattern is a structural superset of the dense Jacobian along a trajectory.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.coupling import CouplingAware, CouplingPattern, ad_union


@pytest.fixture(scope="module")
def asm1_net():
    return aquakin.load_model("asm1")


# --------------------------------------------------------------------------
# The contract (fast, no solve).
# --------------------------------------------------------------------------

def test_cstr_coupling_pattern():
    from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
    from aquakin.plant.cstr import CSTRUnit

    net = aquakin.load_model("asm1")
    unit = CSTRUnit(name="t", model=net, volume=1000.0,
                    input_port_names=["inlet"], conditions={"T": 293.15})
    cp = unit.coupling_pattern()
    n = net.n_species
    # self = the kinetics' structural pattern; inlet = the dilution diagonal.
    assert cp.self_pattern.shape == (n, n)
    assert cp.inlet_pattern.shape == (n, n)
    assert np.array_equal(cp.inlet_pattern, np.eye(n, dtype=bool))
    assert np.array_equal(cp.self_pattern, structural_sparsity_pattern(net))


def test_stateless_coupling_pattern_is_empty():
    from aquakin.plant.mixer import MixerUnit

    net = aquakin.load_model("asm1")
    mixer = MixerUnit(name="m", model=net, input_port_names=["a", "b"])
    cp = mixer.coupling_pattern()
    assert cp.self_pattern.shape == (0, 0)
    assert cp.inlet_pattern is None


def test_settler_coupling_pattern_is_superset_of_dense_jacobian():
    """The settler's AD-derived pattern must be a structural superset of its
    dense RHS Jacobian at several solids profiles (it is exact at any one)."""
    from aquakin.plant.streams import Stream
    from aquakin.plant.takacs import TakacsClarifier

    net = aquakin.load_model("asm1")
    s = TakacsClarifier(name="s", model=net, area=1500.0, height=4.0,
                        underflow_Q=1.8e4, soluble_holdup=True)
    cp = s.coupling_pattern()
    m = s.state_size
    assert cp.self_pattern.shape == (m, m)
    assert cp.inlet_pattern.shape == (m, net.n_species)

    # dense self-Jacobian at a few random positive states must be covered.
    base_C = np.maximum(np.abs(np.asarray(net.default_concentrations())), 1e-3)
    inlet = {s.input_port: Stream(Q=jnp.asarray(2.0e4),
                                  C=jnp.asarray(base_C), model=net)}
    fj = jax.jit(lambda x: jax.jacfwd(lambda z: s.rhs(jnp.asarray(0.0), z,
                                                      inlet, None))(x))
    rng = np.random.default_rng(1)
    base = np.maximum(np.abs(np.asarray(s.initial_state())), 1e-3)
    for _ in range(8):
        y = jnp.asarray(base * 10.0 ** rng.uniform(-2, 2, size=base.shape))
        J = np.asarray(fj(y))
        active = np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-300)
        assert not (active & ~cp.self_pattern).any()      # superset


# --------------------------------------------------------------------------
# The reactive non-BSM units (MBR / SBR / IFAS): contract + superset (#390-392).
# --------------------------------------------------------------------------

_PARTS = ["XI", "XS", "XB_H", "XB_A", "XP", "XND"]


def _superset_holds(unit, make_inputs, *, times=(0.0,), seed=0, n=8):
    """The unit's ``self_pattern`` covers its dense RHS self-Jacobian at diverse
    positive states (and times, for a time-phased unit) -- the superset property."""
    cp = unit.coupling_pattern()
    net = unit.model
    base = np.maximum(np.abs(np.asarray(unit.initial_state())), 1e-3)
    params = net.default_parameters()
    rng = np.random.default_rng(seed)
    for t in times:
        fj = jax.jit(lambda y, _t=t: jax.jacfwd(
            lambda z: unit.rhs(jnp.asarray(_t), z, make_inputs(), params))(y))
        for _ in range(n):
            y = jnp.asarray(base * 10.0 ** rng.uniform(-2, 2, size=base.shape))
            J = np.asarray(fj(y))
            active = np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-300)
            assert not (active & ~cp.self_pattern).any()


def _inlet_is_species_dilution(cp, n):
    """The inlet block is the species dilution diagonal on the head (bulk) rows
    and couples no other state -- the convective ``(Q/V)(C_in - C)`` on species."""
    ip = cp.inlet_pattern
    assert ip.shape[1] == n
    assert np.array_equal(ip[:n, :], np.eye(n, dtype=bool))
    assert not ip[n:, :].any()


def test_mbr_coupling_pattern(asm1_net):
    from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
    from aquakin.plant import Aeration, MBRUnit
    from aquakin.plant.streams import Stream

    net = asm1_net
    unit = MBRUnit("mbr", net, volume=1000.0, aeration=Aeration(kla=240.0),
                   waste_flow=20.0, particulate_species=_PARTS,
                   membrane_area=500.0, fouling_rate=1e-3, fouling_relax=1e-2,
                   conditions={"T": 293.15})
    cp = unit.coupling_pattern()
    n, m = net.n_species, unit.state_size
    assert cp.self_pattern.shape == (m, m)
    # species block = the kinetics' structural pattern; R_f only self-couples.
    assert np.array_equal(cp.self_pattern[:n, :n], structural_sparsity_pattern(net))
    assert cp.self_pattern[n, n]
    assert not cp.self_pattern[n, :n].any() and not cp.self_pattern[:n, n].any()
    _inlet_is_species_dilution(cp, n)

    base_C = jnp.asarray(np.maximum(np.abs(np.asarray(
        net.default_concentrations())), 1e-3))
    _superset_holds(unit, lambda: {
        "feed": Stream(Q=jnp.asarray(500.0), C=base_C, model=net)})


def test_sbr_coupling_pattern(asm1_net):
    from aquakin.plant.sbr import SBRPhase, SBRUnit
    from aquakin.plant.settling import InterfaceSettling, LayeredSettling
    from aquakin.plant.streams import Stream

    net = asm1_net
    phases = [SBRPhase("fill", 1.0, feed=True), SBRPhase("react", 2.0, kla=200.0),
              SBRPhase("settle", 0.5, settle=True),
              SBRPhase("decant", 0.5, decant=True), SBRPhase("idle", 0.5)]
    base_C = jnp.asarray(np.maximum(np.abs(np.asarray(
        net.default_concentrations())), 1e-3))
    for model in (InterfaceSettling(v_settle=400.0, area=200.0),
                  LayeredSettling(n_layers=5, v_settle=400.0, area=200.0)):
        unit = SBRUnit("sbr", net, phases, full_volume=1000.0, feed_flow=500.0,
                       decant_flow=800.0, settling=model,
                       particulate_species=_PARTS, conditions={"T": 293.15})
        cp = unit.coupling_pattern()
        n, m = net.n_species, unit.state_size
        assert cp.self_pattern.shape == (m, m)
        _inlet_is_species_dilution(cp, n)
        # union over a representative time in each phase exercises fill/decant/settle.
        times = [s + 0.5 * float(phases[p].duration)
                 for p, s in enumerate(unit._phase_starts)]
        _superset_holds(unit, lambda: {
            "feed": Stream(Q=jnp.asarray(500.0), C=base_C, model=net)},
            times=times, n=6)


def test_ifas_coupling_pattern(asm1_net):
    from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
    from aquakin.plant import Aeration, IFASUnit
    from aquakin.plant.streams import Stream

    net = asm1_net
    unit = IFASUnit("ifas", net, volume=1000.0, input_port_names=["in"],
                    specific_surface_area=500.0, fill_fraction=0.5,
                    biofilm_thickness=8e-4, n_layers=4,
                    aeration=Aeration(kla=200.0), conditions={"T": 293.15})
    cp = unit.coupling_pattern()
    n, m = net.n_species, unit.state_size
    assert cp.self_pattern.shape == (m, m)
    # each compartment's diagonal sub-block covers the reaction kinetics.
    kin = structural_sparsity_pattern(net)
    assert (cp.self_pattern[:n, :n] & kin == kin).all()       # bulk: full kinetics
    _inlet_is_species_dilution(cp, n)

    base_C = jnp.asarray(np.maximum(np.abs(np.asarray(
        net.default_concentrations())), 1e-3))
    _superset_holds(unit, lambda: {
        "in": Stream(Q=jnp.asarray(500.0), C=base_C, model=net)})


def test_couplingaware_abc_requires_coupling_pattern():
    # A CouplingAware subclass that does not implement coupling_pattern cannot
    # be instantiated (the abstractmethod is unsatisfied).
    class Incomplete(CouplingAware):
        pass

    with pytest.raises(TypeError):
        Incomplete()

    class Complete(CouplingAware):
        def coupling_pattern(self):
            return CouplingPattern(self_pattern=np.zeros((0, 0), dtype=bool))

    Complete()                                            # instantiates fine


def test_reactive_units_are_coupling_aware():
    from aquakin.plant.cstr import CSTRUnit
    from aquakin.plant.digester import ADM1DigesterUnit
    from aquakin.plant.ifas import IFASUnit
    from aquakin.plant.mbr import MBRUnit
    from aquakin.plant.sbr import SBRUnit
    from aquakin.plant.takacs import TakacsClarifier

    for cls in (CSTRUnit, ADM1DigesterUnit, TakacsClarifier,
                MBRUnit, SBRUnit, IFASUnit):
        assert issubclass(cls, CouplingAware)


def test_ad_union_recovers_a_known_pattern():
    # A map y -> [y0*y1, y2] has Jacobian rows {0,1} and {2}; ad_union over
    # diverse states recovers exactly that structural pattern.
    def jac(x):
        return jax.jacfwd(lambda z: jnp.array([z[0] * z[1], z[2]]))(x)

    P = ad_union(jac, np.array([1.0, 1.0, 1.0]), n_states=16)
    expected = np.array([[True, True, False], [False, False, True]])
    assert np.array_equal(P, expected)


# --------------------------------------------------------------------------
# The assembled plant pattern (slow: builds a plant + solves for states).
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_plant_structural_pattern_superset_over_trajectory_bsm1():
    """The assembled plant structural pattern (∪ the IC probe) must cover the
    dense plant Jacobian's strong couplings along a short trajectory -- in
    particular leave *no* within-unit coupling missing (the staleness #388 fixes).
    """
    from aquakin.integrate.colored_jacobian import jacobian_sparsity_pattern
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
    from aquakin.plant.influent import InfluentSeries
    from aquakin.plant.plant import default_atol

    asm1 = aquakin.load_model("asm1")
    p = build_bsm1(model=asm1, use_takacs=True)
    C0 = asm1.concentrations({
        "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
        "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0})
    inf = InfluentSeries(t=jnp.array([0.0, 100.0]), Q=jnp.full((2,), BSM1_Q_AVG),
                         C=jnp.tile(C0, (2, 1)), model=asm1)
    p.add_influent("feed", inf, to="inlet_mix.fresh")
    y0 = bsm1_warm_start(p)
    params = p.default_parameters()
    p.solve(t_span=(0.0, 0.02), t_eval=jnp.array([0.02]), params=params, y0=y0,
            rtol=1e-3, atol=1e-2,
            integrator=aquakin.IntegratorConfig(max_steps=100_000))

    t0 = jnp.asarray(0.0)
    rmap = p._recycle._maybe_recycle_map(t0, p._split_state(y0), params)
    rhs_y0 = lambda y: p._rhs(t0, y, params, recycle_map=rmap)
    probe = jacobian_sparsity_pattern(rhs_y0, y0) > 0
    P = probe | p._colored._structural_plant_pattern(coupling_mask=probe)

    # unit block map
    n = y0.shape[0]
    unit_of = np.full(n, -1)
    for k, (_, (off, size)) in enumerate(p._state_layout.items()):
        unit_of[off:off + size] = k

    te = jnp.linspace(0.0, 5.0, 11)
    atol = default_atol(y0, p.initial_state())
    ys = np.asarray(p.solve(
        t_span=(0.0, 5.0), t_eval=te, params=params, y0=y0,
        rtol=1e-4, atol=atol,
        integrator=aquakin.IntegratorConfig(max_steps=2_000_000)).state)
    denseJ = jax.jit(lambda tt, y: jax.jacfwd(lambda z: p._rhs(
        tt, z, params, recycle_map=p._recycle._maybe_recycle_map(
            jnp.asarray(tt), p._split_state(z), params)))(y))
    missing_within = 0
    for i in range(0, len(te), 3):
        J = np.asarray(denseJ(float(te[i]), jnp.asarray(ys[i])))
        active = np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-300)
        miss = active & ~P
        mi, mj = np.nonzero(miss)
        missing_within += int((unit_of[mi] == unit_of[mj]).sum())
    assert missing_within == 0          # no stale within-unit coupling


def _build_single_unit_plant(kind):
    """A one-reactive-unit plant (MBR / SBR / IFAS) + a constant influent, for the
    assembled-structural-pattern trajectory check. Returns (plant, params, span)."""
    from aquakin.plant import Aeration, IFASUnit, MBRUnit
    from aquakin.plant.influent import InfluentSeries
    from aquakin.plant.plant import Plant
    from aquakin.plant.sbr import SBRPhase, SBRUnit
    from aquakin.plant.settling import InterfaceSettling

    net = aquakin.load_model("asm1")
    inf = InfluentSeries.constant(net, SS=300.0, SNH=40.0, XS=200.0, XB_H=80.0,
                                  Q=1000.0)
    p = Plant(f"{kind}_plant")
    if kind == "mbr":
        p.add_unit(MBRUnit("u", net, volume=1000.0, aeration=Aeration(kla=240.0),
                           waste_flow=20.0, particulate_species=_PARTS,
                           membrane_area=500.0, fouling_rate=1e-3,
                           fouling_relax=1e-2, conditions={"T": 293.15}))
        p.add_influent("feed", inf, to="u.feed")
        span = 2.0
    elif kind == "ifas":
        p.add_unit(IFASUnit("u", net, volume=1000.0, input_port_names=["in"],
                           specific_surface_area=500.0, fill_fraction=0.5,
                           biofilm_thickness=8e-4, n_layers=4,
                           aeration=Aeration(kla=200.0), conditions={"T": 293.15}))
        p.add_influent("feed", inf, to="u.in")
        span = 2.0
    else:  # sbr -- a full cycle, so the trajectory crosses every phase
        phases = [SBRPhase("fill", 0.4, feed=True),
                  SBRPhase("react", 0.6, kla=240.0),
                  SBRPhase("settle", 0.2, settle=True),
                  SBRPhase("decant", 0.2, decant=True), SBRPhase("idle", 0.1)]
        p.add_unit(SBRUnit("u", net, phases, full_volume=1000.0, feed_flow=2000.0,
                          decant_flow=2000.0,
                          settling=InterfaceSettling(v_settle=400.0, area=200.0),
                          particulate_species=_PARTS, conditions={"T": 293.15}))
        p.add_influent("feed", inf, to="u.feed")
        span = 1.5
    return p, p.default_parameters(), span


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["mbr", "sbr", "ifas"])
def test_single_unit_plant_no_within_unit_coupling_missing(kind):
    """The assembled structural pattern (∪ the IC probe) leaves no within-unit
    coupling missing along a trajectory for a plant built on MBR / SBR / IFAS --
    the per-unit ``coupling_pattern()`` of #390-392 wired through the assembler
    (single unit, so "within-unit" is every coupling in the plant Jacobian)."""
    from aquakin.integrate.colored_jacobian import jacobian_sparsity_pattern
    from aquakin.plant.plant import default_atol

    p, params, span = _build_single_unit_plant(kind)
    y0 = p.initial_state()

    # Solve first: this finalizes the state / parameter layouts the probe and the
    # dense-Jacobian RHS below rely on.
    te = jnp.linspace(0.0, span, 9)
    atol = default_atol(y0, p.initial_state())
    ys = np.asarray(p.solve(
        t_span=(0.0, span), t_eval=te, params=params, y0=y0,
        rtol=1e-4, atol=atol,
        integrator=aquakin.IntegratorConfig(max_steps=2_000_000)).state)

    t0 = jnp.asarray(0.0)
    rmap = p._recycle._maybe_recycle_map(t0, p._split_state(y0), params)
    rhs_y0 = lambda y: p._rhs(t0, y, params, recycle_map=rmap)
    probe = jacobian_sparsity_pattern(rhs_y0, y0) > 0
    P = probe | p._colored._structural_plant_pattern(coupling_mask=probe)
    denseJ = jax.jit(lambda tt, y: jax.jacfwd(lambda z: p._rhs(
        tt, z, params, recycle_map=p._recycle._maybe_recycle_map(
            jnp.asarray(tt), p._split_state(z), params)))(y))
    missing = 0
    for i in range(len(te)):
        J = np.asarray(denseJ(float(te[i]), jnp.asarray(ys[i])))
        active = np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-300)
        missing += int((active & ~P).sum())
    assert missing == 0
