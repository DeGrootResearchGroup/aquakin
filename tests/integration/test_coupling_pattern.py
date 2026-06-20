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


# --------------------------------------------------------------------------
# The contract (fast, no solve).
# --------------------------------------------------------------------------

def test_cstr_coupling_pattern():
    from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
    from aquakin.plant.cstr import CSTRUnit

    net = aquakin.load_network("asm1")
    unit = CSTRUnit(name="t", network=net, volume=1000.0,
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

    net = aquakin.load_network("asm1")
    mixer = MixerUnit(name="m", network=net, input_port_names=["a", "b"])
    cp = mixer.coupling_pattern()
    assert cp.self_pattern.shape == (0, 0)
    assert cp.inlet_pattern is None


def test_settler_coupling_pattern_is_superset_of_dense_jacobian():
    """The settler's AD-derived pattern must be a structural superset of its
    dense RHS Jacobian at several solids profiles (it is exact at any one)."""
    from aquakin.plant.streams import Stream
    from aquakin.plant.takacs import TakacsClarifier

    net = aquakin.load_network("asm1")
    s = TakacsClarifier(name="s", network=net, area=1500.0, height=4.0,
                        underflow_Q=1.8e4, soluble_holdup=True)
    cp = s.coupling_pattern()
    m = s.state_size
    assert cp.self_pattern.shape == (m, m)
    assert cp.inlet_pattern.shape == (m, net.n_species)

    # dense self-Jacobian at a few random positive states must be covered.
    base_C = np.maximum(np.abs(np.asarray(net.default_concentrations())), 1e-3)
    inlet = {s.input_port: Stream(Q=jnp.asarray(2.0e4),
                                  C=jnp.asarray(base_C), network=net)}
    fj = jax.jit(lambda x: jax.jacfwd(lambda z: s.rhs(jnp.asarray(0.0), z,
                                                      inlet, None))(x))
    rng = np.random.default_rng(1)
    base = np.maximum(np.abs(np.asarray(s.initial_state())), 1e-3)
    for _ in range(8):
        y = jnp.asarray(base * 10.0 ** rng.uniform(-2, 2, size=base.shape))
        J = np.asarray(fj(y))
        active = np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-300)
        assert not (active & ~cp.self_pattern).any()      # superset


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
    from aquakin.plant.takacs import TakacsClarifier

    for cls in (CSTRUnit, ADM1DigesterUnit, TakacsClarifier):
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

    asm1 = aquakin.load_network("asm1")
    p = build_bsm1(network=asm1, use_takacs=True)
    C0 = asm1.concentrations({
        "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
        "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0})
    inf = InfluentSeries(t=jnp.array([0.0, 100.0]), Q=jnp.full((2,), BSM1_Q_AVG),
                         C=jnp.tile(C0, (2, 1)), network=asm1)
    p.add_influent("feed", inf, to="inlet_mix.fresh")
    y0 = bsm1_warm_start(p)
    params = p.default_parameters()
    p.solve(t_span=(0.0, 0.02), t_eval=jnp.array([0.02]), params=params, y0=y0,
            rtol=1e-3, atol=1e-2, max_steps=100_000)

    t0 = jnp.asarray(0.0)
    rmap = p._maybe_recycle_map(t0, p._split_state(y0), params)
    rhs_y0 = lambda y: p._rhs(t0, y, params, recycle_map=rmap)
    probe = jacobian_sparsity_pattern(rhs_y0, y0) > 0
    P = probe | p._structural_plant_pattern(coupling_mask=probe)

    # unit block map
    n = y0.shape[0]
    unit_of = np.full(n, -1)
    for k, (_, (off, size)) in enumerate(p._state_layout.items()):
        unit_of[off:off + size] = k

    te = jnp.linspace(0.0, 5.0, 11)
    atol = default_atol(y0, p.initial_state())
    ys = np.asarray(p.solve(t_span=(0.0, 5.0), t_eval=te, params=params, y0=y0,
                            rtol=1e-4, atol=atol, max_steps=2_000_000).state)
    denseJ = jax.jit(lambda tt, y: jax.jacfwd(lambda z: p._rhs(
        tt, z, params, recycle_map=p._maybe_recycle_map(
            jnp.asarray(tt), p._split_state(z), params)))(y))
    missing_within = 0
    for i in range(0, len(te), 3):
        J = np.asarray(denseJ(float(te[i]), jnp.asarray(ys[i])))
        active = np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-300)
        miss = active & ~P
        mi, mj = np.nonzero(miss)
        missing_within += int((unit_of[mi] == unit_of[mj]).sum())
    assert missing_within == 0          # no stale within-unit coupling
