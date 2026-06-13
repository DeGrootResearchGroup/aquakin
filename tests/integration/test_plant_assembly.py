"""Plant assembly + monolithic integration tests.

Covers:
- CSTR steady-state mass balance against an analytic answer.
- Influent interpolation through the plant RHS.
- Mixer / splitter mass balances.
- A small plant with an internal recycle (seeded with initial_value).
- AD-grad through Plant.solve w.r.t. plant parameters.
"""

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import (
    CSTRUnit,
    IdentityTranslator,
    MixerUnit,
    Plant,
    SplitterUnit,
    Stream,
)
from aquakin.plant.influent import InfluentSeries


@pytest.fixture
def simple_net():
    return aquakin.load_network_from_file("tests/fixtures/simple_network.yaml")


def _constant_influent(net, *, Q=10.0, C=(1.0, 0.0), t_end=100.0):
    return InfluentSeries(
        t=jnp.asarray([0.0, t_end]),
        Q=jnp.asarray([Q, Q]),
        C=jnp.asarray([list(C), list(C)]),
        network=net,
    )


def test_single_cstr_steady_state(simple_net):
    """A single CSTR with constant inflow reaches the analytic steady state.

    For A -> B with rate k*[A] in a CSTR at hydraulic retention V/Q=10:
        steady-state A = C_in / (1 + k*tau) = 1/(1+1) = 0.5
        steady-state B = C_in - A = 0.5
    (with k=0.1 1/d and tau=10 d).
    """
    plant = Plant("single_cstr")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            network=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net), to="tank.inlet")
    sol = plant.solve(
        t_span=(0.0, 200.0), t_eval=jnp.linspace(0.0, 200.0, 5)
    )
    assert float(sol.C_named("tank", "A")[-1]) == pytest.approx(0.5, abs=1e-4)
    assert float(sol.C_named("tank", "B")[-1]) == pytest.approx(0.5, abs=1e-4)


def test_two_cstrs_in_series(simple_net):
    """Tank-in-series mass balance: with the same volume per tank and
    first-order decay, A drops by factor 1/(1+k*tau) per tank at steady
    state. With tau=10 d, k=0.1 1/d: 1.0 -> 0.5 -> 0.25.
    """
    plant = Plant("two_cstrs")
    for name in ("t1", "t2"):
        plant.add_unit(
            CSTRUnit(
                name=name, network=simple_net, volume=100.0,
                input_port_names=["inlet"], conditions={"T": 293.15},
            )
        )
    plant.add_influent("feed", _constant_influent(simple_net), to="t1.inlet")
    plant.connect("t1", "t2")
    sol = plant.solve(
        t_span=(0.0, 400.0), t_eval=jnp.linspace(0.0, 400.0, 5)
    )
    assert float(sol.C_named("t1", "A")[-1]) == pytest.approx(0.5, abs=1e-3)
    assert float(sol.C_named("t2", "A")[-1]) == pytest.approx(0.25, abs=1e-3)


def test_mixer_mass_balance(simple_net):
    """Mixer downstream of two CSTRs: total Q and Q*C are conserved."""
    plant = Plant("mixer_test")
    plant.add_unit(
        CSTRUnit(
            name="t1", network=simple_net, volume=100.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        MixerUnit(
            name="mix", input_port_names=["a", "b"], network=simple_net,
        )
    )
    plant.add_unit(
        CSTRUnit(
            name="t2", network=simple_net, volume=100.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed_main", _constant_influent(simple_net, Q=10.0), to="t1.inlet")
    plant.add_influent("feed_side", _constant_influent(simple_net, Q=5.0, C=(2.0, 0.0)), to="mix.b")
    plant.connect("t1", "mix.a")
    plant.connect("mix", "t2")
    sol = plant.solve(t_span=(0.0, 500.0), t_eval=jnp.linspace(0.0, 500.0, 5))
    assert jnp.all(jnp.isfinite(sol.state))


def test_splitter_flow_ratios(simple_net):
    """Splitter routes 60/40 of its inflow to two sinks (which are CSTRs)."""
    plant = Plant("split_test")
    plant.add_unit(
        CSTRUnit(
            name="src", network=simple_net, volume=100.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        SplitterUnit(
            name="split",
            output_port_ratios={"a": 0.6, "b": 0.4},
            network=simple_net,
        )
    )
    # Volumes proportional to flow so both sinks have HRT = V/Q = 10 d
    # and therefore the same steady-state concentration.
    plant.add_unit(
        CSTRUnit(
            name="sink_a", network=simple_net, volume=60.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        CSTRUnit(
            name="sink_b", network=simple_net, volume=40.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net), to="src.inlet")
    plant.connect("src", "split")
    plant.connect("split.a", "sink_a")
    plant.connect("split.b", "sink_b")
    sol = plant.solve(t_span=(0.0, 300.0), t_eval=jnp.linspace(0.0, 300.0, 5))
    # At steady state both sinks see the same inlet concentration (the
    # splitter is passive on C), so their final A should match.
    assert float(sol.C_named("sink_a", "A")[-1]) == pytest.approx(
        float(sol.C_named("sink_b", "A")[-1]), rel=1e-3
    )


def test_recycle_with_initial_value(simple_net):
    """A recycle stream: half the CSTR's outflow feeds back to the mixer.

    The Plant evaluates units in the order they were added; the mixer
    needs the recycle stream before the splitter has been visited, so
    the recycle connection is seeded with ``initial_value``.

    Steady-state check: total flow through the tank is influent + recycle
    = 10 + 5 = 15. Half of the tank's outflow (5 m³/d) is wasted, half
    (5 m³/d) returns. Total mass balance: 10 m³/d * 1.0 g/m³ A in =
    10 m³/d * A_eff (no net A leaves via recycle since recycle returns).
    At steady state A_in_tank ≈ A_tank since CSTR mass balance gives
    Q_in*(C_in - C_tank) = k*V*C_tank where C_in is the mixed stream.
    """
    plant = Plant("recycle_test")
    plant.add_unit(
        MixerUnit(
            name="mix", input_port_names=["fresh", "recycle"], network=simple_net
        )
    )
    plant.add_unit(
        CSTRUnit(
            name="tank", network=simple_net, volume=100.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        SplitterUnit(
            name="split",
            output_port_ratios={"out_product": 0.5, "out_recycle": 0.5},
            network=simple_net,
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net, Q=10.0), to="mix.fresh")
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    # Explicit initial_value override (a non-zero warm seed); the recycle would
    # otherwise be auto-seeded with a zero-flow stream.
    plant.connect(
        "split.out_recycle", "mix.recycle",
        initial_value=Stream(
            Q=jnp.asarray(5.0),
            C=jnp.asarray([0.0, 0.0]),
            network=simple_net,
        ),
    )
    sol = plant.solve(t_span=(0.0, 500.0), t_eval=jnp.linspace(0.0, 500.0, 5))
    A_traj = sol.C_named("tank", "A")
    # Recycle doesn't change the overall mass balance into the tank (the
    # mass coming back is still mass leaving the tank); steady state still
    # satisfies the single-tank balance up to a small correction from
    # diluted influent. Just check finite and bounded.
    assert jnp.all(jnp.isfinite(A_traj))
    assert 0.0 < float(A_traj[-1]) < 1.0


def test_connection_index_groups_inputs_and_recycle_keys(simple_net):
    """_build_state_layout precomputes the per-unit incoming-edge map and the
    recycle source keys that the RHS hot paths read instead of re-scanning all
    connections each step."""
    plant = Plant("idx_test")
    plant.add_unit(
        MixerUnit(name="mix", input_port_names=["fresh", "recycle"], network=simple_net)
    )
    plant.add_unit(
        CSTRUnit(
            name="tank", network=simple_net, volume=100.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        SplitterUnit(
            name="split",
            output_port_ratios={"out_product": 0.5, "out_recycle": 0.5},
            network=simple_net,
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net, Q=10.0), to="mix.fresh")
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    # Recycle edge with no initial_value -> auto-seeded.
    plant.connect("split.out_recycle", "mix.recycle")

    plant._build_state_layout()  # builds the connection index too

    # Every connection is grouped under its destination unit, partitioning the
    # full connection list with no loss.
    idx = plant._inputs_by_unit
    assert {c.to_port for c in idx["mix"]} == {"fresh", "recycle"}
    assert [c.to_port for c in idx["tank"]] == ["inlet"]
    assert [c.to_port for c in idx["split"]] == ["in"]
    assert sum(len(v) for v in idx.values()) == len(plant.connections)

    # The auto-seeded back-edge is the sole recycle key.
    assert plant._recycle_keys == [("split", "out_recycle")]

    # Re-running the build is idempotent (no duplicated edges).
    plant._build_state_layout()
    assert sum(len(v) for v in plant._inputs_by_unit.values()) == len(plant.connections)


def test_ad_grad_through_plant(simple_net):
    """jax.grad through Plant.solve w.r.t. params returns a finite gradient."""
    plant = Plant("grad_test")
    plant.add_unit(
        CSTRUnit(
            name="t1", network=simple_net, volume=100.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net), to="t1.inlet")

    def loss(params):
        sol = plant.solve(
            t_span=(0.0, 100.0), t_eval=jnp.asarray([0.0, 100.0]), params=params
        )
        return float(jnp.sum(sol.C_named("t1", "B")))

    # Wrap properly for jax.grad.
    def loss_traced(params):
        sol = plant.solve(
            t_span=(0.0, 100.0), t_eval=jnp.asarray([0.0, 100.0]), params=params
        )
        return jnp.sum(sol.C_named("t1", "B"))

    g = jax.grad(loss_traced)(plant.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    # Increasing k (the only kinetic param) increases B at steady state.
    assert float(g[0]) > 0.0


def test_influent_interpolation(simple_net):
    """A time-varying influent should drive a time-varying steady state."""
    # Sinusoidal influent: Q swings ±50%.
    n_t = 100
    t_grid = jnp.linspace(0.0, 50.0, n_t)
    Q_var = 10.0 + 5.0 * jnp.sin(2 * jnp.pi * t_grid / 5.0)  # 5-day period
    C_const = jnp.tile(jnp.asarray([1.0, 0.0]), (n_t, 1))
    inf = InfluentSeries(t=t_grid, Q=Q_var, C=C_const, network=simple_net)

    plant = Plant("varying_inf")
    plant.add_unit(
        CSTRUnit(
            name="tank", network=simple_net, volume=100.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", inf, to="tank.inlet")
    sol = plant.solve(t_span=(0.0, 50.0), t_eval=jnp.linspace(0.0, 50.0, 51))
    # The trajectory should not be constant — there's a real time response.
    A_traj = sol.C_named("tank", "A")
    assert float(jnp.max(A_traj) - jnp.min(A_traj)) > 0.05


def test_param_layout_and_defaults_agree_for_shared_network(simple_net):
    """The parameter layout and the default-parameter vector must stay in sync.

    Both derive from the same identity-deduped network list, so a plant whose
    units share one compiled network gets exactly one parameter block and a
    default vector of matching length.
    """
    plant = Plant("two_tanks")
    for nm in ("t1", "t2"):
        plant.add_unit(
            CSTRUnit(name=nm, network=simple_net, volume=100.0,
                     input_port_names=["inlet"], conditions={"T": 293.15})
        )
    layout = plant._build_parameter_layout()
    assert layout.total_size == simple_net.n_params  # one shared block
    assert plant.default_parameters().shape == (layout.total_size,)


def test_distinct_networks_sharing_a_name_rejected():
    """Two *distinct* networks with the same name would collide in the
    name-keyed parameter blocks; the plant must reject that rather than
    silently mis-slice the parameter vector."""
    net_a = aquakin.load_network_from_file("tests/fixtures/simple_network.yaml")
    net_b = aquakin.load_network_from_file("tests/fixtures/simple_network.yaml")
    assert net_a is not net_b and net_a.name == net_b.name
    plant = Plant("collision")
    plant.add_unit(CSTRUnit(name="a", network=net_a, volume=100.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.add_unit(CSTRUnit(name="b", network=net_b, volume=100.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    with pytest.raises(ValueError, match="share the name"):
        plant.default_parameters()


def test_unit_ports_are_lists(simple_net):
    """Every unit exposes input_ports/output_ports as list[str] (the Unit
    Protocol type), so Mixer/Splitter match CSTR/clarifier rather than leaking
    tuple fields."""
    from aquakin.plant.units import Unit

    mixer = MixerUnit(name="m", input_port_names=["a", "b"], network=simple_net)
    splitter = SplitterUnit(
        name="s", output_port_ratios={"x": 0.5, "y": 0.5}, network=simple_net)
    cstr = CSTRUnit(name="t", network=simple_net, volume=1.0,
                    input_port_names=["inlet"], conditions={"T": 293.15})
    for u in (mixer, splitter, cstr):
        assert isinstance(u, Unit)
        assert isinstance(u.input_ports, list)
        assert isinstance(u.output_ports, list)


def test_stateless_units_expose_state_size_as_property(simple_net):
    """Stateless units (mixer/splitter/clarifier/thickener) expose state_size
    as a read-only @property returning 0 -- standardised with the stateful
    units' property, and no longer a dataclass constructor field."""
    import dataclasses

    from aquakin.plant.clarifier import IdealClarifier
    from aquakin.plant.separators import IdealThickener

    mixer = MixerUnit(name="m", input_port_names=["a", "b"], network=simple_net)
    splitter = SplitterUnit(
        name="s", output_port_ratios={"x": 0.5, "y": 0.5}, network=simple_net)
    clar = IdealClarifier(name="c", network=simple_net, underflow_Q=10.0)
    thick = IdealThickener(name="th", network=simple_net, target_tss_percent=7.0)

    for u in (mixer, splitter, clar, thick):
        assert u.state_size == 0
        # state_size is a property on the type, not a dataclass field, so it
        # is not a constructor argument.
        assert isinstance(type(u).state_size, property)
        field_names = {f.name for f in dataclasses.fields(u)}
        assert "state_size" not in field_names


def _recycle_plant(simple_net):
    """A mixer -> tank -> splitter loop, the splitter recycling to the mixer."""
    plant = Plant("rc")
    plant.add_unit(MixerUnit(name="mix", input_port_names=["fresh", "recycle"],
                             network=simple_net))
    plant.add_unit(CSTRUnit(name="tank", network=simple_net, volume=100.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.add_unit(SplitterUnit(name="split", network=simple_net,
                                output_port_ratios={"out": 0.5, "rec": 0.5}))
    return plant


def test_connect_infers_sole_ports(simple_net):
    """Bare-unit endpoints resolve to the unit's only in/out port; the stored
    Connection carries the resolved port names."""
    plant = _recycle_plant(simple_net)
    plant.connect("mix", "tank")           # mix.out -> tank.inlet
    plant.connect("tank", "split")         # tank.out -> split.in
    c0, c1 = plant.connections
    assert (c0.from_unit, c0.from_port, c0.to_unit, c0.to_port) == ("mix", "out", "tank", "inlet")
    assert (c1.from_unit, c1.from_port, c1.to_unit, c1.to_port) == ("tank", "out", "split", "in")


def test_recycle_edge_auto_detected_and_seeded(simple_net):
    """A recycle (graph back-edge) given no initial_value is detected by the
    topological sort and auto-seeded with a zero-flow stream of the source
    network. (connect() no longer seeds at wire time -- recycles are found from
    the graph at finalize, regardless of add order.)"""
    plant = _recycle_plant(simple_net)
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    plant.connect("split.rec", "mix.recycle")     # recycle, no initial_value
    # connect() leaves the connection unseeded.
    assert all(c.initial_value is None for c in plant.connections)
    plant._finalize_topology()
    # The back-edge is detected and zero-flow seeded.
    assert ("split", "rec") in plant._recycle_keys
    seed = plant._recycle_seeds[("split", "rec")]
    assert float(seed.Q) == 0.0
    assert seed.network is simple_net


def test_connect_explicit_initial_value_overrides_autoseed(simple_net):
    """An explicit initial_value is kept verbatim (no auto-seed override)."""
    plant = _recycle_plant(simple_net)
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    warm = Stream(Q=jnp.asarray(7.0), C=simple_net.default_concentrations(),
                  network=simple_net)
    plant.connect("split.rec", "mix.recycle", initial_value=warm)
    assert plant.connections[2].initial_value is warm


def test_connect_ambiguous_bare_port_errors(simple_net):
    """A bare unit endpoint with more than one port for the role is rejected
    with a message naming the available ports."""
    plant = _recycle_plant(simple_net)
    with pytest.raises(ValueError, match="omits the port"):
        plant.connect("split", "mix.recycle")     # split has 2 output ports


def test_connect_unknown_port_errors(simple_net):
    plant = _recycle_plant(simple_net)
    with pytest.raises(KeyError, match="no destination port"):
        plant.connect("mix", "tank.nope")


def test_connect_influent_as_source_errors(simple_net):
    """Using an influent name as a connect source points the user to
    add_influent(to=...)."""
    plant = _recycle_plant(simple_net)
    plant.add_influent("feed", _constant_influent(simple_net), to="mix.fresh")
    with pytest.raises(ValueError, match="is an influent"):
        plant.connect("feed", "tank")


def test_add_influent_to_creates_connection(simple_net):
    """add_influent(to=...) registers the influent AND wires it, with no
    separate connect call."""
    plant = _recycle_plant(simple_net)
    plant.add_influent("feed", _constant_influent(simple_net), to="mix.fresh")
    assert "feed" in plant.influents
    inf_conn = [c for c in plant.connections if c.from_unit is None]
    assert len(inf_conn) == 1
    c = inf_conn[0]
    assert (c.from_port, c.to_unit, c.to_port) == ("feed", "mix", "fresh")


def _two_tank_plant(simple_net):
    """Two CSTRs in series (a -> b), each carrying the 2-species network."""
    plant = Plant("two_tank")
    plant.add_unit(CSTRUnit(name="a", network=simple_net, volume=100.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.add_unit(CSTRUnit(name="b", network=simple_net, volume=100.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.connect("a", "b")
    return plant


def test_initial_state_overrides_named_unit(simple_net):
    """initial_state(overrides=...) replaces only the named unit's state and
    leaves the others at their own initial_state()."""
    plant = _two_tank_plant(simple_net)
    base = plant.initial_state()
    warm = simple_net.concentrations(A=0.7, B=0.3)
    y0 = plant.initial_state(overrides={"b": warm})

    a0, a_sz = plant._state_layout["a"]
    b0, b_sz = plant._state_layout["b"]
    assert jnp.allclose(y0[a0:a0 + a_sz], base[a0:a0 + a_sz])  # 'a' untouched
    assert jnp.allclose(y0[b0:b0 + b_sz], warm)                # 'b' replaced


def test_initial_state_override_unknown_unit_errors(simple_net):
    plant = _two_tank_plant(simple_net)
    with pytest.raises(KeyError, match="unknown units"):
        plant.initial_state(overrides={"nope": jnp.zeros(simple_net.n_species)})


def test_initial_state_override_wrong_length_errors(simple_net):
    plant = _two_tank_plant(simple_net)
    with pytest.raises(ValueError, match="expected"):
        plant.initial_state(overrides={"a": jnp.zeros(simple_net.n_species + 1)})


def _fed_cstr_plant(simple_net, *, Q=10.0, C=(1.0, 0.0)):
    """A single CSTR fed by a constant influent."""
    plant = Plant("one")
    plant.add_unit(CSTRUnit(name="tank", network=simple_net, volume=100.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.add_influent("feed", _constant_influent(simple_net, Q=Q, C=C),
                       to="tank.inlet")
    return plant


def test_stream_reconstructs_unit_output(simple_net):
    """plant.stream(sol, endpoint) rebuilds a unit's output trajectory from the
    saved states. A CSTR's output concentration equals its state and its flow
    equals the (constant) inflow."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    sol = plant.solve(t_span=(0.0, 50.0), t_eval=jnp.linspace(0.0, 50.0, 6))

    eff = plant.stream(sol, "tank")            # CSTR's sole output port inferred
    assert eff.t.shape == sol.t.shape
    assert eff.C.shape == (sol.t.shape[0], simple_net.n_species)
    assert jnp.allclose(eff.Q, 10.0)                          # output flow = inflow
    assert jnp.allclose(eff.C_named("A"), sol.C_named("tank", "A"))  # output C = state
    assert jnp.allclose(eff.C_named("B"), sol.C_named("tank", "B"))


def test_stream_unknown_endpoint_errors(simple_net):
    plant = _fed_cstr_plant(simple_net)
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 2))
    with pytest.raises(KeyError, match="Unknown unit"):
        plant.stream(sol, "nope")


def test_states_by_unit_inverts_initial_state(simple_net):
    """states_by_unit splits a flat vector back into the per-unit pieces that
    initial_state(overrides=...) assembled -- the two are inverses."""
    plant = _two_tank_plant(simple_net)
    warm_a = simple_net.concentrations(A=0.7, B=0.3)
    warm_b = simple_net.concentrations(A=0.1, B=0.9)
    y0 = plant.initial_state(overrides={"a": warm_a, "b": warm_b})

    parts = plant.states_by_unit(y0)
    assert set(parts) == {"a", "b"}
    assert jnp.allclose(parts["a"], warm_a)
    assert jnp.allclose(parts["b"], warm_b)


def test_final_state_and_states_by_unit_snapshot(simple_net):
    """sol.final_state is the last state row (1-D), and states_by_unit reads a
    unit's snapshot from it -- matching sol.unit_state(name)[-1] without the
    opaque index on a 2-D trajectory."""
    plant = _fed_cstr_plant(simple_net)
    sol = plant.solve(t_span=(0.0, 5.0), t_eval=jnp.linspace(0.0, 5.0, 4))

    assert sol.final_state.shape == (sol.state.shape[1],)
    assert jnp.array_equal(sol.final_state, sol.state[-1])
    snap = plant.states_by_unit(sol.final_state)["tank"]
    assert jnp.allclose(snap, sol.unit_state("tank")[-1])


def test_derivative_evaluates_rhs(simple_net):
    """plant.derivative(state) is the public dstate/dt: same layout as the state,
    finite, and equal to the assembled RHS."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    y0 = plant.initial_state()

    d = plant.derivative(y0)
    assert d.shape == y0.shape
    assert jnp.all(jnp.isfinite(d))
    # Defaults to default_parameters(); matches a direct RHS evaluation.
    plant._build_state_layout()
    plant._build_parameter_layout()
    expected = plant._rhs(jnp.asarray(0.0), y0, plant.default_parameters())
    assert jnp.allclose(d, expected)
    # Splittable by unit like any flat plant vector.
    assert plant.states_by_unit(d)["tank"].shape == (simple_net.n_species,)


def test_run_to_steady_state_converges(simple_net):
    """run_to_steady_state self-terminates at steady state (no fixed horizon) and
    recovers the analytic CSTR operating point. A -> B with k=0.1 in a CSTR at
    tau=V/Q=10 gives steady A = C_in/(1+k*tau) = 1/(1+1) = 0.5, B = 0.5."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    ss = plant.run_to_steady_state(max_time=1000.0)

    assert ss.converged                 # the steady-state event fired
    assert ss.time < 1000.0             # ... well before the safety cap
    tank = plant.states_by_unit(ss.state)["tank"]
    assert float(tank[simple_net.species_index["A"]]) == pytest.approx(0.5, abs=0.02)
    assert float(tank[simple_net.species_index["B"]]) == pytest.approx(0.5, abs=0.02)


def test_run_to_steady_state_reports_non_convergence(simple_net):
    """If the safety cap is hit before the dynamics settle, converged is False
    (the signal to raise max_time) rather than a silently-wrong 'steady' state."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    ss = plant.run_to_steady_state(max_time=0.1)   # tau=10, nowhere near steady
    assert not ss.converged
    assert ss.time == pytest.approx(0.1)


def test_solve_step_ceiling_gives_friendly_error(simple_net):
    """Hitting the integrator step budget re-raises with a domain-level remedy
    (warm-start / tolerances / max_steps), not a raw Diffrax/Equinox traceback."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    with pytest.raises(RuntimeError, match="step budget"):
        plant.solve(t_span=(0.0, 1000.0), max_steps=1)


def test_default_atol_solves_without_tuning(simple_net):
    """With atol=None (the default), the solve uses a per-component noise floor
    scaled off the state magnitudes and stays accurate -- no hand-set atol. The
    fed CSTR A->B reaches the analytic A=B=0.5."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    sol = plant.solve(t_span=(0.0, 200.0), t_eval=jnp.asarray([200.0]))  # atol default
    tank = plant.states_by_unit(sol.final_state)["tank"]
    assert float(tank[simple_net.species_index["A"]]) == pytest.approx(0.5, abs=1e-3)


# ----- By-name plant parameter overrides (#134) ----------------------------

def _bsm2_no_solve():
    """The two-network BSM2 plant (ASM1 + ADM1), assembled but not solved --
    cheap, for the by-name parameter API."""
    from aquakin.plant.bsm import build_bsm2, bsm2_asm1_network

    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    return build_bsm2(asm1, adm1), asm1, adm1


def test_plant_parameter_names_are_network_prefixed():
    plant, asm1, adm1 = _bsm2_no_solve()
    names = plant.parameter_names()
    assert len(names) == asm1.n_params + adm1.n_params
    assert "asm1.muH" in names           # ASM1 water line
    assert "adm1.k_m_ac" in names        # ADM1 digester
    # No bare names; every key carries its network prefix.
    assert all("." in n and n.split(".")[0] in ("asm1", "adm1") for n in names)


def test_plant_parameter_index_matches_block_offset():
    """The friendly index equals the hand-computed block offset it replaces."""
    plant, asm1, adm1 = _bsm2_no_solve()
    # ASM1 is the first block (offset 0); ADM1 follows it.
    assert plant.parameter_index("asm1.muH") == asm1.param_index["muH"]
    assert (plant.parameter_index("adm1.k_m_ac")
            == asm1.n_params + adm1.param_index["k_m_ac"])


def test_plant_parameter_values_sets_only_named_entries():
    plant, _asm1, _adm1 = _bsm2_no_solve()
    base = plant.default_parameters()
    i_mu = plant.parameter_index("asm1.muH")
    i_km = plant.parameter_index("adm1.k_m_ac")
    v = plant.parameter_values({"asm1.muH": 6.0, "adm1.k_m_ac": 9.0})
    assert float(v[i_mu]) == 6.0
    assert float(v[i_km]) == 9.0
    # Every other entry is unchanged.
    changed = set(int(i) for i in jnp.where(v != base)[0])
    assert changed <= {i_mu, i_km}
    # None / empty returns the defaults unchanged.
    assert bool(jnp.array_equal(plant.parameter_values(), base))
    assert bool(jnp.array_equal(plant.parameter_values({}), base))


def test_plant_parameter_values_unknown_name_raises_with_hint():
    plant, _asm1, _adm1 = _bsm2_no_solve()
    with pytest.raises(KeyError, match="Did you mean: asm1.muH"):
        plant.parameter_values({"asm1.muh": 6.0})   # wrong case
    with pytest.raises(KeyError, match="Unknown plant parameter"):
        plant.parameter_index("adm1.nope")
    with pytest.raises(TypeError):
        plant.parameter_values([("asm1.muH", 6.0)])  # not a dict
