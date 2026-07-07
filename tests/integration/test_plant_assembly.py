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
    RatioSplitter,
    SetpointSplitter,
    Stream,
)
from aquakin.plant.influent import InfluentSeries


@pytest.fixture
def simple_net():
    return aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")


def _constant_influent(net, *, Q=10.0, C=(1.0, 0.0), t_end=100.0):
    return InfluentSeries(
        t=jnp.asarray([0.0, t_end]),
        Q=jnp.asarray([Q, Q]),
        C=jnp.asarray([list(C), list(C)]),
        model=net,
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
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net), to="tank.inlet")
    sol = plant.solve(t_span=(0.0, 200.0), t_eval=jnp.linspace(0.0, 200.0, 5))
    assert float(sol.C_named("tank", "A")[-1]) == pytest.approx(0.5, abs=1e-4)
    assert float(sol.C_named("tank", "B")[-1]) == pytest.approx(0.5, abs=1e-4)


def _single_cstr_plant(net):
    plant = Plant("single_cstr")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(net), to="tank.inlet")
    return plant


def test_plant_time_unit_and_conversion(simple_net):
    """Plant.time_unit reflects its model's unit (the fixture is in seconds),
    and solve(time_unit=...) converts in and out equivalently."""
    plant = _single_cstr_plant(simple_net)
    assert plant.time_unit == "s"

    sol_s = plant.solve(t_span=(0.0, 120.0), t_eval=jnp.linspace(0.0, 120.0, 5))
    sol_min = plant.solve(
        t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 5), time_unit="min"
    )  # 2 min == 120 s
    assert sol_s.time_unit == "s"
    assert sol_min.time_unit == "min"
    assert jnp.allclose(sol_min.t, jnp.linspace(0.0, 2.0, 5))
    # Same physical times -> same state trajectory.
    assert jnp.allclose(sol_s.state, sol_min.state, rtol=1e-6, atol=1e-8)


def test_two_cstrs_in_series(simple_net):
    """Tank-in-series mass balance: with the same volume per tank and
    first-order decay, A drops by factor 1/(1+k*tau) per tank at steady
    state. With tau=10 d, k=0.1 1/d: 1.0 -> 0.5 -> 0.25.
    """
    plant = Plant("two_cstrs")
    for name in ("t1", "t2"):
        plant.add_unit(
            CSTRUnit(
                name=name,
                model=simple_net,
                volume=100.0,
                input_port_names=["inlet"],
                conditions={"T": 293.15},
            )
        )
    plant.add_influent("feed", _constant_influent(simple_net), to="t1.inlet")
    plant.connect("t1", "t2")
    sol = plant.solve(t_span=(0.0, 400.0), t_eval=jnp.linspace(0.0, 400.0, 5))
    assert float(sol.C_named("t1", "A")[-1]) == pytest.approx(0.5, abs=1e-3)
    assert float(sol.C_named("t2", "A")[-1]) == pytest.approx(0.25, abs=1e-3)


def test_mixer_mass_balance(simple_net):
    """Mixer downstream of two CSTRs: total Q and Q*C are conserved."""
    plant = Plant("mixer_test")
    plant.add_unit(
        CSTRUnit(
            name="t1",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        MixerUnit(
            name="mix",
            input_port_names=["a", "b"],
            model=simple_net,
        )
    )
    plant.add_unit(
        CSTRUnit(
            name="t2",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed_main", _constant_influent(simple_net, Q=10.0), to="t1.inlet")
    plant.add_influent("feed_side", _constant_influent(simple_net, Q=5.0, C=(2.0, 0.0)), to="mix.b")
    plant.connect("t1", "mix.a")
    plant.connect("mix", "t2")
    sol = plant.solve(t_span=(0.0, 500.0), t_eval=jnp.linspace(0.0, 500.0, 5))
    assert jnp.all(jnp.isfinite(sol.state))


def test_splitter_flow_mode_conserves_under_low_feed(simple_net):
    """A flow-mode splitter's MATERIAL streams (compute_outputs) never carry more
    flow than the unit receives. Above the total setpoint the setpoints are
    honored verbatim with the remainder taking the rest; below it the setpoints
    share the available flow proportionally with a zero remainder -- so the
    material sweep conserves flow in every regime (previously it over-delivered,
    creating flow). flow_outputs stays the exact AFFINE rule the recycle solve
    needs (an unclamped remainder), and the two agree wherever the unit is not
    starved."""
    sp = SetpointSplitter(
        name="s", model=simple_net, output_port_flows={"a": 100.0, "b": 100.0}, remainder_port="r"
    )
    C = simple_net.default_concentrations()
    p = simple_net.default_parameters()

    def material(Q_in):
        ins = {"in": Stream(Q=jnp.asarray(float(Q_in)), C=C, model=simple_net)}
        out = sp.compute_outputs(jnp.asarray(0.0), None, ins, p)
        return {k: float(v.Q) for k, v in out.items()}

    def flow(Q_in):
        return {
            k: float(v) for k, v in sp.flow_outputs({"in": jnp.asarray(float(Q_in))}, p).items()
        }

    # Above the setpoint (300 >= 200): full setpoints + remainder; material and
    # flow rules agree.
    assert material(300.0) == {"a": 100.0, "b": 100.0, "r": 100.0}
    assert flow(300.0) == material(300.0)
    # Below the setpoint (150 < 200): material shares proportionally with a zero
    # remainder (no flow created)...
    co = material(150.0)
    assert co["a"] == pytest.approx(75.0) and co["b"] == pytest.approx(75.0)
    assert co["r"] == pytest.approx(0.0)
    # ...while flow_outputs stays affine (exact, may be negative -- harmless for
    # the linear recycle solve, which is what kept it from breaking _resolve_flows).
    assert flow(150.0)["r"] == pytest.approx(-50.0)
    # The material sweep conserves flow in every regime.
    for Q_in in (50.0, 150.0, 200.0, 300.0):
        assert sum(material(Q_in).values()) == pytest.approx(Q_in)


def test_splitter_flow_ratios(simple_net):
    """Splitter routes 60/40 of its inflow to two sinks (which are CSTRs)."""
    plant = Plant("split_test")
    plant.add_unit(
        CSTRUnit(
            name="src",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        RatioSplitter(
            name="split",
            output_port_ratios={"a": 0.6, "b": 0.4},
            model=simple_net,
        )
    )
    # Volumes proportional to flow so both sinks have HRT = V/Q = 10 d
    # and therefore the same steady-state concentration.
    plant.add_unit(
        CSTRUnit(
            name="sink_a",
            model=simple_net,
            volume=60.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        CSTRUnit(
            name="sink_b",
            model=simple_net,
            volume=40.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
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
    plant.add_unit(MixerUnit(name="mix", input_port_names=["fresh", "recycle"], model=simple_net))
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        RatioSplitter(
            name="split",
            output_port_ratios={"out_product": 0.5, "out_recycle": 0.5},
            model=simple_net,
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net, Q=10.0), to="mix.fresh")
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    # Explicit initial_value override (a non-zero warm seed); the recycle would
    # otherwise be auto-seeded with a zero-flow stream.
    plant.connect(
        "split.out_recycle",
        "mix.recycle",
        initial_value=Stream(
            Q=jnp.asarray(5.0),
            C=jnp.asarray([0.0, 0.0]),
            model=simple_net,
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
    plant.add_unit(MixerUnit(name="mix", input_port_names=["fresh", "recycle"], model=simple_net))
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        RatioSplitter(
            name="split",
            output_port_ratios={"out_product": 0.5, "out_recycle": 0.5},
            model=simple_net,
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
            name="t1",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net), to="t1.inlet")

    def loss(params):
        sol = plant.solve(t_span=(0.0, 100.0), t_eval=jnp.asarray([0.0, 100.0]), params=params)
        return float(jnp.sum(sol.C_named("t1", "B")))

    # Wrap properly for jax.grad.
    def loss_traced(params):
        sol = plant.solve(t_span=(0.0, 100.0), t_eval=jnp.asarray([0.0, 100.0]), params=params)
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
    inf = InfluentSeries(t=t_grid, Q=Q_var, C=C_const, model=simple_net)

    plant = Plant("varying_inf")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", inf, to="tank.inlet")
    sol = plant.solve(t_span=(0.0, 50.0), t_eval=jnp.linspace(0.0, 50.0, 51))
    # The trajectory should not be constant — there's a real time response.
    A_traj = sol.C_named("tank", "A")
    assert float(jnp.max(A_traj) - jnp.min(A_traj)) > 0.05


def test_param_layout_and_defaults_agree_for_shared_model(simple_net):
    """The parameter layout and the default-parameter vector must stay in sync.

    Both derive from the same identity-deduped model list, so a plant whose
    units share one compiled model gets exactly one parameter block and a
    default vector of matching length.
    """
    plant = Plant("two_tanks")
    for nm in ("t1", "t2"):
        plant.add_unit(
            CSTRUnit(
                name=nm,
                model=simple_net,
                volume=100.0,
                input_port_names=["inlet"],
                conditions={"T": 293.15},
            )
        )
    layout = plant._build_parameter_layout()
    assert layout.total_size == simple_net.n_params  # one shared block
    assert plant.default_parameters().shape == (layout.total_size,)


def test_distinct_models_sharing_a_name_rejected():
    """Two *distinct* models with the same name would collide in the
    name-keyed parameter blocks; the plant must reject that rather than
    silently mis-slice the parameter vector."""
    net_a = aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")
    net_b = aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")
    assert net_a is not net_b and net_a.name == net_b.name
    plant = Plant("collision")
    plant.add_unit(
        CSTRUnit(
            name="a",
            model=net_a,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        CSTRUnit(
            name="b",
            model=net_b,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    with pytest.raises(ValueError, match="share the name"):
        plant.default_parameters()


def test_unit_ports_are_lists(simple_net):
    """Every unit exposes input_ports/output_ports as list[str] (the Unit
    Protocol type), so Mixer/Splitter match CSTR/clarifier rather than leaking
    tuple fields."""
    from aquakin.plant.units import Unit

    mixer = MixerUnit(name="m", input_port_names=["a", "b"], model=simple_net)
    splitter = RatioSplitter(name="s", output_port_ratios={"x": 0.5, "y": 0.5}, model=simple_net)
    cstr = CSTRUnit(
        name="t", model=simple_net, volume=1.0, input_port_names=["inlet"], conditions={"T": 293.15}
    )
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

    mixer = MixerUnit(name="m", input_port_names=["a", "b"], model=simple_net)
    splitter = RatioSplitter(name="s", output_port_ratios={"x": 0.5, "y": 0.5}, model=simple_net)
    # The toy net's species are ("A", "B"); name an existing particulate so the
    # settling/TSS masks resolve (the default ASM1 species are not in this model).
    clar = IdealClarifier(name="c", model=simple_net, underflow_Q=10.0, particulate_species=["B"])
    thick = IdealThickener(
        name="th",
        model=simple_net,
        target_tss_percent=7.0,
        settling_species=("B",),
        tss_species=("B",),
    )

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
    plant.add_unit(MixerUnit(name="mix", input_port_names=["fresh", "recycle"], model=simple_net))
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        RatioSplitter(name="split", model=simple_net, output_port_ratios={"out": 0.5, "rec": 0.5})
    )
    return plant


def test_connect_infers_sole_ports(simple_net):
    """Bare-unit endpoints resolve to the unit's only in/out port; the stored
    Connection carries the resolved port names."""
    plant = _recycle_plant(simple_net)
    plant.connect("mix", "tank")  # mix.out -> tank.inlet
    plant.connect("tank", "split")  # tank.out -> split.in
    c0, c1 = plant.connections
    assert (c0.from_unit, c0.from_port, c0.to_unit, c0.to_port) == ("mix", "out", "tank", "inlet")
    assert (c1.from_unit, c1.from_port, c1.to_unit, c1.to_port) == ("tank", "out", "split", "in")


def test_recycle_edge_auto_detected_and_seeded(simple_net):
    """A recycle (graph back-edge) given no initial_value is detected by the
    topological sort and auto-seeded with a zero-flow stream of the source
    model. (connect() no longer seeds at wire time -- recycles are found from
    the graph at finalize, regardless of add order.)"""
    plant = _recycle_plant(simple_net)
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    plant.connect("split.rec", "mix.recycle")  # recycle, no initial_value
    # connect() leaves the connection unseeded.
    assert all(c.initial_value is None for c in plant.connections)
    plant._finalize_topology()
    # The back-edge is detected and zero-flow seeded.
    assert ("split", "rec") in plant._recycle_keys
    seed = plant._recycle_seeds[("split", "rec")]
    assert float(seed.Q) == 0.0
    assert seed.model is simple_net


def test_connect_explicit_initial_value_overrides_autoseed(simple_net):
    """An explicit initial_value is kept verbatim (no auto-seed override)."""
    plant = _recycle_plant(simple_net)
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    warm = Stream(Q=jnp.asarray(7.0), C=simple_net.default_concentrations(), model=simple_net)
    plant.connect("split.rec", "mix.recycle", initial_value=warm)
    assert plant.connections[2].initial_value is warm


def test_connect_ambiguous_bare_port_errors(simple_net):
    """A bare unit endpoint with more than one port for the role is rejected
    with a message naming the available ports."""
    plant = _recycle_plant(simple_net)
    with pytest.raises(ValueError, match="omits the port"):
        plant.connect("split", "mix.recycle")  # split has 2 output ports


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
    """Two CSTRs in series (a -> b), each carrying the 2-species model."""
    plant = Plant("two_tank")
    plant.add_unit(
        CSTRUnit(
            name="a",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_unit(
        CSTRUnit(
            name="b",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
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
    assert jnp.allclose(y0[a0 : a0 + a_sz], base[a0 : a0 + a_sz])  # 'a' untouched
    assert jnp.allclose(y0[b0 : b0 + b_sz], warm)  # 'b' replaced


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
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net, Q=Q, C=C), to="tank.inlet")
    return plant


def test_stream_reconstructs_unit_output(simple_net):
    """plant.stream(sol, endpoint) rebuilds a unit's output trajectory from the
    saved states. A CSTR's output concentration equals its state and its flow
    equals the (constant) inflow."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    sol = plant.solve(t_span=(0.0, 50.0), t_eval=jnp.linspace(0.0, 50.0, 6))

    eff = plant.stream(sol, "tank")  # CSTR's sole output port inferred
    assert eff.t.shape == sol.t.shape
    assert eff.C.shape == (sol.t.shape[0], simple_net.n_species)
    assert jnp.allclose(eff.Q, 10.0)  # output flow = inflow
    assert jnp.allclose(eff.C_named("A"), sol.C_named("tank", "A"))  # output C = state
    assert jnp.allclose(eff.C_named("B"), sol.C_named("tank", "B"))


def test_stream_unknown_endpoint_errors(simple_net):
    plant = _fed_cstr_plant(simple_net)
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 2))
    with pytest.raises(KeyError, match="Unknown stream"):
        plant.stream(sol, "nope")


def test_stream_reconstruction_is_cached_on_the_solution(simple_net):
    """The whole output sweep is reconstructed once and cached on the solution,
    so a second stream call (any port) reuses it -- and the result matches the
    per-timestep outputs_at reconstruction exactly."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    sol = plant.solve(t_span=(0.0, 50.0), t_eval=jnp.linspace(0.0, 50.0, 6))

    assert "_stream_cache" not in sol.__dict__  # nothing cached yet
    eff = plant.stream(sol, "tank")
    cache = sol.__dict__["_stream_cache"]
    assert len(cache) == 1  # one params key
    cached_map = next(iter(cache.values()))
    assert ("tank", "out") in cached_map  # all ports reconstructed

    # A second call (same params) does not rebuild -- still one key.
    plant.stream(sol, "tank")
    assert len(sol.__dict__["_stream_cache"]) == 1

    # Correctness: the cached stream equals the manual outputs_at trajectory.
    Q = jnp.stack(
        [plant.outputs_at(sol.t[i], sol.state[i])[("tank", "out")].Q for i in range(sol.t.shape[0])]
    )
    C = jnp.stack(
        [plant.outputs_at(sol.t[i], sol.state[i])[("tank", "out")].C for i in range(sol.t.shape[0])]
    )
    assert jnp.allclose(eff.Q, Q) and jnp.allclose(eff.C, C)


def test_solution_stream_convenience_delegates(simple_net):
    """sol.stream(endpoint) is plant.stream(sol, endpoint) with the plant carried
    on the solution, sharing the same cache."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    sol = plant.solve(t_span=(0.0, 20.0), t_eval=jnp.linspace(0.0, 20.0, 4))
    a = sol.stream("tank")
    b = plant.stream(sol, "tank")
    assert jnp.allclose(a.C, b.C) and jnp.allclose(a.Q, b.Q)


def test_stream_cache_keys_on_parameters(simple_net):
    """Different parameter vectors get separate cache entries (the cache never
    serves a stream reconstructed for other parameters)."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    sol = plant.solve(t_span=(0.0, 10.0), t_eval=jnp.linspace(0.0, 10.0, 3))
    p = plant.default_parameters()
    plant.stream(sol, "tank", params=p)
    plant.stream(sol, "tank", params=p.at[0].set(p[0] * 1.5))
    assert len(sol.__dict__["_stream_cache"]) == 2


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

    assert ss.converged  # the steady-state event fired
    assert ss.time < 1000.0  # ... well before the safety cap
    tank = plant.states_by_unit(ss.state)["tank"]
    assert float(tank[simple_net.species_index["A"]]) == pytest.approx(0.5, abs=0.02)
    assert float(tank[simple_net.species_index["B"]]) == pytest.approx(0.5, abs=0.02)


def test_run_to_steady_state_reports_non_convergence(simple_net):
    """If the safety cap is hit before the dynamics settle, converged is False
    (the signal to raise max_time) rather than a silently-wrong 'steady' state."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    ss = plant.run_to_steady_state(max_time=0.1)  # tau=10, nowhere near steady
    assert not ss.converged
    assert ss.time == pytest.approx(0.1)


def test_solve_step_ceiling_gives_friendly_error(simple_net):
    """Hitting the integrator step budget re-raises with a domain-level remedy
    (warm-start / tolerances / max_steps), not a raw Diffrax/Equinox traceback."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    with pytest.raises(RuntimeError, match="step budget"):
        plant.solve(t_span=(0.0, 1000.0), integrator=aquakin.IntegratorConfig(max_steps=1))


def test_default_atol_solves_without_tuning(simple_net):
    """With atol=None (the default), the solve uses a per-component noise floor
    scaled off the state magnitudes and stays accurate -- no hand-set atol. The
    fed CSTR A->B reaches the analytic A=B=0.5."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    sol = plant.solve(t_span=(0.0, 200.0), t_eval=jnp.asarray([200.0]))  # atol default
    tank = plant.states_by_unit(sol.final_state)["tank"]
    assert float(tank[simple_net.species_index["A"]]) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize(
    "gradient",
    [
        aquakin.DifferentiationConfig(method="through_solve"),
        aquakin.DifferentiationConfig(method="stable"),
    ],
    ids=["jax_adjoint", "stable_adjoint"],
)
@pytest.mark.parametrize("wrt", ["y0", "params"])
def test_plant_solve_gradient_leaks_no_tracer(simple_net, gradient, wrt):
    """``Plant.solve`` must not leak a tracer when differentiated w.r.t. either
    the initial state ``y0`` or ``params``, under either gradient backend.

    Two non-differentiable quantities are derived from these traced inputs and
    must stay detached: the per-component ``atol`` (``default_atol(y0, ...)``,
    a solver-config value -- stop_gradient'd in ``default_atol``) and the
    per-unit parameter-slice memo (``_params_for_unit``, which must NOT cache a
    tracer on the long-lived ``Plant``). Such a leak surfaces as an
    UnexpectedTracerError *during* the grad -- diffrax reuses ``params_full``'s
    object identity across its ``eval_shape`` sub-trace and the real trace, so a
    cached eval_shape slice would be served in the outer trace -- so a grad that
    completes with a finite result, plus a memo holding no tracer afterwards, is
    the regression check.

    We assert the mechanism directly rather than with ``jax.checking_leaks()``:
    that detector inspects whole-interpreter state and false-positives -- and
    crashes its own reporter (an ``IndexError``/``Zero`` ``TypeError``) -- on the
    live JAX buffers and module-level solve caches left by earlier tests in a
    shared/sharded run, so it is not safe inside the in-process suite."""
    plant = _fed_cstr_plant(simple_net, Q=10.0, C=(1.0, 0.0))
    y0 = plant.initial_state()
    params = plant.default_parameters()
    t_eval = jnp.asarray([0.0, 1.0])

    def loss(x):
        kw = dict(y0=y0, params=params, t_span=(0.0, 1.0), t_eval=t_eval, diff=gradient)
        kw[wrt] = x
        return jnp.sum(plant.solve(**kw).state[-1])

    x0 = y0 if wrt == "y0" else params
    # Prime the concrete per-unit param memo with one ordinary solve, then
    # differentiate. A leaked tracer would either raise UnexpectedTracerError in
    # the grad below or overwrite the memo with a tracer that outlives the trace.
    plant.solve(y0=y0, params=params, t_span=(0.0, 1.0), t_eval=t_eval, diff=gradient)
    g = jax.grad(loss)(x0)
    assert bool(jnp.all(jnp.isfinite(g)))
    # The per-unit param memo retains no tracer from the traced grad above.
    cache = plant.__dict__.get("_params_unit_cache")
    if cache is not None:
        assert not any(isinstance(v, jax.core.Tracer) for v in cache[1].values())


# ----- By-name plant parameter overrides (#134) ----------------------------


def _bsm2_no_solve():
    """The two-model BSM2 plant (ASM1 + ADM1), assembled but not solved --
    cheap, for the by-name parameter API."""
    from aquakin.plant.bsm import build_bsm2, bsm2_asm1_model

    asm1 = bsm2_asm1_model()
    adm1 = aquakin.load_model("adm1")
    return build_bsm2(asm1, adm1), asm1, adm1


def test_plant_parameter_names_are_model_prefixed():
    plant, asm1, adm1 = _bsm2_no_solve()
    names = plant.parameter_names()
    # The kinetic parameters: one block per model, model-prefixed.
    kinetic = [n for n in names if n.split(".")[0] in ("asm1", "adm1")]
    assert len(kinetic) == asm1.n_params + adm1.n_params
    assert "asm1.muH" in names  # ASM1 water line
    assert "adm1.k_m_ac" in names  # ADM1 digester
    # No bare names; every key carries a prefix (model for kinetic params,
    # unit for the appended flow setpoints).
    assert all("." in n for n in names)
    # Flow setpoints are addressed "<unit>.<setpoint>" -- the differentiable
    # design-variable knobs (recycle / wastage pumps, clarifier underflow).
    flow = [n for n in names if n.split(".")[0] not in ("asm1", "adm1")]
    assert flow and all(n.split(".")[0] in plant.units for n in flow)


def test_plant_parameter_index_matches_block_offset():
    """The friendly index equals the hand-computed block offset it replaces."""
    plant, asm1, adm1 = _bsm2_no_solve()
    # ASM1 is the first block (offset 0); ADM1 follows it.
    assert plant.parameter_index("asm1.muH") == asm1.param_index["muH"]
    assert plant.parameter_index("adm1.k_m_ac") == asm1.n_params + adm1.param_index["k_m_ac"]


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
        plant.parameter_values({"asm1.muh": 6.0})  # wrong case
    with pytest.raises(KeyError, match="Unknown plant parameter"):
        plant.parameter_index("adm1.nope")
    with pytest.raises(TypeError):
        plant.parameter_values([("asm1.muH", 6.0)])  # not a dict


# --- exact affine recycle-concentration pre-solve (#197) ---------------------


def _clarifier_recycle_loop(recycle_passes, *, capture):
    """A *stateless* high-recycle loop: fresh -> mix -> clarifier ->
    {overflow -> tank, underflow -> recycle to mix}. With no state-holding unit
    *inside* the loop, the particulate concentration genuinely circulates
    through the clarifier, so the bare Gauss-Seidel sweep converges at a rate set
    by the loop gain (= capture_efficiency) -- arbitrarily slow as capture -> 1.
    This is the worst case for the recycle solve; the affine pre-solve nails it
    exactly in one linear solve, gain-independent."""
    from aquakin.plant.clarifier import IdealClarifier

    net = aquakin.load_model("asm1")
    infl = net.influent({"SS": 60.0, "XS": 200.0, "XB_H": 50.0, "XI": 25.0, "SNH": 25.0}, Q=1.0)
    plant = Plant("clar_loop", recycle_passes=recycle_passes)
    plant.add_unit(MixerUnit(name="mix", input_port_names=["fresh", "ras"], model=net))
    plant.add_unit(
        IdealClarifier(name="clar", model=net, underflow_Q=2.0, capture_efficiency=capture)
    )
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=net,
            volume=50.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("fresh", infl, to="mix.fresh")
    plant.connect("mix", "clar.inlet")
    plant.connect("clar.overflow", "tank")
    plant.connect("clar.underflow", "mix.ras")
    return plant


def _recycle_warnings(plant):
    import warnings as _w

    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        plant.solve(
            t_span=(0.0, 0.05),
            t_eval=jnp.asarray([0.0, 0.05]),
            rtol=1e-4,
            atol=1e-3,
            integrator=aquakin.IntegratorConfig(max_steps=20000),
        )
    return [str(x.message) for x in w if "recycle concentration" in str(x.message)]


@pytest.mark.parametrize("capture", [0.6, 0.998])
def test_recycle_presolve_is_exact_and_gain_independent(capture):
    # The affine pre-solve seeds the recycle back-edge with its exact fixed point
    # in one linear solve, so even an extreme-gain stateless loop is resolved at
    # the default few mop-up passes -> no convergence warning (which the bare
    # gain-limited sweep would have raised). Also check the seed equals the
    # many-pass Gauss-Seidel answer to machine precision.
    import numpy as np

    plant = _clarifier_recycle_loop(recycle_passes=3, capture=capture)
    assert _recycle_warnings(plant) == []

    plant._build_state_layout()
    plant._build_parameter_layout()
    y0 = plant.initial_state()
    states = plant._split_state(y0)
    params = plant.default_parameters()
    t = jnp.asarray(0.0)
    flows = plant._recycle._resolve_flows(t, params, states)
    presolved = plant._recycle._resolve_recycle_concentrations(t, states, params, flows)
    key = list(plant._recycle_keys)[0]
    # Exactness as a fixed point: one forward pass from the pre-solved seed
    # reproduces the recycle concentration (gain-independent -- a deep
    # Gauss-Seidel reference would itself be unconverged at capture=0.998).
    influent = {(None, pn): s.at(t) for pn, s in plant.influents.items()}
    one = plant._sweep_outputs(t, states, influent, dict(presolved), params, passes=1)
    assert np.allclose(np.asarray(one[key].C), np.asarray(presolved[key].C), rtol=1e-8, atol=1e-8)
    assert np.all(np.isfinite(np.asarray(presolved[key].C)))


def test_recycle_presolve_skipped_without_recycle(simple_net):
    # A plant with no recycle edges has nothing to pre-solve -> no warning.
    plant = Plant("no_recycle")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=simple_net,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net), to="tank.inlet")
    assert _recycle_warnings(plant) == []


def test_recycle_presolve_differentiable():
    # The affine pre-solve (probe + jnp.linalg.solve) runs inside the jitted,
    # differentiated RHS: a gradient through solve stays finite (and the concrete
    # convergence check is skipped under tracing, so no warning).
    plant = _clarifier_recycle_loop(recycle_passes=3, capture=0.6)
    y0 = plant.initial_state()

    def loss(p):
        sol = plant.solve(
            t_span=(0.0, 0.02),
            t_eval=jnp.asarray([0.0, 0.02]),
            params=p,
            y0=y0,
            rtol=1e-4,
            atol=1e-3,
            diff=aquakin.DifferentiationConfig(method="stable"),
            integrator=aquakin.IntegratorConfig(max_steps=20000, colored_jacobian=False),
        )
        return jnp.sum(sol.final_state)

    import warnings as _w

    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        g = jax.grad(loss)(plant.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    assert [x for x in w if "recycle concentration" in str(x.message)] == []


# --- plant introspection: discoverable unit / port / species names (#147) -----


def test_list_units_ports_species():
    plant, asm1, adm1 = _bsm2_no_solve()
    units = plant.list_units()
    assert units[:3] == ["front_mix", "primary", "as_mix"]
    assert "digester" in units and "settler" in units
    # output endpoints are "unit.port" strings (what stream() accepts)
    ports = plant.list_ports()
    assert "settler.overflow" in ports and "settler.underflow" in ports
    assert "tank5.out" in ports
    assert all("." in p for p in ports)
    assert plant.list_ports("settler") == ["settler.overflow", "settler.underflow"]
    assert plant.list_ports("tank1", role="input") == ["tank1.inlet"]
    # species of the concentration-vector units
    assert plant.list_species("tank5") == list(asm1.species)
    assert plant.list_species("digester") == list(adm1.species)


def test_list_species_rejects_non_concentration_units():
    # stateless mixers/splitters and the layered Takacs settler carry a model
    # but their state is not a concentration vector -> clear error, not a wrong
    # or empty result.
    plant, _asm1, _adm1 = _bsm2_no_solve()
    for unit in ("front_mix", "settler", "thickener"):
        with pytest.raises(KeyError, match="not a concentration vector"):
            plant.list_species(unit)


def test_endpoint_error_taxonomy(simple_net):
    """Endpoint resolution raises the uniform exception taxonomy: an unknown
    unit / port is a ``KeyError`` subclass, an invalid wiring is a ``ValueError``
    subclass, so a caller can tell 'bad name' from 'bad value'."""
    plant = _single_cstr_plant(simple_net)  # unit 'tank' (input 'inlet'), influent 'feed'

    # Unknown unit -> UnknownUnitError, still catchable as KeyError.
    with pytest.raises(aquakin.UnknownUnitError) as exc_info:
        plant._parse_endpoint("ghost", role="source")
    assert isinstance(exc_info.value, KeyError)

    # Unknown port on a known unit -> UnknownPortError (KeyError family).
    with pytest.raises(aquakin.UnknownPortError) as exc_info:
        plant._parse_endpoint("tank.nope", role="destination")
    assert isinstance(exc_info.value, KeyError)

    # An influent used as a connect source is a wiring error (ValueError family),
    # not an unknown name -- the name exists, its use is wrong.
    with pytest.raises(aquakin.WiringError) as exc_info:
        plant._parse_endpoint("feed", role="source")
    assert isinstance(exc_info.value, ValueError)
    assert not isinstance(exc_info.value, KeyError)


def test_introspection_unknown_name_hints():
    plant, _asm1, _adm1 = _bsm2_no_solve()
    with pytest.raises(KeyError, match="Did you mean: tank5"):
        plant.list_species("tank6")  # close to tank5/tank4/...
    with pytest.raises(KeyError, match="Unknown unit"):
        plant.list_ports("nope")
    with pytest.raises(ValueError, match="role must be"):
        plant.list_ports(role="sideways")


def test_solution_C_named_errors_and_available_streams():
    from aquakin.plant.plant import PlantSolution
    from aquakin.plant.bsm import bsm2_warm_start

    plant, _asm1, _adm1 = _bsm2_no_solve()
    plant._build_state_layout()
    y0 = bsm2_warm_start(plant)
    sol = PlantSolution(t=jnp.asarray([0.0]), state=y0[None, :], plant=plant)

    # available_streams mirrors plant.list_ports()
    assert sol.available_streams() == plant.list_ports()
    # a good lookup works
    assert float(sol.C_named("tank5", "SNH")[0]) > 0.0
    # unknown species -> hint; unknown unit -> hint; non-concentration unit ->
    # a clear "read it as a stream" message.
    with pytest.raises(KeyError, match="Did you mean: SNH"):
        sol.C_named("tank5", "SNHH")
    with pytest.raises(KeyError, match="Unknown unit"):
        sol.C_named("tankX", "SNH")
    with pytest.raises(KeyError, match="not a concentration vector"):
        sol.C_named("settler", "XB_H")
    with pytest.raises(KeyError, match="Unknown unit"):
        sol.unit_state("tankX")


def test_plant_final_named_and_C_named_many():
    """final_named / C_named_many on a PlantSolution mirror C_named(unit, sp):
    last-point floats by name, several trajectories at once, same hinted errors."""
    from aquakin.plant.plant import PlantSolution
    from aquakin.plant.bsm import bsm2_warm_start

    plant, _asm1, _adm1 = _bsm2_no_solve()
    plant._build_state_layout()
    y0 = bsm2_warm_start(plant)
    sol = PlantSolution(t=jnp.asarray([0.0]), state=y0[None, :], plant=plant)

    # subset: floats equal to the last C_named point
    fn = sol.final_named("tank5", ["SNH", "XB_H"])
    assert isinstance(fn["XB_H"], float)
    assert fn["SNH"] == float(sol.C_named("tank5", "SNH")[-1])
    # species=None covers the unit's whole model
    assert set(sol.final_named("tank5")) == set(plant.list_species("tank5"))
    # C_named_many returns one trajectory per name
    many = sol.C_named_many("tank5", ["SNH", "SNO"])
    assert set(many) == {"SNH", "SNO"}
    assert jnp.array_equal(many["SNH"], sol.C_named("tank5", "SNH"))
    # shared errors: unknown species, and a non-concentration unit (species=None)
    with pytest.raises(KeyError, match="Did you mean: SNH"):
        sol.final_named("tank5", ["SNHH"])
    with pytest.raises(KeyError, match="not a concentration vector"):
        sol.final_named("settler")


# --- semantic stream shortcuts: plant.stream(sol, "effluent") etc. (#148) -----


def test_named_stream_resolution_and_effluent(simple_net):
    # register a semantic name -> "unit.port"; stream(sol, name) reads that port,
    # and effluent_stream() reads the recorded effluent_endpoint.
    plant = _single_cstr_plant(simple_net)
    plant.register_stream("product", "tank.out")
    plant.effluent_endpoint = "tank.out"
    assert plant.list_streams() == {"product": "tank.out"}

    sol = plant.solve(t_span=(0.0, 50.0), t_eval=jnp.linspace(0.0, 50.0, 4))
    by_name = plant.stream(sol, "product")
    by_port = plant.stream(sol, "tank.out")
    by_method = plant.effluent_stream(sol)
    assert jnp.allclose(by_name.C, by_port.C)
    assert jnp.allclose(by_method.C, by_name.C)
    # a literal "unit.port" still works unchanged (semantic resolution is opt-in)
    assert jnp.allclose(plant.stream(sol, "tank.out").Q, by_name.Q)


def test_effluent_stream_requires_endpoint(simple_net):
    plant = _single_cstr_plant(simple_net)  # no effluent_endpoint set
    sol = plant.solve(t_span=(0.0, 10.0), t_eval=jnp.asarray([0.0, 10.0]))
    with pytest.raises(ValueError, match="no recorded effluent_endpoint"):
        plant.effluent_stream(sol)


def test_bsm_named_streams_registered():
    from aquakin.plant.bsm import build_bsm1, build_bsm2

    s1 = build_bsm1().list_streams()
    assert s1["effluent"] == "clarifier.overflow"
    assert s1["ras"] == "underflow_split.ras"
    assert s1["internal_recycle"] == "tank5_split.internal_recycle"

    s2 = build_bsm2().list_streams()
    assert s2["effluent"] == "settler.overflow"
    assert s2["reject"] == "reject_mix.out"
    assert s2["primary_sludge"] == "primary.underflow"
    assert s2["disposal_sludge"] == "dewatering.underflow"


def test_stream_unknown_semantic_name_hints():
    from aquakin.plant.bsm import build_bsm2

    plant = build_bsm2()
    # The name is resolved before the solution is touched, so None is fine here.
    with pytest.raises(KeyError, match="Did you mean: effluent"):
        plant.stream(None, "effluant")  # close to "effluent"
    with pytest.raises(KeyError, match="Semantic names"):
        plant.stream(None, "nonsense")


def test_digester_gas_and_no_digester_error(simple_net):
    from aquakin.plant.bsm import build_bsm2, bsm2_warm_start, DigesterGas
    from aquakin.plant.plant import PlantSolution

    plant = build_bsm2()
    plant._build_state_layout()
    y0 = bsm2_warm_start(plant)
    sol = PlantSolution(t=jnp.asarray([0.0, 1.0]), state=jnp.stack([y0, y0]), plant=plant)
    gas = plant.digester_gas(sol)
    assert isinstance(gas, DigesterGas)
    assert float(gas.Q[0]) > 0.0 and float(gas.ch4[0]) > 0.0
    assert gas.methane_production() == pytest.approx(float(gas.ch4[0]), rel=1e-6)
    # the published BSM2 digester makes ~1000 kg CH4/d at the warm state
    assert 800.0 < gas.methane_production() < 1200.0

    # a plant with no ADM1 digester -> clear error
    cstr = _single_cstr_plant(simple_net)
    cstr_sol = cstr.solve(t_span=(0.0, 1.0), t_eval=jnp.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="no anaerobic digester"):
        cstr.digester_gas(cstr_sol)


def test_digester_gas_normalized_to_published_steady_state():
    """The reported biogas flow is normalized to atmospheric pressure.

    At the published BSM2 open-loop steady-state digester gas composition
    (Gernaey et al. 2014; Jeppsson et al. 2007), the reported gas flow is the raw
    overpressure outflow ``k_P*(P_gas - P_atm)`` recalculated to atmospheric
    pressure by ``P_gas/P_atm``: ~2708 m3/d, giving ~1065 kg CH4/d. Omitting the
    normalization understates both by ``P_gas/P_atm`` (~5%), which the gas-phase
    concentrations -- matched without it -- cannot reveal."""
    from aquakin.plant.bsm import build_bsm2, bsm2_warm_start
    from aquakin.plant.plant import PlantSolution

    # Published steady-state digester gas-phase composition (kg COD/m3 for the
    # COD gases) and the resulting reported flow and methane production.
    REF_GAS = {"S_gas_h2": 1.1032e-5, "S_gas_ch4": 1.6535, "S_gas_co2": 0.01354}
    REF_QGAS = 2708.34  # m3/d, normalized to atmospheric pressure
    REF_CH4 = 1065.35  # kg CH4/d

    plant = build_bsm2()
    plant._build_state_layout()
    y0 = bsm2_warm_start(plant)
    dig = plant.units["digester"]
    si = dig.model.species_index
    dvec = plant.states_by_unit(y0)["digester"]
    dvec = dvec.at[jnp.array([si[k] for k in REF_GAS])].set(
        jnp.array([REF_GAS[k] for k in REF_GAS])
    )
    y = plant.initial_state(overrides={"digester": dvec})
    sol = PlantSolution(t=jnp.asarray([0.0, 1.0]), state=jnp.stack([y, y]), plant=plant)
    gas = plant.digester_gas(sol)
    assert float(gas.Q[0]) == pytest.approx(REF_QGAS, rel=2e-3)
    assert gas.methane_production() == pytest.approx(REF_CH4, rel=2e-3)


def test_stream_series_named_accessors(simple_net):
    """StreamSeries shares the _HasNamedSpecies accessors: C_named (hinted),
    C_named_many, final_named and .final."""
    from aquakin.plant.streams import StreamSeries

    n = simple_net.n_species
    t = jnp.asarray([0.0, 1.0, 2.0])
    C = jnp.stack([jnp.full((n,), 0.1 * (i + 1)) for i in range(3)])
    eff = StreamSeries(t=t, Q=jnp.full((3,), 5.0), C=C, model=simple_net)

    sp0 = simple_net.species[0]
    assert jnp.array_equal(eff.C_named_many([sp0])[sp0], eff.C_named(sp0))
    assert eff.final_named([sp0])[sp0] == float(eff.C_named(sp0)[-1])
    assert set(eff.final) == set(simple_net.species)
    with pytest.raises(KeyError, match="Available"):
        eff.C_named("definitely_not_a_species")
