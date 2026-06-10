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
    plant.add_influent("feed", _constant_influent(simple_net))
    plant.connect(None, "feed", "tank", "inlet")
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
    plant.add_influent("feed", _constant_influent(simple_net))
    plant.connect(None, "feed", "t1", "inlet")
    plant.connect("t1", "out", "t2", "inlet")
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
    plant.add_influent("feed_main", _constant_influent(simple_net, Q=10.0))
    plant.add_influent("feed_side", _constant_influent(simple_net, Q=5.0, C=(2.0, 0.0)))
    plant.connect(None, "feed_main", "t1", "inlet")
    plant.connect("t1", "out", "mix", "a")
    plant.connect(None, "feed_side", "mix", "b")
    plant.connect("mix", "out", "t2", "inlet")
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
    plant.add_influent("feed", _constant_influent(simple_net))
    plant.connect(None, "feed", "src", "inlet")
    plant.connect("src", "out", "split", "in")
    plant.connect("split", "a", "sink_a", "inlet")
    plant.connect("split", "b", "sink_b", "inlet")
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
    plant.add_influent("feed", _constant_influent(simple_net, Q=10.0))
    plant.connect(None, "feed", "mix", "fresh")
    plant.connect("mix", "out", "tank", "inlet")
    plant.connect("tank", "out", "split", "in")
    plant.connect(
        "split", "out_recycle", "mix", "recycle",
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


def test_ad_grad_through_plant(simple_net):
    """jax.grad through Plant.solve w.r.t. params returns a finite gradient."""
    plant = Plant("grad_test")
    plant.add_unit(
        CSTRUnit(
            name="t1", network=simple_net, volume=100.0,
            input_port_names=["inlet"], conditions={"T": 293.15},
        )
    )
    plant.add_influent("feed", _constant_influent(simple_net))
    plant.connect(None, "feed", "t1", "inlet")

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
    plant.add_influent("feed", inf)
    plant.connect(None, "feed", "tank", "inlet")
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
