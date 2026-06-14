"""The optional matplotlib ``sol.plot(...)`` helper (#151)."""

import jax.numpy as jnp
import pytest

matplotlib = pytest.importorskip("matplotlib")   # optional `plot` extra
matplotlib.use("Agg")            # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

import aquakin                   # noqa: E402
from aquakin.plant.cstr import Aeration, CSTRUnit  # noqa: E402
from aquakin.plant.plant import Plant              # noqa: E402
from aquakin.plant.streams import StreamSeries     # noqa: E402


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def batch_solution():
    net = aquakin.load_network("asm1")
    reactor = aquakin.BatchReactor(net, aquakin.OperatingConditions(T=293.15))
    return reactor.solve(net.default_concentrations(),
                         t_span=(0.0, 2.0), t_eval=jnp.linspace(0.0, 2.0, 5))


def test_plot_single_species_labels_axes_with_units(batch_solution):
    ax = batch_solution.plot("SNH")
    assert ax.get_xlabel() == "time [d]"            # asm1 is a days network
    assert ax.get_ylabel() == "SNH [g_N/m³]"        # units from the network
    assert len(ax.lines) == 1


def test_plot_multiple_species_legends(batch_solution):
    ax = batch_solution.plot(["SNH", "SNO", "SS"])
    assert len(ax.lines) == 3
    assert ax.get_legend() is not None
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {"SNH", "SNO", "SS"}


def test_plot_none_plots_every_species(batch_solution):
    ax = batch_solution.plot()
    assert len(ax.lines) == batch_solution.network.n_species


def test_plot_onto_supplied_axes_returns_it(batch_solution):
    fig, ax = plt.subplots()
    out = batch_solution.plot("SNH", ax=ax)
    assert out is ax


def test_plot_unknown_species_hints(batch_solution):
    with pytest.raises(KeyError, match="Did you mean"):
        batch_solution.plot("SNHH")


def test_plot_time_unit_follows_network():
    net = aquakin.load_network("ozone_bromate")        # a seconds network
    reactor = aquakin.BatchReactor(net, net.default_conditions())
    sol = reactor.solve(net.default_concentrations(), t_span=(0.0, 10.0),
                        t_eval=jnp.linspace(0.0, 10.0, 4))
    assert sol.plot("BrO3-").get_xlabel() == "time [s]"


def test_pfr_plot_uses_axial_position():
    net = aquakin.load_network("asm1")
    pfr = aquakin.PlugFlowReactor(
        net, aquakin.SpatialConditions.uniform(8, T=293.15),
        n_points=8, length=5.0, velocity=1.0)
    ax = pfr.solve(net.default_concentrations()).plot("SNH")
    assert ax.get_xlabel() == "axial position [m]"


def test_stream_series_plot():
    net = aquakin.load_network("asm1")
    t = jnp.array([0.0, 1.0, 2.0])
    C = jnp.stack([jnp.full((net.n_species,), 0.1 * (i + 1)) for i in range(3)])
    ss = StreamSeries(t=t, Q=jnp.full((3,), 5.0), C=C, network=net)
    ax = ss.plot("SNH")
    assert ax.get_xlabel() == "time [d]"
    assert ax.get_ylabel() == "SNH [g_N/m³]"


def _single_cstr_plant(net):
    plant = Plant("one")
    plant.add_unit(CSTRUnit(name="tank", network=net, volume=2000.0,
                            input_port_names=["inlet"], conditions={"T": 293.15},
                            aeration=Aeration(kla=120.0, do_sat=8.0)))
    plant.add_influent("feed", net.influent({"SS": 200.0, "XS": 150.0,
                                             "SNH": 30.0, "SALK": 7.0}, Q=5000.0),
                       to="tank.inlet")
    return plant


def test_plant_solution_plot_by_unit():
    net = aquakin.load_network("asm1")
    plant = _single_cstr_plant(net)
    sol = plant.solve(t_span=(0.0, 3.0), t_eval=jnp.linspace(0.0, 3.0, 6))

    ax = sol.plot("tank", "SNH")
    assert ax.get_xlabel() == "time [d]"
    assert ax.get_ylabel() == "SNH [g_N/m³]"
    ax2 = sol.plot("tank", ["SNH", "SS"])
    assert len(ax2.lines) == 2 and ax2.get_legend() is not None
    # an unknown species on a real unit is hinted
    with pytest.raises(KeyError, match="Did you mean"):
        sol.plot("tank", "SNHH")
