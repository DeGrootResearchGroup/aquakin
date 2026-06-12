"""to_dataframe() / to_csv() exporters on the solution classes and StreamSeries."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate.batch import BatchSolution
from aquakin.integrate.biofilm import BiofilmSolution
from aquakin.integrate.pfr import PFRSolution
from aquakin.plant.plant import PlantSolution
from aquakin.plant.streams import StreamSeries


@pytest.fixture
def net(simple_network):
    # Two species A, B, both "mol/L".
    return simple_network


def test_batch_to_dataframe_basic(net):
    t = jnp.array([0.0, 1.0, 2.0])
    C = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
    sol = BatchSolution(t=t, C=C, network=net)

    df = sol.to_dataframe()
    assert list(df.columns) == ["A", "B"]
    assert df.index.name == "t"
    np.testing.assert_allclose(df.index.to_numpy(), np.asarray(t))
    np.testing.assert_allclose(df["A"].to_numpy(), np.asarray(C[:, 0]))
    # Units kept in attrs, columns stay bare species names.
    assert df.attrs["units"] == {"A": "mol/L", "B": "mol/L"}


def test_units_in_columns(net):
    sol = BatchSolution(t=jnp.array([0.0]), C=jnp.array([[1.0, 2.0]]), network=net)
    df = sol.to_dataframe(units_in_columns=True)
    assert list(df.columns) == ["A [mol/L]", "B [mol/L]"]


def test_to_csv_string_embeds_units_by_default(net):
    sol = BatchSolution(t=jnp.array([0.0, 1.0]),
                        C=jnp.array([[1.0, 0.0], [0.5, 0.5]]), network=net)
    csv = sol.to_csv()
    header = csv.splitlines()[0]
    assert "A [mol/L]" in header and "B [mol/L]" in header
    assert header.startswith("t")


def test_to_csv_to_file(net, tmp_path):
    sol = BatchSolution(t=jnp.array([0.0, 1.0]),
                        C=jnp.array([[1.0, 0.0], [0.5, 0.5]]), network=net)
    path = tmp_path / "out.csv"
    sol.to_csv(path)
    text = path.read_text()
    assert "A [mol/L]" in text.splitlines()[0]


def test_pfr_indexed_by_position(net):
    x = jnp.array([0.0, 0.5, 1.0])
    C = jnp.array([[1.0, 0.0], [0.6, 0.4], [0.3, 0.7]])
    sol = PFRSolution(x=x, C=C, network=net)
    df = sol.to_dataframe()
    assert df.index.name == "x"
    np.testing.assert_allclose(df.index.to_numpy(), np.asarray(x))


def test_stream_series_has_flow_column(net):
    t = jnp.array([0.0, 1.0])
    Q = jnp.array([100.0, 120.0])
    C = jnp.array([[1.0, 0.0], [0.5, 0.5]])
    ss = StreamSeries(t=t, Q=Q, C=C, network=net)
    df = ss.to_dataframe()
    # Q precedes the species columns.
    assert list(df.columns) == ["Q", "A", "B"]
    np.testing.assert_allclose(df["Q"].to_numpy(), np.asarray(Q))
    assert df.index.name == "t"


def test_biofilm_bulk_default(net):
    t = jnp.array([0.0, 1.0])
    C = jnp.array([[1.0, 0.0], [0.5, 0.5]])
    n_layers = 2
    profile = jnp.zeros((2, n_layers + 1, 2))
    depth = jnp.array([1e-4, 2e-4])
    sol = BiofilmSolution(t=t, C=C, profile=profile, depth=depth, network=net)
    df = sol.to_dataframe()
    assert list(df.columns) == ["A", "B"]
    assert df.index.name == "t"
    np.testing.assert_allclose(df["A"].to_numpy(), np.asarray(C[:, 0]))


def test_biofilm_profile_multiindex(net):
    t = jnp.array([0.0, 1.0])
    C = jnp.array([[1.0, 0.0], [0.5, 0.5]])
    n_layers = 2
    n_comp = n_layers + 1
    # Distinct values per (time, compartment, species) so we can check layout.
    profile = jnp.arange(2 * n_comp * 2, dtype=float).reshape(2, n_comp, 2)
    depth = jnp.array([1e-4, 2e-4])
    sol = BiofilmSolution(t=t, C=C, profile=profile, depth=depth, network=net)

    df = sol.to_dataframe(profile=True)
    assert list(df.index.names) == ["t", "compartment"]
    assert df.shape[0] == 2 * n_comp
    assert "depth" in df.columns
    # Bulk row (compartment 0) has NaN depth; layer rows carry the mid-depths.
    bulk = df.xs(0, level="compartment")
    assert np.isnan(bulk["depth"].to_numpy()).all()
    layer1 = df.xs(1, level="compartment")
    np.testing.assert_allclose(layer1["depth"].to_numpy(), 1e-4)
    # Species value at (t=0, compartment=0, A) is profile[0,0,0] == 0.
    assert df.loc[(0.0, 0), "A"] == profile[0, 0, 0]


def _fake_plant(net):
    """A minimal stand-in exposing what PlantSolution's exporters use."""
    kinetic = SimpleNamespace(network=net, state_size=2)
    passive = SimpleNamespace(state_size=1)            # no .network
    return SimpleNamespace(
        units={"tank": kinetic, "mixer": passive},
        _state_layout={"tank": (0, 2), "mixer": (2, 1)},
    )


def test_plant_to_dataframe_kinetic_unit(net):
    t = jnp.array([0.0, 1.0])
    state = jnp.array([[1.0, 0.0, 9.0], [0.5, 0.5, 9.0]])
    sol = PlantSolution(t=t, state=state, plant=_fake_plant(net))
    df = sol.to_dataframe("tank")
    assert list(df.columns) == ["A", "B"]
    np.testing.assert_allclose(df["B"].to_numpy(), np.asarray(state[:, 1]))


def test_plant_to_dataframe_passive_unit(net):
    t = jnp.array([0.0, 1.0])
    state = jnp.array([[1.0, 0.0, 9.0], [0.5, 0.5, 8.0]])
    sol = PlantSolution(t=t, state=state, plant=_fake_plant(net))
    df = sol.to_dataframe("mixer")
    assert list(df.columns) == ["state_0"]
    np.testing.assert_allclose(df["state_0"].to_numpy(), np.asarray(state[:, 2]))


def test_plant_to_dataframe_unknown_unit(net):
    sol = PlantSolution(t=jnp.array([0.0]), state=jnp.zeros((1, 3)),
                        plant=_fake_plant(net))
    with pytest.raises(KeyError):
        sol.to_dataframe("nope")
