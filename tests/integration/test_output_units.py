"""Results read back in requested units (``units=`` on the species readers).

The test model's species are in mol/L. Each reader is checked against a closed
form expressed in another unit: a batch solve of first-order decay gives
``A(t) = exp(-0.1 t)`` mol/L, and the single-CSTR plant (V = 100, Q = 10 of
A = 1, k = 0.1, A(0) = 1) gives ``A(t) = 0.5 + 0.5 exp(-0.2 t)`` mol/L.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

T_EVAL = jnp.linspace(0.0, 10.0, 6)


def _batch_A(t, k=0.1):
    return np.exp(-k * np.asarray(t))


def _cstr_A(t):
    return 0.5 + 0.5 * np.exp(-0.2 * np.asarray(t))


@pytest.fixture(scope="module")
def reactor(simple_model):
    return aquakin.BatchReactor(
        simple_model, aquakin.SpatialConditions.uniform(1, T=293.15), rtol=1e-10, atol=1e-12
    )


@pytest.fixture(scope="module")
def batch(reactor, simple_model):
    return reactor.solve(
        simple_model.default_concentrations(),
        params=simple_model.default_parameters(),
        t_span=(0.0, 10.0),
        t_eval=T_EVAL,
    )


# ----- reactor solutions ------------------------------------------------------------


def test_batch_species_in_requested_units(batch):
    expected = _batch_A(batch.t)
    np.testing.assert_allclose(batch.C_named("A"), expected, rtol=1e-7)
    np.testing.assert_allclose(batch.C_named("A", units="mmol/L"), 1e3 * expected, rtol=1e-7)
    np.testing.assert_allclose(batch.C_named("A", units="mol/m3"), 1e3 * expected, rtol=1e-7)
    np.testing.assert_allclose(batch.C_named("A", units="mmol/m3"), 1e6 * expected, rtol=1e-7)
    # The model's own unit (factor exactly 1) leaves the values
    # bit-identical.
    assert np.array_equal(batch.C_named("A", units="mol/L"), batch.C_named("A"))


def test_batch_many_and_final_in_requested_units(batch):
    A = _batch_A(batch.t)
    many = batch.C_named_many(["A", "B"], units={"A": "mmol/L"})
    np.testing.assert_allclose(many["A"], 1e3 * A, rtol=1e-7)
    np.testing.assert_allclose(many["B"], 1.0 - A, rtol=1e-7, atol=1e-12)  # mol/L
    both = batch.C_named_many("B", units="mmol/L")
    np.testing.assert_allclose(both["B"], 1e3 * (1.0 - A), rtol=1e-7, atol=1e-9)
    final = batch.final_named(units="mmol/L")
    assert final["A"] == pytest.approx(1e3 * float(A[-1]), rel=1e-7)
    assert final["B"] == pytest.approx(1e3 * (1.0 - float(A[-1])), rel=1e-7)


def test_batch_table_in_requested_units(batch):
    df = batch.to_dataframe(units={"A": "mmol/L"}, units_in_columns=True)
    np.testing.assert_allclose(df["A [mmol/L]"].to_numpy(), 1e3 * _batch_A(batch.t), rtol=1e-7)
    np.testing.assert_allclose(df["B [mol/L]"].to_numpy(), 1.0 - _batch_A(batch.t), atol=1e-9)
    assert df.attrs["units"] == {"A": "mmol/L", "B": "mol/L"}
    csv = batch.to_csv(units="mmol/L")
    assert csv.splitlines()[0] == "t,A [mmol/L],B [mmol/L]"


def test_converted_reading_is_differentiable(reactor, simple_model):
    # d A(t)/dk = -t exp(-k t) mol/L, so in mmol/L it is 1000 times that.
    C0 = simple_model.default_concentrations()

    def last_A(k):
        sol = reactor.solve(C0, params=jnp.asarray([k]), t_span=(0.0, 10.0), t_eval=T_EVAL)
        return sol.C_named("A", units="mmol/L")[-1]

    g = jax.grad(last_A)(0.1)
    assert float(g) == pytest.approx(1e3 * -10.0 * np.exp(-1.0), rel=1e-5)


@pytest.mark.parametrize(
    "units, match",
    [
        ("g/L", r"Cannot give 'A' in 'g/L': cannot convert"),
        ("parsecs", r"Cannot give 'A' in 'parsecs'"),
    ],
)
def test_units_of_another_quantity_are_refused(batch, units, match):
    with pytest.raises(ValueError, match=match):
        batch.C_named("A", units=units)


def test_units_for_a_species_not_read_are_refused(batch):
    with pytest.raises(ValueError, match=r"units names \['B'\], which are not among"):
        batch.C_named_many(["A"], units={"B": "mmol/L"})
    with pytest.raises(ValueError, match=r"units names \['Z'\]"):
        batch.to_dataframe(units={"Z": "mmol/L"})


# ----- plant solutions and streams ---------------------------------------------------


@pytest.fixture(scope="module")
def cstr_run(simple_model):
    """The single-CSTR plant of ``make_cstr_plant``, solved once."""
    from aquakin.plant import CSTRUnit, Plant
    from aquakin.plant.influent import InfluentSeries

    plant = Plant("single_cstr", time_unit="s")
    plant.add_unit(
        CSTRUnit(
            name="tank",
            model=simple_model,
            volume=100.0,
            input_port_names=["inlet"],
            conditions={"T": 293.15},
        )
    )
    plant.add_influent(
        "feed",
        InfluentSeries(
            t=jnp.asarray([0.0, 100.0]),
            Q=jnp.asarray([10.0, 10.0]),
            C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
            model=simple_model,
        ),
        to="tank.inlet",
    )
    sol = plant.solve(t_span=(0.0, 10.0), t_eval=T_EVAL, rtol=1e-10, atol=1e-12)
    return plant, sol


def test_plant_unit_species_in_requested_units(cstr_run):
    _, sol = cstr_run
    A = _cstr_A(sol.t)
    np.testing.assert_allclose(sol.C_named("tank", "A", units="mmol/L"), 1e3 * A, rtol=1e-7)
    many = sol.C_named_many("tank", ["A", "B"], units={"B": "mmol/L"})
    np.testing.assert_allclose(many["A"], A, rtol=1e-7)
    final = sol.final_named("tank", units="mmol/L")
    assert final["A"] == pytest.approx(1e3 * float(A[-1]), rel=1e-7)
    df = sol.to_dataframe("tank", units="mmol/L")
    np.testing.assert_allclose(df["A"].to_numpy(), 1e3 * A, rtol=1e-7)
    assert df.attrs["units"] == {"A": "mmol/L", "B": "mmol/L"}
    csv = sol.to_csv(unit="tank", units={"A": "mmol/L"})
    assert csv.splitlines()[0] == "t,A [mmol/L],B [mol/L]"


def test_plant_stream_in_requested_units(cstr_run):
    plant, sol = cstr_run
    effluent = plant.stream(sol, "tank.out")
    A = _cstr_A(sol.t)
    np.testing.assert_allclose(effluent.C_named("A", units="mmol/L"), 1e3 * A, rtol=1e-7)
    df = effluent.to_dataframe(units={"A": "mmol/L"})
    np.testing.assert_allclose(df["A"].to_numpy(), 1e3 * A, rtol=1e-7)
    np.testing.assert_allclose(df["Q"].to_numpy(), 10.0)  # flow is left alone
    csv = effluent.to_csv(units="mmol/L")
    assert csv.splitlines()[0] == "t,Q,A [mmol/L],B [mmol/L]"


class _Clock:
    """A stateful unit with no model: its one state integrates d/dt = 1."""

    name = "clock"
    state_size = 1
    @property
    def input_ports(self):
        return []

    @property
    def output_ports(self):
        return []

    def initial_state(self):
        return jnp.zeros(1)

    def compute_outputs(self, t, state, inputs, params, signals=None):
        return {}

    def rhs(self, t, state, inputs, params, signals=None):
        return jnp.ones(1)

    def flow_outputs(self, input_flows, params, ctx=None):
        return {}


def test_plant_table_of_a_unit_without_species_refuses_units():
    from aquakin.plant import Plant

    plant = Plant("clock", time_unit="s")
    plant.add_unit(_Clock())
    sol = plant.solve(t_span=(0.0, 2.0), t_eval=jnp.asarray([0.0, 1.0, 2.0]))
    # Its generic state column is tabulated as is (state = t) ...
    np.testing.assert_allclose(sol.to_dataframe("clock")["state_0"].to_numpy(), [0.0, 1.0, 2.0])
    # ... and has no species unit to convert from.
    with pytest.raises(ValueError, match="'clock' has no model"):
        sol.to_dataframe("clock", units="mg/L")
