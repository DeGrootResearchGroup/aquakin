"""An influent whose flow, temperature and composition are expressions of time.

The expression language itself is ``tests/unit/test_time_expressions.py``; this
is about an influent following one, and a plant being driven by it.
"""

import jax
import numpy as np
import pytest

import aquakin
from aquakin.plant import CSTRUnit, InfluentExpression, Plant

FEED = dict(SI=30.0, SS=69.5, XI=51.2, XS=202.32, XB_H=28.17, SND=6.95, XND=10.59, SALK=7.0)


@pytest.fixture
def model():
    return aquakin.load_model("asm1")


def _tank(model, influent):
    plant = Plant("tank", time_unit="d")
    plant.add_unit(CSTRUnit("tank", model, 1000.0, ["in"], conditions={"T": 293.15}))
    plant.add_influent("feed", influent, to="tank.in")
    return plant


def test_a_diurnal_feed_follows_its_expression(model):
    influent = InfluentExpression(
        model,
        Q="18446 + 4000*sin(2*pi*t)",
        composition={"SNH": "31.6 + 8*sin(2*pi*t)", "SS": 69.5},
    )
    for t, q, snh in [(0.0, 18446.0, 31.6), (0.25, 22446.0, 39.6), (0.75, 14446.0, 23.6)]:
        stream = influent.at(t)
        assert float(stream.Q) == pytest.approx(q)
        assert float(stream.C[model.species_index["SNH"]]) == pytest.approx(snh)
        # A constant among expressions stays constant.
        assert float(stream.C[model.species_index["SS"]]) == pytest.approx(69.5)


def test_a_species_the_composition_does_not_name_is_absent(model):
    """Same convention as InfluentSeries.constant(base="zero"): an unnamed
    species is not in the feed rather than at its YAML reference value."""
    stream = InfluentExpression(model, Q=1.0, composition={"SNH": 5.0}).at(0.0)
    assert float(stream.C[model.species_index["XI"]]) == 0.0
    assert float(stream.C[model.species_index["SNH"]]) == 5.0


def test_a_plain_number_needs_no_expression(model):
    """So a partly-patterned feed costs nothing for the parts that are not."""
    influent = InfluentExpression(model, Q=2000.0, composition={"SNH": 30.0}, T=288.15)
    for t in (0.0, 5.0, 1e6):
        stream = influent.at(t)
        assert float(stream.Q) == 2000.0
        assert float(stream.scalars["T"]) == pytest.approx(288.15)


def test_temperature_is_carried_only_when_given(model):
    assert "T" not in InfluentExpression(model, Q=1.0).at(0.0).scalars
    assert float(
        InfluentExpression(model, Q=1.0, T="285 + 3*sin(2*pi*t)").at(0.25).scalars["T"]
    ) == (pytest.approx(288.0))


def test_the_expressions_keep_their_own_clock(model):
    """An expression written in days means days, even on a plant running on
    hours -- in_time_unit converts the t it is given and the flow it returns,
    not the expression."""
    daily = InfluentExpression(model, Q="2400*t", composition={"SNH": "t"})
    hourly = daily.in_time_unit("h")
    # Six hours in is a quarter of a day: the same point of the pattern...
    assert float(hourly.at(6.0).C[model.species_index["SNH"]]) == pytest.approx(0.25)
    assert float(daily.at(0.25).C[model.species_index["SNH"]]) == pytest.approx(0.25)
    # ...and the flow comes back per hour rather than per day.
    assert float(hourly.at(6.0).Q) == pytest.approx(float(daily.at(0.25).Q) / 24.0)
    # Asking for the clock it already has changes nothing.
    assert daily.in_time_unit(None) is daily


@pytest.mark.parametrize(
    ("units", "key", "expected"),
    [
        ({"Q": "m3/h"}, "Q", 2400.0),  # 100 m3/h on a model running on days
        ({"T": "degC"}, "T", 288.15),  # 15 degC
    ],
)
def test_declared_units_are_converted_as_the_expression_is_read(model, units, key, expected):
    """A string cannot be scaled in place the way a sampled array can, so the
    factor is applied to the result -- which is the same number."""
    influent = InfluentExpression(model, Q=100.0, T=15.0, units=units)
    stream = influent.at(0.0)
    got = float(stream.Q) if key == "Q" else float(stream.scalars["T"])
    assert got == pytest.approx(expected)


def test_a_plant_is_driven_by_the_pattern(model):
    """The point of the feature: a dynamic run that follows a load pattern with
    no table file anywhere."""
    plant = _tank(
        model,
        InfluentExpression(
            model,
            Q="2000 + 500*sin(2*pi*t)",
            composition={**FEED, "SNH": "31.6 + 10*sin(2*pi*t)"},
        ),
    )
    t = np.linspace(0.0, 1.0, 25)
    solution = plant.solve(t_span=(0.0, 1.0), t_eval=t)
    stream = plant.stream(solution, "tank.out")
    Q = np.asarray(stream.Q)
    # The effluent flow is the feed flow, so it carries the swing exactly.
    assert Q.min() == pytest.approx(1500.0)
    assert Q.max() == pytest.approx(2500.0)
    assert np.all(np.isfinite(np.asarray(stream.C)))
    # And the reactor actually sees the load vary, not just the flow.
    snh = np.asarray(stream.C[:, model.species_index["SNH"]])
    assert snh.max() - snh.min() > 1.0


def test_a_gradient_flows_through_a_solve_the_pattern_drives(model):
    """It is evaluated inside the jitted right-hand side, so a calibration
    through a patterned feed has to stay differentiable."""
    plant = _tank(model, InfluentExpression(model, Q="2000 + 500*sin(2*pi*t)", composition=FEED))
    t = np.array([0.0, 0.2])

    def final_snh(params):
        solution = plant.solve(t_span=(0.0, 0.2), t_eval=t, params=params)
        return plant.stream(solution, "tank.out", params).C[-1, model.species_index["SNH"]]

    grad = np.asarray(jax.grad(final_snh)(plant.default_parameters()))
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)


def test_what_is_refused(model):
    with pytest.raises(ValueError, match="not species of model"):
        InfluentExpression(model, Q=1.0, composition={"NOPE": 1.0})
    with pytest.raises(ValueError, match="Did you mean"):
        InfluentExpression(model, Q=1.0, composition={"SNHH": 1.0})
    # An unreadable expression is a load-time error naming the field, not a
    # failure inside the first solve.
    with pytest.raises(ValueError, match="the flow:"):
        InfluentExpression(model, Q="1 +")
    with pytest.raises(ValueError, match="species 'SNH':"):
        InfluentExpression(model, Q=1.0, composition={"SNH": "wobble(t)"})
    with pytest.raises(ValueError, match="the temperature:"):
        InfluentExpression(model, Q=1.0, T="t +")
    with pytest.raises(ValueError, match="not 't', 'Q', 'T' or a species"):
        InfluentExpression(model, Q=1.0, units={"zz": "d"})
    with pytest.raises(ValueError, match="cannot be read"):
        InfluentExpression(model, Q=1.0, units={"Q": "furlongs"})
    with pytest.raises(ValueError, match="units values must be strings"):
        InfluentExpression(model, Q=1.0, units={"Q": 3.0})


def test_a_unit_measuring_another_quantity_is_refused(model):
    """Only the scale may differ. Converting a mass flow to a volume flow would
    need a density, which the influent has no way to know."""
    with pytest.raises(ValueError, match="cannot be converted to"):
        InfluentExpression(model, Q=1.0, units={"Q": "kg/d"})


# ---------------------------------------------------------------------------
# Through a flowsheet spec
# ---------------------------------------------------------------------------


def test_a_pattern_round_trips_through_a_spec_with_no_table_file(tmp_path, model):
    """The point of writing the feed as text: it goes in the spec as it stands,
    so there is no CSV to carry beside the flowsheet."""
    plant = _tank(
        model,
        InfluentExpression(
            model,
            Q="2000 + 500*sin(2*pi*t)",
            composition={**FEED, "SNH": "31.6 + 10*sin(2*pi*t)"},
            T=288.15,
        ),
    )
    spec = plant.to_spec(tmp_path / "flowsheet.yaml")
    entry = spec["units"]["feed"]
    assert entry["Q"] == "2000 + 500*sin(2*pi*t)"
    assert entry["composition"]["SNH"] == "31.6 + 10*sin(2*pi*t)"
    assert entry["composition"]["SS"] == 69.5  # a constant stays a number
    assert entry["series"] is None
    assert not (tmp_path / "data").exists()

    rebuilt = Plant.from_spec(tmp_path / "flowsheet.yaml")
    assert isinstance(rebuilt.influents["feed"], InfluentExpression)
    for t in (0.0, 0.25, 0.75):
        before, after = plant.influents["feed"].at(t), rebuilt.influents["feed"].at(t)
        assert float(before.Q) == pytest.approx(float(after.Q))
        np.testing.assert_allclose(np.asarray(before.C), np.asarray(after.C))
        assert float(before.scalars["T"]) == pytest.approx(float(after.scalars["T"]))


def test_a_feed_of_plain_numbers_is_still_a_sampled_series(tmp_path, model):
    """So every spec written before expressions existed reads back unchanged."""
    from aquakin.plant import InfluentSeries

    plant = _tank(model, InfluentSeries.constant(model, Q=2000.0, T=288.15, **FEED))
    plant.to_spec(tmp_path / "flowsheet.yaml")
    rebuilt = Plant.from_spec(tmp_path / "flowsheet.yaml")
    assert isinstance(rebuilt.influents["feed"], InfluentSeries)


def test_an_unreadable_expression_in_a_spec_names_the_setting(tmp_path, model):
    import yaml

    plant = _tank(model, InfluentExpression(model, Q=2000.0, composition=FEED))
    plant.to_spec(tmp_path / "flowsheet.yaml")
    doc = yaml.safe_load((tmp_path / "flowsheet.yaml").read_text())
    doc["units"]["feed"]["Q"] = "2000 + wobble(t)"
    (tmp_path / "flowsheet.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="'Q'"):
        Plant.from_spec(tmp_path / "flowsheet.yaml")
