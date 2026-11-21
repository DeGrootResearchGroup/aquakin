"""``xy_chart`` drawn to hand-computed geometry, and each study kind's view
against a hand-built structure.

The chart is 420 x 200 with margins left 52, right 12, top 26, bottom 22, so the
plot area is 356 x 152: x runs from 52 to 408 and y from 178 (the lowest tick)
up to 26 (the highest).
"""

import xml.etree.ElementTree as ET

from aquakin_ui.results import NO_VALUE, analysis_view, xy_chart


def _svg(**kwargs):
    return ET.fromstring(xy_chart(**kwargs))


def _texts(svg, cls="ak-chart-axis"):
    return [t.text for t in svg.iter("text") if t.get("class") == cls]


def _points(svg):
    (line,) = [p for p in svg.iter("polyline")]
    return [tuple(map(float, p.split(","))) for p in line.get("points").split()]


DIAGONAL = [(0.0, 0.0), (10.0, 10.0)]


def test_a_line_is_drawn_to_its_ticks():
    """``[0, 10]`` gets y ticks 0, 5, 10 (four steps of 2.5 round up to 5) and
    x ticks every 2 (five steps of 2); the line runs corner to corner."""
    svg = _svg(title="SNH", label="SNH of eff", x_name="t", line=DIAGONAL)
    assert svg.get("viewBox") == "0 0 420 200" and svg.get("aria-label") == "SNH of eff"
    assert _texts(svg, "ak-chart-title") == ["SNH"]
    assert _texts(svg) == ["0", "5", "10", "0", "2", "4", "6", "8", "10"]
    grid = [float(g.get("y1")) for g in svg.iter("line") if g.get("class") == "ak-chart-grid"]
    assert grid == [178.0, 102.0, 26.0]
    assert _points(svg) == [(52.0, 178.0), (408.0, 26.0)]


def test_hovering_a_line_shows_each_sampled_point():
    """One band per point, split halfway between neighbours, each with its
    value in a tooltip."""
    svg = _svg(title="T", label="L", x_name="t", x_unit="d", y_unit="g/m³", line=DIAGONAL)
    bands = [g for g in svg.iter("g") if g.get("class") == "ak-chart-hit"]
    assert [g.find("title").text for g in bands] == ["t = 0 d: 0 g/m³", "t = 10.00 d: 10.00 g/m³"]
    rects = [(float(g.find("rect").get("x")), float(g.find("rect").get("width"))) for g in bands]
    assert rects == [(52.0, 178.0), (230.0, 178.0)]
    assert _texts(svg)[-1] == "time (d)"  # the axis title, from the unit


def test_missing_values_are_left_out_of_the_line():
    svg = _svg(title="T", label="L", x_name="t", line=[(0.0, 0.0), (5.0, None), (10.0, 10.0)])
    assert _points(svg) == [(52.0, 178.0), (408.0, 26.0)]


def test_a_reference_is_a_labelled_hairline_at_its_value():
    svg = _svg(title="T", label="L", x_name="k", line=DIAGONAL, reference=(5.0, "threshold"))
    (ref,) = [g for g in svg.iter("line") if g.get("class") == "ak-chart-ref"]
    assert (float(ref.get("y1")), float(ref.get("y2"))) == (102.0, 102.0)
    label = next(t for t in svg.iter("text") if t.text == "threshold")
    assert float(label.get("y")) == 98.0


def test_a_reference_outside_the_data_widens_the_axis():
    svg = _svg(title="T", label="L", x_name="k", line=DIAGONAL, reference=(20.0, "limit"))
    assert _texts(svg)[:5] == ["0", "5", "10", "15", "20"]
    (ref,) = [g for g in svg.iter("line") if g.get("class") == "ak-chart-ref"]
    assert float(ref.get("y1")) == 26.0


def test_data_and_fit_together_get_a_legend_instead_of_the_axis_title():
    svg = _svg(
        title="T",
        label="L",
        x_name="t",
        line=DIAGONAL,
        dots=[(0.0, 1.0), (10.0, None)],
        x_axis="time",
    )
    texts = _texts(svg)
    assert "fit" in texts and "data" in texts and "time" not in texts
    dots = [c for c in svg.iter("circle") if c.get("class") == "ak-chart-dot"]
    assert len(dots) == 2  # the one data point, and the legend's
    point = dots[1]
    assert (float(point.get("cx")), float(point.get("cy"))) == (52.0, 162.8)
    assert point.find("title").text == "data t = 0: 1.000"


def test_a_data_point_tooltip_carries_its_units():
    svg = _svg(title="T", label="L", x_name="t", x_unit="d", y_unit="g/m³", dots=DIAGONAL)
    tips = [c.find("title").text for c in svg.iter("circle") if c.get("class") == "ak-chart-dot"]
    assert tips == ["data t = 0 d: 0 g/m³", "data t = 10.00 d: 10.00 g/m³"]


def test_markup_in_a_title_is_text():
    svg = xy_chart(title="<b>&", label='"x"', x_name="t", line=DIAGONAL)
    assert "&lt;b&gt;&amp;" in svg and 'aria-label="&quot;x&quot;"' in svg
    assert ET.fromstring(svg).find("text").text == "<b>&"


def test_a_chart_with_no_values_says_so():
    svg = _svg(title="T", label="L", x_name="t", line=[(0.0, None)], dots=[])
    assert _texts(svg) == ["no finite values"] and svg.find("polyline") is None


# -- each study kind's view --------------------------------------------------


def test_a_fisher_view():
    view = analysis_view(
        {
            "kind": "fisher",
            "summary": {
                "values": {"m.a": 2.0, "m.b": 0.5},
                "std": {"m.a": 0.1, "m.b": 0.02},
                "criteria": {"D": 12.5, "A": 0.25},
            },
            "matrix": {"names": ["m.a", "m.b"], "rows": [[100.0, 0.0], [0.0, 2500.0]]},
        }
    )
    assert view == {
        "lines": [],
        "tables": [
            {
                "title": "Parameters",
                "columns": ["parameter", "value", "standard deviation"],
                "rows": [["m.a", "2.000", "0.1000"], ["m.b", "0.5000", "0.02000"]],
            },
            {
                "title": "Design criteria",
                "columns": ["criterion", "value"],
                "rows": [["D", "12.50"], ["A", "0.2500"]],
            },
            {
                "title": "Information matrix (log-parameters where transformed)",
                "columns": ["", "m.a", "m.b"],
                "rows": [["m.a", "100.0", "0"], ["m.b", "0", "2,500"]],
            },
        ],
        "charts": [],
    }


def test_a_fisher_view_without_a_matrix_shows_no_matrix_table():
    view = analysis_view({"kind": "fisher", "summary": {"values": {"m.a": 1.0}, "std": None}})
    assert [t["title"] for t in view["tables"]] == ["Parameters", "Design criteria"]
    assert view["tables"][1]["rows"] == []


def test_a_rank_view_defaults_to_the_d_criterion():
    view = analysis_view(
        {
            "kind": "rank",
            "table": {"candidate": ["eff.SNH", "eff.SNO"], "D": [3.0, 1.5]},
        }
    )
    assert view["tables"] == [
        {
            "title": "Sensor sets, best first",
            "columns": ["rank", "sensor set", "D", "improvement"],
            "rows": [["1", "eff.SNH", "3.000", NO_VALUE], ["2", "eff.SNO", "1.500", NO_VALUE]],
        }
    ]
    assert view["charts"] == [] and len(view["lines"]) == 1


def test_a_calibrate_view_charts_each_fitted_channel():
    view = analysis_view(
        {
            "kind": "calibrate",
            "summary": {
                "fitted": {"m.a": 1.5},
                "std": {},
                "converged": True,
                "iterations": 7,
                "message": "converged",
            },
            "fitted": {
                "t": [0.0, 5.0, 10.0],
                "values": {"eff": {"SNH": [0.0, 5.0, 10.0], "SNO": [10.0, 5.0, 0.0]}},
            },
            "data": {"t": [0.0, 10.0], "values": {"eff": {"SNH": [1.0, 9.0]}}},
        }
    )
    assert view["lines"] == ["The fit converged after 7 iterations (converged)."]
    assert view["tables"] == [
        {
            "title": "Fitted parameters",
            "columns": ["parameter", "value", "standard deviation"],
            "rows": [["m.a", "1.500", NO_VALUE]],
        }
    ]
    assert [c["channel"] for c in view["charts"]] == ["eff.SNH", "eff.SNO"]
    snh, sno = (ET.fromstring(c["svg"]) for c in view["charts"])
    assert snh.get("aria-label") == "SNH of eff: data and fit"
    assert _points(snh) == [(52.0, 178.0), (230.0, 102.0), (408.0, 26.0)]
    assert len([c for c in snh.iter("circle") if c.get("class") == "ak-chart-dot"]) == 3
    # No data for SNO: the fit alone, with the axis title instead of a legend.
    assert [c for c in sno.iter("circle") if c.get("class") == "ak-chart-dot"] == []
    assert _texts(sno)[-1] == "time"


def test_a_profile_view_charts_the_rise_against_its_threshold():
    view = analysis_view(
        {
            "kind": "profile",
            "summary": {"parameter": "m.a", "mle": 1.0, "ci": [None, 1.2], "delta": 5.0},
            "table": {
                "m.a": [0.0, 1.0, 2.0],
                "loss": [11.0, 1.0, 21.0],
                "delta_loss": [10.0, 0.0, 20.0],
            },
        }
    )
    assert view["lines"] == [
        "Best of the grid: m.a = 1.000. 95 % interval: below the grid to 1.200 "
        "(where the curve stays under the threshold)."
    ]
    assert view["tables"] == [
        {
            "title": "Profile",
            "columns": ["m.a", "loss", "rise from the best"],
            "rows": [["0", "11.00", "10.00"], ["1.000", "1.000", "0"], ["2.000", "21.00", "20.00"]],
        }
    ]
    (chart,) = view["charts"]
    assert chart["channel"] == "m.a"
    svg = ET.fromstring(chart["svg"])
    # y ticks 0, 10, 20 (20/4 = 5 rounds up to 10); the threshold at 5 sits a
    # quarter of the way up: 178 - 152/4 = 140.
    (ref,) = [g for g in svg.iter("line") if g.get("class") == "ak-chart-ref"]
    assert float(ref.get("y1")) == 140.0
    assert _points(svg) == [(52.0, 102.0), (230.0, 178.0), (408.0, 26.0)]
    assert "m.a" in _texts(svg)  # the x-axis title


def test_a_profile_of_a_parameter_without_a_table_has_no_chart():
    view = analysis_view({"kind": "profile", "summary": {"parameter": "m.a", "mle": None}})
    assert view["charts"] == [] and view["tables"] == []
    assert view["lines"][0].startswith(f"Best of the grid: m.a = {NO_VALUE}.")
