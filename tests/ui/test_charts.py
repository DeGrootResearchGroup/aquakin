"""``xy_chart`` drawn to hand-computed geometry, and each study kind's view
against a hand-built structure.

The chart is 420 x 200 with margins left 52, right 12, top 26, bottom 22, so the
plot area is 356 x 152: x runs from 52 to 408 and y from 178 (the lowest tick)
up to 26 (the highest).
"""

import xml.etree.ElementTree as ET

from aquakin_ui.results import xy_chart


def _svg(**kwargs):
    return ET.fromstring(xy_chart(**kwargs))


def _texts(svg, cls="ak-chart-axis"):
    return [t.text for t in svg.iter("text") if t.get("class") == cls]


def _points(svg):
    (line,) = [p for p in svg.iter("polyline")]
    return [tuple(map(float, p.split(","))) for p in line.get("points").split()]


DIAGONAL = [(0.0, 0.0), (10.0, 10.0)]


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
