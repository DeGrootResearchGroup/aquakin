"""The results view's tables and charts (``aquakin_ui.results``), from results
shaped as the worker's ``study_results`` gives them."""

import pytest

from aquakin_ui.results import MAX_HOVER, NO_VALUE, channels_of, fmt, line_chart, steady_table


def test_values_read_at_four_significant_figures():
    assert [fmt(v) for v in (None, 0, 18061.2, 385.0, 2.5217, 0.12281, -47.86)] == [
        NO_VALUE,
        "0",
        "18,061",
        "385.0",
        "2.522",
        "0.1228",
        "-47.86",
    ]
    assert fmt(2.5e-5) == "2.500e-05" and fmt(3e12) == "3.000e+12"


def test_a_steady_state_is_a_table_of_channels_by_stream():
    values = {"effluent": {"Q": 18061.0, "TN": 14.13}, "wastage": {"Q": 385.0, "TSS": 5371.0}}
    units = {"effluent": {"Q": "m³/d", "TN": "g_N/m³"}, "wastage": {"TSS": "g/m³"}}
    assert steady_table(values, units) == {
        "columns": ["effluent", "wastage"],
        "rows": [
            {"channel": "Q", "unit": "m³/d", "values": ["18,061", "385.0"]},
            {"channel": "TN", "unit": "g_N/m³", "values": ["14.13", NO_VALUE]},
            {"channel": "TSS", "unit": "g/m³", "values": [NO_VALUE, "5,371"]},
        ],
    }


RUN = {
    "t": [0.0, 0.5, 1.0, 1.5, 2.0],
    "time_unit": "d",
    "units": {"tank.out": {"SNH": "g_N/m³", "Q": "m³/d"}},
    "measured": {"tank.out": {"TN": [20.0, 15.0, 12.5, 12.0, 11.9]}},
    "streams": {
        "tank.out": {"Q": [2000.0] * 5, "SNH": [30.0, 10.0, None, 4.0, 3.9], "TN": [0] * 5}
    },
}


def test_a_run_in_time_is_a_chart_per_channel():
    assert channels_of(RUN, "tank.out") == ["TN", "Q", "SNH"]
    svg = line_chart(RUN, "tank.out", "SNH")
    assert svg.startswith('<svg class="ak-chart"') and svg.endswith("</svg>")
    assert "SNH (g_N/m³)" in svg and "time (d)" in svg
    assert svg.count("<polyline") == 1 and svg.count("<title>") == 4  # the gap is skipped
    assert "t = 1.500 d: 4.000 g_N/m³" in svg
    # A flat series sits in the middle of its axis, labelled in full.
    flat = line_chart(RUN, "tank.out", "Q")
    assert ">2,000<" in flat and "1,900" in flat
    with pytest.raises(KeyError, match="no channel 'NH4'"):
        line_chart(RUN, "tank.out", "NH4")


def test_a_long_run_is_thinned_and_an_empty_one_says_so():
    n = 5000
    run = {"t": [i / 100 for i in range(n)], "streams": {"s": {"Q": [float(i) for i in range(n)]}}}
    svg = line_chart(run, "s", "Q")
    assert svg.count("<title>") <= MAX_HOVER + 1
    assert "time (" not in svg  # no unit given, none shown
    empty = line_chart({"t": [0.0], "streams": {"s": {"Q": [None]}}}, "s", "Q")
    assert "no finite values" in empty and "<polyline" not in empty
