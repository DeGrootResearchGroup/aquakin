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
    # Large values on a fine step are labelled in full, not in thousands.
    big = {"t": [0.0, 1.0], "streams": {"s": {"Q": [18061.0, 18064.0]}}}
    assert ">18,062<" in line_chart(big, "s", "Q")
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


def test_each_analysis_is_shown_as_tables_and_charts():
    from aquakin_ui.results import analysis_view

    fisher = analysis_view(
        {
            "kind": "fisher",
            "summary": {"values": {"m.a": 2.0}, "std": None, "criteria": {"D": None, "A": 1.5}},
            "matrix": {"names": ["m.a"], "rows": [[0.0]]},
        }
    )
    assert "singular" in fisher["lines"][0]
    assert fisher["tables"][0]["rows"] == [["m.a", "2.000", NO_VALUE]]
    assert fisher["tables"][2]["columns"] == ["", "m.a"]

    rank = analysis_view(
        {
            "kind": "rank",
            "criterion": "A",
            "table": {"candidate": ["b", "a"], "A": [0.5, 2.0], "improvement": [None, None]},
        }
    )
    assert rank["tables"][0]["rows"] == [
        ["1", "b", "0.5000", NO_VALUE],
        ["2", "a", "2.000", NO_VALUE],
    ]

    calibrate = analysis_view(
        {
            "kind": "calibrate",
            "summary": {
                "fitted": {"m.a": 1.004},
                "std": {"m.a": 0.0185},
                "converged": True,
                "iterations": 4,
                "message": "ok",
                "sigma_estimate": {"eff.SNH": 0.3},
            },
            "fitted": {"t": [0.0, 1.0], "values": {"eff": {"SNH": [1.0, 2.0]}}},
            "data": {"t": [0.0, 1.0], "values": {"eff": {"SNH": [1.1, None]}}},
        }
    )
    assert calibrate["lines"] == ["The fit converged after 4 iterations (ok)."]
    assert calibrate["tables"][1]["title"] == "Estimated measurement error"
    svg = calibrate["charts"][0]["svg"]
    assert svg.count('class="ak-chart-dot"') == 2  # one data point, and the legend's
    assert ">data<" in svg and ">fit<" in svg

    profile = analysis_view(
        {
            "kind": "profile",
            "summary": {"parameter": "m.a", "mle": 1.0, "ci": [0.9, None], "delta": 1.92},
            "table": {
                "m.a": [0.8, 1.0, 1.2],
                "loss": [3.0, 1.0, 1.5],
                "delta_loss": [2.0, 0.0, 0.5],
            },
        }
    )
    assert "0.9000 to above the grid" in profile["lines"][0]
    assert 'class="ak-chart-ref"' in profile["charts"][0]["svg"]
    assert profile["tables"][0]["rows"][1] == ["1.000", "1.000", "0"]
    # A table without its loss columns gives an empty profile table, not an error.
    bare = analysis_view(
        {"kind": "profile", "summary": {"parameter": "m.a"}, "table": {"m.a": [0.8, 1.0]}}
    )
    assert bare["tables"][0]["rows"] == []
    # A study that did not converge says so; one without a summary says nothing.
    assert (
        "did not converge"
        in analysis_view(
            {"kind": "calibrate", "summary": {"converged": False, "iterations": 9, "message": "x"}}
        )["lines"][0]
    )
    assert analysis_view({"kind": "calibrate"}) == {"lines": [], "tables": [], "charts": []}


def test_a_posterior_shows_its_parameters_diagnostics_and_histograms():
    from aquakin_ui.results import ESS_FLOOR, R_HAT_LIMIT, analysis_view

    parameter = {
        "optimum": 2.0,
        "mean": 2.01,
        "std": 0.05,
        "q05": 1.92,
        "q50": 2.01,
        "q95": 2.09,
        "r_hat": 1.002,
        "ess_bulk": 840.0,
    }
    summary = {
        "fit": "fit",
        "n_chains": 4,
        "n_draws": 1000,
        "prior_space": "transformed",
        "acceptance_rate": [0.3, 0.25],
        "parameters": {"m.a": parameter},
    }
    view = analysis_view(
        {
            "kind": "posterior",
            "summary": summary,
            "histograms": {"m.a": {"edges": [1.9, 2.0, 2.1], "counts": [3, 7]}},
        }
    )
    assert view["lines"] == [
        "4 chains of 1000 draws each, sampling the fit of fit (prior flat in the transformed "
        "coordinates). Acceptance rate per chain: 0.3000, 0.2500."
    ]
    assert view["tables"][0]["columns"][0] == "parameter"
    assert view["tables"][0]["rows"] == [
        ["m.a", "2.000", "2.010", "0.05000", "1.920", "2.010", "2.090", "1.002", "840.0"]
    ]
    (chart,) = view["charts"]
    assert chart["channel"] == "m.a" and "Posterior of m.a" in chart["svg"]

    # Draws that fail the diagnostics are flagged, naming each parameter.
    poor = {**parameter, "r_hat": R_HAT_LIMIT + 0.05, "ess_bulk": ESS_FLOOR - 1}
    flagged = analysis_view(
        {"kind": "posterior", "summary": {**summary, "parameters": {"m.a": poor, "m.b": parameter}}}
    )
    assert flagged["lines"][1] == (
        "Not enough to rely on yet: R-hat above 1.01 for m.a; fewer than 100 effective draws "
        "for m.a. Run it again with more draws."
    )
    assert flagged["charts"] == []


def test_a_design_shows_its_criterion_experiments_and_standard_deviations():
    from aquakin_ui.results import NO_VALUE, analysis_view

    summary = {
        "criterion": "D",
        "subset": [],
        "start": 10.8,
        "optimum": 13.2,
        "converged": True,
        "message": "Optimization terminated successfully",
        "iterations": 8,
        "std": {"r.k": {"start": 0.0045, "optimum": 0.00136}, "r.k2": {"start": None, "optimum": 0.2}},
        "experiments": [{"initial": {"A": 2.0, "B": 0.5}, "t_obs": [0.1667, 1.0]}],
    }
    view = analysis_view({"kind": "design", "summary": summary})
    assert view["lines"][0] == (
        "D criterion, scoring every free parameter: 10.80 at the start, 13.20 at the optimum. "
        "The optimizer converged after 8 iterations (Optimization terminated successfully)."
    )
    experiments, std = view["tables"]
    assert experiments["rows"] == [["1", "A 2.000, B 0.5000", "0.1667, 1.000"]]
    assert std["rows"] == [["r.k", "0.004500", "0.001360"], ["r.k2", NO_VALUE, "0.2000"]]
    scored = analysis_view(
        {"kind": "design", "summary": {**summary, "subset": ["r.k"], "converged": False}}
    )
    assert scored["lines"][0].startswith("D criterion, scoring r.k: ")
    assert "did not converge" in scored["lines"][0]
