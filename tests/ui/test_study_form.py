"""The study dialog's draft (``aquakin_ui.study_form``): settings to editable
values and back, for every kind, against the form aquakin describes."""

import pytest

from aquakin.study import study_form
from aquakin.study.study import _check_kind, _Study
from aquakin_ui.study_form import draft_for, fields_for, missing, study_from

FORM = study_form()

STUDIES = [
    {
        "kind": "simulate",
        "description": "Two weeks",
        "parameters": {"asm1.muH": 5.0},
        "initial_state": "steady",
        "time": {"t_span": [0.0, 14.0], "t_eval": {"start": 0.0, "stop": 14.0, "num": 57}},
        "streams": ["effluent"],
        "observables": [
            {"stream": "effluent", "channels": ["TN", {"measure": "half_SS", "weights": {"SS": 0.5}}]},
            {"stream": "tank3.out"},
        ],
    },
    {
        "kind": "fisher",
        "free": ["asm1.muH", "asm1.KS"],
        "observables": [{"stream": "effluent", "channels": ["SNH", "COD"]}],
        "t_obs": [0.5, 1.0, 1.5],
        "t_span": [0.0, 2.0],
        "sigma": [0.2, 2.0],
        "relative_sigma": True,
        "sigma_floor": 0.01,
        "sensitivity": "reverse",
    },
    {
        "kind": "calibrate",
        "initial_state": "../seed.yaml",
        "free": ["asm1.muH"],
        "observables": [{"stream": "effluent", "channels": ["SNH"]}],
        "data": "obs.csv",
        "loss": "nll",
        "sigma": 0.5,
        "transforms": {"asm1.muH": "positive_log"},
        "use_priors": False,
        "n_starts": 3,
        "laplace": True,
    },
    {
        "kind": "profile",
        "free": ["asm1.muH", "asm1.KS"],
        "observables": [{"stream": "effluent", "channels": ["SNH"]}],
        "data": "obs.csv",
        "estimate_sigma": True,
        "profile": "asm1.muH",
        "grid": {"start": 3.0, "stop": 6.0, "num": 7},
        "delta": 3.0,
    },
    {
        "kind": "rank",
        "free": ["asm1.muH"],
        "candidates": [
            {"name": "ammonia", "observables": [{"stream": "effluent", "channels": ["SNH"]}], "sigma": 0.2},
            {"name": "both", "observables": [{"stream": "effluent", "channels": ["SNH", "SNO"]}], "sigma": [0.2, 0.5]},
        ],
        "base": {"name": "online", "observables": [{"stream": "effluent", "channels": ["Q"]}], "sigma": 10.0},
        "t_obs": {"start": 0.1, "stop": 1.0, "num": 10},
        "criterion": "A",
    },
]  # fmt: skip


@pytest.mark.parametrize("study", STUDIES, ids=[s["kind"] for s in STUDIES])
def test_a_study_round_trips_through_its_draft(study):
    draft = draft_for(FORM, study["kind"], study)
    assert study_from(FORM, study["kind"], draft) == study
    checked = _Study.model_validate(study)
    _check_kind(checked, "here")


def test_a_new_draft_gives_the_defaults_and_a_kind_change_keeps_what_applies():
    draft = draft_for(FORM, "fisher")
    assert study_from(FORM, "fisher", draft) == {"kind": "fisher"}
    assert draft["sensitivity"] == "forward" and draft["t_obs"]["mode"] == "grid"
    fisher = STUDIES[1]
    as_rank = study_from(FORM, "rank", draft_for(FORM, "rank", fisher))
    assert as_rank["free"] == fisher["free"] and "observables" not in as_rank
    assert [f["name"] for f in fields_for(FORM, "rank") if f["required"]] == [
        "free",
        "candidates",
        "t_obs",
    ]


def test_edited_drafts_read_back_and_bad_text_is_named():
    draft = draft_for(FORM, "fisher")
    draft["free"] = ["asm1.muH", ""]
    draft["observables"] = [
        {"stream": "effluent", "channels": ["SNH"]},
        {"stream": "", "channels": []},
    ]
    draft["t_obs"] = {"mode": "list", "start": "", "stop": "", "num": "", "list": "0.5; 1, 2"}
    draft["sigma"] = " 0.3 "
    draft["parameters"] = [{"name": "asm1.KS", "value": "12"}, {"name": "", "value": ""}]
    study = study_from(FORM, "fisher", draft)
    assert study == {
        "kind": "fisher",
        "parameters": {"asm1.KS": 12.0},
        "free": ["asm1.muH"],
        "observables": [{"stream": "effluent", "channels": ["SNH"]}],
        "t_obs": [0.5, 1.0, 2.0],
        "sigma": 0.3,
    }
    for change, message in (
        ({"sigma": "small"}, "Measurement error: 'small' is not a number"),
        ({"sigma": "nan"}, "is not a finite number"),
        (
            {"t_obs": {**draft["t_obs"], "mode": "grid", "start": "1", "stop": "", "num": ""}},
            "give the start",
        ),
        (
            {"t_obs": {**draft["t_obs"], "mode": "grid", "start": "0", "stop": "1", "num": "2.5"}},
            "not a whole number",
        ),
        ({"parameters": [{"name": "asm1.KS", "value": "x"}]}, "Parameters asm1.KS"),
    ):
        with pytest.raises(ValueError, match=message):
            study_from(FORM, "fisher", {**draft, **change})


def test_initial_states_runs_and_sensor_sets_that_are_incomplete():
    draft = draft_for(FORM, "simulate")
    draft["initial_state"] = {"mode": "file", "file": ""}
    with pytest.raises(ValueError, match="give the file of the state"):
        study_from(FORM, "simulate", draft)
    draft = draft_for(FORM, "simulate")
    draft["time"] = {**draft["time"], "dynamic": True, "start": "0", "stop": "2", "unit": "h"}
    assert study_from(FORM, "simulate", draft)["time"] == {"t_span": [0.0, 2.0], "time_unit": "h"}

    draft = draft_for(FORM, "rank")
    draft["candidates"] = [
        {"name": "", "sigma": "", "observables": []},  # an empty row is ignored
        {"name": "", "sigma": "1", "observables": [{"stream": "effluent", "channels": ["SNH"]}]},
    ]
    with pytest.raises(ValueError, match="give each sensor set a name"):
        study_from(FORM, "rank", draft)
    draft["candidates"][1]["name"], draft["candidates"][1]["sigma"] = "a", ""
    with pytest.raises(ValueError, match="Candidates a: give its measurement error"):
        study_from(FORM, "rank", draft)


def test_what_a_study_still_needs_is_named_by_label():
    assert missing(FORM, "fisher", {"kind": "fisher"}) == (
        "Still needed: Free parameters, Measured channels, Sampling times, Measurement error."
    )
    for study in STUDIES:
        assert missing(FORM, study["kind"], study) is None


def test_an_unknown_field_type_is_refused_and_an_empty_number_left_out():
    form = {
        "kinds": [{"kind": "k", "fields": ["x", "n"], "required": []}],
        "fields": {
            "x": {"type": "colour", "label": "X", "default": None, "help": "."},
            "n": {"type": "number", "label": "N", "default": None, "help": "."},
        },
    }
    with pytest.raises(ValueError, match="unknown field type 'colour'"):
        draft_for(form, "k")
    with pytest.raises(ValueError, match="unknown field type 'colour'"):
        study_from(form, "k", {"x": "red", "n": " "})
    assert study_from(
        {**form, "kinds": [{"kind": "k", "fields": ["n"], "required": []}]}, "k", {"n": " "}
    ) == {"kind": "k"}
