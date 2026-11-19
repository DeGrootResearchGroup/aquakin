"""Setting up a study from plain settings: the form description, the plant's
choices, and creating, reading and updating a study."""

import json
from pathlib import Path

import pytest
import yaml

from aquakin.study import (
    StudyError,
    check_study,
    create_study,
    plant_choices,
    read_study,
    run_study,
    study_form,
    update_study,
)
from aquakin.study.study import _Study

BSM1 = Path(__file__).parents[2] / "examples" / "projects" / "bsm1"


def test_the_form_describes_every_setting_and_kind():
    form = study_form()
    json.dumps(form, allow_nan=False)
    kinds = {k["kind"]: k for k in form["kinds"]}
    assert set(kinds) == {"simulate", "fisher", "calibrate", "profile", "rank"}
    # Every setting of study.yaml (but where the flowsheet sits) is in the form,
    # and every field a kind shows is described.
    settings = set(_Study.model_fields) - {"kind", "flowsheet"}
    assert set(form["fields"]) == settings
    for kind in kinds.values():
        assert set(kind["fields"]) <= set(form["fields"])
        assert set(kind["required"]) <= set(kind["fields"])
        assert kind["fields"][0] == "description"
    assert form["fields"]["criterion"]["choices"] == ["D", "A", "E", "modified_E"]
    for name, field in form["fields"].items():
        assert field["help"].endswith("."), name


def test_the_plant_offers_its_streams_parameters_and_channels():
    choices = plant_choices(BSM1 / "flowsheet.yaml")
    json.dumps(choices, allow_nan=False)
    assert choices["streams"][:4] == ["effluent", "internal_recycle", "ras", "wastage"]
    assert "clarifier.overflow" in choices["streams"]
    assert "asm1.muH" in choices["parameters"]
    effluent = choices["channels"]["effluent"]
    assert effluent[:6] == ["Q", "COD", "TKN", "NOx", "TN", "TSS"]  # no T: no influent T
    assert "SNH" in effluent and "TP" not in effluent


@pytest.fixture
def project(tmp_path):
    import shutil

    shutil.copytree(BSM1, tmp_path / "bsm1")
    return tmp_path / "bsm1"


def test_a_study_is_created_read_and_updated(project, tmp_path):
    data = tmp_path / "elsewhere" / "obs.csv"
    data.parent.mkdir()
    data.write_text("t,effluent.SNH\n0.5,1.0\n1.0,1.1\n")
    study = {
        "kind": "calibrate",
        "description": "Fit muH",
        "free": ["asm1.muH"],
        "observables": [{"stream": "effluent", "channels": ["SNH"]}],
        "data": str(data),
        "max_iter": 20,
    }
    folder = create_study(project, "fit", study)
    assert (folder / "obs.csv").read_text() == data.read_text()  # copied in
    written = yaml.safe_load((folder / "study.yaml").read_text())["study"]
    assert written["data"] == "obs.csv" and written["max_iter"] == 20
    assert read_study(folder) == {**study, "data": "obs.csv"}
    assert check_study(folder).state == "not run"

    update_study(folder, {**read_study(folder), "max_iter": 50, "description": "Fit it"})
    assert read_study(folder)["max_iter"] == 50
    assert (folder / "obs.csv").exists()  # the copy is kept, not needed again


def test_bad_studies_are_refused_and_leave_nothing(project):
    simulate = {"kind": "simulate", "streams": ["effluent"]}
    with pytest.raises(StudyError, match="study name 'a b'"):
        create_study(project, "a b", simulate)
    create_study(project, "steady", simulate)
    with pytest.raises(StudyError, match="a study named steady already exists"):
        create_study(project, "steady", simulate)
    with pytest.raises(StudyError, match="max_iter"):
        create_study(project, "bad", {**simulate, "max_iter": "many"})
    with pytest.raises(StudyError, match="a fisher study needs"):
        create_study(project, "bad", {"kind": "fisher"})
    with pytest.raises(StudyError, match="data: missing.csv does not exist"):
        create_study(
            project,
            "bad",
            {
                "kind": "calibrate",
                "free": ["asm1.muH"],
                "observables": [{"stream": "effluent", "channels": ["SNH"]}],
                "data": "missing.csv",
            },
        )
    assert not (project / "studies" / "bad").exists()


def test_an_initial_state_from_another_study_is_copied_in(project):
    first = create_study(project, "first", {"kind": "simulate", "streams": ["effluent"]})
    run_study(first)
    second = create_study(
        project,
        "second",
        {
            "kind": "simulate",
            "streams": ["effluent"],
            "initial_state": "studies/first/results/final_state.yaml",
        },
    )
    assert read_study(second)["initial_state"] == "final_state.yaml"
    assert (second / "final_state.yaml").exists()
    # A file of the same name but other content is not overwritten.
    (first / "results" / "final_state.yaml").write_text("tank1: {SO: 1.0}\n")
    with pytest.raises(StudyError, match="already holds another final_state.yaml"):
        update_study(
            second,
            {**read_study(second), "initial_state": "studies/first/results/final_state.yaml"},
        )


def test_the_worker_answers_the_form_commands(project):
    import io

    from aquakin.__main__ import serve

    requests = [
        {"id": 1, "command": "study_form"},
        {"id": 2, "command": "plant_choices", "spec": str(project / "flowsheet.yaml")},
        {
            "id": 3,
            "command": "create_study",
            "project": str(project),
            "name": "s",
            "study": {"kind": "simulate", "streams": ["effluent"]},
        },
        {"id": 4, "command": "read_study", "path": str(project / "studies" / "s")},
        {
            "id": 5,
            "command": "update_study",
            "path": str(project / "studies" / "s"),
            "study": {"kind": "simulate", "streams": ["ras"]},
        },
    ]
    out = io.StringIO()
    serve(io.StringIO("".join(json.dumps(r) + "\n" for r in requests)), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert all(r["ok"] for r in replies), replies
    assert replies[3]["result"] == {"kind": "simulate", "streams": ["effluent"]}
    assert read_study(project / "studies" / "s")["streams"] == ["ras"]
