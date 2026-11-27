"""Setting up a study from plain settings: the form description, the plant's
choices, and creating, reading and updating a study."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from aquakin.study import (
    StudyError,
    check_study,
    create_study,
    plant_choices,
    read_study,
    run_study,
    simulation_study,
    study_form,
    study_results,
    update_study,
    write_trajectory,
)
from aquakin.study.study import _Study

BSM1 = Path(__file__).parents[2] / "examples" / "projects" / "bsm1"
#: Where a study with a results folder but no finished run stands, as
#: ``study_results`` reports it beside what it read.
NOT_RUN = {"state": "not run", "has_results": False, "reasons": [], "notes": []}


def test_the_form_describes_every_setting_and_kind():
    form = study_form()
    json.dumps(form, allow_nan=False)
    kinds = {k["kind"]: k for k in form["kinds"]}
    assert set(kinds) == {"simulate", "fisher", "calibrate", "profile", "rank", "posterior"}
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
    # A kind that studies a batch reactor shows the reactor's fields instead.
    for kind in kinds.values():
        assert set(kind["reactor_fields"]) <= set(form["fields"])
        assert set(kind["reactor_required"]) <= set(kind["reactor_fields"])
    assert kinds["simulate"]["reactor_fields"] == kinds["rank"]["reactor_fields"] == []
    # A posterior study samples its calibrate study's plant or reactor: no target of its own.
    assert kinds["posterior"]["reactor_fields"] == []
    assert not kinds["posterior"]["flowsheet"]
    assert all(kinds[k]["flowsheet"] for k in kinds if k != "posterior")
    assert kinds["posterior"]["required"] == ["fit"]
    assert form["fields"]["prior_space"]["choices"] == ["transformed", "physical"]
    assert kinds["calibrate"]["reactor_required"] == ["free", "experiments", "observed"]
    for kind in ("calibrate", "profile", "fisher"):
        shown = kinds[kind]["reactor_fields"]
        assert shown[:4] == ["description", "reactor", "experiments", "observed"]
        assert not {"observables", "data", "initial_state", "t_span"} & set(shown)


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
        "sigma": 0.1,
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
    with pytest.raises(StudyError, match=r"data: missing\.csv does not exist"):
        create_study(
            project,
            "bad",
            {
                "kind": "calibrate",
                "free": ["asm1.muH"],
                "observables": [{"stream": "effluent", "channels": ["SNH"]}],
                "data": "missing.csv",
                "sigma": 0.1,
            },
        )
    assert not (project / "studies" / "bad").exists()


def test_an_absolute_flowsheet_is_stored_relative_to_the_study(project):
    """A front end names the spec file it has open; the study keeps the path from
    its own folder, and none at all for the project's flowsheet.yaml."""
    simulate = {"kind": "simulate", "streams": ["effluent"]}
    (project / "plant.yaml").write_text((project / "flowsheet.yaml").read_text())
    other = create_study(project, "other", {**simulate, "flowsheet": str(project / "plant.yaml")})
    assert yaml.safe_load((other / "study.yaml").read_text())["study"]["flowsheet"] == (
        "../../plant.yaml"
    )
    default = create_study(
        project, "default", {**simulate, "flowsheet": str(project / "flowsheet.yaml")}
    )
    assert "flowsheet" not in yaml.safe_load((default / "study.yaml").read_text())["study"]
    update_study(other, {**simulate, "flowsheet": str(project / "flowsheet.yaml")})
    assert read_study(other) == simulate
    # A relative path is the user's own and is kept as written.
    update_study(other, {**simulate, "flowsheet": "../../plant.yaml"})
    assert read_study(other)["flowsheet"] == "../../plant.yaml"


def test_a_flowsheet_on_another_drive_is_kept_absolute(project, monkeypatch):
    """Windows has no relative path between drives (``os.path.relpath`` raises
    ``ValueError``); the study then keeps the absolute path it was given."""
    import aquakin.study.form as form

    def across_drives(path, start):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(form.os.path, "relpath", across_drives)
    elsewhere = str(project / "plant.yaml")
    (project / "plant.yaml").write_text((project / "flowsheet.yaml").read_text())
    folder = create_study(
        project, "elsewhere", {"kind": "simulate", "streams": ["effluent"], "flowsheet": elsewhere}
    )
    assert read_study(folder)["flowsheet"] == elsewhere


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
    with pytest.raises(StudyError, match=r"already holds another final_state\.yaml"):
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
        {"id": 6, "command": "simulation_study", "spec": str(project / "flowsheet.yaml")},
        {"id": 7, "command": "study_results", "path": str(project / "studies" / "s")},
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
    by_id = {r["id"]: r for r in replies}
    assert all(r["ok"] for i, r in by_id.items() if i != 7), replies
    assert "has no results; run it first" in by_id[7]["error"]
    assert by_id[4]["result"] == {"kind": "simulate", "streams": ["effluent"]}
    assert by_id[6]["result"]["streams"] == ["effluent", "wastage"]
    assert read_study(project / "studies" / "s")["streams"] == ["ras"]


def test_a_plain_simulation_saves_what_leaves_the_plant():
    study = simulation_study(BSM1 / "flowsheet.yaml")
    # The effluent and the wasted sludge leave BSM1 (by their stream names);
    # the recycles do not.
    assert study["kind"] == "simulate" and study["streams"] == ["effluent", "wastage"]
    assert study["observables"][0] == {
        "stream": "effluent",
        "channels": ["Q", "COD", "TKN", "NOx", "TN", "TSS"],
    }
    _Study.model_validate(study)


def test_the_results_of_a_steady_simulation_read_with_units(project):
    folder = create_study(project, "simulation", simulation_study(project / "flowsheet.yaml"))
    with pytest.raises(StudyError, match="has no results; run it first"):
        study_results(folder)
    run_study(folder)
    results = study_results(folder)
    json.dumps(results, allow_nan=False)
    assert results["kind"] == "simulate" and results["steady"]
    assert "measured.yaml" in results["files"]
    effluent = results["measured"]["effluent"]
    assert set(effluent) == {"Q", "COD", "TKN", "NOx", "TN", "TSS"}
    assert effluent["TN"] == pytest.approx(
        results["streams"]["effluent"]["SNO"] + effluent["TKN"], rel=1e-9
    )
    units = results["units"]["effluent"]
    assert units["Q"] == "m³/d" and units["SNH"] == units["TN"] == "g_N/m³"
    assert units["COD"] == "g_COD/m³" and units["TSS"] == "g/m³"
    (folder / "results" / "measured.yaml").unlink()  # only the streams were saved
    assert "measured" not in study_results(folder)


def test_a_plant_nothing_leaves_simulates_its_named_streams(tmp_path):
    import aquakin
    from aquakin.plant import CSTRUnit, InfluentSeries, Plant
    from aquakin.study.form import _measure_units

    m = aquakin.load_model("asm1")
    p = Plant("loop", time_unit="d")
    p.add_unit(CSTRUnit("a", m, 1000.0, ["in", "back"], conditions={"T": 293.15}))
    p.add_unit(CSTRUnit("b", m, 1000.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(m, Q=100.0, SS=60.0), to="a.in")
    p.connect("a.out", "b.in")
    p.connect("b.out", "a.back")
    p.to_spec(tmp_path / "flowsheet.yaml")
    with pytest.raises(StudyError, match="no stream leaves the plant"):
        simulation_study(tmp_path / "flowsheet.yaml")
    p.register_stream("loop", "b.out")
    p.to_spec(tmp_path / "named.yaml")
    assert simulation_study(tmp_path / "named.yaml")["streams"] == ["loop"]
    # Without COD species there is no COD (nor TSS) unit to give.
    assert _measure_units(["mol/L", "g_N/m3"]) == {"TKN": "g_N/m3", "NOx": "g_N/m3", "TN": "g_N/m3"}


def test_the_results_of_a_run_in_time_read_by_stream(project):
    import math

    from aquakin.study import write_trajectory

    folder = create_study(
        project,
        "run",
        {
            "kind": "simulate",
            "time": {"t_span": [0.0, 1.0], "time_unit": "h"},
            "streams": ["clarifier.overflow"],
            "observables": [{"stream": "effluent", "channels": ["TN"]}],
        },
    )
    out = folder / "results"
    write_trajectory(
        out / "streams.csv",
        [0.0, 1.0],
        {
            "clarifier.overflow.Q": [1.0, 2.0],
            "clarifier.overflow.SNH": [3.0, math.nan],
            "clarifier.overflow.T": [293.15, 288.15],
        },
    )
    write_trajectory(
        out / "measured.csv", [0.0, 1.0], {"effluent.TN": [5.0, 6.0], "gone.Q": [1.0, 1.0]}
    )
    results = study_results(folder)
    json.dumps(results, allow_nan=False)
    assert not results["steady"] and results["t"] == [0.0, 1.0] and results["time_unit"] == "h"
    overflow = results["streams"]["clarifier.overflow"]
    assert overflow["Q"] == [1.0, 2.0] and overflow["SNH"] == [3.0, None]
    assert overflow["T"] == pytest.approx([20.0, 15.0])  # kelvin on file, shown in °C
    assert results["units"]["clarifier.overflow"]["T"] == "°C"
    # A column of a stream the study no longer names is still read by its name.
    assert results["measured"] == {"effluent": {"TN": [5.0, 6.0]}, "gone": {"Q": [1.0, 1.0]}}
    (out / "measured.csv").unlink()
    assert "measured" not in study_results(folder)
    # Another kind gives its files only; a plant that no longer builds, no units.
    (project / "flowsheet.yaml").write_text("not a flowsheet")
    assert study_results(folder)["units"] == {}
    fisher = folder.parent / "fisher"
    fisher.mkdir()
    (fisher / "study.yaml").write_text(
        "study: {kind: fisher, free: [asm1.muH], observables: [{stream: effluent}], "
        "t_obs: [1.0], sigma: 1.0}\n"
    )
    (fisher / "results").mkdir()
    assert study_results(fisher) == {"kind": "fisher", "files": [], "summary": None, **NOT_RUN}


def test_the_results_of_each_analysis_read_as_plain_data(project):
    from aquakin.study import write_matrix, write_table

    common = {"free": ["asm1.muH"], "observables": [{"stream": "effluent", "channels": ["SNH"]}]}
    data = project / "obs.csv"
    write_trajectory(data, [0.5, 1.0], {"effluent.SNH": [1.0, 1.2], "effluent.TN": [9.0, 9.0]})

    fisher = create_study(project, "f", {"kind": "fisher", **common, "t_obs": [1.0], "sigma": 0.1})
    (fisher / "results").mkdir()
    assert study_results(fisher) == {"kind": "fisher", "files": [], "summary": None, **NOT_RUN}
    write_matrix(fisher / "results" / "fisher.csv", np.array([[4.0]]), ["asm1.muH"])
    (fisher / "results" / "summary.yaml").write_text("values: {asm1.muH: 6.0}\nstd: null\n")
    found = study_results(fisher)
    assert found["matrix"] == {"names": ["asm1.muH"], "rows": [[4.0]]}
    assert found["summary"] == {"values": {"asm1.muH": 6.0}, "std": None}

    fit = create_study(
        project, "c", {"kind": "calibrate", **common, "data": str(data), "sigma": 0.1}
    )
    (fit / "results").mkdir()
    assert study_results(fit) == {  # no results yet: the data alone
        "kind": "calibrate",
        "files": [],
        "summary": None,
        "data": {"t": [0.5, 1.0], "values": {"effluent": {"SNH": [1.0, 1.2]}}},
        **NOT_RUN,
    }
    write_trajectory(fit / "results" / "fitted.csv", [0.5, 1.0], {"effluent.SNH": [1.05, math.inf]})
    (fit / "results" / "fit.yaml").write_text("fitted: {asm1.muH: 5.9}\nconverged: true\n")
    found = study_results(fit)
    assert found["fitted"] == {"t": [0.5, 1.0], "values": {"effluent": {"SNH": [1.05, None]}}}
    assert found["data"]["values"] == {"effluent": {"SNH": [1.0, 1.2]}}  # TN not measured
    assert found["summary"]["converged"] is True
    (fit / "obs.csv").unlink()  # the data gone: the fit alone
    assert "data" not in study_results(fit)

    prof = create_study(
        project,
        "p",
        {"kind": "profile", **common, "data": str(data), "sigma": 0.1, "profile": "asm1.muH",
         "grid": [5.0, 6.0]},
    )  # fmt: skip
    (prof / "results").mkdir()
    assert study_results(prof) == {"kind": "profile", "files": [], "summary": None, **NOT_RUN}
    write_table(
        prof / "results" / "profile.csv", {"asm1.muH": [5.0, 6.0], "delta_loss": [2.0, 0.0]}
    )
    (prof / "results" / "profile.yaml").write_text("parameter: asm1.muH\nci: [5.5, .inf]\n")
    found = study_results(prof)
    assert found["table"]["delta_loss"] == [2.0, 0.0] and found["summary"]["ci"] == [5.5, None]

    rank = create_study(
        project,
        "r",
        {"kind": "rank", "free": ["asm1.muH"], "t_obs": [1.0], "criterion": "A",
         "candidates": [{"name": "a", "observables": common["observables"], "sigma": 0.1}]},
    )  # fmt: skip
    (rank / "results").mkdir()
    assert study_results(rank) == {"kind": "rank", "files": [], "criterion": "A", **NOT_RUN}
    write_table(rank / "results" / "ranking.csv", {"rank": [1], "candidate": ["a"], "A": [0.5]})
    found = study_results(rank)
    assert found["criterion"] == "A" and found["table"]["candidate"] == ["a"]


def test_a_reactor_study_is_created_with_its_data_and_model_where_a_run_reads_them(
    project, tmp_path
):
    """Each experiment's table is copied in; a model file named from the project
    or by an absolute path is stored relative to the study folder, one already
    relative to it is kept, and a shipped model keeps its name."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for run in ("run1", "run2"):
        (elsewhere / f"{run}.csv").write_text("t,SNH\n0.5,1.0\n1.0,0.9\n")
    (project / "models").mkdir()
    (project / "models" / "local.yaml").write_text("model:\n  name: local\n  extends: asm1\n")
    study = {
        "kind": "calibrate",
        "reactor": {"model": "models/local.yaml", "conditions": {"T": 288.15}},
        "experiments": [
            {"data": str(elsewhere / "run1.csv"), "initial": {"SNH": 20.0}},
            {"data": str(elsewhere / "run2.csv")},
        ],
        "observed": ["SNH"],
        "free": ["local.muA"],
        "sigma": 0.1,
    }
    folder = create_study(project, "reactor", study)
    stored = read_study(folder)
    assert stored["reactor"] == {"model": "../../models/local.yaml", "conditions": {"T": 288.15}}
    assert [e["data"] for e in stored["experiments"]] == ["run1.csv", "run2.csv"]
    assert (folder / "run2.csv").read_text() == (elsewhere / "run2.csv").read_text()
    assert check_study(folder).state == "not run"

    absolute = {**stored, "reactor": {"model": str(project / "models" / "local.yaml")}}
    update_study(folder, absolute)
    assert read_study(folder)["reactor"] == {"model": "../../models/local.yaml"}
    update_study(folder, stored)  # relative to the study folder already: kept
    assert read_study(folder) == stored
    shipped = {**stored, "reactor": {"model": "asm1"}, "free": ["asm1.muA"]}
    update_study(folder, shipped)
    assert read_study(folder)["reactor"] == {"model": "asm1"}

    fisher = {
        "kind": "fisher",
        "reactor": {"model": "asm1"},
        "experiments": [{"initial": {"SNH": 20.0}}],
        "observed": ["SNH"],
        "free": ["asm1.muA"],
        "t_obs": [0.5],
        "sigma": 0.1,
    }
    assert read_study(create_study(project, "design", fisher)) == fisher

    with pytest.raises(StudyError, match=r"reactor.model: models/gone.yaml does not exist"):
        create_study(project, "gone", {**study, "reactor": {"model": "models/gone.yaml"}})
    assert not (project / "studies" / "gone").exists()


def test_a_reactor_model_on_another_drive_is_kept_absolute(project, monkeypatch):
    import aquakin.study.form as form

    def across_drives(path, start):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(form.os.path, "relpath", across_drives)
    model = project / "models" / "local.yaml"
    model.parent.mkdir()
    model.write_text("model:\n  name: local\n  extends: asm1\n")
    study = {
        "kind": "fisher",
        "reactor": {"model": str(model)},
        "experiments": [{"initial": {}}],
        "observed": ["SNH"],
        "free": ["local.muA"],
        "t_obs": [0.5],
        "sigma": 0.1,
    }
    folder = create_study(project, "far", study)
    assert read_study(folder)["reactor"] == {"model": str(model)}
