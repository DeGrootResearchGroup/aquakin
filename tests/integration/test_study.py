"""Studies: run, check for staleness, freeze, unfreeze and copy.

A small plant (one ASM1 tank fed a constant influent) keeps the runs fast; the
plant comes from a project folder written with ``Plant.to_spec``, with its ASM1
a model file so an edit to it can be made.
"""

import shutil

import numpy as np
import pytest
import yaml

import aquakin
from aquakin.__main__ import main
from aquakin.plant import (
    CSTRUnit,
    InfluentSeries,
    Plant,
    UnknownUnitError,
    plant_fisher_information,
)
from aquakin.study import (
    StudyError,
    check_study,
    copy_study,
    freeze_study,
    read_matrix,
    read_named_values,
    read_trajectory,
    run_study,
    unfreeze_study,
)

FEED = dict(SI=30.0, SS=69.5, XI=51.2, XS=202.32, XB_H=28.17, SNH=31.56, SND=6.95, XND=10.59)


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "project"
    (folder / "models").mkdir(parents=True)
    model_file = folder / "models" / "asm1_local.yaml"
    model_file.write_text("model:\n  name: asm1_local\n  extends: asm1\n")
    m = aquakin.load_model_from_file(model_file)
    p = Plant("tank")
    p.add_unit(CSTRUnit("tank", m, 1000.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(m, Q=2000.0, **FEED), to="tank.in")
    p.register_stream("effluent", "tank.out")
    p.to_spec(folder / "flowsheet.yaml")
    return folder


def _study(project, name, study):
    folder = project / "studies" / name
    folder.mkdir(parents=True)
    (folder / "study.yaml").write_text(yaml.safe_dump({"study": study}))
    return folder


DYNAMIC = {
    "kind": "simulate",
    "time": {"t_span": [0, 0.5], "t_eval": {"start": 0, "stop": 0.5, "num": 6}},
    "streams": ["effluent"],
    "observables": [{"stream": "effluent", "channels": ["TN", "Q"]}],
}


def test_a_dynamic_simulation_saves_named_results(project):
    folder = _study(project, "run1", DYNAMIC)
    assert check_study(folder).state == "not run"
    status = run_study(folder)
    assert status.state == "fresh" and not status.frozen
    t, streams = read_trajectory(folder / "results" / "streams.csv")
    np.testing.assert_allclose(t, np.linspace(0, 0.5, 6))
    assert "effluent.SNH" in streams and "effluent.Q" in streams
    _, measured = read_trajectory(folder / "results" / "measured.csv")
    assert list(measured) == ["effluent.TN", "effluent.Q"]
    np.testing.assert_allclose(measured["effluent.Q"], 2000.0)
    final = read_named_values(folder / "results" / "final_state.yaml")
    assert final["tank"]["SNH"] == pytest.approx(streams["effluent.SNH"][-1])
    params = read_named_values(folder / "results" / "parameters.yaml")
    assert "asm1_local.muH" in params


def test_a_steady_state_simulation_from_a_named_state(project):
    seed = project / "studies" / "seed.yaml"
    seed.parent.mkdir(parents=True)
    seed.write_text(yaml.safe_dump({"tank": {"XB_H": 2000.0, "XB_A": 100.0}}))
    folder = _study(
        project,
        "steady",
        {
            "kind": "simulate",
            "initial_state": "../seed.yaml",
            "parameters": {"asm1_local.muH": 5},
            "streams": ["effluent"],
        },
    )
    run_study(folder)
    effluent = read_named_values(folder / "results" / "streams.yaml")["effluent"]
    final = read_named_values(folder / "results" / "final_state.yaml")["tank"]
    assert effluent["Q"] == pytest.approx(2000.0) and effluent["SNH"] == final["SNH"]
    assert read_named_values(folder / "results" / "parameters.yaml")["asm1_local.muH"] == 5.0
    # The seed file is one of the study's inputs.
    seed.write_text(yaml.safe_dump({"tank": {"XB_H": 1000.0}}))
    status = check_study(folder)
    assert status.state == "stale" and "the study's settings have changed" in status.reasons


def test_a_live_study_goes_stale_and_says_why(project, monkeypatch):
    folder = _study(project, "live", DYNAMIC)
    run_study(folder)
    spec = project / "flowsheet.yaml"
    original = spec.read_text()
    # Layout, labels and number spelling are not what is simulated.
    spec.write_text(original.replace("volume: 1000.0", "volume: 1000"))
    assert check_study(folder).state == "fresh"
    spec.write_text(original.replace("volume: 1000.0", "volume: 1200.0"))
    assert check_study(folder).reasons == ["the flowsheet has changed"]
    spec.write_text(original)
    model = project / "models" / "asm1_local.yaml"
    model.write_text(model.read_text() + "parameters:\n  muH: {value: 5.0}\n")
    assert check_study(folder).reasons == ["model 'asm1_local' has changed"]
    model.write_text("model:\n  name: asm1_local\n  extends: asm1\n")
    assert check_study(folder).state == "fresh"
    # A new aquakin version is noted, not counted as stale.
    import aquakin.plant.spec as spec_module

    monkeypatch.setattr(spec_module, "aquakin_version", lambda: "9.9.9")
    status = check_study(folder)
    assert status.state == "fresh" and "this is aquakin 9.9.9" in status.notes[0]


def test_edited_results_and_a_failed_run_are_reported(project):
    folder = _study(project, "results", DYNAMIC)
    run_study(folder)
    (folder / "results" / "measured.csv").write_text("t\n0\n")
    status = check_study(folder)
    assert status.state == "damaged" and status.reasons == ["results/measured.csv has been edited"]
    with pytest.raises(StudyError, match="already has results"):
        run_study(folder)
    bad = _study(project, "bad", {**DYNAMIC, "parameters": {"asm1_local.nope": 1.0}})
    with pytest.raises(KeyError, match=r"Unknown plant parameter 'asm1_local\.nope'"):
        run_study(bad)
    status = check_study(bad)
    assert status.state == "failed" and "asm1_local.nope" in status.reasons[0]


def test_a_frozen_study_keeps_its_plant(project):
    folder = _study(project, "frozen", DYNAMIC)
    run_study(folder)
    record = freeze_study(folder)
    snapshot = yaml.safe_load((folder / "snapshot" / "flowsheet.yaml").read_text())
    assert snapshot["models"]["asm1_local"]["source"] == "models/asm1_local.yaml"
    flat = (folder / "snapshot" / "models" / "asm1_local.yaml").read_text()
    assert "extends" not in flat and "muH" in flat  # the chain is merged in
    assert check_study(folder).state == "fresh" and check_study(folder).frozen
    assert (
        record["spec_fingerprint"]
        == yaml.safe_load((folder / "run.yaml").read_text())["spec_fingerprint"]
    )
    before = read_trajectory(folder / "results" / "streams.csv")[1]["effluent.SNH"]

    # The project changes; the frozen study does not.
    (project / "models" / "asm1_local.yaml").write_text(
        "model:\n  name: asm1_local\n  extends: asm1\nparameters:\n  muH: {value: 5.0}\n"
    )
    assert check_study(folder).state == "fresh"
    run_study(folder, replace=True)
    after = read_trajectory(folder / "results" / "streams.csv")[1]["effluent.SNH"]
    np.testing.assert_array_equal(after, before)

    with pytest.raises(StudyError, match="deletes its results"):
        unfreeze_study(folder)
    kept = copy_study(folder, "frozen_copy")
    unfreeze_study(folder, discard_results=True)
    assert not (folder / "results").exists() and not (folder / "snapshot").exists()
    assert check_study(folder).state == "not run"
    assert check_study(kept).state == "fresh" and check_study(kept).frozen


def test_a_shipped_model_is_copied_into_the_snapshot(tmp_path):
    folder = tmp_path / "project"
    m = aquakin.load_model("asm1")
    p = Plant("tank")
    p.add_unit(CSTRUnit("tank", m, 1000.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(m, Q=2000.0, **FEED), to="tank.in")
    p.to_spec(folder / "flowsheet.yaml")
    study = _study(folder, "s", {**DYNAMIC, "streams": ["tank"], "observables": []})
    freeze_study(study)
    snapshot = yaml.safe_load((study / "snapshot" / "flowsheet.yaml").read_text())
    assert snapshot["models"]["asm1"]["copy_of"] == "shipped"
    q = Plant.from_spec(study / "snapshot" / "flowsheet.yaml")
    np.testing.assert_array_equal(
        np.asarray(q.default_parameters()), np.asarray(p.default_parameters())
    )
    y0 = p.initial_state()
    np.testing.assert_array_equal(np.asarray(q.derivative(y0)), np.asarray(p.derivative(y0)))


def test_a_fisher_study_is_the_plant_fisher_information(project):
    t_obs = [0.1, 0.2, 0.3]
    folder = _study(
        project,
        "fisher",
        {
            "kind": "fisher",
            "free": ["asm1_local.muH", "asm1_local.KS"],
            "observables": [{"stream": "effluent", "channels": ["SS", "COD"]}],
            "t_obs": t_obs,
            "t_span": [0, 0.3],
            "sigma": 1,
        },
    )
    run_study(folder)
    matrix, rows, cols = read_matrix(folder / "results" / "fisher.csv")
    assert rows == cols == ["asm1_local.muH", "asm1_local.KS"]
    plant = Plant.from_spec(project / "flowsheet.yaml")
    direct = plant_fisher_information(
        plant,
        [("effluent", ["SS", "COD"])],
        np.asarray(t_obs),
        rows,
        sigma=1.0,
        t_span=(0.0, 0.3),
    )
    np.testing.assert_allclose(matrix, direct.matrix, rtol=1e-10)
    summary = yaml.safe_load((folder / "results" / "summary.yaml").read_text())
    assert set(summary["criteria"]) == {"D", "A", "E", "modified_E"}


def test_the_command_line(project, capsys):
    folder = _study(project, "cli", DYNAMIC)
    assert main(["study", "check", str(folder)]) == 0
    assert main(["study", "run", str(folder)]) == 0
    assert main(["study", "run", str(folder)]) == 2  # results there, no --replace
    assert main(["study", "freeze", str(folder)]) == 0
    assert main(["study", "unfreeze", str(folder)]) == 2
    assert main(["study", "unfreeze", str(folder), "--discard-results"]) == 0
    out = capsys.readouterr()
    assert "fresh, frozen" in out.out and "deletes its results" in out.err


def test_a_bad_study_file_says_where(project):
    folder = _study(project, "bad", {"kind": "simulate", "streams": ["effluent"], "tme": {}})
    with pytest.raises(StudyError, match="tme: Extra inputs"):
        run_study(folder)
    folder = _study(project, "nothing", {"kind": "fisher"})
    with pytest.raises(StudyError, match="needs free, observables, t_obs, sigma"):
        check_study(folder)
    shutil.rmtree(folder)
    with pytest.raises(StudyError, match="does not exist"):
        check_study(folder)


def _synthetic_data(project, folder, muH):
    """Effluent SNH and SS of the project plant run at ``muH``, as a data table."""
    from aquakin.study import write_trajectory

    plant = Plant.from_spec(project / "flowsheet.yaml")
    p = plant.parameter_values({"asm1_local.muH": muH})
    t = np.linspace(0.1, 0.5, 5)
    sol = plant.solve((0.0, 0.5), t_eval=t, params=p)
    s = plant.stream(sol, "effluent", p)
    folder.mkdir(parents=True, exist_ok=True)
    write_trajectory(
        folder / "data.csv",
        t,
        {"effluent.SNH": s.C_named("SNH"), "effluent.SS": s.C_named("SS"), "other.X": t},
    )


FIT = {
    "free": ["asm1_local.muH"],
    "observables": [{"stream": "effluent", "channels": ["SNH", "SS"]}],
    "data": "data.csv",
    "t_span": [0, 0.5],
    "transforms": {"asm1_local.muH": "positive_log"},
    "sigma": [0.05, 0.5],  # one per channel (SNH, SS)
    "max_iter": 40,
    # ASM1's literature prior on muH (centred at 6.0) would pull the estimate.
    "use_priors": False,
}


def test_a_calibration_study_recovers_a_parameter(project):
    """The fit recovers the value the data were made with; without ``laplace``
    there is no posterior, so no std and no covariance file."""
    folder = project / "studies" / "fit"
    _synthetic_data(project, folder, 5.0)
    (folder / "study.yaml").write_text(
        yaml.safe_dump(
            {"study": {"kind": "calibrate", "parameters": {"asm1_local.muH": 3}, **FIT}}
        )
    )
    seen = []
    assert run_study(folder, progress=seen.append).state == "fresh"
    assert seen[0].stage == "setup" and seen[1].stage == "fit"
    fit = yaml.safe_load((folder / "results" / "fit.yaml").read_text())
    assert fit["fitted"]["asm1_local.muH"] == pytest.approx(5.0, rel=1e-3)
    assert fit["std"] is None
    assert not (folder / "results" / "covariance.csv").exists()
    assert read_named_values(folder / "results" / "parameters.yaml")["asm1_local.muH"] == (
        pytest.approx(5.0, rel=1e-3)
    )
    _, fitted = read_trajectory(folder / "results" / "fitted.csv")
    _, data = read_trajectory(folder / "data.csv")
    np.testing.assert_allclose(fitted["effluent.SNH"], data["effluent.SNH"], rtol=1e-4)
    # The data table is one of the study's inputs.
    (folder / "data.csv").write_text((folder / "data.csv").read_text().replace("0.1,", "0.15,"))
    assert "the study's settings have changed" in check_study(folder).reasons


def test_a_run_replaces_results_only_when_it_succeeds(project):
    from aquakin import Cancelled

    folder = _study(project, "keep", {"kind": "simulate", "streams": ["effluent"]})
    run_study(folder)
    before = (folder / "results" / "streams.yaml").read_text()

    def stop(p):
        raise Cancelled()

    with pytest.raises(Cancelled):
        run_study(folder, replace=True, progress=stop)
    status = check_study(folder)
    assert status.state == "fresh" and (folder / "results" / "streams.yaml").read_text() == before
    assert "the last run" in status.notes[-1] and "was cancelled" in status.notes[-1]
    assert not (folder / "results.partial").exists()

    # A run killed outright leaves its partial results; they block nothing.
    (folder / "results.partial").mkdir()
    (folder / "results.partial" / "junk.yaml").write_text("x: 1\n")
    assert run_study(folder, replace=True).state == "fresh"
    assert not (folder / "results.partial").exists() and not (folder / "last_attempt.yaml").exists()
    assert not any("the last run" in n for n in check_study(folder).notes)

    # A study whose only run failed has no results to replace: it runs again as is.
    bad = _study(project, "bad_run", {"kind": "simulate", "streams": ["nowhere"]})
    (bad / "results").mkdir()  # files no finished run recorded (an older aquakin's leftovers)
    with pytest.raises(UnknownUnitError, match="Unknown unit 'nowhere'"):
        run_study(bad)
    assert check_study(bad).state == "failed" and not (bad / "results").exists()
    (bad / "study.yaml").write_text(
        yaml.safe_dump({"study": {"kind": "simulate", "streams": ["effluent"]}})
    )
    assert run_study(bad).state == "fresh"


def test_a_study_names_the_data_columns_it_cannot_find(project):
    folder = project / "studies" / "nodata"
    _synthetic_data(project, folder, 4.0)
    study = {**FIT, "observables": [{"stream": "effluent", "channels": ["SNO"]}], "sigma": 0.1}
    (folder / "study.yaml").write_text(yaml.safe_dump({"study": {"kind": "calibrate", **study}}))
    with pytest.raises(StudyError, match=r"has no column effluent\.SNO"):
        run_study(folder)
    assert check_study(folder).state == "failed"


def test_a_cancelled_run_is_recorded(project):
    from aquakin import Cancelled

    folder = project / "studies" / "cancel"
    _synthetic_data(project, folder, 5.0)
    (folder / "study.yaml").write_text(
        yaml.safe_dump({"study": {"kind": "calibrate", "parameters": {"asm1_local.muH": 3}, **FIT}})
    )

    def stop(p):
        raise Cancelled()

    with pytest.raises(Cancelled):
        run_study(folder, progress=stop)
    assert check_study(folder).state == "cancelled"
    assert not (folder / "results" / "fit.yaml").exists()


def test_a_profile_study(project):
    from aquakin.study import read_table

    folder = project / "studies" / "profile"
    _synthetic_data(project, folder, 4.0)
    study = {
        "kind": "profile",
        **FIT,
        "free": ["asm1_local.muH", "asm1_local.KS"],
        "transforms": {},
        "profile": "asm1_local.muH",
        "grid": [3.5, 4.0],
        "sigma": 0.1,
        "max_iter": 15,
    }
    (folder / "study.yaml").write_text(yaml.safe_dump({"study": study}))
    seen = []
    run_study(folder, progress=seen.append)
    assert {p.stage for p in seen} >= {"setup", "profile"}
    table = read_table(folder / "results" / "profile.csv")
    assert table["asm1_local.muH"] == [3.5, 4.0]
    assert table["delta_loss"][1] == 0.0 and table["delta_loss"][0] > 0.0
    summary = yaml.safe_load((folder / "results" / "profile.yaml").read_text())
    assert summary["mle"] == 4.0 and summary["parameter"] == "asm1_local.muH"


def test_a_rank_study(project):
    from aquakin.study import read_table

    sensors = {
        "ammonia": [{"stream": "effluent", "channels": ["SNH"]}],
        "flow": [{"stream": "effluent", "channels": ["Q"]}],
    }
    folder = _study(
        project,
        "rank",
        {
            "kind": "rank",
            "free": ["asm1_local.muH", "asm1_local.KS"],
            "t_obs": {"start": 0.1, "stop": 0.3, "num": 3},
            "t_span": [0, 0.3],
            "candidates": [{"name": n, "observables": o, "sigma": 0.1} for n, o in sensors.items()],
            "base": {
                "name": "online",
                "observables": [{"stream": "effluent", "channels": ["SNH", "SS"]}],
                "sigma": [0.5, 0.5],
            },
            "criterion": "A",
        },
    )
    run_study(folder)
    ranking = read_table(folder / "results" / "ranking.csv")
    assert ranking["candidate"] == ["ammonia", "flow"] and ranking["rank"] == [1.0, 2.0]
    assert ranking["improvement"][0] > 0.0 and ranking["improvement"][1] == pytest.approx(0.0, abs=0)


def test_a_steady_state_simulation_saves_measured_channels(project):
    folder = _study(
        project,
        "measured",
        {
            "kind": "simulate",
            "observables": [
                {
                    "stream": "effluent",
                    "channels": ["TN", {"measure": "twice_SNH", "weights": {"SNH": 2}}],
                },
                {"stream": "tank"},
            ],
        },
    )
    run_study(folder)
    measured = read_named_values(folder / "results" / "measured.yaml")
    final = read_named_values(folder / "results" / "final_state.yaml")["tank"]
    assert measured["effluent"]["twice_SNH"] == pytest.approx(2 * final["SNH"])
    assert measured["tank"]["SNH"] == pytest.approx(final["SNH"])  # every species


def test_a_singular_fisher_design_is_reported_without_failing(project):
    """Flow alone carries no information about the kinetics: the criteria and
    standard deviations of a singular information are left empty."""
    folder = _study(
        project,
        "flow_only",
        {
            "kind": "fisher",
            "free": ["asm1_local.muH"],
            "observables": [{"stream": "effluent", "channels": ["Q"]}],
            "t_obs": [0.1, 0.2],
            "t_span": [0, 0.2],
            "sigma": 1.0,
        },
    )
    run_study(folder)
    summary = yaml.safe_load((folder / "results" / "summary.yaml").read_text())
    assert summary["std"] is None and summary["criteria"]["A"] == float("inf")


@pytest.mark.parametrize(
    ("study", "message"),
    [
        ({"kind": "simulate"}, "saves at least one stream or observable"),
        ({"kind": "rank", "free": ["a"]}, "a rank study needs candidates, t_obs"),
        (
            {**FIT, "kind": "calibrate", "observables": [{"stream": "effluent"}]},
            "names the channels of each observable",
        ),
        ({**FIT, "kind": "calibrate", "sigma": 1, "estimate_sigma": True}, "not both"),
        (
            {**{k: v for k, v in FIT.items() if k != "sigma"}, "kind": "calibrate"},
            "give the measurement error",
        ),
        ({**FIT, "kind": "calibrate", "sigma": [1, 2, 3]}, "3 errors for 2 measured channels"),
        ({"kind": "simulate", "streams": ["x"], "sigma": "1"}, "sigma"),
    ],
)
def test_a_study_that_cannot_run_says_why(project, study, message):
    folder = _study(project, "bad", study)
    with pytest.raises(StudyError, match=message):
        check_study(folder)


def test_unreadable_study_files(project):
    folder = project / "studies" / "broken"
    folder.mkdir(parents=True)
    (folder / "study.yaml").write_text("study: [unclosed\n")
    with pytest.raises(StudyError, match="not valid YAML"):
        check_study(folder)
    (folder / "study.yaml").write_text("other: 1\n")
    with pytest.raises(StudyError, match="has no 'study:' section"):
        check_study(folder / "study.yaml")


def test_staleness_names_each_kind_of_change(project):
    folder = _study(project, "stale", DYNAMIC)
    run_study(folder)
    (folder / "results" / "streams.csv").unlink()
    assert check_study(folder).reasons == ["results/streams.csv is missing"]
    run_study(folder, replace=True)

    # A model added to the flowsheet, and one taken away.
    record = yaml.safe_load((folder / "run.yaml").read_text())
    record["spec_inputs"] = {**record["spec_inputs"], "model:gone": "x", "table:feed": "y"}
    del record["spec_inputs"]["model:asm1_local"]
    (folder / "run.yaml").write_text(yaml.safe_dump(record))
    reasons = check_study(folder).reasons
    assert "model 'asm1_local' is new" in reasons
    assert "model 'gone' is no longer used" in reasons
    assert "the influent table of 'feed' is no longer used" in reasons

    (project / "flowsheet.yaml").write_text("format: [\n")
    assert "the flowsheet cannot be read" in check_study(folder).reasons[-1]


def test_missing_files_and_misuse_of_freeze_unfreeze_and_copy(project):
    folder = _study(project, "misuse", {**DYNAMIC, "initial_state": "nowhere.yaml"})
    with pytest.raises(StudyError, match=r"initial_state: .*nowhere\.yaml does not exist"):
        run_study(folder)
    data = _study(project, "nodata", {**FIT, "kind": "calibrate", "data": "missing.csv"})
    with pytest.raises(StudyError, match=r"data: .*missing\.csv does not exist"):
        run_study(data)
    (data / "missing.csv").write_text("x,y\n1,2\n")
    with pytest.raises(StudyError, match="not a trajectory table"):
        run_study(data, replace=True)

    with pytest.raises(StudyError, match="is not frozen"):
        unfreeze_study(folder)
    freeze_study(folder)
    with pytest.raises(StudyError, match="already frozen"):
        freeze_study(folder)
    with pytest.raises(StudyError, match="study name 'a b'"):
        copy_study(folder, "a b")
    with pytest.raises(StudyError, match="already exists"):
        copy_study(folder, "nodata")


def test_a_failed_freeze_leaves_no_snapshot(project):
    folder = _study(project, "freeze_fail", DYNAMIC)
    (folder / "snapshot").mkdir()  # left over from before
    (project / "models" / "asm1_local.yaml").write_text("model: [\n")
    with pytest.raises(yaml.YAMLError, match="expected the node content"):
        freeze_study(folder)
    assert not (folder / "snapshot").exists() and not (folder / "frozen.yaml").exists()


def _serve(*requests):
    import io
    import json

    from aquakin.__main__ import serve

    out = io.StringIO()
    serve(io.StringIO("".join(json.dumps(r) + "\n" for r in requests)), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_the_worker_lists_freezes_copies_and_unfreezes_studies(project):
    folder = _study(project, "worker", DYNAMIC)
    (project / "studies" / "broken").mkdir()
    (project / "studies" / "broken" / "study.yaml").write_text("study: {kind: nope}\n")
    p = str(folder)
    replies = _serve(
        {"id": 1, "command": "list_studies", "project": str(project)},
        {"id": 2, "command": "freeze_study", "path": p},
        {"id": 3, "command": "copy_study", "path": p, "name": "worker_copy"},
        {"id": 4, "command": "unfreeze_study", "path": p, "discard_results": True},
        {"id": 5, "command": "list_studies", "project": str(project / "nowhere")},
        {"id": 6, "command": "read_spec", "path": str(project / "flowsheet.yaml")},
    )
    assert all(r["ok"] for r in replies), replies
    broken, worker = replies[0]["result"]
    assert worker["name"] == "worker" and worker["state"] == "not run"
    assert worker["has_results"] is False and worker["path"] == p
    assert worker["flowsheet"] == str((project / "flowsheet.yaml").resolve())
    assert broken["state"] == "unreadable" and "kind" in broken["reasons"][0]
    assert broken["has_results"] is False and broken["flowsheet"] is None
    assert not broken["reasons"][0].startswith("ValidationError")
    assert replies[1]["result"] is True and replies[2]["result"].endswith("worker_copy")
    assert check_study(folder).frozen is False
    assert check_study(project / "studies" / "worker_copy").frozen is True
    assert replies[4]["result"] == [] and replies[5]["result"]["format"] == 2


def test_a_json_run_streams_events_and_stops_on_cancel(project, monkeypatch):
    """``aquakin study run --json``, run in-process: events go to a private copy
    of standard output, and ``cancel`` on standard input stops the run."""
    import json
    import os

    import aquakin.__main__ as cli

    def run(folder, *extra, cancel=False):
        events_r, events_w = os.pipe()
        input_r, input_w = os.pipe()
        monkeypatch.setattr(cli.os, "dup", lambda fd: events_w)
        monkeypatch.setattr(cli.os, "dup2", lambda a, b: None)
        monkeypatch.setattr(cli.sys, "stdin", os.fdopen(input_r))
        if cancel:
            os.write(input_w, b"cancel\n")
        try:
            code = main(["study", "run", "--json", *extra, str(folder)])
        finally:
            os.close(input_w)  # the front end goes away: the listener ends
        with os.fdopen(events_r) as handle:
            return code, [json.loads(line) for line in handle.read().splitlines()]

    folder = _study(project, "json", DYNAMIC)
    code, events = run(folder)
    assert code == 0 and events[0]["event"] == "progress" and events[0]["stage"] == "setup"
    assert events[-1] == {
        "event": "done",
        "state": "fresh",
        "has_results": True,
        "reasons": [],
        "notes": [],
    }

    code, events = run(folder, "--replace", cancel=True)
    assert code == 1 and events[-1] == {"event": "cancelled"}
    status = check_study(folder)  # the earlier results stand
    assert status.state == "fresh" and "was cancelled" in status.notes[-1]
    missing = project / "studies" / "missing"
    code, events = run(missing)
    assert code == 2 and events[-1] == {
        "event": "error",
        "error": f"{missing / 'study.yaml'} does not exist",
        "type": "StudyError",
    }


def _warm_project(tmp_path):
    """A one-tank project whose influent carries a temperature."""
    folder = tmp_path / "warm"
    m = aquakin.load_model("asm1")
    p = Plant("tank", time_unit="d")
    p.add_unit(CSTRUnit("tank", m, 1000.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(m, Q=2000.0, T=288.15, **FEED), to="tank.in")
    p.register_stream("effluent", "tank.out")
    p.to_spec(folder / "flowsheet.yaml")
    return folder


def test_temperature_is_saved_with_the_streams_and_a_steady_start(tmp_path):
    project = _warm_project(tmp_path)
    steady = _study(project, "steady", {"kind": "simulate", "streams": ["effluent"]})
    run_study(steady)
    assert read_named_values(steady / "results" / "streams.yaml")["effluent"]["T"] == pytest.approx(
        288.15
    )
    dynamic = _study(project, "dynamic", {**DYNAMIC, "initial_state": "steady"})
    run_study(dynamic)
    t, columns = read_trajectory(dynamic / "results" / "streams.csv")
    np.testing.assert_allclose(columns["effluent.T"], 288.15)
    # Started at the steady state, the effluent stays there.
    final = read_named_values(steady / "results" / "final_state.yaml")["tank"]
    np.testing.assert_allclose(columns["effluent.SNH"], final["SNH"], rtol=1e-4)


def test_a_frozen_study_run_on_another_plant_is_stale(project):
    folder = _study(project, "refrozen", DYNAMIC)
    run_study(folder)
    freeze_study(folder)
    record = yaml.safe_load((folder / "frozen.yaml").read_text())
    (folder / "frozen.yaml").write_text(yaml.safe_dump({**record, "spec_fingerprint": "other"}))
    status = check_study(folder)
    assert status.state == "stale"
    assert status.reasons == ["the study was frozen on another plant than its results'"]


def test_a_steady_state_that_does_not_converge_fails_the_run(project, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        Plant, "steady_state", lambda self, *a, **k: SimpleNamespace(converged=False, state=None)
    )
    start = _study(project, "nostart", {**DYNAMIC, "initial_state": "steady"})
    with pytest.raises(StudyError, match="steady state used as the initial state did not converge"):
        run_study(start)
    steady = _study(project, "nosteady", {"kind": "simulate", "streams": ["effluent"]})
    with pytest.raises(StudyError, match="the steady state did not converge"):
        run_study(steady)
    assert check_study(steady).state == "failed"


def test_simulations_that_save_only_streams_or_only_channels(project):
    streams = _study(project, "streams_only", {**DYNAMIC, "observables": []})
    run_study(streams)
    assert (streams / "results" / "streams.csv").exists()
    assert not (streams / "results" / "measured.csv").exists()
    channels = _study(project, "channels_only", {**DYNAMIC, "streams": []})
    run_study(channels)
    assert (channels / "results" / "measured.csv").exists()
    assert not (channels / "results" / "streams.csv").exists()


def test_criteria_that_cannot_be_computed_are_left_empty(project, monkeypatch):
    import aquakin

    def refuse(fim, criterion, *a, **k):
        raise ValueError("cannot")

    monkeypatch.setattr(aquakin, "design_criterion", refuse)
    folder = _study(
        project,
        "nocriteria",
        {
            "kind": "fisher",
            "free": ["asm1_local.muH"],
            "observables": [{"stream": "effluent", "channels": ["SS"]}],
            "t_obs": [0.1, 0.2],
            "t_span": [0, 0.2],
            "sigma": 1.0,
        },
    )
    run_study(folder)
    summary = yaml.safe_load((folder / "results" / "summary.yaml").read_text())
    assert summary["criteria"] == {"D": None, "A": None, "E": None, "modified_E": None}
    assert summary["std"]["asm1_local.muH"] > 0.0


def test_a_calibration_with_laplace_writes_its_posterior(project, monkeypatch):
    """With the fit itself stubbed (the plant's Laplace posterior has its own
    tests): the study asks for the posterior, and writes its std and covariance
    by parameter name. The data table is made by hand, since the stub never
    solves."""
    from types import SimpleNamespace

    import aquakin.plant as plant_module
    from aquakin.study import write_trajectory

    folder = project / "studies" / "laplace"
    folder.mkdir(parents=True)
    t = np.linspace(0.1, 0.5, 5)
    write_trajectory(folder / "data.csv", t, {"effluent.SNH": 1.0 + t, "effluent.SS": 2.0 + t})
    (folder / "study.yaml").write_text(
        yaml.safe_dump({"study": {"kind": "calibrate", "laplace": True, **FIT}})
    )
    asked = {}

    def fit(plant, observations, t, free, **kw):
        asked["laplace"] = kw["laplace"]
        return SimpleNamespace(
            params=kw["params"],
            params_named={n: float(kw["params"][plant.parameter_index(n)]) for n in free},
            params_named_std={"asm1_local.muH": 0.25},
            loss=0.0,
            converged=True,
            message="stub",
            n_iter=0,
            at_bound={},
            sigma_estimate=None,
            posterior_cov=np.array([[0.0625]]),
            parameter_names=list(free),
        )

    monkeypatch.setattr(plant_module, "calibrate_plant", fit)
    run_study(folder)
    assert asked["laplace"] is True
    fit_yaml = yaml.safe_load((folder / "results" / "fit.yaml").read_text())
    assert fit_yaml["std"] == {"asm1_local.muH": 0.25}
    cov, rows, columns = read_matrix(folder / "results" / "covariance.csv")
    assert rows == columns == ["asm1_local.muH"]
    np.testing.assert_array_equal(cov, [[0.0625]])


def test_the_fit_is_given_one_error_per_channel_or_estimates_it():
    from aquakin.study.study import _fit_sigma, _Study

    base = {"kind": "calibrate", **FIT}
    assert _fit_sigma(_Study.model_validate({**base, "sigma": None, "estimate_sigma": True})) == (
        "estimate"
    )
    per_channel = _fit_sigma(_Study.model_validate(base))
    assert isinstance(per_channel, np.ndarray) and per_channel.tolist() == [0.05, 0.5]
    assert _fit_sigma(_Study.model_validate({**base, "sigma": 0.2})) == 0.2


def test_a_study_measures_a_channel_of_its_own(project):
    """A measure expression in a study file is a results column of its own name,
    weighing the stream as it says."""
    folder = _study(
        project,
        "custom",
        {
            "kind": "simulate",
            "observables": [
                {
                    "stream": "effluent",
                    "channels": [
                        "SNH",
                        "BOD5 = 0.65*[SS] + 0.65*[XS]",
                        "sTN = TN - [XS]",
                        # The structured form keeps working, now with keywords.
                        {"measure": "half_TN", "keywords": {"TN": 0.5}},
                    ],
                }
            ],
        },
    )
    run_study(folder)
    measured = read_named_values(folder / "results" / "measured.yaml")["effluent"]
    state = read_named_values(folder / "results" / "final_state.yaml")["tank"]
    # Reported under the name, not the text that defines it.
    assert set(measured) == {"SNH", "BOD5", "sTN", "half_TN"}
    assert measured["BOD5"] == pytest.approx(0.65 * (state["SS"] + state["XS"]))
    assert measured["half_TN"] == pytest.approx(0.5 * (measured["sTN"] + state["XS"]))


def test_a_dynamic_study_names_its_custom_columns(project):
    folder = _study(
        project,
        "custom_dynamic",
        {
            **DYNAMIC,
            "observables": [{"stream": "effluent", "channels": ["BOD5 = 0.65*[SS] + 0.65*[XS]"]}],
        },
    )
    run_study(folder)
    _, measured = read_trajectory(folder / "results" / "measured.csv")
    assert list(measured) == ["effluent.BOD5"]


@pytest.mark.parametrize(
    ("channel", "message"),
    [
        # Read when the study is written: it needs no flowsheet.
        ("BOD5 = 0.65*[SS] *", "expected a number"),
        ("x = [SS]*[XS]", "cannot multiply two measured quantities"),
        ("x = 2", "no constant term"),
    ],
)
def test_an_unreadable_measure_is_refused_as_the_study_is_written(project, channel, message):
    folder = _study(
        project,
        "bad",
        {"kind": "simulate", "observables": [{"stream": "effluent", "channels": [channel]}]},
    )
    with pytest.raises(StudyError, match=message):
        run_study(folder)


def test_a_measure_naming_an_unknown_species_says_so_at_the_run(project):
    """The species are resolved against the model of the stream, so they can only
    be checked once the flowsheet is built -- reported as a ``KeyError`` naming
    the stream, exactly as an unknown plain channel already is."""
    folder = _study(
        project,
        "unknown_species",
        {"kind": "simulate", "observables": [{"stream": "effluent", "channels": ["x = [NOPE]"]}]},
    )
    with pytest.raises(KeyError, match="not a species of model"):
        run_study(folder)
    with pytest.raises(KeyError, match="stream 'effluent'"):
        run_study(folder)
