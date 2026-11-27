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
    read_table,
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


def test_a_calibration_study_recovers_a_parameter_and_is_sampled(project):
    """The fit recovers the value the data were made with; without ``laplace``
    there is no posterior, so no std and no covariance file. A posterior study
    samples it all the same (re-entering the fit computes the Laplace posterior
    the sampler starts from), briefly here: this is the plant path, whose fit
    has no cheaper analogue than this one-tank plant."""
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
    post = _study(project, "post", {**POSTERIOR, "n_draws": 4, "n_warmup": 0})
    assert run_study(post).state == "fresh"
    summary = yaml.safe_load((post / "results" / "posterior.yaml").read_text())
    # Re-entered at the saved optimum: the optimizer starts there and stays.
    assert summary["parameters"]["asm1_local.muH"]["optimum"] == pytest.approx(
        fit["fitted"]["asm1_local.muH"], rel=1e-6
    )
    assert summary["n_draws"] == 4 and len(summary["acceptance_rate"]) == 2
    # The data table is one of the study's inputs, and so of its posterior's.
    (folder / "data.csv").write_text((folder / "data.csv").read_text().replace("0.1,", "0.15,"))
    assert "the study's settings have changed" in check_study(folder).reasons
    assert check_study(post).reasons == ["the setup of the calibrate study it samples has changed"]


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
        ({"kind": "design", "free": ["a"]}, "a design study needs reactor"),
        (
            {"kind": "fisher", "base_experiments": [{"initial": {}}]},
            "experiments and observed are the settings of a reactor study",
        ),
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


# ---------------------------------------------------------------------------
# Reactor studies: a batch reactor in place of the flowsheet
# ---------------------------------------------------------------------------

#: A -> B -> C (first-order steps k and k2), a model file extending the
#: first-order decay A -> B, so freezing has an ``extends`` chain to flatten.
CHAIN = """\
model:
  name: decay_chain
  extends: decay.yaml
species:
  - {name: C, description: "Final product", units: mol/L, default_concentration: 0.0}
reactions:
  - name: B_to_C
    description: "First-order decay of B."
    rate: "k2 * [B]"
    parameters:
      k2: {value: 0.05, units: "s-1", bounds: [1.0e-3, 10.0]}
    stoichiometry: {B: -1, C: +1}
"""
TRUE = {"A_to_B.k": 0.1, "B_to_C.k2": 0.05}
T_DATA = np.linspace(2.0, 30.0, 8)
NOISE = 0.01


def _chain(t, a0, k, k2):
    """The closed form of A -> B -> C from A = a0, B = C = 0."""
    a = a0 * np.exp(-k * t)
    b = a0 * k / (k2 - k) * (np.exp(-k * t) - np.exp(-k2 * t))
    return {"A": a, "B": b}


REACTOR = {"model": "../../models/chain.yaml"}
EXPERIMENTS = [
    {"data": "run1.csv", "initial": {"A": 1.0}},
    {"data": "run2.csv", "initial": {"A": 2.0}},
]
REACTOR_FIT = {
    "kind": "calibrate",
    "reactor": REACTOR,
    "experiments": EXPERIMENTS,
    "observed": ["A", "B"],
    "free": ["A_to_B.k", "B_to_C.k2"],
    "parameters": {"A_to_B.k": 0.08, "B_to_C.k2": 0.06},
    "sigma": NOISE,
    "laplace": True,
}


def _reactor_project(root):
    """A project with the chain model and, for any study, noisy data of two
    batches (A0 = 1 and 2) made with the closed form at TRUE."""
    from tests.conftest import FIXTURES

    (root / "models").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "simple_model.yaml", root / "models" / "decay.yaml")
    (root / "models" / "chain.yaml").write_text(CHAIN)
    return root


def _reactor_study(project, name, study):
    from aquakin.study import write_trajectory

    folder = _study(project, name, study)
    rng = np.random.default_rng(7)
    for run, a0 in (("run1", 1.0), ("run2", 2.0)):
        exact = _chain(T_DATA, a0, TRUE["A_to_B.k"], TRUE["B_to_C.k2"])
        noisy = {s: v + NOISE * rng.standard_normal(v.shape) for s, v in exact.items()}
        write_trajectory(folder / f"{run}.csv", T_DATA, noisy)
    return folder


@pytest.fixture(scope="module")
def fitted_reactor(tmp_path_factory):
    """A reactor project whose calibrate study ``fit`` has run once; a test that
    changes it works on a copy (:func:`_copy_project`)."""
    project = _reactor_project(tmp_path_factory.mktemp("reactor") / "project")
    folder = _reactor_study(project, "fit", REACTOR_FIT)
    seen = []
    run_study(folder, progress=seen.append)
    return project, [p.stage for p in seen]


def _copy_project(fitted_reactor, tmp_path):
    project = tmp_path / "project"
    shutil.copytree(fitted_reactor[0], project)
    return project, project / "studies" / "fit"


def test_a_reactor_calibration_study_recovers_its_parameters(fitted_reactor):
    """Two batches fitted jointly recover the rates their data were made with,
    within three standard deviations of the fit; the saved fit reads back with
    ``load_calibration``, and the fitted trajectories are the closed form at
    the fitted rates."""
    from aquakin.study import study_results

    project, stages = fitted_reactor
    folder = project / "studies" / "fit"
    assert stages == ["setup", "fit", "fit"]
    assert check_study(folder).state == "fresh"
    fit = yaml.safe_load((folder / "results" / "fit.yaml").read_text())
    for name, value in TRUE.items():
        assert fit["std"][name] > 0
        assert abs(fit["fitted"][name] - value) < 3 * fit["std"][name], name
    loaded = aquakin.load_calibration(folder / "results")
    assert loaded.params_named == fit["fitted"] and loaded.params_named_std == fit["std"]
    params = read_named_values(folder / "results" / "parameters.yaml")
    assert params == fit["fitted"]
    assert dict(zip(["A_to_B.k", "B_to_C.k2"], map(float, loaded.params))) == params
    covariance, rows, _ = read_matrix(folder / "results" / "covariance.csv")
    assert rows == ["A_to_B.k", "B_to_C.k2"]
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), list(fit["std"].values()), rtol=1e-12)

    t, fitted = read_trajectory(folder / "results" / "fitted.csv")
    np.testing.assert_array_equal(t, T_DATA)
    assert list(fitted) == ["run1.A", "run1.B", "run2.A", "run2.B"]
    for run, a0 in (("run1", 1.0), ("run2", 2.0)):
        exact = _chain(T_DATA, a0, params["A_to_B.k"], params["B_to_C.k2"])
        for species in ("A", "B"):
            # An adaptive solve at rtol 1e-6, checked two decades above it.
            np.testing.assert_allclose(fitted[f"{run}.{species}"], exact[species], rtol=1e-4)

    shown = study_results(folder)
    assert shown["summary"]["fitted"] == fit["fitted"]
    assert shown["fitted"]["values"]["run2"]["B"] == list(fitted["run2.B"])
    _, data = read_trajectory(folder / "run2.csv")
    assert shown["data"]["values"]["run2"]["B"] == list(data["B"])

    with pytest.raises(StudyError, match="already has results"):
        run_study(folder)


def test_a_reactor_study_goes_stale_naming_what_changed(fitted_reactor, tmp_path):
    project, folder = _copy_project(fitted_reactor, tmp_path)
    assert check_study(folder).state == "fresh"
    model = project / "models" / "chain.yaml"
    model.write_text(CHAIN.replace("value: 0.05", "value: 0.06"))
    assert check_study(folder).reasons == ["model '../../models/chain.yaml' has changed"]
    model.write_text(CHAIN)
    # A change to the model it extends reaches it too.
    base = project / "models" / "decay.yaml"
    original = base.read_text()
    base.write_text(original.replace("value: 0.1", "value: 0.2"))
    assert check_study(folder).reasons == ["model '../../models/chain.yaml' has changed"]
    base.write_text(original)
    assert check_study(folder).state == "fresh"

    data = folder / "run2.csv"
    original = data.read_text()
    data.write_text(original.replace("2.0,", "2.5,", 1))
    assert check_study(folder).reasons == ["data table 'run2.csv' has changed"]
    data.write_text(original)
    assert check_study(folder).state == "fresh"

    study = yaml.safe_load((folder / "study.yaml").read_text())
    study["study"]["reactor"]["conditions"] = {"T": 300.0}
    (folder / "study.yaml").write_text(yaml.safe_dump(study))
    assert check_study(folder).reasons == ["the study's settings have changed"]

    data.unlink()
    status = check_study(folder)
    assert status.state == "stale"
    assert status.reasons[-1].startswith("the reactor's model or data cannot be read")
    assert "run2.csv does not exist" in status.reasons[-1]
    # The results still show, beside the data that is left.
    from aquakin.study import study_results

    shown = study_results(folder)
    assert set(shown["data"]["values"]) == {"run1"} and set(shown["fitted"]["values"]) == {
        "run1",
        "run2",
    }


def test_a_frozen_reactor_study_runs_from_its_snapshot(fitted_reactor, tmp_path):
    """The snapshot holds the model with its ``extends`` chain merged in and the
    data; the frozen study stays fresh (the same model and data), reruns to the
    same numbers from the snapshot, and an edit to the live model or data no
    longer reaches it."""
    project, folder = _copy_project(fitted_reactor, tmp_path)
    before = (folder / "results" / "fitted.csv").read_text()
    record = freeze_study(folder)
    assert record["from"] == "../../models/chain.yaml"
    flat = yaml.safe_load((folder / "snapshot" / "models" / "chain.yaml").read_text())
    assert "extends" not in flat["model"] and flat["model"]["name"] == "decay_chain"
    assert [r["name"] for r in flat["reactions"]] == ["A_to_B", "B_to_C"]
    for run in ("run1.csv", "run2.csv"):
        assert (folder / "snapshot" / "data" / run).read_bytes() == (folder / run).read_bytes()
    status = check_study(folder)
    assert status.state == "fresh" and status.frozen

    (project / "models" / "chain.yaml").write_text(CHAIN.replace("value: 0.05", "value: 0.5"))
    (project / "models" / "decay.yaml").unlink()
    (folder / "run1.csv").write_text("t,A,B\n1.0,0.0,0.0\n")
    assert check_study(folder).state == "fresh"
    assert run_study(folder, replace=True).state == "fresh"
    assert (folder / "results" / "fitted.csv").read_text() == before


def test_a_shipped_reactor_model_is_frozen_as_a_copy(tmp_path):
    """A shipped model is fingerprinted by its shipped content, and the snapshot
    holds it flattened: the copy builds the same model."""
    project = tmp_path / "project"
    study = {
        "kind": "fisher",
        "reactor": {"model": "ozone_bromate", "conditions": {"pH": 8.0}},
        "experiments": [{"initial": {"O3": 1.0e-4}}],
        "observed": ["O3"],
        "free": ["O3_decay_OH.k_dec"],
        "t_obs": [60.0],
        "sigma": 1.0e-6,
    }
    folder = _study(project, "shipped", study)
    model = aquakin.load_model("ozone_bromate")
    freeze_study(folder)
    copy = aquakin.load_model_from_file(folder / "snapshot" / "models" / "ozone_bromate.yaml")
    assert copy.species == model.species and copy.parameters == model.parameters
    np.testing.assert_array_equal(
        np.asarray(copy.default_parameters()), np.asarray(model.default_parameters())
    )
    record = yaml.safe_load((folder / "frozen.yaml").read_text())
    from aquakin.plant.spec import _shipped_fingerprint

    assert record["spec_inputs"] == {"model:ozone_bromate": _shipped_fingerprint("ozone_bromate")}


def test_a_reactor_profile_study_is_the_profile_likelihood(tmp_path):
    from aquakin import OptimizerConfig, profile_likelihood
    from aquakin.study import read_table

    project = _reactor_project(tmp_path / "project")
    grid = [0.095, 0.1, 0.105]
    study = {
        **REACTOR_FIT,
        "kind": "profile",
        "experiments": EXPERIMENTS[1:],
        "profile": "A_to_B.k",
        "grid": grid,
    }
    del study["laplace"]
    folder = _reactor_study(project, "profile", study)
    seen = []
    run_study(folder, progress=seen.append)
    assert [p.stage for p in seen] == ["setup", "profile"]

    model = aquakin.load_model_from_file(project / "models" / "chain.yaml")
    _, data = read_trajectory(folder / "run2.csv")
    direct = profile_likelihood(
        aquakin.BatchReactor(model, model.default_conditions()),
        model.concentrations({"A": 2.0}),
        np.stack([data["A"], data["B"]], axis=1),
        T_DATA,
        ["A_to_B.k", "B_to_C.k2"],
        grid=grid,
        profile_param="A_to_B.k",
        initial_params=model.parameter_values(REACTOR_FIT["parameters"]),
        observed_species=["A", "B"],
        loss="nll",
        sigma=NOISE,
        optimizer=OptimizerConfig(n_starts=1, max_iter=500),
    )
    table = read_table(folder / "results" / "profile.csv")
    assert table["A_to_B.k"] == grid
    np.testing.assert_allclose(table["delta_loss"], direct.delta_loss, rtol=1e-10, atol=1e-10)
    summary = yaml.safe_load((folder / "results" / "profile.yaml").read_text())
    np.testing.assert_allclose(summary["ci"], direct.ci, rtol=1e-10)
    assert summary["mle"] == direct.mle and summary["parameter"] == "A_to_B.k"
    lo, hi = summary["ci"]
    assert lo < TRUE["A_to_B.k"] < hi


def test_a_reactor_fisher_study_is_the_fisher_information(tmp_path):
    """Two batches without data: the matrix is ``fisher_information`` of the same
    design, and its k entry is the closed form sum of (t a0 exp(-k t) / sigma)^2
    over the A samples plus the B samples' share."""
    from aquakin import Experiment, fisher_information

    project = _reactor_project(tmp_path / "project")
    t_obs = [5.0, 10.0, 20.0]
    study = {
        "kind": "fisher",
        "reactor": {**REACTOR, "integrator": {"rtol": 1.0e-8}},
        "experiments": [{"initial": {"A": 1.0}}, {"initial": {"A": 2.0}}],
        "observed": ["A", "B"],
        "free": ["A_to_B.k", "B_to_C.k2"],
        "t_obs": t_obs,
        "sigma": [0.01, 0.02],
        "time_unit": "min",
    }
    folder = _study(project, "fisher", study)
    run_study(folder)
    matrix, rows, cols = read_matrix(folder / "results" / "fisher.csv")
    assert rows == cols == ["A_to_B.k", "B_to_C.k2"]

    model = aquakin.load_model_from_file(project / "models" / "chain.yaml")
    reactor = aquakin.BatchReactor(model, model.default_conditions(), settings=aquakin.SolverSettings(
        rtol=1.0e-8,
    ))
    design = [
        Experiment(
            model.concentrations({"A": a0}),
            t_obs=60.0 * np.asarray(t_obs),
            sigma=np.asarray([0.01, 0.02]),
            observed_species=["A", "B"],
        )
        for a0 in (1.0, 2.0)
    ]
    direct = fisher_information(reactor, design, rows)
    np.testing.assert_allclose(matrix, direct.matrix, rtol=1e-10)
    # The A samples alone give the closed form d A / d k = -t a0 exp(-k t); the
    # B samples add a positive share, so the entry is at least that.
    t_s = 60.0 * np.asarray(t_obs)
    from_a = sum(np.sum((t_s * a0 * np.exp(-0.1 * t_s) / 0.01) ** 2) for a0 in (1.0, 2.0))
    assert matrix[0, 0] > from_a
    summary = yaml.safe_load((folder / "results" / "summary.yaml").read_text())
    assert summary["values"] == {"A_to_B.k": 0.1, "B_to_C.k2": 0.05}
    params = read_named_values(folder / "results" / "parameters.yaml")
    assert params == {"A_to_B.k": 0.1, "B_to_C.k2": 0.05}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"flowsheet": "../../flowsheet.yaml"}, "give a flowsheet or a reactor to study, not both"),
        ({"reactor": None}, "experiments and observed are the settings of a reactor study"),
        ({"kind": "simulate"}, "the simulate kind does not take a reactor target yet"),
        ({"kind": "rank"}, "the rank kind does not take a reactor target yet"),
        ({"observed": []}, "a calibrate study needs observed"),
        ({"streams": ["effluent"], "t_span": [0, 1]}, "streams, t_span are settings of a flowsheet"),
        ({"kind": "fisher", "t_obs": [1.0]}, "a fisher study reads no data"),
        ({"experiments": [{"initial": {"A": 1.0}}]}, "experiments 0 give no data table to fit"),
        (
            {"kind": "profile", "profile": "A_to_B.k", "grid": [0.1]},
            "a profile study of a reactor takes one experiment, not 2",
        ),
        (
            {"experiments": [{"data": "run1.csv"}, {"data": "other/run1.csv"}]},
            "run1 names more than one",
        ),
        ({"sigma": [0.1, 0.2, 0.3]}, "3 errors for 2 measured channels"),
        (
            {"experiments": [{"data": "run1.csv", "t_obs": [1.0]}]},
            "experiments 0 give t_obs, but a calibrate study takes its times from each "
            "experiment's data",
        ),
        ({"base_experiments": [{"initial": {}}]}, "base_experiments are a design study's setting"),
        (
            {"kind": "fisher", "experiments": [{"initial": {}, "t_obs": [1.0]}, {"initial": {}}]},
            "give the sampling times, t_obs, for the study or for each experiment",
        ),
        (
            {
                "kind": "design",
                "experiments": [{"initial": {}}],
                "base_experiments": [{"data": "run1.csv"}],
                "t_obs": [1.0],
            },
            "a design study reads no data; its base_experiments give their initial",
        ),
        (
            {
                "kind": "design",
                "experiments": [{"initial": {}}],
                "t_obs": [1.0],
                "optimize_times": False,
            },
            r"optimizes the sampling times \(optimize_times\) or the initial concentrations "
            r"\(initial_bounds\); it has neither",
        ),
        (
            {
                "kind": "fisher",
                "experiments": [{"initial": {}}],
                "t_obs": [1.0],
                "sigma": [0.1, 0.2, 0.3],
            },
            "3 errors for 2 measured channels",
        ),
    ],
)
def test_a_reactor_study_that_cannot_run_says_why(tmp_path, change, message):
    study = {**REACTOR_FIT, **change}
    study = {k: v for k, v in study.items() if v is not None}
    folder = _study(_reactor_project(tmp_path / "project"), "bad", study)
    with pytest.raises(StudyError, match=message):
        check_study(folder)


#: A reactor fit turned into a design of the same reactor, for the refusals at run time.
DESIGN_CHANGE = {
    "kind": "design",
    "experiments": [{"initial": {"A": 1.0}}],
    "t_obs": [5.0],
    "laplace": False,
}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"observed": ["A", "Z"]}, r"observed: the model has no species Z \(its species: A, B, C\)"),
        ({"reactor": {"model": "no_such_model"}}, "'no_such_model' is neither a model file"),
        ({"reactor": {"model": "../../models/gone.yaml"}}, r"reactor.model: .*gone\.yaml does not exist"),
        ({"reactor": {**REACTOR, "conditions": {"pH": 7.0}}}, "reactor.conditions: .*pH"),
        ({"parameters": {"A_to_B.kk": 1.0}}, "parameters: .*A_to_B.kk"),
        ({"experiments": [{"data": "run1.csv", "initial": {"Q": 1.0}}]}, r"experiments\[0\].initial"),
        ({"time_unit": "fortnight"}, "time_unit: Unknown time_unit 'fortnight'"),
        (
            {"experiments": [{"data": "run1.csv", "initial": {}}], "observed": ["A", "C"]},
            r"experiments\[0\].data: run1.csv has no column C \(its columns: A, B\)",
        ),
        ({"experiments": [{"data": "nowhere.csv"}]}, r"experiments\[0\].data: .*nowhere\.csv does not exist"),
        (
            {**DESIGN_CHANGE, "base_experiments": [{"initial": {"Q": 1.0}}]},
            r"base_experiments\[0\].initial",
        ),
        ({**DESIGN_CHANGE, "initial_bounds": {"Q": [0.0, 1.0]}}, "design: Unknown species 'Q' in initial_bounds"),
        ({**DESIGN_CHANGE, "subset": ["A_to_B.kk"]}, "design: Unknown subset name 'A_to_B.kk'"),
    ],
)  # fmt: skip
def test_a_reactor_study_names_what_it_cannot_find(tmp_path, change, message):
    folder = _reactor_study(_reactor_project(tmp_path / "project"), "bad", {**REACTOR_FIT, **change})
    with pytest.raises(StudyError, match=message):
        run_study(folder)


def test_a_reactor_study_whose_data_is_gone_cannot_be_frozen(tmp_path):
    folder = _reactor_study(_reactor_project(tmp_path / "project"), "gone", REACTOR_FIT)
    (folder / "run2.csv").unlink()
    with pytest.raises(StudyError, match=r"experiments\[1\].data: .*run2\.csv does not exist"):
        freeze_study(folder)
    assert not (folder / "snapshot").exists() and not (folder / "frozen.yaml").exists()


def test_the_worker_lists_a_reactor_study_by_its_model(tmp_path):
    project = _reactor_project(tmp_path / "project")
    _reactor_study(project, "fit", REACTOR_FIT)
    (reply,) = _serve({"id": 1, "command": "list_studies", "project": str(project)})
    (listed,) = reply["result"]
    assert listed["flowsheet"] is None and listed["reactor"] == "../../models/chain.yaml"
    assert listed["state"] == "not run" and listed["kind"] == "calibrate"


def test_experiments_sampled_at_other_times_share_one_table():
    """A reactor study's fitted and observed series are written on every time any
    experiment has, empty where an experiment has no sample."""
    from aquakin.study.reactor import _on_one_grid

    t, columns = _on_one_grid(
        [
            ("run1", np.array([1.0, 3.0]), {"A": [10.0, 30.0]}),
            ("run2", np.array([2.0, 3.0]), {"A": [20.0, 31.0], "B": [2.0, 3.1]}),
        ]
    )
    np.testing.assert_array_equal(t, [1.0, 2.0, 3.0])
    assert list(columns) == ["run1.A", "run2.A", "run2.B"]
    np.testing.assert_array_equal(columns["run1.A"], [10.0, np.nan, 30.0])
    np.testing.assert_array_equal(columns["run2.B"], [np.nan, 2.0, 3.1])


# ---------------------------------------------------------------------------
# design: optimizing a reactor's experiments
# ---------------------------------------------------------------------------

#: One sample of A in first-order decay (k = 0.1 per second) carries the
#: information (t A0 exp(-k t) / sigma)^2 about k, largest at t = 1/k = 10 s,
#: with A0 as large as it may be. Times in minutes, to check the units.
DECAY_K, DECAY_SIGMA = 0.1, 0.01
BASE_T, BASE_A0 = 5.0, 1.0  # seconds; a batch already run, sampled once
DESIGN = {
    "kind": "design",
    "reactor": {"model": "../../models/decay.yaml", "integrator": {"rtol": 1.0e-10}},
    "experiments": [{"initial": {"A": 1.0}, "t_obs": [0.05]}],
    "base_experiments": [{"initial": {"A": BASE_A0}, "t_obs": [BASE_T / 60.0]}],
    "observed": ["A"],
    "free": ["A_to_B.k"],
    "sigma": DECAY_SIGMA,
    "time_unit": "min",
    "time_bounds": [0.0, 1.0],
    "initial_bounds": {"A": [0.5, 2.0]},
}


def _decay_information(t, a0):
    """The information about k of one sample of A at t seconds, from A0 = a0."""
    return (t * a0 * np.exp(-DECAY_K * t) / DECAY_SIGMA) ** 2


@pytest.fixture(scope="module")
def designed_decay(tmp_path_factory):
    project = _reactor_project(tmp_path_factory.mktemp("design") / "project")
    folder = _study(project, "design", DESIGN)
    seen = []
    run_study(folder, progress=seen.append)
    return project, folder, seen


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"initial_bounds": ["A"]}, "initial_bounds: Input should be a valid dictionary"),
        ({"initial_bounds": {"A": 0.5}}, r"initial_bounds\.A: Input should be a valid tuple"),
        ({"time_bounds": 1.0}, "time_bounds: Input should be a valid tuple"),
    ],
)
def test_a_design_s_bounds_are_checked_as_it_is_read(tmp_path, change, message):
    folder = _study(_reactor_project(tmp_path / "project"), "bad", {**DESIGN, **change})
    with pytest.raises(StudyError, match=message):
        check_study(folder)


def test_a_design_study_finds_the_closed_form_optimum(designed_decay):
    """The sample moves to t = 1/k (1/6 min) and A0 to its upper bound; the
    standard deviations at the start and at the optimum are the closed form
    1 / sqrt(information), the batch already run included; the criterion is
    the D criterion of those designs."""
    from aquakin import design_criterion, fisher_information

    _, folder, seen = designed_decay
    assert [p.stage for p in seen] == ["setup", "design", "design"]
    design = yaml.safe_load((folder / "results" / "design.yaml").read_text())
    (designed,) = design["experiments"]
    assert designed["t_obs"] == [pytest.approx(1 / (60 * DECAY_K), rel=1e-4)]
    assert designed["initial"] == {"A": pytest.approx(2.0, rel=1e-12)}
    assert design["converged"] and design["criterion"] == "D" and design["subset"] == []
    base = _decay_information(BASE_T, BASE_A0)
    start = base + _decay_information(0.05 * 60, 1.0)
    optimum = base + _decay_information(1 / DECAY_K, 2.0)
    assert design["std"]["A_to_B.k"]["start"] == pytest.approx(start**-0.5, rel=1e-6)
    assert design["std"]["A_to_B.k"]["optimum"] == pytest.approx(optimum**-0.5, rel=1e-6)
    assert design["optimum"] > design["start"]
    model = aquakin.load_model_from_file(folder.parents[1] / "models" / "decay.yaml")
    reactor = aquakin.BatchReactor(model, model.default_conditions(), rtol=1.0e-10)
    experiments = [
        aquakin.Experiment(model.concentrations({"A": a0}), t_obs=np.asarray([t]),
                           sigma=DECAY_SIGMA, observed_species=["A"])
        for t, a0 in ((BASE_T, BASE_A0), (3.0, 1.0))
    ]  # fmt: skip
    fim = fisher_information(reactor, experiments, ["A_to_B.k"])
    assert design["start"] == pytest.approx(float(design_criterion(fim, "D")), rel=1e-8)


def test_a_designed_experiment_pastes_into_a_fisher_study(designed_decay, tmp_path):
    """The written experiments: list, with the batch already run beside it, is a
    fisher study's design (each experiment with its own t_obs), and gives the
    standard deviation the design predicted."""
    from aquakin.study import study_results

    project, folder, _ = designed_decay
    design = yaml.safe_load((folder / "results" / "design.yaml").read_text())
    fisher = {
        key: DESIGN[key] for key in ("reactor", "observed", "free", "sigma", "time_unit")
    }
    fisher.update(
        kind="fisher", experiments=DESIGN["base_experiments"] + design["experiments"]
    )
    pasted = _study(_copy_project((project, None), tmp_path)[0], "pasted", fisher)
    run_study(pasted)
    summary = yaml.safe_load((pasted / "results" / "summary.yaml").read_text())
    assert summary["std"]["A_to_B.k"] == pytest.approx(
        design["std"]["A_to_B.k"]["optimum"], rel=1e-8
    )
    results = study_results(folder)
    assert results["kind"] == "design" and results["files"] == ["design.yaml"]
    assert results["summary"]["experiments"] == design["experiments"]


# ---------------------------------------------------------------------------
# posterior: sampling a calibrate study's fit
# ---------------------------------------------------------------------------

POSTERIOR = {"kind": "posterior", "fit": "fit", "n_chains": 2, "n_draws": 400, "n_warmup": 200}
SHORT_POSTERIOR = {**POSTERIOR, "n_draws": 20, "n_warmup": 20}


@pytest.fixture(scope="module")
def sampled_reactor(fitted_reactor, tmp_path_factory):
    """A copy of the fitted reactor project with a posterior study of its fit,
    run once; a test that changes it works on a copy (:func:`_copy_project`)."""
    project, _ = _copy_project(fitted_reactor, tmp_path_factory.mktemp("sampled"))
    folder = _study(project, "post", POSTERIOR)
    seen = []
    status = run_study(folder, progress=seen.append)
    return project, status, seen


def test_a_posterior_study_samples_its_calibration(sampled_reactor):
    """The draws of the reactor fit's two rates centre on the fit's optimum, with
    the spread of its Laplace posterior: for a well-determined fit both agree,
    within the Monte Carlo error of the draws (the error of a mean is about
    std / sqrt(ESS), of a standard deviation about 1 / sqrt(2 ESS) of it)."""
    project, status, seen = sampled_reactor
    assert status.state == "fresh" and status.running is None
    stages = [p.stage for p in seen]
    assert stages[:2] == ["setup", "fit"] and set(stages[2:]) == {"sample"}
    assert seen[-1].done == seen[-1].total == 600  # warm-up and draws
    fit = yaml.safe_load((project / "studies" / "fit" / "results" / "fit.yaml").read_text())
    summary = yaml.safe_load((project / "studies" / "post" / "results" / "posterior.yaml").read_text())
    assert summary["fit"] == "fit" and summary["prior_space"] == "transformed"
    assert (summary["n_chains"], summary["n_draws"], summary["n_warmup"]) == (2, 400, 200)
    for name in TRUE:
        p = summary["parameters"][name]
        # Re-entered at the saved optimum: the optimizer starts there and stays.
        assert p["optimum"] == pytest.approx(fit["fitted"][name], rel=1e-6)
        assert abs(p["mean"] - p["optimum"]) < 4 * p["std"] / np.sqrt(p["ess_bulk"]), name
        assert p["std"] / fit["std"][name] == pytest.approx(
            1.0, abs=4 / np.sqrt(2 * p["ess_bulk"])
        ), name
        assert p["q05"] < p["q50"] < p["q95"]
    draws = read_table(project / "studies" / "post" / "results" / "draws.csv")
    assert list(draws) == ["chain", "draw", *TRUE]
    assert len(draws["chain"]) == 800 and sorted(set(draws["chain"])) == [0, 1]


def test_posterior_results_read_back_for_the_editor(sampled_reactor):
    from aquakin.study import study_results

    project, _, _ = sampled_reactor
    results = study_results(project / "studies" / "post")
    assert results["kind"] == "posterior" and results["has_results"]
    from aquakin.study import list_studies

    listed = {s["name"]: s for s in list_studies(project)}
    assert listed["post"]["flowsheet"] is None and listed["post"]["state"] == "fresh"
    assert results["files"] == ["draws.csv", "posterior.yaml"]
    for name in TRUE:
        histogram = results["histograms"][name]
        assert sum(histogram["counts"]) == 800 and len(histogram["edges"]) == 31
    assert set(results["summary"]["parameters"]) == set(TRUE)


def test_a_posterior_goes_stale_when_its_calibration_changes(sampled_reactor, tmp_path):
    """About 14 s: it redoes the reactor fit, which is what it tests."""
    from aquakin.study import read_study, update_study

    project, _ = _copy_project((sampled_reactor[0], None), tmp_path)
    post = project / "studies" / "post"
    assert check_study(post).state == "fresh"
    fit = project / "studies" / "fit"
    update_study(fit, {**read_study(fit), "sigma": 2 * NOISE})
    status = check_study(post)
    assert status.state == "stale"
    assert status.reasons == ["the setup of the calibrate study it samples has changed"]
    # Run again on a fit that is not fresh: refused, saying why.
    with pytest.raises(StudyError, match=r"fit: the calibrate study 'fit' is stale \("):
        run_study(post, replace=True)
    # Redone, the fit has other results (a wider error, wider uncertainties).
    run_study(fit, replace=True)
    assert check_study(post).reasons == [
        "the fit of the calibrate study it samples has changed",
        "the setup of the calibrate study it samples has changed",
    ]
    update_study(post, SHORT_POSTERIOR)  # a short run: only the state is checked
    assert run_study(post, replace=True).state == "fresh"


def test_a_frozen_posterior_keeps_the_fit_it_sampled(sampled_reactor, tmp_path):
    """Freezing copies the calibrate study, frozen and with its results, into the
    snapshot: the samples stay fresh, and run again, after the original is gone."""
    project, _ = _copy_project((sampled_reactor[0], None), tmp_path)
    post = project / "studies" / "post"
    freeze_study(post)
    kept = post / "snapshot" / "fit"
    assert (kept / "frozen.yaml").is_file() and (kept / "results" / "fit.yaml").is_file()
    assert not list((project / "studies").glob(".*"))  # the copy frozen beside is gone
    shutil.rmtree(project / "studies" / "fit")
    assert check_study(post).state == "fresh"
    (post / "study.yaml").write_text(yaml.safe_dump({"study": SHORT_POSTERIOR}))
    assert run_study(post, replace=True).state == "fresh"
    unfreeze_study(post, discard_results=True)
    assert not (post / "snapshot").exists()
    with pytest.raises(StudyError, match="fit: this project has no study 'fit'"):
        run_study(post)


def test_a_posterior_of_a_frozen_fit_keeps_that_fit_s_own_snapshot(sampled_reactor, tmp_path):
    """A calibrate study frozen already is copied as it is, not frozen again."""
    project, _ = _copy_project((sampled_reactor[0], None), tmp_path)
    fit = project / "studies" / "fit"
    freeze_study(fit)
    record = (fit / "frozen.yaml").read_text()
    post = project / "studies" / "post"
    freeze_study(post)
    assert (post / "snapshot" / "fit" / "frozen.yaml").read_text() == record
    assert check_study(post).state == "fresh"


def test_posterior_results_without_their_draws_read_as_damaged(sampled_reactor, tmp_path):
    """With draws.csv gone the summary still reads, without histograms."""
    from aquakin.study import study_results

    project, _ = _copy_project((sampled_reactor[0], None), tmp_path)
    post = project / "studies" / "post"
    (post / "results" / "draws.csv").unlink()
    results = study_results(post)
    assert results["state"] == "damaged" and "histograms" not in results
    assert set(results["summary"]["parameters"]) == set(TRUE)


@pytest.mark.parametrize(
    "study, message",
    [
        ({"kind": "posterior"}, "a posterior study needs fit"),
        (
            {**POSTERIOR, "reactor": REACTOR},
            "it takes no flowsheet or reactor of its own",
        ),
        (
            {**POSTERIOR, "flowsheet": "../../other.yaml"},
            "it takes no flowsheet or reactor of its own",
        ),
        ({**POSTERIOR, "fit": "../fit"}, r"by its name \(the folder under studies/\), not '\.\./fit'"),
        ({**POSTERIOR, "n_chains": 1}, "n_chains must be at least 2"),
    ],
)
def test_a_posterior_study_that_cannot_run_is_refused_when_read(project, study, message):
    from aquakin.study import load_study

    folder = _study(project, "post", study)
    with pytest.raises(StudyError, match=message):
        load_study(folder)


def test_a_posterior_needs_a_fresh_calibrate_study(project):
    """A fit that has not run, a study of another kind, and no study at all are
    each refused, naming the problem."""
    fit = _study(project, "fit", {"kind": "calibrate", **FIT})
    post = _study(project, "post", POSTERIOR)
    with pytest.raises(StudyError, match=r"fit: the calibrate study 'fit' is not run; run it"):
        run_study(post)
    other = _study(project, "other", {**POSTERIOR, "fit": "post"})
    with pytest.raises(StudyError, match="fit: 'post' is a posterior study; a posterior study"):
        run_study(other)
    lost = _study(project, "lost", {**POSTERIOR, "fit": "nowhere"})
    with pytest.raises(StudyError, match="fit: this project has no study 'nowhere'"):
        run_study(lost)
    assert fit.is_dir()


def test_a_calibration_being_run_is_not_sampled(sampled_reactor, tmp_path, hold_run):
    """Its results would be replaced while they are read: refused until it ends."""
    project, _ = _copy_project((sampled_reactor[0], None), tmp_path)
    hold_run(project / "studies" / "fit")
    with pytest.raises(StudyError, match="fit: the calibrate study 'fit' is running; sample it"):
        run_study(project / "studies" / "post", replace=True)
