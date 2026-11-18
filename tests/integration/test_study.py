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
from aquakin.plant import CSTRUnit, InfluentSeries, Plant, plant_fisher_information
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
    with pytest.raises(KeyError):
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
    "max_iter": 40,
    # ASM1's literature prior on muH (centred at 6.0) would pull the estimate.
    "use_priors": False,
}


def test_a_calibration_study_recovers_a_parameter(project):
    folder = project / "studies" / "fit"
    _synthetic_data(project, folder, 5.0)
    (folder / "study.yaml").write_text(
        yaml.safe_dump({"study": {"kind": "calibrate", "parameters": {"asm1_local.muH": 3}, **FIT}})
    )
    seen = []
    assert run_study(folder, progress=seen.append).state == "fresh"
    assert seen and seen[0].stage == "fit"
    fit = yaml.safe_load((folder / "results" / "fit.yaml").read_text())
    assert fit["fitted"]["asm1_local.muH"] == pytest.approx(5.0, rel=1e-3)
    assert read_named_values(folder / "results" / "parameters.yaml")["asm1_local.muH"] == (
        pytest.approx(5.0, rel=1e-3)
    )
    _, fitted = read_trajectory(folder / "results" / "fitted.csv")
    _, data = read_trajectory(folder / "data.csv")
    np.testing.assert_allclose(fitted["effluent.SNH"], data["effluent.SNH"], rtol=1e-4)
    # The data table is one of the study's inputs.
    (folder / "data.csv").write_text((folder / "data.csv").read_text().replace("0.1,", "0.15,"))
    assert "the study's settings have changed" in check_study(folder).reasons


def test_a_study_names_the_data_columns_it_cannot_find(project):
    folder = project / "studies" / "nodata"
    _synthetic_data(project, folder, 4.0)
    study = {**FIT, "observables": [{"stream": "effluent", "channels": ["SNO"]}]}
    (folder / "study.yaml").write_text(yaml.safe_dump({"study": {"kind": "calibrate", **study}}))
    with pytest.raises(StudyError, match="has no column effluent.SNO"):
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


@pytest.mark.slow
def test_profile_and_rank_studies(project):
    from aquakin.study import read_table

    folder = project / "studies" / "profile"
    _synthetic_data(project, folder, 4.0)
    study = {
        "kind": "profile",
        **FIT,
        "free": ["asm1_local.muH", "asm1_local.KS"],
        "transforms": {},
        "profile": "asm1_local.muH",
        "grid": {"start": 3.5, "stop": 4.5, "num": 3},
        "loss": "nll",
        "sigma": 0.1,
    }
    (folder / "study.yaml").write_text(yaml.safe_dump({"study": study}))
    run_study(folder)
    table = read_table(folder / "results" / "profile.csv")
    assert table["asm1_local.muH"] == [3.5, 4.0, 4.5]
    summary = yaml.safe_load((folder / "results" / "profile.yaml").read_text())
    assert summary["mle"] == 4.0

    folder = _study(
        project,
        "rank",
        {
            "kind": "rank",
            "free": ["asm1_local.muH", "asm1_local.KS"],
            "t_obs": [0.1, 0.2, 0.3],
            "t_span": [0, 0.3],
            "candidates": [
                {
                    "name": "ammonia",
                    "observables": [{"stream": "effluent", "channels": ["SNH"]}],
                    "sigma": 0.1,
                },
                {
                    "name": "flow",
                    "observables": [{"stream": "effluent", "channels": ["Q"]}],
                    "sigma": 0.1,
                },
            ],
        },
    )
    run_study(folder)
    ranking = read_table(folder / "results" / "ranking.csv")
    assert ranking["candidate"][0] == "ammonia" and ranking["rank"] == [1.0, 2.0]
