"""Plants rebuilt from one spec share their compiled solves, and only they do.

A study run, or the editor's worker, rebuilds its plant from the spec on every
run; each rebuilt plant used to compile its solve again. A plant built by
``Plant.from_spec`` now shares its compiled forward and steady-state solves
with every plant built from a spec of the same fingerprint, as built (model and
table content included). These tests count XLA compilations, check a shared
program's results bit for bit against a plant that compiled its own, and check
that specs that differ in anything the program bakes in, and a plant changed
after it was built, never share.
"""

import jax
import numpy as np
import pytest
import yaml

from aquakin.plant import Plant
from aquakin.plant import solve_cache as solve_cache_mod
from aquakin.plant.solve_cache import SolveCache, clear_shared, shared_size
from aquakin.plant.spec import spec_fingerprint
from aquakin.study import run_study

T_SPAN = (0.0, 20.0)
T_EVAL = np.linspace(0.0, 20.0, 5)


@pytest.fixture
def compiled_names():
    """A list that gains the name of each program XLA compiles while the test
    runs (``jit(rhs_y)``, say), read from JAX's own compile log."""
    import logging

    names = []

    class Names(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith("Finished XLA compilation of "):
                names.append(message.split()[4])

    handler = Names()
    logger = logging.getLogger("jax")
    logger.addHandler(handler)
    with jax.log_compiles(True):
        yield names
    logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def _empty_shared_store():
    """Each test starts and ends with nothing shared, so no test sees another's
    programs."""
    clear_shared()
    yield
    clear_shared()


@pytest.fixture
def project(tmp_path, make_cstr_plant):
    """The one-tank plant of ``make_cstr_plant`` written as a project folder (its
    model a file under ``models/``, so every ``from_spec`` loads it afresh)."""
    folder = tmp_path / "project"
    make_cstr_plant().to_spec(folder / "flowsheet.yaml")
    return folder


def _edited(project, tmp_path, name, edit):
    """A copy of ``project`` with ``edit(spec, folder)`` applied to its spec."""
    import shutil

    folder = tmp_path / name
    shutil.copytree(project, folder)
    path = folder / "flowsheet.yaml"
    spec = yaml.safe_load(path.read_text())
    edit(spec, folder)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return folder


def _solve(plant):
    sol = plant.solve(T_SPAN, t_eval=T_EVAL, rtol=1e-8, atol=1e-10)
    return np.asarray(sol.state)


def _fed_from_a_table(a_end):
    """An edit feeding the tank from ``data/feed.csv``, whose A rises from 1 to
    ``a_end`` over the solve: two such projects differ only in the table's bytes."""

    def edit(spec, folder):
        (folder / "data").mkdir(exist_ok=True)
        (folder / "data" / "feed.csv").write_text(f"t,Q,A,B\n0,10,1,0\n20,10,{a_end},0\n")
        spec["units"]["feed"] |= {"Q": None, "composition": {}, "series": "data/feed.csv"}

    return edit


def test_influent_tables_that_differ_never_share(project, tmp_path, xla_compiles):
    """Two projects whose specs are the same text but whose influent tables
    differ compile one program each, and each gives its own table's result; a
    third with the first one's table compiles nothing."""
    one = _solve(
        Plant.from_spec(
            _edited(project, tmp_path, "one", _fed_from_a_table(2.0)) / "flowsheet.yaml"
        )
    )
    two_folder = _edited(project, tmp_path, "two", _fed_from_a_table(3.0))
    one_spec, two_spec = tmp_path / "one" / "flowsheet.yaml", two_folder / "flowsheet.yaml"
    assert two_spec.read_text() == one_spec.read_text()
    assert spec_fingerprint(two_spec) != spec_fingerprint(one_spec)
    before = len(xla_compiles)
    two = _solve(Plant.from_spec(two_folder / "flowsheet.yaml"))
    assert len(xla_compiles) > before
    assert two[-1, 0] > one[-1, 0]  # more A fed

    before = len(xla_compiles)
    again = _edited(project, tmp_path, "again", _fed_from_a_table(2.0))
    np.testing.assert_array_equal(_solve(Plant.from_spec(again / "flowsheet.yaml")), one)
    assert len(xla_compiles) == before


def _events(plant):
    from aquakin import Event

    refill = Event(at_times=[10.0], apply=lambda t, y, a: y.at[0].set(1.0), name="refill")
    sol = plant.solve(T_SPAN, t_eval=T_EVAL, events=[refill], rtol=1e-8, atol=1e-10)
    return np.asarray(sol.state)


def _forward_fast(plant):
    sol = plant.solve(T_SPAN, t_eval=T_EVAL, forward_fast=True, rtol=1e-8, atol=1e-10)
    return np.asarray(sol.state)


def _steady(plant):
    return np.asarray(plant.steady_state().state)


def _sensitivity(plant):
    _, ys, S = plant.solve_sensitivity(
        plant.default_parameters(), ["simple_decay.A_to_B.k"], t_span=T_SPAN, t_eval=T_EVAL
    )
    return np.concatenate([np.asarray(ys).ravel(), np.asarray(S).ravel()])


# The coloured-Jacobian pattern of the lean forward solve and of the staggered
# sensitivity is probed once per plant (a few milliseconds' compile of the
# plant's right-hand side): it is the plant's own, not part of the shared solve.
_PER_PLANT = {"jit(rhs_y)"}


@pytest.mark.parametrize(
    "run",
    [_solve, _events, _forward_fast, _steady, _sensitivity],
    ids=lambda f: f.__name__.strip("_"),
)
def test_a_rebuilt_plant_reuses_the_compiled_solve_bit_for_bit(project, run, compiled_names):
    """A second plant built from the same spec (its model file loaded again, so
    a different model object) compiles no solve -- dynamic, with located
    events, the lean forward solve, a steady state or a forward sensitivity --
    and its result equals, bit for bit, that of a plant that compiled its own
    program with nothing shared. The plant's identity is the spec's own
    fingerprint."""
    path = project / "flowsheet.yaml"
    first = Plant.from_spec(path)
    assert first._solve_cache.shared_as == (Plant, spec_fingerprint(path))
    run(first)
    assert shared_size() == 1

    second = Plant.from_spec(path)
    assert second.units["tank"].model is not first.units["tank"].model
    before = len(compiled_names)
    shared = run(second)
    assert set(compiled_names[before:]) <= _PER_PLANT

    clear_shared()
    fresh = run(Plant.from_spec(path))
    assert set(compiled_names[before:]) - _PER_PLANT  # it compiled its own solve
    np.testing.assert_array_equal(shared, fresh)


def test_a_shipped_model_is_fingerprinted_as_loaded(tmp_path):
    """A plant on a shipped model is identified by the spec's fingerprint, the
    model digested as ``load_model`` loaded it."""
    from aquakin import load_model
    from aquakin.plant import CSTRUnit, InfluentSeries

    m = load_model("asm1")
    p = Plant("tank", time_unit="d")
    p.add_unit(CSTRUnit("tank", m, 1000.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(m, Q=2000.0, SS=60.0), to="tank.in")
    path = tmp_path / "flowsheet.yaml"
    p.to_spec(path)
    built = Plant.from_spec(path)
    assert built._solve_cache.shared_as == (Plant, spec_fingerprint(path))


def test_the_shared_solve_is_the_closed_form(project):
    """The shared program solves the tank: dA/dt = 0.1 (1 - A) - 0.1 A from
    A = 1, for the plant that compiled it and for a rebuilt one."""
    a = 0.5 + 0.5 * np.exp(-0.2 * T_EVAL)
    for _ in range(2):
        np.testing.assert_allclose(
            _solve(Plant.from_spec(project / "flowsheet.yaml"))[:, 0], a, rtol=1e-6
        )


def _set_volume(spec, folder):
    spec["units"]["tank"]["volume"] = 50.0


def _set_feed(spec, folder):
    spec["units"]["feed"]["composition"]["A"] = 2.0


def _set_rate(spec, folder):
    path = next((folder / "models").glob("*.yaml"))
    path.write_text(path.read_text().replace("value: 0.1", "value: 0.3"))
    for ref in spec["models"].values():
        ref.pop("sha256")  # the edit is deliberate: no warning that the file changed


@pytest.mark.parametrize("edit", [_set_volume, _set_feed, _set_rate], ids=lambda f: f.__name__)
def test_specs_that_differ_in_what_is_compiled_never_share(project, tmp_path, edit, xla_compiles):
    """A spec that differs in a setting, an influent value or its model file's
    content (each a constant of the compiled program) compiles its own solve,
    and gives the result of a plant built from it with nothing shared, not the
    original plant's."""
    original = _solve(Plant.from_spec(project / "flowsheet.yaml"))
    other = _edited(project, tmp_path, "other", edit)
    before = len(xla_compiles)
    changed = _solve(Plant.from_spec(other / "flowsheet.yaml"))
    assert len(xla_compiles) > before
    assert not np.array_equal(changed, original)

    clear_shared()
    np.testing.assert_array_equal(changed, _solve(Plant.from_spec(other / "flowsheet.yaml")))


def test_where_the_files_sit_and_the_plant_name_do_not_stop_sharing(
    project, tmp_path, xla_compiles
):
    """A copy of the project in another folder under another plant name is the
    same plant: it compiles nothing."""

    def rename(spec, folder):
        spec["name"] = "renamed"

    _solve(Plant.from_spec(project / "flowsheet.yaml"))
    other = _edited(project, tmp_path, "elsewhere", rename)
    before = len(xla_compiles)
    _solve(Plant.from_spec(other / "flowsheet.yaml"))
    assert len(xla_compiles) == before


def test_a_plant_changed_after_it_was_built_stops_sharing(project, xla_compiles):
    """A plant changed after ``from_spec`` (a unit added) neither reads nor
    publishes shared programs, and the programs it published before the change
    are withdrawn, so a later plant of the spec compiles its own; the later
    plant's result equals a fresh compile's."""
    from aquakin.plant import CSTRUnit

    first = Plant.from_spec(project / "flowsheet.yaml")
    unchanged = _solve(first)
    assert shared_size() == 1

    first.add_unit(
        CSTRUnit("spare", first.units["tank"].model, 10.0, ["inlet"], conditions={"T": 293.15})
    )
    assert shared_size() == 0
    first.connect("tank.out", "spare.inlet")
    before = len(xla_compiles)
    _solve(first)
    assert len(xla_compiles) > before
    assert shared_size() == 0  # a changed plant publishes nothing

    before = len(xla_compiles)
    later = _solve(Plant.from_spec(project / "flowsheet.yaml"))
    assert len(xla_compiles) > before
    np.testing.assert_array_equal(later, unchanged)


def test_a_plant_whose_temperature_is_set_does_not_read_shared_programs(project, xla_compiles):
    """``set_temperature`` changes a constant of the compiled program, so a
    plant built from the spec and then given another temperature compiles its
    own solve rather than reusing the unchanged plant's."""
    _solve(Plant.from_spec(project / "flowsheet.yaml"))
    warmer = Plant.from_spec(project / "flowsheet.yaml").set_temperature(30.0)
    before = len(xla_compiles)
    _solve(warmer)
    assert len(xla_compiles) > before
    assert warmer.units["tank"].conditions["T"] == pytest.approx(303.15)


def test_the_shared_store_keeps_the_most_recently_used_programs(monkeypatch):
    """Past ``SHARED_LIMIT`` the least recently used program is dropped; a
    program read is the most recently used."""
    monkeypatch.setattr(solve_cache_mod, "SHARED_LIMIT", 2)
    cache = SolveCache()
    cache.share_as("plant")
    cache.put("jit", "a", 1)
    cache.put("jit", "b", 2)
    reader = SolveCache()
    reader.share_as("plant")
    assert reader.get("jit", "a") == 1  # now "b" is the least recently used
    cache.put("steady_jit", "c", 3)
    assert shared_size() == 2
    assert reader.get("jit", "b") is None
    assert (reader.get("jit", "a"), reader.get("steady_jit", "c")) == (1, 3)
    assert reader.jit == {}  # read, not copied: a withdrawn program is never reused
    assert reader.get("jit", None) is None


def test_only_the_solve_stores_are_shared():
    """The continuation and arclength kernels close over a known solution's
    scale that their keys omit, so they cannot be published."""
    cache = SolveCache()
    with pytest.raises(ValueError, match="no shared store 'continuation_kernels'"):
        cache.put("continuation_kernels", "k", 1)


def test_a_second_study_run_compiles_nothing_and_writes_the_same_results(project, xla_compiles):
    """Running a dynamic simulation study again (the plant rebuilt from its
    spec) compiles nothing and writes byte-identical results."""
    folder = project / "studies" / "short"
    folder.mkdir(parents=True)
    study = {
        "kind": "simulate",
        "time": {"t_span": [0, 20], "t_eval": {"start": 0, "stop": 20, "num": 5}},
        "streams": ["tank.out"],
    }
    (folder / "study.yaml").write_text(yaml.safe_dump({"study": study}))
    run_study(folder)
    first = (folder / "results" / "streams.csv").read_bytes()
    before = len(xla_compiles)
    run_study(folder, replace=True)
    assert len(xla_compiles) == before
    assert (folder / "results" / "streams.csv").read_bytes() == first
