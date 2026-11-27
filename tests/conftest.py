"""Shared pytest fixtures."""

from pathlib import Path

import pytest

import aquakin

from . import mem_watchdog

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples"

# Each shipped model with a TSS state: that state's name and, per particulate,
# the parameter converting it to TSS (iTSS_* in ASM2d, iSS_* in ASM3).
TSS_FACTOR_PARAMS = {
    "asm3": ("XSS", {"XI": "iSS_XI", "XS": "iSS_XS", "XH": "iSS_BM", "XA": "iSS_BM",
                     "XSTO": "iSS_STO"}),
    "asm2d": ("XTSS", {"XI": "iTSS_XI", "XS": "iTSS_XS", "XH": "iTSS_BM", "XPAO": "iTSS_BM",
                       "XAUT": "iTSS_BM", "XPP": "iTSS_XPP", "XPHA": "iTSS_XPHA"}),
    "asm2d_tud": ("XTSS", {"XI": "iTSS_XI", "XS": "iTSS_XS", "XH": "iTSS_BM",
                           "XPAO": "iTSS_BM", "XA": "iTSS_BM", "XPP": "iTSS_PP",
                           "XPHA": "iTSS_PHA", "XGLY": "iTSS_GLY"}),
    "asm3_biop": ("XTSS", {"XI": "iTSS_XI", "XS": "iTSS_XS", "XH": "iTSS_BM",
                           "XPAO": "iTSS_BM", "XA": "iTSS_BM", "XPP": "iTSS_XPP",
                           "XPHA": "iTSS_XSTO", "XSTO": "iTSS_XSTO"}),
}

# Fire the memory watchdog at most once per worker process, so an over-limit
# shard reports a single clear failure at the crossing point rather than
# flooding the log with one failure per remaining test.
_watchdog_fired = False


def pytest_runtest_teardown(item, nextitem):
    """Fail loudly when a shard's free memory hits the floor (see mem_watchdog).

    Runs after every test; a no-op unless ``AQUAKIN_TEST_MEM_WATCHDOG=1`` and
    the runner is genuinely low on memory. Turns the otherwise-silent OOM
    runner reclaim into a named failure.
    """
    global _watchdog_fired
    if _watchdog_fired or not mem_watchdog.watchdog_enabled():
        return
    available = mem_watchdog.mem_available_mb()
    floor = mem_watchdog.mem_floor_mb()
    if mem_watchdog.watchdog_should_fire(available, floor):
        _watchdog_fired = True
        pytest.fail(mem_watchdog.watchdog_message(item.nodeid, available, floor), pytrace=False)


# The duration event JAX records for every XLA compilation.
_BACKEND_COMPILE = "/jax/core/compile/backend_compile_duration"


@pytest.fixture
def xla_compiles():
    """A list that gains one entry per XLA compilation while the test runs."""
    import jax

    compiles = []

    def listener(event, duration, **kwargs):
        if event == _BACKEND_COMPILE:
            compiles.append(duration)

    jax.monitoring.register_event_duration_secs_listener(listener)
    yield compiles
    jax.monitoring.unregister_event_duration_listener(listener)


@pytest.fixture(scope="session")
def simple_model():
    """Load the simple A -> B test model, once per session.

    The compiled-solve caches key on the model's identity, so one shared model
    lets every test reuse a solve another test already compiled. Do not mutate
    it: a test that needs to change the model loads its own copy.
    """
    return aquakin.load_model_from_file(FIXTURES / "simple_model.yaml")


@pytest.fixture(scope="session")
def stiff_sewer_model():
    """Load the stiff sewer sulfur-cycle test model, once per session.

    A WATS-based carbon and sulfur model with nitrate-driven sulfide oxidation
    and a positivity limiter: stiff enough that reverse-mode AD through an
    uncapped solve overflows, which is what the stable adjoints and the
    forward-sensitivity solve are tested against. It is a frozen test input,
    not a shipped model. Do not mutate it.
    """
    return aquakin.load_model_from_file(FIXTURES / "wats_sewer_khalil_paper_balanced.yaml")


@pytest.fixture
def make_cstr_plant(simple_model):
    """Build a one-tank plant: a CSTR (V = 100) on the A -> B decay model, fed
    Q = 10 of A = 1 through the influent port ``feed``. Each call returns a new
    plant, with its own caches.

    Small enough to compile and solve in seconds, with a closed-form solution
    (``dA/dt = 0.1 (1 - A) - k A``) and a two-state Jacobian whose only
    off-diagonal entry is the A -> B coupling ``k``.

    ``make(aeration=...)`` gives the tank an :class:`~aquakin.plant.Aeration`
    (on species ``A``, the model having no oxygen), for a plant whose tank is
    actuated through the signal bus.
    """
    import jax.numpy as jnp

    from aquakin.plant import CSTRUnit, Plant
    from aquakin.plant.influent import InfluentSeries

    def make(aeration=None):
        plant = Plant("single_cstr", time_unit="s")
        plant.add_unit(
            CSTRUnit(
                name="tank",
                model=simple_model,
                volume=100.0,
                input_port_names=["inlet"],
                conditions={"T": 293.15},
                aeration=aeration,
            )
        )
        influent = InfluentSeries(
            t=jnp.asarray([0.0, 100.0]),
            Q=jnp.asarray([10.0, 10.0]),
            C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
            model=simple_model,
        )
        plant.add_influent("feed", influent, to="tank.inlet")
        return plant

    return make


@pytest.fixture(scope="session")
def sbr_model():
    """The A -> B decay model with a COD content on both species (so a plant's
    mass balance can be read), for the small SBR plant; loaded once."""
    return aquakin.load_model_from_file(FIXTURES / "sbr_decay_model.yaml")


#: The small SBR's fill-only feed: 10 m3/s during the first 5 s of each 27 s
#: cycle and nothing otherwise (``step`` is 1 from zero up).
SMALL_SBR_FILL_ONLY_Q = "10*(1 - step(mod(t, 27) - 5))"
#: The same volume per cycle, arriving continuously (50 m3 over 27 s).
SMALL_SBR_CONTINUOUS_Q = 50.0 / 27.0


@pytest.fixture(scope="session")
def make_small_sbr_plant(sbr_model):
    """Build a one-SBR plant on the A -> B decay model (in seconds, B taken as
    the settling particulate): a heel of 50 m3 and 50 m3 fed and decanted each
    27 s cycle of fill / react / settle / decant / idle. Each call returns a new
    plant; it compiles and solves in seconds.

    ``make(feed="batch")`` feeds 10 m3/s of A = 1 during the fill phase only,
    so the tank fills to 100 m3 and decants back to 50 m3. ``feed="continuous"``
    and ``feed="storage"`` (a 60 m3 holding volume, half full) feed the same
    50 m3 per cycle continuously. Keyword arguments replace the SBR's settings;
    ``Q=`` replaces the influent's flow (a number or an expression of time).
    """
    from aquakin.plant import InterfaceSettling, Plant, SBRPhase, SBRUnit
    from aquakin.plant.influent import InfluentExpression

    def make(feed="batch", Q=None, **settings):
        phases = [
            SBRPhase("fill", 5.0, feed=True),
            SBRPhase("react", 10.0, mixed=True),
            SBRPhase("settle", 5.0, settle=True),
            SBRPhase("decant", 5.0, decant=True),
            SBRPhase("idle", 2.0),
        ]
        by_mode = {
            "batch": ({"feed_flow": 10.0}, SMALL_SBR_FILL_ONLY_Q),
            "continuous": ({"feed_flow": None}, SMALL_SBR_CONTINUOUS_Q),
            "storage": ({"feed_flow": 10.0, "storage_volume": 60.0}, SMALL_SBR_CONTINUOUS_Q),
        }
        own, default_Q = by_mode[feed]
        sbr = SBRUnit(
            "sbr",
            sbr_model,
            phases,
            **{
                "full_volume": 100.0,
                "decant_flow": 10.0,
                "settling": InterfaceSettling(v_settle=2.0, area=10.0, remix_rate=1.0),
                "particulate_species": ["B"],
                "initial_fraction": 0.5,
                "conditions": {"T": 293.15},
                "feed": feed,
                **own,
                **settings,
            },
        )
        plant = Plant("small_sbr", time_unit="s")
        plant.add_unit(sbr)
        influent = InfluentExpression(
            sbr_model, Q=default_Q if Q is None else Q, composition={"A": 1.0}
        )
        plant.add_influent("feed", influent, to="sbr.feed")
        return plant

    return make


@pytest.fixture
def cstr_plant(make_cstr_plant):
    """One plant from :func:`make_cstr_plant`."""
    return make_cstr_plant()


# --- Tiering: defer full-plant integration tests to the merge-only `slow` job --
#
# Compiling a stiff solve dominates the test cost; building + integrating the
# full BSM2 plant costs ~30 s to compile, which no caching removes (each is a
# distinct configuration). To keep the PR fast-gate cheap, every test that
# requests one of these "build + solve a full plant" *module fixtures* is marked
# ``slow`` here, so it runs only in the merge-to-main `slow` job. The PR gate
# keeps the cheap coverage of those features -- the unit-level logic tests
# (controller math, storage regimes, schedule, delay, metric kernels), the plant
# *assembly* / flow-resolution checks, and the small-plant smokes in
# test_plant_assembly -- which is where a developer's bug surfaces fastest.
#
# To defer a new plant-solving test, give it (or have it use) one of these
# fixtures, or add its fixture name here. A heavy test that does NOT use a
# fixture (e.g. a parametrized ``jax.grad`` check) carries an explicit
# ``@pytest.mark.slow`` in its own file instead.
_SLOW_PLANT_FIXTURES = frozenset({
    "steady",        # test_bsm2: open-loop steady-state solve
    "bsm2_open_loop_10d",  # integration/conftest: shared 10-day open-loop BSM2 solve
    "evaluated",     # test_bsm2_evaluation: EQI/OCI on a solved plant
    "storage_run",   # test_bsm2_storage
    "closed_sol",    # test_bsm2_control: DO closed-loop solve
    "open_sol",      # test_bsm2_control: open-loop contrast solve
    "control_run",   # test_bsm2_reject_control
    "wastage_run",   # test_bsm2_wastage
    "delay_run",     # test_bsm2_hydraulic_delay
    "bypass_run",    # test_bsm2_bypass
    "growth_setup",  # test_profile: calibrate + profile_likelihood on a growth model
})


def pytest_collection_modifyitems(config, items):
    """Mark any test using a full-plant-solving fixture ``slow`` (see above)."""
    slow = pytest.mark.slow
    for item in items:
        if _SLOW_PLANT_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(slow)


def study_project(tmp_path):
    """A one-tank project with one simulate study, written with aquakin."""
    import aquakin
    from aquakin.plant import CSTRUnit, InfluentSeries, Plant

    m = aquakin.load_model("asm1")
    p = Plant("tank")
    p.add_unit(CSTRUnit("tank", m, 1000.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(m, Q=2000.0, SS=60.0, SNH=30.0), to="tank.in")
    project = tmp_path / "project"
    p.to_spec(project / "flowsheet.yaml")
    study = project / "studies" / "short"
    study.mkdir(parents=True)
    (study / "study.yaml").write_text(
        "study:\n  kind: simulate\n  description: half a day\n"
        "  time: {t_span: [0, 0.5], t_eval: {start: 0, stop: 0.5, num: 3}}\n"
        "  streams: [tank]\n"
    )
    return project


#: A second program that holds a study's run lock until its input closes,
#: reporting progress at each line it reads ("cancelled" once it is stopped).
_HOLD = """
import sys
from pathlib import Path
from aquakin.progress import Cancelled, Progress
from aquakin.study.running import RunLock
with RunLock(Path(sys.argv[1])) as run:
    run.report(Progress("fit", 3, 10, "step 3"))
    print("held", flush=True)
    for line in sys.stdin:
        try:
            run.report(Progress("fit", 4, 10, "step 4"))
        except Cancelled:
            print("cancelled", flush=True)
            break
        print("reported", flush=True)
"""


@pytest.fixture
def hold_run():
    """``hold_run(folder) -> Popen``: another process holding a study's run lock
    (``aquakin/study/running.py``) at a first progress report, "step 3".

    Each line written to its input makes it report again: it answers
    "reported", or "cancelled" once the study was cancelled (and then ends).
    Closing its input ends the run. Every holder is ended at the test's end.
    """
    import subprocess
    import sys

    started = []

    def hold(folder):
        proc = subprocess.Popen(
            [sys.executable, "-c", _HOLD, str(folder)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        started.append(proc)
        assert proc.stdout.readline() == "held\n"
        return proc

    yield hold
    for proc in started:
        if not proc.stdin.closed:
            proc.stdin.close()  # ends a holder still running; a closed one has gone
        proc.wait(60)
        proc.stdout.close()
