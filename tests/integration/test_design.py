"""Activated-sludge design-layer tests.

Two layers: the forward sizing (:func:`size_activated_sludge`) and the Takacs
``solids_mass`` accessor are fast pure-Python / no-solve tests in the PR gate;
the achieved-metric post-processor (:func:`sludge_metrics` / ``plant.sludge_age``)
needs real BSM1 plant solves and is marked ``slow``.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin import SludgeMetrics, size_activated_sludge, sludge_metrics
from aquakin.plant.bsm import build_bsm1
from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
from aquakin.plant.influent import InfluentSeries

# ----- Forward sizing (fast, no solve) ------------------------------------


def test_volume_from_hrt_and_wastage_from_srt():
    """V = Q*HRT and (mixed-liquor) Qw = V/SRT."""
    s = size_activated_sludge(SRT=10.0, HRT_h=8.0, Q=18446.0)
    assert s.volume == pytest.approx(18446.0 * 8.0 / 24.0)
    assert s.wastage_flow == pytest.approx(s.volume / 10.0)
    assert pytest.approx(8.0 / 24.0) == s.HRT
    assert s.tank_volumes == (s.volume,)


def test_hrt_days_equivalent_to_hours():
    a = size_activated_sludge(SRT=12.0, HRT=0.5, Q=1000.0)
    b = size_activated_sludge(SRT=12.0, HRT_h=12.0, Q=1000.0)
    assert a.volume == pytest.approx(b.volume)


def test_tank_split_equal_and_fractional():
    eq = size_activated_sludge(SRT=10.0, HRT_h=8.0, Q=18446.0, n_tanks=5)
    assert len(eq.tank_volumes) == 5
    assert sum(eq.tank_volumes) == pytest.approx(eq.volume)
    assert all(v == pytest.approx(eq.volume / 5) for v in eq.tank_volumes)

    frac = size_activated_sludge(
        SRT=10.0, HRT_h=8.0, Q=18446.0, volume_fractions=[0.1, 0.1, 0.267, 0.267, 0.266]
    )
    assert sum(frac.tank_volumes) == pytest.approx(frac.volume)
    assert frac.tank_volumes[0] == pytest.approx(0.1 * frac.volume)


def test_underflow_wasting_uses_thickening_ratio():
    ml = size_activated_sludge(SRT=10.0, HRT_h=8.0, Q=18446.0)
    uf = size_activated_sludge(
        SRT=10.0, HRT_h=8.0, Q=18446.0, wastage_from="underflow", thickening_ratio=2.0
    )
    # Underflow wasting wastes from a 2x-concentrated stream, so half the flow.
    assert uf.wastage_flow == pytest.approx(ml.wastage_flow / 2.0)


def test_recycle_flows_from_ratios():
    s = size_activated_sludge(
        SRT=10.0, HRT_h=8.0, Q=18446.0, internal_recycle_ratio=3.0, ras_ratio=1.0
    )
    assert s.internal_recycle_flow == pytest.approx(3.0 * 18446.0)
    assert s.ras_flow == pytest.approx(18446.0)


def test_sizing_summary_is_a_string():
    s = size_activated_sludge(SRT=10.0, HRT_h=8.0, Q=18446.0, n_tanks=5)
    assert "SRT" in s.summary() and "wastage" in s.summary()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(SRT=-1.0, HRT_h=8.0, Q=18446.0),
        dict(SRT=10.0, HRT_h=8.0, Q=-1.0),
        dict(SRT=10.0, HRT=0.0, Q=18446.0),  # HRT not > 0
        dict(SRT=10.0, Q=18446.0),  # no HRT
        dict(SRT=10.0, HRT=0.3, HRT_h=8.0, Q=18446.0),  # both HRT
        dict(SRT=10.0, HRT_h=8.0, Q=18446.0, wastage_from="bogus"),
        dict(SRT=10.0, HRT_h=8.0, Q=18446.0, wastage_from="underflow", thickening_ratio=0.0),
        dict(SRT=10.0, HRT_h=8.0, Q=18446.0, volume_fractions=[0.5, 0.4]),  # !=1
        dict(SRT=10.0, HRT_h=8.0, Q=18446.0, volume_fractions=[0.5, -0.5, 1.0]),
    ],
)
def test_sizing_validation_errors(kwargs):
    with pytest.raises(ValueError):
        size_activated_sludge(**kwargs)


# ----- Takacs clarifier solids inventory (fast, no plant solve) -----------


def test_takacs_solids_mass():
    """solids_mass = sum over layers of TSS_layer * layer_volume."""
    from aquakin.plant.takacs import TakacsClarifier

    asm1 = aquakin.load_model("asm1")
    clar = TakacsClarifier(name="c", model=asm1, area=1500.0, height=4.0, underflow_Q=18831.0)
    state = clar.initial_state()
    mass = float(clar.solids_mass(state))
    # Independent recomputation from the layered TSS.
    layered = state.reshape((clar.n_layers, clar._n_part))
    tss = jnp.sum(layered * clar._factors_arr[None, :], axis=1)
    layer_vol = clar.area * clar.height / clar.n_layers
    assert mass == pytest.approx(float(jnp.sum(tss) * layer_vol))
    assert mass > 0.0


def test_reactor_autodetection_finds_as_reactors_not_digester():
    """The AS-reactor auto-detection (used by sludge_metrics / plant.sludge_age
    and by the warm-start) must find the activated-sludge CSTRs and exclude the
    digester. This is a no-solve guard for the detection logic, which otherwise
    is only exercised by the slow plant-solve tests -- so a change to how a CSTR
    advertises its aeration (the field the detection keys on) would slip the fast
    gate. Both call sites share the discriminator and must agree."""
    from aquakin.plant.bsm import build_bsm2
    from aquakin.plant.bsm.warmstart import _as_reactor_names
    from aquakin.plant.design import _reactor_units

    bsm1 = build_bsm1()
    expected = ["tank1", "tank2", "tank3", "tank4", "tank5"]
    assert _reactor_units(bsm1, None) == expected
    assert _as_reactor_names(bsm1) == expected

    bsm2 = build_bsm2()
    # Same five AS reactors; the ADM1 digester (volumed but not a CSTR) is out.
    assert _reactor_units(bsm2, None) == expected
    assert _as_reactor_names(bsm2) == expected
    assert "digester" not in _reactor_units(bsm2, None)


# ----- Sizing result fields / summary branches (fast, no solve) -----------


def test_sizing_recycle_flows_none_by_default():
    """No recycle ratios supplied -> the recycle-flow fields stay None and the
    wasting defaults hold (the None branch of the recycle ternaries)."""
    s = size_activated_sludge(SRT=10.0, HRT_h=8.0, Q=1000.0)
    assert s.internal_recycle_flow is None
    assert s.ras_flow is None
    assert s.wastage_from == "mixed_liquor"
    assert s.thickening_ratio == 1.0


def test_sizing_underflow_ratio_stored_and_reported():
    s = size_activated_sludge(
        SRT=10.0, HRT_h=8.0, Q=18446.0, wastage_from="underflow", thickening_ratio=2.0
    )
    assert s.thickening_ratio == 2.0
    assert "from underflow" in s.summary()


def test_sizing_summary_includes_recycle_and_tank_lines():
    """The three conditional summary blocks (tank cascade, internal recycle,
    RAS) render when the corresponding inputs are supplied."""
    s = size_activated_sludge(
        SRT=10.0, HRT_h=8.0, Q=18446.0, n_tanks=5, internal_recycle_ratio=3.0, ras_ratio=1.0
    )
    out = s.summary()
    assert "tank volumes" in out
    assert "internal recycle" in out
    assert "RAS flow" in out


# ----- Endpoint / reactor / influent resolvers (fast, build-only) ---------


def test_pick_endpoint_explicit_valid_invalid_and_autofail():
    from aquakin.plant.design import (
        _EFFLUENT_CANDIDATES,
        _available_endpoints,
        _pick_endpoint,
    )

    plant = build_bsm1()
    available = sorted(_available_endpoints(plant))
    valid = available[0]
    # Explicit + present -> returned unchanged.
    assert _pick_endpoint(plant, valid, _EFFLUENT_CANDIDATES, "effluent") == valid
    # Explicit + absent -> clear error.
    with pytest.raises(ValueError, match="is not an output of this plant"):
        _pick_endpoint(plant, "nope.port", _EFFLUENT_CANDIDATES, "effluent")
    # No explicit, no candidate present -> auto-detect failure.
    with pytest.raises(ValueError, match="Could not auto-detect"):
        _pick_endpoint(plant, None, ("no.such", "also.no"), "effluent")


def test_reactor_units_explicit_known_and_unknown():
    from aquakin.plant.design import _reactor_units

    plant = build_bsm1()
    assert _reactor_units(plant, ["tank1", "tank2"]) == ["tank1", "tank2"]
    with pytest.raises(ValueError, match="Unknown reactor unit"):
        _reactor_units(plant, ["ghost"])


def test_reactor_units_autodetect_fails_without_as_reactors():
    """design.py:366 -- auto-detection on a plant with no activated-sludge
    reactors (only a mixer, which carries no ``aeration`` attribute) errors
    with a request to pass reactor_units= explicitly."""
    from aquakin.plant import MixerUnit, Plant
    from aquakin.plant.design import _reactor_units

    net = aquakin.load_model("asm1")
    plant = Plant("no_as")
    plant.add_unit(MixerUnit(name="mix", input_port_names=["a", "b"], model=net))
    plant._finalize_topology()  # populate _unit_order for the reactor scan
    with pytest.raises(ValueError, match="Could not auto-detect activated-sludge reactors"):
        _reactor_units(plant, None)


def test_pick_influent_branches():
    """`_pick_influent` reads only `plant.influents`, so exercise its four
    branches with lightweight stubs (no plant build/solve needed)."""
    from types import SimpleNamespace

    from aquakin.plant.design import _pick_influent

    a, b, feed = object(), object(), object()
    # Sole influent -> returned.
    assert _pick_influent(SimpleNamespace(influents={"only": a}), None) is a
    # Several, one named "feed" -> the feed.
    assert _pick_influent(SimpleNamespace(influents={"x": a, "feed": feed}), None) is feed
    # Several, no "feed" -> must disambiguate.
    with pytest.raises(ValueError, match="pass influent_name="):
        _pick_influent(SimpleNamespace(influents={"x": a, "y": b}), None)
    # Explicit unknown name -> clear error.
    with pytest.raises(ValueError, match="Unknown influent"):
        _pick_influent(SimpleNamespace(influents={"x": a}), "ghost")


# ----- Achieved metrics from a solved plant (slow: BSM1 plant solves) -----


def _influent(model):
    C0 = model.concentrations(
        {
            "SI": 30.0,
            "SS": 69.5,
            "XI": 51.2,
            "XS": 202.32,
            "XB_H": 28.17,
            "SNH": 31.56,
            "SND": 6.95,
            "XND": 10.59,
            "SALK": 7.0,
        }
    )
    return InfluentSeries(
        t=jnp.array([0.0, 300.0]), Q=jnp.full((2,), BSM1_Q_AVG), C=jnp.tile(C0, (2, 1)), model=model
    )


def _solve(model, influent, Qw=385.0, use_takacs=False):
    plant = build_bsm1(model=model, wastage_flow=Qw, use_takacs=use_takacs)
    plant.add_influent("feed", influent, to="inlet_mix.fresh")
    sol = plant.solve(
        t_span=(0.0, 80.0),
        t_eval=jnp.linspace(70.0, 80.0, 6),
        rtol=1e-4,
        atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=300_000),
    )
    return plant, sol


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


@pytest.fixture(scope="module")
def ideal_run(asm1):
    return _solve(asm1, _influent(asm1), Qw=385.0, use_takacs=False)


@pytest.mark.slow
def test_sludge_metrics_sensible(ideal_run, asm1):
    plant, sol = ideal_run
    m = sludge_metrics(plant, sol)
    assert isinstance(m, SludgeMetrics)
    # Plausible BSM1 ranges.
    assert 2.0 < m.SRT < 30.0
    assert 0.0 < m.FM < 1.0
    assert m.mlss > 0.0 and m.solids_inventory > 0.0
    assert m.solids_wasted > 0.0 and m.solids_effluent > 0.0
    assert m.reactor_units == ["tank1", "tank2", "tank3", "tank4", "tank5"]
    # HRT = total reactor volume / influent flow (recycles excluded).
    V = sum(float(plant.units[n].volume) for n in m.reactor_units)
    assert pytest.approx(V / BSM1_Q_AVG, rel=1e-3) == m.HRT
    assert "SRT" in m.summary()


@pytest.mark.parametrize("bad", ["BOD5", "bod ", "tss", ""])
def test_sludge_metrics_rejects_unknown_substrate(bad):
    """An unrecognized ``substrate`` must raise, not silently fall back to COD
    (which would report an F:M load ~2x off). Validation precedes any use of
    ``plant``/``solution``, so no solve is needed."""
    with pytest.raises(ValueError, match="substrate must be one of"):
        sludge_metrics(None, None, substrate=bad)


@pytest.mark.slow
def test_plant_sludge_age_matches_sludge_metrics(ideal_run):
    plant, sol = ideal_run
    a = plant.sludge_age(sol)
    b = sludge_metrics(plant, sol)
    assert pytest.approx(b.SRT) == a.SRT
    assert pytest.approx(b.HRT) == a.HRT
    assert pytest.approx(b.FM) == a.FM


@pytest.mark.slow
def test_srt_decreases_with_wastage(asm1):
    """Wasting more sludge gives a younger sludge (smaller SRT) -- the
    monotonic relation the target-SRT iteration relies on."""
    inf = _influent(asm1)
    low = sludge_metrics(*_solve(asm1, inf, Qw=250.0)).SRT
    high = sludge_metrics(*_solve(asm1, inf, Qw=600.0)).SRT
    assert low > high


@pytest.mark.slow
def test_takacs_clarifier_inventory_raises_srt(ideal_run, asm1):
    """The Takacs settler holds a real sludge blanket, so counting its solids
    gives a larger system SRT than the stateless ideal clarifier at the same
    wastage flow."""
    _, ideal_sol = ideal_run
    ideal_plant, _ = ideal_run
    srt_ideal = sludge_metrics(ideal_plant, ideal_sol).SRT
    srt_takacs = sludge_metrics(*_solve(asm1, _influent(asm1), Qw=385.0, use_takacs=True)).SRT
    assert srt_takacs > srt_ideal
