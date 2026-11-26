"""Activated-sludge design-layer tests.

Two layers: the forward sizing (:func:`size_activated_sludge`) and the Takacs
``solids_mass`` accessor are fast pure-Python / no-solve tests in the PR gate;
the achieved-metric post-processor (:func:`sludge_metrics` / ``plant.sludge_age``)
is checked against hand computations on solutions built at known states, with
real BSM1 / BSM2 solves (``slow``) only where the plant's response is the point.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import SludgeMetrics, size_activated_sludge, sludge_metrics
from aquakin.plant.bsm import bsm1_warm_start, build_bsm1
from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
from aquakin.plant.influent import InfluentSeries
from tests.conftest import TSS_FACTOR_PARAMS

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
    ("kwargs", "message"),
    [
        (dict(SRT=-1.0, HRT_h=8.0, Q=18446.0), "SRT must be > 0"),
        (dict(SRT=10.0, HRT_h=8.0, Q=-1.0), "Q must be > 0"),
        (dict(SRT=10.0, HRT=0.0, Q=18446.0), "HRT must be > 0"),  # HRT not > 0
        (dict(SRT=10.0, Q=18446.0), "Supply exactly one of HRT"),  # no HRT
        (dict(SRT=10.0, HRT=0.3, HRT_h=8.0, Q=18446.0), "Supply exactly one of HRT"),  # both HRT
        (dict(SRT=10.0, HRT_h=8.0, Q=18446.0, wastage_from="bogus"),
         "wastage_from must be 'mixed_liquor' or 'underflow'"),
        (dict(SRT=10.0, HRT_h=8.0, Q=18446.0, wastage_from="underflow", thickening_ratio=0.0),
         "thickening_ratio must be > 0"),
        (dict(SRT=10.0, HRT_h=8.0, Q=18446.0, volume_fractions=[0.5, 0.4]),  # !=1
         "volume_fractions must sum to 1"),
        (dict(SRT=10.0, HRT_h=8.0, Q=18446.0, volume_fractions=[0.5, -0.5, 1.0]),
         "volume_fractions must all be > 0"),
    ],
)
def test_sizing_validation_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
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


def test_srt_ports_resolve_to_activated_sludge_losses():
    """The SRT solids losses are the secondary-clarifier overflow and the
    clarifier-underflow wastage, not the BSM2 dewatering cake (which also holds
    primary and digested solids, understating the SRT) nor a bypass-combined
    final effluent (whose bypass solids never entered the AS system)."""
    from aquakin.plant import build_a2o
    from aquakin.plant.bsm import build_bsm2
    from aquakin.plant.bsm.bsm2 import InfluentBypass
    from aquakin.plant.design import _srt_ports

    assert _srt_ports(build_bsm1()) == ("clarifier.overflow", "underflow_split.waste")
    assert _srt_ports(build_a2o()) == ("clarifier.overflow", "underflow_split.waste")
    assert _srt_ports(build_bsm2()) == ("settler.overflow", "underflow_split.waste")
    bypassed = build_bsm2(bypass=InfluentBypass())
    assert bypassed.effluent_endpoint == "effluent_mix.out"
    assert _srt_ports(bypassed) == ("settler.overflow", "underflow_split.waste")
    # Explicit ports still win.
    assert _srt_ports(bypassed, "effluent_mix.out", "dewatering.underflow") == (
        "effluent_mix.out",
        "dewatering.underflow",
    )


def test_srt_ports_prefer_recorded_endpoints_over_unit_names():
    """A custom plant whose *primary* clarifier is named ``settler``: the SRT
    losses are its recorded final effluent and registered wastage, not the
    primary overflow that the name fallback would pick (issue #630)."""
    from aquakin.plant import Plant
    from aquakin.plant.clarifier import IdealClarifier
    from aquakin.plant.design import _srt_ports

    asm1 = aquakin.load_model("asm1")
    plant = Plant("custom", time_unit="d")
    plant.add_unit(IdealClarifier(name="settler", model=asm1, underflow_Q=100.0))
    plant.add_unit(IdealClarifier(name="final", model=asm1, underflow_Q=100.0))
    # Without a record the name fallback still applies.
    assert _srt_ports(plant, waste_port="final.underflow")[0] == "settler.overflow"
    with pytest.raises(ValueError, match="wastage"):
        _srt_ports(plant)
    plant.effluent_endpoint = "final.overflow"
    plant.register_stream("wastage", "final.underflow")
    assert _srt_ports(plant) == ("final.overflow", "final.underflow")


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
    plant = Plant("no_as", time_unit="d")
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


# ----- Achieved metrics from a plant solution ------------------------------
#
# SRT, HRT and F:M are functions of the saved states, so most of these checks
# read a solution built at known states (the BSM1 warm start, which is close to
# the BSM1 steady state) without integrating, and compare sludge_metrics with
# the same quantities computed by hand. Only the wastage ordering needs the
# plant's dynamics (the inventory has to respond to the wastage flow) and runs
# real solves, marked ``slow``.


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


def _bsm1(model, Qw=385.0, use_takacs=False):
    plant = build_bsm1(model=model, wastage_flow=Qw, use_takacs=use_takacs)
    plant.add_influent("feed", _influent(model), to="inlet_mix.fresh")
    return plant


def _solve(model, Qw=385.0):
    plant = _bsm1(model, Qw=Qw)
    sol = plant.solve(
        t_span=(0.0, 80.0),
        t_eval=jnp.linspace(70.0, 80.0, 6),
        rtol=1e-4,
        atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=300_000),
    )
    return plant, sol


_REACTORS = ["tank1", "tank2", "tank3", "tank4", "tank5"]


def _warm_solution(plant):
    """A two-point solution with no solve: the BSM1 warm start at t = 0 and the
    same state with every reactor's concentrations 20% higher at t = 1, so the
    window means are not those of a constant."""
    from aquakin.plant.results import PlantSolution

    y0 = jnp.asarray(bsm1_warm_start(plant))
    by_unit = plant.states_by_unit(y0)
    y1 = plant.initial_state(overrides={n: 1.2 * by_unit[n] for n in _REACTORS})
    return PlantSolution(t=jnp.asarray([0.0, 1.0]), state=jnp.stack([y0, y1]), plant=plant)


def _tss_by_hand(C, model):
    """BSM1 TSS: 0.75 g TSS per g COD of the five particulate COD species."""
    idx = [model.species_index[s] for s in ("XS", "XI", "XB_H", "XB_A", "XP")]
    return 0.75 * np.sum(np.asarray(C)[..., idx], axis=-1)


def _srt_by_hand(plant, sol, model):
    """``(SRT, inventory, wasted, effluent)``: the window-mean solids inventory
    (reactors plus any Takács blanket, g) over the window-mean solids leaving in
    the clarifier overflow and the wastage (g/d), from the saved states."""
    t = np.asarray(sol.t)

    def mean(x):
        return float(np.trapezoid(x, t) / (t[-1] - t[0]))

    inventory = sum(
        float(plant.units[n].volume) * _tss_by_hand(sol.unit_state(n), model) for n in _REACTORS
    )
    clarifier = plant.units["clarifier"]
    if clarifier.state_size > 0:
        # Lumped-TSS Takács settler: the first n_layers states are layer TSS.
        assert clarifier.composition_mode == "lumped_tss"
        layers = np.asarray(sol.unit_state("clarifier"))[:, : clarifier.n_layers]
        layer_volume = clarifier.area * clarifier.height / clarifier.n_layers
        inventory = inventory + layers.sum(axis=1) * layer_volume
    eff = plant.stream(sol, "clarifier.overflow")
    waste = plant.stream(sol, "underflow_split.waste")
    eff_rate = np.asarray(eff.Q) * _tss_by_hand(eff.C, model)
    waste_rate = np.asarray(waste.Q) * _tss_by_hand(waste.C, model)
    loss = mean(eff_rate) + mean(waste_rate)
    return mean(inventory) / loss, mean(inventory), mean(waste_rate), mean(eff_rate)


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


@pytest.fixture(scope="module")
def ideal_warm(asm1):
    plant = _bsm1(asm1)
    return plant, _warm_solution(plant)


def test_sludge_metrics_match_hand_computed_srt(ideal_warm, asm1):
    """SRT = mean solids inventory / mean solids leaving (overflow + wastage),
    each term matching a hand computation from the states; HRT = reactor volume
    / influent flow; and the values lie in the BSM1 range at the warm start."""
    plant, sol = ideal_warm
    m = sludge_metrics(plant, sol)
    assert isinstance(m, SludgeMetrics)
    srt, inventory, wasted, effluent = _srt_by_hand(plant, sol, asm1)
    assert pytest.approx(srt, rel=1e-9) == m.SRT
    assert m.solids_inventory == pytest.approx(inventory * 1e-3, rel=1e-9)
    assert m.solids_wasted == pytest.approx(wasted * 1e-3, rel=1e-9)
    assert m.solids_effluent == pytest.approx(effluent * 1e-3, rel=1e-9)
    assert m.reactor_units == _REACTORS
    V = sum(float(plant.units[n].volume) for n in m.reactor_units)
    # The ideal clarifier holds no solids, so the inventory is the reactors'.
    assert m.mlss == pytest.approx(inventory / V, rel=1e-9)
    # HRT = total reactor volume / influent flow (recycles excluded).
    assert pytest.approx(V / BSM1_Q_AVG, rel=1e-9) == m.HRT
    # Plausible BSM1 ranges.
    assert 2.0 < m.SRT < 30.0
    assert 0.0 < m.FM < 1.0
    assert m.solids_wasted > 0.0 and m.solids_effluent > 0.0
    assert "SRT" in m.summary()


@pytest.mark.parametrize("bad", ["BOD5", "bod ", "tss", ""])
def test_sludge_metrics_rejects_unknown_substrate(bad):
    """An unrecognized ``substrate`` must raise, not silently fall back to COD
    (which would report an F:M load ~2x off). Validation precedes any use of
    ``plant``/``solution``, so no solve is needed."""
    with pytest.raises(ValueError, match="substrate must be one of"):
        sludge_metrics(None, None, substrate=bad)


def test_plant_sludge_age_matches_sludge_metrics(ideal_warm):
    plant, sol = ideal_warm
    a = plant.sludge_age(sol)
    b = sludge_metrics(plant, sol)
    assert pytest.approx(b.SRT) == a.SRT
    assert pytest.approx(b.HRT) == a.HRT
    assert pytest.approx(b.FM) == a.FM


@pytest.mark.slow
def test_srt_decreases_with_wastage(asm1):
    """Wasting more sludge gives a younger sludge (smaller SRT) once the plant
    has responded -- the monotonic relation the target-SRT iteration relies on.
    Each SRT is also its solution's hand-computed inventory over losses."""
    srt = {}
    for Qw in (250.0, 600.0):
        plant, sol = _solve(asm1, Qw=Qw)
        srt[Qw] = sludge_metrics(plant, sol).SRT
        assert srt[Qw] == pytest.approx(_srt_by_hand(plant, sol, asm1)[0], rel=1e-9)
    assert srt[250.0] > srt[600.0]


def test_takacs_clarifier_blanket_counts_in_srt(asm1):
    """The Takács settler holds a sludge blanket and sludge_metrics counts it:
    the SRT is (reactor solids + blanket, summed by hand over the settler's
    layers) over the losses, and the blanket makes it materially larger than
    the reactor-only SRT the same losses give."""
    plant = _bsm1(asm1, use_takacs=True)
    sol = _warm_solution(plant)
    m = sludge_metrics(plant, sol)
    srt, inventory, wasted, effluent = _srt_by_hand(plant, sol, asm1)
    assert pytest.approx(srt, rel=1e-9) == m.SRT
    assert m.solids_inventory == pytest.approx(inventory * 1e-3, rel=1e-9)
    reactor_only = m.mlss * m.reactor_volume / (wasted + effluent)
    assert 1.05 * reactor_only < m.SRT


def test_a2o_srt_weighs_the_solids_by_their_itss_factors():
    """On the ASM2d A²O plant every solids term of the SRT is TSS through the
    model's iTSS factors -- the reactors, the per-species Takács blanket, the
    overflow and the wastage -- so the reactor MLSS is the mixed liquor's TSS,
    not 0.75·(XI + XS), the only ASM1-named particulates asm2d has. The F:M
    load is ASM2d's own: its COD counts every COD species (not only the
    ASM1-named SI, XI and XS), and its default BOD weighs them by their
    biodegradable share (it raised ``KeyError: 'SS'``)."""
    from aquakin.plant import a2o_influent, a2o_warm_start, build_a2o
    from aquakin.plant.results import PlantSolution

    asm2d = aquakin.load_model("asm2d")
    p = dict(zip(asm2d.parameters, map(float, asm2d.default_parameters()), strict=True))
    weights = {s: p[f] for s, f in TSS_FACTOR_PARAMS["asm2d"][1].items()}
    weights |= {"XMeOH": 1.0, "XMeP": 1.0}
    w = np.array([weights.get(s, 0.0) for s in asm2d.species])

    plant = build_a2o(asm2d)
    plant.add_influent("feed", a2o_influent(asm2d))
    y0 = jnp.asarray(a2o_warm_start(plant))
    reactors = plant.activated_sludge_reactors()
    by_unit = plant.states_by_unit(y0)
    y1 = plant.initial_state(overrides={n: 1.2 * by_unit[n] for n in reactors})
    sol = PlantSolution(t=jnp.asarray([0.0, 1.0]), state=jnp.stack([y0, y1]), plant=plant)
    m = sludge_metrics(plant, sol, substrate="COD")

    t = np.asarray(sol.t)

    def mean(x):
        return float(np.trapezoid(x, t) / (t[-1] - t[0]))

    reactor = sum(
        float(plant.units[n].volume) * (np.asarray(sol.unit_state(n)) @ w) for n in reactors
    )
    clar = plant.units["clarifier"]
    n_part = len(clar.particulate_species)
    layers = np.asarray(sol.unit_state("clarifier"))[:, : clar.n_layers * n_part]
    layers = layers.reshape(len(t), clar.n_layers, n_part)
    w_part = np.array([weights.get(s, 0.0) for s in clar.particulate_species])
    blanket = (layers @ w_part).sum(axis=1) * clar.area * clar.height / clar.n_layers
    eff = plant.stream(sol, "clarifier.overflow")
    waste = plant.stream(sol, "underflow_split.waste")
    loss = mean(np.asarray(eff.Q) * (np.asarray(eff.C) @ w))
    loss += mean(np.asarray(waste.Q) * (np.asarray(waste.C) @ w))
    V = sum(float(plant.units[n].volume) for n in reactors)

    assert m.mlss == pytest.approx(mean(reactor) / V, rel=1e-9)
    assert m.solids_inventory == pytest.approx((mean(reactor) + mean(blanket)) * 1e-3, rel=1e-9)
    assert pytest.approx((mean(reactor) + mean(blanket)) / loss, rel=1e-9) == m.SRT

    # F:M = substrate load over the reactor solids, by hand from the influent.
    feed = [plant.influents["feed"].at(ti) for ti in t]
    Q = np.array([float(f.Q) for f in feed])
    C = {s: np.array([float(f.C[asm2d.species_index[s]]) for f in feed]) for s in asm2d.species}
    cod = sum(C[s] for s in ("SF", "SA", "SI", "XI", "XS", "XH", "XPAO", "XPHA", "XAUT"))
    bod = 0.25 * (
        C["SF"] + C["SA"] + C["XPHA"] + (1.0 - p["fSI"]) * C["XS"]
        + (1.0 - p["fXI"]) * (C["XH"] + C["XPAO"] + C["XAUT"])
    )
    assert pytest.approx(mean(Q * cod) / mean(reactor), rel=1e-9) == m.FM
    by_bod = sludge_metrics(plant, sol)
    assert pytest.approx(mean(Q * bod) / mean(reactor), rel=1e-9) == by_bod.FM
    assert by_bod.SRT == m.SRT


@pytest.mark.slow
def test_bsm2_srt_counts_activated_sludge_wastage_not_dewatering_cake():
    """On BSM2 the default SRT uses the AS wastage, not the dewatering cake
    (primary + digested solids), which understated the sludge age by ~16%."""
    from aquakin.plant.bsm import bsm2_warm_start, build_bsm2
    from aquakin.plant.bsm.bsm2 import bsm2_constant_influent, bsm2_parameters
    from aquakin.plant.results import PlantSolution

    asm1 = aquakin.load_model("asm1")
    adm1 = aquakin.load_model("adm1")
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    params = bsm2_parameters(asm1, adm1)
    ss = plant.steady_state(params, y0=bsm2_warm_start(plant))
    assert bool(ss.converged)
    sol = PlantSolution(t=jnp.asarray([0.0, 1.0]), state=jnp.stack([ss.state, ss.state]),
                        plant=plant)
    m = sludge_metrics(plant, sol, params)
    explicit = sludge_metrics(plant, sol, params, effluent_port="settler.overflow",
                              waste_port="underflow_split.waste")
    assert pytest.approx(explicit.SRT) == m.SRT
    assert pytest.approx(20.9, abs=0.3) == m.SRT


def test_sludge_metrics_ignores_a_time_unit_override(asm1):
    """``sludge_metrics`` samples the influent at native time, so a solution
    reported in hours gives the same SRT, HRT and F:M as the same solution in
    days. The influent ramps (Q doubles over two days), so sampling it at the
    hour values would change HRT and F:M. No solve: a two-point solution at the
    warm-start state."""
    from aquakin.plant.plant import PlantSolution

    ramp = InfluentSeries(
        t=jnp.array([0.0, 2.0]),
        Q=jnp.array([BSM1_Q_AVG, 2.0 * BSM1_Q_AVG]),
        C=_influent(asm1).C,
        model=asm1,
    )
    plant = build_bsm1(model=asm1)
    plant.add_influent("feed", ramp, to="inlet_mix.fresh")
    y0 = bsm1_warm_start(plant)
    states = jnp.stack([y0, y0])
    days = PlantSolution(t=jnp.asarray([0.0, 1.0]), state=states, plant=plant)
    hours = PlantSolution(
        t=jnp.asarray([0.0, 24.0]), state=states, plant=plant, _requested_time_unit="h"
    )

    a, b = sludge_metrics(plant, days), sludge_metrics(plant, hours)
    assert np.isfinite(a.SRT) and a.SRT > 0.0
    for name in ("SRT", "HRT", "FM"):
        assert getattr(b, name) == pytest.approx(getattr(a, name), rel=1e-12), name
