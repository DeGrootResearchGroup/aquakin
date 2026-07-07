"""Activated-sludge design layer: size from targets, report achieved metrics.

Plants are specified in the quantities a process model *integrates* -- tank
``volume``, fixed pump flows, per-species ``kLa``. But an engineer designs in
the quantities those are derived *from*: the solids retention time (SRT, sludge
age), the hydraulic retention time (HRT) and the food-to-microorganism ratio
(F:M). This module bridges the two directions:

- :func:`size_activated_sludge` -- **forward** design. Given an SRT/HRT target
  and the design flow, return the aeration volume and the wastage flow ``Qw``
  that realise them, plus the recycle pump flows.
- :func:`sludge_metrics` (also reachable as ``plant.sludge_age(solution)``) --
  the **closing** half. Given a *solved* plant, report the SRT/HRT/F:M the model
  actually achieved, by reconstructing the system solids inventory and the
  solids leaving via wastage and effluent. Because SRT is an emergent property
  of ``Qw``, this is what lets the engineer iterate ``Qw`` to a target SRT (see
  ``examples/bsm1_target_srt.py``).

The metrics are model-agnostic in mechanism but use the ASM1 TSS / BOD
aggregates (:mod:`aquakin.plant.metrics`), so they apply to the ASM activated-
sludge models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

from aquakin.plant._constants import EPS_Q
from aquakin.plant.metrics import (
    _time_average as _metrics_time_average,
)
from aquakin.plant.metrics import (
    derived_BOD,
    derived_COD,
    derived_TSS,
)

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.plant.plant import Plant, PlantSolution


# ---------------------------------------------------------------------------
# Forward sizing
# ---------------------------------------------------------------------------


@dataclass
class ActivatedSludgeSizing:
    """The reactor volume and flows that realise an SRT/HRT design target.

    Attributes
    ----------
    SRT : float
        Target solids retention time (sludge age), days.
    HRT : float
        Hydraulic retention time, days.
    Q : float
        Design influent flow, m³/d.
    volume : float
        Total aeration volume, m³ (= ``Q × HRT``).
    wastage_flow : float
        Sludge-wastage flow ``Qw`` (m³/d) that hits the target SRT.
    tank_volumes : tuple of float
        Per-tank volumes when the basin is split into a cascade (else a
        single-element tuple equal to ``volume``).
    internal_recycle_flow : float or None
        Internal (mixed-liquor) recycle flow, if an ``internal_recycle_ratio``
        was supplied (= ratio × Q).
    ras_flow : float or None
        Return-activated-sludge flow, if a ``ras_ratio`` was supplied.
    wastage_from : str
        Where the wastage is drawn -- ``"mixed_liquor"`` (hydraulic SRT control,
        ``Qw = V/SRT``, concentration-independent) or ``"underflow"``
        (``Qw = V/(SRT × thickening_ratio)``).
    thickening_ratio : float
        Underflow-to-reactor TSS ratio used for underflow wasting (1 for
        mixed-liquor wasting).
    """

    SRT: float
    HRT: float
    Q: float
    volume: float
    wastage_flow: float
    tank_volumes: tuple = ()
    internal_recycle_flow: Optional[float] = None
    ras_flow: Optional[float] = None
    wastage_from: str = "mixed_liquor"
    thickening_ratio: float = 1.0

    def summary(self) -> str:
        """A human-readable one-block summary of the sizing."""
        lines = [
            "Activated-sludge sizing:",
            f"  design flow Q     = {self.Q:10.1f}  m3/d",
            f"  target SRT        = {self.SRT:10.2f}  d",
            f"  HRT               = {self.HRT * 24.0:10.2f}  h  ({self.HRT:.3f} d)",
            f"  aeration volume   = {self.volume:10.1f}  m3",
        ]
        if len(self.tank_volumes) > 1:
            vols = ", ".join(f"{v:.0f}" for v in self.tank_volumes)
            lines.append(f"  tank volumes      = [{vols}]  m3")
        lines.append(
            f"  wastage Qw        = {self.wastage_flow:10.2f}  m3/d (from {self.wastage_from})"
        )
        if self.internal_recycle_flow is not None:
            lines.append(f"  internal recycle  = {self.internal_recycle_flow:10.1f}  m3/d")
        if self.ras_flow is not None:
            lines.append(f"  RAS flow          = {self.ras_flow:10.1f}  m3/d")
        return "\n".join(lines)


def size_activated_sludge(
    *,
    SRT: float,
    Q: float,
    HRT: Optional[float] = None,
    HRT_h: Optional[float] = None,
    n_tanks: int = 1,
    volume_fractions: Optional[list] = None,
    wastage_from: str = "mixed_liquor",
    thickening_ratio: float = 1.0,
    internal_recycle_ratio: Optional[float] = None,
    ras_ratio: Optional[float] = None,
) -> ActivatedSludgeSizing:
    """Size an activated-sludge basin from SRT / HRT design targets.

    The two standard sizing relations:

    - **Volume from HRT:** ``V = Q × HRT`` -- the aeration volume is the design
      flow times the chosen hydraulic retention time.
    - **Wastage from SRT:** the sludge age is the system solids mass over the
      rate solids are wasted. Two wasting strategies:

      * ``wastage_from="mixed_liquor"`` (default, "hydraulic"/Garrett SRT
        control): wasting mixed liquor straight from the aeration basin gives
        ``SRT = V·X / (Qw·X) = V/Qw`` independent of the (unknown, time-varying)
        solids concentration, so ``Qw = V/SRT`` exactly.
      * ``wastage_from="underflow"``: wasting from the thickened RAS line gives
        ``SRT = V·X / (Qw·X_r)``, so ``Qw = V / (SRT × thickening_ratio)`` with
        ``thickening_ratio = X_r/X`` (the underflow-to-reactor TSS ratio, > 1).

    This is the *nominal* design; the realised SRT also depends on the effluent
    solids loss and (for underflow wasting) the actual thickening, so confirm it
    on a solved plant with :func:`sludge_metrics`.

    Parameters
    ----------
    SRT : float
        Target solids retention time (days). Must be > 0.
    Q : float
        Design (average) influent flow (m³/d). Must be > 0.
    HRT : float, optional
        Hydraulic retention time in **days**. Supply exactly one of ``HRT`` /
        ``HRT_h``.
    HRT_h : float, optional
        Hydraulic retention time in **hours** (the usual engineering unit).
    n_tanks : int, optional
        Split the aeration volume into this many equal tanks (a CSTR cascade).
        Default 1. Ignored when ``volume_fractions`` is given.
    volume_fractions : list of float, optional
        Explicit per-tank volume fractions (must be positive and sum to 1); sets
        ``tank_volumes`` and overrides ``n_tanks``.
    wastage_from : str, optional
        ``"mixed_liquor"`` (default) or ``"underflow"`` -- see above.
    thickening_ratio : float, optional
        Underflow-to-reactor TSS ratio for ``wastage_from="underflow"`` (must be
        > 0; ignored for mixed-liquor wasting). Default 1.
    internal_recycle_ratio, ras_ratio : float, optional
        If given, also report the internal-recycle / RAS pump flows as
        ``ratio × Q``.

    Returns
    -------
    ActivatedSludgeSizing
        Volume, wastage flow, tank split and recycle flows.

    Raises
    ------
    ValueError
        On non-positive SRT/Q/HRT, an ambiguous or missing HRT, a bad
        ``wastage_from``, a non-positive ``thickening_ratio``, or
        ``volume_fractions`` that are non-positive / wrong length / do not sum
        to 1.

    Examples
    --------
    >>> s = size_activated_sludge(SRT=10.0, HRT_h=8.0, Q=18446.0, n_tanks=5)
    >>> round(s.volume), round(s.wastage_flow)
    (6149, 615)
    """
    if SRT <= 0:
        raise ValueError(f"SRT must be > 0; got {SRT}")
    if Q <= 0:
        raise ValueError(f"Q must be > 0; got {Q}")
    if (HRT is None) == (HRT_h is None):
        raise ValueError("Supply exactly one of HRT (days) or HRT_h (hours).")
    HRT_days = float(HRT) if HRT is not None else float(HRT_h) / 24.0
    if HRT_days <= 0:
        raise ValueError(f"HRT must be > 0; got {HRT_days} d")
    if wastage_from not in ("mixed_liquor", "underflow"):
        raise ValueError(
            f"wastage_from must be 'mixed_liquor' or 'underflow'; got {wastage_from!r}"
        )
    if thickening_ratio <= 0:
        raise ValueError(f"thickening_ratio must be > 0; got {thickening_ratio}")

    volume = float(Q) * HRT_days

    # Tank split.
    if volume_fractions is not None:
        fracs = [float(f) for f in volume_fractions]
        if any(f <= 0 for f in fracs):
            raise ValueError("volume_fractions must all be > 0.")
        if abs(sum(fracs) - 1.0) > 1e-6:
            raise ValueError(f"volume_fractions must sum to 1; got {sum(fracs)}")
        tank_volumes = tuple(volume * f for f in fracs)
    elif n_tanks > 1:
        tank_volumes = tuple(volume / n_tanks for _ in range(n_tanks))
    else:
        tank_volumes = (volume,)

    # Wastage to hit the SRT.
    ratio = 1.0 if wastage_from == "mixed_liquor" else float(thickening_ratio)
    wastage_flow = volume / (float(SRT) * ratio)

    return ActivatedSludgeSizing(
        SRT=float(SRT),
        HRT=HRT_days,
        Q=float(Q),
        volume=volume,
        wastage_flow=wastage_flow,
        tank_volumes=tank_volumes,
        internal_recycle_flow=(
            None if internal_recycle_ratio is None else float(internal_recycle_ratio) * float(Q)
        ),
        ras_flow=(None if ras_ratio is None else float(ras_ratio) * float(Q)),
        wastage_from=wastage_from,
        thickening_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Achieved metrics from a solved plant
# ---------------------------------------------------------------------------


@dataclass
class SludgeMetrics:
    """Activated-sludge operating metrics achieved by a solved plant.

    All quantities are time-averaged over the solution window (so for a
    steady-state run they are the steady values).

    Attributes
    ----------
    SRT : float
        Solids retention time / sludge age (days) = system solids inventory /
        rate of solids leaving (wastage + effluent).
    HRT : float
        Hydraulic retention time (days) = total reactor volume / influent flow
        (the recycle flows do not count toward HRT).
    FM : float
        Food-to-microorganism ratio (g BOD / g TSS / d) = influent BOD load /
        reactor solids mass (reported on a TSS basis).
    mlss : float
        Mixed-liquor suspended solids -- the mean reactor TSS (g/m³).
    reactor_volume : float
        Total aeration volume counted (m³).
    solids_inventory : float
        System solids mass (kg TSS) -- reactors plus any secondary-clarifier
        sludge blanket.
    solids_wasted : float
        Solids leaving via the wastage stream (kg TSS/d).
    solids_effluent : float
        Solids leaving via the effluent stream (kg TSS/d).
    influent_flow : float
        Influent flow used for HRT / F:M (m³/d).
    influent_bod_load : float
        Influent BOD load (kg BOD/d).
    reactor_units : list of str
        The reactor units whose volume and solids were counted.
    notes : str
        Notes on the computation.
    """

    SRT: float
    HRT: float
    FM: float
    mlss: float
    reactor_volume: float
    solids_inventory: float
    solids_wasted: float
    solids_effluent: float
    influent_flow: float
    influent_bod_load: float
    reactor_units: list = field(default_factory=list)
    notes: str = (
        "Time-averaged over the solution window. SRT counts reactor + secondary-"
        "clarifier solids over wastage + effluent loss; HRT = reactor volume / "
        "influent flow; F:M = influent BOD load / reactor TSS mass."
    )

    def summary(self) -> str:
        """A human-readable one-block summary of the achieved metrics."""
        return "\n".join(
            [
                "Achieved activated-sludge metrics:",
                f"  SRT (sludge age) = {self.SRT:10.2f}  d",
                f"  HRT              = {self.HRT * 24.0:10.2f}  h  ({self.HRT:.3f} d)",
                f"  F:M              = {self.FM:10.3f}  g BOD / g TSS / d",
                f"  MLSS             = {self.mlss:10.1f}  g/m3",
                f"  solids inventory = {self.solids_inventory:10.1f}  kg TSS",
                f"  solids wasted    = {self.solids_wasted:10.1f}  kg TSS/d",
                f"  solids effluent  = {self.solids_effluent:10.1f}  kg TSS/d",
                f"  reactors counted = {self.reactor_units}",
            ]
        )


# Effluent / wastage endpoint candidates, in preference order (BSM2 first).
_EFFLUENT_CANDIDATES = ("effluent_mix.out", "settler.overflow", "clarifier.overflow")
_WASTE_CANDIDATES = ("dewatering.underflow", "underflow_split.waste")
_VALID_SUBSTRATES = frozenset({"BOD", "COD"})


def _available_endpoints(plant) -> set:
    """The ``"unit.port"`` strings the plant can produce."""
    return {f"{name}.{port}" for name, unit in plant.units.items() for port in unit.output_ports}


def _pick_endpoint(plant, explicit, candidates, role):
    """Resolve a stream endpoint: an explicit one, else the first candidate the
    plant exposes."""
    available = _available_endpoints(plant)
    if explicit is not None:
        if explicit not in available:
            raise ValueError(
                f"{role} port {explicit!r} is not an output of this plant. "
                f"Available: {sorted(available)}"
            )
        return explicit
    for cand in candidates:
        if cand in available:
            return cand
    raise ValueError(
        f"Could not auto-detect the {role} port; pass it explicitly. "
        f"Tried {candidates}; available: {sorted(available)}"
    )


def _reactor_units(plant, explicit):
    """Auto-detect the activated-sludge reactor CSTRs (volume + aeration), or
    validate an explicit list. The digester and other volumed units have no
    ``aeration`` field and are excluded."""
    if explicit is not None:
        for name in explicit:
            if name not in plant.units:
                raise ValueError(f"Unknown reactor unit {name!r}.")
        return list(explicit)
    reactors = plant.activated_sludge_reactors(require_volume=True)
    if not reactors:
        raise ValueError(
            "Could not auto-detect activated-sludge reactors; pass reactor_units=[...] explicitly."
        )
    return reactors


def _time_average(t, values):
    """Trapezoidal time-average of ``values(t)`` over ``[t0, t1]`` (single source
    of truth: the shared :func:`aquakin.plant.metrics._time_average` kernel,
    which also returns the single sample for a one-point steady-state window).
    Wrapped here only to keep the local ``(t, values)`` argument order and the
    ``float`` return."""
    return float(_metrics_time_average(values, t))


def _pick_influent(plant, influent_name):
    """Resolve the external influent series for HRT / F:M."""
    influents = plant.influents
    if influent_name is not None:
        if influent_name not in influents:
            raise ValueError(f"Unknown influent {influent_name!r}; have {list(influents)}.")
        return influents[influent_name]
    if len(influents) == 1:
        return next(iter(influents.values()))
    if "feed" in influents:
        return influents["feed"]
    raise ValueError(
        f"Multiple influents {list(influents)}; pass influent_name= to pick the main feed."
    )


def sludge_metrics(
    plant: "Plant",
    solution: "PlantSolution",
    params: Optional[jnp.ndarray] = None,
    *,
    reactor_units: Optional[list] = None,
    influent_name: Optional[str] = None,
    effluent_port: Optional[str] = None,
    waste_port: Optional[str] = None,
    substrate: str = "BOD",
) -> SludgeMetrics:
    """Achieved SRT / HRT / F:M from a solved activated-sludge plant.

    Closes the design loop: SRT is an emergent property of the wastage flow, so
    rather than guessing ``Qw`` this reports the sludge age the model actually
    produced. Time-averaged over the solution window.

    Parameters
    ----------
    plant : Plant
        The solved plant (e.g. from :func:`aquakin.plant.bsm.build_bsm1`).
    solution : PlantSolution
        A solution from ``plant.solve``. For a representative steady SRT, solve
        to (near) steady state and save a few late points.
    params : jnp.ndarray, optional
        Plant parameters used for the run (defaults to the plant defaults).
    reactor_units : list of str, optional
        Aeration reactors to count. Defaults to the auto-detected ASM CSTRs.
    influent_name : str, optional
        Which external influent is the main feed (for HRT / F:M). Defaults to
        the sole influent, or the one named ``"feed"``.
    effluent_port, waste_port : str, optional
        ``"unit.port"`` of the final effluent and the wastage stream. Auto-
        detected for BSM1/BSM2 when omitted.
    substrate : str, optional
        Substrate measure for the F:M load -- ``"BOD"`` (default) or ``"COD"``.

    Returns
    -------
    SludgeMetrics
        SRT, HRT, F:M and the intermediate inventories / loads.

    Examples
    --------
    >>> m = aquakin.plant.design.sludge_metrics(plant, solution)  # doctest: +SKIP
    >>> print(m.summary())                                        # doctest: +SKIP
    """
    substrate_key = substrate.upper()
    if substrate_key not in _VALID_SUBSTRATES:
        raise ValueError(
            f"substrate must be one of {sorted(_VALID_SUBSTRATES)}; got {substrate!r}."
        )

    params_full = plant.default_parameters() if params is None else jnp.asarray(params)
    reactors = _reactor_units(plant, reactor_units)
    model = plant.units[reactors[0]].model
    t = solution.t

    # ----- System solids inventory (g): reactors + secondary clarifier. -----
    reactor_volume = sum(float(plant.units[n].volume) for n in reactors)
    reactor_solids = jnp.zeros_like(t)  # (n_t,) g
    for name in reactors:
        X = solution.unit_state(name)  # (n_t, n_species)
        reactor_solids = reactor_solids + derived_TSS(X, model) * float(plant.units[name].volume)

    clarifier_solids = jnp.zeros_like(t)
    for name, unit in plant.units.items():
        # Any stateful separator that can report its sludge blanket (the Takács
        # clarifier); the stateless IdealClarifier holds ~0 inventory.
        if hasattr(unit, "solids_mass") and unit.state_size > 0:
            states = solution.unit_state(name)  # (n_t, state_size)
            clarifier_solids = clarifier_solids + jax.vmap(unit.solids_mass)(states)

    system_solids = reactor_solids + clarifier_solids  # (n_t,) g
    inventory_mean = _time_average(t, system_solids)  # g
    reactor_solids_mean = _time_average(t, reactor_solids)  # g

    # ----- Solids leaving via wastage + effluent (g/d). -----
    eff_port = _pick_endpoint(
        plant,
        effluent_port or getattr(plant, "effluent_endpoint", None),
        _EFFLUENT_CANDIDATES,
        "effluent",
    )
    w_port = _pick_endpoint(plant, waste_port, _WASTE_CANDIDATES, "wastage")
    eff = plant.stream(solution, eff_port, params_full)
    waste = plant.stream(solution, w_port, params_full)
    eff_solids_rate = eff.Q * derived_TSS(eff.C, model)  # (n_t,) g/d
    waste_solids_rate = waste.Q * derived_TSS(waste.C, model)
    loss_mean = _time_average(t, eff_solids_rate + waste_solids_rate)  # g/d

    SRT = inventory_mean / (loss_mean + EPS_Q)  # days

    # ----- HRT and F:M from the external influent. -----
    influent = _pick_influent(plant, influent_name)
    inf_streams = [influent.at(ti) for ti in t]
    inf_Q = jnp.asarray([s.Q for s in inf_streams])  # (n_t,)
    inf_C = jnp.stack([s.C for s in inf_streams])  # (n_t, n_species)
    Q_mean = _time_average(t, inf_Q)
    HRT = reactor_volume / (Q_mean + EPS_Q)  # days

    load_fn = derived_BOD if substrate_key == "BOD" else derived_COD
    bod_load_rate = inf_Q * load_fn(inf_C, model)  # (n_t,) g/d
    bod_load_mean = _time_average(t, bod_load_rate)  # g/d
    # F:M is the substrate load over the reactor (aeration-basin) solids mass.
    FM = bod_load_mean / (reactor_solids_mean + EPS_Q)  # 1/d
    mlss = reactor_solids_mean / (reactor_volume + EPS_Q)  # g/m3

    return SludgeMetrics(
        SRT=SRT,
        HRT=HRT,
        FM=FM,
        mlss=mlss,
        reactor_volume=reactor_volume,
        solids_inventory=inventory_mean * 1e-3,  # kg
        solids_wasted=_time_average(t, waste_solids_rate) * 1e-3,
        solids_effluent=_time_average(t, eff_solids_rate) * 1e-3,
        influent_flow=Q_mean,
        influent_bod_load=bod_load_mean * 1e-3,
        reactor_units=reactors,
    )
