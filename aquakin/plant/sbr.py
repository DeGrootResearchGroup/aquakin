"""Sequencing batch reactor (SBR): one tank cycling through timed phases.

An SBR treats wastewater in batches in a single tank, cycling fill -> react ->
settle -> decant -> idle. Volume rises during fill and falls during decant; the
biology reacts throughout; aeration is switched per phase (aerobic react vs
anoxic/settle); and the settle phase clarifies the supernatant that the decant
draws off as the treated effluent. SBRs dominate small / decentralized plants,
where the whole works is often a single SBR.

:class:`SBRUnit` is a plant unit with variable-volume state ``[C, V]`` plus the
internal state of a pluggable :class:`~aquakin.plant.settling.SettlingModel`. Its
phases are scheduled in time; the unit reports its phase-transition times as
located :class:`~aquakin.Event` boundaries (``cycle_events``), which the plant
collects so the integrator lands exactly on each switch -- so the discontinuous
flow/aeration changes are resolved, not stepped across. Within a phase the
behaviour is a smooth, differentiable ODE.

How the tank takes its feed is the required ``feed`` mode, since a feed the
tank does not take would leave the plant unaccounted for:

- ``"continuous"`` (a continuous-inflow SBR, ICEAS type): the tank takes the
  inlet flow whenever it arrives, in every phase, and its volume follows the
  inflow minus the decant.
- ``"storage"`` (an influent holding tank ahead of the SBR, held inside the
  unit): a holding volume, with its own volume and concentration states,
  receives the inlet flow continuously, and the reactor draws ``feed_flow``
  from it during fill phases. What the holding volume cannot hold leaves
  through the ``overflow`` port, and a draw it cannot supply (running empty) is
  cut back to the inflow; :func:`storage_report` reports both.
- ``"batch"`` (a reservoir-fed SBR): the inlet must deliver ``feed_flow`` during
  fill phases and nothing in any other phase. Each solve checks this against
  the influent feeding the unit and refuses one that does not; the tank then
  takes the inlet flow as it arrives, so even the tolerance of that check
  cannot lose mass.

The effluent leaves at ``decant_flow`` during the decant phase, at the clarified
concentration the settling model reports, and is zero in every other phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate.events import Event
from aquakin.plant._constants import check_positive
from aquakin.plant.balance import unit_terms
from aquakin.plant.coupling import CouplingAware
from aquakin.plant.errors import WiringError
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.kinetic import AdjustableTemperatureUnit
from aquakin.plant.settling import SettlingModel
from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel

_EPS_V = 1e-9  # guard 1/V when the tank is near-empty

#: The feed modes, with what each models (the ``feed`` setting has no default).
FEED_MODES = {
    "continuous": "the tank takes the inlet flow whenever it arrives (ICEAS type)",
    "storage": (
        "a holding volume inside the unit receives the inlet flow and the tank "
        "draws feed_flow from it during fill phases"
    ),
    "batch": "the inlet delivers feed_flow during fill phases and nothing otherwise",
}

#: The holding volume's level band, as a fraction of ``storage_volume``, over
#: which the draw pump cuts out (from two bands down to one) and the overflow
#: takes over (from one band below full up to full). Ramping both linearly keeps
#: the right-hand side continuous, so the solve and its gradient need no event
#: at either level; one band is the pump's dead volume.
STORAGE_BAND = 0.01

#: A batch-fed SBR's inlet flow must equal ``feed_flow`` in a fill phase and zero
#: otherwise to within this fraction of ``feed_flow``.
BATCH_FLOW_RTOL = 1e-6

#: The inset from each phase boundary, as a fraction of the phase, of the first
#: and last times at which the batch check reads the inlet flow (a step in a
#: sampled influent needs two samples, so it ramps over their spacing).
_BATCH_CHECK_INSET = 1e-6

_DEFAULT_STORAGE_FRACTION = 0.5


def _mode_list() -> str:
    return "; ".join(f"'{m}' ({what})" for m, what in FEED_MODES.items())


@dataclass(frozen=True)
class SBRPhase:
    """One phase of an SBR cycle.

    Parameters
    ----------
    name : str
        Phase label (e.g. ``"fill"``, ``"react"``, ``"settle"``, ``"decant"``,
        ``"idle"``), used in the event log.
    duration : float
        Phase length (d), on the plant's clock.
    feed : bool
        A fill phase: the tank draws ``feed_flow`` from its holding volume
        (``feed="storage"``), or the inlet delivers it (``feed="batch"``). A
        continuously fed tank takes its inflow in every phase.
    decant : bool
        Draw clarified effluent out at ``decant_flow`` during this phase (decant).
    kla : float
        Oxygen mass-transfer coefficient (1/d) applied during this phase (aerobic
        react). 0 (default) = no aeration (anoxic / settle / fill).
    settle : bool
        Let the contents settle this phase (the settling model's clarity grows).
    mixed : bool, optional
        Whether the tank is actively mixed this phase (re-suspending settled
        solids, so the settling model's clarity relaxes back toward fully mixed).
        Default ``None`` derives it as ``feed or kla > 0`` -- a fed or aerated
        phase is mixed; a quiescent decant/idle phase is not, so its clarity is
        held rather than washed out. A settle phase is never mixed (settling takes
        precedence). Set explicitly to mark an unaerated-but-mechanically-mixed
        phase (e.g. an anoxic mixed react).
    """

    name: str
    duration: float
    feed: bool = False
    decant: bool = False
    kla: float = 0.0
    settle: bool = False
    mixed: bool | None = None


@dataclass
class SBRUnit(AdjustableTemperatureUnit, CouplingAware):
    """A sequencing batch reactor: variable-volume, time-phased single tank.

    Parameters
    ----------
    name : str
    model : CompiledModel
    phases : sequence of SBRPhase
        One cycle, in order. The cycle repeats with period ``sum(durations)``.
    full_volume : float
        Maximum (full) liquid volume (m³).
    feed_flow : float or None
        Fill-pump flow (m³/d) during ``feed`` phases: drawn from the holding
        volume with ``feed="storage"``, and the flow the inlet must deliver with
        ``feed="batch"``. Leave it unset (``None``) with ``feed="continuous"``,
        where the tank takes the inlet flow as it comes.
    decant_flow : float
        Effluent flow (m³/d) drawn during ``decant`` phases.
    settling : SettlingModel
        Pluggable settle/decant clarification strategy.
    particulate_species : sequence of str
        Species that settle (e.g. the ASM ``X*`` solids); the settling model
        clarifies these in the decant draw.
    initial_fraction : float
        Starting fill level ``V0 / full_volume`` (the heel left after the
        previous decant). Default 0.25.
    conditions : dict
        Per-model condition values (e.g. ``{"T": 293.15}``), as for a CSTR. The
        kinetics always run at this static ``T`` (change it with
        :meth:`set_temperature` or ``Plant.set_temperature``): unlike a CSTR the
        SBR does not follow its inlet temperature, since its feed may arrive
        during fill phases only, and a heat-balance temperature model does not
        track it, since its liquid volume varies over the cycle.
    do_sat : float
        Oxygen saturation (g/m³) in the aeration term ``kLa*(do_sat - C_O2)``.
    oxygen_species : str
        The aerated species (default ``"SO"``); ignored if absent from the model.
    initial_concentrations : jnp.ndarray, optional
        Initial bulk concentration vector (defaults to the model defaults).
    cycle_origin : float
        Time (d) at which phase 0 of a cycle begins (default 0).
    input_port : str
        Name of the feed port (default ``"feed"``).
    output_port : str
        Name of the effluent port (default ``"effluent"``).
    feed : {"continuous", "storage", "batch"}
        Required: how the tank takes the flow arriving on its feed port.
        ``"continuous"``: whenever it arrives, in every phase (a
        continuous-inflow SBR); the volume follows the inflow minus the decant.
        ``"storage"``: a holding volume inside the unit receives it and the tank
        draws ``feed_flow`` from it during fill phases; overflow leaves through
        ``overflow_port``. ``"batch"``: the inlet must deliver ``feed_flow``
        during fill phases and nothing otherwise, which each solve checks.
    storage_volume : float, optional
        Capacity of the holding volume (m³). Required with ``feed="storage"``,
        unset otherwise.
    storage_initial_fraction : float
        Starting level of the holding volume, as a fraction of
        ``storage_volume`` (default 0.5). With ``feed="storage"`` only.
    storage_initial_concentrations : jnp.ndarray, optional
        Initial concentration vector of the holding volume (defaults to the
        model defaults). With ``feed="storage"`` only.
    overflow_port : str
        Name of the port the holding volume overflows through (default
        ``"overflow"``); the unit has it only with ``feed="storage"``.
    """

    name: str
    model: CompiledModel
    phases: Sequence[SBRPhase]
    full_volume: float
    feed_flow: float | None
    decant_flow: float
    settling: SettlingModel
    particulate_species: Sequence[str] = ()
    initial_fraction: float = 0.25
    conditions: dict = field(default_factory=dict)
    do_sat: float = 8.0
    oxygen_species: str = "SO"
    initial_concentrations: jnp.ndarray | None = None
    cycle_origin: float = 0.0
    input_port: str = "feed"
    output_port: str = "effluent"
    feed: Literal["continuous", "storage", "batch"] | None = None
    storage_volume: float | None = None
    storage_initial_fraction: float = _DEFAULT_STORAGE_FRACTION
    storage_initial_concentrations: jnp.ndarray | None = None
    overflow_port: str = "overflow"

    def __post_init__(self) -> None:
        check_positive(self.name, "full_volume", self.full_volume)
        self._check_feed()
        if not self.phases:
            raise ValueError(f"SBRUnit '{self.name}' needs at least one phase.")
        if any(p.duration <= 0 for p in self.phases):
            raise ValueError(f"SBRUnit '{self.name}' phase durations must be > 0.")
        if not (0.0 < self.initial_fraction <= 1.0):
            raise ValueError(f"SBRUnit '{self.name}' initial_fraction must be in (0, 1].")
        self._init_conditions(self.conditions)
        for s in self.particulate_species:
            if s not in self.model.species_index:
                raise ValueError(
                    f"SBRUnit '{self.name}' particulate species '{s}' not in the model."
                )

        # Phase schedule (concrete floats; static).
        durations = [float(p.duration) for p in self.phases]
        self._period = float(sum(durations))
        cum = []
        acc = 0.0
        for d in durations:
            acc += d
            cum.append(acc)
        self._interior_breaks = jnp.asarray(cum[:-1])  # within-cycle boundaries
        self._phase_starts = [0.0] + cum[:-1]  # phase start offsets

        # Per-phase flag/value arrays, indexed by phase index.
        self._feed = jnp.asarray([1.0 if p.feed else 0.0 for p in self.phases])
        self._decant = jnp.asarray([1.0 if p.decant else 0.0 for p in self.phases])
        self._kla = jnp.asarray([float(p.kla) for p in self.phases])
        self._settle = jnp.asarray([1.0 if p.settle else 0.0 for p in self.phases])

        # Mixing flag: a settle phase is never mixed; otherwise use the explicit
        # `mixed` if given, else derive it as fed-or-aerated. Quiescent decant/idle
        # phases are thus not mixed, so the settling model holds clarity there.
        def _is_mixed(p: SBRPhase) -> bool:
            if p.settle:
                return False
            if p.mixed is not None:
                return bool(p.mixed)
            return bool(p.feed or p.kla > 0.0)

        self._mixed = jnp.asarray([1.0 if _is_mixed(p) else 0.0 for p in self.phases])

        # Oxygen-transfer mask (a single aerated species).
        n = self.model.n_species
        if self.oxygen_species in self.model.species_index:
            o2 = self.model.species_index[self.oxygen_species]
            self._o2_mask = jnp.zeros((n,)).at[o2].set(1.0)
        else:
            self._o2_mask = jnp.zeros((n,))

        # Bind the settling model's particulate mask.
        self.settling.bind(self.model, list(self.particulate_species))

    def _check_feed(self) -> None:
        """Refuse a missing or unknown ``feed`` mode, and settings it does not use."""
        if self.feed is None:
            raise ValueError(
                f"SBRUnit '{self.name}' needs a feed mode, feed=..., saying how the tank "
                f"takes the flow arriving on its feed port: {_mode_list()}."
            )
        if self.feed not in FEED_MODES:
            raise ValueError(
                f"SBRUnit '{self.name}' feed={self.feed!r} is not a feed mode; use one of "
                f"{_mode_list()}."
            )
        if self.feed == "continuous" and self.feed_flow is not None:
            raise ValueError(
                f"SBRUnit '{self.name}' takes the inlet flow as it arrives with "
                f"feed='continuous', so feed_flow does not apply; leave it unset "
                f"(feed_flow=None)."
            )
        if self.feed != "continuous":
            check_positive(self.name, "feed_flow", self.feed_flow)
        if self.feed == "storage":
            check_positive(self.name, "storage_volume", self.storage_volume)
            if not (0.0 < self.storage_initial_fraction <= 1.0):
                raise ValueError(
                    f"SBRUnit '{self.name}' storage_initial_fraction must be in (0, 1]."
                )
            return
        unset = {
            "storage_volume": self.storage_volume is None,
            "storage_initial_fraction": (
                self.storage_initial_fraction == _DEFAULT_STORAGE_FRACTION
            ),
            "storage_initial_concentrations": self.storage_initial_concentrations is None,
        }
        given = [name for name, is_unset in unset.items() if not is_unset]
        if given:
            raise ValueError(
                f"SBRUnit '{self.name}' has no holding volume with feed={self.feed!r}, so "
                f"{', '.join(given)} does not apply; leave it unset, or use feed='storage'."
            )

    @property
    def _storage(self) -> bool:
        return self.feed == "storage"

    # ----- protocol: identity / layout -----------------------------------
    @property
    def state_size(self) -> int:
        n = self.model.n_species
        storage = n + 1 if self._storage else 0
        return n + 1 + self.settling.extra_state_size() + storage

    def state_names(self) -> tuple[str, ...]:
        """Names of the state entries, in order: the species concentrations, the
        liquid volume ``V``, the settling model's states, then, with
        ``feed="storage"``, the holding volume's concentrations
        (``storage.<species>``) and volume (``storage.V``)."""
        storage = (
            (*(f"storage.{s}" for s in self.model.species), "storage.V") if self._storage else ()
        )
        return (*self.model.species, "V", *self.settling.extra_state_names(), *storage)

    def liquid_volume(self, state: jnp.ndarray):
        """The tank's current liquid volume (the settling-state tail carries no
        mass), whose concentrations head the state."""
        return self._split(state)[1]

    def component_inventory(self, state: jnp.ndarray, content: dict, params) -> dict:
        """The unit's component inventory: the tank's ``V*C``, plus the holding
        volume's with ``feed="storage"`` (the settling state carries no mass).

        Parameters
        ----------
        state : jnp.ndarray
            The unit's state.
        content : dict
            ``{component: (n_species,) content per unit concentration}``.
        params : jnp.ndarray
            The unit's parameter vector (unused).

        Returns
        -------
        dict
            ``{component: grams}``.
        """
        C, V, _ = self._split(state)
        held = [(float(V), np.asarray(C))]
        if self._storage:
            C_s, V_s = self._split_storage(state)
            held.append((float(V_s), np.asarray(C_s)))
        return {q: sum(v * float(np.dot(c, vec)) for v, c in held) for q, vec in content.items()}

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port, self.overflow_port] if self._storage else [self.output_port]

    def _initial(self, given) -> jnp.ndarray:
        return self.model.default_concentrations() if given is None else jnp.asarray(given)

    def initial_state(self) -> jnp.ndarray:
        V0 = jnp.asarray([self.initial_fraction * float(self.full_volume)])
        parts = [
            self._initial(self.initial_concentrations),
            V0,
            self.settling.initial_extra_state(),
        ]
        if self._storage:
            V_s0 = self.storage_initial_fraction * float(self.storage_volume)
            parts += [self._initial(self.storage_initial_concentrations), jnp.asarray([V_s0])]
        return jnp.concatenate(parts)

    # ----- phase helpers --------------------------------------------------
    def _phase_index(self, t: jnp.ndarray) -> jnp.ndarray:
        """Index of the active phase at time ``t`` (cyclic, jit/AD-safe)."""
        tau = jnp.mod(t - self.cycle_origin, self._period)
        return jnp.searchsorted(self._interior_breaks, tau, side="right")

    def _split(self, state: jnp.ndarray):
        """``(C, V, settling state)`` of the tank."""
        n = self.model.n_species
        return state[:n], state[n], state[n + 1 : n + 1 + self.settling.extra_state_size()]

    def _split_storage(self, state: jnp.ndarray):
        """``(C, V)`` of the holding volume: the state's tail (``feed="storage"``)."""
        n = self.model.n_species
        return state[-(n + 1) : -1], state[-1]

    def storage_flows(self, t, state: jnp.ndarray, inflow) -> dict:
        """The holding volume's flows at ``t`` (``feed="storage"``).

        The draw is ``feed_flow`` in a fill phase while the level is above two
        :data:`STORAGE_BAND` of ``storage_volume``; below that it ramps down to
        what arrives (at most the demand), reached at one band, so the holding
        volume never runs dry. The overflow is the surplus of inflow over the
        draw, ramping in over the band below full, so the level never exceeds
        ``storage_volume``. Both ramps are continuous in the level, so neither
        needs a located event and the gradient is that of the model.

        Parameters
        ----------
        t : float or jnp.ndarray
            Plant time, which selects the phase.
        state : jnp.ndarray
            The unit's state.
        inflow : float or jnp.ndarray
            Flow arriving on the feed port (m³/d).

        Returns
        -------
        dict
            ``draw`` (into the tank), ``overflow`` (out of ``overflow_port``)
            and ``shortfall`` (the demand the holding volume could not supply),
            each a flow (m³/d).
        """
        demand = self._feed[self._phase_index(jnp.asarray(t))] * self.feed_flow
        V_s = self._split_storage(state)[1]
        capacity = float(self.storage_volume)
        band = STORAGE_BAND * capacity
        pumping = jnp.clip((V_s - band) / band, 0.0, 1.0)
        draw = pumping * demand + (1.0 - pumping) * jnp.minimum(demand, inflow)
        full = jnp.clip((V_s - (capacity - band)) / band, 0.0, 1.0)
        overflow = full * jnp.maximum(inflow - draw, 0.0)
        return {"draw": draw, "overflow": overflow, "shortfall": demand - draw}

    def _tank_feed(self, t, state: jnp.ndarray, inlet: Stream):
        """``(Q, C)`` of what enters the tank: the inlet itself, or the draw from
        the holding volume at its concentrations."""
        if not self._storage:
            return inlet.Q, inlet.C
        return self.storage_flows(t, state, inlet.Q)["draw"], self._split_storage(state)[0]

    # ----- the batch feed's inlet check -----------------------------------
    def check_inlet_flows(self, inlets: dict, t0: float, t1: float) -> None:
        """Refuse a ``feed="batch"`` inlet that does not deliver ``feed_flow``
        during fill phases and nothing otherwise over ``(t0, t1)``
        (:class:`~aquakin.plant.units.InletFlowCheckedUnit`).

        The inlet flow is read just inside both ends and at the middle of every
        phase in the span, and at every sample time of a sampled influent
        within it, and must match to :data:`BATCH_FLOW_RTOL` of ``feed_flow``.
        Other feed modes take whatever arrives and check nothing.

        Parameters
        ----------
        inlets : dict
            ``{port: influent or None}``: the influent feeding each input port
            on the plant's clock, ``None`` when another unit feeds it.
        t0, t1 : float
            The solve span, on the plant's clock.

        Raises
        ------
        WiringError
            If another unit feeds a batch-fed tank, so its flow cannot be checked.
        ValueError
            If the influent's flow departs from the fill schedule, naming the
            first time it does.
        """
        if self.feed != "batch":
            return
        source = inlets.get(self.input_port)
        if source is None:
            raise WiringError(
                f"SBRUnit '{self.name}' has feed='batch', which needs an influent on "
                f"'{self.input_port}' so its flow can be checked against the fill "
                f"schedule before the solve; another unit feeds it, or nothing does. Use "
                f"feed='continuous' (the tank takes the flow as it arrives) or "
                f"feed='storage' (a holding volume receives it)."
            )
        times, expected = self._batch_check_points(t0, t1, source)
        with jax.ensure_compile_time_eval():
            Q = np.asarray(jax.vmap(lambda t: source.at(t).Q)(jnp.asarray(times)))
        Q = np.broadcast_to(Q, times.shape)
        off = np.abs(Q - expected) > BATCH_FLOW_RTOL * float(self.feed_flow)
        if off.any():
            i = int(np.argmax(off))
            where = "a fill phase" if expected[i] > 0.0 else "a phase that is not a fill"
            raise ValueError(
                f"SBRUnit '{self.name}' has feed='batch', but its inlet delivers "
                f"Q={Q[i]:.6g} at t={times[i]:.6g}, in {where}, where the schedule "
                f"wants {expected[i]:.6g} (feed_flow during fill phases, 0 otherwise). "
                f"Feed it only during fill phases at feed_flow, or use "
                f"feed='continuous' or feed='storage'."
            )

    def _batch_check_points(self, t0: float, t1: float, source):
        """The times the batch check reads the inlet flow at, and the flow the
        schedule wants at each."""
        samples = np.asarray(source.t) if isinstance(source, InfluentSeries) else np.empty(0)
        times: list[float] = []
        expected: list[float] = []
        k = int(np.floor((t0 - self.cycle_origin) / self._period))
        while self.cycle_origin + k * self._period < t1:
            base = self.cycle_origin + k * self._period
            for phase, start in zip(self.phases, self._phase_starts):
                a = max(base + start, t0)
                b = min(base + start + float(phase.duration), t1)
                if b <= a:
                    continue
                inset = _BATCH_CHECK_INSET * (b - a)
                lo, hi = np.searchsorted(samples, [a + inset, b - inset])
                points = [a + inset, 0.5 * (a + b), b - inset, *samples[lo:hi]]
                times += points
                expected += [float(self.feed_flow) if phase.feed else 0.0] * len(points)
            k += 1
        return np.asarray(times), np.asarray(expected)

    # ----- structural coupling (issue #388) ------------------------------
    def coupling_pattern(self):
        """Structural Jacobian sparsity, over phases.

        State is ``[C (n_species), V, <settling-model state>]``. The within-phase
        RHS is smooth, but the active phase is a constant per phase (it switches at
        the located cycle events, not with the state), so the structural superset
        is the **union over phases**. Two nonlinearities feed the ``self`` block:
        the reaction kinetics (a saturated Monod term is numerically invisible to
        a probe, so the model's syntactic AST pattern is unioned into the species
        sub-block) and the settling clarity dynamics + the ``1/V`` convection
        (smooth nonlinearities whose branches AD over diverse states exercises, as
        for the Takacs settler -- unioned via :func:`ad_union` at each phase's
        representative time). ``inlet``: the convective dilution diagonal on the
        species; the feed couples no other state. With ``feed="storage"`` the
        inlet feeds the holding volume (``[C_s, V_s]`` at the tail) instead and
        the tank draws from it: the tank's species couple to the holding
        volume's species and level, as do the tank volume and the holding
        volume's own rows; the outlets read the tank (effluent) and the holding
        volume (overflow).
        """
        import jax
        import numpy as np

        from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
        from aquakin.plant.coupling import CouplingPattern, ad_union

        net = self.model
        n = net.n_species
        m = self.state_size
        params = net.default_parameters()
        state0 = np.asarray(self.initial_state())
        base_C = jnp.asarray(np.maximum(np.abs(np.asarray(net.default_concentrations())), 1e-3))
        q = self.feed_flow if self.feed_flow is not None else self.full_volume / self._period
        inlet = {self.input_port: Stream(Q=jnp.asarray(q), C=base_C, model=net)}

        # AD over diverse states, unioned over a representative time in each phase
        # (so the fill/decant/settle couplings all appear). A Monod term is invisible
        # to this, so the rate AST is unioned in below.
        self_pat = np.zeros((m, m), dtype=bool)
        for p, start in enumerate(self._phase_starts):
            t_p = jnp.asarray(self.cycle_origin + start + 0.5 * float(self.phases[p].duration))

            def jac(s, _t=t_p):
                return jax.jacfwd(lambda x: self.rhs(_t, x, inlet, params))(s)

            self_pat |= ad_union(jac, state0)
        self_pat[:n, :n] |= structural_sparsity_pattern(net)  # saturated kinetics
        np.fill_diagonal(self_pat, True)

        # Feed enters only through ``q_in/V*(C_in - C)`` -- diagonal in species, no
        # coupling into V or the settling state -- into the tank, or into the
        # holding volume when there is one.
        eye = np.eye(n, dtype=bool)
        inlet_pat = np.zeros((m, n), dtype=bool)
        if not self._storage:
            inlet_pat[:n, :] = eye
            return CouplingPattern(self_pattern=self_pat, inlet_pattern=inlet_pat)
        s0 = m - n - 1  # the holding volume's block [C_s, V_s]
        inlet_pat[s0 : s0 + n, :] = eye
        self_pat[:n, s0 : s0 + n] |= eye  # the draw carries the held concentrations
        self_pat[s0 : s0 + n, s0 : s0 + n] |= eye
        self_pat[: n + 1, m - 1] = True  # the draw (tank species, V) follows the level
        self_pat[s0:, m - 1] = True
        out_pat = np.zeros((n, m), dtype=bool)
        out_pat[:, :n] = eye
        out_pat[:, n:s0] = True  # V and the settling state (the clarified decant)
        out_pat[:, s0 : s0 + n] = eye  # the overflow
        out_pat[:, m - 1] = True
        return CouplingPattern(
            self_pattern=self_pat, inlet_pattern=inlet_pat, output_pattern=out_pat
        )

    # ----- protocol: behaviour -------------------------------------------
    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        C, V, extra = self._split(state)
        idx = self._phase_index(t)
        q_out = self._decant[idx] * self.decant_flow
        c_decant = self.settling.decant_multiplier(C, V, extra) * C
        out = {self.output_port: Stream(Q=q_out, C=c_decant, model=self.model)}
        if self._storage:
            overflow = self.storage_flows(t, state, inputs[self.input_port].Q)["overflow"]
            C_s = self._split_storage(state)[0]
            out[self.overflow_port] = Stream(Q=overflow, C=C_s, model=self.model)
        return out

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Effluent flow = ``decant_flow`` during a decant phase, else 0; with
        ``feed="storage"`` the overflow too, from the holding volume's level.
        Reads the time and the state from the flow context."""
        t = jnp.zeros(()) if (ctx is None or ctx.t is None) else ctx.t
        idx = self._phase_index(t)
        out = {self.output_port: self._decant[idx] * self.decant_flow}
        if self._storage:
            flows = self.storage_flows(t, jnp.asarray(ctx.state), input_flows[self.input_port])
            out[self.overflow_port] = flows["overflow"]
        return out

    def _reaction_conditions(self, inputs: dict[str, Stream], signals: dict | None) -> dict:
        """The static conditions: the SBR runs its kinetics at its static ``T``
        (see the class docstring)."""
        return self._condition_arrays

    def _aeration(self, idx: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
        """The aeration term ``kLa*(do_sat - C)`` on the oxygen species, at the
        ``kLa`` of phase ``idx`` (zero in an unaerated phase)."""
        return self._kla[idx] * self._o2_mask * (self.do_sat - C)

    # ----- the mass balance's transfer and volume contracts --------------
    @property
    def required_signals(self) -> tuple[str, ...]:
        """No control signals: the SBR's aeration follows its phase schedule."""
        return ()

    def oxygen_transfer(
        self,
        t,
        state: jnp.ndarray,
        inputs: dict,
        params: jnp.ndarray,
        signals: dict | None,
    ):
        """The aeration transfer the mass balance integrates
        (:class:`~aquakin.plant.units.OxygenTransferUnit`): the same term
        :meth:`rhs` adds, at the phase :meth:`rhs` is in at ``t``.

        Parameters
        ----------
        t : float or jnp.ndarray
            Plant time, which selects the active phase.
        state : jnp.ndarray
            The unit's state ``[C, V, <settling state>, ...]``.
        inputs : dict[str, Stream]
            The unit's input streams (unused).
        params : jnp.ndarray
            The unit's parameter vector (unused).
        signals : dict or None
            The control-signal bus (unused).

        Returns
        -------
        oxygen, stripped : jnp.ndarray
            Per-species transfer rate per unit volume, shape ``(n_species,)``:
            the aeration term on the oxygen species, and zeros, since the SBR
            co-strips no other gas.
        """
        C = self._split(state)[0]
        idx = self._phase_index(jnp.asarray(t))
        return self._aeration(idx, C), jnp.zeros_like(C)

    def reaction_volumes(self, state: jnp.ndarray, params: jnp.ndarray):
        """The volume each species' reaction term acts in: the tank's current
        liquid volume ``V`` from the state, for every species (the holding
        volume does not react).

        Parameters
        ----------
        state : jnp.ndarray
            The unit's state ``[C, V, <settling state>, ...]``.
        params : jnp.ndarray
            The unit's parameter vector (unused).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_species,)`` (m³).
        """
        return np.full(self.model.n_species, float(self.liquid_volume(state)))

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        C, V, extra = self._split(state)
        idx = self._phase_index(t)
        inlet = inputs[self.input_port]
        q_in, C_in = self._tank_feed(t, state, inlet)
        q_out = self._decant[idx] * self.decant_flow
        settle_on = self._settle[idx]
        mix_on = self._mixed[idx]

        V_safe = jnp.maximum(V, _EPS_V)
        c_decant = self.settling.decant_multiplier(C, V, extra) * C

        # Variable-volume mass balance: d(VC)/dt = Q_in C_in - Q_out C_decant + V r,
        # with dV/dt = Q_in - Q_out, gives the concentration form below. Drawing a
        # clarified (low-particulate) decant therefore concentrates the retained
        # solids -- mass is conserved whatever the settling model reports.
        convection = q_in / V_safe * (C_in - C) - q_out / V_safe * (c_decant - C)

        chemistry = self._reaction(C, params, self._reaction_conditions(inputs, signals))

        aeration = self._aeration(idx, C)

        dC = convection + chemistry + aeration
        dV = jnp.reshape(q_in - q_out, (1,))
        d_extra = self.settling.extra_rhs(C, V, extra, settle_on, mix_on)
        if not self._storage:
            return jnp.concatenate([dC, dV, d_extra])
        # The holding volume is a non-reacting mixed tank: the inlet in, the draw
        # and the overflow out at its own concentrations.
        C_s, V_s = self._split_storage(state)
        flows = self.storage_flows(t, state, inlet.Q)
        dC_s = inlet.Q / jnp.maximum(V_s, _EPS_V) * (inlet.C - C_s)
        dV_s = jnp.reshape(inlet.Q - flows["draw"] - flows["overflow"], (1,))
        return jnp.concatenate([dC, dV, d_extra, dC_s, dV_s])

    # ----- located phase-transition events -------------------------------
    def cycle_events(self, t0: float, t1: float) -> list[Event]:
        """The phase-transition times in ``(t0, t1)`` as one located time event.

        The plant collects this so the integrator lands exactly on every phase
        switch -- the flow/aeration discontinuities are resolved at the boundary
        rather than stepped across. Returns ``[]`` when no switch falls inside
        the span (e.g. a sub-phase-length solve)."""
        t0 = float(t0)
        t1 = float(t1)
        times: list[float] = []
        # First whole-cycle index whose phase starts could fall at/after t0.
        k = int((t0 - self.cycle_origin) // self._period) - 1
        while True:
            base = self.cycle_origin + k * self._period
            if base - self._period > t1:
                break
            for off in self._phase_starts:
                tt = base + off
                if t0 < tt < t1:
                    times.append(tt)
            if base > t1:
                break
            k += 1
        times = sorted(set(round(t, 12) for t in times))
        if not times:
            return []
        return [Event(at_times=times, name=f"{self.name}_phase")]


@dataclass(frozen=True)
class StorageReport:
    """What a storage-fed SBR's holding volume did over a solve.

    Attributes
    ----------
    unit : str
        The SBR's name.
    t : numpy.ndarray
        The saved times, on the plant's clock, shape ``(n_t,)``.
    level : numpy.ndarray
        The holding volume's liquid volume (m³), shape ``(n_t,)``.
    inflow, draw, overflow, shortfall : numpy.ndarray
        Flows (m³ per plant time unit), shape ``(n_t,)``: arriving on the feed
        port, drawn into the tank, overflowing, and demanded but not drawn
        because the holding volume ran empty.
    capacity : float
        ``storage_volume`` (m³).
    overflow_volume, shortfall_volume : float
        The overflow and the shortfall integrated over the saved times
        (trapezoids, so accurate to the ``t_eval`` sampling) (m³).
    """

    unit: str
    t: np.ndarray
    level: np.ndarray
    inflow: np.ndarray
    draw: np.ndarray
    overflow: np.ndarray
    shortfall: np.ndarray
    capacity: float
    overflow_volume: float
    shortfall_volume: float

    def summary(self) -> str:
        """A short text report: the level range, and any overflow or shortfall."""
        lines = [
            f"SBR '{self.unit}' holding volume: level {self.level.min():.4g} to "
            f"{self.level.max():.4g} m³ of {self.capacity:.4g} m³",
            f"  overflowed: {self.overflow_volume:.4g} m³",
            f"  ran empty: {self.shortfall_volume:.4g} m³ of fill demand not drawn",
        ]
        return "\n".join(lines)


def storage_report(plant, solution, unit: str, params=None) -> StorageReport:
    """Report what a ``feed="storage"`` SBR's holding volume did over a solve.

    The holding volume overflows through its own port, which the mass balance
    counts as an outflow, and running empty cuts the fill draw back to what
    arrives; this reports both as flows over the saved times, evaluated with the
    inputs the solve's right-hand side gave the unit.

    Parameters
    ----------
    plant : Plant
        The plant that was solved.
    solution : PlantSolution
        Its solution.
    unit : str
        The SBR's name.
    params : jnp.ndarray, optional
        The parameters the solve used (defaults to the plant's).

    Returns
    -------
    StorageReport

    Raises
    ------
    ValueError
        If ``unit`` is not an SBR with ``feed="storage"``.
    """
    sbr = plant.units[unit]
    if not (isinstance(sbr, SBRUnit) and sbr.feed == "storage"):
        raise ValueError(
            f"'{unit}' is not an SBRUnit with feed='storage'; it has no holding volume."
        )
    params = plant.default_parameters() if params is None else params

    def term(u, t, state, inputs, signals, unit_params):
        inflow = inputs[u.input_port].Q
        flows = u.storage_flows(t, state, inflow)
        return (
            u._split_storage(state)[1],
            inflow,
            flows["draw"],
            flows["overflow"],
            flows["shortfall"],
        )

    level, inflow, draw, overflow, shortfall = (
        np.asarray(a) for a in unit_terms(plant, solution, params, [unit], term)[unit]
    )
    t = np.asarray(solution.t_native)
    return StorageReport(
        unit=unit,
        t=t,
        level=level,
        inflow=inflow,
        draw=draw,
        overflow=overflow,
        shortfall=shortfall,
        capacity=float(sbr.storage_volume),
        overflow_volume=float(np.trapezoid(overflow, t)),
        shortfall_volume=float(np.trapezoid(shortfall, t)),
    )
