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

The feed is drawn at the unit's own ``feed_flow`` (a fill pump) during the fill
phase, taking its composition from the connected feed stream; the effluent
leaves at ``decant_flow`` during the decant phase, at the clarified
concentration the settling model reports. Both are zero in every other phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import jax.numpy as jnp

from aquakin.integrate.events import Event
from aquakin.plant.settling import SettlingModel
from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork

_EPS_V = 1e-9  # guard 1/V when the tank is near-empty


@dataclass(frozen=True)
class SBRPhase:
    """One phase of an SBR cycle.

    Parameters
    ----------
    name : str
        Phase label (e.g. ``"fill"``, ``"react"``, ``"settle"``, ``"decant"``,
        ``"idle"``), used in the event log.
    duration : float
        Phase length in the plant's time unit.
    feed : bool
        Draw feed in at the unit's ``feed_flow`` during this phase (fill).
    decant : bool
        Draw clarified effluent out at ``decant_flow`` during this phase (decant).
    kla : float
        Oxygen mass-transfer coefficient applied during this phase (aerobic
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
    mixed: Optional[bool] = None


@dataclass
class SBRUnit:
    """A sequencing batch reactor: variable-volume, time-phased single tank.

    Parameters
    ----------
    name : str
    network : CompiledNetwork
    phases : sequence of SBRPhase
        One cycle, in order. The cycle repeats with period ``sum(durations)``.
    full_volume : float
        Maximum (full) liquid volume.
    feed_flow : float
        Fill-pump flow drawn during ``feed`` phases.
    decant_flow : float
        Effluent flow drawn during ``decant`` phases.
    settling : SettlingModel
        Pluggable settle/decant clarification strategy.
    particulate_species : sequence of str
        Species that settle (e.g. the ASM ``X*`` solids); the settling model
        clarifies these in the decant draw.
    initial_fraction : float
        Starting fill level ``V0 / full_volume`` (the heel left after the
        previous decant). Default 0.25.
    conditions : dict
        Per-network condition values (e.g. ``{"T": 293.15}``), as for a CSTR.
    do_sat : float
        Oxygen saturation in the aeration term ``kLa*(do_sat - C_O2)``.
    oxygen_species : str
        The aerated species (default ``"SO"``); ignored if absent from the network.
    initial_concentrations : jnp.ndarray, optional
        Initial bulk concentration vector (defaults to the network defaults).
    cycle_origin : float
        Time at which phase 0 of a cycle begins (default 0).
    input_port, output_port : str
        Port names (``"feed"`` / ``"effluent"``).
    """

    name: str
    network: "CompiledNetwork"
    phases: Sequence[SBRPhase]
    full_volume: float
    feed_flow: float
    decant_flow: float
    settling: SettlingModel
    particulate_species: Sequence[str] = ()
    initial_fraction: float = 0.25
    conditions: dict = field(default_factory=dict)
    do_sat: float = 8.0
    oxygen_species: str = "SO"
    initial_concentrations: Optional[jnp.ndarray] = None
    cycle_origin: float = 0.0
    input_port: str = "feed"
    output_port: str = "effluent"

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError(f"SBRUnit '{self.name}' needs at least one phase.")
        if any(p.duration <= 0 for p in self.phases):
            raise ValueError(f"SBRUnit '{self.name}' phase durations must be > 0.")
        if not (0.0 < self.initial_fraction <= 1.0):
            raise ValueError(
                f"SBRUnit '{self.name}' initial_fraction must be in (0, 1].")
        missing = [c for c in self.network.conditions_required
                   if c not in self.conditions]
        if missing:
            raise ValueError(
                f"SBRUnit '{self.name}' is missing condition values for: "
                f"{missing}. Provided: {list(self.conditions)}.")
        for s in self.particulate_species:
            if s not in self.network.species_index:
                raise ValueError(
                    f"SBRUnit '{self.name}' particulate species '{s}' not in "
                    f"the network.")

        # Phase schedule (concrete floats; static).
        durations = [float(p.duration) for p in self.phases]
        self._period = float(sum(durations))
        cum = []
        acc = 0.0
        for d in durations:
            acc += d
            cum.append(acc)
        self._interior_breaks = jnp.asarray(cum[:-1])      # within-cycle boundaries
        self._phase_starts = [0.0] + cum[:-1]              # phase start offsets

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
        n = self.network.n_species
        if self.oxygen_species in self.network.species_index:
            o2 = self.network.species_index[self.oxygen_species]
            self._o2_mask = jnp.zeros((n,)).at[o2].set(1.0)
        else:
            self._o2_mask = jnp.zeros((n,))

        # Condition arrays for the rate evaluation (as on a CSTR).
        self._condition_arrays = {
            k: jnp.asarray([float(v)]) for k, v in self.conditions.items()}

        # Bind the settling model's particulate mask.
        self.settling.bind(self.network, list(self.particulate_species))

    # ----- protocol: identity / layout -----------------------------------
    @property
    def state_size(self) -> int:
        return self.network.n_species + 1 + self.settling.extra_state_size()

    def liquid_volume(self, state: jnp.ndarray):
        """The current liquid volume (the settling-state tail carries no mass).

        The explicit mass-balance inventory contract (see
        :func:`aquakin.plant.balance._unit_inventory`): inventory is ``V*C`` with
        the concentration head block.
        """
        return self._split(state)[1]

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        C0 = (self.network.default_concentrations()
              if self.initial_concentrations is None
              else jnp.asarray(self.initial_concentrations))
        V0 = jnp.asarray([self.initial_fraction * float(self.full_volume)])
        return jnp.concatenate([C0, V0, self.settling.initial_extra_state()])

    # ----- phase helpers --------------------------------------------------
    def _phase_index(self, t: jnp.ndarray) -> jnp.ndarray:
        """Index of the active phase at time ``t`` (cyclic, jit/AD-safe)."""
        tau = jnp.mod(t - self.cycle_origin, self._period)
        return jnp.searchsorted(self._interior_breaks, tau, side="right")

    def _split(self, state: jnp.ndarray):
        n = self.network.n_species
        return state[:n], state[n], state[n + 1:]

    # ----- protocol: behaviour -------------------------------------------
    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        C, V, extra = self._split(state)
        idx = self._phase_index(t)
        q_out = self._decant[idx] * self.decant_flow
        c_decant = self.settling.decant_multiplier(C, V, extra) * C
        return {self.output_port: Stream(Q=q_out, C=c_decant, network=self.network)}

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray,
                     ctx=None) -> dict:
        """Effluent flow = ``decant_flow`` during a decant phase, else 0. Reads
        the time from the flow context (a scheduled split)."""
        t = jnp.zeros(()) if (ctx is None or ctx.t is None) else ctx.t
        idx = self._phase_index(t)
        return {self.output_port: self._decant[idx] * self.decant_flow}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> jnp.ndarray:
        C, V, extra = self._split(state)
        idx = self._phase_index(t)
        q_in = self._feed[idx] * self.feed_flow
        q_out = self._decant[idx] * self.decant_flow
        kla = self._kla[idx]
        settle_on = self._settle[idx]
        mix_on = self._mixed[idx]

        V_safe = jnp.maximum(V, _EPS_V)
        C_in = inputs[self.input_port].C
        c_decant = self.settling.decant_multiplier(C, V, extra) * C

        # Variable-volume mass balance: d(VC)/dt = Q_in C_in - Q_out C_decant + V r,
        # with dV/dt = Q_in - Q_out, gives the concentration form below. Drawing a
        # clarified (low-particulate) decant therefore concentrates the retained
        # solids -- mass is conserved whatever the settling model reports.
        convection = q_in / V_safe * (C_in - C) - q_out / V_safe * (c_decant - C)

        stoich = self.network.compute_stoich(params)
        rates = self.network.rates(C, params, self._condition_arrays, 0)
        chemistry = stoich.T @ rates

        aeration = kla * self._o2_mask * (self.do_sat - C)

        dC = convection + chemistry + aeration
        dV = jnp.reshape(q_in - q_out, (1,))
        d_extra = self.settling.extra_rhs(C, V, extra, settle_on, mix_on)
        return jnp.concatenate([dC, dV, d_extra])

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
