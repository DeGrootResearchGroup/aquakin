"""Plant flowsheet assembly and monolithic Diffrax integration.

A :class:`Plant` owns a set of :class:`Unit`-Protocol-conforming components,
the directed connections between them, and time-varying influent sources.
Calling :meth:`Plant.solve` builds one flat state vector by concatenating
each unit's internal state, integrates the whole thing under Diffrax, and
returns a :class:`PlantSolution`.

The RHS uses a two-pass design per evaluation:

1. **Stream pass** — walk units in evaluation order, computing each unit's
   output streams from its current internal state and its input streams.
   Internal recycles work naturally because output streams are functions
   of *current* state, not future state.
2. **Derivative pass** — compute each unit's ``dstate`` from its current
   state and input streams (now all known from pass 1). Concatenate.

The evaluation order is the user-supplied unit ordering. Cycles in the
connection graph are allowed (recycles); the user marks edges that close
cycles by ordering downstream units before upstream consumers — see
:meth:`Plant.connect` documentation.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence, Union

import diffrax
import jax
import jax.numpy as jnp

from aquakin.core.hints import did_you_mean
from aquakin.core.network import CompiledNetwork
from aquakin.integrate._common import (
    DifferentiationConfig,
    IntegratorConfig,
    _coerce_atol,
    _run_diffeqsolve,
    concrete_settings_key,
    default_atol,
    forward_adjoint,
    friendly_solve_errors,
    to_native_time,
)
from aquakin.integrate.discrete_adjoint import esdirk_adjoint_solve
from aquakin.integrate.events import Event, solve_with_events
from aquakin.plant import sensitivity as _sensitivity
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.plant_types import (
    Connection,
    ParameterLayout,
    PlantCheck,
    PlantSolution,
    SteadyStateResult,
)
from aquakin.plant.recycle import RecycleResolver
from aquakin.plant.streams import Stream, StreamSeries
from aquakin.plant.temperature import (
    OPERATING_T_SIGNAL as _OPERATING_T_SIGNAL,
)
from aquakin.plant.temperature import AlgebraicTemperature, TemperatureModel
from aquakin.plant.translators import IdentityTranslator, StateTranslator
from aquakin.plant.units import Unit

# Reserved key under which :meth:`Plant._split_state` exposes the appended
# temperature-state block in the per-unit state map (not a real unit, so it is
# never iterated as one -- ``_unit_order`` does not contain it).
_TEMPERATURE_KEY = "__temperature__"


def _senses_concentration(unit) -> bool:
    """Whether a control loop can read a measured species from ``unit``'s state.

    The signal bus reconstructs the sensed value as ``state[sensor][meas_idx]``,
    so the sensor's state must carry the species concentrations as its leading
    entries -- true for a reactor (CSTR, MBR, SBR, digester, storage), false for a
    stateless mixer/splitter/ideal-clarifier. A unit whose state has the
    concentration vector as a prefix (``state_size >= n_species``) qualifies.
    """
    net = getattr(unit, "network", None)
    return net is not None and getattr(unit, "state_size", 0) >= net.n_species


def _concrete_teval_key(t_eval):
    """Hashable cache key for a concrete ``t_eval`` (or ``None``).

    Returns ``(key, is_concrete)``. A traced ``t_eval`` (the solve running under
    an outer trace) cannot be materialised into a key, so ``is_concrete`` is
    ``False`` and the caller bypasses the compiled-solve cache.
    """
    if t_eval is None:
        return None, True
    try:
        return tuple(float(x) for x in jnp.asarray(t_eval)), True
    except (jax.errors.ConcretizationTypeError, TypeError):
        return None, False


class Plant:
    """A plant flowsheet of :class:`Unit` components.

    Build a plant by:

    1. Constructing each unit (CSTRUnit, MixerUnit, etc.).
    2. Adding them via :meth:`add_unit` in the order you want their
       RHS evaluated (downstream-first if there's a recycle).
    3. Adding influent sources via :meth:`add_influent`.
    4. Connecting units via :meth:`connect`.
    5. Calling :meth:`solve`.

    Parameters
    ----------
    name : str
    recycle_passes : int, optional
        Number of Gauss-Seidel *mop-up* passes per RHS evaluation (default 3).
        Both the recycle flows AND concentrations are first resolved **exactly**
        and gain-independently -- flows by :meth:`_resolve_flows`, concentrations
        by :meth:`_resolve_recycle_concentrations` (an affine probe + linear
        solve) -- so for any linear topology (every shipped plant) the recycle
        back-edges are seeded at their exact fixed point and these passes do no
        work. They only refine the residual of a genuinely *non-affine* in-cycle
        unit (an ASM↔ADM translator inside a pure-stateless loop, which the
        shipped units cannot form), so the default 3 is ample and rarely matters.
        It is a *fixed* count (not iterate-to-tolerance) because the RHS is jitted
        and differentiated. The first non-traced :meth:`solve` runs a one-time
        convergence diagnostic on the mop-up residual and warns only if even the
        exact pre-solve plus these passes has not converged (a non-affine loop);
        raise ``recycle_passes`` until it clears, or set ``recycle_tol``.
    recycle_tol : float, optional
        Relative tolerance for the **adaptive** recycle resolution, **on by
        default** (``1e-8``). The recycle back-edge *streams* (flow,
        concentration, temperature) are iterated to this tolerance -- correct for
        *any* topology, not just the low-gain BSM reject loop. The alternative
        fixed ``recycle_passes`` mop-up converges in ``log(tol)/log(rho)`` passes
        where ``rho`` is the nonlinear flow<->concentration coupling's spectral
        radius (~0.0066 for BSM, so 3 passes is ample), but ``rho`` is
        topology-dependent and not bounded below 1: a recycle-heavy plant with a
        strong concentration-dependent in-loop flow can leave the fixed count
        silently under-converged. The adaptive solve (warm-started from the exact
        affine seed, an adaptive :func:`jax.lax.while_loop` wrapped in
        :func:`jax.lax.custom_root`) iterates until the *actual* residual clears,
        so it converges for any ``rho < 1``, stops early on a low-gain plant, and
        -- via the implicit-function-theorem tangent -- gives a gradient that is
        exact and O(1) in the pass count. ``1e-8`` is well below the typical solver
        ``rtol`` and a strict improvement on the old fixed-3-pass default (~1e-6 for
        BSM) at ~neutral cost (~3 iterations from the affine seed). Set
        ``recycle_tol=None`` to fall back to the fixed ``recycle_passes`` path. See
        :meth:`_adaptive_recycle_refine`.
    recycle_max_passes : int, optional
        Cap on the adaptive ``recycle_tol`` iteration (default 100), the
        worst-case guard for a near-unit-gain loop.
    """

    def __init__(
        self,
        name: str,
        *,
        recycle_passes: int = 3,
        recycle_tol: Optional[float] = 1e-8,
        recycle_max_passes: int = 100,
    ) -> None:
        self.name = name
        if recycle_passes < 1:
            raise ValueError(f"recycle_passes must be >= 1; got {recycle_passes}")
        self.recycle_passes = int(recycle_passes)
        # Adaptive recycle-concentration resolution (on by default). The recycle
        # back-edge concentrations are iterated to this relative tolerance (an
        # adaptive ``lax.while_loop`` warm-started from the exact affine seed,
        # wrapped in ``jax.lax.custom_root`` for the IFT gradient); ``None`` falls
        # back to the fixed ``recycle_passes`` Gauss-Seidel mop-up. The fixed
        # count is calibrated to the BSM reject-loop gain (it converges in ~2-3
        # passes), but the passes-to-tolerance is ``log(tol)/log(rho)`` where
        # ``rho`` is the nonlinear flow<->concentration coupling's spectral
        # radius -- topology-dependent and not bounded below 1 for an arbitrary
        # recycle-heavy plant, where the fixed count would silently under-
        # converge. The adaptive solve iterates until the *actual* residual
        # clears, so it is correct for any ``rho < 1`` and the gradient (via the
        # IFT tangent) is O(1) in the pass count. On by default (``1e-8``); the
        # fixed-pass path (``None``) is the bit-identical historic behaviour.
        if recycle_tol is not None and recycle_tol <= 0:
            raise ValueError(f"recycle_tol must be > 0 or None; got {recycle_tol}")
        self.recycle_tol = recycle_tol
        if recycle_max_passes < 1:
            raise ValueError(f"recycle_max_passes must be >= 1; got {recycle_max_passes}")
        self.recycle_max_passes = int(recycle_max_passes)
        self.units: dict[str, Unit] = {}
        # Units in the order they were added (the user's order; arbitrary).
        self._insertion_order: list[str] = []
        # The RHS *evaluation* order: a topological sort of the feed-forward
        # connection graph, computed by _finalize_topology() (recycles are the
        # graph back-edges, detected automatically, not by add order).
        self._unit_order: list[str] = []
        self.connections: list[Connection] = []
        self.influents: dict[str, InfluentSeries] = {}
        # Canonical entry / exit endpoints, set by the plant builders so callers
        # never hard-code a "unit.port". A builder whose front/effluent ports move
        # with its options (e.g. a BSM2 influent bypass relocating the entry to
        # ``bypass_split.in`` and the effluent to ``effluent_mix.out``) records the
        # right ones here, so ``plant.add_influent(series)`` and the metric
        # evaluators read them instead of a guessable literal. ``None`` on a plant
        # whose builder did not set them.
        self.influent_endpoint: Optional[str] = None
        self.effluent_endpoint: Optional[str] = None
        # Semantic stream shortcuts: engineering names (``"effluent"``, ``"ras"``,
        # ``"internal_recycle"``, ``"primary_sludge"``, ``"reject"`` ...) -> the
        # internal ``"unit.port"`` they resolve to, registered by the builder
        # (see :meth:`register_stream`). ``plant.stream(sol, "effluent")`` then
        # reads the right port without the user knowing it is
        # ``"tank5_split.internal_recycle"``; :meth:`list_streams` lists them.
        self.named_streams: dict[str, str] = {}
        # How reactor temperature is handled (see aquakin.plant.temperature). The
        # default carries no state and reproduces the historic instantaneous
        # flow-weighted behaviour exactly; a HeatBalanceTemperature gives each
        # finite-volume unit a dynamic temperature state. Swappable on the built
        # plant (clears the compiled-solve cache).
        self.temperature_model: TemperatureModel = AlgebraicTemperature()
        # Filled in at solve() time:
        self._state_layout: dict[str, tuple[int, int]] = {}
        self._total_state_size: int = 0
        # Appended temperature-state block: the tracked unit names (in state
        # order) and the (start, size) slice of the flat state vector. Empty for
        # the default AlgebraicTemperature.
        self._temperature_units: list[str] = []
        # Volumes (m^3) of the tracked units, in the same order -- precomputed
        # once so the heat-balance RHS need not re-read float(unit.volume) per call.
        self._temperature_volumes: jnp.ndarray = jnp.zeros((0,))
        self._temperature_block: tuple[int, int] = (0, 0)
        self._parameter_layout: ParameterLayout = ParameterLayout()
        self._network_param_index: dict[str, int] = {}
        # Static connection adjacency, rebuilt by _build_state_layout():
        self._inputs_by_unit: dict[str, list[Connection]] = {}
        self._recycle_keys: list[tuple[Optional[str], str]] = []
        # The recycle (back-edge) connections and, for any auto-detected one with
        # no explicit ``initial_value``, a zero-flow seed stream. Set by
        # _finalize_topology().
        self._recycle_conns: list[Connection] = []
        self._recycle_seeds: dict[tuple[str, str], Stream] = {}
        # One-time guards for the concrete recycle diagnostics (warnings, run
        # once per plant on the first non-traced solve): the flow-affinity check
        # and the concentration-sweep convergence check.
        self._flow_affinity_checked: bool = False
        self._recycle_convergence_checked: bool = False
        # Recycle-flow / recycle-concentration resolution, owning its own
        # tri-state map-constant caches (see aquakin.plant.recycle).
        self._recycle = RecycleResolver(self)
        # Colored-Jacobian root finder (built once, concretely, on the first
        # colored_jacobian=True solve): (root_finder, n_colors, ok). ``ok`` False
        # means the setup guard found the colored Jacobian disagreed with the
        # dense one at the start state, so the solve falls back to the dense path.
        self._colored_root_finder: Optional[tuple] = None
        # Colored-Jacobian builder for the stable_adjoint BACKWARD pass (built
        # once, concretely, on the first colored_jacobian=True stable_adjoint
        # solve): (builder_or_None, n_colors, ok). Distinct from
        # _colored_root_finder (the forward root finder, n states) -- this colors
        # the AUGMENTED (time-carrying, n+1) primal rhs the discrete adjoint
        # differentiates. ``ok`` False => the guard found a colored/dense mismatch
        # at the start state, so the backward falls back to dense jacfwd.
        self._colored_adjoint_builder: Optional[tuple] = None
        # Colored-Jacobian builder for the PTC STEADY-STATE iteration (built once,
        # concretely, on the first colored_jacobian=True steady_state call):
        # (builder_or_None, n_colors, ok). Colors the autonomous steady residual
        # Jacobian dF/dy. The PTC operating-point neighbourhood is narrow, so the
        # start-state pattern stays valid throughout (unlike a wide dynamic run).
        self._colored_steady_builder: Optional[tuple] = None
        # One-time guard for materialising the DO controllers a CSTRUnit's
        # closed-loop Aeration spec requires (see _materialize_aeration).
        self._aeration_materialized: bool = False
        # Per-instance cache of the jit-compiled forward solve, keyed by call
        # signature + solver settings. The plant RHS closes over the (static)
        # unit graph, so once compiled the same solve is reused across repeated
        # solves of this plant -- e.g. a parameter sweep / Monte Carlo that
        # builds the plant once and solves it many times, or a warm-started
        # steady-state-then-dynamic run. Without it every solve rebuilds the RHS
        # closure and Diffrax recompiles the whole stiff plant (~tens of seconds)
        # each call. Assumes the plant is not structurally mutated after the
        # first solve (units/connections fixed), as the reactors assume too.
        self._jit_cache: dict = {}
        # Compiled PTC steady-state forward solves, keyed by settings. The eager
        # ``jax.lax.while_loop`` in ``ptc_forward`` re-traces and recompiles on
        # every call (~12-17 s for BSM2), so a persisted jitted solver lets a
        # repeated concrete ``steady_state`` (a sweep / multistart / figure
        # regen) pay that compile once and reuse it (~40 ms run thereafter).
        self._steady_jit_cache: dict = {}

    # ----- assembly --------------------------------------------------------

    def add_unit(self, unit: Unit) -> None:
        """Register a unit.

        Units may be added in any order: the plant computes the RHS evaluation
        order itself by topologically sorting the connection graph at solve time
        (recycles are the graph back-edges, detected automatically -- you do not
        have to add a downstream unit before its upstream consumer). The add
        order is used only as a deterministic tie-break for the sort and to order
        the per-network parameter blocks.
        """
        if unit.name in self.units:
            raise ValueError(f"Unit '{unit.name}' already added")
        self.units[unit.name] = unit
        self._insertion_order.append(unit.name)
        # Seed the evaluation order with the insertion order so it is non-empty
        # before the first solve (helpers that just enumerate units read it);
        # _finalize_topology() reorders it into a valid topological order.
        self._unit_order.append(unit.name)

    def add_influent(
        self,
        name: str,
        series: InfluentSeries,
        to: Optional[str] = None,
        *,
        translator: Optional[StateTranslator] = None,
    ) -> None:
        """Register an external time-varying influent stream, optionally wiring it.

        Parameters
        ----------
        name : str
            Identifier for this influent (used in stream keys and error
            messages).
        series : InfluentSeries
            The time-varying influent data.
        to : str, optional
            Destination endpoint to feed this influent into, as ``"unit.port"``
            (or bare ``"unit"`` to use the unit's sole input port). When given,
            the connection is made here, so no separate :meth:`connect` call is
            needed. The destination unit must already be added. When omitted it
            defaults to the plant's :attr:`influent_endpoint` (set by the
            builders, so ``plant.add_influent("feed", series)`` wires to the
            canonical front without hard-coding a port); if that is also unset
            the influent is registered but inert -- influents are not valid
            :meth:`connect` sources, so wiring happens here.
        translator : StateTranslator, optional
            Translator for the influent -> unit connection. Defaults to identity
            when the influent and destination share a network.
        """
        if name in self.influents:
            raise ValueError(f"Influent '{name}' already added")
        self.influents[name] = series
        if to is None:
            to = self.influent_endpoint
        if to is not None:
            to_unit, to_port = self._parse_endpoint(to, role="destination")
            translator = self._default_translator(
                series.network,
                to_unit,
                translator,
                f"influent '{name}'",
                f"{to_unit}.{to_port}",
            )
            self.connections.append(
                Connection(
                    from_unit=None,
                    from_port=name,
                    to_unit=to_unit,
                    to_port=to_port,
                    translator=translator,
                    initial_value=None,
                )
            )

    def set_temperature(self, celsius: float, *, units: Optional[Iterable[str]] = None) -> "Plant":
        """Set the operating temperature of the reactors, in **degrees Celsius**.

        One knob for "run the plant at this temperature": converts to Kelvin and
        writes the static ``T`` condition of every temperature-bearing reactor,
        so a re-solve runs the kinetics -- including the Arrhenius
        ``temperature_corrections`` -- at that temperature. (For a network whose
        corrections are referenced to this temperature, the correction is unity
        and the run reproduces the calibrated operating point; a colder/warmer
        setting drives the kinetics away from it.)

        By default it targets the activated-sludge reactors (the units exposing
        :meth:`CSTRUnit.set_temperature` with a ``T`` condition) and **leaves a
        fixed-temperature unit like the heated anaerobic digester untouched** --
        the digester is a separate ADM1 unit without that method. Pass ``units``
        to target a specific set by name.

        Parameters
        ----------
        celsius : float
            Operating temperature in °C (converted to Kelvin internally; the
            ASM/ADM condition fields are Kelvin).
        units : iterable of str, optional
            Unit names to set. Defaults to every reactor that supports it.

        Returns
        -------
        Plant
            ``self`` (so calls can be chained after ``build_*``).

        Raises
        ------
        ValueError
            If a name in ``units`` is unknown or does not support a temperature.
        """
        kelvin = float(celsius) + 273.15
        if units is None:
            targets = [
                n
                for n, u in self.units.items()
                if hasattr(u, "set_temperature")
                and "T" in getattr(u.network, "conditions_required", ())
            ]
        else:
            targets = list(units)
            for name in targets:
                if name not in self.units:
                    raise ValueError(f"Unknown unit '{name}'.")
                if not hasattr(self.units[name], "set_temperature"):
                    raise ValueError(f"Unit '{name}' does not support set_temperature.")
        for name in targets:
            self.units[name].set_temperature(kelvin)
        # The compiled solve bakes in the (now-changed) condition values, so it
        # must be recompiled on the next solve.
        self._jit_cache.clear()
        self._steady_jit_cache.clear()
        return self

    def set_temperature_model(self, model: TemperatureModel) -> "Plant":
        """Select how reactor temperature is handled (see :mod:`aquakin.plant.temperature`).

        Pass :class:`~aquakin.plant.temperature.HeatBalanceTemperature` to give
        each finite-volume liquid unit a dynamic temperature state (a first-order
        heat balance), or the default
        :class:`~aquakin.plant.temperature.AlgebraicTemperature` for the
        instantaneous flow-weighted behaviour. Changes the flat state-vector
        length (an appended temperature block), so it clears the compiled-solve
        cache; rebuild ``y0`` (e.g. via ``bsm2_warm_start``) after calling it.
        Returns ``self`` for chaining.
        """
        self.temperature_model = model
        self._jit_cache.clear()
        self._steady_jit_cache.clear()
        # The colored-Jacobian builders cache seed matrices / sparsity patterns
        # sized for the concrete state. The appended temperature block changes the
        # state length, so a previously-built builder is stale and would be
        # dimension-mismatched on the next colored solve -- reset them too (unlike
        # set_temperature, which leaves the state size and pattern unchanged).
        self._colored_root_finder = None
        self._colored_adjoint_builder = None
        self._colored_steady_builder = None
        return self

    def connect(
        self,
        source: str,
        dest: str,
        *,
        translator: Optional[StateTranslator] = None,
        initial_value: Optional[Stream] = None,
    ) -> None:
        """Wire a stream from one unit's output to another unit's input.

        Both endpoints are ``"unit.port"`` strings, read as ``source -> dest``.
        The port may be omitted (bare ``"unit"``) when the unit has exactly one
        port for that role -- a single output (source) or single input (dest) --
        so only multi-port units like mixers and splitters need an explicit
        port. To feed an external influent, use
        :meth:`add_influent` with ``to=...`` rather than :meth:`connect`.

        Parameters
        ----------
        source : str
            Source endpoint, ``"unit.port"`` or bare ``"unit"`` for the unit's
            sole output port.
        dest : str
            Destination endpoint, ``"unit.port"`` or bare ``"unit"`` for the
            unit's sole input port.
        translator : StateTranslator, optional
            State translator to apply as the stream crosses the connection.
            Defaults to :class:`IdentityTranslator` when source and destination
            share a network; required when they differ.
        initial_value : Stream, optional
            Seed for the stream's first RHS pass on a *recycle* edge. Recycles
            are detected automatically as the back-edges of the connection graph
            (you do not mark them, and the units may be added in any order); a
            recycle with no ``initial_value`` is auto-seeded with a zero-flow
            stream of the source network. Pass it only to override that with a
            non-zero warm start (e.g. a temperature-carrying seed to ignite
            temperature propagation around the loop on the first pass). Ignored
            for feed-forward edges. Both endpoints must already be added.

        Examples
        --------
        >>> plant.connect("tank1", "tank2")                 # sole ports inferred
        >>> plant.connect("tank5_split.to_clarifier", "clarifier")
        >>> plant.connect("split.recycle", "mix.recycle")   # recycle: auto-detected
        """
        src_unit, src_port = self._parse_endpoint(source, role="source")
        dst_unit, dst_port = self._parse_endpoint(dest, role="destination")
        translator = self._default_translator(
            self._unit_network(src_unit),
            dst_unit,
            translator,
            f"{src_unit}.{src_port}",
            f"{dst_unit}.{dst_port}",
        )
        self.connections.append(
            Connection(
                from_unit=src_unit,
                from_port=src_port,
                to_unit=dst_unit,
                to_port=dst_port,
                translator=translator,
                initial_value=initial_value,
            )
        )

    def check(self, *, raise_on_error: bool = False) -> PlantCheck:
        """Validate the wiring before solving: find unfed ports / dangling outputs.

        Every unit input port must be fed by exactly one stream (an influent or
        another unit's output); an **unfed** input port has no source, so the RHS
        sweep cannot resolve it and the solve fails. Output ports consumed by no
        connection are reported too, but as information -- a terminal stream that
        leaves the plant (final effluent, wasted sludge, disposal cake, biogas)
        is legitimately unconsumed.

        Parameters
        ----------
        raise_on_error : bool, optional
            If True, raise ``ValueError`` when any input port is unfed (the
            actionable error) instead of only reporting it. Default False.

        Returns
        -------
        PlantCheck
            ``unfed_ports`` (errors), ``dangling_outputs`` (info), ``recycles``
            (the detected back-edges), with ``.ok`` and ``.summary()``.
        """
        self._finalize_topology()
        fed = {(c.to_unit, c.to_port) for c in self.connections}
        consumed = {(c.from_unit, c.from_port) for c in self.connections if c.from_unit is not None}
        unfed, dangling = [], []
        for name in self._insertion_order:
            unit = self.units[name]
            for port in unit.input_ports:
                if (name, port) not in fed:
                    unfed.append(f"{name}.{port}")
            for port in unit.output_ports:
                if (name, port) not in consumed:
                    dangling.append(f"{name}.{port}")
        result = PlantCheck(
            unfed_ports=unfed,
            dangling_outputs=dangling,
            recycles=[f"{u}.{p}" for (u, p) in self._recycle_keys],
        )
        if raise_on_error and not result.ok:
            raise ValueError(
                f"Plant '{self.name}' has unfed input ports (no stream wired "
                f"in): {unfed}. Wire them with connect()/add_influent() before "
                f"solving."
            )
        return result

    def _parse_endpoint(self, spec: str, *, role: str) -> tuple[str, str]:
        """Resolve a ``"unit.port"`` / ``"unit"`` endpoint string to ``(unit, port)``.

        ``role`` is ``"source"`` (resolves against the unit's output ports) or
        ``"destination"`` (input ports). A bare unit infers its sole port for
        that role; an unknown unit, an unknown port, or an ambiguous bare unit
        (more than one port for the role) raises a clear error.
        """
        unit_name, _, port = spec.partition(".")
        if unit_name not in self.units:
            if unit_name in self.influents:
                raise ValueError(
                    f"'{unit_name}' is an influent, not a unit; wire it with "
                    f"add_influent('{unit_name}', series, to=...), not connect()."
                )
            raise KeyError(f"Unknown unit '{unit_name}' in endpoint '{spec}'.")
        ports = (
            self.units[unit_name].output_ports
            if role == "source"
            else self.units[unit_name].input_ports
        )
        if port:
            if port not in ports:
                raise KeyError(
                    f"Unit '{unit_name}' has no {role} port '{port}'; available: {list(ports)}."
                )
            return unit_name, port
        if len(ports) != 1:
            raise ValueError(
                f"Endpoint '{spec}' omits the port, but unit '{unit_name}' has "
                f"{len(ports)} {role} ports {list(ports)}; name one explicitly "
                f"as '{unit_name}.<port>'."
            )
        return unit_name, ports[0]

    def _default_translator(
        self, source_network, dest_unit, translator, src_label, dst_label
    ) -> StateTranslator:
        """Return ``translator`` if given, else the default for the endpoints.

        The default is :class:`IdentityTranslator` when the source and
        destination share a network; a cross-network connection with no
        explicit translator is an error.
        """
        if translator is not None:
            return translator
        target_network = self._unit_network(dest_unit)
        if source_network is target_network:
            return IdentityTranslator(target_network)
        # Two *different* network objects. Distinguish the common mistake -- two
        # separate instances of the same model (e.g. calling bsm2_asm1_network()
        # twice, so the plant and the influent carry different temperature
        # corrections / parameters) -- from a genuine cross-network connection.
        same_model = getattr(source_network, "name", object()) == getattr(
            target_network, "name", None
        ) and getattr(source_network, "species", object()) == getattr(
            target_network, "species", None
        )
        if same_model:
            raise ValueError(
                f"Connection {src_label} -> {dst_label}: the two endpoints use "
                f"different *instances* of the same network "
                f"('{getattr(source_network, 'name', '?')}'). Build the network "
                f"once and pass that same object to both the plant and the "
                f"influent, so their parameters and temperature corrections "
                f"match -- two instances are silently inconsistent. (If they are "
                f"genuinely different models, pass an explicit translator=.)"
            )
        raise ValueError(
            f"Connection {src_label} -> {dst_label} crosses networks "
            f"('{getattr(source_network, 'name', '?')}' -> "
            f"'{getattr(target_network, 'name', '?')}'); supply an explicit "
            f"translator."
        )

    def _finalize_topology(self) -> None:
        """Compute the RHS evaluation order and the recycle (back-edge) set.

        Topologically sorts the feed-forward connection graph so units can be
        added in any order. A connection carrying an explicit ``initial_value``
        is treated as a declared recycle and cut first (so its seed sits on the
        cut edge); any cycle remaining in the rest is broken deterministically --
        Kahn's algorithm with an insertion-order tie-break, and when it stalls
        the earliest-added remaining unit has its still-active incoming edges cut
        (those become auto-detected recycles, zero-flow seeded). Sets
        :attr:`_unit_order`, :attr:`_recycle_keys`, :attr:`_recycle_conns` and the
        zero-flow :attr:`_recycle_seeds`. Deterministic, so the state/parameter
        layouts are stable, and for a plant whose units were already added in a
        valid evaluation order it reproduces that order and recycle set exactly.
        """
        units = self._insertion_order
        pos = {u: i for i, u in enumerate(units)}
        unit_conns = [c for c in self.connections if c.from_unit is not None]
        forced_ids = {id(c) for c in unit_conns if c.initial_value is not None}

        incoming = {u: [] for u in units}
        succ = {u: [] for u in units}
        for c in unit_conns:
            if id(c) not in forced_ids:
                incoming[c.to_unit].append(c)
                succ[c.from_unit].append(c)
        indeg = {u: len(incoming[u]) for u in units}

        remaining = set(units)
        cut_ids: set[int] = set()
        order: list[str] = []
        while remaining:
            ready = sorted((u for u in remaining if indeg[u] == 0), key=pos.get)
            if not ready:
                # Cycle: break it at the earliest-added remaining unit by cutting
                # its still-active incoming edges (sources not yet emitted).
                node = min(remaining, key=pos.get)
                for c in incoming[node]:
                    if c.from_unit in remaining and id(c) not in cut_ids:
                        cut_ids.add(id(c))
                        indeg[node] -= 1
                ready = [node]
            node = ready[0]
            order.append(node)
            remaining.discard(node)
            for c in succ[node]:
                if id(c) not in cut_ids and c.to_unit in remaining:
                    indeg[c.to_unit] -= 1

        self._unit_order = order
        forced = [c for c in unit_conns if id(c) in forced_ids]
        auto_cut = [c for c in unit_conns if id(c) in cut_ids]
        self._recycle_conns = forced + auto_cut
        self._recycle_keys = [(c.from_unit, c.from_port) for c in self._recycle_conns]
        self._recycle_seeds = {}
        for c in auto_cut:
            net = self.units[c.from_unit].network
            self._recycle_seeds[(c.from_unit, c.from_port)] = Stream(
                Q=jnp.asarray(0.0), C=net.default_concentrations(), network=net
            )

    def _unit_network(self, unit_name: str) -> CompiledNetwork:
        unit = self.units[unit_name]
        if not hasattr(unit, "network"):
            raise KeyError(
                f"Unit '{unit_name}' has no 'network' attribute; cannot infer translator default."
            )
        return unit.network

    # ----- layout ----------------------------------------------------------

    def _materialize_aeration(self) -> None:
        """Auto-wire a PI controller for every closed-loop ``Aeration`` spec.

        A ``CSTRUnit`` whose ``aeration`` carries a ``do_setpoint`` reads a kLa
        control signal in its ``rhs`` but does not itself supply the controller.
        This creates the missing controllers from the units' specs: tanks are
        grouped by their aeration ``controller`` id (the shared-controller case)
        or, when none is given, each gets its own (per-tank control). For each
        group one :class:`~aquakin.plant.control.PIController` is added -- sensing
        the group's ``sensor`` unit, publishing the signal the tanks consume --
        and the sensor is connected to it. Runs once (idempotent) at topology
        setup, before the state layout is assigned, so the controllers get state
        slices and the signal-bus validation sees them.

        Tanks sharing a controller must agree on its setpoint, sensor and tuning;
        only the per-tank ``gain`` may differ.
        """
        if self._aeration_materialized:
            return
        self._aeration_materialized = True

        from aquakin.plant.control import PIController
        from aquakin.plant.cstr import _aeration_signal_name

        groups: dict[str, list[tuple[str, "Unit", object]]] = {}
        for name in list(self._insertion_order):
            aer = getattr(self.units[name], "aeration", None)
            if aer is None or not aer.is_closed_loop:
                continue
            groups.setdefault(aer.controller_id(name), []).append((name, self.units[name], aer))

        for cid, members in groups.items():
            first_name, first_unit, a0 = members[0]
            key = (
                a0.do_setpoint,
                a0.sensor,
                a0.species,
                a0.Kp,
                a0.Ti,
                a0.Tt,
                a0.kla_offset,
                a0.kla_min,
                a0.kla_max,
            )
            for nm, _u, a in members[1:]:
                if (
                    a.do_setpoint,
                    a.sensor,
                    a.species,
                    a.Kp,
                    a.Ti,
                    a.Tt,
                    a.kla_offset,
                    a.kla_min,
                    a.kla_max,
                ) != key:
                    raise ValueError(
                        f"CSTRUnits sharing aeration controller '{cid}' "
                        f"('{first_name}', '{nm}') must agree on its setpoint, "
                        f"sensor, species and PI tuning; only gain may differ."
                    )
            sensor = a0.sensor if a0.sensor is not None else first_name
            if sensor not in self.units:
                raise ValueError(
                    f"Aeration controller '{cid}' senses unit '{sensor}', which "
                    f"is not in the plant."
                )
            if not _senses_concentration(self.units[sensor]):
                raise ValueError(
                    f"Aeration controller '{cid}' senses unit '{sensor}', whose "
                    f"state is not a concentration vector. A DO sensor must read a "
                    f"reactor that carries the species concentrations in its state "
                    f"(e.g. a CSTRUnit), not a mixer/splitter/clarifier. Set "
                    f"sensor= to such a reactor."
                )
            # The controller unit takes the shared id as its name (so it is
            # referenceable); a per-tank controller gets a derived name that
            # cannot collide with the tank it controls.
            ctrl_name = a0.controller if a0.controller is not None else f"{first_name}_aeration"
            self.add_unit(
                PIController(
                    name=ctrl_name,
                    network=first_unit.network,
                    measured_species=a0.species,
                    setpoint=a0.do_setpoint,
                    Kp=a0.Kp,
                    Ti=a0.Ti,
                    Tt=a0.Tt,
                    offset=a0.kla_offset,
                    out_min=a0.kla_min,
                    out_max=a0.kla_max,
                    signal_name=_aeration_signal_name(cid),
                )
            )
            # Tap the sensor's first output port explicitly: the controller reads
            # the sensed value from the sensor's state (a reactor concentration),
            # so any output port carries it, but a bare endpoint is ambiguous for a
            # multi-output sensor (e.g. an MBR's permeate/waste).
            sensor_port = self.units[sensor].output_ports[0]
            self.connect(f"{sensor}.{sensor_port}", f"{ctrl_name}.measured")

    def _materialize_dosing(self) -> None:
        """Auto-wire a PI controller for every feedback :class:`DosingUnit`.

        A ``DosingUnit`` with a ``setpoint`` reads a dose-flow control signal in
        its ``compute_outputs`` but does not itself supply the controller. This
        creates the missing controllers from the units' specs -- the dosing
        analogue of :meth:`_materialize_aeration` -- grouping units by their
        ``controller`` id (shared controller) or giving each its own. For each
        group one :class:`~aquakin.plant.control.PIController` is added, sensing
        the group's ``sensor`` reactor and publishing the dose-flow signal the
        units consume, and the sensor is connected to it. Idempotent; runs at
        topology setup before the state layout is assigned.

        Units sharing a controller must agree on its setpoint, sensor, measured
        species and PI tuning; only the per-unit ``gain`` may differ.
        """
        if getattr(self, "_dosing_materialized", False):
            return
        self._dosing_materialized = True

        from aquakin.plant.control import PIController
        from aquakin.plant.dosing import DosingUnit, dose_signal_name

        groups: dict[str, list[tuple[str, "DosingUnit"]]] = {}
        for name in list(self._insertion_order):
            unit = self.units[name]
            if isinstance(unit, DosingUnit) and unit.is_closed_loop:
                groups.setdefault(unit.controller_id(), []).append((name, unit))

        for cid, members in groups.items():
            first_name, d0 = members[0]
            key = (
                d0.setpoint,
                d0.sensor,
                d0.measured_species,
                d0.Kp,
                d0.Ti,
                d0.Tt,
                d0.flow_offset,
                d0.flow_min,
                d0.flow_max,
            )
            for nm, d in members[1:]:
                if (
                    d.setpoint,
                    d.sensor,
                    d.measured_species,
                    d.Kp,
                    d.Ti,
                    d.Tt,
                    d.flow_offset,
                    d.flow_min,
                    d.flow_max,
                ) != key:
                    raise ValueError(
                        f"DosingUnits sharing controller '{cid}' ('{first_name}', "
                        f"'{nm}') must agree on setpoint, sensor, measured species "
                        f"and PI tuning; only gain may differ."
                    )
            if d0.sensor not in self.units:
                raise ValueError(
                    f"Dosing controller '{cid}' senses unit '{d0.sensor}', which "
                    f"is not in the plant."
                )
            if not _senses_concentration(self.units[d0.sensor]):
                raise ValueError(
                    f"Dosing controller '{cid}' senses unit '{d0.sensor}', whose "
                    f"state is not a concentration vector. A feedback dose must "
                    f"measure a reactor that carries the species concentrations in "
                    f"its state (e.g. a CSTRUnit), not a mixer/splitter/clarifier. "
                    f"Set sensor= to such a reactor."
                )
            self.add_unit(
                PIController(
                    name=cid,
                    network=self.units[d0.sensor].network,
                    measured_species=d0.measured_species,
                    setpoint=d0.setpoint,
                    Kp=d0.Kp,
                    Ti=d0.Ti,
                    Tt=d0.Tt,
                    offset=d0.flow_offset,
                    out_min=d0.flow_min,
                    out_max=d0.flow_max,
                    signal_name=dose_signal_name(cid),
                )
            )
            self.connect(d0.sensor, f"{cid}.measured")

    def _build_state_layout(self) -> None:
        """Assign each unit a contiguous slice of the flat state vector.

        First topologically sorts the units (:meth:`_finalize_topology`, which
        also derives the recycle set), then assigns the state slices in that
        evaluation order and (re)builds the input-adjacency cache. All depend
        only on the wiring, fixed by the time the plant is solved -- the single
        topology-setup entry point that :meth:`solve` and every direct-RHS caller
        invoke first.
        """
        self._materialize_aeration()
        self._materialize_dosing()
        self._finalize_topology()
        layout: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name in self._unit_order:
            size = self.units[name].state_size
            layout[name] = (cursor, size)
            cursor += size
        self._state_layout = layout
        # Append the temperature-state block at the tail (the FlowSetpoint
        # parameter-block pattern, but for state): every per-unit slice above
        # keeps its index, so warm-starts and states_by_unit are unaffected.
        self._temperature_units = self.temperature_model.tracked_units(self)
        self._temperature_volumes = jnp.asarray(
            [float(self.units[n].volume) for n in self._temperature_units], dtype=float
        )
        temp_size = self.temperature_model.state_size(self)
        self._temperature_block = (cursor, temp_size)
        cursor += temp_size
        self._total_state_size = cursor
        self._build_connection_index()
        self._validate_control_signals()

    def _validate_control_signals(self) -> None:
        """Cross-check the control-signal bus: every published signal name is
        unique, and every signal a unit *consumes* is *published* by some unit.

        A unit under closed-loop control (a ``CSTRUnit`` whose ``aeration`` has a
        DO setpoint) reads ``signals[name]`` in its ``rhs``; if no controller publishes
        ``name`` -- a forgotten or mistyped wiring -- that read is a bare
        ``KeyError`` from deep inside the first jitted solve. This runs at
        topology setup (before the RHS is traced) and raises a clear error
        naming the unit, the missing signal, and the available signals instead.

        Two controllers publishing the *same* signal name are also rejected here:
        the bus is gathered with ``dict.update`` (see :meth:`_compute_signals`),
        so a duplicate would silently overwrite -- whichever unit runs later wins
        and the other's output is discarded while its integral state keeps
        winding. That is caught up front as a clear error rather than a silent
        wrong closed loop.

        Conservative by design: a unit that publishes signals declares their
        names via ``signal_names``; if any signal *producer* (one exposing
        ``signal_outputs``) does not declare ``signal_names``, the published set
        is unknown, so validation is skipped rather than risk rejecting a valid
        plant.
        """
        unknown_publisher = any(
            hasattr(u, "signal_outputs") and not hasattr(u, "signal_names")
            for u in self.units.values()
        )
        if unknown_publisher:
            return
        published: set[str] = set()
        publisher_of: dict[str, str] = {}
        for name, unit in self.units.items():
            for sig in getattr(unit, "signal_names", ()) or ():
                if sig in publisher_of:
                    raise ValueError(
                        f"Control signal '{sig}' is published by both "
                        f"'{publisher_of[sig]}' and '{name}'. Signal names must be "
                        f"unique -- the bus gathers them by name, so a duplicate "
                        f"would silently overwrite. Give the controllers distinct "
                        f"signal names."
                    )
                publisher_of[sig] = name
                published.add(sig)
        for name, unit in self.units.items():
            for sig in getattr(unit, "required_signals", ()) or ():
                if sig not in published:
                    raise ValueError(
                        f"Unit '{name}' consumes control signal '{sig}', which "
                        f"no unit in this plant publishes. Published signals: "
                        f"{sorted(published) or '(none)'}. Add a controller "
                        f"(e.g. a PIController) that publishes '{sig}', or fix "
                        f"the signal name."
                    )

    def _build_connection_index(self) -> None:
        """Group connections by destination unit for the RHS hot paths.

        ``_inputs_by_unit`` lets :meth:`_collect_inputs` and :meth:`_resolve_flows`
        look up a unit's incoming edges in O(in-degree) instead of re-scanning all
        connections (the scan ran ``recycle_passes x n_units`` times per stream
        sweep plus once per unit in the derivative pass). The recycle set is
        computed separately by :meth:`_finalize_topology`.
        """
        inputs_by_unit: dict[str, list[Connection]] = {name: [] for name in self._unit_order}
        for conn in self.connections:
            inputs_by_unit.setdefault(conn.to_unit, []).append(conn)
            # A pH-feedback translator on the source side (needs_src_pH, e.g.
            # ADM->ASM) reads the source unit's state-derived pH. An external
            # influent has no source unit (from_unit is None) and no state, so the
            # feedback would silently fall back to the fixed pH_adm. Not reachable
            # in shipped plants (the digester is always a real source unit); warn
            # if it is ever wired that way rather than failing silently.
            if conn.from_unit is None and getattr(conn.translator, "needs_src_pH", False):
                warnings.warn(
                    f"Translator on the influent edge into '{conn.to_unit}."
                    f"{conn.to_port}' declares needs_src_pH but has no source "
                    f"unit (it is fed by an external influent), so its pH "
                    f"feedback falls back to the fixed pH_adm. Feed it from the "
                    f"pH-bearing unit (e.g. the digester) for state-derived pH.",
                    stacklevel=2,
                )
        self._inputs_by_unit = inputs_by_unit

    def _ordered_networks(self) -> list[CompiledNetwork]:
        """The distinct kinetic networks, in first-appearance order.

        Deduplicated by object **identity** (two units sharing the same
        compiled network see one block). Parameter blocks are keyed by
        ``network.name``, so two *distinct* networks sharing a name would
        collide; reject that here rather than silently mis-slice the
        parameter vector. This is the single source of truth for both
        :meth:`_build_parameter_layout` and :meth:`default_parameters`, so the
        layout and the default vector can never disagree.
        """
        seen: list[CompiledNetwork] = []
        for name in self._insertion_order:
            net = getattr(self.units[name], "network", None)
            if net is None:
                continue
            if any(s is net for s in seen):
                continue
            if any(s.name == net.name for s in seen):
                raise ValueError(
                    f"Two distinct networks share the name '{net.name}'. "
                    f"Parameter blocks are keyed by network name, so names "
                    f"must be unique across the plant's networks."
                )
            seen.append(net)
        return seen

    @property
    def time_unit(self) -> Optional[str]:
        """The plant's native integration time unit, or ``None``.

        ``t_span`` / ``t_eval`` passed to :meth:`solve` are in this unit (unless
        a ``time_unit=`` override is given). It is the unit shared by every
        kinetic network's rate constants (:attr:`CompiledNetwork.time_unit`) --
        ``"d"`` for the BSM-family plants (ASM1 + ADM1 are both in days).
        Returns ``None`` if the plant has no kinetic network, or its networks
        disagree on (or do not declare) a time unit -- in which case a
        ``time_unit=`` conversion in :meth:`solve` cannot be applied.
        """
        units = {net.time_unit for net in self._ordered_networks()}
        if len(units) == 1:
            return next(iter(units))
        return None

    def _build_parameter_layout(self) -> ParameterLayout:
        """Concatenate kinetic-parameter vectors for every distinct network.

        Each network contributes one block, in the order networks first
        appear via the unit-list iteration.
        """
        blocks: dict[str, tuple[int, int]] = {}
        cursor = 0
        net_index: dict[str, int] = {}
        for i, net in enumerate(self._ordered_networks()):
            blocks[net.name] = (cursor, net.n_params)
            net_index[net.name] = i
            cursor += net.n_params

        # Flow-setpoint blocks: one per unit carrying differentiable flow
        # setpoints, appended after the kinetic blocks (so kinetic indices are
        # unchanged). Each unit's setpoint defaults seed these slots; the unit
        # reads the live values from the tail of its ``params_unit`` slice.
        flow_blocks: dict[str, tuple[int, int]] = {}
        for name in self._unit_order:
            defaults = self._unit_flow_defaults(self.units[name])
            if defaults:
                flow_blocks[name] = (cursor, len(defaults))
                cursor += len(defaults)

        self._network_param_index = net_index
        self._parameter_layout = ParameterLayout(
            network_param_blocks=blocks,
            unit_flow_blocks=flow_blocks,
            total_size=cursor,
        )
        return self._parameter_layout

    @staticmethod
    def _unit_flow_defaults(unit) -> list[float]:
        """A flow-bearing unit's ordered setpoint defaults (``[]`` otherwise)."""
        fn = getattr(unit, "flow_param_defaults", None)
        return list(fn()) if fn is not None else []

    def _kinetic_param_size(self) -> int:
        """Total length of the kinetic (network) parameter blocks."""
        layout = self._parameter_layout
        return sum(size for _, size in layout.network_param_blocks.values())

    def _coerce_params(self, params: jnp.ndarray) -> jnp.ndarray:
        """Accept a full parameter vector, or a kinetic-only one (padded).

        The full vector is ``[kinetic networks | flow setpoints]``. A vector of
        only the kinetic length -- e.g. one built by concatenating the network
        defaults, the convention before flow setpoints became parameters -- is
        padded with the default flow setpoints, so existing parameter vectors
        keep working unchanged.
        """
        self._build_parameter_layout()
        params = jnp.asarray(params)
        total = self._parameter_layout.total_size
        kinetic = self._kinetic_param_size()
        if params.shape == (total,):
            return params
        if params.shape == (kinetic,) and total > kinetic:
            flow_defaults = self.default_parameters()[kinetic:]
            return jnp.concatenate([params, flow_defaults])
        raise ValueError(
            f"params has shape {params.shape}, expected ({total},) "
            f"(or the kinetic-only ({kinetic},), padded with flow setpoints)."
        )

    def default_parameters(self) -> jnp.ndarray:
        """Concatenated default parameters: kinetic networks then flow setpoints."""
        nets = self._ordered_networks()
        kinetic = (
            jnp.concatenate([net.default_parameters() for net in nets]) if nets else jnp.zeros((0,))
        )
        flow_defaults: list[float] = []
        for name in self._unit_order:
            flow_defaults.extend(self._unit_flow_defaults(self.units[name]))
        if flow_defaults:
            return jnp.concatenate([kinetic, jnp.asarray(flow_defaults, dtype=float)])
        return kinetic

    def _plant_param_index(self) -> dict[str, int]:
        """Map ``"<network>.<param>"`` -> index in the flat plant parameter vector.

        Built from the per-network parameter blocks (:attr:`network_param_blocks`)
        and each network's own ``param_index``. The key prefixes a network's
        namespaced parameter name with the network name, so the same flat vector
        the integrator uses gets a friendly by-name address.
        """
        self._build_parameter_layout()
        blocks = self._parameter_layout.network_param_blocks
        index: dict[str, int] = {}
        for net in self._ordered_networks():
            start, _ = blocks[net.name]
            for pname, local in net.param_index.items():
                index[f"{net.name}.{pname}"] = start + local
        # Flow setpoints are addressed ``"<unit>.<setpoint>"`` (e.g.
        # ``"underflow_split.ras"``), the differentiable design-variable knobs.
        for name, (fstart, _) in self._parameter_layout.unit_flow_blocks.items():
            local_names = self.units[name].flow_param_local_names()
            for j, local_name in enumerate(local_names):
                index[f"{name}.{local_name}"] = fstart + j
        return index

    def parameter_names(self) -> list[str]:
        """The plant's calibratable parameter names, ``"<network>.<param>"``.

        Each kinetic network contributes its (namespaced) parameter names
        prefixed by the network name -- e.g. ``"asm1.muH"`` or
        ``"adm1.k_hyd_ch"`` -- which are the keys accepted by
        :meth:`parameter_values` / :meth:`parameter_index`.
        """
        return list(self._plant_param_index())

    def parameter_index(self, name: str) -> int:
        """Flat index of ``name`` (``"<network>.<param>"``) in the parameter vector.

        The companion to :meth:`parameter_values` for code that needs the
        *position* rather than a new vector -- e.g. ``jax.grad`` with respect to
        one parameter, which can't go through :meth:`parameter_values` (that
        materialises concrete values) -- without hand-computing the network
        block offset. Raises ``KeyError`` with a close-match hint for an unknown
        name.

        Examples
        --------
        >>> gidx = plant.parameter_index("adm1.k_m_ac")
        >>> grad = jax.grad(lambda th: f(params.at[gidx].set(th)))(theta0)
        """
        index = self._plant_param_index()
        if name not in index:
            suffix = did_you_mean(name, list(index))
            raise KeyError(
                f"Unknown plant parameter '{name}'. Keys are "
                f"'<network>.<param>' (see plant.parameter_names()).{suffix}"
            )
        return index[name]

    def parameter_values(self, overrides: Optional[dict] = None, /) -> jnp.ndarray:
        """Plant parameter vector: the defaults with named entries overridden.

        The plant analogue of ``CompiledNetwork.parameter_values``. Keys are
        ``"<network>.<param>"`` -- the network name plus the network's own
        (namespaced) parameter name -- so you can bump one rate in a
        multi-network plant (e.g. BSM2's ASM1 water line + ADM1 digester)
        without hunting the block offset and index by hand.

        Parameters
        ----------
        overrides : dict[str, float], optional
            Map of ``"<network>.<param>"`` to a new value. Unknown names raise a
            ``KeyError`` with a close-match hint. ``None`` returns the defaults.

        Returns
        -------
        jnp.ndarray
            The flat parameter vector to pass to :meth:`solve`.

        Examples
        --------
        >>> params = plant.parameter_values({"asm1.muH": 4.0, "adm1.k_hyd": 10.0})
        >>> plant.solve(t_span=(0.0, 200.0), params=params)

        See Also
        --------
        parameter_names : the valid keys.
        """
        base = self.default_parameters()
        if overrides is None:
            return base
        if not isinstance(overrides, dict):
            raise TypeError(
                "overrides must be a dict of '<network>.<param>' -> value; got "
                f"{type(overrides).__name__}."
            )
        if not overrides:
            return base
        index = self._plant_param_index()
        idxs, vals = [], []
        for name, value in overrides.items():
            if name not in index:
                suffix = did_you_mean(name, list(index))
                raise KeyError(
                    f"Unknown plant parameter '{name}'. Keys are "
                    f"'<network>.<param>' (see plant.parameter_names()).{suffix}"
                )
            idxs.append(index[name])
            vals.append(float(value))
        return base.at[jnp.asarray(idxs)].set(jnp.asarray(vals, dtype=base.dtype))

    # ----- introspection: discover unit / port / species names ---------------

    def _unit_or_raise(self, name: str) -> "Unit":
        """Return ``self.units[name]`` or a ``KeyError`` with a close-match hint."""
        if name not in self.units:
            suffix = did_you_mean(name, list(self.units))
            raise KeyError(
                f"Unknown unit '{name}'. Units (see plant.list_units()): "
                f"{self.list_units()}.{suffix}"
            )
        return self.units[name]

    def list_units(self) -> list[str]:
        """The plant's unit names, in the order added.

        Each name keys :attr:`units`, is the ``unit`` part of the ``"unit.port"``
        endpoints :meth:`stream` / :meth:`connect` accept, and is a key of
        :meth:`states_by_unit` / :meth:`PlantSolution.unit_state`. See
        :meth:`list_ports` for the ports and :meth:`list_species` for a kinetic
        unit's species. Available before solving (it is plant structure).
        """
        return list(self._insertion_order)

    def list_ports(self, unit: Optional[str] = None, *, role: str = "output") -> list[str]:
        """The ``"unit.port"`` endpoint strings, for discovering stream args.

        ``role="output"`` (default) returns every unit-*output* endpoint -- the
        strings :meth:`stream` reconstructs and :meth:`connect` reads from as a
        source. ``role="input"`` returns the input endpoints (:meth:`connect`
        destinations). Pass ``unit`` to restrict to one unit. No reading the
        builder source to find a port string.
        """
        if role not in ("output", "input"):
            raise ValueError(f"role must be 'output' or 'input'; got {role!r}.")
        names = [unit] if unit is not None else list(self._insertion_order)
        out: list[str] = []
        for name in names:
            u = self._unit_or_raise(name)
            ports = u.output_ports if role == "output" else u.input_ports
            out.extend(f"{name}.{p}" for p in ports)
        return out

    def activated_sludge_reactors(self, *, require_volume: bool = True) -> list[str]:
        """The activated-sludge reactor units (CSTR / MBR), in plant order.

        Identified by the CSTR-only ``aeration`` attribute -- the digester and
        the other volumed units lack it. ``require_volume`` (the default)
        additionally requires a ``volume`` field: warm-starts and sizing need
        the volume, whereas the mixing-energy term wants every mechanically
        mixed reactor regardless. The single source of truth behind the
        warm-start / design / evaluation reactor heuristics.
        """
        names = []
        for name in self._unit_order:
            unit = self.units[name]
            if not hasattr(unit, "aeration"):
                continue
            if require_volume and not hasattr(unit, "volume"):
                continue
            names.append(name)
        return names

    @staticmethod
    def _is_concentration_unit(unit) -> bool:
        """Whether a unit's state *is* its network's concentration vector (so it
        can be indexed by species: a CSTR or the digester, not a stateless
        mixer/splitter/ideal-clarifier nor the layered Takacs settler)."""
        return hasattr(unit, "network") and getattr(unit, "state_size", 0) == unit.network.n_species

    def list_species(self, unit: str) -> list[str]:
        """The species names of a concentration-vector unit's network.

        These are the valid names for :meth:`PlantSolution.C_named` and
        :meth:`PlantSolution.to_dataframe` on that unit. Raises ``KeyError`` for
        an unknown unit (with a hint), or for a unit whose state is not a
        concentration vector (a mixer/splitter/clarifier) -- use
        :meth:`stream` to read those as flow streams instead.
        """
        u = self._unit_or_raise(unit)
        if not self._is_concentration_unit(u):
            indexable = [
                n for n in self._insertion_order if self._is_concentration_unit(self.units[n])
            ]
            raise KeyError(
                f"Unit '{unit}' state is not a concentration vector, so it has "
                f"no per-species columns (read it as a stream with "
                f"plant.stream(sol, ...) instead). Concentration units: "
                f"{indexable}."
            )
        return list(u.network.species)

    def _params_for_unit(self, unit_name: str, params_full: jnp.ndarray) -> jnp.ndarray:
        """Slice out one unit's parameters: its kinetic network block, then (for
        a flow-bearing unit) its appended flow-setpoint block.

        One RHS step calls this hundreds of times (the flow probe, the
        concentration probe, the output sweep, the signal bus and the dstate
        loop each re-request every unit's slice), always with the *same*
        ``params_full`` object and producing the *same* slice. So the per-unit
        slices are built once per distinct ``params_full`` and memoised: within
        a single (traced) RHS evaluation ``params_full`` is one object, so the
        map is built once and every later request is a dict lookup. Keyed by
        object identity (``is``), with a strong reference held, so an identity
        can never be reused for a different array while the map is live.
        """
        # The memo must never be stored on the long-lived ``self`` when
        # ``params_full`` is a JAX tracer: diffrax reuses a tracer's object
        # identity across its ``eqx.filter_eval_shape`` sub-trace and the real
        # trace, so a slice built in the sub-trace could be served in the outer
        # trace (a stale tracer), and the cached ``(params_full, built)`` tuple
        # outlives the trace -- ``jax.checking_leaks`` flags it as a leaked
        # tracer, the canonical UnexpectedTracerError antipattern. So only a
        # CONCRETE ``params_full`` is memoised on ``self`` (the eager forward
        # solve, the results-level stream reconstruction); under a trace the
        # slices are static-index slices that XLA constant-folds and CSE-dedupes
        # anyway, so recomputing them costs nothing and nothing is cached.
        if isinstance(params_full, jax.core.Tracer):
            return self._slice_unit_params(unit_name, params_full)
        cache = self.__dict__.get("_params_unit_cache")
        if cache is None or cache[0] is not params_full:
            built = {name: self._slice_unit_params(name, params_full) for name in self._unit_order}
            cache = (params_full, built)
            self._params_unit_cache = cache
        return cache[1][unit_name]

    def _slice_unit_params(self, unit_name: str, params_full: jnp.ndarray) -> jnp.ndarray:
        """Compute one unit's parameter slice (uncached; see
        :meth:`_params_for_unit`)."""
        unit = self.units[unit_name]
        net = getattr(unit, "network", None)
        if net is None:
            kinetic = jnp.zeros((0,))
        else:
            start, size = self._parameter_layout.network_param_blocks[net.name]
            # ``start``/``size`` are static Python ints, so a static slice lets
            # XLA constant-fold the index instead of emitting a dynamic_slice op.
            kinetic = params_full[start : start + size]
        flow_block = self._parameter_layout.unit_flow_blocks.get(unit_name)
        if flow_block is None:
            return kinetic
        fstart, fsize = flow_block
        # Read the unit's flow setpoints from its block -- unless the parameter
        # vector is the kinetic-only length (the pre-flow convention, or a
        # results-level reconstruction passing the same vector the run used): then
        # fall back to the unit's default setpoints (a slice would otherwise
        # silently clamp and read garbage). ``shape`` is static, so this branch
        # is resolved at trace time.
        if params_full.shape[0] >= fstart + fsize:
            flow = params_full[fstart : fstart + fsize]
        else:
            flow = jnp.asarray(
                self._unit_flow_defaults(self.units[unit_name]), dtype=params_full.dtype
            )
        return jnp.concatenate([kinetic, flow])

    def initial_state(self, overrides: Optional[dict[str, jnp.ndarray]] = None) -> jnp.ndarray:
        """Concatenated initial state from each unit's ``initial_state()``.

        Parameters
        ----------
        overrides : dict[str, array-like], optional
            Map of unit name to an initial-state vector that replaces that
            unit's own ``initial_state()``. Each vector must have length equal
            to the unit's ``state_size``. Use this to warm-start a plant --
            e.g. seed the activated-sludge reactors with a healthy biomass
            before settling a slow digester -- without reaching into the
            internal state layout::

                warm = asm1.concentrations(XB_H=2244.0, ...)   # (n_species,)
                y0 = plant.initial_state(overrides={t: warm for t in tanks})

        Returns
        -------
        jnp.ndarray
            The flat initial-state vector, shape ``(total_state_size,)``.

        Raises
        ------
        KeyError
            If an override names a unit not in the plant.
        ValueError
            If an override vector length does not match the unit's state size.
        """
        self._build_state_layout()
        if self._total_state_size == 0:
            return jnp.zeros((0,))
        overrides = overrides or {}
        unknown = set(overrides) - set(self.units)
        if unknown:
            raise KeyError(
                f"initial_state overrides name unknown units {sorted(unknown)}; "
                f"valid units are {sorted(self.units)}."
            )
        pieces: list[jnp.ndarray] = []
        for name in self._unit_order:
            if name in overrides:
                vec = jnp.asarray(overrides[name])
                expected = self.units[name].state_size
                if vec.shape != (expected,):
                    raise ValueError(
                        f"initial_state override for unit '{name}' has shape "
                        f"{vec.shape}, expected ({expected},)."
                    )
                pieces.append(vec)
            else:
                pieces.append(self.units[name].initial_state())
        if self._temperature_block[1]:
            pieces.append(self.temperature_model.initial_state(self))
        return jnp.concatenate(pieces)

    def states_by_unit(self, state_full: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Split a flat plant vector into a ``{unit_name: sub-vector}`` map.

        The inverse of :meth:`initial_state` with ``overrides``: that assembles
        a flat vector from per-unit pieces, this splits one back apart. Works on
        any flat plant vector in the plant's state layout -- an initial state, a
        solution snapshot (:attr:`PlantSolution.final_state`), or a derivative
        from :meth:`derivative`::

            dig = plant.states_by_unit(sol.final_state)["digester"]
            rate = plant.states_by_unit(plant.derivative(y0))["tank5"]

        Parameters
        ----------
        state_full : jnp.ndarray
            A flat plant vector, shape ``(total_state_size,)``.

        Returns
        -------
        dict
            ``{unit_name: sub-vector}``, each of shape ``(unit.state_size,)`` in
            the unit-addition order.
        """
        self._build_state_layout()
        return self._split_state(jnp.asarray(state_full))

    # ----- RHS -------------------------------------------------------------

    def _split_state(self, state_full: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Split the flat state vector into a per-unit ``{name: state}`` map.

        The appended temperature-state block (if any) is exposed under the
        reserved :data:`_TEMPERATURE_KEY`, not a real unit name -- ``_unit_order``
        does not contain it, so it is never iterated as a unit.
        """
        states: dict[str, jnp.ndarray] = {}
        for name in self._unit_order:
            start, size = self._state_layout[name]
            # ``start``/``size`` are static Python ints; a static slice lets XLA
            # constant-fold the index rather than emit a dynamic_slice op.
            states[name] = state_full[start : start + size]
        tstart, tsize = self._temperature_block
        if tsize:
            states[_TEMPERATURE_KEY] = state_full[tstart : tstart + tsize]
        return states

    def _temperatures_by_unit(self, states: dict[str, jnp.ndarray]) -> dict:
        """Map each tracked unit to its current temperature state value.

        Empty for the default :class:`AlgebraicTemperature` (no tracked units)
        and when the influent carries no temperature (the channel is inactive,
        so overriding an outlet to a number would be inconsistent with the
        all-``None`` temperature field). When non-empty, it drives both the
        outlet-temperature override in :meth:`_sweep_outputs` and the operating
        temperature a reactor reads in its ``rhs``.
        """
        temp_state = states.get(_TEMPERATURE_KEY)
        if temp_state is None or not self._temperature_units:
            return {}
        carries_T = any(getattr(s, "T", None) is not None for s in self.influents.values())
        if not carries_T:
            return {}
        return {name: temp_state[i] for i, name in enumerate(self._temperature_units)}

    def _resolve_streams(
        self,
        t: jnp.ndarray,
        states: dict[str, jnp.ndarray],
        params_full: jnp.ndarray,
        signals: Optional[dict] = None,
        design: Optional[dict] = None,
        recycle_map: Optional[list] = None,
        flow_map: Optional[jnp.ndarray] = None,
    ) -> tuple[
        dict[tuple[Optional[str], str], Stream],
        dict[tuple[str, str], Stream],
    ]:
        """Resolve every unit's output streams at one instant.

        Builds the influent + recycle-seed registry, resolves the (exact,
        concentration-decoupled) recycle flow network, and runs the fixed-pass
        Gauss-Seidel output sweep. Returns ``(all_outputs, influent_streams)``;
        the second is needed by :meth:`_collect_inputs`.

        ``design["influent"]`` (optional) maps an influent port name to a
        ``{"Q": ..., "C": ..., "T": ...}`` dict of differentiable arrays, used
        instead of the recorded :class:`InfluentSeries` at ``t`` (``T`` optional)
        -- the hook that makes a steady-state / sweep differentiable w.r.t. the
        influent load. (Plain arrays, not a :class:`Stream`, because the Stream
        carries a non-JAX ``network`` field and so cannot be a differentiated
        leaf; the network is supplied here from the recorded influent.)
        """
        # The control-signal bus is computed from the states up front (controllers
        # sense reactor states), so it is available to every unit's
        # compute_outputs during the sweep -- a feedback-dosing unit reads its
        # dose flow there. Callers other than _rhs (outputs_at, the steady-state
        # solve) leave it None and it is computed here; _rhs passes the bus it
        # also reuses for the dstate pass.
        if signals is None:
            signals = self._compute_signals(t, states, params_full)
        influent_override = (design or {}).get("influent", {})
        streams: dict[tuple[Optional[str], str], Stream] = {}
        for port_name, series in self.influents.items():
            if port_name in influent_override:
                ov = influent_override[port_name]
                base = series.at(t)
                # An influent override may be an ABSOLUTE replacement (a constant
                # design, e.g. the steady-state IFT samples the influent at one
                # time and overrides it) and/or a MULTIPLICATIVE scale on the
                # time-sampled series (a dynamic operating-parameter sensitivity:
                # scale the flow or a species' load). Both default to identity, so
                # an absent key / a scale of 1.0 leaves the recorded influent
                # unchanged. ``C_scale`` may be a scalar or a per-species array.
                Q = ov.get("Q", base.Q) * ov.get("Q_scale", 1.0)
                C = ov.get("C", base.C) * ov.get("C_scale", 1.0)
                streams[(None, port_name)] = Stream(
                    Q=Q, C=C, network=series.network, T=ov.get("T", base.T)
                )
            else:
                streams[(None, port_name)] = series.at(t)
        resolved_flows = self._recycle._resolve_flows(
            t, params_full, states, design=design, flow_map=flow_map
        )
        # Seed the recycle back-edges with their exact (affine) fixed point, so
        # the Gauss-Seidel mop-up starts at the answer for any linear topology
        # (gain-independent); it then only refines a genuinely non-affine
        # in-cycle unit, if any. Falls back to the zero auto-seed when there are
        # no recycle edges.
        seeded = self._recycle._resolve_recycle_concentrations(
            t, states, params_full, resolved_flows, signals, recycle_map=recycle_map
        )
        all_outputs = self._sweep_outputs(t, states, streams, seeded, params_full, signals=signals)
        return all_outputs, streams

    def outputs_at(
        self,
        t: jnp.ndarray,
        state_full: jnp.ndarray,
        params: Optional[jnp.ndarray] = None,
    ) -> dict[tuple[str, str], Stream]:
        """Reconstruct every unit's output streams at one ``(t, state)``.

        The plant integrates unit *states*, not the inter-unit streams, so a
        stream such as the clarifier effluent is recomputed from the state on
        demand. This resolves the full output sweep (including recycle edges)
        and returns the ``{(unit, port): Stream}`` map. See :meth:`stream` for
        a named-stream trajectory over a whole solution.

        Parameters
        ----------
        t : float
            Time (plant units), used to interpolate the influents.
        state_full : jnp.ndarray
            Flat plant state at ``t`` (e.g. one row of ``PlantSolution.state``).
        params : jnp.ndarray, optional
            Plant parameters (defaults to :meth:`default_parameters`).

        Returns
        -------
        dict
            ``{(unit_name, port_name): Stream}`` for every unit output.
        """
        self._build_state_layout()
        self._build_parameter_layout()
        params_full = self.default_parameters() if params is None else self._coerce_params(params)
        states = self._split_state(jnp.asarray(state_full))
        recycle_map = self._recycle._maybe_recycle_map(jnp.asarray(t), states, params_full)
        flow_map = self._recycle._maybe_flow_map(jnp.asarray(t), states, params_full)
        all_outputs, _ = self._resolve_streams(
            jnp.asarray(t), states, params_full, recycle_map=recycle_map, flow_map=flow_map
        )
        return all_outputs

    def _cached_streams(self, solution: "PlantSolution", params_full):
        """Reconstruct **every** unit-output stream over a whole solution, once.

        The plant integrates unit *states*, not the inter-unit streams, so each
        stream is recomputed from the saved states. This does that for all
        ``(unit, port)`` outputs in a single pass over the saved times, and
        **caches** the result on the solution keyed by the parameter vector --
        so a sequence of ``plant.stream(sol, ...)`` calls (effluent, RAS,
        wastage, ...) costs one reconstruction, not one per stream.

        Returns ``{(unit, port): (Q (n_t,), C (n_t, n_species))}``. Under tracing
        (a differentiated/jitted solve) the parameter vector cannot be
        materialised into a cache key, so the pass runs uncached.
        """
        key, concrete = _concrete_teval_key(params_full)
        cache = solution.__dict__.setdefault("_stream_cache", {})
        if concrete and key in cache:
            return cache[key]
        self._build_state_layout()
        self._build_parameter_layout()
        ts = jnp.asarray(solution.t)
        states_flat = jnp.asarray(solution.state)

        # The recycle concentration map M is state-invariant for a fixed-pump
        # plant, so precompute it once (from the first saved state) and reuse it
        # across every reconstructed time -- skipping the per-time per-edge probe
        # sweeps, exactly as the forward RHS does. None when the map is
        # state-coupled (the resolver then probes per call). Built from
        # params_full so a differentiated reconstruction still flows the gradient
        # through M (the vmap below is traced when params_full is a tracer).
        recycle_map = self._recycle._maybe_recycle_map(
            ts[0], self._split_state(states_flat[0]), params_full
        )
        flow_map = self._recycle._maybe_flow_map(
            ts[0], self._split_state(states_flat[0]), params_full
        )

        # Reconstruct every (unit, port) output stream at one saved time, keeping
        # only the (Q, C) arrays (the Stream's `network` is a static, non-JAX
        # field). vmap then batches this over all saved times in a single XLA
        # program -- one vectorised sweep instead of a Python loop of per-step
        # sweeps (each a recycle-flow + concentration solve), which dominated
        # the cost of evaluating a long dynamic run.
        def _one(t_i, state_row):
            states = self._split_state(state_row)
            outs, _ = self._resolve_streams(
                t_i, states, params_full, recycle_map=recycle_map, flow_map=flow_map
            )
            return {k: (s.Q, s.C) for k, s in outs.items()}

        result = jax.vmap(_one)(ts, states_flat)
        if concrete:
            cache[key] = result
        return result

    def stream(
        self,
        solution: "PlantSolution",
        endpoint: str,
        params: Optional[jnp.ndarray] = None,
    ) -> StreamSeries:
        """Reconstruct a named output stream's trajectory from a solution.

        **The plant integrates unit states, not the inter-unit streams**, so a
        stream such as the secondary-clarifier effluent is *not stored* -- it is
        recomputed (flow + concentration over time) from the saved states on
        demand. The whole output sweep is reconstructed once and **cached on the
        solution** (see :meth:`_cached_streams`), so a sequence of ``stream``
        calls for different ports reuses one reconstruction.

        Parameters
        ----------
        solution : PlantSolution
            A solution returned by :meth:`solve` (carries ``t`` and ``state``).
        endpoint : str
            A registered **semantic name** (``"effluent"``, ``"ras"``,
            ``"internal_recycle"``, ``"primary_sludge"``, ``"reject"`` ... -- see
            :meth:`list_streams`) **or** a literal ``"unit.port"`` (e.g.
            ``"clarifier.overflow"``; the port may be omitted for a single-output
            unit). The semantic name is resolved to its port first, so an engineer
            asks for ``"effluent"`` without knowing the internal port.
        params : jnp.ndarray, optional
            The plant parameters used for the run (defaults to
            :meth:`default_parameters`). Output reconstruction is
            parameter-independent for the shipped units, so the default is
            usually fine.

        Returns
        -------
        StreamSeries
            ``t``, ``Q``, ``C`` (shape ``(n_t, n_species)``) and ``network``,
            with a ``C_named(species)`` accessor.
        """
        resolved = self.named_streams.get(endpoint)
        if resolved is None and "." not in endpoint and endpoint not in self.units:
            # Looks like a semantic name (no port, not a unit) but isn't
            # registered -> hint at the available ones rather than the bare
            # "Unknown unit" that _parse_endpoint would give.
            suffix = did_you_mean(endpoint, list(self.named_streams))
            raise KeyError(
                f"Unknown stream '{endpoint}'. Semantic names "
                f"(plant.list_streams()): {sorted(self.named_streams)}; or pass "
                f"a 'unit.port' (plant.list_ports()).{suffix}"
            )
        endpoint = resolved if resolved is not None else endpoint
        unit, port = self._parse_endpoint(endpoint, role="source")
        params_full = self.default_parameters() if params is None else self._coerce_params(params)
        Q, C = self._cached_streams(solution, params_full)[(unit, port)]
        org = self._reconstruct_stream_org(solution, (unit, port), params_full)
        return StreamSeries(t=solution.t, Q=Q, C=C, network=self.units[unit].network, org=org)

    def _reconstruct_stream_org(self, solution, key, params_full):
        """Reconstruct an output stream's indicator-organism trajectory, or
        ``None`` if the stream carries no indicator.

        The reconstruction cache keeps only ``(Q, C)``; the indicator density
        (carried like the temperature scalar) is surfaced here on demand for a
        disinfection effluent. Whether the stream carries an indicator is a static
        structural property, so it is decided once at the first saved state; an
        indicator-agnostic stream (every BSM stream with no disinfection unit)
        returns ``None`` and does no extra work."""
        states0 = self._split_state(jnp.asarray(solution.state)[0])
        probe, _ = self._resolve_streams(jnp.asarray(solution.t)[0], states0, params_full)
        if probe[key].org is None:
            return None

        def _org_one(t_i, state_row):
            outs, _ = self._resolve_streams(t_i, self._split_state(state_row), params_full)
            return outs[key].org

        return jax.vmap(_org_one)(jnp.asarray(solution.t), jnp.asarray(solution.state))

    def register_stream(self, name: str, endpoint: str) -> "Plant":
        """Register a **semantic name** for an output ``"unit.port"`` endpoint.

        After this, ``plant.stream(sol, name)`` reads that port -- so an engineer
        asks for ``"effluent"`` / ``"ras"`` / ``"digester_gas"`` instead of the
        internal port. Builders (``build_bsm1`` / ``build_bsm2``) register the
        plant's named streams; call this to add your own. Returns ``self`` for
        chaining.
        """
        self.named_streams[name] = endpoint
        return self

    def list_streams(self) -> dict[str, str]:
        """The registered semantic stream names ``{name: "unit.port"}``.

        The engineering shortcuts :meth:`stream` accepts (``"effluent"``,
        ``"ras"``, ...) and the internal port each resolves to -- the discoverable
        companion to :meth:`list_ports`, so you never read the builder source to
        find a port string. Empty on a plant whose builder registered none.
        """
        return dict(self.named_streams)

    def effluent_stream(
        self, solution: "PlantSolution", params: Optional[jnp.ndarray] = None
    ) -> StreamSeries:
        """The plant's final effluent stream (a shortcut for the most-read one).

        Reads the builder-recorded :attr:`effluent_endpoint` -- the right port
        even when an option moved it (a BSM2 influent bypass relocates the
        effluent to ``effluent_mix.out``). Equivalent to
        ``plant.stream(sol, plant.effluent_endpoint)``.
        """
        if self.effluent_endpoint is None:
            raise ValueError(
                "This plant has no recorded effluent_endpoint (its builder did "
                "not set one); pass an explicit endpoint to plant.stream(sol, ...)."
            )
        return self.stream(solution, self.effluent_endpoint, params)

    def digester_gas(self, solution: "PlantSolution", params: Optional[jnp.ndarray] = None):
        """The anaerobic digester's biogas trajectory.

        A :class:`~aquakin.plant.bsm.evaluation.DigesterGas` with the biogas flow
        ``Q`` (m³/d), the CH₄/CO₂/H₂ partial pressures and the CH₄ mass flow
        ``ch4`` (kg/d), plus ``.methane_production()`` (time-averaged kg CH₄/d).
        Unlike :meth:`stream`, the biogas is *derived* from the ADM1 headspace
        state (not a material port). Raises if the plant has no ADM1 digester.
        """
        from aquakin.plant.bsm.evaluation import digester_gas

        return digester_gas(self, solution, params)

    def mass_balance(
        self,
        solution: "PlantSolution",
        *,
        components=("COD", "N", "P"),
        influent_ports: Optional[list] = None,
        effluent_ports: Optional[list] = None,
        params: Optional[jnp.ndarray] = None,
    ):
        """Results-level mass-balance closure of this plant over a solved window.

        Accounts, per component (COD / N / P), the component that flowed **in**
        (influents), **out** (terminal material streams), left as **gas** (O₂
        transferred by aeration, digester biogas COD, denitrification N₂ + its
        oxidised COD) and **accumulated** (inventory change across every unit) --
        and reports the closure ``imbalance = in − out − gas − accumulation``,
        which is ~0 for a trustworthy result. The gas terms are computed
        independently of the in/out/accumulation bookkeeping (from the aeration
        term, the digester headspace and an activated-sludge reaction integral),
        so the imbalance is a genuine check rather than a definition.

        Everything is reported on one canonical gram basis (g COD / g N / g P),
        so inventories and fluxes sum across networks of different units (the ASM
        water line in g/m³, the ADM digester in kg/m³ and kmol/m³) via the
        shipped :func:`aquakin.composition_table`.

        Parameters
        ----------
        solution : PlantSolution
            A solved trajectory (the closure is over ``solution.t[0]..t[-1]``).
            The gas integrals are exact at steady state and otherwise accurate to
            the ``t_eval`` sampling.
        components : tuple of str, optional
            Which components to balance (default ``("COD", "N", "P")``); a network
            that does not carry one contributes nothing to it.
        influent_ports, effluent_ports : list of str, optional
            Override the boundary streams. By default the influents are every
            registered influent and the effluents are the plant's terminal
            (dangling) output ports (final effluent, wasted sludge, disposal
            cake) -- the biogas is handled as a gas term, not a material port.
        params : jnp.ndarray, optional
            Plant parameters used for the run (defaults to
            :meth:`default_parameters`).

        Returns
        -------
        MassBalance
            Per-component :class:`~aquakin.plant.balance.ComponentBalance`
            (index by name, e.g. ``mb["N"]``), with ``.closed(rtol)`` and
            ``.summary()``.
        """
        from aquakin.plant.balance import mass_balance

        return mass_balance(
            self,
            solution,
            components=components,
            influent_ports=influent_ports,
            effluent_ports=effluent_ports,
            params=params,
        )

    def sludge_age(self, solution: "PlantSolution", params: Optional[jnp.ndarray] = None, **kwargs):
        """Achieved SRT / HRT / F:M of this activated-sludge plant.

        Convenience wrapper for :func:`aquakin.plant.design.sludge_metrics` --
        the design loop's closing half: the solids retention time (sludge age)
        is an emergent property of the wastage flow, so this reports what the
        solved model actually achieved instead of requiring it be guessed.

        Parameters
        ----------
        solution : PlantSolution
            A solution from :meth:`solve` (ideally near steady state).
        params : jnp.ndarray, optional
            Plant parameters used for the run.
        **kwargs
            Forwarded to :func:`~aquakin.plant.design.sludge_metrics`
            (``reactor_units``, ``influent_name``, ``effluent_port``,
            ``waste_port``, ``substrate``).

        Returns
        -------
        SludgeMetrics
            SRT, HRT, F:M and the intermediate inventories / loads.
        """
        # Lazy import: the design layer depends on plant types, so importing it
        # at module load would be circular.
        from aquakin.plant.design import sludge_metrics

        return sludge_metrics(self, solution, params, **kwargs)

    def derivative(
        self,
        state: jnp.ndarray,
        params: Optional[jnp.ndarray] = None,
        *,
        t: float = 0.0,
    ) -> jnp.ndarray:
        """Evaluate the assembled flowsheet RHS once: ``dstate/dt`` at ``state``.

        A public single evaluation of the same right-hand side :meth:`solve`
        integrates (recycles resolved by the fixed-pass sweep). Useful for
        inspecting the dynamics -- sign, magnitude, finiteness -- at a state
        without running a full solve. Split the result with
        :meth:`states_by_unit`::

            d = plant.derivative(y0, params)
            snh_rate = plant.states_by_unit(d)["tank5"][net.species_index["SNH"]]

        Parameters
        ----------
        state : jnp.ndarray
            Flat plant state, shape ``(total_state_size,)``.
        params : jnp.ndarray, optional
            Plant parameters (defaults to :meth:`default_parameters`).
        t : float, optional
            Time at which to evaluate, for time-varying influents (default 0).

        Returns
        -------
        jnp.ndarray
            ``dstate/dt``, shape ``(total_state_size,)`` -- the same layout as
            ``state``.
        """
        self._build_state_layout()
        self._build_parameter_layout()
        params_full = self.default_parameters() if params is None else self._coerce_params(params)
        return self._rhs(jnp.asarray(float(t)), jnp.asarray(state), params_full)

    def _rhs(
        self,
        t: jnp.ndarray,
        state_full: jnp.ndarray,
        params_full: jnp.ndarray,
        design: Optional[dict] = None,
        recycle_map: Optional[list] = None,
        flow_map: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Step 1: split state by unit.
        states = self._split_state(state_full)

        # Step 2: control signals, computed from the reactor states BEFORE the
        # sweep (see :meth:`_compute_signals`) so a feedback-dosing unit can read
        # its dose-flow signal while its output stream is being computed.
        signals = self._compute_signals(t, states, params_full)

        # Step 3: resolve influent + recycle streams and run the output sweep,
        # threading the signal bus into every unit's ``compute_outputs`` (a
        # dosing unit reads its dose flow there). ``design`` (optional) carries
        # differentiable design-variable overrides -- currently the influent
        # streams -- so a steady-state / sweep can take gradients w.r.t. them.
        all_outputs, streams = self._resolve_streams(
            t,
            states,
            params_full,
            signals=signals,
            design=design,
            recycle_map=recycle_map,
            flow_map=flow_map,
        )

        # Collect each unit's converged input streams ONCE for the dstate pass.
        # The streams are fixed after the sweep, so re-collecting per unit would
        # repeat the translator calls and Stream allocations on every RHS step --
        # and on the differentiated tape. Same values, so the result is
        # unchanged. (The signal bus is computed up front from the states, above,
        # so it does not reuse this map.)
        inputs_by_unit = {
            name: self._collect_inputs(name, all_outputs, streams, states, params_full)
            for name in self._unit_order
        }

        # Step 4: compute dstates from final input streams and the (always
        # passed) control-signal bus. Every unit's rhs has the same signature;
        # an uncontrolled unit ignores ``signals``. Under a heat-balance
        # temperature model, a reactor's operating temperature (its lagged tank
        # state) is threaded in through a reserved signal key, and the flow-
        # weighted inlet (Q, T) of each tracked unit is captured for the
        # temperature-state derivative.
        temp_by_unit = self._temperatures_by_unit(states)
        inlet_by_unit: dict = {}
        dstates: list[jnp.ndarray] = []
        for name in self._unit_order:
            unit = self.units[name]
            inputs = inputs_by_unit[name]
            params_unit = self._params_for_unit(name, params_full)
            unit_signals = signals
            if temp_by_unit:
                unit_signals = {**signals, _OPERATING_T_SIGNAL: temp_by_unit.get(name)}
                if name in temp_by_unit:
                    inlet_by_unit[name] = self._inlet_flow_temperature(inputs)
            dstate = unit.rhs(t, states[name], inputs, params_unit, unit_signals)
            dstates.append(dstate)
        if self._temperature_block[1]:
            dstates.append(
                self.temperature_model.state_rhs(self, states[_TEMPERATURE_KEY], inlet_by_unit)
            )
        return jnp.concatenate(dstates) if dstates else jnp.zeros((0,))

    @staticmethod
    def _inlet_flow_temperature(inputs: dict[str, Stream]):
        """Total inlet flow and flow-weighted inlet temperature of a unit.

        Returns ``(Q_in, T_in)``; ``T_in`` is ``None`` if any inlet is
        temperature-agnostic (the same gate the unit mixers use). The heat
        balance ``V dT/dt = Q_in (T_in - T)`` reads both.
        """
        Q_total = jnp.zeros(())
        heat = jnp.zeros(())
        have_T = True
        for s in inputs.values():
            Q_total = Q_total + s.Q
            if getattr(s, "T", None) is None:
                have_T = False
            else:
                heat = heat + s.Q * s.T
        if not have_T:
            return (Q_total, None)
        return (Q_total, heat / (Q_total + 1e-12))

    def _compute_signals(
        self,
        t: jnp.ndarray,
        states: dict[str, jnp.ndarray],
        params_full: jnp.ndarray,
    ) -> dict:
        """Gather the control-signal bus from every controller unit -- BEFORE the
        stream sweep.

        Units exposing ``signal_outputs`` (e.g. PI controllers) read a sensed
        reactor variable and their own state and write named scalar signals into
        a shared dict; that dict is threaded into every unit's ``compute_outputs``
        and ``rhs`` as ``signals``. A controller senses a reactor's
        concentration, which IS that unit's state, so the sensed value is taken
        directly from ``states`` and the bus is computed *before* the sweep. That
        ordering is what lets a feedback-dosing unit read its dose-flow signal in
        ``compute_outputs`` -- a dose changes the output stream, computed in the
        sweep, so the signal must already exist. Each controller's sensed
        ``inputs`` stream is reconstructed from the sensor unit's state
        (``C = state``); the sensor must therefore be a reactor whose output
        concentration is its state (a :class:`CSTRUnit`). Whether a unit produces
        signals is a class-level property, so the branch is static (jit-safe).
        """
        signals: dict = {}
        for name in self._unit_order:
            unit = self.units[name]
            if not hasattr(unit, "signal_outputs"):
                continue
            # Reconstruct each sensed source's stream from its state (a reactor's
            # output concentration is its state), so no swept stream is needed.
            sensed_outputs: dict[tuple[str, str], Stream] = {}
            for conn in self._inputs_by_unit.get(name, ()):
                if conn.from_unit is None:
                    continue
                sensed_outputs[(conn.from_unit, conn.from_port)] = Stream(
                    Q=jnp.zeros(()),
                    C=states[conn.from_unit],
                    network=self.units[conn.from_unit].network,
                )
            inputs = self._collect_inputs(name, sensed_outputs, {})
            signals.update(
                unit.signal_outputs(
                    t, states[name], inputs, self._params_for_unit(name, params_full)
                )
            )
        return signals

    def signals_at(
        self,
        t: jnp.ndarray,
        state_full: jnp.ndarray,
        params: Optional[jnp.ndarray] = None,
    ) -> dict:
        """Reconstruct the control-signal bus at one ``(t, state)``.

        The plant integrates unit states, not the control signals, so a signal
        such as a DO controller's manipulated ``kLa`` is recomputed from the
        state on demand -- the signal analogue of :meth:`outputs_at`. Returns
        ``{}`` for an open-loop plant (no controllers).

        Parameters
        ----------
        t : float
            Time (plant units), used to interpolate the influents.
        state_full : jnp.ndarray
            Flat plant state at ``t`` (e.g. one row of ``PlantSolution.state``).
        params : jnp.ndarray, optional
            Plant parameters (defaults to :meth:`default_parameters`).

        Returns
        -------
        dict
            ``{signal_name: scalar}`` for every published control signal.
        """
        self._build_state_layout()
        self._build_parameter_layout()
        params_full = self.default_parameters() if params is None else self._coerce_params(params)
        states = self._split_state(jnp.asarray(state_full))
        # The bus is computed from the reactor states alone (controllers sense
        # states), so no stream sweep is needed to reconstruct it.
        return self._compute_signals(jnp.asarray(t), states, params_full)

    def _structural_plant_pattern(self, coupling_mask=None) -> "np.ndarray":
        """Assemble the plant's structural Jacobian sparsity from each unit's
        emitted couplings, for the colored pattern (issue #388).

        The numerical probe at the start state captures the plant's **linear,
        always-on** couplings but drops every **nonlinear** coupling that is
        saturated at the warm-start operating point and only switches on once a
        dynamic influent drives the plant off it -- reaction kinetics (Monod / pH
        switches), the Takacs settling velocity, and the ASM<->ADM interface
        branches. Those are the stiff couplings, so a stale pattern wrecks the
        chord-Newton convergence (a ~6x slowdown).

        Each unit emits its own structural sparsity (:class:`CouplingAware`,
        :meth:`coupling_pattern`): a ``self`` block (d rhs / d own state) and an
        ``inlet`` block (d rhs / d inlet concentration). This assembler places the
        ``self`` blocks on the diagonal and composes each ``inlet`` block with the
        species coupling of the stream feeding it -- identity for a same-network
        feed, the translator's emitted ``coupling_pattern()`` for an ASM<->ADM
        feed -- to form the off-diagonal blocks, restricted to the unit pairs the
        probe shows actually coupled (the recycle's real reach). The result is a
        structural superset that cannot go stale for any influent. ``coupling_mask``
        is the probe pattern (used only to restrict off-diagonal placement to real
        couplings, keeping the coloring tight).
        """
        import numpy as np

        from aquakin.plant.translators import translator_coupling_pattern

        N = self._total_state_size
        P = np.zeros((N, N), dtype=bool)

        # Each unit emits its structural Jacobian sparsity (CouplingAware): a
        # self block (d rhs / d own state) and an inlet block (d rhs / d inlet
        # concentration). Reactors derive self from the rate AST (saturated Monod
        # terms are numerically invisible to a probe), the Takacs settler by AD
        # over diverse solids profiles, stateless units are empty.
        cps = {}
        for name, unit in self.units.items():
            fn = getattr(unit, "coupling_pattern", None)
            if fn is not None:
                cps[name] = (unit, fn())

        # Diagonal: each unit's self_pattern, placed on its own state block.
        for name, (unit, cp) in cps.items():
            sp = np.asarray(cp.self_pattern, dtype=bool)
            if sp.size == 0:
                continue
            off, _ = self._state_layout[name]
            k = sp.shape[0]
            P[off : off + k, off : off + k] |= sp

        # Off-diagonal: J[A][B] = inlet_pattern_A composed with the species
        # coupling of the stream feeding A from B. For a same-network feed that
        # coupling is the identity; for a cross-network feed (ASM<->ADM) it is the
        # translator's emitted pattern. The source unit B's *output* is linear in
        # its state (a reactor outputs its state; the settler reads a layer), so B
        # ranges over the concentration units (output == state) and the only
        # nonlinear, stale parts are A's inlet response and the translator -- both
        # captured here. Placement is restricted to the unit pairs the probe shows
        # actually coupled (the recycle's real reach), keeping the pattern tight.
        tcache: dict[tuple, np.ndarray] = {}

        def _translator(src_net, tgt_net):
            key = (id(src_net), id(tgt_net))
            if key not in tcache:
                tcache[key] = None
                for conn in self.connections:
                    T = getattr(conn, "translator", None)
                    if (
                        T is not None
                        and T.source_network is src_net
                        and T.target_network is tgt_net
                    ):
                        tcache[key] = np.asarray(translator_coupling_pattern(T), dtype=bool)
                        break
            return tcache[key]

        conc = {nm: u for nm, u in self.units.items() if self._is_concentration_unit(u)}
        for aname, (aunit, cp) in cps.items():
            if cp.inlet_pattern is None:
                continue
            ip = np.asarray(cp.inlet_pattern, dtype=bool)  # (a_rows, a_nsp)
            a_off, _ = self._state_layout[aname]
            a_rows = ip.shape[0]
            a_net = getattr(aunit, "network", None)
            for bname, bunit in conc.items():
                if bname == aname:
                    continue
                b_net = bunit.network
                b_off, _ = self._state_layout[bname]
                b_cols = b_net.n_species
                if (
                    coupling_mask is not None
                    and not coupling_mask[a_off : a_off + a_rows, b_off : b_off + b_cols].any()
                ):
                    continue  # not really coupled
                if a_net is b_net:
                    coupling = np.eye(a_net.n_species, dtype=bool)
                else:
                    coupling = _translator(b_net, a_net)  # (a_nsp, b_nsp)
                    if coupling is None:
                        continue
                block = (ip.astype(np.int8) @ coupling.astype(np.int8)) > 0
                P[a_off : a_off + a_rows, b_off : b_off + b_cols] |= block
        return P

    def _colored_jacobian_solver(self, solver, t0, y0, params, rtol, atol):
        """Return ``solver`` reconfigured to use a colored-AD Jacobian root
        finder, or ``solver`` unchanged when the colored path is unavailable.

        Built **once per plant** (concretely): derive the plant Jacobian sparsity
        pattern, color it, pack a :class:`ColoredVeryChord`, and **guard** it by
        comparing the colored and dense Jacobians at the start state -- falling
        back to the dense path (returning ``solver`` unchanged, with a warning) if
        they disagree. Reused on later solves. Skipped under tracing if not yet
        built (the probe/guard need concrete arrays), so a first traced solve
        falls back to dense. The colored matrix equals the dense one when the
        pattern is a superset, so the step sequence is unchanged; a pattern miss
        only costs solver steps, never accuracy.
        """
        from aquakin.integrate.colored_jacobian import (
            build_colored_root_finder,
            colored_jacobian_guard,
            jacobian_sparsity_pattern,
        )

        if self._colored_root_finder is None:
            if isinstance(params, jax.core.Tracer) or isinstance(y0, jax.core.Tracer):
                return solver  # can't build under trace; fall back
            t0a = jnp.asarray(float(t0))
            states0 = self._split_state(y0)
            rmap = self._recycle._maybe_recycle_map(t0a, states0, params)
            fmap = self._recycle._maybe_flow_map(t0a, states0, params)

            def rhs_y(y):
                return self._rhs(t0a, y, params, recycle_map=rmap, flow_map=fmap)

            atol_arr = jnp.asarray(atol)
            probe = jacobian_sparsity_pattern(rhs_y, y0) > 0
            structural = self._structural_plant_pattern(coupling_mask=probe)
            rf, n_colors = build_colored_root_finder(
                rhs_y,
                y0,
                rtol=10.0 * rtol,
                atol=10.0 * atol_arr,
                probe_pattern=probe,
                extra_pattern=structural,
            )
            ok = colored_jacobian_guard(rhs_y, y0, rf, context="colored_jacobian=True")
            self._colored_root_finder = (rf, n_colors, ok)

        rf, n_colors, ok = self._colored_root_finder
        if not ok:
            return solver
        # Build through the single-source-of-truth helper: it injects the colored
        # root finder into the user-supplied solver, or into the canonical Kvaerno5
        # when none is given -- so the colored path constructs no solver of its own.
        from aquakin.integrate._common import build_implicit_solver

        return build_implicit_solver(rtol, atol, solver=solver, colored_root_finder=rf)

    def _colored_adjoint_jacobian_builder(self, t0, rtol, atol):
        """Derive (once) the sparsity-colored ``df/dy`` Jacobian builder for the
        stable_adjoint backward pass, from the plant's **own default operating
        point**.

        The cap-free reverse-mode backward is dominated (~82% on BSM2) by per-step
        dense ``df/dy`` Jacobian builds -- one Jacobian-vector product per state.
        For a large block-sparse plant, coloring the Jacobian computes it in one
        JVP per *color* (BSM2: ~45 vs 167), cutting that cost while staying exact
        (the colored matrix equals the dense one when the pattern is a superset).

        The colored pattern is a **structural** property of the plant -- the unit
        coupling graph plus the recycle block structure -- so it is built from the
        plant's **default** state/params, NOT the solve's ``y0``/``params``. That
        decouples it from the solve routing and the AD trace: it builds lazily on
        first use even under a gradient trace, because it touches only concrete
        plant defaults (wrapped in ``ensure_compile_time_eval`` so the probe is not
        staged into the gradient). Whether to *use* the result is a separate,
        size-based decision (:attr:`_COLORED_BACKWARD_MIN_STATES`), so no build-time
        benchmark is needed -- which is what removes the old dependency on a
        concrete ``stable_adjoint`` solve to trigger and time the build.

        Caches ``(builder_or_None, n_colors, ok, n_states, rf)`` on
        ``_colored_adjoint_builder``: the augmented colored builder, the color
        count, whether the default-state guard passed, the plant state count (for
        the size gate), and the ``ColoredVeryChord``.

        Built for the **augmented** (time-carrying, ``n+1``) primal right-hand side
        the discrete adjoint actually differentiates. Guarded against the dense
        Jacobian at the default state; a mismatch falls back to dense (with a
        warning).
        """
        import numpy as np

        from aquakin.integrate.colored_jacobian import (
            build_colored_root_finder,
            colored_jacobian_guard,
            jacobian_sparsity_pattern,
            materialize_colored_jacobian,
        )
        from aquakin.integrate.discrete_adjoint import _autonomize

        if self._colored_adjoint_builder is not None:
            return self._colored_adjoint_builder[0]

        # Build from the plant's OWN default operating point -- a concrete,
        # always-available representative state -- so the pattern (a structural
        # property) is independent of the solve's inputs and computable even when
        # the only caller is a gradient (whose y0/params are tracers).
        # ``ensure_compile_time_eval`` keeps the concrete probe from being staged
        # into the enclosing gradient computation.
        with jax.ensure_compile_time_eval():
            y0 = self.initial_state()
            params = self.default_parameters()
            t0f = float(t0)
            rmap = self._recycle._maybe_recycle_map(jnp.asarray(t0f), self._split_state(y0), params)
            fmap = self._recycle._maybe_flow_map(jnp.asarray(t0f), self._split_state(y0), params)

            def primal(t, y, p):
                return self._rhs(t, y, p, recycle_map=rmap, flow_map=fmap)

            # The discrete adjoint integrates the autonomized [y; tau] state, so
            # the backward Jacobians -- and hence the coloring -- are of that.
            rhs_aug, y0_aug = _autonomize(primal, y0, t0f)

            def rhs_aug_y(ya):
                return rhs_aug(0.0, ya, params)

            # The backward feeds J directly into ``I - dt.gamma.J^T`` and the
            # transposed solve, so a missed coupling **silently corrupts the
            # gradient**. So the pattern must be a *complete* structural superset.
            # Use the per-component structural pattern (each unit's equation-derived
            # ``coupling_pattern()``, assembled by :meth:`_structural_plant_pattern`)
            # for the plant ``df/dy`` block, embedded in the augmented ``[y; tau]``
            # layout's top-left block; the probe at the default state supplies the
            # recycle block structure (connectivity-based, so any representative
            # state reveals it -- which is why the default state serves as well as
            # the real y0).
            n = int(y0.shape[0])
            plain_probe = (
                jacobian_sparsity_pattern(lambda y: primal(jnp.asarray(t0f), y, params), y0) > 0
            )
            structural = self._structural_plant_pattern(coupling_mask=plain_probe)
            aug_extra = np.zeros((n + 1, n + 1), dtype=bool)
            aug_extra[:n, :n] = plain_probe | structural

            # rtol/atol only set the (unused) chord tolerances on the returned
            # object; the seed matrix / coloring / pattern is all we use. A scalar
            # atol avoids augmenting the per-component vector for the probe.
            atol_s = float(jnp.max(jnp.asarray(atol)))
            rf, n_colors = build_colored_root_finder(
                rhs_aug_y, y0_aug, rtol=10.0 * rtol, atol=10.0 * atol_s, extra_pattern=aug_extra
            )
            ok = colored_jacobian_guard(
                rhs_aug_y, y0_aug, rf, context="colored_jacobian (stable_adjoint)"
            )

        # ``builder`` is applied later in the backward to the real (traced) f/y;
        # it closes over the concrete ``rf`` (seed/coloring/pattern) built above.
        def builder(f, y):
            return materialize_colored_jacobian(rf, f, y)

        # ``rf`` is cached too: it is reused as the *forward* solve's root finder
        # so the adjoint's forward pass can color its per-step Jacobian as well.
        self._colored_adjoint_builder = (
            builder if ok else None,
            n_colors,
            ok,
            n,
            rf if ok else None,
        )
        return self._colored_adjoint_builder[0]

    # Backward colored Jacobian is auto-enabled when the plant has at least this
    # many states. The win is set by absolute plant size, not sparsity: a small
    # plant pays more in per-build overhead than it saves (BSM1, 65 states:
    # colored is ~2x slower per build despite 4.7x fewer colors), while a large
    # one benefits (BSM2, 167 states). The threshold sits between them. The colored
    # backward is exact (== dense), so a mis-set gate only ever costs a little
    # speed, never correctness -- which is what lets a deterministic size gate
    # replace the old (concrete-solve-dependent) build-time benchmark.
    _COLORED_BACKWARD_MIN_STATES = 100

    def colored_jacobian_decision(self):
        """Report the auto colored-backward-Jacobian decision:
        ``("colored" | "dense", n_states)``, or ``None`` before the state layout
        can be built.

        Deterministic from the plant size (:attr:`_COLORED_BACKWARD_MIN_STATES`):
        the colored backward is auto-enabled iff the plant has at least that many
        states. (With ``colored_jacobian=True``/``False`` the choice is forced and
        this size gate does not apply.)
        """
        try:
            n = int(self.initial_state().shape[0])
        except Exception:
            return None
        choice = "colored" if n >= self._COLORED_BACKWARD_MIN_STATES else "dense"
        return (choice, n)

    def _colored_steady_jacobian_builder(self, rhs, y0, theta, *, tol):
        """Derive (once, concretely) a colored-AD materializer for the PTC steady
        residual Jacobian ``dF/dy``, or ``None`` to fall back to dense ``jacfwd``.

        The PTC iteration (:func:`aquakin.plant.steady.ptc_forward`) forms the
        full plant Jacobian ``dF/dy`` -- the same block-sparse object the
        integrator's implicit-stage and the stable_adjoint backward color -- once
        per Newton step (~tens of times for BSM2). Coloring builds it in one
        Jacobian-vector product per *color* (BSM2: ~45 vs 167 columns) instead of
        ``n``, while reconstructing the identical matrix on the pattern's support.

        Unlike the dynamic solve, this is well suited to coloring: PTC marches to
        a single operating point in a narrow neighbourhood, so the warm-start
        probe usually suffices on its own (the dynamic run's wide load excursion
        is what makes start-state-missed couplings a problem). The builder unions
        it with the complete per-component structural pattern regardless, matching
        the forward and stable_adjoint builders. Built and **guarded** against the
        dense Jacobian at ``y0`` once (falling back to dense with a warning on a
        mismatch), then cached and reused -- a parameter/design sweep keeps the
        same structural pattern.

        Returns a builder ``(F, y) -> dF/dy``, or ``None`` (dense) when called
        under a trace (the probe needs concrete arrays) or when the guard fails.
        """

        from aquakin.integrate.colored_jacobian import (
            build_colored_root_finder,
            colored_jacobian_guard,
            jacobian_sparsity_pattern,
            materialize_colored_jacobian,
        )

        if self._colored_steady_builder is None:
            if any(
                isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves((theta, y0))
            ):
                return None  # can't build under trace; fall back

            def rhs_y(y):
                return rhs(y, theta)

            tol_s = float(jnp.max(jnp.asarray(tol)))
            # Use the per-component structural pattern (each unit's equation-derived
            # coupling_pattern, assembled by _structural_plant_pattern) unioned with
            # the warm-start probe, matching the forward and stable_adjoint builders.
            # PTC stays in a narrow neighbourhood so the start-state probe is usually
            # enough on its own, but the structural superset is complete regardless
            # of how far the warm start sits from the steady state.
            plain_probe = jacobian_sparsity_pattern(rhs_y, y0) > 0
            structural = self._structural_plant_pattern(coupling_mask=plain_probe)
            rf, n_colors = build_colored_root_finder(
                rhs_y,
                y0,
                rtol=tol_s,
                atol=tol_s,
                probe_pattern=plain_probe,
                extra_pattern=structural,
            )
            ok = colored_jacobian_guard(
                rhs_y, y0, rf, context="colored_jacobian=True (steady_state)"
            )

            def builder(f, y):
                return materialize_colored_jacobian(rf, f, y)

            self._colored_steady_builder = (builder if ok else None, n_colors, ok)

        return self._colored_steady_builder[0]

    def _sweep_outputs(
        self,
        t: jnp.ndarray,
        states: dict[str, jnp.ndarray],
        streams: dict[tuple[Optional[str], str], Stream],
        seeded: dict[tuple[str, str], Stream],
        params_full: jnp.ndarray,
        passes: Optional[int] = None,
        signals: Optional[dict] = None,
    ) -> dict[tuple[str, str], Stream]:
        """Resolve unit outputs with the fixed-pass Gauss-Seidel recycle sweep.

        Recycle edges (downstream -> upstream) create cyclic data dependencies.
        We resolve them with a fixed number of passes (``recycle_passes``,
        default 3) over the unit-output computations: each pass refines the
        stream estimates on recycle edges. The count is fixed, not
        iterate-to-tolerance, because this RHS is jitted and differentiated --
        a data-dependent loop is not allowed. Convergence is fast because most
        recycle-stream concentrations are a source unit's *state* (read
        directly, e.g. a CSTR's output is its own tank concentration), so the
        only iterated chain is the short mixer/splitter/clarifier path. For
        BSM-family topologies the streams reach a fixed point in 2 passes
        (verified to machine precision in tests/integration/test_bsm1.py), so
        the default 3 is a safe margin. ``streams`` is updated in place; the
        returned ``all_outputs`` includes the recycle seeds.

        ``passes`` overrides ``self.recycle_passes`` for this call -- used by the
        one-time convergence diagnostic (:meth:`_check_recycle_convergence`),
        which sweeps to a deeper count to see whether the default has converged.
        """
        n_passes = self.recycle_passes if passes is None else passes
        # Under a HeatBalanceTemperature, a tracked unit's outlet leaves at its
        # *tank* temperature (a state, constant w.r.t. its inlets this RHS), not
        # the flow-weighted inlet T the unit self-computes. Override it here so the
        # lagged temperature propagates downstream and (since this method also runs
        # the affine recycle-temperature probe) through that exact solve. Empty for
        # the algebraic default -> a pure no-op.
        temp_by_unit = self._temperatures_by_unit(states)
        all_outputs: dict[tuple[str, str], Stream] = {}
        all_outputs.update(seeded)
        for _pass in range(n_passes):
            for name in self._unit_order:
                unit = self.units[name]
                inputs = self._collect_inputs(name, all_outputs, streams, states, params_full)
                params_unit = self._params_for_unit(name, params_full)
                outputs = unit.compute_outputs(t, states[name], inputs, params_unit, signals)
                override_T = temp_by_unit.get(name)
                for port, stream in outputs.items():
                    if override_T is not None:
                        stream = stream.with_T(override_T)
                    all_outputs[(name, port)] = stream
                    streams[(name, port)] = stream
        return all_outputs

    def _warn_if_flow_nonaffine(self, recycled, x) -> None:
        """Warn if the recycle-flow solve is inconsistent (non-affine flow rule).

        ``recycled`` is the forward pass re-evaluated at the solved recycle flows
        ``x``; for an affine flow network they are equal. A meaningful residual
        means the affine ``(I - A) x = b`` solve does not actually solve the flow
        network -- see :meth:`_resolve_flows`.
        """
        resid = float(jnp.max(jnp.abs(jnp.asarray(recycled) - jnp.asarray(x))))
        scale = float(jnp.max(jnp.abs(jnp.asarray(x)))) + 1.0
        if resid > 1e-6 * scale:
            warnings.warn(
                f"Recycle-flow solve is inconsistent (residual {resid:.3g} on "
                f"flows of order {scale:.3g}): a unit's flow rule is non-affine "
                f"in the recycle flows -- typically a threshold-mode SplitterUnit "
                f"or a StorageTank bypass whose inlet varies with a recycle flow "
                f"and crosses its kink. Plant._resolve_flows assumes affine flow "
                f"rules, so the resolved recycle flows (and the steady state they "
                f"drive) may be inaccurate. Feed such a unit from a recycle-"
                f"independent source (e.g. an external influent).",
                stacklevel=2,
            )

    def _collect_inputs(
        self,
        unit_name: str,
        all_outputs: dict[tuple[str, str], Stream],
        streams: dict[tuple[Optional[str], str], Stream],
        states: Optional[dict[str, jnp.ndarray]] = None,
        params_full: Optional[jnp.ndarray] = None,
    ) -> dict[str, Stream]:
        """Find the input stream for each port of ``unit_name``.

        A translator that declares ``needs_dest_pH`` (the ASM->ADM interface)
        has a pH-dependent inorganic-carbon charge balance, which BSM2
        evaluates at the digester pH. It is fed the destination (digester) unit's
        instantaneous, state-derived pH so the feed it produces is
        BSM2-consistent. This needs ``states``/``params_full``; when they are absent
        (e.g. the control-signal sweep) the translator falls back to its fixed
        ``pH_adm``.
        """
        inputs: dict[str, Stream] = {}
        for conn in self._inputs_by_unit.get(unit_name, ()):
            if conn.from_unit is None:
                src = streams[(None, conn.from_port)]
            else:
                key = (conn.from_unit, conn.from_port)
                if key in all_outputs:
                    src = all_outputs[key]
                elif key in streams:
                    src = streams[key]
                else:
                    raise RuntimeError(
                        f"Stream {conn.from_unit}.{conn.from_port} not yet "
                        f"available for {conn.to_unit}.{conn.to_port}. "
                        f"Recycle edges must be seeded with initial_value."
                    )
            digester_pH = None
            if states is not None:
                if getattr(conn.translator, "needs_dest_pH", False):
                    # ASM->ADM: the digester is the destination.
                    digester_pH = self._unit_operating_pH(unit_name, states, params_full)
                elif getattr(conn.translator, "needs_src_pH", False) and conn.from_unit is not None:
                    # ADM->ASM: the digester is the source.
                    digester_pH = self._unit_operating_pH(conn.from_unit, states, params_full)
            inputs[conn.to_port] = src.with_C(
                conn.translator.translate(src.C, digester_pH=digester_pH)
            )
        return inputs

    def _unit_operating_pH(self, unit_name, states, params_full):
        """The state-derived pH of ``unit_name`` for a pH-coupled translator.

        Returns ``None`` (interface uses its fixed fallback) if the unit exposes
        no ``operating_pH`` -- so the feedback is opt-in per unit.
        """
        op = getattr(self.units[unit_name], "operating_pH", None)
        if op is None:
            return None
        return op(states[unit_name], self._params_for_unit(unit_name, params_full))

    # ----- solve -----------------------------------------------------------

    def solve(
        self,
        t_span: tuple[float, float],
        t_eval: Union[jnp.ndarray, None] = None,
        params: Optional[jnp.ndarray] = None,
        *,
        rtol: float = 1e-6,
        atol: Union[float, jnp.ndarray, None] = None,
        y0: Optional[jnp.ndarray] = None,
        integrator: IntegratorConfig = IntegratorConfig(),
        diff: DifferentiationConfig = DifferentiationConfig(),
        time_unit: Optional[str] = None,
        event: Optional[diffrax.Event] = None,
        events: Optional[Sequence["Event"]] = None,
        progress_meter: Optional["diffrax.AbstractProgressMeter"] = None,
        forward_fast: bool = False,
    ) -> PlantSolution:
        """Integrate the plant over ``t_span``.

        Parameters
        ----------
        t_span : (float, float)
            ``(t_start, t_end)`` in the plant's time unit (``plant.time_unit``,
            typically days for BSM-family) unless ``time_unit`` is given.
        t_eval : jnp.ndarray, optional
            Save times. If ``None`` only the endpoint is saved.
        params : jnp.ndarray, optional
            Flat plant parameter vector. Defaults to :meth:`default_parameters`.
        rtol, atol : float or array, optional
            Solver tolerances. ``atol=None`` auto-scales a per-component noise
            floor off the operating magnitudes.
        y0 : jnp.ndarray, optional
            Warm-start initial state (e.g. a previously-computed steady state),
            shape ``(total_state_size,)``. Defaults to the per-unit initial states.
        integrator : IntegratorConfig, optional
            Integrator / step configuration. ``order`` selects Kvaerno3 (default,
            fast) or Kvaerno5; ``factormax`` caps the PID step-growth (default 3);
            ``dtmax`` caps the step (set it only for a reverse gradient *through*
            the solve, ``diff.method='through_solve'``); ``max_steps`` is the step
            budget; ``solver`` is an explicit override (honoured verbatim);
            ``colored_jacobian`` ({"auto", True, False}) selects sparse colored-AD
            materialisation of the per-step Jacobian. ``"auto"`` (default) governs
            the ``mode='reverse', method='stable'`` backward decision -- it measures
            whether the colored ``df/dy`` build pays and enables it only then (a
            large plant like BSM2 yes, a small one like BSM1 no), and leaves the
            forward solve dense; ``True`` forces coloring on both paths; ``False``
            disables it. Reported by :meth:`colored_jacobian_decision`.
        diff : DifferentiationConfig, optional
            How a gradient / sensitivity flows through the solve.
            ``mode='reverse', method='stable'`` (the default) routes a plain
            concrete forward solve to the fast cached path and a reverse-mode
            differentiated solve to the cap-free hand-written discrete adjoint
            (:func:`~aquakin.esdirk_adjoint_solve`) -- finite for a stiff plant
            with no ``dtmax`` to tune. ``mode='reverse', method='through_solve'``
            differentiates *through* the diffrax solve (``RecursiveCheckpointAdjoint``;
            needs a ``dtmax`` cap for a stiff plant). ``mode='forward',
            method='through_solve'`` builds a forward-capable adjoint so
            ``jax.jvp`` / ``jacfwd`` flow through the solve. ``mode='forward',
            method='stable'`` is the augmented variational solve -- not supported
            here; call :meth:`solve_sensitivity` / :meth:`dynamic_sensitivity`.
            ``check_finite`` raises on a non-finite gradient; ``adjoint_max_steps``
            bounds the discrete-adjoint backward-scan buffer (it must exceed the
            forward step count); ``adjoint_low_memory`` recomputes the backward
            stages instead of saving the ``~n_stages``x dense buffer (memory for
            compute, gradient unchanged). Passing ``event=`` pins the forward
            ``jax_adjoint`` path.
        time_unit : str, optional
            The unit ``t_span`` / ``t_eval`` are in (``"s"``/``"min"``/``"h"``/
            ``"d"``). Default ``None`` uses the plant's native time unit. When
            given, the input times are converted to the native unit for the solve
            and ``solution.t`` is reported back in ``time_unit``.
        event : diffrax.Event, optional
            The low-level single terminating event (used internally by
            :meth:`run_to_steady_state`); pins the forward ``jax_adjoint`` path.
        events : sequence of Event, optional
            Located plant-wide discontinuities (on/off pumps, SBR phase switches,
            dosing on/off, tank-level limits). The monolithic solve is split into
            segments at the firings and ``solution.events_log`` records them.
            Time-only events keep ``jax.grad`` finite; a state event makes the run
            a forward simulation. Mutually exclusive with ``event=`` and with
            ``diff=DifferentiationConfig(method='stable')`` reverse differentiation.
        progress_meter : diffrax.AbstractProgressMeter, optional
            A diffrax progress meter for a long forward solve.
        forward_fast : bool, optional
            Use the lean non-AD forward integrator (no diffrax adjoint machinery):
            a much faster compile + run for a one-off forward solve, but the result
            is NOT differentiable and it needs concrete ``params``/``y0``.

        Returns
        -------
        PlantSolution
        """
        # Resolve the public config objects into the internal primitive locals the
        # rest of this method (and the build helpers) operate on. IntegratorConfig
        # carries the step machinery; DifferentiationConfig carries the AD mode.
        solver = integrator.solver
        factormax = integrator.factormax
        colored_jacobian = integrator.colored_jacobian
        dtmax = integrator.dtmax
        max_steps = int(integrator.max_steps)
        order = int(integrator.order)
        # DifferentiationConfig -> the legacy gradient / adjoint routing.
        # reverse + stable    -> "auto"        (concrete fwd -> cached; diff -> discrete adjoint)
        # reverse + through   -> "jax_adjoint" (differentiate through the diffrax solve)
        # forward + through   -> adjoint=forward_adjoint() (jvp/jacfwd through the solve)
        # forward + stable    -> the augmented [y; S] variational solve (use solve_sensitivity)
        if diff.mode not in ("reverse", "forward"):
            raise ValueError(f"diff.mode must be 'reverse' or 'forward'; got {diff.mode!r}.")
        if diff.method not in ("stable", "through_solve"):
            raise ValueError(
                f"diff.method must be 'stable' or 'through_solve'; got {diff.method!r}."
            )
        adjoint = None
        if diff.mode == "reverse":
            gradient = "auto" if diff.method == "stable" else "jax_adjoint"
        else:  # forward
            if diff.method == "stable":
                raise ValueError(
                    "diff=DifferentiationConfig(mode='forward', method='stable') is "
                    "the augmented variational solve; call plant.solve_sensitivity "
                    "(or plant.dynamic_sensitivity) for it, not plant.solve."
                )
            # forward + through_solve: jvp/jacfwd through the diffrax solve.
            gradient = "jax_adjoint"
            adjoint = forward_adjoint()
        adjoint_max_steps = int(diff.adjoint_max_steps)
        adjoint_low_memory = bool(diff.adjoint_low_memory)

        # The reverse stable method forms its own cap-free discrete adjoint and
        # controls its own steps, so an explicit dtmax cap alongside it is a usage
        # error (dtmax is meaningful only when differentiating *through* the diffrax
        # solve, i.e. method='through_solve'). Catch it here, before the "auto"
        # routing below would otherwise silently treat dtmax as a request to pin
        # the jax_adjoint path.
        if diff.mode == "reverse" and diff.method == "stable" and dtmax is not None:
            raise ValueError(
                "diff=DifferentiationConfig(method='stable') forms its own discrete "
                "adjoint and controls its own steps; do not also pass "
                "integrator=IntegratorConfig(dtmax=...) (the stable method is "
                "cap-free). Use method='through_solve' if you need the dtmax cap."
            )

        if gradient not in ("auto", "jax_adjoint", "stable_adjoint"):
            raise ValueError(
                f"gradient must be 'auto', 'jax_adjoint' or 'stable_adjoint'; got {gradient!r}."
            )
        if colored_jacobian not in (True, False, "auto"):
            raise ValueError(
                f"colored_jacobian must be True, False or 'auto'; got {colored_jacobian!r}."
            )
        self._build_state_layout()
        self._build_parameter_layout()
        if params is None:
            params = self.default_parameters()
        else:
            params = self._coerce_params(params)
        # Initial state: the per-unit defaults, or a caller-supplied warm start
        # (e.g. a previously-computed steady state -- the standard way to start a
        # dynamic-influent run, avoiding the stiff clean-start transient).
        if y0 is None:
            y0 = self.initial_state()
        else:
            y0 = jnp.asarray(y0)
            if y0.shape != (self._total_state_size,):
                raise ValueError(f"y0 has shape {y0.shape}, expected ({self._total_state_size},)")
        # Default atol is a per-component noise floor scaled off the operating
        # magnitudes (the warm start y0 and the per-unit defaults), so a g/m³
        # plant solves without the old fixed 1e-9 forcing the step ceiling.
        if atol is None:
            atol_eff = default_atol(y0, self.initial_state())
        else:
            atol_eff = _coerce_atol(atol, self._total_state_size)
        # Convert t_span / t_eval from a caller-supplied time_unit into the
        # plant's native (rate-constant) unit; _time_factor scales back on output.
        t_span, t_eval, _time_factor = to_native_time(self.time_unit, time_unit, t_span, t_eval)
        t0, t1 = float(t_span[0]), float(t_span[1])
        if not (t1 > t0):
            raise ValueError(f"t_span end must exceed start; got ({t0}, {t1}).")
        if t_eval is not None:
            t_eval = jnp.asarray(t_eval)

        # Auto-collect located phase-transition events that units declare (an
        # SBRUnit's cycle boundaries), merged with any user-supplied events, so
        # the integrator lands exactly on every phase switch without the caller
        # hand-listing them. cycle_events takes the native-time span.
        unit_events: list[Event] = []
        for unit in self.units.values():
            collect = getattr(unit, "cycle_events", None)
            if collect is not None:
                unit_events.extend(collect(t0, t1))
        if unit_events:
            events = (list(events) + unit_events) if events is not None else unit_events

        # Validate incompatible argument combinations BEFORE any concrete solve
        # work (the affinity/recycle-map probes below call _resolve_flows, which
        # would raise an obscure error on an intentionally-minimal plant). The
        # events dispatch itself stays after the concrete checks so a valid events
        # solve still gets the cached recycle map.
        if events is not None:
            if event is not None:
                raise ValueError(
                    "pass either events= (the user-facing located-event API) or "
                    "the low-level event= (a single diffrax terminating event), "
                    "not both."
                )
            # The located-event solve runs a segmented solve through the diffrax
            # path (with the resolved adjoint), so the cap-free reverse *stable*
            # discrete adjoint -- which has no segmented form -- cannot back it.
            # ``method="stable"`` is the default and routes a concrete events solve
            # to the segmented jax_adjoint path fine; only an explicit request to
            # differentiate the events solve via the stable discrete adjoint is the
            # error, which the runtime trace below reports. Here we reject the two
            # integrator opt-ins the segmented solve cannot honour.
            if integrator.solver is not None or integrator.colored_jacobian is True:
                raise ValueError(
                    "integrator.solver / colored_jacobian=True are not supported "
                    "with events=; the located-event solve manages its own "
                    "integrator. Drop them."
                )

        # forward_fast is the lean non-AD forward integrator: no diffrax adjoint /
        # optimistix / lineax machinery, so the result is NOT differentiable and it
        # needs concrete inputs to build its colored-Jacobian pattern once.
        if forward_fast:
            if events is not None:
                raise ValueError(
                    "forward_fast is not supported with events=; the located-event "
                    "solve manages its own integrator. Drop one."
                )
            if gradient == "stable_adjoint":
                raise ValueError(
                    "forward_fast is a non-differentiable forward path; it is "
                    "incompatible with gradient='stable_adjoint'. For a reverse-mode "
                    "gradient use the default solve."
                )
            if any(isinstance(v, jax.core.Tracer) for v in (params, y0)):
                raise ValueError(
                    "forward_fast requires concrete params/y0 -- it is a non-AD "
                    "fast path that is not differentiable and cannot be traced "
                    "(no jax.grad / jax.jit). For gradients or jit use the default "
                    "solve (which routes through the differentiable diffrax path)."
                )

        # One-time, concrete check that the recycle-flow solve is self-consistent
        # (every flow rule affine in the recycle flows). Skipped under tracing
        # (params/y0 are JAX tracers -- can't compare/warn) and guarded to run
        # once per plant. It only warns, never blocks the solve. Run before the
        # events branch too, so an events-only plant (an SBR/control study) still
        # gets these diagnostics and the cached recycle map.
        #
        # LIMITATION: this probes affinity only at (t0, y0). A unit with a
        # piecewise-linear flow rule -- a threshold-mode SplitterUnit (influent
        # bypass) or a StorageTank level-gated bypass -- that is on one side of
        # its kink at t0 but crosses it *later* in a dynamic run is NOT caught:
        # _resolve_flows then linearises across the kink at that time with no
        # warning. The resolved flows stay exact only while every unit remains in
        # the affine regime it occupied at t0. (Re-probing at sampled times would
        # close this; deferred as a known limitation -- see issue #255.)
        if not any(isinstance(v, jax.core.Tracer) for v in (params, y0)):
            if not self._flow_affinity_checked:
                self._flow_affinity_checked = True
                self._recycle._resolve_flows(
                    jnp.asarray(t0),
                    params,
                    states=self._split_state(y0),
                    check_affine=True,
                )
            # Companion diagnostic: does the fixed-pass recycle concentration
            # sweep converge at recycle_passes? (warns once, never blocks.)
            if not self._recycle_convergence_checked:
                self._recycle_convergence_checked = True
                self._recycle._check_recycle_convergence(
                    jnp.asarray(t0), self._split_state(y0), params
                )
            # Is the recycle affine map M state-independent? If so, it can be
            # precomputed once per solve and reused -- a large per-RHS saving.
            if self._recycle._recycle_map_constant is None:
                self._recycle._check_recycle_map_constant(jnp.asarray(t0), y0, params)
            # Is the recycle *flow* map A state-independent? If so it is cached
            # once per solve too, skipping the per-RHS flow probe.
            if self._recycle._flow_map_constant is None:
                self._recycle._check_flow_map_constant(jnp.asarray(t0), y0, params)
            # (The colored backward Jacobian builder is derived in the
            # stable_adjoint branch below, only when that path is actually taken,
            # so a forward jax_adjoint solve never builds it.)

        if events is not None:
            # (argument combinations validated above, before the concrete checks)
            return self._solve_with_events(
                t0,
                t1,
                t_eval,
                params,
                y0,
                events,
                rtol=rtol,
                atol=atol_eff,
                dtmax=dtmax,
                adjoint=adjoint,
                max_steps=max_steps,
                time_factor=_time_factor,
                time_unit=time_unit,
                order=order,
                factormax=factormax,
                solver=solver,
            )

        if gradient == "auto":
            # A concrete forward solve takes the fast cached jax_adjoint path; a
            # solve under reverse-mode differentiation (params/y0 are tracers)
            # takes the cap-free stable_adjoint, so a stiff plant gradient is
            # finite by default. event=/adjoint=/dtmax= are jax_adjoint-only, so
            # their presence pins jax_adjoint.
            differentiating = any(
                isinstance(v, jax.core.Tracer) for v in (params, y0) if v is not None
            )
            pin_jax = event is not None or adjoint is not None or dtmax is not None
            gradient = "jax_adjoint" if (pin_jax or not differentiating) else "stable_adjoint"

        if gradient == "stable_adjoint":
            if adjoint is not None or dtmax is not None:
                raise ValueError(
                    "gradient='stable_adjoint' forms its own discrete adjoint and "
                    "controls its own steps; do not also pass adjoint= or dtmax=."
                )
            if event is not None:
                raise ValueError(
                    "event= (e.g. a steady-state terminating event) is only "
                    "supported on the forward gradient='jax_adjoint' path."
                )
            # solver=/factormax= ARE supported here: the discrete adjoint builds
            # its backward from the forward solver's Butcher tableau generically
            # (any ESDIRK with a tableau -- e.g. the cheaper 4-stage Kvaerno3 in
            # place of the default 7-stage Kvaerno5), and factormax caps the
            # per-step growth of its PID controller, exactly as on the forward
            # path. A non-ESDIRK solver raises a clear error from the tableau
            # extraction. ``solver=None`` keeps the default Kvaerno5.
            # colored_jacobian colors the per-step df/dy Jacobian build in the
            # BACKWARD pass (its dominant cost), exact (machine-precision) vs dense.
            # On a concrete solve, derive + cache the colored builder and -- for
            # "auto" -- measure whether it actually pays (the build is cheaper than
            # dense); under a first traced solve (before it is built) it is None ->
            # dense fallback. The decision: True forces it on, "auto" enables it
            # only when the measured build speedup clears the margin (so it is on
            # for a large block-sparse plant like BSM2 and off for a small one like
            # BSM1, where the colored build is slower), False never colors.
            # The colored backward Jacobian is enabled deterministically by plant
            # SIZE (the reliable signal -- see _COLORED_BACKWARD_MIN_STATES), not a
            # build-time benchmark, so no concrete probe solve is needed: the
            # builder is derived from the plant's default operating point and so
            # builds even here under the gradient's own trace. ``True`` forces it on
            # (any size), ``"auto"`` enables it past the size gate, ``False`` never
            # colors.
            n_states = int(y0.shape[0])
            want_colored = colored_jacobian is True or (
                colored_jacobian == "auto" and n_states >= self._COLORED_BACKWARD_MIN_STATES
            )
            if want_colored and self._colored_adjoint_builder is None:
                self._colored_adjoint_jacobian_builder(t0, rtol, atol_eff)
            cab = self._colored_adjoint_builder
            use_colored = bool(want_colored and cab is not None and cab[2])  # built + guard ok
            jac_builder = cab[0] if use_colored else None
            # The infrastructure to also color the *forward* solve's per-step
            # implicit Jacobian is in place (``cab[4]`` is the ColoredVeryChord for
            # the autonomized forward RHS, and ``esdirk_adjoint_solve`` accepts it
            # as ``forward_root_finder``), but it is NOT auto-enabled: unlike the
            # colored *backward* (which feeds J directly into the transposed solve,
            # exact on a superset pattern), the colored forward feeds J into an
            # iterative chord whose decoupled-Newton convergence point depends on
            # the J approximation, so a colored-vs-dense difference shifts the
            # forward trajectory at the ~Newton-tolerance level (~1e-4) -- it would
            # break the bit-identical ``colored_jacobian=True`` == dense invariant.
            # Enabling it as a default needs the structural pattern reconciled so
            # the colored and dense forward chords converge identically.
            fwd_root_finder = None
            # Cap-free reverse-mode gradient through the stiff plant solve: the
            # forward is a robust adaptive ESDIRK solve and the reverse is the
            # per-step transposed-solve discrete adjoint over the saved
            # trajectory, which stays finite at any step size. The compiled solve
            # is cached per instance and reused across repeat *forward* solves
            # (the parameter-sweep case), keyed like the forward path but tagged
            # so it never collides with it. ``t_eval`` is baked into the compiled
            # closure (the adjoint marks it non-differentiable, so it cannot be a
            # traced runtime argument), so the key carries its values. The cache
            # is used only when the inputs are concrete: under a trace (a gradient
            # through the solve, or an enclosing jit) the adjoint's ``custom_vjp``
            # is traced directly into the outer computation -- routing it through
            # an inner ``jax.jit`` does not compose with an outer reverse-mode
            # pass -- which is the path the calibration gradient takes anyway.
            # The discrete-adjoint backward scan walks a saved trajectory whose
            # length is adjoint_max_steps (DifferentiationConfig.adjoint_max_steps);
            # the forward solve must also fit in that buffer, so this single budget
            # serves both roles here.
            adj_steps = adjoint_max_steps
            settings = concrete_settings_key(rtol, atol_eff, None, None, adj_steps)
            teval_key, teval_concrete = _concrete_teval_key(t_eval)
            under_trace = isinstance(params, jax.core.Tracer) or isinstance(y0, jax.core.Tracer)
            # The forward ESDIRK solver (e.g. the cheaper Kvaerno3) and factormax
            # change the compiled solve, so they key the cache -- the solver by
            # class, exactly as the forward jax_adjoint path does. The ESDIRK
            # ``order`` (3 vs 5) also changes the compiled solve, so it is keyed.
            # ``adjoint_low_memory`` selects the recompute backward (no saved
            # dense-stage buffer), a different compiled solve, so it is keyed too.
            solver_key = type(solver).__name__ if solver is not None else None
            cache_key = (
                None
                if (settings is None or not teval_concrete or under_trace)
                else (
                    "stable_adjoint",
                    t0,
                    t1,
                    teval_key,
                    settings,
                    use_colored,
                    solver_key,
                    factormax,
                    order,
                    adjoint_low_memory,
                )
            )
            with friendly_solve_errors(adj_steps, what="plant solve"):
                if cache_key is not None:
                    jitted = self._jit_cache.get(cache_key)
                    if jitted is None:
                        jitted = self._build_jitted_stable_adjoint_solve(
                            t0,
                            t1,
                            t_eval,
                            rtol=rtol,
                            atol=atol_eff,
                            max_steps=adj_steps,
                            forward_root_finder=fwd_root_finder,
                            solver=solver,
                            factormax=factormax,
                            order=order,
                            low_memory=adjoint_low_memory,
                        )
                        self._jit_cache[cache_key] = jitted
                    ys = jitted(y0, params)
                else:
                    # Under-trace (the calibration-gradient path): not cached, but
                    # still hoists the recycle probe via the cached-map primal RHS
                    # (the same exact split as the jitted closure) and uses the
                    # colored backward Jacobian builder + colored forward root
                    # finder when requested.
                    ys = self._esdirk_stable_adjoint(
                        y0,
                        params,
                        t0,
                        t1,
                        t_eval,
                        rtol=rtol,
                        atol=atol_eff,
                        max_steps=adj_steps,
                        jacobian_builder=jac_builder,
                        forward_root_finder=fwd_root_finder,
                        solver=solver,
                        factormax=factormax,
                        order=order,
                        low_memory=adjoint_low_memory,
                    )
            if t_eval is None:
                ts = jnp.asarray([t1])
                ys = ys[None, :]
            else:
                ts = jnp.asarray(t_eval)
            if _time_factor != 1.0:
                ts = ts / _time_factor  # native -> requested unit
            sol = PlantSolution(t=ts, state=ys, plant=self)
            if time_unit is not None:
                sol._requested_time_unit = time_unit
            return sol

        # Forward (jax_adjoint) path. The compiled solve is cached per instance
        # and reused across repeat solves of this plant (see ``_jit_cache``). An
        # event-terminated solve (e.g. run_to_steady_state) is not cached -- the
        # event closure varies and that path is run once -- and a traced call
        # (settings key None) bypasses the cache. Caching does not change the
        # first solve; it only avoids recompiling on subsequent ones.
        # Forward-fast: the lean non-AD integrator (no diffrax adjoint / optimistix
        # / lineax). It needs the colored-Jacobian pattern (for a cheap per-step
        # Jacobian); build + guard it, and fall back to the diffrax path if the
        # guard fails. Concrete-only (validated above). Its compiled solve is
        # cached per instance like the diffrax path.
        if forward_fast:
            self._colored_jacobian_solver(None, t0, y0, params, rtol, atol_eff)
            crf = self._colored_root_finder
            if crf is not None and crf[2]:
                te = t_eval if t_eval is not None else jnp.asarray([float(t1)])
                fkey = (
                    "forward_fast",
                    t0,
                    t1,
                    tuple(te.shape),
                    concrete_settings_key(rtol, atol_eff, None, None, max_steps),
                    self._recycle._recycle_map_constant,
                )
                jitted = self._jit_cache.get(fkey)
                if jitted is None:
                    jitted = self._build_jitted_forward_fast(t0, t1, rtol=rtol, atol=atol_eff)
                    self._jit_cache[fkey] = jitted
                with friendly_solve_errors(max_steps, what="plant forward_fast solve"):
                    ts, ys = jitted(y0, params, te)
                if _time_factor != 1.0:
                    ts = ts / _time_factor
                sol = PlantSolution(t=ts, state=ys, plant=self)
                if time_unit is not None:
                    sol._requested_time_unit = time_unit
                return sol
            warnings.warn(
                "forward_fast: the colored-Jacobian start-state guard failed; "
                "falling back to the diffrax forward path for this plant.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Colored-AD Jacobian: reconfigure the solver to materialise the per-step
        # implicit Jacobian by sparse column compression (a large saving on the
        # dense flowsheet Jacobian). Built+guarded once per plant; falls back to
        # the dense solver if the start-state guard fails or it can't be built
        # (a first traced solve). Numerically identical when the pattern is a
        # superset, so it composes with solver=/factormax= and the cached map.
        # Forward coloring is enabled by an explicit ``colored_jacobian=True``.
        # The default ``"auto"`` governs only the (exact) stable_adjoint BACKWARD
        # decision; it leaves the forward solve dense, because forward coloring
        # swaps the implicit linear solver and so is not guaranteed bit-identical
        # -- making it the all-solves default needs its own full-suite validation
        # (a deliberate follow-up). ``True`` opts into both.
        if colored_jacobian is True:
            solver = self._colored_jacobian_solver(solver, t0, y0, params, rtol, atol_eff)
        colored_active = colored_jacobian is True and (
            self._colored_root_finder is not None and self._colored_root_finder[2]
        )

        settings = concrete_settings_key(rtol, atol_eff, adjoint, dtmax, max_steps)
        sig = (t0, t1, None if t_eval is None else tuple(t_eval.shape))
        # The solver is keyed by class name: a fresh stock solver instance (the
        # common case) shares the cache with another of the same class, while a
        # different class (Kvaerno3 vs the default Kvaerno5) keys separately. A
        # custom-*configured* instance of an otherwise-default class would share
        # the default's entry -- documented on the ``solver=`` argument.
        solver_key = None if solver is None else type(solver).__name__
        # The recycle-map reuse flags are part of the compiled solve (they select
        # the cached-M / cached-MT code path). They are set once per instance
        # before the first build, so they are constant for a given plant -- keying
        # on them only guards the rare case where the very first solve was traced
        # (flags still None) and a later concrete solve sets them.
        recycle_key = (
            self._recycle._recycle_map_constant,
            self._recycle._recycle_T_map_constant,
            self._recycle._flow_map_constant,
        )
        # A progress meter is a one-off diagnostic (and carries host state), so it
        # bypasses the compiled-solve cache rather than being keyed into it.
        cache_key = (
            None
            if (settings is None or event is not None or progress_meter is not None)
            else (sig, settings, solver_key, factormax, recycle_key, colored_active, order)
        )
        jitted = self._jit_cache.get(cache_key) if cache_key is not None else None
        if jitted is None:
            jitted = self._build_jitted_solve(
                t0,
                t1,
                t_eval is not None,
                event=event,
                rtol=rtol,
                atol=atol_eff,
                adjoint=adjoint,
                dtmax=dtmax,
                max_steps=max_steps,
                progress_meter=progress_meter,
                solver=solver,
                factormax=factormax,
                order=order,
            )
            if cache_key is not None:
                self._jit_cache[cache_key] = jitted

        with friendly_solve_errors(max_steps, what="plant solve"):
            if t_eval is None:
                ts, ys = jitted(y0, params)
            else:
                ts, ys = jitted(y0, params, t_eval)
        if _time_factor != 1.0:
            ts = ts / _time_factor  # native -> requested unit
        sol = PlantSolution(t=ts, state=ys, plant=self)
        if time_unit is not None:
            sol._requested_time_unit = time_unit
        return sol

    def _solve_with_events(
        self,
        t0,
        t1,
        t_eval,
        params,
        y0,
        events,
        *,
        rtol,
        atol,
        dtmax,
        adjoint,
        max_steps,
        time_factor,
        time_unit,
        order=5,
        factormax=None,
        solver=None,
    ):
        """Run the monolithic plant solve with located events (the ``events=``
        path).

        Hands the assembled plant RHS to :func:`solve_with_events`, which splits
        the solve at the event times and applies their resets to the flat plant
        state between segments. Not routed through the per-instance jit cache: the
        driver is an eager segment loop (a state event's firing count is
        data-dependent), while time-only events still differentiate because each
        segment is a plain differentiable plant sub-solve.

        The state-invariant recycle map is precomputed once (from ``params``, so a
        time-event gradient still flows through it) and reused across every
        segment's RHS calls -- the same per-RHS saving the forward solve gets;
        ``None`` (the probe path) when the map is state-coupled or the constancy
        check has not run.
        """
        recycle_map = self._recycle._maybe_recycle_map(
            jnp.asarray(t0), self._split_state(y0), params
        )
        flow_map = self._recycle._maybe_flow_map(jnp.asarray(t0), self._split_state(y0), params)

        def rhs(t, y, args):
            return self._rhs(t, y, args, recycle_map=recycle_map, flow_map=flow_map)

        with friendly_solve_errors(max_steps, what="plant solve"):
            res = solve_with_events(
                rhs,
                y0,
                params,
                t0=t0,
                t1=t1,
                t_eval=t_eval,
                events=events,
                rtol=rtol,
                atol=atol,
                dtmax=dtmax,
                adjoint=adjoint,
                max_steps=max_steps,
                order=order,
                factormax=factormax,
                solver=solver,
            )
        ts = res.ts / time_factor if time_factor != 1.0 else res.ts
        sol = PlantSolution(t=ts, state=res.ys, plant=self, events_log=res.log)
        if time_unit is not None:
            sol._requested_time_unit = time_unit
        return sol

    def _build_jitted_solve(
        self,
        t0,
        t1,
        has_t_eval,
        *,
        event,
        rtol,
        atol,
        adjoint,
        dtmax,
        max_steps,
        progress_meter=None,
        solver=None,
        factormax=None,
        order=5,
    ):
        """Build the jit-compiled forward solve for one call signature.

        The returned ``_solve`` closes over the (static) plant RHS and the solver
        settings; ``y0``/``params``/``t_eval`` are its runtime arguments, so the
        same compiled solve serves any parameter or initial-state vector of the
        cached shapes (the parameter-sweep case). ``t_eval`` is a runtime
        argument (not baked in) so different save times of the same shape reuse
        one compile.

        When the recycle affine map ``M`` is state-independent
        (:meth:`_check_recycle_map_constant`), it is computed **once** per solve
        from the runtime ``params`` (so the gradient still flows and a parameter
        sweep stays correct) and passed into every RHS call as ``recycle_map=``,
        collapsing the per-RHS recycle resolution from ``n_recycle_edges + 1``
        sweeps to one. Bit-identical to probing it every call.
        """
        t0_arr = jnp.asarray(float(t0))

        def make_rhs(y0, params):
            states0 = self._split_state(y0)
            recycle_map = self._recycle._maybe_recycle_map(t0_arr, states0, params)
            flow_map = self._recycle._maybe_flow_map(t0_arr, states0, params)

            def rhs(t, y, args):
                return self._rhs(t, y, args, recycle_map=recycle_map, flow_map=flow_map)

            return rhs

        kw = dict(
            t0=t0,
            t1=t1,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
            dtmax=dtmax,
            max_steps=max_steps,
            event=event,
            progress_meter=progress_meter,
            solver=solver,
            factormax=factormax,
            order=order,
        )

        if has_t_eval:

            @jax.jit
            def _solve(y0, params, t_eval):
                sol = _run_diffeqsolve(
                    make_rhs(y0, params),
                    y0=y0,
                    args=params,
                    saveat=diffrax.SaveAt(ts=t_eval),
                    **kw,
                )
                return sol.ts, sol.ys

            return _solve

        @jax.jit
        def _solve(y0, params):
            sol = _run_diffeqsolve(
                make_rhs(y0, params),
                y0=y0,
                args=params,
                saveat=diffrax.SaveAt(t1=True),
                **kw,
            )
            return sol.ts, sol.ys

        return _solve

    def _build_jitted_forward_fast(self, t0, t1, *, rtol, atol):
        """Build the lean non-AD forward solve for one signature.

        Wraps :func:`~aquakin.integrate.forward_solve.forward_solve` -- a plain
        ``lax.while_loop`` adaptive ESDIRK with simplified Newton + colored
        Jacobian + direct ``lu_factor``/``lu_solve``, no diffrax adjoint /
        optimistix / lineax. The per-step Jacobian is still colored forward-mode
        AD (the same exact matrix the differentiable path uses), but the solve is
        NOT differentiable. ``y0``/``params``/``t_eval`` are runtime arguments so
        one compile serves a parameter sweep; the colored pattern (built+guarded
        before this is called) is baked into the closure. The recycle map is
        precomputed once per solve from ``params`` (the cached-M path).
        """
        from aquakin.integrate.forward_solve import forward_solve

        rf = self._colored_root_finder[0]
        S, col_of, pattern = rf.seed_matrix, rf.color_of, rf.pattern
        t0a = jnp.asarray(float(t0))

        @jax.jit
        def _solve(y0, params, t_eval):
            rmap = self._recycle._maybe_recycle_map(t0a, self._split_state(y0), params)
            fmap = self._recycle._maybe_flow_map(t0a, self._split_state(y0), params)

            def rhs(t, y, args):
                return self._rhs(t, y, args, recycle_map=rmap, flow_map=fmap)

            def jac(t, y, args):
                _, lin = jax.linearize(lambda yy: rhs(t, yy, args), y)
                JS = jax.vmap(lin, in_axes=1, out_axes=1)(S)
                return JS[:, col_of] * pattern

            ys = forward_solve(
                rhs, jac, y0, params, float(t0), float(t1), t_eval, rtol=rtol, atol=atol
            )
            return t_eval, ys

        return _solve

    def _build_jitted_stable_adjoint_solve(
        self,
        t0,
        t1,
        t_eval,
        *,
        rtol,
        atol,
        max_steps,
        forward_root_finder=None,
        solver=None,
        factormax=None,
        order=5,
        low_memory=False,
    ):
        """Build the jit-compiled cap-free stable-adjoint solve for one signature.

        Mirrors :meth:`_build_jitted_solve` but wraps
        :func:`~aquakin.esdirk_adjoint_solve` (the hand-written discrete adjoint),
        returning the saved states. ``time_dependent`` carries integration time in
        the state so the explicit time dependence of the plant RHS (a time-varying
        influent) is captured exactly; ``max_steps`` bounds the saved-trajectory
        buffer the backward scan walks. ``y0``/``params`` are runtime arguments,
        so one compile serves any parameter or initial-state vector of the cached
        shapes; ``t_eval`` is closed over because the discrete adjoint marks it
        non-differentiable (a traced ``t_eval`` cannot enter that slot).
        """

        @jax.jit
        def _solve(y0, params):
            return self._esdirk_stable_adjoint(
                y0,
                params,
                t0,
                t1,
                t_eval,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
                forward_root_finder=forward_root_finder,
                solver=solver,
                factormax=factormax,
                order=order,
                low_memory=low_memory,
            )

        return _solve

    def _esdirk_stable_adjoint(
        self,
        y0,
        params,
        t0,
        t1,
        t_eval,
        *,
        rtol,
        atol,
        max_steps,
        jacobian_builder=None,
        forward_root_finder=None,
        solver=None,
        factormax=None,
        order=5,
        low_memory=False,
    ):
        """Cap-free reverse-mode plant solve with the cached recycle map hoisted
        out of the backward pass.

        Caches the state-invariant recycle map once per solve and reuses it on the
        *primal* RHS (the forward solve + every backward stage / df/dy Jacobian),
        so the expensive per-call recycle probe is lifted out of the hot
        reverse-adjoint loop. The plain ``rhs`` -- which recomputes the map from
        ``params`` -- still drives the df/dtheta vjp, so the dM/dtheta term
        (nonzero only for flow-setpoint params) is exact; the gradient is
        bit-identical to probing on every call, just faster.

        The cached map must be ``stop_gradient``'d: it is a params-derived value
        closed over inside the discrete-adjoint custom VJP, and its parameter
        dependence is accounted for by the vjp's ``rhs``, not by this closure.
        When the map is not state-invariant (``_recycle_map_constant`` not True,
        e.g. a scheduled pump) the cache is ``None`` and the primal falls back to
        ``rhs`` (probe per call -- correct, just unoptimised). Shared by the
        cached jit closure and the under-trace gradient path so both optimise.
        """

        def rhs(t, y, args):
            return self._rhs(t, y, args)

        rmap = self._recycle._maybe_recycle_map(jnp.asarray(t0), self._split_state(y0), params)
        fmap = self._recycle._maybe_flow_map(jnp.asarray(t0), self._split_state(y0), params)
        if rmap is None:
            primal_rhs = None
        else:
            rmap = jax.lax.stop_gradient(rmap)
            # The cached flow map A is a params-derived value (it depends on the
            # flow-setpoint block); stop_gradient it before closing over it -- its
            # parameter dependence is the param-vjp's job, which keeps the
            # map-recomputing `rhs` (flow_map left None) so a flow-setpoint
            # gradient still captures dA/dtheta.
            fmap = None if fmap is None else jax.lax.stop_gradient(fmap)

            def primal_rhs(t, y, args):
                return self._rhs(t, y, args, recycle_map=rmap, flow_map=fmap)

        return esdirk_adjoint_solve(
            rhs,
            y0,
            params,
            (t0, t1),
            t_eval,
            solver=solver,
            order=order,
            factormax=factormax,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            time_dependent=True,
            primal_rhs=primal_rhs,
            jacobian_builder=jacobian_builder,
            forward_root_finder=forward_root_finder,
            low_memory=low_memory,
        )

    def run_to_steady_state(
        self,
        params: Optional[jnp.ndarray] = None,
        y0: Optional[jnp.ndarray] = None,
        *,
        max_time: float = 1000.0,
        ss_rtol: float = 1e-3,
        ss_atol: float = 1e-3,
        rtol: float = 1e-6,
        atol: Union[float, jnp.ndarray, None] = None,
        atol_factor: float = 1e-6,
        max_steps: int = 500_000,
    ) -> "SteadyStateResult":
        """Integrate forward until the plant settles to steady state.

        A single continuous adaptive solve that **terminates itself** the instant
        the dynamics die out -- diffrax's steady-state event halts when the
        vector field is approximately zero (``||dstate/dt|| <= ss_atol +
        ss_rtol*||state||``), the standard "march in time until the residual
        dies" criterion. There is no fixed horizon to guess and no chunked
        re-integration: ``max_time`` is only a safety cap, reached only if the
        plant has *not* settled (then ``converged`` is ``False``).

        The absolute tolerance for the underlying solve defaults to a
        per-component noise floor scaled off the operating magnitudes
        (``atol_i = atol_factor * max(|y0_i|, |default_i|)``, with a small global
        floor) -- the SUNDIALS/Hairer "vector atol" recommendation for states
        spanning many orders of magnitude -- so the engineer does not hand-tune
        tolerances for a g/m³ plant.

        Parameters
        ----------
        params : jnp.ndarray, optional
            Plant parameters (defaults to :meth:`default_parameters`).
        y0 : jnp.ndarray, optional
            Initial state / warm start (defaults to :meth:`initial_state`). A
            healthy warm start reaches steady state far faster than a cold start.
        max_time : float
            Safety cap on integration time (plant units, typically days).
        ss_rtol, ss_atol : float
            Steady-state event tolerances on ``||dstate/dt||`` (the convergence
            criterion). ``ss_rtol=1e-3`` ~ "no state changes faster than ~0.1%
            per unit time".
        rtol : float
            Relative tolerance of the underlying integrator.
        atol : float or jnp.ndarray, optional
            Absolute tolerance of the integrator. ``None`` (default) auto-scales
            per component (see above); pass a value to override.
        atol_factor : float
            Scale factor for the auto per-component ``atol``.
        max_steps : int
            Maximum integrator steps.

        Returns
        -------
        SteadyStateResult
            ``state`` (the operating point), ``converged``, ``time``, and the
            underlying ``solution``.

        Examples
        --------
        >>> ss = plant.run_to_steady_state(params, y0=warm)
        >>> ss.converged
        True
        >>> snh = plant.states_by_unit(ss.state)["tank5"][asm1.species_index["SNH"]]
        """
        self._build_state_layout()
        y0 = self.initial_state() if y0 is None else jnp.asarray(y0)
        if atol is None:
            atol = default_atol(y0, self.initial_state(), atol_factor=atol_factor)
        event = diffrax.Event(diffrax.steady_state_event(rtol=ss_rtol, atol=ss_atol))
        sol = self.solve(
            t_span=(0.0, float(max_time)),
            params=params,
            y0=y0,
            rtol=rtol,
            atol=atol,
            event=event,
            integrator=IntegratorConfig(max_steps=max_steps),
        )
        t_final = float(sol.t[-1])
        converged = bool(t_final < float(max_time))
        return SteadyStateResult(
            state=sol.state[-1],
            converged=converged,
            time=t_final,
            solution=sol,
            method="forward",
        )

    def _steady_continuation_fallback(
        self,
        rhs,
        params,
        y0,
        scale_floor,
        continuation_from,
        continuation_kwargs,
        *,
        dt0,
        dt_max,
        growth_cap,
        max_iter,
        tol,
        nonneg,
        influent_time,
    ):
        """Continuation fallback for a non-converged direct PTC solve.

        Reach the steady state at ``params`` by deforming from the known solution
        ``continuation_from = (params_known, y_known)`` -- predictor-corrector with
        the same PTC as the corrector -- for a parameter set whose direct warm
        start is out of basin. Returns a converged :class:`SteadyStateResult`, or
        ``None`` if continuation is not configured or did not reach the target (a
        fold the natural-parameter path cannot turn, left to the forward backstop /
        a future pseudo-arclength layer). The jitted kernels are cached per
        settings, so a screen over many targets from one known point compiles once.
        """
        if continuation_from is None:
            return None
        from aquakin.plant.steady import continuation_solve, make_continuation_kernels

        pk, yk = continuation_from
        pk = self._coerce_params(pk)
        yk = jnp.asarray(yk)
        ckey = (
            float(dt0),
            float(dt_max),
            float(growth_cap),
            int(max_iter),
            float(tol),
            bool(nonneg),
            float(influent_time),
        )
        cache = getattr(self, "_continuation_kernel_cache", None)
        if cache is None:
            cache = {}
            self._continuation_kernel_cache = cache
        kernels = cache.get(ckey)
        if kernels is None:
            ptc_kw = dict(
                dt0=dt0,
                dt_max=dt_max,
                growth_cap=growth_cap,
                max_iter=max_iter,
                tol=tol,
                scale_floor=jnp.asarray(scale_floor),
                nonneg=nonneg,
            )
            kernels = make_continuation_kernels(rhs, None, ptc_kw)
            cache[ckey] = kernels
        cres = continuation_solve(
            rhs, pk, yk, params, kernels=kernels, **(continuation_kwargs or {})
        )
        if bool(cres.converged):
            return SteadyStateResult(
                state=cres.state,
                converged=True,
                method="continuation",
                iterations=int(cres.corrector_iterations),
                residual=float(cres.residual),
            )
        return None

    def _steady_arclength_fallback(
        self,
        rhs,
        params,
        scale_floor,
        continuation_from,
        *,
        tol,
        nonneg,
        influent_time,
        arclength_kwargs=None,
    ):
        """Pseudo-arclength fallback: track the steady-state branch to a far target.

        Reaches operating points behind a near-singular ``dF/dy`` (where PTC and
        natural-parameter continuation overshoot -- the augmented arclength operator
        does not invert it), AND classifies existence: it detects when the operating
        branch **folds before the target** (a saddle-node bifurcation -- the
        operating point is past the survival limit, e.g. digester washout, so it
        does not exist there). Returns a converged result (``method="arclength"``,
        ``operating_point_exists=True``), a past-fold result (``method="past_fold"``,
        ``operating_point_exists=False`` -- a screen should *exclude* it), or
        ``None`` if inconclusive (the caller falls through to the forward backstop).
        Kernels are cached per settings (the ``rhs`` and the known-solution-derived
        scale are fixed across a sweep from one known point).
        """
        if continuation_from is None:
            return None
        from aquakin.plant.steady import arclength_continuation_solve, make_arclength_kernels

        pk, yk = continuation_from
        pk = self._coerce_params(pk)
        yk = jnp.asarray(yk)
        scale = jnp.maximum(jnp.abs(yk), 1e-3)
        ptc_kw = dict(scale_floor=jnp.asarray(scale_floor), tol=tol, nonneg=nonneg)
        ckey = (float(tol), bool(nonneg), float(influent_time))
        cache = getattr(self, "_arclength_kernel_cache", None)
        if cache is None:
            cache = {}
            self._arclength_kernel_cache = cache
        kernels = cache.get(ckey)
        if kernels is None:
            kernels = make_arclength_kernels(rhs, scale, ptc_kw)
            cache[ckey] = kernels
        res = arclength_continuation_solve(
            rhs,
            pk,
            yk,
            params,
            kernels=kernels,
            scale=scale,
            ptc_kwargs=ptc_kw,
            **(arclength_kwargs or {}),
        )
        if res.status == "converged":
            return SteadyStateResult(
                state=res.state,
                converged=True,
                method="arclength",
                iterations=int(res.corrector_iterations),
                residual=float(res.residual),
                operating_point_exists=True,
            )
        if res.status == "past_fold":
            return SteadyStateResult(
                state=res.state,
                converged=False,
                method="past_fold",
                iterations=int(res.corrector_iterations),
                residual=float(res.residual),
                operating_point_exists=False,
            )
        return None

    def steady_state(
        self,
        params: Optional[jnp.ndarray] = None,
        y0: Optional[jnp.ndarray] = None,
        *,
        influent_time: float = 0.0,
        dt0: float = 1e-2,
        dt_max: float = 1e10,
        growth_cap: float = 10.0,
        max_iter: int = 400,
        tol: float = 1e-6,
        scale_floor: Optional[float] = None,
        nonneg: bool = True,
        design: Optional[dict] = None,
        colored_jacobian: bool = False,
        fallback: bool = True,
        fallback_kwargs: Optional[dict] = None,
        continuation_from: Optional[tuple] = None,
        continuation_kwargs: Optional[dict] = None,
        arclength: bool = True,
        arclength_kwargs: Optional[dict] = None,
    ) -> "SteadyStateResult":
        """Solve the plant steady state algebraically by pseudo-transient continuation.

        Finds the root of the plant right-hand side ``F(y) = dy/dt = 0`` directly,
        rather than integrating forward until the dynamics die out
        (:meth:`run_to_steady_state`). Pseudo-transient continuation takes
        damped-Newton steps ``(V/dt - J) dy = F`` with the exact AD Jacobian
        ``J`` and a per-state pseudo-time ``V/dt`` that ramps from a stable
        time-stepping move (far from the root) to a full Newton step (near it),
        so it is robust on stiff plants (long-SRT digesters) where a plain
        Newton root-find stalls. Typically reaches steady state ~10x faster than
        the forward solve and to a tighter residual; see
        :mod:`aquakin.plant.steady`.

        **Differentiable.** The returned ``state`` carries the
        implicit-function-theorem gradient with respect to ``params`` (the
        iteration itself is gradient-blocked), so ``jax.grad`` of a loss on the
        steady state flows to the plant parameters -- the intended path for
        design sweeps. The diagnostic fields (``converged``, ``iterations``,
        ``residual``) and the forward fallback are only available in eager use;
        under an outer ``jit``/``grad`` trace they are traced values and the
        fallback is skipped.

        Parameters
        ----------
        params : jnp.ndarray, optional
            Plant parameters (defaults to :meth:`default_parameters`).
        y0 : jnp.ndarray, optional
            Warm start (defaults to :meth:`initial_state`). A healthy warm start
            (e.g. ``bsm2_warm_start``) converges in fewer iterations.
        influent_time : float
            Time at which to read the (constant) influent for the steady
            residual. The steady state is only well defined for a constant load;
            for a time-varying influent this samples it at ``influent_time``.
        dt0, dt_max, growth_cap : float
            Pseudo-transient controls: initial / maximum pseudo-timestep and the
            per-step SER growth cap (see :func:`aquakin.plant.steady.ptc_forward`).
        max_iter : int
            PTC iteration cap.
        tol : float
            Convergence tolerance on the scaled residual.
        scale_floor : float or array, optional
            Floor on ``|y|`` in the per-state pseudo-time / residual scaling (so
            near-zero states do not distort the damping or the convergence
            criterion). Default ``None`` builds a **per-state** floor
            ``max(|y0|, 1e-6)`` -- each state scaled by its own warm-start
            magnitude, which roughly halves the iteration count on a stiff
            multi-network plant (the flat scalar floor over-damps small-magnitude
            states). Pass a scalar or per-state array to override.
        nonneg : bool
            Clamp the state to ``>= 0`` each step (concentrations are
            non-negative).
        design : dict, optional
            Differentiable design-variable overrides folded into the quantity the
            implicit-function-theorem gradient is taken w.r.t., so a design sweep
            can ``jax.grad`` / ``jacobian`` the steady state through them.
            Currently supports the **influent load**:
            ``design={"influent": {port: {"Q": ..., "C": ..., "T": ...}}}`` (plain
            arrays; ``T`` optional), which replaces the recorded influent at
            ``influent_time``. Example -- sensitivity of effluent ammonia to the
            influent ammonia load::

                jax.grad(lambda c: plant.steady_state(
                    p, y0, design={"influent": {"feed": {"Q": Q, "C": c}}}
                ).state[eff_idx])(C_influent)
        colored_jacobian : bool
            Materialize the PTC iteration Jacobian ``dF/dy`` by column-compressed
            colored AD (one Jacobian-vector product per color rather than per
            state; BSM2 46 colors vs 167) instead of dense ``jax.jacfwd``. The
            operating point is unchanged (the colored matrix equals the dense one
            on its sparsity-pattern support: bit-identical on a single-network
            plant, identical to PTC tolerance on a multi-network plant where the
            recycle solve differs by round-off); only the per-iteration Jacobian
            cost drops. Built and guarded once concretely (falls back to dense
            with a warning on a start-state mismatch, or under a ``jit``/``grad``
            trace where the probe cannot run). **Benefit is regime-specific**
            (measured BSM2): the Jacobian build is ~2.4x cheaper and the whole
            solve ~1.9x faster *run-only under jit*, but an un-jitted one-shot
            call is compile/trace-bound, so the one-time pattern build makes it
            ~0.8x (slower). Worth enabling only when the steady solve is run
            **repeatedly under jit** (differentiable design sweeps, where compile
            amortizes) or for a much larger plant; not for a single steady state.
            Default ``False``.
        fallback : bool
            If PTC does not converge within ``max_iter`` (eager use only), fall
            back to :meth:`run_to_steady_state` and return that result (with
            ``method="ptc->forward"``).
        fallback_kwargs : dict, optional
            Extra keyword arguments for the forward fallback.

        Returns
        -------
        SteadyStateResult
            ``state`` (the operating point, IFT-differentiable), ``converged``,
            ``method="ptc"``, ``iterations`` and ``residual``.

        Examples
        --------
        >>> ss = plant.steady_state(params, y0=warm)
        >>> ss.converged, ss.iterations
        (True, 75)
        >>> g = jax.grad(lambda p: plant.steady_state(p, y0=warm).state[idx])(params)
        """
        from aquakin.plant.steady import ptc_forward, solve_steady_state

        self._build_state_layout()
        self._build_parameter_layout()
        params = self.default_parameters() if params is None else self._coerce_params(params)
        y0 = self.initial_state() if y0 is None else jnp.asarray(y0)
        t = jnp.asarray(float(influent_time))

        # Per-state pseudo-time / residual scaling. The PTC step damping V and the
        # convergence criterion both use ``max(|y|, scale_floor)``; a flat scalar
        # floor over-damps the small-magnitude states (gas fractions ~1e-3,
        # dissolved hydrogen ~1e-7), throttling their residual and so the SER ramp,
        # which roughly doubles the iteration count. Defaulting the floor to each
        # state's own warm-start magnitude ``max(|y0|, 1e-6)`` gives every state a
        # magnitude-consistent scale -- ~half the iterations on BSM2 (80 -> 38),
        # neutral on BSM1, same root. The small 1e-6 absolute floor anchors
        # near-zero states (a pure-|y| relative scale destabilises the iteration).
        # An explicit ``scale_floor`` (scalar or per-state array) is honoured.
        if scale_floor is None:
            scale_floor = jnp.maximum(jnp.abs(y0), 1e-6)

        # When design variables are supplied, the differentiated quantity is the
        # pytree ``theta = (params, design)`` -- the implicit-function-theorem
        # gradient then flows to BOTH the kinetic parameters and the design
        # overrides (e.g. the influent load). Without design, theta is just the
        # parameter vector (unchanged path).
        if design is None:

            def rhs(y, theta):
                return self._rhs(t, y, theta)

            theta = params
        else:

            def rhs(y, theta):
                p, d = theta
                return self._rhs(t, y, p, design=d)

            theta = (params, design)

        # Colored PTC needs a leak-free, cached-recycle-map forward rhs: the
        # per-call recycle probing in ``_rhs(recycle_map=None)`` leaks a traced
        # intermediate under the ``jit`` used for the structural pattern probe on
        # a multi-network plant. The cached map equals the probed map, so the
        # operating point is unchanged; it is built only on the concrete,
        # constant-map, design-free path (else fall back to dense). The gradient
        # keeps the map-recomputing ``rhs`` (primal_rhs is forward-only), so a
        # flow-setpoint parameter retains its recycle-map dependence.
        jac_fn = None
        primal_rhs = None
        if colored_jacobian:
            concrete = not any(
                isinstance(leaf, jax.core.Tracer)
                for leaf in jax.tree_util.tree_leaves((params, y0))
            )
            if design is None and concrete:
                self._recycle._check_recycle_map_constant(t, y0, params)
                self._recycle._check_flow_map_constant(t, y0, params)
                rmap = self._recycle._maybe_recycle_map(t, self._split_state(y0), params)
                fmap = self._recycle._maybe_flow_map(t, self._split_state(y0), params)
                if rmap is not None:

                    def primal_rhs(y, theta):
                        return self._rhs(t, y, theta, recycle_map=rmap, flow_map=fmap)

                    jac_fn = self._colored_steady_jacobian_builder(primal_rhs, y0, theta, tol=tol)
            if jac_fn is None:
                primal_rhs = None  # colored unavailable -> dense forward too

        # Compiled-solve cache (the single-run compile lever). The eager
        # ``while_loop`` in ``ptc_forward`` re-traces and recompiles on EVERY call
        # (~12-17 s for BSM2, dominated by the plant-RHS ``jacfwd``), so a repeated
        # concrete ``steady_state`` -- a temperature/SRT sweep, multistart, or
        # regenerating a figure -- pays that compile each time. Persisting a jitted
        # forward solver and reusing it lets JAX skip the recompile (~40 ms run
        # thereafter). Cached only on the **dense, design-free, concrete** path:
        # the ``rhs`` recomputes the recycle map from the argument ``params``, so
        # one compiled solver is correct for any params (a sweep); the colored
        # primal bakes a params-derived map, the ``design`` path differentiates a
        # pytree, and a traced (gradient) call needs the IFT ``custom_vjp`` and is
        # amortized by the caller's own ``jit`` -- those keep ``solve_steady_state``.
        under_trace = any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree_util.tree_leaves((theta, y0, scale_floor))
        )
        if design is None and jac_fn is None and not under_trace:
            key = (
                float(dt0),
                float(dt_max),
                float(growth_cap),
                int(max_iter),
                float(tol),
                bool(nonneg),
                float(influent_time),
            )
            jitted = self._steady_jit_cache.get(key)
            if jitted is None:

                def _fwd(y0_, params_, scale_floor_):
                    return ptc_forward(
                        rhs,
                        params_,
                        y0_,
                        dt0=dt0,
                        dt_max=dt_max,
                        growth_cap=growth_cap,
                        max_iter=max_iter,
                        tol=tol,
                        scale_floor=scale_floor_,
                        nonneg=nonneg,
                    )

                jitted = jax.jit(_fwd)
                self._steady_jit_cache[key] = jitted
            y_star, residual_a, iters_a, conv_a = jitted(y0, params, jnp.asarray(scale_floor))
            converged = bool(conv_a)
            if not converged:
                alt = self._steady_continuation_fallback(
                    rhs,
                    params,
                    y0,
                    scale_floor,
                    continuation_from,
                    continuation_kwargs,
                    dt0=dt0,
                    dt_max=dt_max,
                    growth_cap=growth_cap,
                    max_iter=max_iter,
                    tol=tol,
                    nonneg=nonneg,
                    influent_time=influent_time,
                )
                if alt is not None:
                    return alt
                if arclength:
                    arc = self._steady_arclength_fallback(
                        rhs,
                        params,
                        scale_floor,
                        continuation_from,
                        tol=tol,
                        nonneg=nonneg,
                        influent_time=influent_time,
                        arclength_kwargs=arclength_kwargs,
                    )
                    if arc is not None:
                        return arc
                if fallback:
                    fb = self.run_to_steady_state(params, y0=y0, **(fallback_kwargs or {}))
                    fb.method = "ptc->forward"
                    return fb
            return SteadyStateResult(
                state=y_star,
                converged=converged,
                method="ptc",
                iterations=int(iters_a),
                residual=float(residual_a),
            )

        res = solve_steady_state(
            rhs,
            theta,
            y0,
            jac_fn=jac_fn,
            primal_rhs=primal_rhs,
            dt0=dt0,
            dt_max=dt_max,
            growth_cap=growth_cap,
            max_iter=max_iter,
            tol=tol,
            scale_floor=scale_floor,
            nonneg=nonneg,
        )

        # Eager use gets concrete diagnostics and the forward fallback; under an
        # outer trace ``bool(...)`` raises, so we keep the traced values and skip
        # the (un-jittable) fallback branch.
        try:
            converged = bool(res.converged)
            iterations = int(res.iterations)
            residual = float(res.residual)
        except jax.errors.ConcretizationTypeError:
            # Under an outer jit/grad trace the diagnostics are traced values;
            # return them as-is and skip the (un-jittable) fallback branch.
            return SteadyStateResult(
                state=res.state,
                converged=res.converged,
                method="ptc",
                iterations=res.iterations,
                residual=res.residual,
            )

        if not converged:
            alt = self._steady_continuation_fallback(
                rhs,
                params,
                y0,
                scale_floor,
                continuation_from,
                continuation_kwargs,
                dt0=dt0,
                dt_max=dt_max,
                growth_cap=growth_cap,
                max_iter=max_iter,
                tol=tol,
                nonneg=nonneg,
                influent_time=influent_time,
            )
            if alt is not None:
                return alt
            if arclength:
                arc = self._steady_arclength_fallback(
                    rhs,
                    params,
                    scale_floor,
                    continuation_from,
                    tol=tol,
                    nonneg=nonneg,
                    influent_time=influent_time,
                    arclength_kwargs=arclength_kwargs,
                )
                if arc is not None:
                    return arc
            if fallback:
                fb = self.run_to_steady_state(params, y0=y0, **(fallback_kwargs or {}))
                fb.method = "ptc->forward"
                return fb

        return SteadyStateResult(
            state=res.state,
            converged=converged,
            method="ptc",
            iterations=iterations,
            residual=residual,
        )

    # -- Operating-condition sensitivity inputs (shared steady + dynamic) -------
    #
    # A single spec + helper so every sensitivity entry point -- the steady-state
    # implicit-function-theorem path and the dynamic augmented-variational path,
    # and their dgsm screens -- accepts an IDENTICAL operating-input description.
    # This is the unification point: a new operating kind is added here once and
    # both stacks gain it, and the parity guard test asserts they stay in step.
    def _parse_operating(self, operating):
        """Validate operating-condition sensitivity specs into ``op_meta``.

        Each spec is one of

        * ``{"kind": "influent_flow", "port": p}`` -- a differentiable
          multiplicative scale (nominal ``1.0``) on that influent's flow, or
        * ``{"kind": "influent_concentration", "port": p, "species": s}`` -- a
          scale on one species' influent load.

        Returns a list of ``(port, is_flow, species_idx, n_species)`` tuples, one
        per operating parameter, in the order given.
        """
        op_meta = []
        for spec in operating or []:
            port = spec["port"]
            if port not in self.influents:
                raise KeyError(
                    f"operating influent port {port!r} is not an influent of this "
                    f"plant; have {sorted(self.influents)}."
                )
            net = self.influents[port].network
            kind = spec.get("kind")
            if kind == "influent_flow":
                op_meta.append((port, True, -1, int(net.n_species)))
            elif kind == "influent_concentration":
                sp = spec["species"]
                if sp not in net.species_index:
                    raise KeyError(
                        f"operating species {sp!r} is not in the influent network "
                        f"for port {port!r}."
                    )
                op_meta.append((port, False, int(net.species_index[sp]), int(net.n_species)))
            else:
                raise ValueError(
                    f"operating spec 'kind' must be 'influent_flow' or "
                    f"'influent_concentration'; got {kind!r}."
                )
        return op_meta

    @staticmethod
    def _operating_design(op_meta, op_vals):
        """Build the influent ``design`` dict that applies the operating scales
        ``op_vals`` (one per ``op_meta`` entry, nominal ``1.0``).

        A flow spec sets ``Q_scale``; a concentration spec sets the species' entry
        of a per-species ``C_scale`` (ones elsewhere). The design threads through
        the same ``_resolve_streams`` / ``_flow_one_pass`` override the
        steady-state IFT uses, so the cached recycle/flow maps stay exact.
        """
        influent: dict = {}
        for k, (port, is_flow, sp_idx, nsp) in enumerate(op_meta):
            d = influent.setdefault(port, {})
            if is_flow:
                d["Q_scale"] = op_vals[k]
            else:
                cs = d.get("C_scale", jnp.ones(nsp))
                d["C_scale"] = cs.at[sp_idx].set(op_vals[k])
        return {"influent": influent}

    # ------------------------------------------------------------------
    # Sensitivity / uncertainty-quantification surface.
    #
    # Implemented as free functions taking the plant as their first argument
    # in :mod:`aquakin.plant.sensitivity`, and bound here as methods so the
    # public ``plant.steady_state_dgsm(...)`` API is unchanged. They only use
    # the public solve API plus the parameter/state-layout helpers.
    # ------------------------------------------------------------------
    steady_state_sensitivity = _sensitivity.steady_state_sensitivity
    steady_state_dgsm = _sensitivity.steady_state_dgsm
    solve_sensitivity = _sensitivity.solve_sensitivity
    dynamic_sensitivity = _sensitivity.dynamic_sensitivity
    dynamic_dgsm = _sensitivity.dynamic_dgsm
