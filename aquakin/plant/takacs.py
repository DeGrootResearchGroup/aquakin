"""Takács 1-D layered secondary clarifier (Takács, Patry & Nolasco 1991).

The clarifier is modelled as a stack of well-mixed layers with vertical
flux exchange:

- **Soluble species** (e.g. SS, SNH, SNO) don't settle. By default they are
  computed by mass balance on the inlet flow split between overflow and
  underflow: Q_overflow + Q_underflow = Q_in, and C_soluble is identical in
  both (pass-through, no holdup). With the opt-in ``soluble_holdup=True`` they
  instead occupy the layers as well-mixed states advected by the bulk flow
  (convection only, no settling), so the clarifier's liquid volume damps the
  soluble signal -- the overflow soluble is the top layer, the underflow the
  bottom. A non-reacting soluble reaches a uniform = feed concentration at
  steady state (overflow = underflow = feed), so the holdup changes only the
  dynamic response, not the steady state.
- **Particulate species** (e.g. XS, XB_H, XB_A) settle at a velocity
  given by the Takács double-exponential function::

      v_s(X) = max(0, min(v0', v0 * (exp(-rh * (X - X_min))
                                     - exp(-rp * (X - X_min)))))

  where X_min = fns * X_in (the non-settleable fraction). Each layer's
  particulate mass balance combines convective transport (down for
  layers at or below the feed; up for layers above) with the settling
  flux from the layer above and to the layer below.

The clarifier exposes ``overflow`` and ``underflow`` output ports. The
overflow flow rate is set as a property of the unit; the underflow takes
the remainder.

References
----------
Takács, I., Patry, G.G., & Nolasco, D. (1991). A dynamic model of the
clarification-thickening process. Water Research, 25(10), 1263-1271.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import (
    ASM1_SETTLING_SPECIES,
    ASM1_TSS_FACTOR,
    ASM1_TSS_SPECIES,
    species_indices,
)
from aquakin.plant._flow_split import (
    split_controlled_flows,
    validate_controlled_split,
)
from aquakin.plant.coupling import CouplingAware
from aquakin.plant.flow_setpoint import FlowParameterized, FlowSetpoint
from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


# A standard ASM1 ↔ TSS conversion used to identify the "particulate
# mass" that settles. Each particulate species contributes to the
# settling solid concentration in proportion to its COD-to-TSS factor.
# XND settles with XS but is N attached to it, not separate solids, so it
# carries a zero TSS factor.
_DEFAULT_TSS_FACTORS = {
    **dict.fromkeys(ASM1_TSS_SPECIES, ASM1_TSS_FACTOR),
    "XND": 0.0,
}

# Standard BSM1 Takács parameter set (Alex et al. 2008, Table 1.7).
_BSM1_TAKACS_DEFAULTS = dict(
    v0=474.0,  # max theoretical settling velocity, m/d
    vmax=250.0,  # max practical settling velocity, m/d
    rh=5.76e-4,  # hindered settling parameter, m³/g
    rp=2.86e-3,  # flocculant settling parameter, m³/g
    fns=2.28e-3,  # non-settleable fraction
    # Clarification-zone flux-limiting threshold (g/m³): above the feed, the
    # downward settling flux is limited by the layer below only when that layer
    # exceeds this concentration (Takács 1991). NOT a settling-velocity cutoff.
    X_threshold=3000.0,
)


@dataclass
class TakacsClarifier(FlowParameterized, CouplingAware):
    """A 10-layer 1-D secondary clarifier.

    Parameters
    ----------
    name : str
    model : CompiledModel
        The model whose species ordering applies to the inlet and
        outlet streams.
    area : float
        Cross-sectional area (m²).
    height : float
        Total settler depth (m). Divided equally among layers.
    n_layers : int
        Number of layers (default 10 per BSM1).
    feed_layer : int
        Index of the layer where the inlet enters (0-based from the
        bottom; BSM1 uses index 5 for a 10-layer clarifier, counting
        from the bottom).
    overflow_Q : float, optional
        Fixed overflow (clarified effluent) flow rate, m³/d; underflow takes
        ``Q_in - overflow_Q``. Supply exactly one of ``overflow_Q`` /
        ``underflow_Q``.
    underflow_Q : float, optional
        Fixed underflow flow rate, m³/d (the controlled RAS+wastage pump flow
        ``Q_r + Q_w``); the effluent overflow is then the remainder and tracks
        the feed (``Q_e = Q_f - Q_u``). Preferred under dynamic influent -- a
        fixed overflow forces a near-singular recycle-flow gain. Supply exactly
        one of ``overflow_Q`` / ``underflow_Q``.
    particulate_species : list[str]
        Species names that settle. Defaults to ASM1's standard set:
        XS, XI, XB_H, XB_A, XP, XND.
    settling_params : dict[str, float]
        Takács settling parameters (v0, vmax, rh, rp, fns, X_threshold).
        Defaults to the standard BSM1 values.
    composition_mode : str
        How the particulate composition of the outlet streams is determined.

        - ``"per_species"`` (default): the state carries every particulate
          species per layer, shape ``(n_layers, n_part)``. Each layer keeps its
          own evolving composition, and the outlet particulates are read straight
          from the relevant boundary layer. This carries per-layer composition
          memory.
        - ``"lumped_tss"``: the state carries a single total-suspended-solids
          value per layer, shape ``(n_layers,)``. The settling/convection
          dynamics act on that one TSS variable, and each outlet stream's
          particulate composition is the *instantaneous feed* composition scaled
          by the boundary-layer-to-feed TSS ratio. The two modes agree at steady
          state but diverge under dynamic flow, because the lumped form has no
          per-layer composition memory.
    soluble_holdup : bool
        If ``True`` (default ``False``), the soluble species also occupy the
        layers as well-mixed states advected by the bulk flow (no settling), so
        the clarifier's liquid volume (~``area × height``) damps the soluble
        effluent signal. The overflow soluble is the top layer, the underflow
        the bottom; a soluble holdup block of shape ``(n_layers, n_soluble)`` is
        appended to the tail of the state vector (so the particulate state
        layout and ``state_size`` are unchanged when this is off). Orthogonal to
        ``composition_mode``. A non-reacting soluble reaches a uniform = feed
        concentration at steady state, so this leaves every steady state
        unchanged and only adds the dynamic hydraulic smoothing (the BSM2
        ``settler1dv5`` behaviour, where each soluble is carried per layer).
    input_port : str
    overflow_port : str
    underflow_port : str
    """

    name: str
    model: "CompiledModel"
    area: float
    height: float
    overflow_Q: "float | None" = None
    underflow_Q: "float | None" = None
    n_layers: int = 10
    feed_layer: int = 5
    # Initial-condition operating point for a settled-blanket start (state-point
    # analysis). When ``init_underflow_Q`` is given, ``initial_state`` seeds a
    # settled profile -- clarification layers near the non-settleable floor, a
    # thickened bottom blanket -- instead of a uniform fill, which avoids the
    # violent startup transient (the blanket forming from scratch through the
    # flux kinks). ``init_feed_tss`` is the expected feed TSS (MLSS); if None it
    # is taken from the model's default particulate concentrations.
    init_underflow_Q: "float | None" = None
    init_feed_tss: "float | None" = None
    particulate_species: list[str] = field(default_factory=lambda: list(ASM1_SETTLING_SPECIES))
    settling_params: dict[str, float] = field(default_factory=lambda: dict(_BSM1_TAKACS_DEFAULTS))
    tss_factors: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_TSS_FACTORS))
    composition_mode: str = "per_species"
    soluble_holdup: bool = False
    input_port: str = "inlet"
    overflow_port: str = "overflow"
    underflow_port: str = "underflow"

    def __post_init__(self) -> None:
        # Ordered: index resolution (validates species) feeds the geometry
        # precompute, which reads the resolved indices; the settling-param
        # unpack is independent and runs last.
        self._validate()
        self._resolve_species_indices()
        self._precompute_geometry()
        self._unpack_settling_params()

    def _validate(self) -> None:
        """Validate the configuration that does not depend on resolved indices."""
        if self.composition_mode not in ("per_species", "lumped_tss"):
            raise ValueError(
                f"TakacsClarifier '{self.name}': composition_mode must be "
                f"'per_species' or 'lumped_tss'; got {self.composition_mode!r}."
            )
        if self.feed_layer < 0 or self.feed_layer >= self.n_layers:
            raise ValueError(f"feed_layer must be in [0, {self.n_layers}); got {self.feed_layer}")
        validate_controlled_split(
            f"TakacsClarifier '{self.name}'", self.overflow_Q, self.underflow_Q
        )

    def _resolve_species_indices(self) -> None:
        """Resolve particulate / soluble species to model indices and TSS factors."""
        self._part_indices = species_indices(
            self.model,
            self.particulate_species,
            what=f"TakacsClarifier '{self.name}': particulate species",
        )
        self._part_tss_factors = [self.tss_factors.get(sp, 1.0) for sp in self.particulate_species]
        self._n_part = len(self._part_indices)
        # Soluble = everything not in particulate.
        self._soluble_indices = [
            i for i in range(self.model.n_species) if i not in self._part_indices
        ]
        self._n_sol = len(self._soluble_indices)

    def _precompute_geometry(self) -> None:
        """Precompute the constant index/factor arrays and per-layer geometric
        masks ONCE, so the per-RHS-call hot path does not rebuild them with
        ``jnp.asarray`` of a Python list / ``astype`` every step."""
        self._part_idx_arr = jnp.asarray(self._part_indices)
        self._sol_idx_arr = jnp.asarray(self._soluble_indices)
        self._factors_arr = jnp.asarray(self._part_tss_factors)
        layer_idx = jnp.arange(self.n_layers)
        self._is_below_feed = (layer_idx < self.feed_layer).astype(jnp.float64)
        self._is_above_feed = (layer_idx > self.feed_layer).astype(jnp.float64)
        self._is_feed = (layer_idx == self.feed_layer).astype(jnp.float64)

    def _unpack_settling_params(self) -> None:
        """Wire the controlled-outflow setpoint and unpack the settling params to
        scalar attributes read by ``rhs`` without per-call Python branching."""
        # The controlled outflow is a differentiable flow setpoint -- but only
        # when it is a constant; a *scheduled* setpoint (a PiecewiseConstant with
        # ``.at(t)``) is left as the schedule (not a single parameter).
        self._setpoints = {}
        ctrl_name = "underflow_Q" if self.underflow_Q is not None else "overflow_Q"
        ctrl_val = getattr(self, ctrl_name)
        if ctrl_val is not None and not hasattr(ctrl_val, "at"):
            self._setpoints[ctrl_name] = FlowSetpoint(float(ctrl_val), 0)
        self._v0 = float(self.settling_params["v0"])
        self._vmax = float(self.settling_params["vmax"])
        self._rh = float(self.settling_params["rh"])
        self._rp = float(self.settling_params["rp"])
        self._fns = float(self.settling_params["fns"])
        self._X_threshold = float(self.settling_params["X_threshold"])

    @property
    def _part_block_size(self) -> int:
        """Size of the particulate portion of the state (the head block)."""
        if self.composition_mode == "lumped_tss":
            return self.n_layers
        return self.n_layers * self._n_part

    @property
    def state_size(self) -> int:
        size = self._part_block_size
        if self.soluble_holdup:
            size += self.n_layers * self._n_sol
        return size

    def _unpack(self, state: jnp.ndarray):
        """Split the flat state into ``(particulate_state, soluble_layered)``.

        The particulate head block keeps its existing layout (per the
        ``composition_mode``); the soluble holdup, when enabled, is a tail block
        reshaped to ``(n_layers, n_soluble)``. Returns ``soluble_layered = None``
        when ``soluble_holdup`` is off.
        """
        if not self.soluble_holdup:
            return state, None
        pb = self._part_block_size
        part = state[:pb]
        sol = state[pb:].reshape((self.n_layers, self._n_sol))
        return part, sol

    @property
    def volume(self) -> float:
        """Total liquid volume (m³) = settling area × depth.

        Exposed so a :class:`~aquakin.plant.temperature.HeatBalanceTemperature`
        model gives the settler a (well-mixed) bulk temperature state; it does
        not enter the settling physics, which are layered.
        """
        return float(self.area) * float(self.height)

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.overflow_port, self.underflow_port]

    def coupling_pattern(self):
        """Structural Jacobian sparsity (issue #388), AD-derived.

        The Takacs settling velocity ``v_s(X)`` and the feed flux-split are
        nonlinear, so the settler's couplings switch regime with the solids load
        and a single-operating-point probe goes stale on them. Unlike Monod
        kinetics (whose saturated terms are numerically invisible at any state),
        the settling law is a smooth nonlinearity whose every branch is exercised
        by sampling diverse physical profiles -- so AD of the RHS unioned over
        such states (:func:`aquakin.plant.coupling.ad_union`) gives a structural
        superset. ``params=None`` resolves the default flow setpoints, so this is
        a standalone derivation. ``self`` = d(rhs)/d(layer state); ``inlet`` =
        d(rhs)/d(feed concentration).
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        from aquakin.plant.coupling import CouplingPattern, ad_union
        from aquakin.plant.streams import Stream

        net = self.model
        state0 = np.asarray(self.initial_state())
        base_C = np.asarray(net.default_concentrations())
        Q0 = jnp.asarray(2.0e4)  # representative positive throughput
        t0 = jnp.asarray(0.0)

        def make_inputs(C):
            return {self.input_port: Stream(Q=Q0, C=C, model=net)}

        inlet0 = make_inputs(jnp.asarray(np.maximum(np.abs(base_C), 1e-3)))
        self_jac = lambda s: jax.jacfwd(lambda x: self.rhs(t0, x, inlet0, None))(s)
        self_pat = ad_union(self_jac, state0)

        state_fixed = jnp.asarray(np.maximum(np.abs(state0), 1e-3))
        inlet_jac = lambda c: jax.jacfwd(lambda C: self.rhs(t0, state_fixed, make_inputs(C), None))(
            c
        )
        inlet_pat = ad_union(inlet_jac, base_C)
        return CouplingPattern(self_pattern=self_pat, inlet_pattern=inlet_pat)

    def initial_state(self) -> jnp.ndarray:
        part_state = self._particulate_initial_state()
        if not self.soluble_holdup:
            return part_state
        # Seed every layer at the model's default soluble concentrations; a
        # non-reacting soluble relaxes to the feed concentration anyway, so the
        # seed only affects the initial transient.
        defaults = self.model.default_concentrations()
        sol_defaults = jnp.asarray([float(defaults[i]) for i in self._soluble_indices])
        sol_block = jnp.tile(sol_defaults, self.n_layers)  # (n_layers * n_sol,)
        return jnp.concatenate([part_state, sol_block])

    def _particulate_initial_state(self) -> jnp.ndarray:
        """The particulate head block of the initial state (layout per
        ``composition_mode``); unchanged by ``soluble_holdup``."""
        defaults = self.model.default_concentrations()
        part_defaults = jnp.asarray([float(defaults[i]) for i in self._part_indices])

        if self.init_underflow_Q is None:
            # Backward-compatible uniform seed: every layer at the inlet's
            # default particulate concentration.
            if self.composition_mode == "lumped_tss":
                # The single per-layer TSS is the default composition summed to
                # total solids, identical in every layer.
                tss = float(jnp.sum(part_defaults * self._factors_arr))
                return jnp.full((self.n_layers,), tss)
            return jnp.tile(part_defaults, self.n_layers)

        tss = self._initial_blanket_tss(part_defaults)
        if self.composition_mode == "lumped_tss":
            return tss

        # Apportion each layer's TSS across particulate species by the feed
        # composition: C_k = d_k * (TSS_layer / X_f), so sum_k C_k * f_k = TSS.
        X_f = float(jnp.sum(part_defaults * self._factors_arr))
        scale = tss / X_f  # (n_layers,)
        state = scale[:, None] * part_defaults[None, :]  # (n_layers, n_part)
        return state.reshape(-1)

    def _initial_blanket_tss(self, part_defaults: jnp.ndarray) -> jnp.ndarray:
        """Per-layer TSS (g/m³) for a settled-blanket start, shape ``(n_layers,)``.

        Builds a state-point-analysis profile -- clarification (above-feed)
        layers near the non-settleable floor, a thickened bottom blanket -- from
        the model's default particulate concentrations and the blanket
        operating point. Layer 0 is the bottom (underflow); ``n_layers - 1`` is
        the top (effluent). Shared by both composition modes.

        Parameters
        ----------
        part_defaults : jnp.ndarray
            Per-species default particulate concentrations, shape ``(n_part,)``.

        Returns
        -------
        jnp.ndarray
            Per-layer total-suspended-solids profile, shape ``(n_layers,)``.
        """
        # --- State-point-analysis settled blanket ----------------------
        # Feed TSS (MLSS) the blanket is built for. The clarifier feed at
        # startup is the reactors' initial mixed-liquor, so default to the TSS
        # implied by the model's particulate defaults.
        factors = self._factors_arr
        X_f = (
            float(self.init_feed_tss)
            if self.init_feed_tss is not None
            else float(jnp.sum(part_defaults * factors))
        )
        # Thickening ratio Q_feed / Q_underflow. For a well-functioning
        # clarifier (clean effluent) a solids balance puts the underflow at
        # X_u = X_f * Q_feed / Q_underflow; here Q_feed = Q_overflow + Q_under.
        # In underflow-controlled mode the design overflow is not given, so use
        # the blanket underflow as a stand-in (ratio ~2, right for BSM loading) --
        # this only seeds the initial profile, which then relaxes.
        Q_u = float(self.init_underflow_Q)
        design_over = self.overflow_Q if self.overflow_Q is not None else self.init_underflow_Q
        ratio = (float(design_over) + Q_u) / Q_u
        X_u = X_f * ratio
        # Clarification (above-feed) layers sit at the non-settleable floor.
        X_clar = max(self._fns * X_f, 1.0)

        # Per-layer TSS, layer 0 = bottom (underflow) ... n-1 = top (effluent).
        # Below feed and the feed layer carry the mixed-liquor TSS; the very
        # bottom layer carries the thickened blanket; clarification layers are
        # clear. (Matches the reference settled profile's flat-then-spike shape.)
        layer_idx = jnp.arange(self.n_layers)
        tss = jnp.where(layer_idx > self.feed_layer, X_clar, X_f)
        tss = tss.at[0].set(X_u)
        return tss

    def _layered(self, state: jnp.ndarray) -> jnp.ndarray:
        """Reshape the flat state into ``(n_layers, n_part)``."""
        return state.reshape((self.n_layers, self._n_part))

    def solids_mass(self, state: jnp.ndarray) -> jnp.ndarray:
        """Total settleable-solids mass held in the clarifier (g).

        Sums each layer's TSS over the equal layer volumes
        (``area × height / n_layers``). Used by the activated-sludge design
        layer to include the secondary-clarifier sludge inventory in the
        system solids-retention-time calculation.

        Parameters
        ----------
        state : jnp.ndarray
            The clarifier's flat state vector: shape ``(n_layers × n_part,)`` in
            ``"per_species"`` mode, ``(n_layers,)`` in ``"lumped_tss"`` mode.

        Returns
        -------
        jnp.ndarray
            Scalar total settleable-solids mass (g).
        """
        part_state, _ = self._unpack(state)
        if self.composition_mode == "lumped_tss":
            tss_per_layer = part_state  # the particulate block IS the per-layer TSS
        else:
            layered = self._layered(part_state)  # (n_layers, n_part)
            tss_per_layer = jnp.sum(layered * self._factors_arr[None, :], axis=1)
        layer_volume = self.area * self.height / self.n_layers
        return jnp.sum(tss_per_layer) * layer_volume

    def component_inventory(self, state, content, params):
        """Canonical-component inventory held in the settler blanket
        (``{component: grams}``).

        Implements the shared stateful-unit inventory contract consumed by
        :func:`aquakin.plant.balance.mass_balance`, so the balance never has to
        know this unit's layered state layout. The blanket is summed over the
        layers at the per-layer volume (``area × height / n_layers``):

        - In ``per_species`` mode the particulate head block is
          ``(n_layers, n_part)`` and each species' content is summed directly.
        - In ``lumped_tss`` mode the head block is one TSS value per layer with
          no per-species split (the lumped model scales the outlet particulates
          from the feed), so the stored TSS is distributed over the particulate
          species by the model-default solids composition to recover the
          per-component content per unit TSS -- exact for COD (the
          composition-independent factor ratio), the default solids composition
          for N/P (a small inventory term).

        With ``soluble_holdup`` the soluble tail block ``(n_layers, n_sol)``
        adds its own convective-only inventory (the liquid holdup), same layout
        in both modes.

        Parameters
        ----------
        state : array
            The settler's flat state vector.
        content : dict of str to ndarray
            ``{component: (n_species,) canonical content}`` for this model.
        params : jnp.ndarray
            This unit's parameter vector (unused; part of the shared contract).

        Returns
        -------
        dict of str to float
            ``{component: grams}`` held in the blanket (plus the soluble holdup
            when enabled).
        """
        import numpy as np

        layer_vol = float(self.area) * float(self.height) / int(self.n_layers)
        sv = np.asarray(state)
        pb = int(self._part_block_size)
        out = {}
        if self.composition_mode == "lumped_tss":
            solids_mass = layer_vol * float(np.sum(sv[:pb]))  # total g TSS held
            defaults = np.asarray(self.model.default_concentrations())
            frac = np.asarray([float(defaults[i]) for i in self._part_indices])
            factors = np.asarray(self._part_tss_factors)
            tss_per_unit = float(np.sum(frac * factors))
            for comp, vec in content.items():
                part_content = np.asarray([vec[i] for i in self._part_indices])
                per_tss = (
                    float(np.sum(frac * part_content)) / tss_per_unit if tss_per_unit > 0 else 0.0
                )
                out[comp] = solids_mass * per_tss
        else:
            prof = sv[:pb].reshape(int(self.n_layers), self._n_part)
            for comp, vec in content.items():
                part_content = np.asarray([vec[i] for i in self._part_indices])
                out[comp] = layer_vol * float(np.sum(prof * part_content[None, :]))
        if self.soluble_holdup:
            sol = sv[pb:].reshape(int(self.n_layers), self._n_sol)
            for comp, vec in content.items():
                sol_content = np.asarray([vec[i] for i in self._soluble_indices])
                out[comp] = out.get(comp, 0.0) + layer_vol * float(
                    np.sum(sol * sol_content[None, :])
                )
        return out

    def _tss(self, layer_C: jnp.ndarray) -> jnp.ndarray:
        """Total settleable solids (g/m³) in a layer from per-species
        particulate concentrations."""
        return jnp.sum(layer_C * self._factors_arr)

    def _settling_velocity(self, X: jnp.ndarray, X_min: jnp.ndarray) -> jnp.ndarray:
        """Takács double-exponential settling velocity.

        ``X`` is total solids (g/m³) in a layer; ``X_min`` is the
        non-settleable threshold based on the inflow.
        """
        # The reference Takács expression clamps only the *output* velocity, not
        # the exponent argument: below the non-settleable floor (X < X_min) the
        # double exponential is already <= 0 (since rp > rh), so the outer
        # clip(., 0, .) yields v = 0 -- the same result without an inner
        # max(X-X_min, 0). The arguments stay O(1) (rh, rp ~ 1e-3; |X-X_min| is
        # bounded by X_min ~ f_ns*feed_tss), so no exp overflow.
        excess = X - X_min
        v_takacs = self._v0 * (jnp.exp(-self._rh * excess) - jnp.exp(-self._rp * excess))
        return jnp.clip(v_takacs, 0.0, self._vmax)

    def _flow_setpoints(self) -> "dict[str, FlowSetpoint]":
        return self._setpoints

    def _resolve_setpoint(self, q, t, params, name):
        """Evaluate a setpoint: a time schedule, a differentiable parameter, or a
        constant. A scheduled setpoint (``.at(t)``) takes precedence; otherwise a
        constant controlled flow is read through its :class:`FlowSetpoint` (so it
        is differentiable), falling back to the raw value."""
        if q is None:
            return None
        if hasattr(q, "at"):
            return q.at(t)
        if name in self._setpoints:
            return self._setpoints[name].resolve(self._flow_params(params))
        return q

    def _split_flows(self, Q_in: jnp.ndarray, clamp: bool, t, params):
        return split_controlled_flows(
            self._resolve_setpoint(self.overflow_Q, t, params, "overflow_Q"),
            self._resolve_setpoint(self.underflow_Q, t, params, "underflow_Q"),
            Q_in,
            clamp,
        )

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        """Split the feed into a clarified overflow and a thickened underflow.

        Returns
        -------
        dict of str to Stream
            The two outlet streams keyed by port name: ``overflow_port`` (the
            clarified top-layer effluent) and ``underflow_port`` (the thickened
            bottom-layer sludge/RAS). Solubles pass through unchanged (or carry
            the held top/bottom layers when ``soluble_holdup`` is on);
            particulates take their respective boundary layers.
        """
        s_in = inputs[self.input_port]
        # Overflow = top layer (index n_layers - 1). Underflow = bottom (0).
        # Soluble species pass through (same concentration in both outlets).
        # The controlled flow is fixed and the other is the remainder, clamped
        # into [0, Q_in] so neither outflow goes negative (a negative RAS would
        # make the downstream mixer produce negative concentrations).
        overflow_Q, underflow_Q = self._split_flows(s_in.Q, clamp=True, t=t, params=params)

        n_species = self.model.n_species
        sol_idx, part_idx = self._sol_idx_arr, self._part_idx_arr
        part_state, sol_layered = self._unpack(state)

        # Soluble outlets: feed pass-through by default; the held top/bottom
        # layers when soluble_holdup is on (so the liquid volume has damped them).
        if self.soluble_holdup:
            sol_overflow = sol_layered[self.n_layers - 1]
            sol_underflow = sol_layered[0]
        else:
            sol_overflow = sol_underflow = s_in.C[sol_idx]

        state = part_state  # the particulate logic below reads the head block
        if self.composition_mode == "lumped_tss":
            # Particulates carry no per-layer composition memory: each outlet
            # takes the *instantaneous feed* particulate composition scaled by
            # the boundary-layer-to-feed TSS ratio, so a layer thicker than the
            # feed concentrates every particulate by the same factor and a clear
            # layer dilutes them. (At steady state this matches the per-species
            # reading; under dynamic flow it has no composition lag.)
            C_in_part = s_in.C[part_idx]  # (n_part,)
            tss_feed = jnp.sum(C_in_part * self._factors_arr)
            # Guard the zero-feed division: with no feed solids the scaling is
            # undefined, so the outlets just carry the feed particulates through.
            safe_tss = jnp.where(tss_feed > 0.0, tss_feed, 1.0)
            scale_over = jnp.where(tss_feed > 0.0, state[self.n_layers - 1] / safe_tss, 1.0)
            scale_under = jnp.where(tss_feed > 0.0, state[0] / safe_tss, 1.0)
            part_overflow = C_in_part * scale_over
            part_underflow = C_in_part * scale_under
        else:
            # Particulates take the top layer for overflow and the bottom layer
            # for underflow, straight from the per-layer composition state.
            layered = self._layered(state)  # (n_layers, n_part)
            part_overflow = layered[self.n_layers - 1]
            part_underflow = layered[0]

        # Build the per-species C vectors with two scatters each (not a Python
        # loop of scalar scatters): solubles pass through; particulates as above.
        C_overflow = (
            jnp.zeros((n_species,)).at[sol_idx].set(sol_overflow).at[part_idx].set(part_overflow)
        )
        C_underflow = (
            jnp.zeros((n_species,)).at[sol_idx].set(sol_underflow).at[part_idx].set(part_underflow)
        )

        return {
            self.overflow_port: Stream(
                Q=overflow_Q, C=C_overflow, model=self.model, scalars=s_in.scalars
            ),
            self.underflow_port: Stream(
                Q=underflow_Q, C=C_underflow, model=self.model, scalars=s_in.scalars
            ),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Linear flow rule for the recycle-flow solve: the controlled flow
        (``underflow_Q`` or ``overflow_Q``) is the setpoint at the current time
        and the other outflow is the remainder, so the map stays affine and
        ``_resolve_flows`` is exact. A scheduled setpoint reads the time from
        ``ctx``; a constant setpoint ignores it. The clamp in compute_outputs/rhs
        is the concentration-stage safeguard, inactive at the steady-state feed."""
        Q_in = input_flows[self.input_port]
        t = None if ctx is None else ctx.t
        Q_over, Q_under = self._split_flows(Q_in, clamp=False, t=t, params=params)
        return {self.overflow_port: Q_over, self.underflow_port: Q_under}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> jnp.ndarray:
        s_in = inputs[self.input_port]
        Q_in = s_in.Q
        # Controlled flow fixed, the other the remainder, clamped into [0, Q_in]
        # so the up/down convective velocities stay non-negative (see
        # compute_outputs): a negative underflow would make the RAS recycle a
        # negative-flow stream and destabilise the solve.
        overflow_Q, underflow_Q = self._split_flows(Q_in, clamp=True, t=t, params=params)

        # Particulate inlet concentrations — gather via the precomputed index array.
        C_in_part = s_in.C[self._part_idx_arr]  # (n_part,)

        # Inlet TSS sets the non-settleable floor X_min.
        tss_in = jnp.sum(C_in_part * self._factors_arr)
        X_min = self._fns * tss_in

        h_layer = self.height / self.n_layers
        v_up = overflow_Q / self.area
        v_down = underflow_Q / self.area

        part_state, sol_layered = self._unpack(state)

        if self.composition_mode == "lumped_tss":
            # A single total-solids variable per layer. The settling velocity is
            # already TSS-based, so the lumped dynamics are the identical Takács
            # flux/convection math applied to the one TSS variable -- treat the
            # state as an (n_layers, 1) array and run the SAME component-agnostic
            # helpers the per-species path uses, with a unit TSS factor and the
            # feed TSS as the single inlet "concentration". The result equals the
            # per-species dTSS/dt aggregated over species for the same profile.
            tss = part_state[:, None]  # (n_layers, 1)
            settling = self._settling_divergence(tss, X_min, h_layer, factors=jnp.ones((1,)))
            convection = self._convection(tss, tss_in[None], Q_in, v_up, v_down, h_layer)
            part_dstate = (convection + settling).reshape((-1,))
        else:
            layered = self._layered(part_state)  # (n_layers, n_part)
            # The per-layer derivative is convection (bulk up/down transport plus
            # the feed inflow) plus the Takács settling divergence.
            settling = self._settling_divergence(layered, X_min, h_layer)
            convection = self._convection(layered, C_in_part, Q_in, v_up, v_down, h_layer)
            part_dstate = (convection + settling).reshape((-1,))

        if not self.soluble_holdup:
            return part_dstate

        # Soluble holdup: bulk convection through the layers, no settling (the
        # feed soluble enters at the feed layer and is carried up/down by the
        # same v_up/v_down field). The shared _convection serves both blocks.
        C_in_sol = s_in.C[self._sol_idx_arr]  # (n_sol,)
        sol_dstate = self._convection(sol_layered, C_in_sol, Q_in, v_up, v_down, h_layer).reshape(
            (-1,)
        )
        return jnp.concatenate([part_dstate, sol_dstate])

    def _settling_divergence(
        self,
        layered: jnp.ndarray,
        X_min: jnp.ndarray,
        h_layer: float,
        factors: "jnp.ndarray | None" = None,
    ) -> jnp.ndarray:
        """Net Takács (1991) settling flux per layer, shape ``(n_layers, n_comp)``.

        Solids settle down the column at the bulk velocity ``v_s(TSS)``. At
        each interface the downward flux is the *limiting* (minimum) of the two
        adjacent layers' potential fluxes ``v_s(X)*X`` -- EXCEPT in the
        clarification zone (interfaces above the feed) while the receiving
        layer is dilute (``<= X_threshold``), where the upper layer settles
        freely instead of being held up by the min-flux rule (this is what
        keeps the effluent clean). Below the feed the min-flux rule applies
        everywhere, so a dense sludge blanket self-limits its own loading.

        The interface TSS flux is apportioned to columns by the upper (source)
        layer's composition: all particulates in a layer settle at the same
        bulk velocity, so column ``k``'s flux is ``flux_tss * X_k / TSS`` of
        that layer (no extra TSS factor -- that would mis-scale to the column's
        TSS flux). Summing ``flux_per_col * factor`` over columns recovers
        ``flux_tss`` exactly, conserving total settleable solids.

        Component-count agnostic (the column count is inferred from
        ``layered.shape[1]``), so the per-species particulate state and the
        single-column lumped-TSS state both route through here: the per-species
        caller passes the species TSS ``factors``, and the lumped caller a
        one-element ``[1.0]`` (the state already *is* the TSS, so its fraction
        of the layer total is 1).
        """
        if factors is None:
            factors = self._factors_arr
        n_comp = layered.shape[1]
        tss_per_layer = jnp.sum(layered * factors[None, :], axis=1)
        # Interface i lies between layer i (below, receiving) and i+1 (above).
        tss_above = tss_per_layer[1:]  # upper layer at each interface (n-1,)
        tss_below = tss_per_layer[:-1]  # lower (receiving) layer at each interface
        f_above = self._settling_velocity(tss_above, X_min) * tss_above
        f_below = self._settling_velocity(tss_below, X_min) * tss_below

        min_flux = jnp.minimum(f_above, f_below)
        interface_idx = jnp.arange(self.n_layers - 1)
        is_clarification = interface_idx >= self.feed_layer  # static bool
        below_threshold = tss_below <= self._X_threshold
        flux_tss = jnp.where(is_clarification & below_threshold, f_above, min_flux)

        species_frac_above = layered[1:, :] / (tss_above[:, None] + 1e-12)
        flux_per_species = flux_tss[:, None] * species_frac_above
        # (n_layers - 1, n_comp) — downward positive, at interface i.

        # flux_per_species[i] is the flux from layer i+1 down to layer i: it
        # enters layer i from above and leaves layer i+1 below. Pad with a zero
        # row for the no-flux top/bottom boundaries.
        flux_in_from_above = (
            jnp.concatenate([flux_per_species, jnp.zeros((1, n_comp))], axis=0) / h_layer
        )  # (n_layers, n_comp); top layer has 0
        flux_out_to_below = (
            jnp.concatenate([jnp.zeros((1, n_comp)), flux_per_species], axis=0) / h_layer
        )  # (n_layers, n_comp); bottom layer has 0
        return flux_in_from_above - flux_out_to_below

    def _convection(
        self,
        layered: jnp.ndarray,
        C_in_part: jnp.ndarray,
        Q_in: jnp.ndarray,
        v_up: jnp.ndarray,
        v_down: jnp.ndarray,
        h_layer: float,
    ) -> jnp.ndarray:
        """Net convective transport per layer (inflow minus outflow), shape
        ``(n_layers, n_comp)``.

        Below the feed the bulk moves down at ``v_down``, above it up at
        ``v_up``; the feed layer takes the inlet and sheds both ways. The zone
        masks are pure geometry, precomputed in ``__post_init__``. Component-count
        agnostic (inferred from ``layered.shape[1]``) so it serves both the
        particulate transport and the optional soluble-holdup transport.
        """
        # Padded "layer above"/"layer below" each i; the zero rows give
        # no convective flux across the top/bottom boundaries.
        zero_row = jnp.zeros((1, layered.shape[1]))
        below = jnp.concatenate([layered[1:, :], zero_row], axis=0)
        above = jnp.concatenate([zero_row, layered[:-1, :]], axis=0)

        is_below_feed = self._is_below_feed
        is_above_feed = self._is_above_feed
        is_feed = self._is_feed

        # Inflow: downflow from above in the underflow zone, upflow from below
        # in the clarification zone, and the inlet at the feed layer.
        conv_in_down = (v_down / h_layer) * below * is_below_feed[:, None]
        conv_in_up = (v_up / h_layer) * above * is_above_feed[:, None]
        feed_inflow = (Q_in / (self.area * h_layer)) * C_in_part[None, :] * is_feed[:, None]
        conv_in = conv_in_down + conv_in_up + feed_inflow

        # Outflow: v_down below feed, v_up above, (v_up + v_down) at the feed.
        conv_out = (
            (v_down / h_layer) * layered * is_below_feed[:, None]
            + (v_up / h_layer) * layered * is_above_feed[:, None]
            + ((v_up + v_down) / h_layer) * layered * is_feed[:, None]
        )
        return conv_in - conv_out
