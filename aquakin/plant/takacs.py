"""Takács 1-D layered secondary clarifier (Takács, Patry & Nolasco 1991).

The clarifier is modelled as a stack of well-mixed layers with vertical
flux exchange:

- **Soluble species** (e.g. SS, SNH, SNO) don't settle. They are computed
  by mass balance on the inlet flow split between overflow and underflow:
  Q_overflow + Q_underflow = Q_in, and C_soluble is identical in both.
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
)
from aquakin.plant._flow_split import (
    split_controlled_flows,
    validate_controlled_split,
)
from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


# A standard ASM1 ↔ TSS conversion used to identify the "particulate
# mass" that settles. Each particulate species contributes to the
# settling solid concentration in proportion to its COD-to-TSS factor.
# XND settles with XS but is N attached to it, not separate solids, so it
# carries a zero TSS factor.
_DEFAULT_TSS_FACTORS = {
    **{sp: ASM1_TSS_FACTOR for sp in ASM1_TSS_SPECIES},
    "XND": 0.0,
}

# Standard BSM1 Takács parameter set (Alex et al. 2008, Table 1.7).
_BSM1_TAKACS_DEFAULTS = dict(
    v0=474.0,        # max theoretical settling velocity, m/d
    vmax=250.0,      # max practical settling velocity, m/d
    rh=5.76e-4,      # hindered settling parameter, m³/g
    rp=2.86e-3,      # flocculant settling parameter, m³/g
    fns=2.28e-3,     # non-settleable fraction
    # Clarification-zone flux-limiting threshold (g/m³): above the feed, the
    # downward settling flux is limited by the layer below only when that layer
    # exceeds this concentration (Takács 1991). NOT a settling-velocity cutoff.
    X_threshold=3000.0,
)


@dataclass
class TakacsClarifier:
    """A 10-layer 1-D secondary clarifier.

    Parameters
    ----------
    name : str
    network : CompiledNetwork
        The network whose species ordering applies to the inlet and
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
    input_port : str
    overflow_port : str
    underflow_port : str
    """

    name: str
    network: "CompiledNetwork"
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
    # is taken from the network's default particulate concentrations.
    init_underflow_Q: "float | None" = None
    init_feed_tss: "float | None" = None
    particulate_species: list[str] = field(
        default_factory=lambda: list(ASM1_SETTLING_SPECIES)
    )
    settling_params: dict[str, float] = field(default_factory=lambda: dict(_BSM1_TAKACS_DEFAULTS))
    tss_factors: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_TSS_FACTORS))
    input_port: str = "inlet"
    overflow_port: str = "overflow"
    underflow_port: str = "underflow"

    def __post_init__(self) -> None:
        # Validate species and resolve indices.
        self._part_indices: list[int] = []
        self._part_tss_factors: list[float] = []
        for sp in self.particulate_species:
            if sp not in self.network.species_index:
                raise ValueError(
                    f"TakacsClarifier '{self.name}': particulate species "
                    f"'{sp}' not in network."
                )
            self._part_indices.append(self.network.species_index[sp])
            self._part_tss_factors.append(self.tss_factors.get(sp, 1.0))
        n_part = len(self._part_indices)

        # Per-layer geometry.
        if self.feed_layer < 0 or self.feed_layer >= self.n_layers:
            raise ValueError(
                f"feed_layer must be in [0, {self.n_layers}); got {self.feed_layer}"
            )
        validate_controlled_split(
            f"TakacsClarifier '{self.name}'", self.overflow_Q, self.underflow_Q
        )
        # Soluble = everything not in particulate.
        self._soluble_indices = [
            i for i in range(self.network.n_species) if i not in self._part_indices
        ]

        # The state vector is (n_layers, n_part) flattened into 1-D.
        # Initial state: all zero (clean reactor at start).
        self._n_part = n_part

        # Precompute the constant index/factor arrays and per-layer geometric
        # masks ONCE, so the per-RHS-call hot path does not rebuild them with
        # ``jnp.asarray`` of a Python list / ``astype`` every step.
        self._part_idx_arr = jnp.asarray(self._part_indices)
        self._sol_idx_arr = jnp.asarray(self._soluble_indices)
        self._factors_arr = jnp.asarray(self._part_tss_factors)
        layer_idx = jnp.arange(self.n_layers)
        self._is_below_feed = (layer_idx < self.feed_layer).astype(jnp.float64)
        self._is_above_feed = (layer_idx > self.feed_layer).astype(jnp.float64)
        self._is_feed = (layer_idx == self.feed_layer).astype(jnp.float64)

        # Stash settling params as a closure to use in rhs without Python
        # branching per call.
        self._v0 = float(self.settling_params["v0"])
        self._vmax = float(self.settling_params["vmax"])
        self._rh = float(self.settling_params["rh"])
        self._rp = float(self.settling_params["rp"])
        self._fns = float(self.settling_params["fns"])
        self._X_threshold = float(self.settling_params["X_threshold"])

    @property
    def state_size(self) -> int:
        return self.n_layers * self._n_part

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.overflow_port, self.underflow_port]

    def initial_state(self) -> jnp.ndarray:
        defaults = self.network.default_concentrations()
        part_defaults = jnp.asarray([float(defaults[i]) for i in self._part_indices])

        if self.init_underflow_Q is None:
            # Backward-compatible uniform seed: every layer at the inlet's
            # default particulate concentration.
            return jnp.tile(part_defaults, self.n_layers)

        # --- State-point-analysis settled blanket ----------------------
        # Feed TSS (MLSS) the blanket is built for. The clarifier feed at
        # startup is the reactors' initial mixed-liquor, so default to the TSS
        # implied by the network's particulate defaults.
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
        design_over = (
            self.overflow_Q if self.overflow_Q is not None else self.init_underflow_Q
        )
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

        # Apportion each layer's TSS across particulate species by the feed
        # composition: C_k = d_k * (TSS_layer / X_f), so sum_k C_k * f_k = TSS.
        scale = tss / X_f                                   # (n_layers,)
        state = scale[:, None] * part_defaults[None, :]     # (n_layers, n_part)
        return state.reshape(-1)

    def _layered(self, state: jnp.ndarray) -> jnp.ndarray:
        """Reshape the flat state into ``(n_layers, n_part)``."""
        return state.reshape((self.n_layers, self._n_part))

    def _tss(self, layer_C: jnp.ndarray) -> jnp.ndarray:
        """Total settleable solids (g/m³) in a layer from per-species
        particulate concentrations."""
        return jnp.sum(layer_C * self._factors_arr)

    def _settling_velocity(self, X: jnp.ndarray, X_min: jnp.ndarray) -> jnp.ndarray:
        """Takács double-exponential settling velocity.

        ``X`` is total solids (g/m³) in a layer; ``X_min`` is the
        non-settleable threshold based on the inflow.
        """
        excess = jnp.maximum(X - X_min, 0.0)
        v_takacs = self._v0 * (jnp.exp(-self._rh * excess) - jnp.exp(-self._rp * excess))
        # Clamp to [0, vmax].
        return jnp.clip(v_takacs, 0.0, self._vmax)

    def _split_flows(self, Q_in: jnp.ndarray, clamp: bool):
        return split_controlled_flows(
            self.overflow_Q, self.underflow_Q, Q_in, clamp
        )

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict[str, Stream]:
        s_in = inputs[self.input_port]
        layered = self._layered(state)  # (n_layers, n_part)
        # Overflow = top layer (index n_layers - 1). Underflow = bottom (0).
        # Soluble species pass through (same concentration in both outlets).
        # The controlled flow is fixed and the other is the remainder, clamped
        # into [0, Q_in] so neither outflow goes negative (a negative RAS would
        # make the downstream mixer produce negative concentrations).
        overflow_Q, underflow_Q = self._split_flows(s_in.Q, clamp=True)

        # Build the per-species C vectors with two scatters each (not a Python
        # loop of scalar scatters): solubles pass through; particulates take the
        # top layer for overflow and the bottom layer for underflow.
        n_species = self.network.n_species
        sol_idx, part_idx = self._sol_idx_arr, self._part_idx_arr
        sol_C = s_in.C[sol_idx]
        C_overflow = (
            jnp.zeros((n_species,))
            .at[sol_idx].set(sol_C)
            .at[part_idx].set(layered[self.n_layers - 1])
        )
        C_underflow = (
            jnp.zeros((n_species,))
            .at[sol_idx].set(sol_C)
            .at[part_idx].set(layered[0])
        )

        return {
            self.overflow_port: Stream(Q=overflow_Q, C=C_overflow, network=self.network),
            self.underflow_port: Stream(Q=underflow_Q, C=C_underflow, network=self.network),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray) -> dict:
        """Linear flow rule for the recycle-flow solve: the controlled flow
        (``underflow_Q`` or ``overflow_Q``) is constant and the other outflow is
        the remainder, so the map stays affine and ``_resolve_flows`` is exact.
        The clamp in compute_outputs/rhs is the concentration-stage safeguard,
        inactive at the steady-state feed."""
        Q_in = input_flows[self.input_port]
        Q_over, Q_under = self._split_flows(Q_in, clamp=False)
        return {self.overflow_port: Q_over, self.underflow_port: Q_under}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        s_in = inputs[self.input_port]
        Q_in = s_in.Q
        # Controlled flow fixed, the other the remainder, clamped into [0, Q_in]
        # so the up/down convective velocities stay non-negative (see
        # compute_outputs): a negative underflow would make the RAS recycle a
        # negative-flow stream and destabilise the solve.
        overflow_Q, underflow_Q = self._split_flows(Q_in, clamp=True)

        layered = self._layered(state)  # (n_layers, n_part)
        # Particulate inlet concentrations — gather via the precomputed index array.
        C_in_part = s_in.C[self._part_idx_arr]  # (n_part,)

        # Inlet TSS sets the non-settleable floor X_min.
        tss_in = jnp.sum(C_in_part * self._factors_arr)
        X_min = self._fns * tss_in

        h_layer = self.height / self.n_layers
        v_up = overflow_Q / self.area
        v_down = underflow_Q / self.area

        # The per-layer derivative is convection (bulk up/down transport plus
        # the feed inflow) plus the Takács settling divergence.
        settling = self._settling_divergence(layered, X_min, h_layer)
        convection = self._convection(
            layered, C_in_part, Q_in, v_up, v_down, h_layer
        )
        dstate = convection + settling
        return dstate.reshape((-1,))

    def _settling_divergence(
        self, layered: jnp.ndarray, X_min: jnp.ndarray, h_layer: float
    ) -> jnp.ndarray:
        """Net Takács (1991) settling flux per layer, shape ``(n_layers, n_part)``.

        Solids settle down the column at the bulk velocity ``v_s(TSS)``. At
        each interface the downward flux is the *limiting* (minimum) of the two
        adjacent layers' potential fluxes ``v_s(X)*X`` -- EXCEPT in the
        clarification zone (interfaces above the feed) while the receiving
        layer is dilute (``<= X_threshold``), where the upper layer settles
        freely instead of being held up by the min-flux rule (this is what
        keeps the effluent clean). Below the feed the min-flux rule applies
        everywhere, so a dense sludge blanket self-limits its own loading.

        The interface TSS flux is apportioned to species by the upper (source)
        layer's composition: all particulates in a layer settle at the same
        bulk velocity, so species ``k``'s flux is ``flux_tss * X_k / TSS`` of
        that layer (no extra TSS factor -- that would mis-scale to the species'
        TSS flux). Summing ``flux_per_species * factor`` over species recovers
        ``flux_tss`` exactly, conserving total settleable solids.
        """
        tss_per_layer = jnp.sum(layered * self._factors_arr[None, :], axis=1)
        # Interface i lies between layer i (below, receiving) and i+1 (above).
        tss_above = tss_per_layer[1:]   # upper layer at each interface (n-1,)
        tss_below = tss_per_layer[:-1]  # lower (receiving) layer at each interface
        f_above = self._settling_velocity(tss_above, X_min) * tss_above
        f_below = self._settling_velocity(tss_below, X_min) * tss_below

        min_flux = jnp.minimum(f_above, f_below)
        interface_idx = jnp.arange(self.n_layers - 1)
        is_clarification = interface_idx >= self.feed_layer        # static bool
        below_threshold = tss_below <= self._X_threshold
        flux_tss = jnp.where(is_clarification & below_threshold, f_above, min_flux)

        species_frac_above = layered[1:, :] / (tss_above[:, None] + 1e-12)
        flux_per_species = flux_tss[:, None] * species_frac_above
        # (n_layers - 1, n_part) — downward positive, at interface i.

        # flux_per_species[i] is the flux from layer i+1 down to layer i: it
        # enters layer i from above and leaves layer i+1 below. Pad with a zero
        # row for the no-flux top/bottom boundaries.
        flux_in_from_above = jnp.concatenate(
            [flux_per_species, jnp.zeros((1, self._n_part))], axis=0
        ) / h_layer   # (n_layers, n_part); top layer has 0
        flux_out_to_below = jnp.concatenate(
            [jnp.zeros((1, self._n_part)), flux_per_species], axis=0
        ) / h_layer   # (n_layers, n_part); bottom layer has 0
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
        ``(n_layers, n_part)``.

        Below the feed the bulk moves down at ``v_down``, above it up at
        ``v_up``; the feed layer takes the inlet and sheds both ways. The zone
        masks are pure geometry, precomputed in ``__post_init__``.
        """
        # Padded "layer above"/"layer below" each i; the zero rows give
        # no convective flux across the top/bottom boundaries.
        zero_row = jnp.zeros((1, self._n_part))
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
