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

from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


# A standard ASM1 ↔ TSS conversion used to identify the "particulate
# mass" that settles. Each particulate species contributes to the
# settling solid concentration in proportion to its COD-to-TSS factor.
_DEFAULT_TSS_FACTORS = {
    "XS": 0.75, "XI": 0.75, "XB_H": 0.75, "XB_A": 0.75, "XP": 0.75,
    "XND": 0.0,  # XND is N attached to XS, not separate solids
}

# Standard BSM1 Takács parameter set (Alex et al. 2008, Table 1.7).
_BSM1_TAKACS_DEFAULTS = dict(
    v0=474.0,        # max theoretical settling velocity, m/d
    vmax=250.0,      # max practical settling velocity, m/d
    rh=5.76e-4,      # hindered settling parameter, m³/g
    rp=2.86e-3,      # flocculant settling parameter, m³/g
    fns=2.28e-3,     # non-settleable fraction
    X_threshold=3000.0,  # threshold below which v_s = 0, g/m³
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
    overflow_Q : float
        Overflow (clarified effluent) volumetric flow rate, m³/d.
        Underflow takes ``Q_in - overflow_Q``.
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
    overflow_Q: float
    n_layers: int = 10
    feed_layer: int = 5
    particulate_species: list[str] = field(default_factory=lambda: [
        "XS", "XI", "XB_H", "XB_A", "XP", "XND"
    ])
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
        if self.overflow_Q < 0:
            raise ValueError(f"overflow_Q must be non-negative; got {self.overflow_Q}")
        # Soluble = everything not in particulate.
        self._soluble_indices = [
            i for i in range(self.network.n_species) if i not in self._part_indices
        ]

        # The state vector is (n_layers, n_part) flattened into 1-D.
        # Initial state: all zero (clean reactor at start).
        self._n_part = n_part
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
        # Seed every layer at the inlet's default particulate concentration
        # (so the integrator starts in a sensible regime).
        defaults = self.network.default_concentrations()
        part_defaults = jnp.asarray([float(defaults[i]) for i in self._part_indices])
        # Tile to (n_layers, n_part) then flatten.
        return jnp.tile(part_defaults, self.n_layers)

    def _layered(self, state: jnp.ndarray) -> jnp.ndarray:
        """Reshape the flat state into ``(n_layers, n_part)``."""
        return state.reshape((self.n_layers, self._n_part))

    def _tss(self, layer_C: jnp.ndarray) -> jnp.ndarray:
        """Total settleable solids (g/m³) in a layer from per-species
        particulate concentrations."""
        factors = jnp.asarray(self._part_tss_factors)
        return jnp.sum(layer_C * factors)

    def _settling_velocity(self, X: jnp.ndarray, X_min: jnp.ndarray) -> jnp.ndarray:
        """Takács double-exponential settling velocity.

        ``X`` is total solids (g/m³) in a layer; ``X_min`` is the
        non-settleable threshold based on the inflow.
        """
        excess = jnp.maximum(X - X_min, 0.0)
        v_takacs = self._v0 * (jnp.exp(-self._rh * excess) - jnp.exp(-self._rp * excess))
        # Clamp to [0, vmax].
        return jnp.clip(v_takacs, 0.0, self._vmax)

    def _settling_flux(
        self,
        X_above: jnp.ndarray,
        X_below: jnp.ndarray,
        X_min: jnp.ndarray,
    ) -> jnp.ndarray:
        """Bürger-Diehl-style flux across the boundary between two layers.

        Returns flux density (g/m²/d) — positive flux is downward.
        Standard Takács-flux at a boundary is min of the upper and lower
        layer fluxes, with a threshold-based suppression below X_threshold.
        """
        v_above = self._settling_velocity(X_above, X_min)
        v_below = self._settling_velocity(X_below, X_min)
        f_above = v_above * X_above
        f_below = v_below * X_below
        # Standard minimum-flux assumption (Takács 1991 eq. 11).
        flux = jnp.minimum(f_above, f_below)
        # Suppress the flux through layers above the feed when X is below
        # the threshold (prevents back-mixing into the clarification zone).
        return jnp.where(X_above < self._X_threshold, f_above, flux)

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
        overflow_Q = jnp.asarray(self.overflow_Q)
        underflow_Q = s_in.Q - overflow_Q

        # Build the per-species C vectors for the two outputs.
        n_species = self.network.n_species
        C_overflow = jnp.zeros((n_species,))
        C_underflow = jnp.zeros((n_species,))
        # Solubles: pass through.
        for i in self._soluble_indices:
            C_overflow = C_overflow.at[i].set(s_in.C[i])
            C_underflow = C_underflow.at[i].set(s_in.C[i])
        # Particulates: top layer for overflow, bottom layer for underflow.
        for k, i in enumerate(self._part_indices):
            C_overflow = C_overflow.at[i].set(layered[self.n_layers - 1, k])
            C_underflow = C_underflow.at[i].set(layered[0, k])

        return {
            self.overflow_port: Stream(Q=overflow_Q, C=C_overflow, network=self.network),
            self.underflow_port: Stream(Q=underflow_Q, C=C_underflow, network=self.network),
        }

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        s_in = inputs[self.input_port]
        Q_in = s_in.Q
        overflow_Q = jnp.asarray(self.overflow_Q)
        underflow_Q = Q_in - overflow_Q

        layered = self._layered(state)  # (n_layers, n_part)
        # Particulate inlet concentrations — gather via index array (vectorised).
        part_idx = jnp.asarray(self._part_indices)
        C_in_part = s_in.C[part_idx]  # (n_part,)
        factors = jnp.asarray(self._part_tss_factors)

        # Inlet TSS for the non-settleable threshold.
        tss_in = jnp.sum(C_in_part * factors)
        X_min = self._fns * tss_in

        h_layer = self.height / self.n_layers
        v_up = overflow_Q / self.area
        v_down = underflow_Q / self.area

        # Per-layer TSS and settling velocities.
        tss_per_layer = jnp.sum(layered * factors[None, :], axis=1)  # (n_layers,)

        # Interface settling-flux density at boundary i (between layers i
        # and i+1). Downward positive. Computed pairwise via vectorised
        # min-flux assumption.
        tss_above = tss_per_layer[1:]   # (n_layers - 1,)
        tss_below = tss_per_layer[:-1]  # (n_layers - 1,)
        v_above = self._settling_velocity(tss_above, X_min)
        v_below = self._settling_velocity(tss_below, X_min)
        f_above = v_above * tss_above
        f_below = v_below * tss_below
        # Pure Takács minimum-flux assumption (1991 paper). Both layers'
        # potential settling fluxes are computed; the boundary takes the
        # limiting (smaller) one. This keeps the model self-stabilising:
        # when the layer below grows very dense, its settling velocity
        # collapses, and the inflowing flux from above is throttled to
        # match — preventing unbounded accumulation in the underflow.
        flux_tss = jnp.minimum(f_above, f_below)  # (n_layers - 1,)

        # Per-species settling flux: tss_flux × (X_species_above / TSS_above) × factor.
        species_ratio_above = layered[1:, :] / (tss_above[:, None] + 1e-12)
        flux_per_species = flux_tss[:, None] * species_ratio_above * factors[None, :]
        # (n_layers - 1, n_part) — downward positive, at interface i between layers i and i+1.

        # Convection. Use jnp.zeros((n_layers, n_part)) for "no flux at top/bottom".
        # Upward convection above feed layer; downward below; mixed at feed.
        zero_row = jnp.zeros((1, self._n_part))

        # conv_in_down[i] = inflow from layer i+1 to layer i (downflow zone)
        # conv_in_up[i]   = inflow from layer i-1 to layer i (upflow zone)
        below = jnp.concatenate([layered[1:, :], zero_row], axis=0)  # padded "layer above" each i
        above = jnp.concatenate([zero_row, layered[:-1, :]], axis=0)  # padded "layer below" each i

        # Build per-layer (n_layers, n_part) convection terms.
        layer_idx = jnp.arange(self.n_layers)
        is_below_feed = (layer_idx < self.feed_layer).astype(jnp.float64)
        is_above_feed = (layer_idx > self.feed_layer).astype(jnp.float64)
        is_feed = (layer_idx == self.feed_layer).astype(jnp.float64)

        # Downflow convection contributes only in/at the underflow zone.
        conv_in_down = (v_down / h_layer) * below * is_below_feed[:, None]
        # Upflow convection in/at the clarification zone.
        conv_in_up = (v_up / h_layer) * above * is_above_feed[:, None]
        # Feed-layer inflow (per-species inlet concentration).
        feed_inflow = (Q_in / (self.area * h_layer)) * C_in_part[None, :] * is_feed[:, None]

        conv_in = conv_in_down + conv_in_up + feed_inflow

        # Outflow from each layer.
        # Below feed (downflow zone): v_down * X / h
        # Above feed: v_up * X / h
        # Feed layer: (v_up + v_down) * X / h
        conv_out = (
            (v_down / h_layer) * layered * is_below_feed[:, None]
            + (v_up / h_layer) * layered * is_above_feed[:, None]
            + ((v_up + v_down) / h_layer) * layered * is_feed[:, None]
        )

        # Settling fluxes — mass flows down the column. flux_per_species[i]
        # is the flux from layer i+1 down to layer i.
        # For layer i (not top): flux into layer i from above = flux_per_species[i] / h
        # For layer i (not bottom): flux out of layer i to below = flux_per_species[i-1] / h
        flux_in_from_above = jnp.concatenate(
            [flux_per_species, jnp.zeros((1, self._n_part))], axis=0
        ) / h_layer   # (n_layers, n_part); top layer has 0
        flux_out_to_below = jnp.concatenate(
            [jnp.zeros((1, self._n_part)), flux_per_species], axis=0
        ) / h_layer   # (n_layers, n_part); bottom layer has 0

        dstate = conv_in - conv_out + flux_in_from_above - flux_out_to_below
        return dstate.reshape((-1,))
