"""Ideal point separator — a simpler alternative to the full Takács model.

The :class:`IdealClarifier` assumes instantaneous, perfect separation:
particulates split between overflow and underflow such that the underflow
is concentrated by a fixed *thickening ratio* relative to the inlet (or,
equivalently, the overflow has a fixed *capture efficiency* of particulates).
Solubles pass through unchanged.

This is appropriate for BSM1-style demos where the focus is the upstream
biology and the clarifier serves mainly to recycle biomass via the RAS.
It produces deterministic, stable behaviour without the per-layer
mass-balance dynamics. For literature-comparable BSM1 effluent metrics,
use the layered :class:`aquakin.plant.takacs.TakacsClarifier` instead
(``build_bsm1(use_takacs=True)``); both expose the same ports, so they are
drop-in interchangeable in a flowsheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import ASM1_SETTLING_SPECIES
from aquakin.plant._flow_split import (
    split_controlled_flows,
    validate_controlled_split,
)
from aquakin.plant.flow_setpoint import FlowParameterized, FlowSetpoint
from aquakin.plant.streams import Stream
from aquakin.plant.units import StatelessUnit

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


@dataclass
class IdealClarifier(StatelessUnit, FlowParameterized):
    """Stateless ideal solid/liquid separator.

    Soluble species pass through to both outflow streams at the inlet
    concentration. Particulate species are split:

    - ``capture_efficiency`` (default 0.998): fraction of inflowing
      particulate mass that goes to the underflow.
    - The overflow gets the remainder.

    The underflow's particulate concentrations are determined by mass
    balance: ``C_under = capture * Q_in * C_in / Q_under`` per species.
    The overflow concentrations follow similarly with ``(1 - capture)``.

    Parameters
    ----------
    name : str
    model : CompiledModel
    overflow_Q : float, optional
        Fixed overflow (effluent) flow rate; underflow takes the remainder.
        Supply exactly one of ``overflow_Q`` or ``underflow_Q``.
    underflow_Q : float, optional
        Fixed underflow flow rate (the controlled RAS+wastage pump flow,
        ``Q_r + Q_w``); the overflow (effluent) is then the remainder and
        tracks the feed (``Q_e = Q_f - Q_u``, the BSM convention). Preferred
        for plants with dynamic influent -- a fixed overflow forces a
        near-singular recycle-flow gain. Supply exactly one of ``overflow_Q``
        or ``underflow_Q``.
    capture_efficiency : float
        Fraction of inflowing particulate mass directed to the underflow.
        Default 0.998 (typical BSM1 clarifier; corresponds to ~99.8% of
        biomass returned via RAS).
    particulate_species : list[str]
        Species names treated as particulates. Default is the ASM1 set.
    input_port : str
    overflow_port : str
    underflow_port : str
    """

    name: str
    model: "CompiledModel"
    overflow_Q: "float | None" = None
    underflow_Q: "float | None" = None
    capture_efficiency: float = 0.998
    particulate_species: list[str] = field(default_factory=lambda: list(ASM1_SETTLING_SPECIES))
    input_port: str = "inlet"
    overflow_port: str = "overflow"
    underflow_port: str = "underflow"

    # state_size / initial_state / rhs come from StatelessUnit.

    def __post_init__(self) -> None:
        if not (0.0 <= self.capture_efficiency <= 1.0):
            raise ValueError(f"capture_efficiency must be in [0, 1]; got {self.capture_efficiency}")
        validate_controlled_split(
            f"IdealClarifier '{self.name}'", self.overflow_Q, self.underflow_Q
        )
        # The controlled outflow (underflow or overflow) is a differentiable
        # flow setpoint; the other outflow is the remainder. Both the flow rule
        # and the material split read it through one FlowSetpoint.
        if self.underflow_Q is not None:
            self._ctrl = "underflow"
            self._setpoints = {"underflow_Q": FlowSetpoint(float(self.underflow_Q), 0)}
        else:
            self._ctrl = "overflow"
            self._setpoints = {"overflow_Q": FlowSetpoint(float(self.overflow_Q), 0)}
        self._part_indices = [
            self.model.species_index[s]
            for s in self.particulate_species
            if s in self.model.species_index
        ]
        # Pre-build a (n_species,) mask: 1.0 for particulates, 0.0 for solubles.
        mask = jnp.zeros((self.model.n_species,))
        for i in self._part_indices:
            mask = mask.at[i].set(1.0)
        self._particulate_mask = mask

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.overflow_port, self.underflow_port]

    def _flow_setpoints(self) -> "dict[str, FlowSetpoint]":
        return self._setpoints

    def _split_flows(self, Q_in: jnp.ndarray, params: jnp.ndarray, clamp: bool):
        val = self._setpoints[f"{self._ctrl}_Q"].resolve(self._flow_params(params))
        if self._ctrl == "underflow":
            return split_controlled_flows(None, val, Q_in, clamp)
        return split_controlled_flows(val, None, Q_in, clamp)

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> dict[str, Stream]:
        s_in = inputs[self.input_port]
        Q_in = s_in.Q
        # The non-controlled outflow is the remainder, clamped into [0, Q_in]:
        # neither overflow nor underflow can go negative (the mass/Q blow-up
        # hazard during the recycle-flow startup transient; closes issue #17).
        # Mass-conserving (the two outflows sum to Q_in) and inactive at steady
        # state.
        Q_over, Q_under = self._split_flows(Q_in, params, clamp=True)
        cap = jnp.asarray(self.capture_efficiency)

        # Per-species partitioning. Solubles split with flow ratio.
        # Particulates: ``cap`` fraction of mass goes to underflow.
        mass_in = Q_in * s_in.C  # (n_species,) g/d
        part_mask = self._particulate_mask
        sol_mask = 1.0 - part_mask

        # Soluble outflow concentrations: same as inlet (pass through).
        sol_C_under = s_in.C * sol_mask
        sol_C_over = s_in.C * sol_mask

        # Particulate outflow concentrations: mass partitioned by capture.
        # mass_under_p = cap * mass_in_p
        # mass_over_p = (1 - cap) * mass_in_p
        mass_under_p = cap * mass_in * part_mask
        mass_over_p = (1.0 - cap) * mass_in * part_mask

        part_C_under = mass_under_p / (Q_under + 1e-12)
        part_C_over = mass_over_p / (Q_over + 1e-12)

        C_over = sol_C_over + part_C_over
        C_under = sol_C_under + part_C_under

        return {
            self.overflow_port: Stream(Q=Q_over, C=C_over, model=self.model, T=s_in.T),
            self.underflow_port: Stream(Q=Q_under, C=C_under, model=self.model, T=s_in.T),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Linear flow rule for the recycle-flow solve: the controlled flow
        (``underflow_Q`` or ``overflow_Q``) is constant and the other outflow is
        the remainder, so the map stays affine and ``Plant._resolve_flows`` is
        exact. The clamp in ``compute_outputs`` is the concentration-stage
        safeguard, inactive at steady state."""
        Q_in = input_flows[self.input_port]
        Q_over, Q_under = self._split_flows(Q_in, params, clamp=False)
        return {self.overflow_port: Q_over, self.underflow_port: Q_under}
