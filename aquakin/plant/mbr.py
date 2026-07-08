"""Membrane bioreactor (MBR): a high-MLSS aerated reactor + membrane separation.

An MBR replaces the secondary clarifier with a membrane that retains essentially
all the solids in the reactor, so the biomass concentration (MLSS) runs high and
the permeate -- the treated effluent -- is near solids-free. Solids leave only by
sludge wasting, which decouples the solids retention time (SRT = V / Q_waste)
from the hydraulic retention time. It is a dominant modern configuration (small
footprint, high-quality effluent).

:class:`MBRUnit` is a fixed-volume aerated reactor (reusing the CSTR kinetics and
the :class:`~aquakin.plant.cstr.Aeration` machinery, so it gets open-loop or
auto-wired DO control like a CSTR) with two outlets:

* ``permeate`` -- the membrane filtrate: solubles pass, particulates are rejected
  at ``rejection`` (~all), so the effluent is near solids-free.
* ``waste``    -- the wasted mixed liquor at the full reactor MLSS, drawn at the
  ``waste_flow`` setpoint; this is the only solids sink, so it sets the SRT.

The reactor volume is held constant (a permeate pump matches the feed less
wasting): ``Q_permeate = Q_in - Q_waste``. A simple membrane-fouling state tracks
a fouling resistance that grows with the permeate flux and partially relaxes, and
:meth:`MBRUnit.tmp` reports the trans-membrane pressure from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant.coupling import CouplingAware
from aquakin.plant.cstr import Aeration, AerationUnit, aeration_transfer
from aquakin.plant.streams import Stream, mixed_scalars

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


@dataclass
class MBRUnit(AerationUnit, CouplingAware):
    """A membrane bioreactor: high-MLSS aerated reactor with membrane separation.

    Parameters
    ----------
    name : str
    model : CompiledModel
    volume : float
        Reactor liquid volume (held constant; the permeate pump matches the feed
        less wasting).
    aeration : Aeration, optional
        Aeration spec, exactly as for a :class:`~aquakin.plant.cstr.CSTRUnit`
        (open-loop ``kla`` or a DO ``do_setpoint`` the plant auto-wires a
        controller for). ``None`` = no aeration.
    waste_flow : float
        Mixed-liquor wasting flow (the only solids sink; sets SRT = V/waste_flow).
        Default 0 (no wasting -- solids then accumulate, see the SBR note).
    rejection : float
        Membrane particulate rejection in [0, 1]; the permeate carries
        ``(1 - rejection)`` of each particulate. Default 0.999.
    particulate_species : sequence of str
        Species the membrane rejects (the solids, e.g. the ASM ``X*``). Solubles
        pass through unhindered.
    membrane_area : float
        Membrane area, for the permeate flux ``J = Q_permeate / area`` that drives
        fouling and TMP.
    fouling_rate : float
        Fouling-resistance growth per unit flux (``dR_f/dt`` gains
        ``fouling_rate * J``). Default 0 (no fouling).
    fouling_relax : float
        First-order recovery rate of the fouling resistance (reversible fouling /
        relaxation), 1/time. Default 0.
    membrane_resistance : float
        Clean-membrane resistance ``R_m`` in the TMP relation.
    tmp_viscosity : float
        Permeate viscosity factor in ``TMP = tmp_viscosity * J * (R_m + R_f)``.
        Default 1 (TMP reported in the model's consistent flux*resistance units).
    conditions : dict
        Per-model condition values (e.g. ``{"T": 293.15}``).
    initial_concentrations : jnp.ndarray, optional
        Initial bulk concentrations (defaults to the model defaults).
    initial_fouling : float
        Initial fouling resistance (default 0).
    input_port, permeate_port, waste_port : str
        Port names (``"feed"`` / ``"permeate"`` / ``"waste"``).
    """

    name: str
    model: CompiledModel
    volume: float
    aeration: Aeration | None = None
    waste_flow: float = 0.0
    rejection: float = 0.999
    particulate_species: Sequence[str] = ()
    membrane_area: float = 1.0
    fouling_rate: float = 0.0
    fouling_relax: float = 0.0
    membrane_resistance: float = 1.0
    tmp_viscosity: float = 1.0
    conditions: dict = field(default_factory=dict)
    initial_concentrations: jnp.ndarray | None = None
    initial_fouling: float = 0.0
    input_port: str = "feed"
    permeate_port: str = "permeate"
    waste_port: str = "waste"

    def __post_init__(self) -> None:
        if not (0.0 <= self.rejection <= 1.0):
            raise ValueError(
                f"MBRUnit '{self.name}' rejection must be in [0, 1], got {self.rejection}."
            )
        if self.waste_flow < 0.0:
            raise ValueError(f"MBRUnit '{self.name}' waste_flow must be >= 0.")
        missing = [c for c in self.model.conditions_required if c not in self.conditions]
        if missing:
            raise ValueError(
                f"MBRUnit '{self.name}' is missing condition values for: "
                f"{missing}. Provided: {list(self.conditions)}."
            )
        for s in self.particulate_species:
            if s not in self.model.species_index:
                raise ValueError(
                    f"MBRUnit '{self.name}' particulate species '{s}' not in the model."
                )

        # Permeate multiplier: 1 for solubles, (1 - rejection) for particulates.
        n = self.model.n_species
        mask = jnp.zeros((n,))
        if self.particulate_species:
            idx = [self.model.species_index[s] for s in self.particulate_species]
            mask = mask.at[jnp.asarray(idx, dtype=int)].set(1.0)
        self._perm_mult = 1.0 - self.rejection * mask

        # Aeration vectors via the AerationUnit mixin (the same CSTR machinery,
        # incl. auto-wired DO control).
        self._setup_aeration()
        self._condition_arrays = {k: jnp.asarray([float(v)]) for k, v in self.conditions.items()}

    # ----- protocol: identity / layout -----------------------------------
    @property
    def state_size(self) -> int:
        return self.model.n_species + 1  # bulk C + fouling resistance R_f

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.permeate_port, self.waste_port]

    def initial_state(self) -> jnp.ndarray:
        C0 = (
            self.model.default_concentrations()
            if self.initial_concentrations is None
            else jnp.asarray(self.initial_concentrations)
        )
        return jnp.concatenate([C0, jnp.asarray([float(self.initial_fouling)])])

    def _split(self, state):
        n = self.model.n_species
        return state[:n], state[n]

    def liquid_volume(self, state: jnp.ndarray):
        """The (fixed) reactor liquid volume.

        The explicit mass-balance inventory contract (see
        :func:`aquakin.plant.balance._unit_inventory`): the trailing fouling
        resistance ``R_f`` carries no mass, so the inventory is ``volume*C``.
        """
        return jnp.asarray(self.volume)

    def coupling_pattern(self):
        """Structural Jacobian sparsity (issue #388).

        State is ``[C (n_species), R_f]``. ``self``: the reaction kinetics'
        structural pattern on the species block (a saturated Monod term is
        numerically invisible to a probe, so the syntactic AST dependency is
        needed), plus the fouling resistance's self-relaxation diagonal entry.
        ``R_f`` is driven by the permeate flux (a flow setpoint, not the species),
        so it is decoupled from the concentrations: ``dR_f/dt`` does not read
        ``C``, and ``dC/dt`` does not read ``R_f`` (see :meth:`rhs`). ``inlet``:
        the convective dilution diagonal on the species (the permeate/waste split
        is a fixed-flow setpoint), with no inlet coupling into ``R_f``.
        """
        import numpy as np

        from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
        from aquakin.plant.coupling import CouplingPattern

        n = self.model.n_species
        self_pat = np.zeros((n + 1, n + 1), dtype=bool)
        self_pat[:n, :n] = structural_sparsity_pattern(self.model)
        self_pat[n, n] = True  # dR_f/dt depends on R_f
        inlet_pat = np.zeros((n + 1, n), dtype=bool)
        inlet_pat[:n, :] = np.eye(n, dtype=bool)  # dilution: each C_i <- C_in,i
        return CouplingPattern(self_pattern=self_pat, inlet_pattern=inlet_pat)

    # ----- membrane diagnostics ------------------------------------------
    def tmp(self, fouling_resistance: jnp.ndarray, permeate_flow: jnp.ndarray) -> jnp.ndarray:
        """Trans-membrane pressure ``tmp_viscosity * J * (R_m + R_f)`` at the
        given fouling resistance and permeate flow (``J = Q/area``)."""
        J = permeate_flow / self.membrane_area
        return self.tmp_viscosity * J * (self.membrane_resistance + fouling_resistance)

    # ----- protocol: behaviour -------------------------------------------
    def _flows(self, q_in: jnp.ndarray):
        """Permeate / waste split for an inflow ``q_in`` (constant volume)."""
        q_waste = jnp.asarray(float(self.waste_flow))
        q_perm = jnp.maximum(q_in - q_waste, 0.0)
        return q_perm, q_waste

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        C, _R_f = self._split(state)
        q_in = inputs[self.input_port].Q
        q_perm, q_waste = self._flows(q_in)
        scalars_out = mixed_scalars(inputs, self.input_ports)  # carry inlet scalars on
        return {
            self.permeate_port: Stream(
                Q=q_perm, C=self._perm_mult * C, model=self.model, scalars=scalars_out
            ),
            self.waste_port: Stream(Q=q_waste, C=C, model=self.model, scalars=scalars_out),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        q_perm, q_waste = self._flows(input_flows[self.input_port])
        return {self.permeate_port: q_perm, self.waste_port: q_waste}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        C, R_f = self._split(state)
        s_in = inputs[self.input_port]
        q_in = s_in.Q
        q_perm, q_waste = self._flows(q_in)

        # Constant-volume mass balance: d(VC)/dt = Q_in C_in - Q_perm C_perm
        # - Q_waste C + V(r + aer), with C_perm = perm_mult * C (membrane rejects
        # the particulates). Solids leave only via the waste draw, so the MLSS
        # concentrates -- the defining MBR behaviour, SRT = V / Q_waste.
        convection = (q_in * s_in.C - q_perm * (self._perm_mult * C) - q_waste * C) / self.volume

        # Use the flow-weighted inlet temperature (seasonal influent) for both the
        # kinetics and the aeration, exactly as the CSTR does; fall back to the
        # static condition when no inlet carries a temperature.
        T_in = self._mixed_inlet_T(inputs)
        if T_in is not None and "T" in self._condition_arrays:
            conditions = {**self._condition_arrays, "T": jnp.reshape(T_in, (1,))}
        else:
            conditions = self._condition_arrays

        stoich = self.model.compute_stoich(params)
        rates = self.model.rates(C, params, conditions, 0)
        chemistry = stoich.T @ rates

        T_eff = T_in if T_in is not None else self.conditions.get("T")
        aeration = aeration_transfer(self._av, C, T_eff, signals, self.model)

        dC = convection + chemistry + aeration

        # Membrane fouling: resistance grows with the permeate flux and relaxes
        # (reversible fouling), reaching a quasi-steady fouled TMP.
        J = q_perm / self.membrane_area
        dR_f = jnp.reshape(self.fouling_rate * J - self.fouling_relax * R_f, (1,))
        return jnp.concatenate([dC, dR_f])
