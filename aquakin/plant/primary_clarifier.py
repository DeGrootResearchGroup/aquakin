"""BSM2 primary clarifier (Otterpohl & Freund 1992 / Gernaey et al. 2014).

A well-mixed holding tank (a CSTR with *no* reaction) whose content is split
into a clarified effluent and a thickened primary sludge by a retention-time-
dependent particulate-removal efficiency:

    n_COD = f_corr · (2.88·f_X − 0.118) · (1.45 + 6.15·ln(HRT_minutes))   [%]
    n_X   = clip(n_COD / f_X, 0, 100)                                     [%]

``n_X`` is the fraction of *particulate* COD removed to the sludge; solubles are
not removed. The primary-sludge (underflow) flow is a fixed fraction of the
feed, ``Q_u = f_PS · Q_in`` (concentration-independent, so the flow split is
exact for the recycle-flow pre-solve), and the thickening factor is
``E = Q_in / Q_u = 1 / f_PS``.

The HRT is computed from a **first-order filtered** inflow, as in BSM2:

    HRT = V / (Q_f + 0.001),    dQ_f/dt = (Q_in − Q_f) / t_m

so under a flow peak the removal efficiency lags the flow by ``t_m`` rather than
dropping instantly. The filter is identity at steady state but not dynamically:
on the BSM2 dynamic influent, evaluating the HRT on the instantaneous flow lets
too many particulates through at the peaks and raises the average effluent
ammonia by about 3%. With ``t_m > 0`` the state is ``[C..., Q_f]``; ``t_m = 0``
drops the filter (the HRT follows the instantaneous inflow) and the state is the
concentration vector alone. Only the removal efficiency reads ``Q_f`` — the flow
split stays on the instantaneous inflow, so it remains exact for the recycle-flow
pre-solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import ASM1_SETTLING_SPECIES, MINUTES_PER_DAY, species_mask
from aquakin.plant.flow_setpoint import FlowParameterized, FlowSetpoint
from aquakin.plant.streams import Stream, mixed_feed, mixed_scalars, total_flow

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


@dataclass
class PrimaryClarifier(FlowParameterized):
    """Otterpohl–Freund dynamic primary clarifier.

    Parameters
    ----------
    name : str
    model : CompiledModel
    volume : float
        Clarifier liquid volume (m³); sets the hydraulic retention time.
    input_port_names : list[str]
        Incoming-stream ports; multiple inflows are Q-weighted-mixed (the raw
        influent plus any recycled reject water).
    f_corr : float
        Efficiency correction factor ``rho`` (BSM2: 0.65).
    f_X : float
        Mean particulate-COD / total-COD ratio ``K`` (BSM2: 0.85).
    f_PS : float
        Primary-sludge (underflow) flow as a fraction of the feed (BSM2: 0.007).
    t_m : float
        Time constant (d) of the first-order inflow filter the HRT is computed
        from (BSM2: 0.125). Default ``0``: no filter, the HRT follows the
        instantaneous inflow and the state is the concentration vector alone.
        With ``t_m > 0`` the filtered flow is one extra trailing state entry.
    initial_flow : float, optional
        Initial filtered flow (m³/d), required when ``t_m > 0``. Set it to the
        design inflow, so a steady start or a warm start begins at the filter's
        steady state.
    settling_species : tuple[str, ...]
        Particulates removed to the sludge. Defaults to the ASM1 settling set.
    effluent_port, sludge_port : str
    """

    name: str
    model: CompiledModel
    volume: float
    input_port_names: list[str] = field(default_factory=lambda: ["inlet"])
    f_corr: float = 0.65
    f_X: float = 0.85
    f_PS: float = 0.007
    t_m: float = 0.0
    initial_flow: float | None = None
    settling_species: tuple[str, ...] = ASM1_SETTLING_SPECIES
    effluent_port: str = "effluent"
    sludge_port: str = "underflow"

    def __post_init__(self) -> None:
        if not (0.0 < self.f_PS < 1.0):
            raise ValueError(
                f"PrimaryClarifier '{self.name}': f_PS must be in (0, 1); got {self.f_PS}"
            )
        if self.t_m < 0.0:
            raise ValueError(f"PrimaryClarifier '{self.name}': t_m must be >= 0; got {self.t_m}")
        if self.t_m > 0.0 and self.initial_flow is None:
            raise ValueError(
                f"PrimaryClarifier '{self.name}': t_m > 0 filters the inflow, so "
                f"initial_flow (the design inflow, m3/d) must be given."
            )
        self._settle_mask = species_mask(self.model, self.settling_species, what="settling species")
        # Primary-sludge fraction as a differentiable setpoint, read by both the
        # flow rule and the material split.
        self._setpoints = {"f_PS": FlowSetpoint(float(self.f_PS), 0)}

    def _flow_setpoints(self) -> dict[str, FlowSetpoint]:
        return self._setpoints

    @property
    def _filtered(self) -> bool:
        return self.t_m > 0.0

    @property
    def state_size(self) -> int:
        return self.model.n_species + (1 if self._filtered else 0)

    def liquid_volume(self, state: jnp.ndarray):
        """The liquid volume held: the fixed tank volume.

        Declares the concentration vector as the head of the state (the trailing
        filtered-flow entry, when present, is not a concentration), so the mass
        balance and the per-species readers index it as ``state[:n_species]``.
        """
        return self.volume

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        return [self.effluent_port, self.sludge_port]

    def initial_state(self) -> jnp.ndarray:
        C0 = self.model.default_concentrations()
        if not self._filtered:
            return C0
        return jnp.concatenate([C0, jnp.asarray([float(self.initial_flow)])])

    def _hrt_flow(self, state: jnp.ndarray, Q_in: jnp.ndarray) -> jnp.ndarray:
        """The flow the HRT is computed from: the filtered flow, or ``Q_in``."""
        return state[self.model.n_species] if self._filtered else Q_in

    def _removal_fraction(self, Q: jnp.ndarray) -> jnp.ndarray:
        """Particulate-COD removal fraction n_X in [0, 1] from the HRT ``V/Q``."""
        # The two floors here are deliberate *physical* floors, not the shared
        # numerical EPS_Q divide-by-zero guard: 1e-3 m3/d caps the HRT at a large
        # but finite value as the feed vanishes (a zero-flow clarifier does not
        # retain forever), and 1e-6 min keeps the log argument in domain. Their
        # magnitudes are modelling choices, so they stay local.
        hrt_days = self.volume / (Q + 1e-3)
        hrt_min = hrt_days * MINUTES_PER_DAY
        n_cod = (
            self.f_corr
            * (2.88 * self.f_X - 0.118)
            * (1.45 + 6.15 * jnp.log(jnp.maximum(hrt_min, 1e-6)))
        )
        n_x = jnp.clip(n_cod / self.f_X, 0.0, 100.0) / 100.0
        return n_x

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        """Split the feed into a clarified effluent and a settled sludge stream.

        Returns
        -------
        dict of str to Stream
            The two outlet streams keyed by port name: ``effluent_port`` (the
            clarified overflow) and ``sludge_port`` (the underflow at flow
            ``f_PS·Q_in``). Solubles pass through; particulates lose the removal
            fraction to the sludge.
        """
        C = state[: self.model.n_species]
        Q_in = total_flow(inputs[name].Q for name in self.input_port_names)
        # Flow-weighted inlet side-channel scalars (temperature, ...), passed
        # through to both outlets.
        scalars_out = mixed_scalars(inputs, self.input_port_names)

        f_PS = self._setpoints["f_PS"].resolve(self._flow_params(params))
        Qu = f_PS * Q_in
        E = 1.0 / f_PS  # thickening factor Q_in/Q_u
        n_x = self._removal_fraction(self._hrt_flow(state, Q_in))

        # ff_i = fraction of species i that stays in the effluent. Solubles
        # (settle_mask=0) keep ff=1; particulates lose n_x to the sludge.
        ff = 1.0 - self._settle_mask * n_x
        C_eff = jnp.maximum(ff * C, 0.0)
        C_sludge = jnp.maximum(((1.0 - ff) * E + ff) * C, 0.0)

        return {
            self.effluent_port: Stream(Q=Q_in - Qu, C=C_eff, model=self.model, scalars=scalars_out),
            self.sludge_port: Stream(Q=Qu, C=C_sludge, model=self.model, scalars=scalars_out),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Exact linear flow rule: underflow = f_PS·Q_in, effluent the rest."""
        Q_in = total_flow(input_flows[name] for name in self.input_port_names)
        f_PS = self._setpoints["f_PS"].resolve(self._flow_params(params))
        Qu = f_PS * Q_in
        return {self.effluent_port: Q_in - Qu, self.sludge_port: Qu}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        # Well-mixed holding tank: convection only (no reaction).
        n = self.model.n_species
        Q_total, C_in = mixed_feed(inputs, self.input_port_names)
        dC = (Q_total / self.volume) * (C_in - state[:n])
        if not self._filtered:
            return dC
        dQ_f = (Q_total - state[n]) / self.t_m
        return jnp.concatenate([dC, jnp.reshape(dQ_f, (1,))])
