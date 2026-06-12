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

Only the well-mixed concentration vector is carried as state; the reference's
separate first-order flow-smoothing state (time constant ``t_m``, used solely to
smooth the HRT under fast flow transients) is omitted — it is identity at steady
state, the target of the open-loop BSM2 build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import ASM1_SETTLING_SPECIES
from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


@dataclass
class PrimaryClarifier:
    """Otterpohl–Freund dynamic primary clarifier.

    Parameters
    ----------
    name : str
    network : CompiledNetwork
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
    settling_species : tuple[str, ...]
        Particulates removed to the sludge. Defaults to the ASM1 settling set.
    effluent_port, sludge_port : str
    """

    name: str
    network: "CompiledNetwork"
    volume: float
    input_port_names: list[str] = field(default_factory=lambda: ["inlet"])
    f_corr: float = 0.65
    f_X: float = 0.85
    f_PS: float = 0.007
    settling_species: tuple[str, ...] = ASM1_SETTLING_SPECIES
    effluent_port: str = "effluent"
    sludge_port: str = "underflow"

    def __post_init__(self) -> None:
        if not (0.0 < self.f_PS < 1.0):
            raise ValueError(
                f"PrimaryClarifier '{self.name}': f_PS must be in (0, 1); "
                f"got {self.f_PS}"
            )
        mask = jnp.zeros((self.network.n_species,))
        for sp in self.settling_species:
            if sp in self.network.species_index:
                mask = mask.at[self.network.species_index[sp]].set(1.0)
        self._settle_mask = mask

    @property
    def state_size(self) -> int:
        return self.network.n_species

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        return [self.effluent_port, self.sludge_port]

    def initial_state(self) -> jnp.ndarray:
        return self.network.default_concentrations()

    def _removal_fraction(self, Q_in: jnp.ndarray) -> jnp.ndarray:
        """Particulate-COD removal fraction n_X in [0, 1] from the HRT."""
        hrt_days = self.volume / (Q_in + 1e-3)
        hrt_min = hrt_days * 1440.0
        n_cod = self.f_corr * (2.88 * self.f_X - 0.118) * (
            1.45 + 6.15 * jnp.log(jnp.maximum(hrt_min, 1e-6))
        )
        n_x = jnp.clip(n_cod / self.f_X, 0.0, 100.0) / 100.0
        return n_x

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
    ) -> dict[str, Stream]:
        Q_in = jnp.zeros(())
        for name in self.input_port_names:
            Q_in = Q_in + inputs[name].Q
        # Flow-weighted inlet temperature, passed through to both outlets.
        T_out = None
        if all(inputs[n].T is not None for n in self.input_port_names):
            heat = jnp.zeros(())
            for name in self.input_port_names:
                heat = heat + inputs[name].Q * inputs[name].T
            T_out = heat / (Q_in + 1e-12)

        Qu = self.f_PS * Q_in
        E = 1.0 / self.f_PS                       # thickening factor Q_in/Q_u
        n_x = self._removal_fraction(Q_in)

        # ff_i = fraction of species i that stays in the effluent. Solubles
        # (settle_mask=0) keep ff=1; particulates lose n_x to the sludge.
        ff = 1.0 - self._settle_mask * n_x
        C_eff = jnp.maximum(ff * state, 0.0)
        C_sludge = jnp.maximum(((1.0 - ff) * E + ff) * state, 0.0)

        return {
            self.effluent_port: Stream(
                Q=Q_in - Qu, C=C_eff, network=self.network, T=T_out
            ),
            self.sludge_port: Stream(Q=Qu, C=C_sludge, network=self.network, T=T_out),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Exact linear flow rule: underflow = f_PS·Q_in, effluent the rest."""
        Q_in = jnp.zeros(())
        for name in self.input_port_names:
            Q_in = Q_in + input_flows[name]
        Qu = self.f_PS * Q_in
        return {self.effluent_port: Q_in - Qu, self.sludge_port: Qu}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: "dict | None" = None,
    ) -> jnp.ndarray:
        # Well-mixed holding tank: convection only (no reaction).
        Q_total = jnp.zeros(())
        mass_total = jnp.zeros((self.network.n_species,))
        for name in self.input_port_names:
            s = inputs[name]
            Q_total = Q_total + s.Q
            mass_total = mass_total + s.Q * s.C
        C_in = mass_total / (Q_total + 1e-12)
        return (Q_total / self.volume) * (C_in - state)
