"""Ideal %TSS sludge separators — BSM2 thickener and dewatering units.

Both are the same stateless ideal model (Gernaey et al. 2014 BSM2): the
underflow is concentrated to a target solids fraction (``%TSS``) and a fixed
fraction of the incoming solids (``tss_removal_percent``) is captured to the
underflow, the rest leaving with the (reject-water) overflow. Solubles are not
separated — they pass through at the inlet concentration, split only by flow.

The two BSM2 uses differ only in the target solids fraction:

- **thickener** — ``target_tss_percent = 7``  (thickens the secondary wastage)
- **dewatering** — ``target_tss_percent = 28`` (dewaters the digester output)

Unlike the secondary clarifier's flow-controlled RAS pump, the underflow flow
here is *concentration-dependent*: ``Q_underflow = Q_in · removal/(100·f)`` with
``f = target_TSS / TSS_in``, so a denser feed is thickened into a smaller
underflow. ``compute_outputs`` implements this exactly; ``flow_outputs`` (used
only to seed the linear recycle-flow pre-solve) uses a fixed nominal underflow
fraction, since it cannot see the concentration — the concentration sweep then
carries the true split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import (
    ASM1_SETTLING_SPECIES,
    ASM1_TSS_FACTOR,
    ASM1_TSS_SPECIES,
)
from aquakin.plant.streams import Stream
from aquakin.plant.units import StatelessUnit

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.network import CompiledNetwork


@dataclass
class IdealThickener(StatelessUnit):
    """Stateless ideal %TSS separator (BSM2 thickener / dewatering).

    Parameters
    ----------
    name : str
    network : CompiledNetwork
    target_tss_percent : float
        Target solids fraction of the underflow, in percent (BSM2: 7 for the
        thickener, 28 for dewatering). The underflow is concentrated so its TSS
        equals ``target_tss_percent · 10000`` mg/L (``%`` → mg/L).
    tss_removal_percent : float
        Fraction of inflowing solids captured to the underflow, in percent
        (default 98 per BSM2). The remainder leaves with the overflow (reject).
    settling_species : tuple[str, ...]
        Particulates that are separated (concentrated to the underflow).
        Defaults to the ASM1 settling set (includes XND).
    tss_species : tuple[str, ...]
        Particulates that contribute to the TSS used for the thickening factor.
    tss_factor : float
        COD→TSS conversion factor for ``tss_species`` (Copp 2002, 0.75).
    nominal_underflow_fraction : float
        Fixed underflow/feed flow ratio used *only* by ``flow_outputs`` to seed
        the linear recycle-flow pre-solve (which cannot see concentration). The
        true, concentration-dependent split is applied in ``compute_outputs``.
    input_port, overflow_port, underflow_port : str
    """

    name: str
    network: "CompiledNetwork"
    target_tss_percent: float
    tss_removal_percent: float = 98.0
    settling_species: tuple[str, ...] = ASM1_SETTLING_SPECIES
    tss_species: tuple[str, ...] = ASM1_TSS_SPECIES
    tss_factor: float = ASM1_TSS_FACTOR
    nominal_underflow_fraction: float = 0.02
    input_port: str = "inlet"
    overflow_port: str = "overflow"
    underflow_port: str = "underflow"

    # state_size / initial_state / rhs come from StatelessUnit.

    def __post_init__(self) -> None:
        if not (0.0 <= self.tss_removal_percent <= 100.0):
            raise ValueError(
                f"IdealThickener '{self.name}': tss_removal_percent must be in "
                f"[0, 100]; got {self.tss_removal_percent}"
            )
        if self.target_tss_percent <= 0.0:
            raise ValueError(
                f"IdealThickener '{self.name}': target_tss_percent must be "
                f"positive; got {self.target_tss_percent}"
            )
        n = self.network.n_species
        idx = self.network.species_index
        # Mask of separated particulates (1.0 where concentrated/thinned).
        settle = jnp.zeros((n,))
        for sp in self.settling_species:
            if sp in idx:
                settle = settle.at[idx[sp]].set(1.0)
        self._settle_mask = settle
        self._soluble_mask = 1.0 - settle
        # TSS contribution vector (factor on the tss_species, 0 elsewhere).
        tss = jnp.zeros((n,))
        for sp in self.tss_species:
            if sp in idx:
                tss = tss.at[idx[sp]].set(self.tss_factor)
        self._tss_vec = tss
        self._target_tss = float(self.target_tss_percent) * 1.0e4  # %TSS -> mg/L
        self._removal_frac = float(self.tss_removal_percent) / 100.0

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.overflow_port, self.underflow_port]

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
        C = s_in.C

        tss_in = jnp.sum(self._tss_vec * C)
        # Thickening factor f = target_TSS / TSS_in. f > 1 means the feed can be
        # concentrated to the target; f <= 1 means it is already at/above target,
        # so everything leaves with the underflow (overflow is empty).
        f = self._target_tss / jnp.maximum(tss_in, 1e-12)
        can = f > 1.0
        Qu_factor = self._removal_frac / jnp.maximum(f, 1e-12)
        thin_factor = (1.0 - self._removal_frac) / jnp.maximum(1.0 - Qu_factor, 1e-12)

        # Per-species multipliers for the two outlets. Settled particulates scale
        # by f (underflow) / thin_factor (overflow); solubles pass through (×1)
        # to the underflow and split by flow to the overflow (also ×1, the flow
        # split carries the partition).
        uf_scale = jnp.where(
            can, self._settle_mask * f + self._soluble_mask, 1.0
        )
        of_scale = jnp.where(
            can, self._settle_mask * thin_factor + self._soluble_mask, 0.0
        )
        C_under = C * uf_scale
        C_over = C * of_scale

        Q_under = jnp.where(can, Q_in * Qu_factor, Q_in)
        Q_over = jnp.where(can, Q_in * (1.0 - Qu_factor), 0.0)

        return {
            self.underflow_port: Stream(Q=Q_under, C=C_under, network=self.network, T=s_in.T),
            self.overflow_port: Stream(Q=Q_over, C=C_over, network=self.network, T=s_in.T),
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Nominal linear flow rule for the recycle-flow pre-solve.

        The true underflow flow is concentration-dependent (see the class
        docstring), which the concentration-free flow network cannot express, so
        this uses the fixed ``nominal_underflow_fraction``. It only seeds the
        recycle edges; ``compute_outputs`` applies the exact split during the
        concentration sweep.
        """
        Q_in = input_flows[self.input_port]
        frac = jnp.asarray(float(self.nominal_underflow_fraction))
        return {
            self.underflow_port: Q_in * frac,
            self.overflow_port: Q_in * (1.0 - frac),
        }
