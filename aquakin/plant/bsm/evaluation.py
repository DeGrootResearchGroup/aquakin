"""BSM2 performance evaluation: EQI / OCI from a plant solution.

The generic metric kernels live in :mod:`aquakin.plant.metrics`; this module
wires them to a concrete BSM2 flowsheet (built by :func:`build_bsm2`). It
reconstructs the quantities the indices need from a solved
:class:`~aquakin.plant.plant.PlantSolution` -- the secondary-effluent stream,
the (possibly control-varying) aeration kLa of the aerated reactors, the pumped
recycle flows, and the wasted-sludge mass flow -- and returns the headline
**EQI** (effluent quality index) and **OCI** (operational cost index).

The aeration term reads the *actual* kLa over the run: for an open-loop plant
that is each aerated tank's fixed ``kla``; under closed-loop DO control
(``build_bsm2(do_control=True)``) it is the controller's manipulated signal,
recovered per saved state via :meth:`Plant.signals_at`. So evaluating an
open- and a closed-loop run side by side quantifies what the control buys.

OCI here is the BSM1-form index (aeration + pumping + 5x sludge production);
BSM2's full OCI additionally credits methane production and charges mixing /
sludge-heating energy -- those terms are not yet included (see ``oci_note``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp

from aquakin.plant.metrics import (
    aeration_energy,
    derived_TSS,
    effluent_averages,
    effluent_quality_index,
    operational_cost_index,
    pumping_energy,
)

# Default BSM2 port names (as wired by build_bsm2).
_EFFLUENT_PORT = "settler.overflow"
_DISPOSAL_PORT = "dewatering.underflow"
_INTERNAL_RECYCLE_PORT = "tank5_split.internal_recycle"
_RAS_PORT = "underflow_split.ras"
_WASTE_PORT = "underflow_split.waste"
_DO_SATURATION = 8.0  # gO2/m^3


@dataclass
class BSM2Evaluation:
    """Headline BSM2 performance indices from a solved plant.

    Attributes
    ----------
    eqi : float
        Effluent Quality Index (kg pollutant / day), lower is better.
    oci : float
        Operational Cost Index (BSM1 form: ``AE + PE + 5*sludge``).
    aeration_energy : float
        Aeration energy (kWh/d).
    pumping_energy : float
        Pumping energy of the AS recycles (kWh/d).
    sludge_production : float
        Wasted-sludge TSS mass flow to disposal (kg TSS/d), time-averaged.
    effluent : dict
        Time/flow-weighted average effluent concentrations (COD, BOD, TSS, TKN,
        SNH, SNO; g/m^3) from :func:`effluent_averages`.
    aerated_tanks : list[str]
        The reactors whose aeration was counted.
    oci_note : str
        What the OCI omits relative to the full BSM2 definition.
    """

    eqi: float
    oci: float
    aeration_energy: float
    pumping_energy: float
    sludge_production: float
    effluent: dict = field(default_factory=dict)
    aerated_tanks: list = field(default_factory=list)
    oci_note: str = (
        "BSM1-form OCI (aeration + pumping + 5x sludge); BSM2 mixing/heating "
        "energy and the methane-production credit are not included."
    )


def _aerated_tanks(plant) -> list:
    """Reactors with aeration -- a fixed ``kla`` or a ``controlled_kla`` on SO."""
    tanks = []
    for name in plant._unit_order:
        unit = plant.units[name]
        if getattr(unit, "kla", None) or getattr(unit, "controlled_kla", None):
            tanks.append(name)
    return tanks


def _kla_history(plant, solution, params, tanks) -> jnp.ndarray:
    """Reconstruct each aerated tank's kLa at every saved time, ``(n_t, n_tanks)``.

    A tank under DO control reads its kLa from the control signal (recovered via
    :meth:`Plant.signals_at`); otherwise it uses its fixed ``kla``.
    """
    need_signals = any(plant.units[n].controlled_kla for n in tanks)
    rows = []
    for i in range(solution.t.shape[0]):
        sig = (plant.signals_at(solution.t[i], solution.state[i], params)
               if need_signals else {})
        row = []
        for n in tanks:
            unit = plant.units[n]
            if unit.controlled_kla and "SO" in unit.controlled_kla:
                signal_name, gain = unit.controlled_kla["SO"]
                row.append(float(sig[signal_name]) * gain)
            else:
                row.append(float(unit.kla.get("SO", 0.0)))
        rows.append(row)
    return jnp.asarray(rows)


def _time_average(t: jnp.ndarray, values: jnp.ndarray) -> float:
    """Trapezoidal time-average of ``values(t)`` over ``[t0, t1]``."""
    T = float(t[-1] - t[0])
    return float(jnp.trapezoid(values, t) / (T + 1e-12))


def evaluate_bsm2(
    plant,
    solution,
    params: Optional[jnp.ndarray] = None,
    *,
    effluent_port: str = _EFFLUENT_PORT,
    disposal_port: str = _DISPOSAL_PORT,
    internal_recycle_port: str = _INTERNAL_RECYCLE_PORT,
    ras_port: str = _RAS_PORT,
    waste_port: str = _WASTE_PORT,
    do_saturation: float = _DO_SATURATION,
) -> BSM2Evaluation:
    """Compute the BSM2 performance indices from a solved plant.

    Parameters
    ----------
    plant : Plant
        A BSM2 plant from :func:`build_bsm2` (open- or closed-loop).
    solution : PlantSolution
        A solution from ``plant.solve`` over the evaluation window. Use a fine
        enough ``t_eval`` to resolve the influent dynamics; the indices are
        trapezoidal time-integrals over the saved points.
    params : jnp.ndarray, optional
        The plant parameters used for the run (defaults to the plant defaults).
    effluent_port, disposal_port, internal_recycle_port, ras_port, waste_port :
        str, optional
        Stream endpoints to reconstruct; the defaults match ``build_bsm2``.
    do_saturation : float, optional
        DO saturation used in the aeration-energy formula (gO2/m^3).

    Returns
    -------
    BSM2Evaluation
        EQI, OCI and the component terms.
    """
    network = plant.units["tank1"].network

    # ----- Effluent quality (secondary clarifier overflow). -----
    eff = plant.stream(solution, effluent_port, params=params)
    eqi = effluent_quality_index(eff.t, eff.C, eff.Q, network)
    averages = effluent_averages(eff.t, eff.C, eff.Q, network)

    # ----- Aeration energy (actual kLa over the run). -----
    tanks = _aerated_tanks(plant)
    kla_hist = _kla_history(plant, solution, params, tanks)
    volumes = jnp.asarray([float(plant.units[n].volume) for n in tanks])
    AE = aeration_energy(solution.t, kla_hist, volumes, saturation=do_saturation)

    # ----- Pumping energy (AS internal recycle + RAS + wastage). -----
    Q_int = plant.stream(solution, internal_recycle_port, params=params).Q
    Q_ras = plant.stream(solution, ras_port, params=params).Q
    Q_was = plant.stream(solution, waste_port, params=params).Q
    PE = pumping_energy(solution.t, Q_int, Q_ras, Q_was)

    # ----- Sludge production (TSS mass flow leaving to disposal, kg/d). -----
    disposal = plant.stream(solution, disposal_port, params=params)
    tss_mass_flow = derived_TSS(disposal.C, network) * disposal.Q * 1e-3
    sludge = _time_average(solution.t, tss_mass_flow)

    oci = operational_cost_index(AE, PE, sludge)

    return BSM2Evaluation(
        eqi=eqi, oci=oci, aeration_energy=AE, pumping_energy=PE,
        sludge_production=sludge, effluent=averages, aerated_tanks=tanks,
    )
