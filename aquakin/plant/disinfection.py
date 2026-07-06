"""Disinfection unit operations: UV dose-response and chlorine CT / log-removal.

Effluent-permit work needs disinfection sizing and a pathogen log-removal credit.
The reaction *models* ``uv_h2o2`` / ``ozone_bromate`` model the oxidation
chemistry, but neither is a disinfection *unit op* that reduces a pathogen
indicator in the flowsheet. This module adds two, plus the credit physics behind
them:

* :class:`UVUnit` -- a UV reactor. The **dose** is the average fluence rate times
  the exposure time (``V/Q``), corrected for the water's UV transmittance (UVT);
  a log-linear **dose-response** (with optional tailing) gives the
  log-inactivation.
* :class:`ChlorineContactUnit` -- a chlorine contact tank. The applied chlorine
  **residual** is a dynamic state that decays first-order; the **CT** (residual ×
  the T10 contact time) drives a CT-based log-removal credit. T10 comes from a
  baffling factor (``T10 = baffling·V/Q``) or, for a non-ideal contactor, from a
  measured/simulated residence-time distribution (:func:`t10_from_rtd`, reusing
  :mod:`aquakin.utils.rtd`).

Both **pass the process (ASM) stream through unchanged** -- disinfection does not
materially change COD/N/P at this fidelity -- and reduce the **indicator-organism
density carried on the stream** (``Stream.org``, the disinfection analogue of the
temperature scalar), falling back to the unit's design ``inlet_density`` when the
inlet carries none. So the indicator is tracked through the flowsheet (mixers
flow-weight it) and the disinfection unit applies the log-inactivation, matching
the behaviour of the commercial process simulators.

The credit physics is exposed as pure, AD-clean functions (``uv_dose`` /
``uv_log_inactivation`` / ``ct_value`` / ``ct_log_removal`` / ``t10_from_baffling``
/ ``t10_from_rtd``) so a contactor can be sized or a credit computed standalone.

Units: the standalone functions are unit-agnostic (the caller keeps them
consistent). The units convert the plant residence time ``V/Q`` (in the plant's
time unit -- days for the BSM family) to seconds for the UV dose (fluence rate in
mW/cm², dose in mJ/cm²); the chlorine T10/CT stay in the plant time unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

from aquakin.plant.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel

_EPS_Q = 1e-9  # guard 1/Q when the flow is ~zero
_SECONDS_PER_DAY = 86400.0


# --- UV credit physics ------------------------------------------------------


def uvt_intensity_factor(uvt, uvt_ref):
    """First-order UV-transmittance correction on the average fluence rate.

    The average intensity in a UV reactor falls as the water absorbs more UV. With
    ``uvt`` / ``uvt_ref`` the % transmittance (per cm) at operating / reference
    conditions, this returns the linear ratio ``uvt / uvt_ref`` -- a first-order
    correction; an exact treatment needs the reactor geometry and a
    dose-distribution (UVDGM) model. ``uvt is None`` returns 1 (no correction)."""
    if uvt is None or uvt_ref is None:
        return 1.0
    return jnp.asarray(uvt) / jnp.asarray(uvt_ref)


def uv_dose(intensity, exposure_time, *, uvt=None, uvt_ref=None):
    """UV dose (fluence) = average fluence rate × exposure time.

    ``intensity`` is the average fluence rate (e.g. mW/cm²) and ``exposure_time``
    the contact time in the matching time unit (seconds → dose in mJ/cm²),
    optionally scaled by the UVT correction :func:`uvt_intensity_factor`."""
    return jnp.asarray(intensity) * uvt_intensity_factor(uvt, uvt_ref) * jnp.asarray(exposure_time)


def uv_log_inactivation(dose, d10, *, max_log=None):
    """Log-inactivation from a UV dose by a log-linear dose-response.

    ``log10(N/N0) = dose / d10`` where ``d10`` is the dose for one log of
    inactivation (mJ/cm²/log). Optionally capped at ``max_log`` to represent
    tailing (a resistant subpopulation)."""
    log = jnp.asarray(dose) / d10
    if max_log is not None:
        log = jnp.minimum(log, max_log)
    return log


# --- chlorine credit physics ------------------------------------------------


def t10_from_baffling(volume, flow, baffling_factor):
    """T10 contact time from the baffling factor: ``T10 = baffling·V/Q``.

    The baffling factor is the ratio of the T10 (the time below which 10 % of the
    flow exits) to the mean hydraulic residence time ``V/Q`` -- ~0.1 for an
    unbaffled tank, ~0.7 for a well-baffled one, 1.0 for ideal plug flow."""
    return baffling_factor * jnp.asarray(volume) / (jnp.asarray(flow) + _EPS_Q)


def t10_from_rtd(t, C):
    """T10 contact time from a measured/simulated residence-time distribution.

    Reuses :func:`aquakin.utils.rtd.percentile_time` (the 10th percentile -- the
    ``q=0.10`` quantile -- of the cumulative residence-time distribution) -- the
    non-ideal-contactor alternative to a baffling-factor estimate."""
    from aquakin.utils.rtd import percentile_time

    return percentile_time(t, C, 0.10)


def ct_value(residual, t10):
    """CT = disinfectant residual × T10 contact time (e.g. mg·min/L)."""
    return jnp.asarray(residual) * jnp.asarray(t10)


def ct_log_removal(ct, ct_per_log, *, max_log=None):
    """Log-removal from a CT value by a CT-based credit (Chick–Watson form).

    ``log = CT / ct_per_log`` where ``ct_per_log`` is the CT that earns one log of
    inactivation (from the regulatory CT tables for the target organism, pH and
    temperature). Optionally capped at ``max_log``."""
    log = jnp.asarray(ct) / ct_per_log
    if max_log is not None:
        log = jnp.minimum(log, max_log)
    return log


def _apply_log_removal(org_in, log_inactivation):
    """Reduce an indicator density by a log-inactivation: ``N = N0·10^(-log)``."""
    return org_in * 10.0 ** (-log_inactivation)


# --- UV reactor unit --------------------------------------------------------


@dataclass
class UVUnit:
    """A UV disinfection reactor (stateless): dose-driven log-inactivation.

    Passes the process stream through unchanged and reduces the indicator-organism
    density (``Stream.org``, else ``inlet_density``) by the dose-response
    log-inactivation. The dose is ``intensity · exposure · UVT-factor`` with the
    exposure time the (baffling-scaled) residence time ``V/Q`` converted to
    seconds.

    Parameters
    ----------
    name : str
    model : CompiledModel
    volume : float
        Reactor volume (m³); the exposure time is ``baffling_factor·V/Q``.
    intensity : float
        Average UV fluence rate (mW/cm²) at the reference UVT.
    d10 : float
        Dose for one log of inactivation (mJ/cm²/log).
    uvt, uvt_ref : float, optional
        Operating / reference UV transmittance (% per cm) for the first-order
        intensity correction (:func:`uvt_intensity_factor`). Both ``None`` = no
        correction.
    max_log : float, optional
        Cap on the log-inactivation (tailing). ``None`` = uncapped.
    baffling_factor : float
        T10/HRT short-circuiting factor on the exposure time. Default 1.0.
    inlet_density : float
        Indicator density used when the inlet stream carries none (``org`` is
        ``None``). Default 0.
    input_port, output_port : str
    """

    name: str
    model: "CompiledModel"
    volume: float
    intensity: float
    d10: float
    uvt: Optional[float] = None
    uvt_ref: Optional[float] = None
    max_log: Optional[float] = None
    baffling_factor: float = 1.0
    inlet_density: float = 0.0
    input_port: str = "in"
    output_port: str = "out"

    # ----- stateless ------------------------------------------------------
    @property
    def state_size(self) -> int:
        return 0

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        return jnp.zeros((0,))

    def rhs(self, t, state, inputs, params, signals=None) -> jnp.ndarray:
        return jnp.zeros((0,))

    # ----- behaviour ------------------------------------------------------
    def exposure_seconds(self, flow) -> jnp.ndarray:
        """Exposure time (s): the baffling-scaled residence ``V/Q`` (plant time
        unit, days) converted to seconds."""
        hrt_days = self.baffling_factor * self.volume / (jnp.asarray(flow) + _EPS_Q)
        return hrt_days * _SECONDS_PER_DAY

    def log_inactivation(self, flow) -> jnp.ndarray:
        """The UV log-inactivation at the given throughflow."""
        dose = uv_dose(
            self.intensity, self.exposure_seconds(flow), uvt=self.uvt, uvt_ref=self.uvt_ref
        )
        return uv_log_inactivation(dose, self.d10, max_log=self.max_log)

    def compute_outputs(self, t, state, inputs, params, signals=None) -> dict:
        s_in = inputs[self.input_port]
        org_carried = s_in.scalars.get("org")
        org_in = org_carried if org_carried is not None else self.inlet_density
        org_out = _apply_log_removal(org_in, self.log_inactivation(s_in.Q))
        return {
            self.output_port: Stream(
                Q=s_in.Q, C=s_in.C, model=self.model, scalars={**s_in.scalars, "org": org_out}
            )
        }

    def flow_outputs(self, input_flows: dict, params, ctx=None) -> dict:
        return {self.output_port: input_flows[self.input_port]}


# --- chlorine contact unit --------------------------------------------------


@dataclass
class ChlorineContactUnit:
    """A chlorine contact tank: CT-driven log-removal with residual decay.

    The applied chlorine **residual** is a dynamic state (a completely-mixed tank
    with first-order decay, ``dCl/dt = (Q/V)(dose − Cl) − k_decay·Cl``); the CT
    credit ``residual × T10`` drives the log-removal. Passes the process stream
    through and reduces the indicator-organism density (``Stream.org``, else
    ``inlet_density``). ``dechlorinate`` reports the discharged residual as zero
    (the credit is already earned in the tank) -- e.g. a downstream bisulfite dose.

    Parameters
    ----------
    name : str
    model : CompiledModel
    volume : float
        Contact-tank volume (m³).
    dose : float
        Applied chlorine dose at the inlet (g Cl/m³ ≡ mg/L).
    ct_per_log : float
        CT (residual × T10) that earns one log of inactivation, in the same units
        as ``residual × time`` (from the regulatory CT tables for the organism /
        pH / temperature). **Time-unit trap:** ``T10 = baffling·V/Q`` is in the
        plant's time unit (days for the BSM models, whose rate constants are
        ``1/d``), while the regulatory CT tables are in **mg·min/L**. Supply
        ``ct_per_log`` in the *plant* time unit -- e.g. for a days-based plant
        multiply a mg·min/L table value by ``1/1440`` (min→day) -- or both the CT
        and the credit are off by ``1440×``. ``V``/``Q`` and ``decay_rate`` are in
        that same plant time unit.
    decay_rate : float
        First-order chlorine-decay rate (1/time, plant time unit). Default 0.
    baffling_factor : float
        T10/HRT factor for the contact time (``T10 = baffling·V/Q``). Default 0.5.
    max_log : float, optional
        Cap on the log-removal. ``None`` = uncapped.
    inlet_density : float
        Indicator density used when the inlet carries none. Default 0.
    initial_residual : float
        Initial chlorine residual state (g/m³). Default 0.
    dechlorinate : bool
        If true, the reported discharged residual is 0. Default False.
    input_port, output_port : str
    """

    name: str
    model: "CompiledModel"
    volume: float
    dose: float
    ct_per_log: float
    decay_rate: float = 0.0
    baffling_factor: float = 0.5
    max_log: Optional[float] = None
    inlet_density: float = 0.0
    initial_residual: float = 0.0
    dechlorinate: bool = False
    input_port: str = "in"
    output_port: str = "out"

    def __post_init__(self) -> None:
        if self.volume <= 0.0:
            raise ValueError(f"ChlorineContactUnit '{self.name}': volume must be > 0.")
        if self.ct_per_log <= 0.0:
            raise ValueError(f"ChlorineContactUnit '{self.name}': ct_per_log must be > 0.")

    # ----- one-state: chlorine residual -----------------------------------
    @property
    def state_size(self) -> int:
        return 1

    @property
    def input_ports(self) -> list[str]:
        return [self.input_port]

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        return jnp.asarray([float(self.initial_residual)])

    # ----- behaviour ------------------------------------------------------
    def ct(self, residual, flow) -> jnp.ndarray:
        """CT = residual × T10, with ``T10 = baffling·V/Q`` (plant time unit)."""
        t10 = t10_from_baffling(self.volume, flow, self.baffling_factor)
        return ct_value(residual, t10)

    def log_removal(self, residual, flow) -> jnp.ndarray:
        """The chlorine CT log-removal at the given residual and throughflow."""
        return ct_log_removal(self.ct(residual, flow), self.ct_per_log, max_log=self.max_log)

    def compute_outputs(self, t, state, inputs, params, signals=None) -> dict:
        s_in = inputs[self.input_port]
        residual = state[0]
        org_carried = s_in.scalars.get("org")
        org_in = org_carried if org_carried is not None else self.inlet_density
        org_out = _apply_log_removal(org_in, self.log_removal(residual, s_in.Q))
        return {
            self.output_port: Stream(
                Q=s_in.Q, C=s_in.C, model=self.model, scalars={**s_in.scalars, "org": org_out}
            )
        }

    def flow_outputs(self, input_flows: dict, params, ctx=None) -> dict:
        return {self.output_port: input_flows[self.input_port]}

    def rhs(self, t, state, inputs, params, signals=None) -> jnp.ndarray:
        # Completely-mixed contact tank: the inlet chlorine is the applied dose;
        # the residual fills toward it and decays first-order.
        s_in = inputs[self.input_port]
        residual = state[0]
        loading = s_in.Q / self.volume * (self.dose - residual)
        d_res = loading - self.decay_rate * residual
        return jnp.reshape(d_res, (1,))

    def discharged_residual(self, state) -> float:
        """The chlorine residual leaving the tank (0 if dechlorinated)."""
        return 0.0 if self.dechlorinate else float(jnp.asarray(state)[0])
