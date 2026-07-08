"""Effluent quality, operational cost, and derived-quantity metrics.

The headline numbers BSM1 reports are:

- **EQI** (Effluent Quality Index) — weighted sum of total suspended
  solids, COD, BOD, TKN, and NO₃-N in the effluent, integrated over
  the simulation window. Lower is better.
- **OCI** (Operational Cost Index) — aeration energy + pumping energy +
  sludge production + mixing energy.

Both are defined in Copp 2002 / Alex 2008 with specific weighting factors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.plant._constants import (
    ASM1_TSS_FACTOR,
    ASM1_TSS_SPECIES,
    EPS_Q,
    HOURS_PER_DAY,
    SECONDS_PER_DAY,
)

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


# ASM1 → TSS conversion (Copp 2002): TSS = 0.75 * (X_S + X_I + X_BH + X_BA + X_P).
_TSS_FACTOR = ASM1_TSS_FACTOR
_TSS_SPECIES = ASM1_TSS_SPECIES

# EQI weighting factors (g pollutant / m³)⁻¹ from Copp 2002 / Alex 2008.
_EQI_WEIGHTS = {
    "TSS": 2.0,
    "COD": 1.0,
    "BOD": 2.0,
    "TKN": 30.0,  # total Kjeldahl nitrogen
    "NO": 10.0,  # nitrate-nitrogen
}


def _species_idx(model: CompiledModel, names) -> jnp.ndarray:
    """Index array for the named species that exist in the model."""
    return jnp.asarray([model.species_index[s] for s in names if s in model.species_index])


def _is_stream(x) -> bool:
    """True for a :class:`~aquakin.plant.streams.StreamSeries` (duck-typed)."""
    return hasattr(x, "C") and hasattr(x, "model") and hasattr(x, "t")


def _conc_model(C, model):
    """Accept a ``StreamSeries`` (use its ``C`` + ``model``) or an explicit
    ``(C, model)`` pair. Lets ``derived_TSS(effluent)`` work."""
    if _is_stream(C):
        return C.C, (model if model is not None else C.model)
    return C, model


def _composition(model, params=None):
    """``(i_XB, i_XP, f_P)`` for the derived TKN and BOD quantities.

    Read from ``params`` when given (the composition the simulation actually
    used, so a calibrated or benchmark-specific value such as the BSM2
    ``i_XB = 0.08`` is honoured), else from the model's declared defaults, else
    the standard ASM1 values. This keeps the post-processed nitrogen and BOD
    consistent with the model the states came from instead of a fixed constant.
    """

    def g(name, fallback):
        idx = getattr(model, "param_index", {}).get(name) if model else None
        if idx is None:
            return fallback
        if params is not None:
            return float(params[idx])
        return float(model.default_parameters()[idx])

    return g("i_XB", 0.086), g("i_XP", 0.06), g("f_P", 0.08)


def _effluent_args(stream_or_t, C, Q, model):
    """Normalise the effluent-metric arguments.

    Accepts either a ``StreamSeries`` -- in which case the *second* positional is
    an optional ``model`` override (the stream already carries ``t/C/Q``), so
    ``effluent_quality_index(eff)`` and ``effluent_quality_index(eff, model)``
    both work -- or the explicit ``(t, C, Q, model)`` form.
    """
    if _is_stream(stream_or_t):
        s = stream_or_t
        return s.t, s.C, s.Q, (C if C is not None else s.model)
    return stream_or_t, C, Q, model


def time_average(integrand, t, axis: int = 0):
    """Trapezoidal time-average of ``integrand`` over the window ``[t0, t1]``.

    The **single** public helper behind every time-averaged index -- the
    ``design``, ``aeration_system``, ``ghg`` and ``evaluation`` modules all call
    it directly rather than re-wrapping it. The argument order is always
    ``(values, t)`` (values first, times second); keep it that way so no module
    re-introduces an inverted-signature local copy.

    For a **single saved point** -- exactly what :meth:`Plant.run_to_steady_state`
    returns (the terminal state only) -- the window has zero width, but the
    time-average of a constant *is* that constant, so the single sample is
    returned directly. This yields the meaningful **instantaneous (steady-state)
    value** for a one-point solution instead of dividing by a zero window (which
    previously raised ``ZeroDivisionError`` in ``aeration_energy``, or gave a
    spurious zero in the guarded kernels). The metric kernels here are called
    eagerly on concrete arrays after a solve, so the ``t.shape[0]`` branch is a
    static check.

    Parameters
    ----------
    integrand : array
        Values at the save times, with the time axis at ``axis``.
    t : array
        Save times, shape ``(n_t,)``.
    axis : int
        The time axis of ``integrand`` (default 0).

    Returns
    -------
    jnp.ndarray
        The time-average, reduced along ``axis`` (a 0-d array for a 1-D
        ``integrand``). Callers that need a Python ``float`` wrap the result in
        ``float(...)``, as the scalar metric kernels here do.
    """
    t = jnp.asarray(t)
    integrand = jnp.asarray(integrand)
    if t.shape[0] <= 1:
        return jnp.take(integrand, 0, axis=axis)
    return jnp.trapezoid(integrand, t, axis=axis) / (t[-1] - t[0])


# Every derived quantity below indexes ``C`` with ``C[..., i]`` (a scalar
# column) and, where it sums, sums over ``axis=-1``. That works for both a 1-D
# state vector ``(n_species,)`` -> scalar and a 2-D trajectory
# ``(n_t, n_species)`` -> vector, so no rank branch is needed.


def derived_TSS(C, model: CompiledModel | None = None) -> jnp.ndarray:
    """Total suspended solids from ASM1 particulate species.

    ``TSS = 0.75 × (XS + XI + XB_H + XB_A + XP)``. Scalar for 1-D ``C``, a
    leading-axis vector for 2-D ``C``. ``C`` may be a concentration array (with
    an explicit ``model``) or a ``StreamSeries`` (model taken from it).
    """
    C, model = _conc_model(C, model)
    return _TSS_FACTOR * jnp.sum(C[..., _species_idx(model, _TSS_SPECIES)], axis=-1)


def derived_COD(C, model: CompiledModel | None = None) -> jnp.ndarray:
    """Total COD = SI + SS + XI + XS + XB_H + XB_A + XP.

    ``C`` may be a concentration array (with ``model``) or a ``StreamSeries``.
    """
    C, model = _conc_model(C, model)
    species = ("SI", "SS", "XI", "XS", "XB_H", "XB_A", "XP")
    return jnp.sum(C[..., _species_idx(model, species)], axis=-1)


def derived_BOD(C, model: CompiledModel | None = None, *, f_P: float | None = None) -> jnp.ndarray:
    """BOD₅ proxy = 0.25 × (SS + XS + (1 - f_P) × (XB_H + XB_A)), Copp 2002.

    ``f_P`` defaults to the model's declared inert-fraction (the standard ASM1
    0.08 when undeclared). ``C`` may be a concentration array (with ``model``)
    or a ``StreamSeries``.
    """
    C, model = _conc_model(C, model)
    if f_P is None:
        _, _, f_P = _composition(model)
    i = model.species_index
    return 0.25 * (
        C[..., i["SS"]] + C[..., i["XS"]] + (1.0 - f_P) * (C[..., i["XB_H"]] + C[..., i["XB_A"]])
    )


def derived_TKN(
    C, model: CompiledModel | None = None, *, i_XB: float | None = None, i_XP: float | None = None
) -> jnp.ndarray:
    """Total Kjeldahl Nitrogen = S_NH + S_ND + X_ND + i_XB × (XB_H + XB_A)
    + i_XP × (XP + XI).

    ``i_XB`` / ``i_XP`` default to the model's declared N-fractions (the
    standard ASM1 0.086 / 0.06 when undeclared); the BSM2 parameter set uses
    ``i_XB = 0.08``. ``C`` may be a concentration array (with ``model``) or a
    ``StreamSeries``.
    """
    C, model = _conc_model(C, model)
    if i_XB is None or i_XP is None:
        d_XB, d_XP, _ = _composition(model)
        i_XB = d_XB if i_XB is None else i_XB
        i_XP = d_XP if i_XP is None else i_XP
    i = model.species_index
    return (
        C[..., i["SNH"]]
        + C[..., i["SND"]]
        + C[..., i["XND"]]
        + i_XB * (C[..., i["XB_H"]] + C[..., i["XB_A"]])
        + i_XP * (C[..., i["XP"]] + C[..., i["XI"]])
    )


def effluent_averages(
    stream_or_t,
    C_traj=None,
    Q_traj=None,
    model: CompiledModel | None = None,
    *,
    params=None,
) -> dict[str, float]:
    """Time-flow-weighted average effluent concentrations.

    Accepts either a :class:`~aquakin.plant.streams.StreamSeries` (the usual
    ``plant.stream(sol, "clarifier.overflow")`` result) -- ``effluent_averages(eff)``
    -- or the explicit ``(t, C_traj, Q_traj, model)`` form.

    Parameters
    ----------
    stream_or_t : StreamSeries or jnp.ndarray
        A reconstructed effluent stream, or the save-time vector ``(n_t,)``.
    C_traj : jnp.ndarray, optional
        Effluent concentration trajectory, shape ``(n_t, n_species)`` (only when
        ``stream_or_t`` is a time vector).
    Q_traj : jnp.ndarray, optional
        Effluent flow rate trajectory, shape ``(n_t,)`` (likewise).
    model : CompiledModel, optional
        Model; taken from the stream when one is passed.

    Returns
    -------
    dict[str, float]
        Time-averaged COD, BOD, TSS, TKN, SNH, SNO (g/m³).
    """
    t, C_traj, Q_traj, model = _effluent_args(stream_or_t, C_traj, Q_traj, model)
    i_XB, i_XP, f_P = _composition(model, params)
    t = jnp.asarray(t)
    single_point = t.shape[0] <= 1
    # Use trapezoidal integration over time.
    dt = jnp.diff(t)
    # Flow-weight via Q.
    weight = 0.5 * (Q_traj[:-1] + Q_traj[1:]) * dt  # (n_t - 1,)
    total_w = jnp.sum(weight)

    def time_avg(values: jnp.ndarray) -> float:
        # A single saved point (a steady-state solution) has a zero-width window;
        # the flow-weighted average of a constant is that sample.
        if single_point:
            return float(values[0])
        v_mid = 0.5 * (values[:-1] + values[1:])
        return float(jnp.sum(v_mid * weight) / (total_w + EPS_Q))

    return {
        "TSS": time_avg(derived_TSS(C_traj, model)),
        "COD": time_avg(derived_COD(C_traj, model)),
        "BOD": time_avg(derived_BOD(C_traj, model, f_P=f_P)),
        "TKN": time_avg(derived_TKN(C_traj, model, i_XB=i_XB, i_XP=i_XP)),
        "SNH": time_avg(C_traj[:, model.species_index["SNH"]]),
        "SNO": time_avg(C_traj[:, model.species_index["SNO"]]),
    }


def effluent_quality_index(
    stream_or_t,
    C_traj=None,
    Q_traj=None,
    model: CompiledModel | None = None,
    *,
    params=None,
) -> float:
    """EQI per Copp 2002 / Alex 2008.

    ``EQI = (1 / T) × ∫ Q × (B_TSS×TSS + B_COD×COD + B_BOD×BOD
                              + B_TKN×TKN + B_NO×SNO) dt × 1e-3``

    Units: kg pollutant / day, averaged over the simulation window. Accepts either
    a :class:`~aquakin.plant.streams.StreamSeries` -- ``effluent_quality_index(eff)``
    -- or the explicit ``(t, C_traj, Q_traj, model)`` form.
    """
    t, C_traj, Q_traj, model = _effluent_args(stream_or_t, C_traj, Q_traj, model)
    i_XB, i_XP, f_P = _composition(model, params)
    TSS_t = derived_TSS(C_traj, model)
    COD_t = derived_COD(C_traj, model)
    BOD_t = derived_BOD(C_traj, model, f_P=f_P)
    TKN_t = derived_TKN(C_traj, model, i_XB=i_XB, i_XP=i_XP)
    SNO_t = C_traj[:, model.species_index["SNO"]]

    integrand = Q_traj * (
        _EQI_WEIGHTS["TSS"] * TSS_t
        + _EQI_WEIGHTS["COD"] * COD_t
        + _EQI_WEIGHTS["BOD"] * BOD_t
        + _EQI_WEIGHTS["TKN"] * TKN_t
        + _EQI_WEIGHTS["NO"] * SNO_t
    )
    return float(time_average(integrand, t) * 1e-3)


def aeration_energy(
    t: jnp.ndarray,
    kla_history: jnp.ndarray,
    volumes: jnp.ndarray,
    saturation: float = 8.0,
) -> float:
    """Aeration energy (kWh/d) per Copp 2002 eq.

    AE = (S_sat / (T × 1.8 × 1000)) × ∫ Σ_i V_i × kLa_i(t) dt

    Parameters
    ----------
    t : (n_t,) save times in days
    kla_history : (n_t, n_aerated_tanks) kLa value at each save time
    volumes : (n_aerated_tanks,) liquid volume of each aerated tank
    saturation : float
        Dissolved-oxygen saturation concentration (mg/L).

    Returns
    -------
    float
        Aeration energy in kWh/d, time-averaged over ``t``.
    """
    kla_history = jnp.asarray(kla_history)
    volumes = jnp.asarray(volumes)
    integrand = jnp.sum(kla_history * volumes[None, :], axis=1)
    return float(saturation / (1.8 * 1000.0) * time_average(integrand, t))


def pumping_energy(
    t: jnp.ndarray,
    Q_internal: jnp.ndarray,
    Q_ras: jnp.ndarray,
    Q_was: jnp.ndarray,
) -> float:
    """Pumping energy (kWh/d) per Copp 2002 eq.

    PE = (1 / T) × ∫ (0.004 × Q_internal + 0.008 × Q_ras + 0.05 × Q_was) dt

    Returns
    -------
    float
        Pumping energy in kWh/d, time-averaged over ``t``.
    """
    integrand = 0.004 * Q_internal + 0.008 * Q_ras + 0.05 * Q_was
    return float(time_average(integrand, t))


def operational_cost_index(
    aeration: float,
    pumping: float,
    sludge_production: float,
    mixing: float = 0.0,
) -> float:
    """OCI (BSM1 form):

    OCI = aeration + pumping + mixing + 5 × sludge_production

    The original Copp (2002) index omits the mixing term (``mixing=0``); the
    updated open-loop benchmark adds the mechanical-mixing energy of the
    unaerated reactors, so the two conventions differ only by that term.
    Sludge_production is the time-averaged TSS mass flow leaving via
    wastage + the change in plant TSS inventory.
    """
    return float(aeration + pumping + mixing + 5.0 * sludge_production)


# ---- BSM2 OCI component kernels (Gernaey et al. 2014) -----------------------

# Default BSM2 pumping-energy factors (kWh/m³), one per pumped stream.
_BSM2_PUMP_FACTORS = {
    "internal": 0.004,
    "ras": 0.008,
    "wastage": 0.05,
    "primary_underflow": 0.075,
    "thickener_underflow": 0.060,
    "dewatering_underflow": 0.004,
}


def pumping_energy_bsm2(
    t: jnp.ndarray,
    flows: dict[str, jnp.ndarray],
    factors: dict[str, float] | None = None,
) -> float:
    """Pumping energy (kWh/d) for the full BSM2 pump set.

    ``PE = (1/T) × ∫ Σ_k PF_k × Q_k dt`` over the pumped streams: the AS internal
    recirculation, sludge recycle and wastage, plus the primary / thickener /
    dewatering underflows. ``flows`` maps a stream key to its ``(n_t,)`` flow
    trajectory; ``factors`` maps the same keys to per-m³ energy factors (default
    :data:`_BSM2_PUMP_FACTORS`). Keys present in ``flows`` but not ``factors``
    (or vice versa) are ignored.

    Returns
    -------
    float
        Pumping energy in kWh/d, time-averaged over ``t``.
    """
    factors = _BSM2_PUMP_FACTORS if factors is None else factors
    integrand = jnp.zeros_like(jnp.asarray(t))
    for key, Q in flows.items():
        if key in factors:
            integrand = integrand + float(factors[key]) * jnp.asarray(Q)
    return float(time_average(integrand, t))


def mixing_energy(
    t: jnp.ndarray,
    kla_history: jnp.ndarray,
    volumes: jnp.ndarray,
    digester_volume: float,
    kla_threshold: float = 20.0,
    reactor_unit: float = 0.005,
    digester_unit: float = 0.005,
) -> float:
    """Mixing energy (kWh/d) per Gernaey et al. 2014.

    A reactor needs mechanical mixing only while it is *not* aerated
    (``kLa < kla_threshold``); an aerated tank is mixed by the aeration. The
    anaerobic digester is always mechanically mixed. With unit mixing powers in
    kW/m³ (default 0.005 for both)::

        ME = 24 × [ Σ_i reactor_unit × V_i × frac_unaerated_i
                    + digester_unit × V_digester ]

    where ``frac_unaerated_i`` is the time fraction reactor ``i`` has
    ``kLa < kla_threshold``.

    Parameters
    ----------
    t : (n_t,) save times in days.
    kla_history : (n_t, n_reactors) kLa per reactor at each save time.
    volumes : (n_reactors,) reactor liquid volumes.
    digester_volume : float
        Anaerobic-digester liquid volume.

    Returns
    -------
    float
        Mixing energy in kWh/d.
    """
    kla_history = jnp.asarray(kla_history)
    volumes = jnp.asarray(volumes)
    # Time fraction each reactor is below the aeration threshold.
    unaerated = (kla_history < kla_threshold).astype(jnp.float64)  # (n_t, n_reac)
    frac = time_average(unaerated, t, axis=0)  # (n_reac,)
    reactor_mix = reactor_unit * jnp.sum(volumes * frac)
    digester_mix = digester_unit * float(digester_volume)
    return float(HOURS_PER_DAY * (reactor_mix + digester_mix))


def carbon_mass(
    t: jnp.ndarray,
    Q_carbon: jnp.ndarray,
    carbon_conc: float,
) -> float:
    """External-carbon mass dose (kg COD/d), time-averaged.

    ``= (1/T) × ∫ Q_carbon(t) × carbon_conc dt × 1e-3`` (the dose flow times the
    source COD concentration, g→kg).

    Returns
    -------
    float
        External-carbon mass dose in kg COD/d, time-averaged over ``t``.
    """
    integrand = jnp.asarray(Q_carbon) * float(carbon_conc) * 1e-3
    return float(time_average(integrand, t))


def heating_energy(
    t: jnp.ndarray,
    Q_feed: jnp.ndarray,
    T_feed_C: jnp.ndarray,
    T_target_C: float = 35.0,
    rho: float = 1000.0,
    cp: float = 4.186,
) -> float:
    """Digester sludge-heating energy (kWh/d) per Gernaey et al. 2014.

    Energy to raise the digester feed from ``T_feed_C`` to ``T_target_C``::

        Heatpower [kW] = (T_target − T_feed) × Q_feed × rho × cp / 86400
        HE [kWh/d]     = 24 × time-average(Heatpower)

    with water density ``rho`` (kg/m³) and specific heat ``cp`` (kJ/kg·°C).
    Temperatures are in **Celsius**. A feed already above the target contributes
    negative heating (no cooling credit is taken here; the OCI applies the
    methane offset separately).

    Parameters
    ----------
    Q_feed : (n_t,) digester feed flow (m³/d).
    T_feed_C : (n_t,) or scalar feed temperature (°C).

    Returns
    -------
    float
        Digester sludge-heating energy in kWh/d, time-averaged over ``t``.
    """
    heatpower = (
        (float(T_target_C) - jnp.asarray(T_feed_C))
        * jnp.asarray(Q_feed)
        * rho
        * cp
        / SECONDS_PER_DAY
    )  # kW
    return float(HOURS_PER_DAY * time_average(heatpower, t))


def bsm2_oci_terms(
    aeration: float,
    pumping: float,
    mixing: float,
    sludge_production: float,
    carbon: float,
    methane: float,
    heating: float,
) -> list:
    """Itemized BSM2 OCI contributions -- the single source of the OCI weights.

    Returns a list of ``(key, value, contribution)`` rows, where ``value`` is the
    raw physical term and ``contribution`` is its signed addition to the OCI
    (``None`` for the raw ``heating`` term, which enters the index non-linearly
    through ``net_heating = max(0, heating − 7·methane)``).
    :func:`operational_cost_index_bsm2` sums the contributions and the BSM2
    evaluation ``report()`` renders them, so the Gernaey-2014 weights live here
    only (not duplicated in the report renderer).
    """
    net_heating = max(0.0, heating - 7.0 * methane)
    return [
        ("aeration", aeration, aeration),
        ("pumping", pumping, pumping),
        ("mixing", mixing, mixing),
        ("sludge", sludge_production, 3.0 * sludge_production),
        ("carbon", carbon, 3.0 * carbon),
        ("methane", methane, -6.0 * methane),
        ("heating", heating, None),
        ("net_heating", net_heating, net_heating),
    ]


def operational_cost_index_bsm2(
    aeration: float,
    pumping: float,
    mixing: float,
    sludge_production: float,
    carbon: float,
    methane: float,
    heating: float,
) -> float:
    """Full BSM2 OCI (Gernaey et al. 2014):

    ``OCI = AE + PE + ME + 3·sludge + 3·carbon − 6·methane
            + max(0, heating − 7·methane)``

    Energies in kWh/d; sludge and carbon in kg/d; methane in kg CH₄/d. The
    methane credit and the methane-offset heating term reward biogas recovery.
    Sums the itemized contributions from :func:`bsm2_oci_terms` (the single
    source of the weights).
    """
    terms = bsm2_oci_terms(aeration, pumping, mixing, sludge_production, carbon, methane, heating)
    return float(sum(c for _, _, c in terms if c is not None))
