"""Effluent quality, operational cost, and derived-quantity metrics.

The headline numbers BSM1 reports are:

- **EQI** (Effluent Quality Index) — weighted sum of total suspended
  solids, COD, BOD, TKN, and NO₃-N in the effluent, integrated over
  the simulation window. Lower is better.
- **OCI** (Operational Cost Index) — aeration energy + pumping energy +
  sludge production + mixing energy.

The kernels implement the **updated** BSM1 evaluation (Alex et al. 2008), which
BSM2 (Gernaey et al. 2014) also uses. The original BSM1 evaluation (Copp 2002)
differs in three places: the EQI nitrogen weights (``B_TKN = 20``, ``B_NO = 20``
rather than 30 and 10), the aeration energy (an empirical quadratic in kLa rather
than ``S_sat·V·kLa / 1800``), and the pumping energy (one factor of 0.04 kWh/m³
for every pumped flow rather than 0.004 / 0.008 / 0.05 for the internal recycle,
RAS and wastage). The TSS, COD, BOD₅ and TKN definitions are the same in both.
Compare results against the reference values of the updated evaluation.

References
----------
Copp, J.B. (ed.) (2002). *The COST Simulation Benchmark: Description and
Simulator Manual*. Office for Official Publications of the European
Communities, Luxembourg.

Alex, J., Benedetti, L., Copp, J., Gernaey, K.V., Jeppsson, U., Nopens, I.,
Pons, M.-N., Rieger, L., Rosen, C., Steyer, J.-P., Vanrolleghem, P. &
Winkler, S. (2008). *Benchmark Simulation Model no. 1 (BSM1)*. Technical
report, IWA Task Group on Benchmarking of Control Strategies for WWTPs.

Gernaey, K.V., Jeppsson, U., Vanrolleghem, P.A. & Copp, J.B. (eds.) (2014).
*Benchmarking of Control Strategies for Wastewater Treatment Plants*. IWA
Scientific and Technical Report No. 23. IWA Publishing, London.
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
    cod_content,
)

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


# ASM1 → TSS conversion (Copp 2002): TSS = 0.75 * (X_S + X_I + X_BH + X_BA + X_P).
_TSS_FACTOR = ASM1_TSS_FACTOR
_TSS_SPECIES = ASM1_TSS_SPECIES

# EQI weighting factors (g pollutant / m³)⁻¹ of the updated BSM1 evaluation
# (Alex et al. 2008), also used by BSM2. The original BSM1 evaluation (Copp 2002)
# weights TKN and nitrate at 20 each instead of 30 and 10.
_EQI_WEIGHTS = {
    "TSS": 2.0,
    "COD": 1.0,
    "BOD": 2.0,
    "TKN": 30.0,  # total Kjeldahl nitrogen
    "SNO": 10.0,  # nitrate-nitrogen (the SNO species column)
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
    """Total COD: every COD-bearing species weighted by its COD content.

    The species and their content come from the model (:func:`cod_content`:
    its composition, else its declared units), on the lab-COD convention, so
    for ASM1 this is ``SI + SS + XI + XS + XB_H + XB_A + XP`` and for ASM2d
    ``SF + SA + SI + XI + XS + XH + XPAO + XPHA + XAUT``. ``C`` may be a
    concentration array (with ``model``) or a ``StreamSeries``.

    Raises
    ------
    ValueError
        If the model does not say which of its species carry COD.
    """
    C, model = _conc_model(C, model)
    cod = cod_content(model)
    w = jnp.asarray(list(cod.values()))
    return jnp.sum(C[..., _species_idx(model, cod)] * w, axis=-1)


def _biodegradable_weights(model: CompiledModel, params=None) -> list[tuple[float, list[int]]]:
    """The biodegradable COD of each species, grouped by weight.

    A COD-bearing species that no reaction consumes is inert (``SI``, ``XI``,
    ``XP``). Every other one is biodegradable, less the share of its COD that
    the reactions consuming it turn into inert COD -- the inert fraction a
    decaying biomass leaves (``f_P`` in ASM1, ``fXI`` in ASM2d and ASM3) or
    that hydrolysis releases (``fSI``) -- read from the stoichiometry at
    ``params``. This is the generalization of the Copp (2002) BOD weights
    ``SS + XS + (1 - f_P)(XB_H + XB_A)``.

    Returns
    -------
    list of (float, list of int)
        ``(weight, species indices)`` per distinct weight, in model
        order of first appearance, so ASM1 sums ``SS + XS`` and
        ``XB_H + XB_A`` before weighting, as the Copp expression does.

    Raises
    ------
    ValueError
        If the reactions consuming a species leave different inert fractions,
        so its biodegradable share is not one number.
    """
    import numpy as np

    cod = cod_content(model, params)
    S = np.asarray(model.compute_stoich(model.default_parameters() if params is None else params))
    col = model.species_index
    inert = [s for s in cod if not np.any(S[:, col[s]] < 0.0)]
    degradable = [s for s in cod if s not in inert]
    groups: dict[float, list[int]] = {}
    for s in degradable:
        ratios = []
        for r in np.flatnonzero(S[:, col[s]] < 0.0):
            consumed = sum(-cod[d] * float(S[r, col[d]]) for d in degradable if S[r, col[d]] < 0.0)
            produced = sum(cod[i] * float(S[r, col[i]]) for i in inert)
            ratios.append(produced / consumed)
        if max(ratios) - min(ratios) > 1e-9:
            raise ValueError(
                f"model '{model.name}': the reactions consuming {s} leave different inert "
                "fractions, so its BOD cannot be determined; use substrate=\"COD\" "
                "(derived_COD) instead."
            )
        groups.setdefault((1.0 - ratios[0]) * cod[s], []).append(col[s])
    return list(groups.items())


def derived_BOD(
    C, model: CompiledModel | None = None, *, f_P: float | None = None, params=None
) -> jnp.ndarray:
    """BOD₅ proxy: 0.25 × the biodegradable COD (Copp 2002).

    The biodegradable COD is every COD-bearing species the model's reactions
    consume, less the inert COD those reactions leave, read from the model's
    stoichiometry. For ASM1 it is the Copp (2002) expression
    ``0.25 × (SS + XS + (1 - f_P) × (XB_H + XB_A))``; for ASM2d,
    ``0.25 × (SF + SA + XPHA + (1 - fSI) XS + (1 - fXI)(XH + XPAO + XAUT))``.

    Parameters
    ----------
    C : jnp.ndarray or StreamSeries
        Concentrations, shape ``(..., n_species)``, or a stream (its ``C`` and
        model are used).
    model : CompiledModel, optional
        The model of ``C``; taken from the stream when one is passed.
    f_P : float, optional
        Overrides the model's ``f_P`` parameter (the inert fraction of decayed
        ASM1 biomass).
    params : jnp.ndarray, optional
        Model parameter vector whose stoichiometry gives the inert fractions
        (the composition the simulation used); the model defaults if omitted.

    Returns
    -------
    jnp.ndarray
        BOD₅, scalar for a 1-D ``C``, a leading-axis vector for a 2-D one.

    Raises
    ------
    ValueError
        If ``f_P`` is given for a model without an ``f_P`` parameter, if the
        model does not say which species carry COD, or if its stoichiometry
        does not determine a species' biodegradable share (use
        :func:`derived_COD` then).
    """
    C, model = _conc_model(C, model)
    if f_P is not None:
        if "f_P" not in model.param_index:
            raise ValueError(
                f"model '{model.name}' has no f_P parameter; its inert fractions come from "
                "its stoichiometry, so pass params= instead of f_P=."
            )
        base = model.default_parameters() if params is None else jnp.asarray(params)
        params = base.at[model.param_index["f_P"]].set(f_P)
    # Sum each weight's species before weighting them, adding in model order,
    # so ASM1 evaluates exactly the Copp expression.
    total = jnp.zeros(jnp.shape(C)[:-1])
    for k, (weight, cols) in enumerate(_biodegradable_weights(model, params)):
        group = C[..., cols[0]]
        for c in cols[1:]:
            group = group + C[..., c]
        term = group if weight == 1.0 else weight * group
        total = term if k == 0 else total + term
    return 0.25 * total


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
    i_XB, i_XP, _ = _composition(model, params)
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
        "BOD": time_avg(derived_BOD(C_traj, model, params=params)),
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
    """EQI with the updated BSM1 / BSM2 weights (Alex et al. 2008).

    ``EQI = (1 / T) × ∫ Q × (B_TSS×TSS + B_COD×COD + B_BOD×BOD + B_TKN×TKN + B_NO×SNO) dt × 1e-3``

    with ``B_TSS = 2``, ``B_COD = 1``, ``B_BOD = 2``, ``B_TKN = 30`` and
    ``B_NO = 10``. These are the weights of the updated BSM1 evaluation, also
    used by BSM2. The original BSM1 evaluation (Copp 2002) uses ``B_TKN = 20``
    and ``B_NO = 20``, so its EQI differs from this one for the same effluent.

    Units: kg pollutant / day, averaged over the simulation window. Accepts either
    a :class:`~aquakin.plant.streams.StreamSeries` -- ``effluent_quality_index(eff)``
    -- or the explicit ``(t, C_traj, Q_traj, model)`` form.
    """
    t, C_traj, Q_traj, model = _effluent_args(stream_or_t, C_traj, Q_traj, model)
    i_XB, i_XP, _ = _composition(model, params)
    TSS_t = derived_TSS(C_traj, model)
    COD_t = derived_COD(C_traj, model)
    BOD_t = derived_BOD(C_traj, model, params=params)
    TKN_t = derived_TKN(C_traj, model, i_XB=i_XB, i_XP=i_XP)
    SNO_t = C_traj[:, model.species_index["SNO"]]

    integrand = Q_traj * (
        _EQI_WEIGHTS["TSS"] * TSS_t
        + _EQI_WEIGHTS["COD"] * COD_t
        + _EQI_WEIGHTS["BOD"] * BOD_t
        + _EQI_WEIGHTS["TKN"] * TKN_t
        + _EQI_WEIGHTS["SNO"] * SNO_t
    )
    return float(time_average(integrand, t) * 1e-3)


def aeration_energy(
    t: jnp.ndarray,
    kla_history: jnp.ndarray,
    volumes: jnp.ndarray,
    saturation: float = 8.0,
) -> float:
    """Aeration energy (kWh/d) of the updated BSM1 / BSM2 evaluation.

    AE = (S_sat / (T × 1.8 × 1000)) × ∫ Σ_i V_i × kLa_i(t) dt

    This is the aeration energy of the updated BSM1 evaluation (Alex et al.
    2008), also used by BSM2. The original BSM1 evaluation (Copp 2002) used an
    empirical quadratic in kLa instead.

    Parameters
    ----------
    t : (n_t,) save times in days
    kla_history : (n_t, n_aerated_tanks) kLa value at each save time: the
        delivered process kLa (a tank's fixed ``kla`` or its controller output
        times its gain, times ``alpha`` and, with the temperature correction
        on, ``kla_theta**(T - ref_T)``;
        :meth:`~aquakin.plant.cstr.AerationUnit.delivered_oxygen_kla`). The
        1.8 kg O₂/kWh is an empirical field efficiency, so it applies to the
        transfer actually delivered and a tank's ``alpha`` is not counted
        twice. The clean-water blower formulas of
        :mod:`aquakin.plant.aeration_system` take the clean-water kLa instead.
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
    """Pumping energy (kWh/d) of the updated BSM1 evaluation.

    PE = (1 / T) × ∫ (0.004 × Q_internal + 0.008 × Q_ras + 0.05 × Q_was) dt

    The per-stream factors (kWh/m³) are those of the updated BSM1 evaluation
    (Alex et al. 2008), taken from BSM2. The original BSM1 evaluation (Copp
    2002) used 0.04 kWh/m³ for every pumped flow.

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

    The updated BSM1 index (Alex et al. 2008) with the mechanical-mixing energy
    of the unaerated reactors. Its full form also adds ``3 × external carbon``,
    which is zero for the open-loop BSM1 (no carbon dosing) and is not taken
    here. The original-evaluation OCI has the same structure but computes the
    aeration and pumping energies differently (see :func:`aeration_energy`,
    :func:`pumping_energy`), so pass those terms in the convention you want.
    ``sludge_production`` is the time-averaged TSS mass flow leaving via
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
    ``kLa < kla_threshold``. The kLa is the clean-water kLa the blowers deliver
    (before ``alpha`` and the temperature factor), since it is the air flow,
    not the oxygen it transfers, that mixes the tank.

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

    ``OCI = AE + PE + ME + 3·sludge + 3·carbon − 6·methane + max(0, heating − 7·methane)``

    Energies in kWh/d; sludge and carbon in kg/d; methane in kg CH₄/d. The
    methane credit and the methane-offset heating term reward biogas recovery.
    Sums the itemized contributions from :func:`bsm2_oci_terms` (the single
    source of the weights).
    """
    terms = bsm2_oci_terms(aeration, pumping, mixing, sludge_production, carbon, methane, heating)
    return float(sum(c for _, _, c in terms if c is not None))
