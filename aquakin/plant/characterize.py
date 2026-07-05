"""Influent characterization: lab / SCADA measurements -> ASM1 state vector.

A municipal influent is measured as aggregates -- total COD, TKN, ammonia,
alkalinity, and (if available) filtered / flocculated COD -- not as the 13
ASM1 state variables an `InfluentSeries` needs. :func:`characterize_influent`
and the lower-level :func:`fractionate` split those aggregates into ASM1 states,
following the SUMO Sumo1 raw-influent fractionation reduced to ASM1.

The COD is split first by filtration, then by biodegradability:

- **soluble** ``SCOD`` (flocculated filtered), **colloidal** ``CCOD`` (filtered
  minus flocculated), **particulate** (total minus filtered);
- soluble -> unbiodegradable ``SU`` + biodegradable ``SB`` (incl. VFA); colloidal
  -> ``CU`` + ``CB``; particulate -> unbiodegradable ``XU``, heterotrophs
  ``XOHO``, endogenous products ``XE``, and biodegradable ``XB`` (the remainder).

Reduced to ASM1 (colloidal behaves as slowly-hydrolysed particulate):

    SI   = SU                 # soluble unbiodegradable
    SS   = SB                 # soluble biodegradable (incl. VFA)
    XI   = CU + XU            # colloidal + particulate unbiodegradable
    XS   = CB + XB            # colloidal + particulate biodegradable
    XB_H = XOHO               # ordinary heterotrophs
    XP   = XE                 # endogenous products
    XB_A = 0                  # autotrophs ~ 0 in raw influent

A measured ``filtered_cod`` / ``flocculated_filtered_cod`` /
``soluble_inert_cod`` drives the corresponding split; absent, the SUMO default
fraction is used. Nitrogen: ``SNH`` from ammonia (or ``f_snh*TKN``); ``SND`` from
the soluble-biodegradable N content; ``XND`` as the TKN-balance remainder using
ASM1's own ``i_XB`` / ``i_XP``. Alkalinity (mg CaCO3/L) converts to ASM1's
``SALK`` (mol charge / m3) by dividing by 50. The fraction defaults are the SUMO
Sumo1 raw-influent tool's.

All splits are plain arithmetic, so :func:`fractionate` works element-wise on
scalars *or* arrays -- the per-row path :func:`read_influent_csv` uses to map an
aggregate-measurement CSV (a COD / TKN time series) to ASM1 states on load.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# The ASM1 state names this module produces (the 13 ASM1 species).
ASM1_STATES = (
    "SI",
    "SS",
    "XI",
    "XS",
    "XB_H",
    "XB_A",
    "XP",
    "SO",
    "SNO",
    "SNH",
    "SND",
    "XND",
    "SALK",
)


@dataclass(frozen=True)
class InfluentFractions:
    """Fractionation parameters for :func:`characterize_influent` / :func:`fractionate`.

    Defaults are the SUMO Sumo1 raw-influent tool's municipal values; the
    nitrogen-balance contents ``iN_xb`` / ``iN_xp`` are ASM1's ``i_XB`` / ``i_XP``.

    Parameters
    ----------
    f_sccod : float
        Filtered COD (1.5 um, incl. colloids) as a fraction of total COD. Used
        only when ``filtered_cod`` is not measured.
    f_scod : float
        Flocculated-filtered (truly soluble) COD as a fraction of total COD.
        Used only when ``flocculated_filtered_cod`` is not measured.
    f_su : float
        Soluble unbiodegradable COD as a fraction of filtered COD. Used only
        when ``soluble_inert_cod`` is not measured.
    f_xu : float
        Particulate unbiodegradable COD as a fraction of total COD.
    f_oho : float
        Heterotroph biomass COD as a fraction of total COD.
    f_xe : float
        Endogenous-product COD as a fraction of the heterotroph COD.
    f_cu : float
        Colloidal unbiodegradable COD as a fraction of colloidal COD.
    f_snh : float
        Ammonia as a fraction of TKN. Used only when ``ammonia`` is not measured.
    iN_sb : float
        Nitrogen content of soluble biodegradable substrate (g N / g COD), used
        to set ``SND``.
    iN_xb, iN_xp : float
        Nitrogen content of biomass / endogenous products (g N / g COD), ASM1's
        ``i_XB`` / ``i_XP``; used to close the TKN balance onto ``XND``.
    caco3_eq : float
        g CaCO3 per equivalent: ``SALK [mol/m3] = alkalinity [mg CaCO3/L] / caco3_eq``.
    """

    f_sccod: float = 0.405
    f_scod: float = 0.202
    f_su: float = 0.118
    f_xu: float = 0.14
    f_oho: float = 0.05
    f_xe: float = 0.20
    f_cu: float = 0.20
    f_snh: float = 0.70
    iN_sb: float = 0.04
    iN_xb: float = 0.086
    iN_xp: float = 0.06
    caco3_eq: float = 50.0
    default_alkalinity_mol: float = 7.0  # SALK when alkalinity is not supplied


def fractionate(
    *,
    total_cod,
    tkn,
    ammonia=None,
    nox=0.0,
    alkalinity=None,
    filtered_cod=None,
    flocculated_filtered_cod=None,
    soluble_inert_cod=None,
    fractions: InfluentFractions = InfluentFractions(),
) -> dict:
    """Split influent aggregate measurements into ASM1 state values.

    Every argument may be a scalar or a same-shaped array (the array path is how
    a measurement time series is fractionated per row). Returns a mapping of ASM1
    state name -> value (same scalar/array shape), suitable for
    ``model.influent(...)`` or assembling an :class:`InfluentSeries`. See the
    module docstring for the scheme.

    Parameters
    ----------
    total_cod, tkn : float or array
        Total COD (g COD/m3) and total Kjeldahl nitrogen (g N/m3). Required.
    ammonia, nox, alkalinity : float or array, optional
        Ammonia (g N/m3; else ``f_snh*tkn``), nitrate+nitrite (g N/m3; default 0
        -> ``SNO``), and alkalinity (mg CaCO3/L -> ``SALK``; else a default).
    filtered_cod, flocculated_filtered_cod, soluble_inert_cod : float or array, optional
        Measured COD sub-fractions (g COD/m3). When given they drive the split;
        when absent the corresponding default fraction is used.
    fractions : InfluentFractions
        The fraction parameters (SUMO Sumo1 defaults).

    Raises
    ------
    ValueError
        If ``ammonia`` exceeds ``tkn`` (TKN includes ammonia).

    Warns
    -----
    UserWarning
        If the COD fractionation does not close -- a COD fraction clamped
        negative (unusual ``filtered_cod`` / ``flocculated_filtered_cod``), so
        the ASM1 COD states sum to more than ``total_cod``.
    """
    f = fractions

    # TKN includes ammonia, so ammonia > TKN is an inconsistent measurement that
    # would force the organic-N pools negative; reject it rather than clamp.
    if ammonia is not None and np.any(np.asarray(ammonia) > np.asarray(tkn)):
        raise ValueError("ammonia exceeds tkn (TKN includes ammonia); check the measurements.")

    sccod = filtered_cod if filtered_cod is not None else f.f_sccod * total_cod
    scod = (
        flocculated_filtered_cod if flocculated_filtered_cod is not None else f.f_scod * total_cod
    )
    ccod = sccod - scod  # colloidal COD
    pcod = total_cod - sccod  # particulate COD

    su = soluble_inert_cod if soluble_inert_cod is not None else f.f_su * sccod
    sb = scod - su  # soluble biodegradable (incl. VFA)
    cu = f.f_cu * ccod
    cb = ccod - cu  # colloidal biodegradable
    xu = f.f_xu * total_cod
    oho = f.f_oho * total_cod
    xe = f.f_xe * oho
    xb = pcod - xu - oho - xe  # particulate biodegradable

    SI = np.maximum(su, 0.0)
    SS = np.maximum(sb, 0.0)
    XI = np.maximum(cu + xu, 0.0)
    XS = np.maximum(cb + xb, 0.0)
    XB_H = np.maximum(oho, 0.0)
    XP = np.maximum(xe, 0.0)

    # The six COD states partition total_cod exactly (sum == total_cod) when no
    # fraction is negative. A negative fraction clamped to 0 (unusual filtered /
    # flocculated splits) ADDS COD, so the partition no longer closes -- warn
    # rather than silently return a non-conserving influent.
    cod_sum = SI + SS + XI + XS + XB_H + XP
    if np.any(np.asarray(cod_sum) > np.asarray(total_cod) * (1.0 + 1e-6) + 1e-9):
        warnings.warn(
            "Influent COD fractionation does not close: a COD fraction clamped "
            "negative (check filtered_cod / flocculated_filtered_cod), so the "
            "ASM1 COD states sum to more than total_cod.",
            stacklevel=2,
        )

    SNH = ammonia if ammonia is not None else f.f_snh * tkn
    SNO = nox
    SND = np.maximum(f.iN_sb * SS, 0.0)  # soluble biodegradable organic N
    # particulate biodegradable organic N closes the TKN balance: TKN excludes
    # nitrate, and ASM1 carries biomass/product N via i_XB / i_XP.
    XND = np.maximum(tkn - SNH - SND - f.iN_xb * XB_H - f.iN_xp * XP, 0.0)

    SALK = (
        alkalinity / f.caco3_eq
        if alkalinity is not None
        else _broadcast_like(f.default_alkalinity_mol, total_cod)
    )

    zero = _broadcast_like(0.0, total_cod)
    return {
        "SI": SI,
        "SS": SS,
        "XI": XI,
        "XS": XS,
        "XB_H": XB_H,
        "XB_A": zero,
        "XP": XP,
        "SO": zero,
        "SNO": _broadcast_like(SNO, total_cod),
        "SNH": SNH,
        "SND": SND,
        "XND": XND,
        "SALK": SALK,
    }


def _broadcast_like(value, ref):
    """Return ``value`` shaped like ``ref`` (a scalar stays a float; an array
    reference makes a constant column), so every state in the returned mapping
    has matching shape."""
    if np.ndim(ref) == 0:
        return float(value) if np.ndim(value) == 0 else value
    return np.full(np.shape(ref), value) if np.ndim(value) == 0 else value


def characterize_influent(
    model,
    *,
    flow,
    total_cod,
    tkn,
    ammonia=None,
    nox=0.0,
    alkalinity=None,
    filtered_cod=None,
    flocculated_filtered_cod=None,
    soluble_inert_cod=None,
    fractions: InfluentFractions = InfluentFractions(),
    T=None,
):
    """Build a constant :class:`InfluentSeries` from influent measurements.

    Fractionates the measured aggregates into ASM1 states (see :func:`fractionate`
    and the module docstring) and returns a constant-in-time influent at flow
    ``flow``. The model must declare the ASM1 states.

    Parameters
    ----------
    model : CompiledModel
        An ASM1 (or ASM1-state-compatible) model.
    flow : float
        Volumetric flow (m3/d).
    total_cod, tkn, ammonia, nox, alkalinity, filtered_cod,
    flocculated_filtered_cod, soluble_inert_cod, fractions :
        Passed to :func:`fractionate`.
    T : float, optional
        Influent temperature (Kelvin), carried onto the series.

    Returns
    -------
    InfluentSeries

    Examples
    --------
    >>> net = aquakin.load_model("asm1")
    >>> inf = characterize_influent(net, flow=24000.0, total_cod=420.0,
    ...                             tkn=34.4, ammonia=24.0, alkalinity=330.0)
    """
    _require_asm1_states(model)
    states = fractionate(
        total_cod=total_cod,
        tkn=tkn,
        ammonia=ammonia,
        nox=nox,
        alkalinity=alkalinity,
        filtered_cod=filtered_cod,
        flocculated_filtered_cod=flocculated_filtered_cod,
        soluble_inert_cod=soluble_inert_cod,
        fractions=fractions,
    )
    return model.influent({k: float(v) for k, v in states.items()}, Q=float(flow), T=T, base="zero")


def _require_asm1_states(model) -> None:
    missing = [s for s in ASM1_STATES if s not in model.species_index]
    if missing:
        raise ValueError(
            f"characterize_influent needs the ASM1 state variables; model "
            f"'{model.name}' is missing {missing}. The fractionation targets "
            f"the ASM1 13-state vector."
        )
