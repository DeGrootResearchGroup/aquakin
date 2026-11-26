"""Continuous-flow stirred-tank reactor (CSTR) with kinetics + aeration.

The CSTR is the workhorse unit for activated-sludge plants: each ASM tank
in BSM1 is a CSTR. The mass-balance equation for species i is::

    dC_i/dt = (Q_in / V) * (C_in,i - C_i)        # convection
              + S^T r(C, p, conditions)[i]       # chemistry
              + kLa_i * (C_sat,i - C_i)          # mass transfer (DO in aerobic)

Per-species ``kLa`` and ``C_sat`` make it trivial to model both anoxic
tanks (kLa=0 everywhere) and aerobic tanks (kLa_DO > 0) with the same
unit class. Dissolved gases stripped by the same air (N2O, NO, ...) are
further species with ``kLa_gas = ratio * kLa_O2`` and a saturation of ~0
(see ``Aeration(stripped=...)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquakin.core.gases import (
    AIR_MOLE_FRACTIONS,
    ATMOSPHERE_PA,
    saturation,
    temperature_and_pH,
    volatile_concentration,
)
from aquakin.core.temperature import T_REF_20C
from aquakin.plant._constants import check_positive
from aquakin.plant.coupling import CouplingAware
from aquakin.plant.kinetic import AdjustableTemperatureUnit
from aquakin.plant.streams import Stream, mixed_scalars

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.model import CompiledModel


def oxygen_saturation(T_K):
    """Clean-water dissolved-oxygen saturation (mg/L) at 1 atm, zero salinity.

    The Benson--Krause / APHA Standard Methods correlation: ``ln(C_s)`` as a
    quartic in ``1/T`` with ``T`` in Kelvin. Gives ~9.09 mg/L at 20 degC and
    ~7.56 at 30 degC -- the ~15-20% swing across the activated-sludge operating
    band. Used only as a **ratio** ``C_s(T)/C_s(T_ref)`` to temperature-correct a
    user-supplied saturation, so the absolute calibration (salinity, pressure)
    cancels; those enter separately as the ``beta`` / ``pressure_factor``
    multipliers on :class:`Aeration`. Pure ``jnp`` so it is jit/AD-clean.

    Parameters
    ----------
    T_K : float or jnp.ndarray
        Temperature in Kelvin.

    Returns
    -------
    jnp.ndarray
        Saturation dissolved-oxygen concentration in mg/L (== g/m3).
    """
    T = jnp.asarray(T_K, dtype=float)
    ln_cs = (
        -139.34411 + 1.575701e5 / T - 6.642308e7 / T**2 + 1.243800e10 / T**3 - 8.621949e11 / T**4
    )
    return jnp.exp(ln_cs)


def oxygen_saturation_bsm2(T_K):
    """Dissolved-oxygen saturation (mg/L) by the IWA benchmark (BSM1/BSM2) van't
    Hoff correlation, normalised to 8.0 mg/L at 15 degC.

    This is the exact saturation the activated-sludge benchmark uses to
    temperature-correct the oxygen driving force, ``SO_sat(T) = 0.9997743214 *
    8/10.5 * 56.12 * 6791.5 * exp(-66.7354 + 87.4755/theta + 24.4526*ln(theta))``
    with ``theta = T_K/100``. It gives 8.0 at 15 degC, ~9.0 at 9.5 degC and ~7.2
    at 20.5 degC. Use it (rather than :func:`oxygen_saturation`, the Benson--Krause
    correlation, whose temperature *shape* differs by ~0.5 %) when matching the
    benchmark exactly. Pure ``jnp`` so it is jit/AD-clean.

    Parameters
    ----------
    T_K : float or jnp.ndarray
        Temperature in Kelvin.

    Returns
    -------
    jnp.ndarray
        Saturation dissolved-oxygen concentration in mg/L (== g/m3).
    """
    theta = jnp.asarray(T_K, dtype=float) / 100.0
    return (
        0.9997743214
        * 8.0
        / 10.5
        * (56.12 * 6791.5 * jnp.exp(-66.7354 + 87.4755 / theta + 24.4526 * jnp.log(theta)))
    )


# ``Aeration(stripped=...)`` value that strips every gas the model declares.
STRIP_MODEL_GASES = "model"

# Saturation-model registry: name -> C_s(T) callable used for the temperature
# correction ratio C_s(T)/C_s(ref_T).
_SATURATION_MODELS = {
    "benson_krause": oxygen_saturation,
    "bsm2": oxygen_saturation_bsm2,
}


#: The modes of an :class:`Aeration`, exactly one of which is given.
_AERATION_MODES = ("kla", "do_setpoint", "kla_signal")


@dataclass(frozen=True)
class Aeration:
    """Aeration / dissolved-oxygen spec for a :class:`CSTRUnit`.

    The quantity a designer actually thinks in, instead of a raw mass-transfer
    coefficient on a state variable. Choose exactly one mode.

    **Open loop** -- a fixed mass-transfer coefficient (the saturation defaults
    sensibly, so the common case is one number)::

        Aeration(kla=120)                 # do_sat defaults to 8.0 gO2/m3
        Aeration(kla=120, do_sat=9.0)

    **Closed loop** -- a dissolved-oxygen setpoint. The plant auto-wires a PI
    controller that manipulates this tank's kLa to hold the setpoint::

        Aeration(do_setpoint=2.0)         # this tank holds DO = 2.0 on its own

    Several tanks share one controller by giving the same ``controller`` id -- the
    BSM2 design, one sensor driving several reactors at per-tank gains::

        Aeration(do_setpoint=2.0, controller="do", sensor="tank4", gain=1.0)
        Aeration(do_setpoint=2.0, controller="do", sensor="tank4", gain=0.5)

    **From a signal** -- the kLa is a control signal some other unit publishes
    (an :class:`~aquakin.plant.control.ExpressionController`, or any signal
    producer); nothing is auto-wired::

        Aeration(kla_signal="kla_tank3")  # kLa = signal * gain

    Parameters
    ----------
    kla : float, optional
        Open-loop mass-transfer coefficient (1/time). Give exactly one of
        ``kla``, ``do_setpoint`` and ``kla_signal``.
    do_setpoint : float, optional
        Closed-loop dissolved-oxygen target (same units as ``species``).
    kla_signal : str, optional
        Name of the control signal that sets this tank's kLa (1/time, times
        ``gain``). A unit of the plant must publish it. It is treated as a
        closed-loop kLa: neither ``alpha`` nor the theta correction scales it,
        while the saturation correction and co-stripped gases apply as usual.
    do_sat : float
        Saturation concentration in the aeration term ``kLa*(do_sat - C)``.
        Default 8.0 (gO2/m3, the usual ASM oxygen saturation).
    species : str
        The aerated species (default ``"SO"``).
    controller : str, optional
        Closed-loop only. Shared-controller id: tanks giving the same id share one
        PI controller (and the controller unit takes this name). ``None`` (the
        default) gives this tank its own dedicated controller.
    sensor : str, optional
        Closed-loop only. Name of the unit whose oxygen the controller measures.
        Defaults to the controlled tank itself (per-tank control).
    gain : float
        Closed loop or ``kla_signal``. This tank's share of the controller's kLa
        output (default 1.0), so one shared controller can drive tanks at
        different rates.
    Kp, Ti, Tt, kla_offset, kla_min, kla_max : float
        Closed-loop PI tuning and output bounds. Defaults are the BSM2 DO loop:
        Kp=25, Ti=0.002 d, Tt=0.001 d, offset 120, bounds [0, 360] d^-1.

    temperature_correction : bool
        First of the oxygen-transfer corrections -- all **off / identity by
        default**, so a default ``Aeration`` is bit-faithful to the IWA benchmark
        constant-saturation, constant-kLa definition. When ``True``,
        temperature-correct the oxygen driving force using the
        tank's operating temperature (the flow-weighted inlet ``T`` the kinetics
        already use, falling back to the static ``T`` condition). The saturation
        is scaled by the clean-water ratio ``C_s(T)/C_s(ref_T)``
        (:func:`oxygen_saturation`) and -- for an open-loop fixed ``kla`` -- the
        transfer coefficient by ``kla_theta**(T - ref_T)``. ``False`` (default)
        leaves both constant. Removes the internal inconsistency whereby a warm
        run already speeds the (Arrhenius) biology while the oxygen saturation
        stays pinned. A closed-loop controlled ``kla`` is **not** theta-scaled
        (the controller already manipulates it to hold the setpoint), but its
        driving-force saturation still gets the ``C_s(T)`` correction.
    ref_T : float
        Reference temperature (Kelvin) at which ``do_sat`` and ``kla`` are
        specified, so the correction is unity there. Default 293.15 (20 degC).
    alpha : float
        Process kLa-transfer factor (alpha_F): a constant multiplier on ``kla``
        (clean-water -> process water; typically 0.4-0.7 in activated sludge).
        Default 1.0. Applied to the open-loop ``kla`` only.
    beta : float
        Salinity saturation factor: a constant multiplier on ``do_sat``
        (typically 0.95-0.99). Default 1.0.
    pressure_factor : float
        Elevation / barometric saturation factor: a constant multiplier on
        ``do_sat`` (``< 1`` at altitude). Default 1.0.
    kla_theta : float
        Arrhenius base for the open-loop ``kla(T) = kla*kla_theta**(T-ref_T)``
        correction (only used when ``temperature_correction`` is on). Default
        1.024, the standard value.
    saturation_model : str
        Which clean-water saturation correlation supplies the temperature-
        correction ratio ``C_s(T)/C_s(ref_T)``: ``"benson_krause"`` (default, the
        APHA correlation) or ``"bsm2"`` (the IWA benchmark van't Hoff formula,
        :func:`oxygen_saturation_bsm2`). Use ``"bsm2"`` to match the benchmark's
        oxygen driving force exactly; the two differ by ~0.5 % in shape. Only used
        when ``temperature_correction`` is on.
    stripped : dict, optional
        Dissolved gases stripped by the same aeration, alongside oxygen. Maps a
        species name to its mass-transfer ratio ``kLa_gas / kLa_O2`` (a float,
        saturation 0) or to a ``(ratio, saturation)`` pair. Each gas gets the
        term ``ratio * kLa_O2 * (sat - C)`` in the tank's mass balance, so it
        leaves the liquid inside the solve rather than through the effluent.
        ``kLa_O2`` is the tank's actual oxygen transfer coefficient: the
        open-loop ``kla`` (with ``alpha`` and, when enabled, the temperature
        factor), or under closed loop the controller signal times ``gain``. By
        penetration theory the ratio is the square root of the diffusivity
        ratio, close to 1 for N2O and NO (``{"SN2O": 1.0, "SNO": 1.0}``). The
        gas saturation is not temperature-corrected. ``"model"`` instead strips
        every dissolved gas the model declares in its ``gases:`` block (other
        than the aerated species): each at its ``kL_ratio``, toward the
        saturation in equilibrium with its share of ``air`` at the tank's
        temperature, ``H(T) * y * 101325 Pa * pressure_factor``; inorganic
        carbon transfers as its H2CO3* fraction at the tank's pH. Change a gas's
        solubility or transfer ratio in the model's ``gases:`` block. Default
        ``None`` (only oxygen is transferred). A tank with no blowers but some
        surface exchange is ``Aeration(kla=<surface kLa>, stripped="model")``.
    air : dict, optional
        With ``stripped="model"``: mole fractions of the gas the liquid meets,
        ``{gas: fraction}`` for library gases (``"N2"``, ``"CO2"``, ...),
        overriding those of dry air (:data:`aquakin.core.gases.AIR_MOLE_FRACTIONS`)
        for the gases given, e.g. ``{"CO2": 0.02}`` for a covered tank whose
        headspace holds CO2. Default ``None`` (dry air).
    """

    kla: float | None = None
    do_setpoint: float | None = None
    kla_signal: str | None = field(default=None, metadata={"written_when_set": True})
    do_sat: float = 8.0
    species: str = "SO"
    controller: str | None = None
    sensor: str | None = None
    gain: float = 1.0
    Kp: float = 25.0
    Ti: float = 0.002
    Tt: float = 0.001
    kla_offset: float = 120.0
    kla_min: float = 0.0
    kla_max: float = 360.0
    temperature_correction: bool = False
    ref_T: float = T_REF_20C
    alpha: float = 1.0
    beta: float = 1.0
    pressure_factor: float = 1.0
    kla_theta: float = 1.024
    saturation_model: str = "benson_krause"
    stripped: dict | tuple | str | None = None
    air: dict | tuple | None = field(default=None, metadata={"written_when_set": True})

    def __post_init__(self) -> None:
        # Normalize ``stripped`` to a hashable tuple of (species, ratio, sat)
        # triples (the dataclass is frozen and hashable).
        object.__setattr__(self, "stripped", _normalize_stripped(self.stripped, self.species))
        object.__setattr__(self, "air", _normalize_air(self.air, self.stripped))
        given = [m for m in _AERATION_MODES if getattr(self, m) is not None]
        if not given:
            raise ValueError(
                "Aeration requires exactly one of kla= (open loop), do_setpoint= "
                "(closed loop) or kla_signal= (kLa from a control signal); none was given."
            )
        if len(given) > 1:
            raise ValueError(
                f"Aeration requires exactly one of kla=, do_setpoint= or kla_signal=; "
                f"got {' and '.join(given)}."
            )
        if self.saturation_model not in _SATURATION_MODELS:
            raise ValueError(
                f"Unknown saturation_model '{self.saturation_model}'; choose "
                f"from {sorted(_SATURATION_MODELS)}."
            )
        if self.kla is not None and self.kla < 0.0:
            raise ValueError(f"Aeration kla must be >= 0, got {self.kla}.")
        for name in ("alpha", "beta", "pressure_factor"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"Aeration {name} must be >= 0, got {getattr(self, name)}.")
        if self.kla_theta <= 0.0:
            raise ValueError(f"Aeration kla_theta must be > 0, got {self.kla_theta}.")

    @property
    def is_closed_loop(self) -> bool:
        return self.do_setpoint is not None

    def controller_id(self, unit_name: str) -> str:
        """The id that groups tanks onto one controller: the shared ``controller``
        if given, else the tank's own name (a dedicated per-tank controller)."""
        return self.controller if self.controller is not None else unit_name

    @property
    def reads_signal(self) -> bool:
        """Whether the kLa is read from the signal bus (closed loop, or ``kla_signal``)."""
        return self.kla is None

    def signal_name(self, unit_name: str) -> str:
        """The control-signal name the tank reads: ``kla_signal`` when given,
        otherwise the one its auto-wired controller publishes, derived from the
        controller id so a shared controller's tanks all resolve to the same
        signal."""
        if self.kla_signal is not None:
            return self.kla_signal
        return _aeration_signal_name(self.controller_id(unit_name))


def _normalize_stripped(stripped, aerated_species: str) -> tuple:
    """``Aeration.stripped`` as a tuple of ``(species, ratio, saturation)``.

    Accepts ``None``, a ``{species: ratio}`` / ``{species: (ratio, sat)}``
    mapping, or an already-normalized tuple of triples.
    """
    if stripped is None:
        return ()
    if isinstance(stripped, str):
        if stripped != STRIP_MODEL_GASES:
            raise ValueError(
                f"Aeration stripped must be {STRIP_MODEL_GASES!r}, a {{species: ratio}} "
                f"mapping or a tuple of (species, ratio, sat) triples; got {stripped!r}."
            )
        return stripped
    if isinstance(stripped, dict):
        items = []
        for sp, val in stripped.items():
            ratio, sat = val if isinstance(val, (tuple, list)) else (val, 0.0)
            items.append((sp, ratio, sat))
    else:
        items = list(stripped)
    out = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError(
                "Aeration stripped must be a {species: ratio} or {species: (ratio, sat)} "
                f"mapping, or a tuple of (species, ratio, sat) triples; got entry {item!r}."
            )
        sp, ratio, sat = item
        sp, ratio, sat = str(sp), float(ratio), float(sat)
        if sp == aerated_species:
            raise ValueError(
                f"Aeration stripped gas '{sp}' is the aerated species itself; list only "
                "the other dissolved gases."
            )
        if ratio < 0.0:
            raise ValueError(f"Aeration stripped ratio for '{sp}' must be >= 0, got {ratio}.")
        if sat < 0.0:
            raise ValueError(f"Aeration stripped saturation for '{sp}' must be >= 0, got {sat}.")
        out.append((sp, ratio, sat))
    names = [sp for sp, _r, _s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Aeration stripped gases repeat a species: {names}.")
    return tuple(out)


def _normalize_air(air, stripped) -> tuple:
    """``Aeration.air`` as a sorted tuple of ``(gas, mole fraction)`` pairs.

    Raises
    ------
    ValueError
        If ``air`` is given without ``stripped="model"``, names a gas outside
        the library, or gives a fraction outside [0, 1].
    """
    if not air:  # None, or the empty list a spec writes for it
        return ()
    if stripped != STRIP_MODEL_GASES:
        raise ValueError(
            f"Aeration air is the gas the model's declared gases equilibrate with; "
            f"it needs stripped={STRIP_MODEL_GASES!r}."
        )
    items = air.items() if isinstance(air, dict) else air
    out = []
    for gas, fraction in items:
        if gas not in AIR_MOLE_FRACTIONS:
            raise ValueError(
                f"Aeration air names unknown gas '{gas}'; the library gases are "
                f"{sorted(AIR_MOLE_FRACTIONS)}."
            )
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"Aeration air fraction of '{gas}' must be in [0, 1], got {fraction}.")
        out.append((str(gas), float(fraction)))
    return tuple(sorted(out))


def _aeration_signal_name(controller_id: str) -> str:
    return f"_aer_{controller_id}_kla"


@dataclass(frozen=True)
class AerationVectors:
    """Precomputed per-species aeration vectors + temperature-correction config.

    Built once from an :class:`Aeration` spec by :func:`build_aeration_vectors`
    and applied each RHS call by :func:`aeration_transfer`. Shared by
    :class:`CSTRUnit` (the suspended tank) and the bulk of
    :class:`~aquakin.plant.ifas.IFASUnit`, so the closed-loop control, the
    constant alpha/beta/pressure folds and the temperature correction live in one
    place.
    """

    kla_vec: jnp.ndarray  # (n_species,) fixed open-loop kLa
    sat_vec: jnp.ndarray  # (n_species,) saturation (beta/P folded)
    controlled: dict  # species -> (signal_name, gain)
    ref_T: float
    kla_theta: float
    temp_correct: bool
    saturation_model: str = "benson_krause"
    # Co-stripped gases: species -> kLa_gas / kLa_O2. A closed-loop gas also
    # appears in ``controlled`` with gain * ratio.
    stripped: dict = field(default_factory=dict)
    # 1 at the aerated (oxygen) species, 0 elsewhere: the C_s(T)/C_s(ref)
    # temperature ratio applies to the oxygen saturation only.
    o2_mask: jnp.ndarray | None = None
    # stripped="model": ((DissolvedGas, partial pressure in Pa), ...), whose
    # saturation and volatile fraction are evaluated at the tank's conditions.
    gases: tuple = ()


def build_aeration_vectors(aeration, model, unit_name: str) -> AerationVectors:
    """Translate an :class:`Aeration` spec into the per-species RHS vectors.

    The constant (temperature-independent) corrections fold straight in: beta
    (salinity) and pressure_factor (elevation) onto the saturation, alpha
    (transfer fouling) onto the open-loop kLa. A closed-loop species records its
    ``(signal_name, gain)`` instead of a fixed kLa. All factors default to 1.0
    / off, so a plain ``Aeration(kla=...)`` or ``aeration=None`` is unchanged.
    """
    controlled: dict[str, tuple[str, float]] = {}
    stripped: dict[str, float] = {}
    kla_vec = jnp.zeros((model.n_species,))
    sat_vec = jnp.zeros((model.n_species,))
    o2_mask = None
    ref_T, kla_theta, temp_correct = T_REF_20C, 1.024, False
    saturation_model = "benson_krause"
    gases: tuple = ()
    if aeration is not None:
        if aeration.species not in model.species_index:
            raise ValueError(
                f"'{unit_name}' aeration species '{aeration.species}' is not in the model."
            )
        idx = model.species_index[aeration.species]
        sat_vec = sat_vec.at[idx].set(
            float(aeration.do_sat) * float(aeration.beta) * float(aeration.pressure_factor)
        )
        o2_mask = jnp.zeros((model.n_species,)).at[idx].set(1.0)
        if aeration.reads_signal:
            controlled[aeration.species] = (aeration.signal_name(unit_name), aeration.gain)
        else:
            kla_vec = kla_vec.at[idx].set(float(aeration.kla) * float(aeration.alpha))
        # Co-stripped gases ride on the oxygen kLa: ratio * (open-loop kla, or
        # the controller signal * gain).
        listed = aeration.stripped
        if listed == STRIP_MODEL_GASES:
            gases = _declared_gases(aeration, model, unit_name)
            listed = tuple((entry.species, entry.kL_ratio, 0.0) for entry, _p in gases)
        for sp, ratio, sat in listed:
            if sp not in model.species_index:
                raise ValueError(f"'{unit_name}' aeration stripped gas '{sp}' is not in the model.")
            g = model.species_index[sp]
            stripped[sp] = ratio
            sat_vec = sat_vec.at[g].set(sat)
            if aeration.reads_signal:
                controlled[sp] = (aeration.signal_name(unit_name), aeration.gain * ratio)
            else:
                kla_vec = kla_vec.at[g].set(float(aeration.kla) * float(aeration.alpha) * ratio)
        ref_T = float(aeration.ref_T)
        kla_theta = float(aeration.kla_theta)
        temp_correct = bool(aeration.temperature_correction)
        saturation_model = aeration.saturation_model
    return AerationVectors(
        kla_vec,
        sat_vec,
        controlled,
        ref_T,
        kla_theta,
        temp_correct,
        saturation_model,
        stripped=stripped,
        o2_mask=o2_mask,
        gases=gases,
    )


def _declared_gases(aeration, model, unit_name: str) -> tuple:
    """``((DissolvedGas, partial pressure in Pa), ...)`` for ``stripped="model"``:
    every gas the model declares except the aerated species, each at its share
    of the air the tank is aerated with."""
    if model.gases is None:
        raise ValueError(
            f"'{unit_name}' aeration has stripped={STRIP_MODEL_GASES!r}, but model "
            f"'{model.name}' declares no dissolved gases; add a 'gases:' block to the "
            "model, or list the stripped gases as {species: ratio}."
        )
    air = {**AIR_MOLE_FRACTIONS, **dict(aeration.air)}
    pressure = ATMOSPHERE_PA * float(aeration.pressure_factor)
    return tuple(
        (entry, air[entry.gas] * pressure)
        for entry in model.gases.entries
        if entry.species != aeration.species
    )


def aeration_transfer(av: AerationVectors, C, T_eff, signals, model, conditions=None, params=None):
    """The per-species mass-transfer term ``kLa * (C_sat - C)`` for state ``C``.

    Applies the temperature correction (saturation by ``C_s(T)/C_s(ref)``, the
    open-loop kLa by ``theta**(T-ref)``) when enabled and a temperature is
    available, then overrides any closed-loop species' kLa with its control
    signal (``signal * gain``). ``T_eff`` is the operating temperature (the
    flow-weighted inlet, else the static condition); pass ``None`` to skip the
    correction. A gas stripped as one the model declares (``stripped="model"``)
    is driven by ``kLa * (H(T) p - C_volatile)`` at the temperature and pH of
    ``conditions`` (the kinetics' conditions) and ``params``, which it then
    requires.
    """
    kla_vec, sat_vec = av.kla_vec, av.sat_vec
    if av.temp_correct and T_eff is not None:
        c_s = _SATURATION_MODELS[av.saturation_model]
        sat_ratio = c_s(T_eff) / c_s(av.ref_T)
        if av.o2_mask is not None and av.stripped:
            # Correct the oxygen saturation only, not a co-stripped gas's.
            sat_vec = sat_vec * (1.0 + av.o2_mask * (sat_ratio - 1.0))
        else:
            sat_vec = sat_vec * sat_ratio
        kla_vec = kla_vec * av.kla_theta ** (T_eff - av.ref_T)
    if av.controlled:
        if signals is None:
            raise ValueError(
                "Closed-loop aeration needs the control-signal bus, but rhs() "
                "was called without one (signals=None). A controlled reactor must "
                "be solved inside its plant, which supplies the signals "
                f"{sorted({sig for sig, _gain in av.controlled.values()})}; there is "
                "no open-loop kLa to fall back on, so running it this way would "
                "silently leave it unaerated."
            )
        kla_vec = _apply_controlled_kla(av, kla_vec, signals, model)
    if av.gases:
        sat_vec, C = _declared_gas_driving(av, sat_vec, C, model, conditions, params)
    return kla_vec * (sat_vec - C)


def _declared_gas_driving(av: AerationVectors, sat_vec, C, model, conditions, params):
    """The saturation vector and the transferable concentrations with each
    declared gas's entries evaluated at the tank's temperature and pH."""
    if conditions is None:
        raise ValueError(
            "Aeration of the model's declared gases needs the tank's conditions "
            "(temperature, pH); pass conditions= and params= to aeration_transfer."
        )
    T, pH = temperature_and_pH(model, C, params, conditions)
    volatile = C
    for entry, p in av.gases:
        sat_vec = sat_vec.at[entry.index].set(saturation(entry, T, p))
        volatile = volatile.at[entry.index].set(
            volatile_concentration(entry, C[entry.index], T, pH)
        )
    return sat_vec, volatile


def _apply_controlled_kla(av: AerationVectors, kla_vec, signals, model):
    """``kla_vec`` with each closed-loop species' entry set to its control
    signal times its gain (oxygen at ``gain``, a co-stripped gas at
    ``gain * ratio``). The one place the controller output becomes a kLa, shared
    by the RHS transfer term and the post-solve readers."""
    for sp, (signal_name, gain) in av.controlled.items():
        kla_vec = kla_vec.at[model.species_index[sp]].set(signals[signal_name] * gain)
    return kla_vec


class AerationUnit:
    """Mixin for a plant unit that aerates through an :class:`Aeration` spec.

    Translates the ``Aeration`` spec into the per-species RHS vectors once and
    exposes the readers the aeration-energy / O2-balance code and the control bus
    use, so a CSTR, an IFAS/MBBR tank and an MBR share one definition instead of
    each re-deriving them. A consumer sets ``self.aeration`` / ``self.model`` /
    ``self.name`` and calls :meth:`_setup_aeration` from its ``__post_init__``.
    """

    def _setup_aeration(self) -> None:
        """Build the per-species aeration vectors (fixed-kLa / saturation, the
        closed-loop signal map, the temperature-correction config) into
        ``self._av``. Call from ``__post_init__`` once ``aeration`` / ``model``
        / ``name`` are set."""
        self._av = build_aeration_vectors(self.aeration, self.model, self.name)

    def aeration_term(self, C, inputs: dict, params, signals: dict | None):
        """The aeration mass-transfer term ``dC/dt`` the unit's ``rhs`` adds,
        at the operating conditions the inputs and signals give.

        Parameters
        ----------
        C : jnp.ndarray
            The aerated liquid's concentrations, shape ``(n_species,)``.
        inputs : dict[str, Stream]
            The unit's input streams.
        params : jnp.ndarray
            The model's parameter vector.
        signals : dict or None
            The control-signal bus.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_species,)``, per the plant's time unit.
        """
        conditions, T_op = self._operating_conditions(inputs, signals)
        return aeration_transfer(self._av, C, T_op, signals, self.model, conditions, params)

    @property
    def required_signals(self) -> tuple[str, ...]:
        """Control-signal names this unit reads from the bus in ``rhs`` (the
        closed-loop aeration signal, if any). The plant validates these are
        published -- by the controller it auto-wires from the ``Aeration`` spec --
        before solving."""
        # A co-stripped gas under closed loop reads the same signal as oxygen,
        # so de-duplicate (order kept).
        return tuple(dict.fromkeys(sig for sig, _gain in self._av.controlled.values()))

    @property
    def stripped_gases(self) -> dict:
        """The gases co-stripped with oxygen, ``{species: kLa_gas / kLa_O2}``
        (empty when the tank strips none)."""
        return dict(self._av.stripped)

    @property
    def saturation_vector(self) -> jnp.ndarray:
        """Per-species saturation of the aeration term, shape ``(n_species,)``:
        the oxygen ``do_sat`` (with ``beta`` and ``pressure_factor`` folded in)
        and each stripped gas's saturation; zero elsewhere. Not temperature
        corrected."""
        return self._av.sat_vec

    def kla_vector(self, signals: dict | None = None) -> jnp.ndarray:
        """Per-species mass-transfer coefficient, shape ``(n_species,)``.

        The open-loop kLa (with ``alpha`` folded in) on the aerated species and
        each stripped gas, with every closed-loop entry replaced by its control
        signal times its gain. The temperature correction is not applied.

        Parameters
        ----------
        signals : dict, optional
            The control-signal bus at the time of interest
            (:meth:`Plant.signals_at`). Needed only when the tank is under
            closed-loop control; ``None`` otherwise.

        Returns
        -------
        jnp.ndarray
            kLa per species (1/time), zero for species aeration does not touch.

        Raises
        ------
        ValueError
            If the tank is under closed-loop control and ``signals`` is ``None``.
        """
        if not self._av.controlled:
            return self._av.kla_vec
        if signals is None:
            raise ValueError(
                f"'{self.name}' is under closed-loop aeration control, so its kLa "
                "is the controller signal; pass the signal bus (plant.signals_at)."
            )
        return _apply_controlled_kla(self._av, self._av.kla_vec, signals, self.model)

    def oxygen_kla(self, signals: dict | None = None):
        """The tank's oxygen kLa: the entry of :meth:`kla_vector` for the
        aerated species named by ``aeration.species`` (``"SO"`` in ASM1,
        ``"SO2"`` in ASM3-based models), or 0 for an unaerated tank.

        Parameters
        ----------
        signals : dict, optional
            The control-signal bus, needed only under closed-loop control (see
            :meth:`kla_vector`).

        Returns
        -------
        jnp.ndarray
            Scalar oxygen kLa (1/time).
        """
        if self.aeration is None:
            return jnp.asarray(0.0)
        return self.kla_vector(signals)[self.model.species_index[self.aeration.species]]

    def oxygen_transfer(
        self,
        t,
        state: jnp.ndarray,
        inputs: dict,
        params: jnp.ndarray,
        signals: dict | None,
    ):
        """The aeration transfer the mass balance integrates
        (:class:`~aquakin.plant.units.OxygenTransferUnit`): :meth:`aeration_term`,
        as the unit's ``rhs`` adds it, split into the oxygen and the other gases.

        Parameters
        ----------
        t : float or jnp.ndarray
            Plant time (unused: the aeration is not time-phased).
        state : jnp.ndarray
            The unit's state; its leading ``n_species`` entries are the
            concentrations aeration acts on.
        inputs : dict[str, Stream]
            The unit's input streams, which set its operating temperature.
        params : jnp.ndarray
            The unit's parameter vector.
        signals : dict or None
            The control-signal bus, needed only under closed-loop control.

        Returns
        -------
        oxygen, stripped : jnp.ndarray
            Per-species transfer rate per unit volume, shape ``(n_species,)``:
            on the aerated oxygen species only, and on the co-stripped gases
            only. Both are zero for an unaerated tank.
        """
        n = self.model.n_species
        if self.aeration is None:
            return jnp.zeros((n,)), jnp.zeros((n,))
        transfer = self.aeration_term(state[:n], inputs, params, signals)
        o2_mask = jnp.zeros((n,)).at[self.model.species_index[self.aeration.species]].set(1.0)
        return transfer * o2_mask, transfer * (1.0 - o2_mask)

    # Readers onto the canonical ``self._av`` store, for tests that introspect
    # the per-species aeration vectors.
    @property
    def _kla_vec(self) -> jnp.ndarray:
        return self._av.kla_vec

    @property
    def _sat_vec(self) -> jnp.ndarray:
        return self._av.sat_vec

    @property
    def _controlled_kla(self) -> dict:
        return self._av.controlled


@dataclass
class CSTRUnit(AdjustableTemperatureUnit, AerationUnit, CouplingAware):
    """A single continuous-flow stirred tank with kinetics + aeration.

    Parameters
    ----------
    name : str
        Unit identifier.
    model : CompiledModel
        Kinetic model whose rate functions provide the chemistry term.
    volume : float
        Tank liquid volume (m³).
    input_port_names : list[str]
        Names of the incoming-stream ports. Multiple inflows are summed
        (treated as a built-in mixer) before the mass balance.
    conditions : dict[str, float]
        Spatially-uniform condition values (e.g. ``{"T": 293.15}``). One
        value per condition declared by the model.
    aeration : Aeration, optional
        How the tank is aerated. ``None`` (default) is an anoxic/anaerobic tank
        (no aeration). ``Aeration(kla=120)`` is open-loop aeration at a fixed
        mass-transfer coefficient; ``Aeration(do_setpoint=2.0)`` is closed-loop
        dissolved-oxygen control, for which the plant auto-wires a PI controller
        on this tank's kLa (see :class:`Aeration`).
    output_port : str
        Name of the single output port.
    """

    name: str
    model: CompiledModel
    volume: float
    input_port_names: list[str]
    conditions: dict[str, float] = field(default_factory=dict)
    aeration: Aeration | None = None
    output_port: str = "out"

    def __post_init__(self) -> None:
        check_positive(self.name, "volume", self.volume)
        self._init_conditions(self.conditions)
        # Translate the Aeration spec into the per-species RHS vectors (the
        # fixed-kLa / saturation vectors, the closed-loop signal map, and the
        # temperature-correction config) -- the AerationUnit mixin, shared with
        # the IFAS and MBR units.
        self._setup_aeration()

    @property
    def state_size(self) -> int:
        return self.model.n_species

    def state_names(self) -> tuple[str, ...]:
        """Names of the state entries, in order: the species concentrations."""
        return tuple(self.model.species)

    @property
    def input_ports(self) -> list[str]:
        return list(self.input_port_names)

    @property
    def output_ports(self) -> list[str]:
        return [self.output_port]

    def initial_state(self) -> jnp.ndarray:
        return self.model.default_concentrations()

    def coupling_pattern(self):
        """Structural Jacobian sparsity.

        ``self`` is the reaction kinetics' coupling from the rate AST (a saturated
        Monod term is numerically invisible at any single state, so the syntactic
        dependency is needed, not a probe). ``inlet`` is the convective dilution
        ``(Q/V)(C_in - C)``, which couples each species' derivative to its own
        inlet concentration only -- the identity.
        """
        import numpy as np

        from aquakin.integrate.colored_jacobian import structural_sparsity_pattern
        from aquakin.plant.coupling import CouplingPattern

        n = self.model.n_species
        return CouplingPattern(
            self_pattern=structural_sparsity_pattern(self.model),
            inlet_pattern=np.eye(n, dtype=bool),
        )

    def compute_outputs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> dict[str, Stream]:
        # Total inflow Q. The outflow equals the total inflow (constant
        # volume assumption; no accumulation of water).
        Q_total = jnp.zeros(())
        for name in self.input_port_names:
            Q_total = Q_total + inputs[name].Q
        return {
            self.output_port: Stream(
                Q=Q_total,
                C=state,
                model=self.model,
                scalars=mixed_scalars(inputs, self.input_port_names),
            )
        }

    def flow_outputs(self, input_flows: dict, params: jnp.ndarray, ctx=None) -> dict:
        """Outflow equals total inflow (constant-volume reactor)."""
        Q_total = jnp.zeros(())
        for name in self.input_port_names:
            Q_total = Q_total + input_flows[name]
        return {self.output_port: Q_total}

    def rhs(
        self,
        t: jnp.ndarray,
        state: jnp.ndarray,
        inputs: dict[str, Stream],
        params: jnp.ndarray,
        signals: dict | None = None,
    ) -> jnp.ndarray:
        # Mix inflows (Q-weighted).
        Q_total = jnp.zeros(())
        mass_total = jnp.zeros((self.model.n_species,))
        for name in self.input_port_names:
            s = inputs[name]
            Q_total = Q_total + s.Q
            mass_total = mass_total + s.Q * s.C
        C_in = mass_total / (Q_total + 1e-12)

        # Convection.
        convection = (Q_total / self.volume) * (C_in - state)

        # Chemistry at the operating temperature (the plant-supplied lagged tank
        # temperature, else the flow-weighted inlet, else the static condition),
        # through the canonical dCdt. Convection and aeration are added on top,
        # unlimited (the reaction term is the only one the limiter guards).
        conditions = self._reaction_conditions(inputs, signals)
        chemistry = self._reaction(state, params, conditions)
        # Aeration (mass transfer), with the optional driving-force correction at
        # the same temperature; aeration_transfer then overrides any closed-loop
        # kLa with its control signal.
        aeration = self.aeration_term(state, inputs, params, signals)

        return convection + chemistry + aeration
