"""Dissolved gases: the built-in gas library and a model's ``gases:`` block.

A model declares which of its species are dissolved gases (``gases:``); the
plant units that own a gas phase -- an aerated tank's air, a digester's
headspace -- read the compiled declaration and the library below to transfer
those gases. The biochemical model itself never carries a gas phase, so the
same model runs in every unit.

Gas-liquid transfer of gas ``i`` follows

    rate_i = kLa_i * (C_i,volatile - C*_i),    C*_i = H_i(T) * p_i,

where ``C_i,volatile`` is the transferable form of the dissolved species (all of
it, or for inorganic carbon only the un-ionised H2CO3*), ``p_i`` the gas's
partial pressure in the gas the liquid meets, and ``H_i(T)`` its Henry's-law
solubility.

**Library values.** Henry's-law solubilities ``H^cp`` at 298.15 K (mol m^-3
Pa^-1) and their temperature dependence ``d ln H / d(1/T)`` (K) are the first
evaluated-review entries of Sander, R. (2023), Compilation of Henry's law
constants (version 5.0.0) for water as solvent, Atmos. Chem. Phys. 23,
10901-12440, doi:10.5194/acp-23-10901-2023: Burkholder et al. (2019) for O2,
N2, CO2 and CH4, Fernandez-Prini et al. (2003) for H2. The diffusivities in
water at 25 C (m^2/s) are from the CRC Handbook of Chemistry and Physics,
"Diffusion of gases in water" (Jaehne et al. 1987 for CO2, CH4 and H2;
Himmelblau 1964 and Landolt-Boernstein for O2 and N2). A gas's liquid-film
coefficient relative to oxygen's follows penetration theory,
``kL_i / kL_O2 = (D_i / D_O2) ** 0.5``. The water vapour pressure is the BSM2
form (Rosen & Jeppsson 2006).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp

from aquakin.core.ph_solver import equilibrium_constants

# Reference temperature of the library values (K).
T_REF = 298.15
# Universal gas constant (J mol^-1 K^-1).
R_GAS = 8.314462618


@dataclass(frozen=True)
class GasProperties:
    """Solubility and diffusivity of one gas in water.

    Attributes
    ----------
    henry : float
        Henry's-law solubility ``H^cp`` at 298.15 K, mol m^-3 Pa^-1.
    henry_temperature : float
        ``d ln H^cp / d(1/T)``, K: solubility at ``T`` is
        ``henry * exp(henry_temperature * (1/T - 1/298.15))``.
    diffusivity : float
        Diffusion coefficient in water at 25 C, m^2 s^-1.
    """

    henry: float
    henry_temperature: float
    diffusivity: float


GAS_LIBRARY: dict[str, GasProperties] = {
    "O2": GasProperties(henry=1.3e-5, henry_temperature=1500.0, diffusivity=2.42e-9),
    "N2": GasProperties(henry=6.4e-6, henry_temperature=1300.0, diffusivity=2.0e-9),
    "CO2": GasProperties(henry=3.4e-4, henry_temperature=2300.0, diffusivity=1.91e-9),
    "CH4": GasProperties(henry=1.4e-5, henry_temperature=1600.0, diffusivity=1.84e-9),
    "H2": GasProperties(henry=7.8e-6, henry_temperature=530.0, diffusivity=5.11e-9),
}

# Mole fraction of each library gas in dry air near sea level: O2, N2 from the
# U.S. Standard Atmosphere (1976); CO2, CH4 and H2 at present-day background
# levels (about 420 ppm, 1.9 ppm and 0.53 ppm). Aeration equilibrates a
# dissolved gas toward its fraction of this air.
AIR_MOLE_FRACTIONS: dict[str, float] = {
    "O2": 0.20946,
    "N2": 0.78084,
    "CO2": 4.2e-4,
    "CH4": 1.9e-6,
    "H2": 5.3e-7,
}
# One standard atmosphere (Pa).
ATMOSPHERE_PA = 101325.0

# Acid/base systems whose un-ionised form is the volatile one.
VALID_VOLATILE_FRACTIONS = ("carbonate",)


@dataclass(frozen=True)
class DissolvedGas:
    """One entry of a compiled ``gases:`` block.

    Attributes
    ----------
    species : str
        The dissolved species that carries the gas.
    index : int
        Its position in the model's state vector.
    gas : str
        The gas it transfers as (a :data:`GAS_LIBRARY` key).
    molar_mass : float
        Converts the species' state units to mol/L (``mol/L = state /
        molar_mass``), as in ``speciation:``.
    henry : float
        ``H^cp`` at 298.15 K, mol m^-3 Pa^-1 (the library value unless the
        model overrides it).
    henry_temperature : float
        ``d ln H^cp / d(1/T)``, K.
    kL_ratio : float
        Liquid-film transfer coefficient relative to oxygen's.
    volatile_fraction : str or None
        The acid/base system whose un-ionised form transfers, or ``None`` when
        the whole dissolved species does.
    """

    species: str
    index: int
    gas: str
    molar_mass: float
    henry: float
    henry_temperature: float
    kL_ratio: float
    volatile_fraction: str | None = None


@dataclass(frozen=True)
class ModelGases:
    """A model's compiled ``gases:`` block.

    Attributes
    ----------
    entries : tuple of DissolvedGas
        One per declared dissolved gas, in declaration order.
    temperature_field : str
        The condition holding the temperature.
    temperature_units : str
        ``"kelvin"`` or ``"celsius"``.
    pH_field : str or None
        The condition (supplied, or produced by ``speciation:``) holding the
        pH, needed when an entry has a ``volatile_fraction``.
    """

    entries: tuple[DissolvedGas, ...]
    temperature_field: str
    temperature_units: str
    pH_field: str | None

    def by_gas(self, gas: str) -> DissolvedGas:
        """The entry for ``gas``.

        Raises
        ------
        KeyError
            If the model declares no species for ``gas``.
        """
        for entry in self.entries:
            if entry.gas == gas:
                return entry
        raise KeyError(f"The model declares no dissolved {gas}; it declares {self.gases}.")

    @property
    def gases(self) -> list[str]:
        """The declared gases, in declaration order."""
        return [entry.gas for entry in self.entries]

    def kelvin(self, T):
        """The temperature condition value in kelvin."""
        return T + 273.15 if self.temperature_units == "celsius" else T


def kL_ratio(gas: str) -> float:
    """A library gas's liquid-film coefficient relative to oxygen's,
    ``(D_gas / D_O2) ** 0.5`` (penetration theory).

    Examples
    --------
    >>> round(kL_ratio("CO2"), 3)
    0.888
    """
    return math.sqrt(GAS_LIBRARY[gas].diffusivity / GAS_LIBRARY["O2"].diffusivity)


def henry(entry: DissolvedGas, T_kelvin):
    """Henry's-law solubility of ``entry`` at ``T_kelvin``, mol m^-3 Pa^-1."""
    return entry.henry * jnp.exp(entry.henry_temperature * (1.0 / T_kelvin - 1.0 / T_REF))


def saturation(entry: DissolvedGas, T_kelvin, partial_pressure):
    """The dissolved concentration in equilibrium with ``partial_pressure``
    (Pa), in the species' own state units."""
    mol_per_m3 = henry(entry, T_kelvin) * partial_pressure
    return mol_per_m3 * entry.molar_mass / 1000.0


def partial_pressure(entry: DissolvedGas, T_kelvin, gas_concentration):
    """The partial pressure (Pa) of a gas-phase concentration given in the
    species' own state units per m^3 of gas (ideal gas)."""
    mol_per_m3 = gas_concentration * 1000.0 / entry.molar_mass
    return mol_per_m3 * R_GAS * T_kelvin


def volatile_concentration(entry: DissolvedGas, C, T_kelvin, pH):
    """The transferable part of the dissolved species ``C`` (state units).

    For inorganic carbon this is H2CO3*, the fraction
    ``h^2 / (h^2 + K1 h + K1 K2)`` of the total at ``pH``, with the carbonate
    constants at ``T_kelvin`` (activities taken as concentrations); for any
    other gas the whole species.
    """
    if entry.volatile_fraction is None:
        return C
    K = equilibrium_constants(T_kelvin)
    h = jnp.power(10.0, -pH)
    k1, k2 = K["co3_1"], K["co3_2"]
    return C * h * h / (h * h + k1 * h + k1 * k2)


def temperature_and_pH(model, C, params, condition_arrays):
    """The temperature (K) and pH a model's gas transfer uses.

    Both come from the condition arrays the kinetics run at; the pH is
    computed from the state ``C`` by the model's speciation when it is not a
    supplied condition, and is ``None`` when the ``gases:`` block names no pH.

    Parameters
    ----------
    model : CompiledModel
        A model with a ``gases:`` block.
    C : jnp.ndarray
        Concentrations, shape ``(n_species,)``.
    params : jnp.ndarray
        The model's parameter vector.
    condition_arrays : dict of str to jnp.ndarray
        The conditions the kinetics run at, one location.

    Returns
    -------
    T : jnp.ndarray
        Temperature (K).
    pH : jnp.ndarray or None
    """
    gases = model.gases
    T = gases.kelvin(jnp.reshape(condition_arrays[gases.temperature_field], (-1,))[0])
    if gases.pH_field is None:
        return T, None
    if gases.pH_field in condition_arrays:
        return T, jnp.reshape(condition_arrays[gases.pH_field], (-1,))[0]
    derived = model.derived_condition_fn(C, params, condition_arrays, 0)
    return T, jnp.reshape(derived[gases.pH_field], ())


def water_vapour_pressure(T_kelvin):
    """Saturated water vapour pressure (Pa), the BSM2 form
    ``0.0313 bar * exp(5290 K * (1/298.15 - 1/T))``.

    Examples
    --------
    >>> round(float(water_vapour_pressure(308.15)), 1)
    5566.8
    """
    return 0.0313e5 * jnp.exp(5290.0 * (1.0 / T_REF - 1.0 / T_kelvin))


def compile_gases(spec, species_index: dict[str, int], condition_fields: set[str]):
    """Compile a model spec's ``gases:`` block, or return ``None`` without one.

    Parameters
    ----------
    spec : ModelSpec
        The validated model spec.
    species_index : dict of str to int
        Species name to state index.
    condition_fields : set of str
        Every condition a rate may read: the declared ones and those a
        ``speciation:`` block produces.

    Returns
    -------
    ModelGases or None

    Raises
    ------
    ValueError
        If an entry names an unknown species, two entries name the same
        gas, or a condition the block reads is not available.
    """
    block = getattr(spec, "gases", None)
    if block is None:
        return None
    if block.temperature_field not in condition_fields:
        raise ValueError(
            f"gases.temperature_field '{block.temperature_field}' is not a condition of the "
            f"model; available: {sorted(condition_fields)}."
        )
    entries = []
    for species, cfg in block.species.items():
        if species not in species_index:
            raise ValueError(f"gases: '{species}' is not a species of the model.")
        if cfg.volatile_fraction is not None and block.pH_field not in condition_fields:
            raise ValueError(
                f"gases: '{species}' has volatile_fraction '{cfg.volatile_fraction}', which "
                f"needs a pH; set gases.pH_field to a condition of the model (got "
                f"{block.pH_field!r})."
            )
        props = GAS_LIBRARY[cfg.gas]
        entries.append(
            DissolvedGas(
                species=species,
                index=species_index[species],
                gas=cfg.gas,
                molar_mass=cfg.molar_mass,
                henry=props.henry if cfg.henry is None else cfg.henry,
                henry_temperature=(
                    props.henry_temperature
                    if cfg.henry_temperature is None
                    else cfg.henry_temperature
                ),
                kL_ratio=kL_ratio(cfg.gas) if cfg.kL_ratio is None else cfg.kL_ratio,
                volatile_fraction=cfg.volatile_fraction,
            )
        )
    gases = [e.gas for e in entries]
    if len(set(gases)) != len(gases):
        raise ValueError(f"gases: each gas may be carried by one species only; got {gases}.")
    return ModelGases(
        entries=tuple(entries),
        temperature_field=block.temperature_field,
        temperature_units=block.temperature_units,
        pH_field=block.pH_field,
    )
