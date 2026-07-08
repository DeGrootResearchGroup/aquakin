"""Greenhouse-gas (GHG) accounting for a plant solution.

A wastewater treatment plant's carbon footprint has three contributions, all
expressed here as a CO₂-equivalent mass flow (kg CO₂e/d) via 100-year global
warming potentials (GWPs):

* **Direct N₂O** -- nitrous oxide stripped from the aerated reactors. N₂O is a
  potent greenhouse gas (GWP ~273) and the dominant *direct* footprint of an
  activated-sludge plant. It is produced by the nitrifier pathways resolved in
  the N₂O kinetic models (a tracked dissolved ``SN2O`` state); the stripping
  to atmosphere is a reactor concern, computed here from the dissolved
  concentration and the aeration intensity (:func:`stripped_n2o`).
* **Energy CO₂e** -- the indirect footprint of the electricity the plant draws
  (aeration + pumping + mixing), via a grid carbon-intensity factor.
* **Methane** -- a *credit* when the digester biogas offsets fossil energy, and
  a *fugitive emission* for the fraction of CH₄ that leaks unburned (GWP ~27 for
  biogenic methane).

The kernels here are generic (they take energy / mass numbers); the BSM2
plant-coupled entry points (:func:`aquakin.plant.bsm.direct_n2o_emission`,
which reconstructs the stripped N₂O from a solved plant) live alongside
``evaluate_bsm2`` and feed :func:`carbon_footprint`.

The GWP and grid-factor defaults are documented, representative values, not
universal constants -- override them for a specific accounting standard
(IPCC assessment report) or grid.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import jax.numpy as jnp

from aquakin.plant.metrics import time_average

# IPCC AR6 (2021) 100-year global warming potentials (kg CO₂e / kg gas).
# N₂O and *biogenic* CH₄ (the wastewater case -- carbon of recent biological
# origin); fossil CH₄ is slightly higher (~29.8). Override per accounting
# standard (AR5: N₂O 265, CH₄ 28; AR4: N₂O 298, CH₄ 25).
GWP_N2O = 273.0
GWP_CH4 = 27.0

# Molar-mass ratio N₂O / N₂ = 44 / 28: converts an N₂O-N mass (the dissolved
# state is referenced to its nitrogen) to an N₂O gas mass.
_N2O_PER_N = 44.0 / 28.0

# Representative grid carbon intensity (kg CO₂e / kWh). Highly region- and
# year-dependent; supply the actual factor for a real accounting.
DEFAULT_GRID_FACTOR = 0.4


# The three scalar CO₂e converters below are **post-solve, eager-only** reporting
# helpers: they take concrete per-day mass/energy flows already reduced from a
# solution and coerce with ``float(...)``, so they are not differentiable and not
# meant for the traced hot path (unlike the ``jnp``-based ``stripped_n2o`` history
# integral). Feed them the eager flows from an evaluation; do not call them under
# ``jax.jit`` / ``jax.grad``.


def co2e_from_energy(energy_kwh_per_d: float, grid_factor: float) -> float:
    """Indirect CO₂e from electricity use (kg CO₂e/d).

    ``= energy × grid_factor``, the energy draw (kWh/d) times the grid carbon
    intensity (kg CO₂e/kWh). Post-solve, eager-only (see the note above).
    """
    return float(energy_kwh_per_d) * float(grid_factor)


def n2o_n_to_co2e(n2o_n_kg_per_d: float, gwp: float = GWP_N2O) -> float:
    """CO₂e of an N₂O emission given as an **N₂O-N** mass flow (kg CO₂e/d).

    ``= n2o_n × (44/28) × GWP`` -- the N₂O-N mass is first converted to N₂O gas
    mass (molar-mass ratio 44/28), then weighted by the N₂O GWP.
    """
    return float(n2o_n_kg_per_d) * _N2O_PER_N * float(gwp)


def methane_to_co2e(ch4_kg_per_d: float, gwp: float = GWP_CH4) -> float:
    """CO₂e of a methane mass flow (kg CO₂e/d). ``= ch4 × GWP``."""
    return float(ch4_kg_per_d) * float(gwp)


def stripped_n2o(
    t: jnp.ndarray,
    kla_o2_history: jnp.ndarray,
    s_n2o_history: jnp.ndarray,
    volumes: jnp.ndarray,
    *,
    kla_ratio: float = 1.0,
    s_n2o_sat: float = 0.0,
) -> float:
    """Time-averaged N₂O stripped from the aerated reactors (kg N₂O-N/d).

    Each reactor strips dissolved N₂O at the aeration mass-transfer rate::

        G_i(t) = kLa_{N2O,i} × (S_{N2O,i} − S*_{N2O}) × V_i

    summed over reactors and time-averaged over the window. The N₂O transfer
    coefficient is taken as the oxygen ``kLa`` scaled by ``kla_ratio`` (the
    diffusivity ratio D_{N2O}/D_{O2} ≈ 1), so an *unaerated* tank (``kLa = 0``)
    strips nothing -- only the aerated tanks emit. The atmospheric saturation
    ``S*_{N2O}`` is ~0.

    Parameters
    ----------
    t : (n_t,) save times (days).
    kla_o2_history : (n_t, n_reactors) oxygen ``kLa`` per reactor (1/d).
    s_n2o_history : (n_t, n_reactors) dissolved N₂O-N concentration (g N/m³).
    volumes : (n_reactors,) reactor liquid volumes (m³).
    kla_ratio : float
        N₂O-to-O₂ mass-transfer-coefficient ratio (default 1.0).
    s_n2o_sat : float
        Atmospheric N₂O saturation concentration (g N/m³, default 0).

    Returns
    -------
    float
        Stripped N₂O-N mass flow (kg N/d), time-averaged.
    """
    kla = jnp.asarray(kla_o2_history) * float(kla_ratio)
    s = jnp.asarray(s_n2o_history) - float(s_n2o_sat)
    volumes = jnp.asarray(volumes)
    # g N/d summed over reactors, then g→kg.
    flux = jnp.sum(kla * s * volumes[None, :], axis=1) * 1e-3  # (n_t,) kg N/d
    return float(time_average(flux, t))


@dataclass
class CarbonFootprint:
    """A plant's greenhouse-gas footprint as a CO₂-equivalent mass flow.

    ``str(fp)`` / :meth:`report` give a labeled breakdown; the raw fields stay
    available for programmatic use and :meth:`kpis` exposes the headline numbers
    for a scenario KPI table.

    Attributes
    ----------
    direct_n2o : float
        Direct N₂O emission (kg N₂O-N/d) stripped from the aerated reactors.
    energy_kwh : float
        Total electricity draw (kWh/d) attributed to the footprint
        (aeration + pumping + mixing).
    ch4_fugitive : float
        Fugitive (unburned, leaked) methane (kg CH₄/d).
    biogas_recovered_kwh : float
        Electricity the recovered biogas displaces (kWh/d) -- an avoided-emission
        credit, valued at the grid factor.
    grid_factor : float
        Grid carbon intensity used (kg CO₂e/kWh).
    gwp_n2o, gwp_ch4 : float
        The GWPs used.
    direct_n2o_co2e, energy_co2e, ch4_fugitive_co2e, biogas_credit_co2e : float
        The CO₂e contributions (kg CO₂e/d); ``biogas_credit_co2e`` is the avoided
        emission (subtracted from the total).
    total_co2e : float
        Net carbon footprint (kg CO₂e/d).
    """

    direct_n2o: float
    energy_kwh: float
    ch4_fugitive: float
    biogas_recovered_kwh: float
    grid_factor: float
    gwp_n2o: float
    gwp_ch4: float
    direct_n2o_co2e: float
    energy_co2e: float
    ch4_fugitive_co2e: float
    biogas_credit_co2e: float
    total_co2e: float
    note: str = (
        "GHG footprint as CO2e/d (IPCC AR6 GWPs: N2O 273, biogenic CH4 27). "
        "Direct N2O is stripped from the aerated reactors; energy CO2e uses the "
        "supplied grid carbon intensity; the biogas credit is the avoided grid "
        "emission of the recovered biogas energy. Defaults are representative -- "
        "override the GWPs / grid factor for a specific standard or grid."
    )

    def kpis(self) -> dict:
        """Headline GHG KPIs (kg CO₂e/d unless noted) for a comparison table."""
        return {
            "GHG total (kgCO2e/d)": self.total_co2e,
            "N2O direct (kgCO2e/d)": self.direct_n2o_co2e,
            "Energy (kgCO2e/d)": self.energy_co2e,
            "CH4 fugitive (kgCO2e/d)": self.ch4_fugitive_co2e,
            "Biogas credit (kgCO2e/d)": -self.biogas_credit_co2e,
        }

    def report(self) -> str:
        title = "Carbon footprint (CO2e)"
        terms = [
            ("Direct N2O", self.direct_n2o, "kg N/d", self.direct_n2o_co2e),
            ("Energy (grid)", self.energy_kwh, "kWh/d", self.energy_co2e),
            ("CH4 fugitive", self.ch4_fugitive, "kg CH4/d", self.ch4_fugitive_co2e),
            ("Biogas credit", self.biogas_recovered_kwh, "kWh/d", -self.biogas_credit_co2e),
        ]
        width = max(len(lbl) for lbl, *_ in terms)
        lines = [
            title,
            "=" * len(title),
            f"  Net footprint = {self.total_co2e:14.1f}  kg CO2e/d (lower is better)",
            "",
            f"  {'source':<{width}}  {'amount':>12}  {'unit':<9}  {'kg CO2e/d':>12}",
        ]
        for lbl, val, unit, co2e in terms:
            lines.append(f"  {lbl:<{width}}  {val:12.3f}  {unit:<9}  {co2e:12.1f}")
        lines.append("")
        lines += textwrap.wrap(
            self.note, width=76, initial_indent="  Note: ", subsequent_indent="        "
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


def carbon_footprint(
    energy_kwh_per_d: float,
    *,
    grid_factor: float = DEFAULT_GRID_FACTOR,
    n2o_emission: float = 0.0,
    methane_production: float = 0.0,
    ch4_fugitive_fraction: float = 0.0,
    biogas_recovered_kwh: float = 0.0,
    gwp_n2o: float = GWP_N2O,
    gwp_ch4: float = GWP_CH4,
) -> CarbonFootprint:
    """Assemble a :class:`CarbonFootprint` from energy use and gas emissions.

    Parameters
    ----------
    energy_kwh_per_d : float
        Total electricity draw (kWh/d) -- e.g. ``AE + PE + ME`` from a
        :class:`~aquakin.plant.bsm.BSM2Evaluation`.
    grid_factor : float
        Grid carbon intensity (kg CO₂e/kWh).
    n2o_emission : float
        Direct N₂O emission as an **N₂O-N** mass flow (kg N/d), e.g. from
        :func:`aquakin.plant.bsm.direct_n2o_emission`.
    methane_production : float
        Digester methane production (kg CH₄/d).
    ch4_fugitive_fraction : float
        Fraction of the produced methane that leaks unburned (0--1). The fugitive
        CH₄ is ``ch4_fugitive_fraction × methane_production``.
    biogas_recovered_kwh : float
        Electricity the recovered (combusted) biogas displaces (kWh/d) -- an
        avoided-emission credit at ``grid_factor``. Defaults to 0 (no credit);
        a typical value is the lower heating value of the non-fugitive CH₄ times a
        CHP electrical efficiency.
    gwp_n2o, gwp_ch4 : float
        Global warming potentials (kg CO₂e/kg gas).

    Returns
    -------
    CarbonFootprint
    """
    ch4_fugitive = float(ch4_fugitive_fraction) * float(methane_production)
    direct_n2o_co2e = n2o_n_to_co2e(n2o_emission, gwp_n2o)
    energy_co2e = co2e_from_energy(energy_kwh_per_d, grid_factor)
    ch4_fugitive_co2e = methane_to_co2e(ch4_fugitive, gwp_ch4)
    biogas_credit_co2e = co2e_from_energy(biogas_recovered_kwh, grid_factor)
    total = direct_n2o_co2e + energy_co2e + ch4_fugitive_co2e - biogas_credit_co2e
    return CarbonFootprint(
        direct_n2o=float(n2o_emission),
        energy_kwh=float(energy_kwh_per_d),
        ch4_fugitive=ch4_fugitive,
        biogas_recovered_kwh=float(biogas_recovered_kwh),
        grid_factor=float(grid_factor),
        gwp_n2o=float(gwp_n2o),
        gwp_ch4=float(gwp_ch4),
        direct_n2o_co2e=direct_n2o_co2e,
        energy_co2e=energy_co2e,
        ch4_fugitive_co2e=ch4_fugitive_co2e,
        biogas_credit_co2e=biogas_credit_co2e,
        total_co2e=total,
    )
