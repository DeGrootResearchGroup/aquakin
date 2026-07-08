"""Monetised operating-cost (OPEX) accounting on top of the OCI.

The Operational Cost Index (:func:`aquakin.plant.operational_cost_index_bsm2`)
is a *dimensionless weighted* cost; for a client deliverable the same component
flows -- energy (kWh/d), external carbon (kg COD/d), wasted sludge (kg TSS/d)
and the digester biogas credit (kg CH₄/d) -- are better expressed in currency.

This module turns those physical flows into an OPEX (currency/d) using a small
set of unit prices (:class:`CostFactors`), optionally adds an annualised CAPEX
term and a GHG carbon price, and reports a labeled breakdown
(:class:`OperatingCost`). The prices are the *only* site-specific inputs; the
physical flows come from a :class:`~aquakin.plant.bsm.BSM2Evaluation` (or BSM1)
already computed by ``evaluate_bsm2``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Days per year for annualising CAPEX and reporting annual OPEX.
_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class CostFactors:
    """Unit prices for monetising a plant's operating cost.

    All prices are in the chosen ``currency`` per the named physical unit.
    Defaults are representative order-of-magnitude values (USD); set them to a
    site's actual tariffs for a real estimate.

    Attributes
    ----------
    currency : str
        Currency label for the report (informational).
    energy_price : float
        Electricity price (currency / kWh).
    carbon_price : float
        External-carbon (e.g. methanol) price (currency / kg COD dosed).
    sludge_disposal_price : float
        Sludge disposal / hauling cost (currency / kg TSS).
    biogas_value : float
        Value of recovered biogas methane (currency / kg CH₄) -- a credit. Set 0
        to ignore the credit (e.g. flared biogas).
    ghg_price : float
        Carbon price applied to the net CO₂e footprint (currency / kg CO₂e),
        default 0 (no carbon charge). Only used when a footprint is supplied.
    capex_annual : float
        Annualised capital cost (currency / yr), spread evenly over the year,
        default 0.
    """

    currency: str = "USD"
    energy_price: float = 0.12
    carbon_price: float = 0.50
    sludge_disposal_price: float = 0.35
    biogas_value: float = 0.20
    ghg_price: float = 0.0
    capex_annual: float = 0.0


@dataclass(frozen=True)
class OperatingCost:
    """A plant's monetised cost as a per-day breakdown.

    ``str(cost)`` / :meth:`report` give a labeled breakdown; :meth:`kpis` exposes
    the headline numbers for a scenario KPI table.

    Attributes
    ----------
    currency : str
        Currency label.
    energy_cost, carbon_cost, sludge_cost : float
        OPEX components (currency/d): electricity, external carbon, sludge
        disposal.
    biogas_credit : float
        Biogas-methane value (currency/d) -- subtracted from the OPEX.
    ghg_cost : float
        Carbon charge on the net CO₂e footprint (currency/d), 0 when no footprint
        / carbon price is given.
    opex_per_day : float
        Net operating cost (currency/d) = energy + carbon + sludge − biogas +
        GHG charge.
    capex_per_day : float
        Annualised capital cost spread over the year (currency/d).
    total_per_day : float
        OPEX + CAPEX (currency/d).
    annual_total : float
        ``total_per_day × 365`` (currency/yr).
    """

    currency: str
    energy_cost: float
    carbon_cost: float
    sludge_cost: float
    biogas_credit: float
    ghg_cost: float
    opex_per_day: float
    capex_per_day: float
    total_per_day: float
    annual_total: float

    def kpis(self) -> dict:
        """Headline cost KPIs for a comparison table (currency/d, annual/yr)."""
        c = self.currency
        return {
            f"OPEX ({c}/d)": self.opex_per_day,
            f"Total cost ({c}/d)": self.total_per_day,
            f"Annual cost ({c}/yr)": self.annual_total,
        }

    def report(self) -> str:
        c = self.currency
        title = f"Operating cost ({c})"
        terms = [
            ("Energy", self.energy_cost),
            ("External carbon", self.carbon_cost),
            ("Sludge disposal", self.sludge_cost),
            ("Biogas credit", -self.biogas_credit),
            ("GHG carbon charge", self.ghg_cost),
        ]
        width = max(len(lbl) for lbl, _ in terms)
        lines = [
            title,
            "=" * len(title),
            f"  Total = {self.total_per_day:14.2f}  {c}/d ({self.annual_total:,.0f} {c}/yr)",
            "",
            f"  {'item':<{width}}  {c + '/d':>14}",
        ]
        for lbl, val in terms:
            lines.append(f"  {lbl:<{width}}  {val:14.2f}")
        lines.append(f"  {'OPEX subtotal':<{width}}  {self.opex_per_day:14.2f}")
        if self.capex_per_day:
            lines.append(f"  {'CAPEX (annualised)':<{width}}  {self.capex_per_day:14.2f}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


def operating_cost(
    *,
    energy_kwh_per_d: float,
    carbon_kg_cod_per_d: float = 0.0,
    sludge_kg_tss_per_d: float = 0.0,
    methane_kg_per_d: float = 0.0,
    factors: CostFactors | None = None,
    co2e_per_d: float | None = None,
) -> OperatingCost:
    """Monetise a plant's operating cost from its physical flows.

    Parameters
    ----------
    energy_kwh_per_d : float
        Total electricity draw (kWh/d) -- e.g. ``AE + PE + ME``.
    carbon_kg_cod_per_d : float
        External-carbon dose (kg COD/d).
    sludge_kg_tss_per_d : float
        Wasted-sludge mass flow to disposal (kg TSS/d).
    methane_kg_per_d : float
        Digester methane production (kg CH₄/d), valued at ``factors.biogas_value``.
    factors : CostFactors, optional
        Unit prices (defaults to the representative :class:`CostFactors`).
    co2e_per_d : float, optional
        Net CO₂e footprint (kg CO₂e/d). When given and ``factors.ghg_price > 0``,
        a carbon charge ``co2e_per_d × ghg_price`` is added to the OPEX.

    Returns
    -------
    OperatingCost
    """
    f = factors if factors is not None else CostFactors()
    energy_cost = float(energy_kwh_per_d) * f.energy_price
    carbon_cost = float(carbon_kg_cod_per_d) * f.carbon_price
    sludge_cost = float(sludge_kg_tss_per_d) * f.sludge_disposal_price
    biogas_credit = float(methane_kg_per_d) * f.biogas_value
    ghg_cost = 0.0 if co2e_per_d is None else float(co2e_per_d) * f.ghg_price
    opex = energy_cost + carbon_cost + sludge_cost - biogas_credit + ghg_cost
    capex_per_day = f.capex_annual / _DAYS_PER_YEAR
    total = opex + capex_per_day
    return OperatingCost(
        currency=f.currency,
        energy_cost=energy_cost,
        carbon_cost=carbon_cost,
        sludge_cost=sludge_cost,
        biogas_credit=biogas_credit,
        ghg_cost=ghg_cost,
        opex_per_day=opex,
        capex_per_day=capex_per_day,
        total_per_day=total,
        annual_total=total * _DAYS_PER_YEAR,
    )
