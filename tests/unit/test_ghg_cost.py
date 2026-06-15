"""Unit tests for the GHG / cost reporting kernels and KPI comparison.

These are pure numerical kernels (no ODE solve), so they belong in the fast
gate. The plant-coupled ``direct_n2o_emission`` is exercised in
``tests/integration/test_bsm2_evaluation.py`` on the shared BSM2 solve.
"""

import math

import jax.numpy as jnp
import pytest

import aquakin as ak
from aquakin.plant.ghg import GWP_CH4, GWP_N2O, _N2O_PER_N


# --- GHG kernels -------------------------------------------------------------

def test_co2e_from_energy_linear():
    assert ak.co2e_from_energy(1000.0, 0.4) == pytest.approx(400.0)
    assert ak.co2e_from_energy(0.0, 0.4) == 0.0


def test_n2o_n_to_co2e_uses_molar_ratio_and_gwp():
    # 1 kg N2O-N -> 44/28 kg N2O -> * GWP.
    assert ak.n2o_n_to_co2e(1.0) == pytest.approx(_N2O_PER_N * GWP_N2O)
    # Override GWP.
    assert ak.n2o_n_to_co2e(2.0, gwp=300.0) == pytest.approx(2.0 * 44 / 28 * 300.0)


def test_methane_to_co2e():
    assert ak.methane_to_co2e(10.0) == pytest.approx(10.0 * GWP_CH4)


def test_stripped_n2o_only_aerated_tanks_emit():
    # Two tanks, two times. Tank 0 aerated (kLa>0), tank 1 anoxic (kLa=0).
    t = jnp.array([0.0, 1.0])
    kla = jnp.array([[100.0, 0.0], [100.0, 0.0]])      # (n_t, n_reac)
    s_n2o = jnp.array([[0.5, 5.0], [0.5, 5.0]])         # g N/m3
    vol = jnp.array([1000.0, 2000.0])
    # Only tank 0 strips: 100 * 0.5 * 1000 = 50000 g N/d = 50 kg N/d.
    assert ak.stripped_n2o(t, kla, s_n2o, vol) == pytest.approx(50.0)


def test_stripped_n2o_ratio_and_saturation():
    t = jnp.array([0.0, 1.0])
    kla = jnp.array([[200.0], [200.0]])
    s_n2o = jnp.array([[1.0], [1.0]])
    vol = jnp.array([500.0])
    # kLa_N2O = 0.5 * 200 = 100; (S - S*) = 1 - 0.2 = 0.8; * 500 = 40000 g/d.
    got = ak.stripped_n2o(t, kla, s_n2o, vol, kla_ratio=0.5, s_n2o_sat=0.2)
    assert got == pytest.approx(40.0)


def test_carbon_footprint_total_is_sum_with_credit_subtracted():
    fp = ak.carbon_footprint(
        5000.0, grid_factor=0.4, n2o_emission=10.0,
        methane_production=1000.0, ch4_fugitive_fraction=0.02,
        biogas_recovered_kwh=3000.0,
    )
    assert fp.direct_n2o_co2e == pytest.approx(ak.n2o_n_to_co2e(10.0))
    assert fp.energy_co2e == pytest.approx(2000.0)
    assert fp.ch4_fugitive == pytest.approx(20.0)            # 2% of 1000
    assert fp.ch4_fugitive_co2e == pytest.approx(20.0 * GWP_CH4)
    assert fp.biogas_credit_co2e == pytest.approx(1200.0)    # 3000 kWh * 0.4
    expected = (fp.direct_n2o_co2e + fp.energy_co2e
                + fp.ch4_fugitive_co2e - fp.biogas_credit_co2e)
    assert fp.total_co2e == pytest.approx(expected)


def test_carbon_footprint_defaults_zero_emissions():
    fp = ak.carbon_footprint(1000.0, grid_factor=0.5)
    assert fp.direct_n2o_co2e == 0.0
    assert fp.ch4_fugitive_co2e == 0.0
    assert fp.biogas_credit_co2e == 0.0
    assert fp.total_co2e == pytest.approx(500.0)


def test_carbon_footprint_report_and_kpis():
    fp = ak.carbon_footprint(1000.0, grid_factor=0.4, n2o_emission=5.0)
    text = str(fp)
    assert "Carbon footprint" in text and "kg CO2e/d" in text
    kpis = fp.kpis()
    assert kpis["GHG total (kgCO2e/d)"] == pytest.approx(fp.total_co2e)
    # The biogas KPI is reported as a (negative) credit.
    assert kpis["Biogas credit (kgCO2e/d)"] == pytest.approx(-fp.biogas_credit_co2e)


# --- Cost kernels ------------------------------------------------------------

def test_operating_cost_components_and_total():
    f = ak.CostFactors(energy_price=0.10, carbon_price=0.5,
                       sludge_disposal_price=0.3, biogas_value=0.2)
    oc = ak.operating_cost(
        energy_kwh_per_d=5000.0, carbon_kg_cod_per_d=100.0,
        sludge_kg_tss_per_d=2000.0, methane_kg_per_d=1000.0, factors=f,
    )
    assert oc.energy_cost == pytest.approx(500.0)
    assert oc.carbon_cost == pytest.approx(50.0)
    assert oc.sludge_cost == pytest.approx(600.0)
    assert oc.biogas_credit == pytest.approx(200.0)
    assert oc.ghg_cost == 0.0
    assert oc.opex_per_day == pytest.approx(500 + 50 + 600 - 200)
    assert oc.capex_per_day == 0.0
    assert oc.total_per_day == pytest.approx(oc.opex_per_day)
    assert oc.annual_total == pytest.approx(oc.total_per_day * 365.0)


def test_operating_cost_capex_annualised_and_ghg_charge():
    f = ak.CostFactors(energy_price=0.0, carbon_price=0.0,
                       sludge_disposal_price=0.0, biogas_value=0.0,
                       ghg_price=0.05, capex_annual=365000.0)
    oc = ak.operating_cost(energy_kwh_per_d=0.0, co2e_per_d=1000.0, factors=f)
    assert oc.ghg_cost == pytest.approx(50.0)        # 1000 * 0.05
    assert oc.capex_per_day == pytest.approx(1000.0)  # 365000 / 365
    assert oc.total_per_day == pytest.approx(50.0 + 1000.0)


def test_operating_cost_co2e_none_means_no_charge():
    f = ak.CostFactors(ghg_price=10.0)
    oc = ak.operating_cost(energy_kwh_per_d=100.0, factors=f, co2e_per_d=None)
    assert oc.ghg_cost == 0.0


def test_operating_cost_report_and_kpis():
    oc = ak.operating_cost(energy_kwh_per_d=1000.0, factors=ak.CostFactors())
    assert "Operating cost" in str(oc)
    kpis = oc.kpis()
    assert any("OPEX" in k for k in kpis)
    assert any("Annual" in k for k in kpis)


# --- KPI comparison ----------------------------------------------------------

def test_kpi_comparison_union_columns_and_blanks():
    fp = ak.carbon_footprint(1000.0, grid_factor=0.4)
    oc = ak.operating_cost(energy_kwh_per_d=1000.0)
    kc = ak.kpi_comparison({"footprint": fp, "cost": oc})
    assert kc.names == ["footprint", "cost"]
    # Columns are the union, first-seen order: footprint KPIs then cost KPIs.
    assert "GHG total (kgCO2e/d)" in kc.kpi_names
    assert any("OPEX" in k for k in kc.kpi_names)
    # A KPI only one report provides is NaN for the other.
    col = kc.column("GHG total (kgCO2e/d)")
    assert col["footprint"] == pytest.approx(fp.total_co2e)
    assert math.isnan(col["cost"])


def test_kpi_comparison_accepts_plain_dict():
    kc = ak.kpi_comparison({"a": {"EQI": 5.0, "OCI": 3.0},
                            "b": {"EQI": 4.0, "OCI": 6.0}})
    assert kc.kpi_names == ["EQI", "OCI"]
    assert kc.best("EQI", minimize=True) == "b"
    assert kc.best("OCI", minimize=False) == "b"


def test_kpi_comparison_best_skips_nan():
    kc = ak.kpi_comparison({"a": {"x": 5.0}, "b": {"y": 1.0}})
    # Only "a" has a finite "x".
    assert kc.best("x") == "a"


def test_kpi_comparison_table_renders():
    kc = ak.kpi_comparison({"base": {"EQI": 5000.0, "OCI": 1234.5}})
    text = kc.table()
    assert "EQI" in text and "OCI" in text and "base" in text


def test_kpi_comparison_rejects_bad_report():
    with pytest.raises(TypeError):
        ak.kpi_comparison({"bad": object()})
