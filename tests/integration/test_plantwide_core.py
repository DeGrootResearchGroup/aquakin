"""The core level of the unified plant-wide model (examples/plantwide/models).

What is checked, each against something independent of the model file:

- every process conserves COD, N and C (the declared compositions), and the
  rate expressions are dimensionally consistent;
- the coefficients the balances solve equal the textbook ASM2d expressions;
- under aerobic and anoxic conditions, with no anaerobic organisms present, the
  reaction terms equal the shipped ``asm2d`` model's (the same processes and
  values; ASM2d's extra phosphate and alkalinity terms are saturated);
- a continuously fed mesophilic digester running this model through the plant
  layer reaches a steady state that closes the COD, N and C balances between
  feed and effluent, with every anaerobic group sustained by growth;
- the gradient of a batch digestion result matches a central difference.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import CSTRUnit, Plant
from aquakin.plant.influent import InfluentSeries

MODEL = Path(__file__).parents[2] / "examples" / "plantwide" / "models" / "plantwide_core.yaml"
ANAEROBES = ("XFE", "XPRO", "XAC", "XH2M")


@pytest.fixture(scope="module")
def core():
    return aquakin.load_model_from_file(MODEL)


def _vector(model, values):
    return jnp.array([values.get(s, 0.0) for s in model.species])


def _composition_matrix(model, quantities):
    comp = model.composition()
    return np.array([[comp.get(s, {}).get(q, 0.0) for q in quantities] for s in model.species])


def test_every_process_conserves_cod_n_and_c_and_units_are_clean(core):
    assert core.check_conservation() == []
    assert core.check_units() == []
    assert core.time_unit == "d"


def test_solved_coefficients_match_the_asm2d_expressions(core):
    p = core.default_parameters()
    S = np.asarray(core.compute_stoich(p))
    row = {name: S[i] for i, name in enumerate(core.reaction_names)}
    col = core.species_index
    YH, YAUT, Y_ac = 0.625, 0.24, 0.05
    assert row["aerobic_growth_on_SF"][col["SO2"]] == pytest.approx(-(1 - YH) / YH, rel=1e-12)
    assert row["growth_of_nitrifiers"][col["SO2"]] == pytest.approx(
        -(64 / 14 - YAUT) / YAUT, rel=1e-12
    )
    assert row["anoxic_growth_on_SA"][col["SN2"]] == pytest.approx(
        (1 - YH) / (40 / 14 * YH), rel=1e-12
    )
    # Acetoclastic methanogenesis: the carbon not in methane or biomass is CO2.
    assert row["acetoclastic_methanogenesis"][col["SIC"]] == pytest.approx(
        0.375 - (1 - Y_ac) * 0.1875 - Y_ac * 0.36, rel=1e-12
    )


@pytest.mark.parametrize("SO2", [2.0, 0.0])
def test_aerobic_and_anoxic_reaction_terms_equal_asm2d(core, SO2):
    """With no anaerobes present, the shared processes are ASM2d's without its
    heterotrophic fermentation, which this model leaves to the acidogens.

    ASM2d also limits growth by phosphate and alkalinity; with both at 1e6 those
    Monod terms are 1 - O(1e-8), which sets the tolerance.
    """
    asm2d = aquakin.load_model("asm2d")
    state = dict(
        SO2=SO2, SF=30.0, SA=10.0, SNH4=15.0, SNO3=5.0, SN2=10.0, SI=30.0,
        XI=500.0, XS=200.0, XH=1500.0, XAUT=100.0,
    )  # fmt: skip
    c_core = _vector(core, {**state, "SIC": 60.0})
    c_asm2d = _vector(asm2d, {**state, "SPO4": 1e6, "SALK": 1e6})
    r_core = core.dCdt(c_core, core.default_parameters(), core.default_conditions().fields, 0)
    p_asm2d = asm2d.default_parameters().at[asm2d.param_index["qfe"]].set(0.0)
    r_asm2d = asm2d.dCdt(c_asm2d, p_asm2d, asm2d.default_conditions().fields, 0)
    for s in state:
        np.testing.assert_allclose(
            r_core[core.species_index[s]], r_asm2d[asm2d.species_index[s]], rtol=1e-6, atol=1e-9
        )


FEED = dict(
    SF=300.0, SA=300.0, SI=100.0, SNH4=40.0, SIC=60.0, XI=14000.0, XS=22000.0,
    XH=9000.0, XAUT=500.0, XFE=1.0, XPRO=1.0, XAC=1.0, XH2M=1.0,
)  # fmt: skip


@pytest.fixture(scope="module")
def digester_steady_state(core):
    """A 2000 m3 digester fed 100 m3/d of mixed sludge (20 d), at 35 C, solved for
    800 d from a seeded start."""
    plant = Plant("digester", time_unit="d")
    seed = {**FEED, "XFE": 500.0, "XPRO": 100.0, "XAC": 500.0, "XH2M": 200.0}
    plant.add_unit(
        CSTRUnit(
            name="digester",
            model=core,
            volume=2000.0,
            input_port_names=["inlet"],
            conditions={"T": 308.15, "pH": 7.1},
        )
    )
    feed = _vector(core, FEED)
    plant.add_influent(
        "sludge",
        InfluentSeries(
            t=jnp.asarray([0.0, 1000.0]),
            Q=jnp.asarray([100.0, 100.0]),
            C=jnp.stack([feed, feed]),
            model=core,
        ),
        to="digester.inlet",
    )
    y0 = plant.initial_state(overrides={"digester": _vector(core, seed)})
    sol = plant.solve(t_span=(0.0, 800.0), t_eval=jnp.asarray([700.0, 800.0]), y0=y0)
    return np.asarray(sol.unit_state("digester"))


def test_digester_steady_state_closes_the_cod_n_and_c_balances(core, digester_steady_state):
    """At steady state the effluent carries exactly what the feed brought in.

    There is no gas phase at this level, so methane leaves dissolved and the
    balance is over the liquid alone.
    """
    late, final = digester_steady_state
    np.testing.assert_allclose(late, final, rtol=1e-4, atol=1e-6)  # steady
    E = _composition_matrix(core, ["COD", "N", "C"])
    np.testing.assert_allclose(final @ E, np.asarray(_vector(core, FEED)) @ E, rtol=1e-5)


def test_digester_sustains_every_anaerobic_group_and_makes_methane(core, digester_steady_state):
    """Each anaerobic group stays far above its feed level of 1 g/m3, so it grows
    in the digester rather than passing through, and most of the biodegradable
    COD fed leaves as methane."""
    final = digester_steady_state[-1]
    idx = core.species_index
    for group in ANAEROBES:
        assert final[idx[group]] > 50.0, group
    biodegradable_in = FEED["SF"] + FEED["SA"] + FEED["XS"] + 0.9 * (FEED["XH"] + FEED["XAUT"])
    assert final[idx["SCH4"]] / biodegradable_in == pytest.approx(0.83, abs=0.02)


def test_gradient_of_batch_methane_matches_a_central_difference(core):
    """d(methane after one day)/d(k_m_ac) by reverse mode equals a central
    difference with a relative step of 1e-3.

    About 18 s, almost all of it compiling the reverse-mode stiff solve of the
    19-state model; the one-day batch is already the shortest that exercises
    the methanogens.
    """
    seed = {**FEED, "XFE": 500.0, "XPRO": 100.0, "XAC": 500.0, "XH2M": 200.0}
    reactor = aquakin.BatchReactor(core, aquakin.SpatialConditions.uniform(T=308.15, pH=7.1))
    C0 = _vector(core, seed)
    k = core.param_index["k_m_ac"]

    def methane(params):
        return reactor.solve(C0, t_span=(0.0, 1.0), params=params).C_named("SCH4")[-1]

    p = core.default_parameters()
    grad = jax.grad(methane)(p)[k]
    h = 1e-3 * p[k]
    fd = (methane(p.at[k].add(h)) - methane(p.at[k].add(-h))) / (2 * h)
    assert jax.config.x64_enabled
    assert float(grad) == pytest.approx(float(fd), rel=1e-4)
    assert float(grad) > 0.0
