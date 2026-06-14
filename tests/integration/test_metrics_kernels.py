"""Direct kernel tests for the effluent / cost metric functions.

These exercise the public metric kernels on hand-built concentration vectors
with closed-form expected values -- no plant solve. They were previously
covered only transitively (``derived_BOD`` / ``derived_TKN`` via
``effluent_averages`` and the EQI; ``operational_cost_index`` via
``evaluate_bsm1``), so a regression in a coefficient could slip through.
"""

import jax.numpy as jnp
import pytest

import aquakin


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def test_derived_BOD_matches_closed_form(asm1):
    # BOD5 proxy = 0.25 * (SS + XS + (1 - f_P) * (XB_H + XB_A)), f_P = 0.08.
    C = asm1.concentrations(
        {"SS": 10.0, "XS": 20.0, "XB_H": 100.0, "XB_A": 50.0}, base="zero"
    )
    expected = 0.25 * (10.0 + 20.0 + (1.0 - 0.08) * (100.0 + 50.0))  # = 42.0
    assert float(aquakin.derived_BOD(C, asm1)) == pytest.approx(expected)


def test_derived_TKN_matches_closed_form(asm1):
    # TKN = SNH + SND + XND + i_XB*(XB_H + XB_A) + i_XP*(XP + XI),
    # with i_XB = 0.086, i_XP = 0.06.
    C = asm1.concentrations(
        {"SNH": 5.0, "SND": 2.0, "XND": 3.0, "XB_H": 100.0, "XB_A": 50.0,
         "XP": 30.0, "XI": 40.0},
        base="zero",
    )
    expected = (5.0 + 2.0 + 3.0
                + 0.086 * (100.0 + 50.0)
                + 0.06 * (30.0 + 40.0))            # = 27.1
    assert float(aquakin.derived_TKN(C, asm1)) == pytest.approx(expected)


def test_derived_kernels_vectorize_over_a_trajectory(asm1):
    # The kernels index C[..., i], so a 2-D (n_t, n_species) trajectory maps to a
    # leading-axis vector with no rank branch.
    row = asm1.concentrations(
        {"SS": 10.0, "XS": 20.0, "XB_H": 100.0, "XB_A": 50.0}, base="zero"
    )
    traj = jnp.stack([row, 2.0 * row])             # (2, n_species)
    bod = aquakin.derived_BOD(traj, asm1)
    assert bod.shape == (2,)
    # linear in C, so doubling the state doubles BOD
    assert float(bod[1]) == pytest.approx(2.0 * float(bod[0]))


def test_operational_cost_index_bsm1_form():
    # BSM1 OCI = aeration + pumping + 5 * sludge_production.
    assert aquakin.operational_cost_index(10.0, 5.0, 2.0) == pytest.approx(25.0)
    # weight is exactly 5 on the sludge term
    base = aquakin.operational_cost_index(10.0, 5.0, 0.0)
    assert aquakin.operational_cost_index(10.0, 5.0, 1.0) - base == pytest.approx(5.0)
    assert isinstance(aquakin.operational_cost_index(1.0, 1.0, 1.0), float)
