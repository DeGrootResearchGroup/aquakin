"""Tests for the BSM2 ideal %TSS separators (thickener / dewatering)."""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant._constants import ASM1_TSS_FACTOR, ASM1_TSS_SPECIES
from aquakin.plant.separators import IdealThickener
from aquakin.plant.streams import Stream


@pytest.fixture
def asm1():
    return aquakin.load_network("asm1")


def _inlet(asm1, Q, overrides):
    C = asm1.default_concentrations()
    for sp, v in overrides.items():
        C = C.at[asm1.species_index[sp]].set(v)
    return Stream(Q=jnp.asarray(float(Q)), C=C, network=asm1)


def _tss(asm1, C):
    return float(sum(ASM1_TSS_FACTOR * C[asm1.species_index[sp]] for sp in ASM1_TSS_SPECIES))


def _run(unit, s_in):
    out = unit.compute_outputs(jnp.asarray(0.0), jnp.zeros((0,)), {"inlet": s_in},
                               jnp.zeros((0,)))
    return out["underflow"], out["overflow"]


def test_underflow_hits_target_tss(asm1):
    """The underflow is concentrated to exactly target_tss_percent (in mg/L)."""
    unit = IdealThickener(name="thk", network=asm1, target_tss_percent=7.0)
    s_in = _inlet(asm1, 300.0, {"XI": 1000.0, "XS": 0.0, "XB_H": 2000.0,
                                "XB_A": 100.0, "XP": 500.0})
    under, _ = _run(unit, s_in)
    assert _tss(asm1, under.C) == pytest.approx(7.0 * 1e4, rel=1e-6)  # 70000 mg/L


def test_dewatering_target(asm1):
    unit = IdealThickener(name="dw", network=asm1, target_tss_percent=28.0)
    s_in = _inlet(asm1, 100.0, {"XI": 1000.0, "XB_H": 2000.0, "XP": 500.0})
    under, _ = _run(unit, s_in)
    assert _tss(asm1, under.C) == pytest.approx(28.0 * 1e4, rel=1e-6)


def test_flow_and_solids_mass_balance(asm1):
    """Per-species mass conserved; flows sum to the feed; removal fraction of the
    solids reports to the underflow."""
    unit = IdealThickener(name="thk", network=asm1, target_tss_percent=7.0,
                          tss_removal_percent=98.0)
    Q = 300.0
    over = {"XI": 1000.0, "XS": 50.0, "XB_H": 2000.0, "XB_A": 100.0, "XP": 500.0,
            "XND": 5.0, "SS": 30.0, "SNH": 20.0}
    s_in = _inlet(asm1, Q, over)
    under, overflow = _run(unit, s_in)

    # Flow split conserves volume.
    assert float(under.Q + overflow.Q) == pytest.approx(Q, rel=1e-9)

    # Per-species mass balance: Qu*Cu + Qo*Co == Qin*Cin.
    mass_in = Q * s_in.C
    mass_out = float(under.Q) * under.C + float(overflow.Q) * overflow.C
    assert jnp.allclose(mass_out, mass_in, rtol=1e-9, atol=1e-9)

    # Solubles pass through at the inlet concentration in BOTH outlets.
    for sp in ("SI", "SS", "SNH", "SND", "SALK"):
        i = asm1.species_index[sp]
        assert float(under.C[i]) == pytest.approx(float(s_in.C[i]), rel=1e-9)
        assert float(overflow.C[i]) == pytest.approx(float(s_in.C[i]), rel=1e-9)

    # 98% of the settleable solids mass reports to the underflow.
    i_xi = asm1.species_index["XI"]
    frac_under = float(under.Q * under.C[i_xi]) / float(Q * s_in.C[i_xi])
    assert frac_under == pytest.approx(0.98, rel=1e-6)


def test_overconcentrated_feed_all_to_underflow(asm1):
    """A feed already above the target %TSS cannot be thickened: everything
    leaves with the underflow and the overflow is empty."""
    unit = IdealThickener(name="thk", network=asm1, target_tss_percent=7.0)
    # TSS_in = 0.75 * 100000 = 75000 mg/L > 70000 target.
    s_in = _inlet(asm1, 300.0, {"XI": 100000.0, "XS": 0.0, "XB_H": 0.0,
                                "XB_A": 0.0, "XP": 0.0})
    under, overflow = _run(unit, s_in)
    assert float(overflow.Q) == pytest.approx(0.0, abs=1e-9)
    assert float(under.Q) == pytest.approx(300.0, rel=1e-9)
    assert jnp.allclose(under.C, s_in.C, rtol=1e-9, atol=1e-9)


def test_grad_through_separator(asm1):
    """compute_outputs is differentiable w.r.t. the inlet concentration."""
    import jax
    unit = IdealThickener(name="thk", network=asm1, target_tss_percent=7.0)
    C0 = asm1.default_concentrations().at[asm1.species_index["XB_H"]].set(2500.0)

    def loss(C):
        s_in = Stream(Q=jnp.asarray(300.0), C=C, network=asm1)
        under, _ = _run(unit, s_in)
        return jnp.sum(under.C)

    g = jax.grad(loss)(C0)
    assert jnp.all(jnp.isfinite(g))
