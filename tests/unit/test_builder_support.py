"""Shared plant-builder helpers (issue #463 de-duplication)."""

import pytest

import aquakin
from aquakin.plant._builder_support import reactor_conditions, recycle_pump_flows


def test_recycle_pump_flows_are_constant_setpoints():
    """Qa/Qr are ratio*design-flow, Qw is the wastage, and the clarifier underflow
    is Qr + Qw -- fixed volumetric setpoints, not fractions of throughput."""
    Qa, Qr, Qw, Q_underflow = recycle_pump_flows(
        internal_ratio=3.0, ras_ratio=1.0, Q_design=1000.0, wastage=50.0
    )
    assert Qa == pytest.approx(3000.0)
    assert Qr == pytest.approx(1000.0)
    assert Qw == pytest.approx(50.0)
    assert Q_underflow == pytest.approx(Qr + Qw)  # underflow = RAS + wastage


def test_reactor_conditions_uses_public_defaults():
    """The reactor conditions dict is the model's declared defaults for exactly the
    required fields, via the public condition_defaults() accessor."""
    net = aquakin.load_model("asm1")
    conds = reactor_conditions(net)
    assert set(conds) == set(net.conditions_required)
    for name, value in conds.items():
        assert value == pytest.approx(net.condition_defaults()[name])
