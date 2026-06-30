"""The influent-T / network-ref_T consistency guard in ``bsm2_constant_influent``.

The influent temperature and the network's Arrhenius reference ``ref_T`` are
independent knobs that must agree. A large mismatch silently rescales every
rate correction by ``theta**(T - ref_T)`` -- a 14.86 °C inlet on the plain
20 °C ``load_network("asm1")`` cuts nitrification by ~40 %. The guard warns
(naming both values) when they differ by more than the expected BSM2 inlet
offset; it must stay silent for the benchmark-faithful pairing and for the
temperature-agnostic ``T=None`` default.
"""

import warnings

import pytest

import aquakin
from aquakin.plant.bsm.bsm2 import (
    BSM2_AS_TEMPERATURE_K,
    BSM2_CONSTANT_INFLUENT_T,
    bsm2_asm1_network,
    bsm2_constant_influent,
)


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def test_mismatched_network_warns(asm1):
    """14.86 °C inlet on the plain 20 °C network (ref_T 293.15) warns, naming both."""
    with pytest.warns(UserWarning, match="ref_T") as record:
        bsm2_constant_influent(asm1, T=BSM2_CONSTANT_INFLUENT_T)
    msg = str(record[0].message)
    assert f"{BSM2_CONSTANT_INFLUENT_T:.5g}" in msg  # the influent T
    assert "293.15" in msg                            # the network ref_T


def test_benchmark_pairing_is_silent():
    """The faithful pairing (15 °C-referenced network + 14.86 °C inlet) does not warn."""
    net = bsm2_asm1_network()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bsm2_constant_influent(net, T=BSM2_CONSTANT_INFLUENT_T)


def test_temperature_agnostic_default_is_silent(asm1):
    """The ``T=None`` default carries no temperature, so there is nothing to check."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bsm2_constant_influent(asm1)


def test_exact_reference_is_silent():
    """An inlet exactly at the network reference is trivially consistent."""
    net = bsm2_asm1_network()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bsm2_constant_influent(net, T=BSM2_AS_TEMPERATURE_K)
