"""Validation: the primary clarifier reproduces the BSM2 reference unit exactly.

Independent unit-level check (no plant solve), companion to
``test_takacs_vs_bsm1_reference``. The benchmark primary clarifier is a
well-mixed holding tank with a hydraulic-retention-time-dependent particulate
removal applied at the outlet:

    n_COD = f_corr * (2.88*f_X - 0.118) * (1.45 + 6.15*ln(HRT_minutes))   [%]
    n_X   = n_COD / f_X                                                    [%]
    ff_i  = 1 - n_X/100  (settleable species)  or  1  (solubles)
    effluent_i = ff_i * x_i ;   sludge_i = ((1 - ff_i)*E + ff_i) * x_i

with the underflow a fixed fraction of the inflow (Q_u = f_PS*Q, E = 1/f_PS).
An independent transcription of those equations is compared to
``PrimaryClarifier.compute_outputs`` for a representative tank state. At a flow
equal to the (reference) smoothed flow the agreement is at the floor of double
precision, confirming the removal law, the effluent/sludge split, and the
settleable-species set are reproduced exactly.

The benchmark additionally low-pass-filters the flow that drives the HRT (a 3 h
lag) before the removal law; aquakin uses the instantaneous inflow. The two are
identical at constant flow (the smoothing is identity at steady state); under a
flow transient they differ, which this test documents but does not treat as an
error.

References
----------
Gernaey, K.V. et al. (2014). Benchmarking of Control Strategies for Wastewater
Treatment Plants. IWA Scientific and Technical Report No. 23.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.streams import Stream

# BSM2 primary-clarifier parameters.
_F_CORR, _F_X, _F_PS, _VOL = 0.65, 0.85, 0.007, 900.0
# Settleable (particulate) ASM1 species removed by the primary clarifier.
_PARTICULATE = ("XI", "XS", "XB_H", "XB_A", "XP", "XND")
# Representative well-mixed tank composition (g m^-3).
_STATE = {"SI": 28.0, "SS": 59.0, "XI": 94.0, "XS": 357.0, "XB_H": 51.0,
          "XB_A": 0.0, "XP": 0.0, "SO": 0.0, "SNO": 0.0, "SNH": 31.0,
          "SND": 6.5, "XND": 19.0, "SALK": 7.0}


def _reference_outputs(asm1, x, Q, Q_smooth):
    """Independent transcription of the BSM2 primary-clarifier outlet streams."""
    si = asm1.species_index
    Qu = _F_PS * Q
    E = Q / Qu
    tt = _VOL / (Q_smooth + 0.001)
    nCOD = _F_CORR * (2.88 * _F_X - 0.118) * (1.45 + 6.15 * np.log(tt * 24.0 * 60.0))
    ff_part = 1.0 - (nCOD / _F_X) / 100.0
    eff = np.array(x, dtype=float)
    sludge = np.array(x, dtype=float)
    for sp in asm1.species:
        i = si[sp]
        ff = ff_part if sp in _PARTICULATE else 1.0
        eff[i] = max(ff * x[i], 0.0)
        sludge[i] = max(((1.0 - ff) * E + ff) * x[i], 0.0)
    return eff, sludge, Q - Qu, Qu


def _aquakin_outputs(clar, asm1, state, Q):
    s_in = Stream(Q=jnp.asarray(float(Q)), C=state, model=asm1)
    out = clar.compute_outputs(0.0, state, {clar.input_port_names[0]: s_in},
                               asm1.default_parameters())
    eff, sludge = out[clar.effluent_port], out[clar.sludge_port]
    return (np.asarray(eff.C), np.asarray(sludge.C),
            float(eff.Q), float(sludge.Q))


@pytest.fixture(scope="module")
def setup():
    asm1 = aquakin.load_model("asm1")
    clar = PrimaryClarifier(name="p", model=asm1, volume=_VOL, f_PS=_F_PS)
    state = np.zeros(asm1.n_species)
    for k, v in _STATE.items():
        state[asm1.species_index[k]] = v
    return asm1, clar, jnp.asarray(state)


@pytest.mark.validation
def test_primclar_matches_reference_at_matched_flow(setup):
    """Removal law, effluent/sludge split, and species mask are exact."""
    asm1, clar, state = setup
    x = np.asarray(state)
    for Q in (20648.0, 10000.0, 60000.0):
        # Reference HRT uses the smoothed flow; at steady state it equals Q, so
        # supplying Q isolates everything except the smoothing.
        b_eff, b_sl, b_Qe, b_Qu = _reference_outputs(asm1, x, Q, Q)
        a_eff, a_sl, a_Qe, a_Qu = _aquakin_outputs(clar, asm1, state, Q)
        assert a_Qe == pytest.approx(b_Qe, rel=1e-12)
        assert a_Qu == pytest.approx(b_Qu, rel=1e-12)
        eff_err = np.max(np.abs(a_eff - b_eff) / np.maximum(np.abs(b_eff), 1.0))
        sl_err = np.max(np.abs(a_sl - b_sl) / np.maximum(np.abs(b_sl), 1.0))
        assert eff_err < 1e-12, f"Q={Q}: effluent off by {eff_err}"
        assert sl_err < 1e-12, f"Q={Q}: sludge off by {sl_err}"


@pytest.mark.validation
def test_primclar_smoothing_is_identity_at_steady_flow(setup):
    """The flow-smoothing difference vanishes when inflow == smoothed flow."""
    asm1, clar, state = setup
    x = np.asarray(state)
    Q = 20648.0
    b_eff, _, _, _ = _reference_outputs(asm1, x, Q, Q)   # smoothed == instantaneous
    a_eff, _, _, _ = _aquakin_outputs(clar, asm1, state, Q)
    assert np.allclose(a_eff, b_eff, rtol=1e-12, atol=1e-12)
