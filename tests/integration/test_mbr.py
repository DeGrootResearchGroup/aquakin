"""Membrane bioreactor: high-MLSS reactor + membrane separation.

The membrane retains the solids (high MLSS, near solids-free permeate), solids
leave only by wasting (SRT = V/Q_waste, decoupled from HRT), and a fouling state
drives the trans-membrane pressure. Aeration reuses the CSTR machinery, including
auto-wired DO control.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import Aeration, MBRUnit
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.plant import Plant
from aquakin.plant.streams import Stream

_PARTS = ["XI", "XS", "XB_H", "XB_A", "XP", "XND"]


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


def _mbr(asm1, **kw):
    kw.setdefault("aeration", Aeration(kla=240.0))
    kw.setdefault("waste_flow", 20.0)
    kw.setdefault("particulate_species", _PARTS)
    kw.setdefault("conditions", {"T": 293.15})
    kw.setdefault("volume", 1000.0)
    kw.setdefault("rejection", 0.999)
    kw.setdefault("membrane_area", 500.0)
    return MBRUnit("mbr", asm1, **kw)


def _plant(asm1, mbr, Q=1000.0):
    p = Plant("mbr_plant")
    p.add_unit(mbr)
    p.add_influent("feed", InfluentSeries.constant(
        asm1, SS=300.0, SNH=40.0, XS=200.0, XB_H=80.0, Q=Q), to="mbr.feed")
    return p


# --- construction / validation ---------------------------------------------

def test_state_and_ports(asm1):
    mbr = _mbr(asm1)
    assert mbr.state_size == asm1.n_species + 1          # C + fouling resistance
    assert mbr.output_ports == ["permeate", "waste"]
    assert mbr.input_ports == ["feed"]


def test_validation(asm1):
    with pytest.raises(ValueError, match="rejection must be in"):
        _mbr(asm1, rejection=1.5)
    with pytest.raises(ValueError, match="waste_flow must be"):
        _mbr(asm1, waste_flow=-1.0)
    with pytest.raises(ValueError, match="missing condition"):
        MBRUnit("m", asm1, volume=1.0, particulate_species=_PARTS)
    with pytest.raises(ValueError, match="particulate species"):
        _mbr(asm1, particulate_species=["NOPE"])


# --- membrane separation ----------------------------------------------------

def test_permeate_rejects_particulates_passes_solubles(asm1):
    mbr = _mbr(asm1, rejection=0.99)
    C = asm1.concentrations({"XB_H": 2000.0, "SS": 5.0})
    s_in = Stream(Q=jnp.asarray(1000.0), C=asm1.default_concentrations(),
                  model=asm1)
    state = jnp.concatenate([C, jnp.zeros((1,))])
    outs = mbr.compute_outputs(jnp.asarray(0.0), state, {"feed": s_in},
                               asm1.default_parameters())
    perm, waste = outs["permeate"], outs["waste"]
    xbh, ss = asm1.species_index["XB_H"], asm1.species_index["SS"]
    assert float(perm.C[xbh]) == pytest.approx(0.01 * 2000.0)   # (1-rej)*reactor
    assert float(perm.C[ss]) == pytest.approx(5.0)              # soluble passes
    assert float(waste.C[xbh]) == pytest.approx(2000.0)         # waste at full MLSS
    # constant volume: permeate + waste = feed
    assert float(perm.Q + waste.Q) == pytest.approx(1000.0)


def test_flow_split_constant_volume(asm1):
    mbr = _mbr(asm1, waste_flow=30.0)
    out = mbr.flow_outputs({"feed": jnp.asarray(1000.0)}, asm1.default_parameters())
    assert float(out["permeate"]) == pytest.approx(970.0)
    assert float(out["waste"]) == pytest.approx(30.0)


# --- the reactor: high MLSS, SRT control ------------------------------------

def test_membrane_retains_solids_high_mlss(asm1):
    """At a 1-day HRT the membrane retains the biomass (high MLSS) where a
    clarifier-less CSTR would wash it out; the permeate is near solids-free."""
    mbr = _mbr(asm1)
    p = _plant(asm1, mbr)
    sol = p.solve(t_span=(0.0, 40.0), t_eval=jnp.linspace(0.0, 40.0, 81))
    assert np.all(np.isfinite(np.asarray(sol.state)))
    xbh = asm1.species_index["XB_H"]
    react = float(sol.state[-1, xbh])
    perm = p.stream(sol, "mbr.permeate")
    assert react > 500.0                                       # solids retained
    assert float(perm.C[-1, xbh]) == pytest.approx(0.001 * react, rel=1e-3)


def test_srt_set_by_wasting(asm1):
    """More wasting -> shorter SRT -> lower steady MLSS."""
    def mlss(qw):
        mbr = _mbr(asm1, waste_flow=qw)
        sol = _plant(asm1, mbr).solve(t_span=(0.0, 40.0), t_eval=jnp.array([40.0]))
        return float(sol.state[-1, asm1.species_index["XB_H"]])
    assert mlss(10.0) > mlss(40.0)                            # less waste -> more MLSS


# --- fouling / TMP ----------------------------------------------------------

def test_fouling_grows_and_drives_tmp(asm1):
    mbr = _mbr(asm1, fouling_rate=1e-3, fouling_relax=0.1)
    p = _plant(asm1, mbr)
    sol = p.solve(t_span=(0.0, 20.0), t_eval=jnp.linspace(0.0, 20.0, 41))
    R_f = np.asarray(sol.state[:, asm1.n_species])
    assert R_f[-1] > R_f[0] >= 0.0                            # fouling builds
    assert R_f[-1] == pytest.approx(R_f[-2], abs=1e-3) or R_f[-1] > R_f[1]  # toward quasi-steady
    perm = p.stream(sol, "mbr.permeate")
    tmp_lo = float(mbr.tmp(jnp.asarray(R_f[1]), perm.Q[1]))
    tmp_hi = float(mbr.tmp(jnp.asarray(R_f[-1]), perm.Q[-1]))
    assert tmp_hi > tmp_lo                                    # TMP rises with fouling


def test_no_fouling_by_default(asm1):
    mbr = _mbr(asm1)                                          # fouling_rate=0
    p = _plant(asm1, mbr)
    sol = p.solve(t_span=(0.0, 10.0), t_eval=jnp.array([10.0]))
    assert float(sol.state[-1, asm1.n_species]) == pytest.approx(0.0)


# --- aeration reuse (DO control auto-wiring) --------------------------------

def test_open_loop_aeration(asm1):
    mbr = _mbr(asm1, aeration=Aeration(kla=180.0))
    assert mbr.required_signals == ()
    assert float(mbr._av.kla_vec[asm1.species_index["SO"]]) == 180.0


def test_do_control_auto_wires_a_controller(asm1):
    mbr = _mbr(asm1, aeration=Aeration(do_setpoint=2.0))
    p = _plant(asm1, mbr)
    p._build_state_layout()                                   # materialises + validates
    assert mbr.required_signals == ("_aer_mbr_kla",)
    assert "mbr_aeration" in p.units                          # auto-wired PI controller
    sol = p.solve(t_span=(0.0, 5.0), t_eval=jnp.array([5.0]))
    assert np.all(np.isfinite(np.asarray(sol.state)))


# --- seasonal temperature ---------------------------------------------------

def test_inlet_temperature_propagates_and_drives_kinetics(asm1):
    """The MBR carries the flow-weighted inlet temperature onto its outlet streams
    and feeds it to the (Arrhenius) kinetics, like a CSTR -- not the static
    condition. A warmer feed nitrifies faster, so the ammonia derivative differs."""
    mbr = _mbr(asm1)
    C = asm1.concentrations({"XB_H": 2000.0, "XB_A": 120.0, "SNH": 30.0, "SO": 2.0})
    state = jnp.concatenate([C, jnp.zeros((1,))])
    p = asm1.default_parameters()
    cold = Stream(Q=jnp.asarray(1000.0), C=asm1.default_concentrations(),
                  model=asm1, scalars={"T": jnp.asarray(283.15)})
    warm = Stream(Q=jnp.asarray(1000.0), C=asm1.default_concentrations(),
                  model=asm1, scalars={"T": jnp.asarray(303.15)})
    outs = mbr.compute_outputs(jnp.asarray(0.0), state, {"feed": cold}, p)
    assert float(outs["permeate"].scalars["T"]) == pytest.approx(283.15)  # carried downstream
    assert float(outs["waste"].scalars["T"]) == pytest.approx(283.15)
    snh = asm1.species_index["SNH"]
    dC_cold = mbr.rhs(jnp.asarray(0.0), state, {"feed": cold}, p)
    dC_warm = mbr.rhs(jnp.asarray(0.0), state, {"feed": warm}, p)
    assert float(dC_cold[snh]) != float(dC_warm[snh])           # T drives kinetics


# --- mass balance -----------------------------------------------------------

def test_mass_balance_closes(asm1):
    """Regression for #348: a plant containing an MBR must close its mass balance.
    The inventory reads the fixed reactor volume (not the fouling resistance), and
    the aeration-O2 / reaction-gas term treats the MBR as a reactive aerated unit."""
    mbr = _mbr(asm1)
    p = _plant(asm1, mbr)
    sol = p.solve(t_span=(0.0, 60.0), t_eval=jnp.linspace(40.0, 60.0, 21))  # near steady
    mb = p.mass_balance(sol)
    for q in ("COD", "N"):
        infl = mb[q].inflow
        assert abs(mb[q].imbalance) < 0.02 * abs(infl)          # closes to a few %


# --- AD ---------------------------------------------------------------------

def test_grad_through_mbr(asm1):
    mbr = _mbr(asm1, fouling_rate=1e-3, fouling_relax=0.1)
    p = _plant(asm1, mbr)
    base = p.default_parameters()
    g = jax.grad(lambda s: jnp.sum(
        p.solve(t_span=(0.0, 5.0), t_eval=jnp.array([5.0]), params=base * s).state ** 2))(1.0)
    assert jnp.isfinite(g)
