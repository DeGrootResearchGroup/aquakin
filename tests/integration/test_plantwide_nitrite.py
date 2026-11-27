"""The nitrite level of the plant-wide model (``plantwide_nitrite``), generated
by ``examples/plantwide/models/_make_plantwide_nitrite.py``.

Checked against independent references:

- conservation: COD, N, C and P close in every reaction, and the units check
  is clean;
- without nitrite the level is the bio-P level: every inherited process runs
  at the same rate, each nitrate-reducing process now as its first step (to
  nitrite) at Hiatt & Grady's (2008) share of it, while its second step
  (nitrite to N2) is idle;
- the anammox stoichiometry is Strous et al.'s (1998): 1.32 mol nitrite
  consumed and 0.26 mol nitrate made per mol ammonium, 1.02 mol N2;
- the AOB and NOB growth rates, evaluated by hand from their sources (Monod
  on ammonium and nitrite, free ammonia and free nitrous acid inhibition after
  Hiatt & Grady 2008, oxygen affinities of Daebel et al. 2007), and the
  anammox growth rate of Hao et al. (2002) at 30 C and 20 C;
- the gradient of a nitrifier batch matches a central difference.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

MODELS = Path(__file__).parents[2] / "examples" / "plantwide" / "models"
T20, PH = 293.15, 7.0


@pytest.fixture(scope="module")
def nitrite():
    return aquakin.load_model_from_file(MODELS / "plantwide_nitrite.yaml")


@pytest.fixture(scope="module")
def bio_p():
    return aquakin.load_model_from_file(MODELS / "plantwide_bio_p.yaml")


def _vector(model, values):
    return jnp.array([values.get(s, 0.0) for s in model.species])


def _conditions():
    return {"T": jnp.asarray([T20]), "pH": jnp.asarray([PH])}


def test_every_reaction_conserves_cod_n_c_and_p_and_units_are_clean(nitrite):
    assert nitrite.check_conservation() == []
    assert nitrite.check_units() == []
    assert {"SNO2", "XAOB", "XNOB", "XAMX"} <= set(nitrite.species)
    assert "XAUT" not in nitrite.species


def test_without_nitrite_every_process_runs_as_in_the_bio_p_level(nitrite, bio_p):
    """A mixed liquor with heterotrophs, PAOs and GAOs (with their storage),
    nitrate and a little oxygen, no nitrite and no nitrifiers: each process the
    two levels share runs at the same rate, each nitrate-reducing process of the
    bio-P level runs here as its first step at 0.28 / 0.8 of the rate (Hiatt &
    Grady's nitrate step against ASM2d's anoxic factor), and its second step
    does not run. The tolerance is the switches' nitrite term at zero."""
    state = dict(
        SO2=0.05, SF=20.0, SA=10.0, SPRO=5.0, SNH4=15.0, SNO3=5.0, SPO4=10.0, SIC=60.0,
        SI=30.0, XI=500.0, XS=200.0, XH=1500.0,
        XPAO=600.0, XPP=150.0, XGLY_PAO=120.0, XPHA_PAO_AC=40.0, XPHA_PAO_PR=10.0,
        XGAOC=300.0, XGLY_GAOC=80.0, XPHA_GAOC_AC=30.0, XPHA_GAOC_PR=5.0,
        XGAOA=200.0, XGLY_GAOA=50.0, XPHA_GAOA_AC=10.0, XPHA_GAOA_PR=15.0,
    )  # fmt: skip
    r_n = dict(zip(nitrite.reaction_names, np.asarray(
        nitrite.rates(_vector(nitrite, state), nitrite.default_parameters(), _conditions(), 0))))
    r_b = dict(zip(bio_p.reaction_names, np.asarray(
        bio_p.rates(_vector(bio_p, state), bio_p.default_parameters(), _conditions(), 0))))  # fmt: skip
    split = [name for name in r_b if f"{name}_NO3_NO2" in r_n]
    assert len(split) == 23  # 3 heterotroph and 20 PAO / GAO anoxic processes
    for name, rate in r_b.items():
        if name in split:
            assert r_n[f"{name}_NO3_NO2"] == pytest.approx(0.28 / 0.8 * rate, rel=1e-10, abs=1e-12), name
            assert r_n[f"{name}_NO2_N2"] == 0.0, name
        elif name in r_n:
            assert r_n[name] == pytest.approx(rate, rel=1e-10, abs=1e-12), name
    assert any(r > 0.0 for name, r in r_b.items() if name in split)


def test_anammox_stoichiometry_is_strous(nitrite):
    """Strous et al. (1998): NH4 : NO2 : NO3 : N2 = 1 : 1.32 : 0.26 : 1.02 (mol).
    The model gives the nitrite and nitrate coefficients; the ammonium and N2
    follow from the COD and N balances, so the ratios check those balances
    against the published ones (N2 to Strous's two figures)."""
    S = np.asarray(nitrite.compute_stoich(nitrite.default_parameters()))
    row = S[nitrite.reaction_names.index("growth_of_anammox")]
    col = nitrite.species_index
    nh4 = -row[col["SNH4"]]
    assert -row[col["SNO2"]] / nh4 == pytest.approx(1.32, rel=1e-9)
    assert row[col["SNO3"]] / nh4 == pytest.approx(0.26, rel=1e-9)
    assert row[col["SN2"]] / 2.0 / nh4 == pytest.approx(1.02, abs=0.005)


def _rates_at(model, state, T, pH):
    conditions = {"T": jnp.asarray([T]), "pH": jnp.asarray([pH])}
    rates = model.rates(_vector(model, state), model.default_parameters(), conditions, 0)
    return dict(zip(model.reaction_names, np.asarray(rates)))


@pytest.mark.parametrize(("T", "pH"), [(293.15, 7.0), (288.15, 7.8)])
def test_nitrifier_growth_is_monod_on_ammonium_and_nitrite_inhibited_by_fa_and_fna(nitrite, T, pH):
    """AOB and NOB growth written out by hand: AOB 0.9 1/d with theta 1.072
    (Jones et al. 2005), NOB 0.78 1/d with theta 1.11 (Hiatt & Grady 2008;
    Mehrani et al. 2021); Monod on total ammonium (0.6, Mannucci et al. 2020)
    and total nitrite (0.1); apparent floc oxygen half-saturations 0.65 and
    0.23 (Daebel et al. 2007); free ammonia and free nitrous acid from Hiatt &
    Grady's eqs 2-5, inhibiting with their constants (AOB: FA 1.0, FNA 0.1;
    NOB: FA 0.2, FNA 0.04). The nutrient terms are the core's. All in closed
    form, hence rel=1e-10."""
    state = dict(SO2=1.5, SNH4=30.0, SNO2=8.0, SPO4=5.0, XAOB=100.0, XNOB=50.0)
    rates = _rates_at(nitrite, state, T, pH)
    p = {k: float(nitrite.default_parameters()[i]) for k, i in nitrite.param_index.items()}
    fa = 30.0 * 10**pH / (np.exp(6344.0 / T) + 10**pH)
    fna = 8.0 / (np.exp(-2300.0 / T) * 10**pH)
    mu_aob = 0.9 * 1.072 ** (T - 293.15)
    mu_nob = 0.78 * 1.11 ** (T - 293.15)
    p_limit = 5.0 / (p["KP_H"] + 5.0)
    aob = (
        mu_aob * 30.0 / (0.6 + 30.0) * 1.5 / (0.65 + 1.5) * 1.0 / (1.0 + fa) * 0.1 / (0.1 + fna)
        * p_limit * 100.0
    )  # fmt: skip
    nob = (
        mu_nob * 8.0 / (0.1 + 8.0) * 1.5 / (0.23 + 1.5) * 0.04 / (0.04 + fna) * 0.2 / (0.2 + fa)
        * 30.0 / (p["KNH4_H"] + 30.0) * p_limit * 50.0
    )  # fmt: skip
    assert rates["growth_of_AOB"] == pytest.approx(aob, rel=1e-10)
    assert rates["growth_of_NOB"] == pytest.approx(nob, rel=1e-10)


@pytest.mark.parametrize("T", [293.15, 283.15])
def test_nitrifier_decay_is_the_werf_measurement(nitrite, T):
    """Aerobic decay: AOB 0.17 1/d with theta 1.029 (WERF, Jones et al. 2005;
    Friedrich et al. 2017), NOB 0.089 1/d with theta 1.036 (Friedrich et al.).
    At 10 C, 0.128 and 0.062, Friedrich et al.'s tabulated values (to their
    three figures)."""
    state = dict(SO2=8.0, SNH4=10.0, SPO4=5.0, XAOB=100.0, XNOB=50.0)
    rates = _rates_at(nitrite, state, T, 7.0)
    p = {k: float(nitrite.default_parameters()[i]) for k, i in nitrite.param_index.items()}
    aerobic = 8.0 / (p["KO2_H"] + 8.0)
    switch = aerobic + p["eta_b_NOX_AUT"] * 0.0 + p["eta_b_AN"] * (1.0 - aerobic)
    assert rates["decay_of_XAOB"] == pytest.approx(0.17 * 1.029 ** (T - 293.15) * switch * 100.0, rel=1e-10)
    assert rates["decay_of_XNOB"] == pytest.approx(0.089 * 1.036 ** (T - 293.15) * switch * 50.0, rel=1e-10)
    if T == 283.15:
        assert rates["decay_of_XAOB"] / (switch * 100.0) == pytest.approx(0.128, abs=5e-4)
        assert rates["decay_of_XNOB"] / (switch * 50.0) == pytest.approx(0.062, abs=5e-4)


def test_anammox_growth_is_hao_at_30_and_20_c(nitrite):
    """Hao et al. (2002), from Strous et al. (1998): 0.072 1/d at 30 C, with
    oxygen inhibition 0.01 and affinities 0.07 (ammonium) and 0.05 (nitrite);
    at 20 C the activation energy of 70 kJ/mol gives 0.028 1/d (Hao et al.
    quote 0.03). theta is rounded to 10 figures, hence rel=1e-8."""
    state = dict(SO2=0.02, SNH4=0.1, SNO2=0.2, SPO4=5.0, XAMX=100.0)
    p = {k: float(nitrite.default_parameters()[i]) for k, i in nitrite.param_index.items()}
    terms = 0.1 / (0.07 + 0.1) * 0.2 / (0.05 + 0.2) * 0.01 / (0.01 + 0.02) * 5.0 / (p["KP_H"] + 5.0) * 100.0
    arrhenius = np.exp(-70000.0 / 8.314 * (1.0 / 293.15 - 1.0 / 303.15))
    assert 0.072 * arrhenius == pytest.approx(0.028, abs=5e-4)
    assert _rates_at(nitrite, state, 303.15, 7.0)["growth_of_anammox"] == pytest.approx(0.072 * terms, rel=1e-8)
    assert _rates_at(nitrite, state, 293.15, 7.0)["growth_of_anammox"] == pytest.approx(
        0.072 * arrhenius * terms, rel=1e-8
    )


def _nitrifier_batch(nitrite):
    """An aerated AOB and NOB batch, without heterotrophs or anammox."""
    p = nitrite.default_parameters().at[nitrite.param_index["muAMX"]].set(0.0)
    c = _vector(nitrite, dict(SO2=8.0, SNH4=40.0, SNO3=1.0, SIC=500.0, SPO4=50.0, XAOB=30.0, XNOB=20.0))
    reactor = aquakin.BatchReactor(
        nitrite, aquakin.SpatialConditions.uniform(T=T20, pH=PH), rtol=1e-10, atol=1e-10
    )
    return reactor, c, p


@pytest.mark.slow
def test_gradient_of_batch_nitrite_matches_a_central_difference(nitrite):
    """d(nitrite after 0.1 d)/d(muAOB) by reverse mode against a central
    difference with a relative step of 1e-4.

    About 37 s, nearly all compiling the reverse-mode solve of this
    111-reaction model, so it runs in the slow tier; the fast tier checks a
    reverse-mode gradient through the same solve path on the bio-P level
    (``test_plantwide_bio_p.py``)."""
    r_n, c_n, p_n = _nitrifier_batch(nitrite)
    k = nitrite.param_index["muAOB"]

    def nitrite_at(params):
        return r_n.solve(c_n, t_span=(0.0, 0.1), params=params).C_named("SNO2")[-1]

    grad = jax.grad(nitrite_at)(p_n)[k]
    h = 1e-4 * p_n[k]
    fd = (nitrite_at(p_n.at[k].add(h)) - nitrite_at(p_n.at[k].add(-h))) / (2 * h)
    assert jax.config.x64_enabled
    assert float(grad) > 0.0
    assert float(grad) == pytest.approx(float(fd), rel=1e-5)
