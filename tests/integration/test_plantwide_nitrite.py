"""The nitrite level of the plant-wide model (``plantwide_nitrite``), generated
by ``examples/plantwide/models/_make_plantwide_nitrite.py``.

Checked against independent references:

- conservation: COD, N, C and P close in every reaction, and the units check
  is clean;
- without nitrite the level is the bio-P level: every inherited process runs
  at the same rate, each nitrate-reducing process now as its first step (to
  nitrite) while its second step (nitrite to N2) is idle;
- the anammox stoichiometry is Strous et al.'s (1998): 1.32 mol nitrite
  consumed and 0.26 mol nitrate made per mol ammonium, 1.02 mol N2;
- with the kinetics aligned, an AOB and NOB batch follows the shipped
  ``asm3_2step`` (Kaelin et al. 2009) while ammonium lasts, and its gradient
  matches a central difference.
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
    bio-P level runs at the same rate as its first step here, and its second
    step does not run. The tolerance is the switches' nitrite term at zero."""
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
            assert r_n[f"{name}_NO3_NO2"] == pytest.approx(rate, rel=1e-10, abs=1e-12), name
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


def _aligned_batch(nitrite):
    """An aerated nitrifier batch (oxygen in excess, no heterotrophs) in both
    models, with asm3_2step given this level's AOB and NOB constants and both
    decays off (asm3_2step's is endogenous respiration, this level's
    death-regeneration). This level's NOB also needs ammonium for its biomass
    and phosphate, and asm3_2step's alkalinity; each is made non-limiting."""
    asm3 = aquakin.load_model("asm3_2step")
    p_asm3 = asm3.default_parameters()
    for name, value in dict(
        KAOBNH4=0.7, KAOBO2=0.25, KNOBNO2=0.1, KNOBO2=0.25, YAOB=0.15, YNOB=0.09,
        bAOB=0.0, bNOB=0.0, KNALK=1e-12,
    ).items():  # fmt: skip
        p_asm3 = p_asm3.at[asm3.param_index[name]].set(value)
    p_n = nitrite.default_parameters()
    for name, value in dict(bAOB=0.0, bNOB=0.0, KNH4_H=1e-9, KP_H=1e-9, muAMX=0.0).items():
        p_n = p_n.at[nitrite.param_index[name]].set(value)
    start = dict(SO2=200.0, SNH4=40.0, SNO3=1.0, XAOB=30.0, XNOB=20.0)
    c_n = _vector(nitrite, {**start, "SIC": 500.0, "SPO4": 50.0})
    c_asm3 = _vector(asm3, {**start, "SALK": 0.05})
    tol = dict(rtol=1e-10, atol=1e-10)
    r_n = aquakin.BatchReactor(nitrite, aquakin.SpatialConditions.uniform(T=T20, pH=PH), **tol)
    r_asm3 = aquakin.BatchReactor(asm3, aquakin.SpatialConditions.uniform(T=T20), **tol)
    return (r_n, c_n, p_n), (r_asm3, c_asm3, p_asm3)


def test_aligned_nitrifier_batch_follows_asm3_2step(nitrite):
    """To 0.2 d, while ammonium lasts (after it, asm3_2step's NOB keeps growing
    and drives ammonium negative, this level's stops): ammonium, nitrite,
    nitrate, oxygen and both biomasses to rtol 1e-7, a decade above the solves'
    rtol. About 12 s, compiling both solves."""
    (r_n, c_n, p_n), (r_asm3, c_asm3, p_asm3) = _aligned_batch(nitrite)
    t = jnp.linspace(0.0, 0.2, 5)
    s_n = r_n.solve(c_n, t_span=(0.0, 0.2), t_eval=t, params=p_n)
    s_asm3 = r_asm3.solve(c_asm3, t_span=(0.0, 0.2), t_eval=t, params=p_asm3)
    for s in ("SNH4", "SNO2", "SNO3", "SO2", "XAOB", "XNOB"):
        np.testing.assert_allclose(s_n.C_named(s), s_asm3.C_named(s), rtol=1e-7, atol=1e-8)
    assert float(s_n.C_named("SNO2")[-1]) > 5.0  # nitrite accumulates between the steps


@pytest.mark.slow
def test_gradient_of_batch_nitrite_matches_a_central_difference(nitrite):
    """d(nitrite after 0.1 d)/d(muAOB) by reverse mode against a central
    difference with a relative step of 1e-4.

    About 37 s, nearly all compiling the reverse-mode solve of this
    111-reaction model, so it runs in the slow tier; the fast tier checks a
    reverse-mode gradient through the same solve path on the bio-P level
    (``test_plantwide_bio_p.py``)."""
    (r_n, c_n, p_n), _ = _aligned_batch(nitrite)
    k = nitrite.param_index["muAOB"]

    def nitrite_at(params):
        return r_n.solve(c_n, t_span=(0.0, 0.1), params=params).C_named("SNO2")[-1]

    grad = jax.grad(nitrite_at)(p_n)[k]
    h = 1e-4 * p_n[k]
    fd = (nitrite_at(p_n.at[k].add(h)) - nitrite_at(p_n.at[k].add(-h))) / (2 * h)
    assert jax.config.x64_enabled
    assert float(grad) > 0.0
    assert float(grad) == pytest.approx(float(fd), rel=1e-5)
