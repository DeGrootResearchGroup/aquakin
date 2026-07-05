"""The asm3_2step model: ASM3 with explicit nitrite (two-step nitrification
and two-step denitrification), after Kaelin et al. (2009).

These check the structure (the two split state variables, 19 processes), the
headline two-step signatures (nitrite as an explicit intermediate of both
nitrification and denitrification), and that gradients flow. COD/N/charge
closure is covered by tests/integration/test_asm_continuity.py.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture
def net():
    return aquakin.load_model("asm3_2step")


def _batch(net, t_end, t_n=21, **C0):
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    r = aquakin.BatchReactor(net, cond)
    t = jnp.linspace(0.0, t_end, t_n)
    sol = r.solve(net.concentrations(C0), params=net.default_parameters(),
                  t_span=(0.0, t_end), t_eval=t)
    return t, sol


def test_structure(net):
    # 15 compounds: SNOX split into SNO2 + SNO3, XA split into XAOB + XNOB.
    assert net.n_species == 15
    assert net.n_reactions == 19          # Kaelin's 20 processes minus reactor aeration
    for s in ("SNO2", "SNO3", "XAOB", "XNOB"):
        assert s in net.species, s
    assert "SNOX" not in net.species and "XA" not in net.species


def test_two_step_nitrification(net):
    # Aerobic batch (a large dissolved-O2 reservoir stands in for aeration, which
    # a BatchReactor does not supply), both nitrifier groups seeded but NOB the
    # slower/smaller pool: NH4 -> NO2 -> NO3, with nitrite a genuine transient
    # intermediate (peaks while AOB outpaces NOB, then NOB clears it to nitrate).
    t, sol = _batch(net, 1.0, SO2=300.0, SNH4=30.0, SNO2=0.0, SNO3=0.0,
                    XAOB=80.0, XNOB=25.0, XH=0.0, XSTO=0.0, SS=0.0, SALK=0.05)
    nh4 = np.asarray(sol.C_named("SNH4"))
    no2 = np.asarray(sol.C_named("SNO2"))
    no3 = np.asarray(sol.C_named("SNO3"))
    assert nh4[-1] < 1.0                          # ammonia oxidised
    assert no3[-1] > 10.0                          # nitrate produced (NOB catch up)
    assert no2.max() > 1.0                         # nitrite is a real intermediate
    assert no2[-1] < no2.max()                     # and a transient (peaks then falls)
    assert np.all(np.isfinite(sol.C))


def test_two_step_denitrification_nitrite_hump(net):
    # Anoxic, nitrate + substrate: NO3 -> NO2 -> N2. The nitrite "hump"
    # (Kaelin Fig 4) is the headline two-step denitrification signature.
    t, sol = _batch(net, 0.15, SO2=0.0, SNO3=12.0, SNO2=0.0, SS=200.0,
                    XH=500.0, XSTO=20.0, XAOB=0.0, XNOB=0.0, SNH4=5.0, SALK=0.01)
    no3 = np.asarray(sol.C_named("SNO3"))
    no2 = np.asarray(sol.C_named("SNO2"))
    n2 = np.asarray(sol.C_named("SN2"))
    assert no3[-1] < 0.5                            # nitrate fully reduced
    assert no2.max() > 0.5                          # nitrite accumulates transiently
    assert no2[0] < no2.max() and no2[-1] < no2.max()   # ... a hump, not monotone
    assert n2[-1] > n2[0]                            # N2 produced
    # Total oxidised N (NO3 + NO2) converts to N2: mass conserved.
    assert (no3 + no2 + n2)[-1] == pytest.approx((no3 + no2 + n2)[0], abs=0.2)


def test_nob_washout_leaves_nitrite(net):
    # With AOB but no NOB, nitritation runs but nitratation cannot: nitrogen
    # stops at nitrite (the nitrite-shunt the two-step model exists to capture).
    t, sol = _batch(net, 1.0, SO2=300.0, SNH4=30.0, SNO2=0.0, SNO3=0.0,
                    XAOB=80.0, XNOB=0.0, XH=0.0, XSTO=0.0, SS=0.0, SALK=0.05)
    no2 = np.asarray(sol.C_named("SNO2"))
    no3 = np.asarray(sol.C_named("SNO3"))
    assert no2[-1] > 10.0                           # ammonia oxidised, stuck at nitrite
    assert no3[-1] < 1e-6                           # no nitrate without NOB


def test_grad_through_solve_is_finite(net):
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    r = aquakin.BatchReactor(net, cond)
    C0 = net.concentrations({"SO2": 0.0, "SNO3": 12.0, "SS": 200.0, "XH": 500.0,
                             "XSTO": 20.0})

    def loss(p):
        sol = r.solve(C0, params=p, t_span=(0.0, 0.05),
                      t_eval=jnp.linspace(0.0, 0.05, 4))
        return jnp.sum(sol.C_named("SNO2"))

    g = jax.grad(loss)(net.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    # The nitrite trajectory depends on the first denitrification-step rate.
    assert float(g[net.param_index["etaH_NO3"]]) != 0.0


def test_temperature_slows_nitrification_in_the_cold(net):
    # AOB carries the strongest temperature dependency (theta ~ 1.13): at 10 C
    # the nitritation rate drops well below the 20 C value.
    C = net.concentrations({"SO2": 6.0, "SNH4": 30.0, "XAOB": 80.0, "SALK": 0.05})
    p = net.default_parameters()
    i = net.reaction_names.index("Aerobic_growth_of_XAOB_nitritation")
    r20 = float(net.rates(C, p, {"T": jnp.array([293.15])}, 0)[i])
    r10 = float(net.rates(C, p, {"T": jnp.array([283.15])}, 0)[i])
    assert r10 < 0.5 * r20
