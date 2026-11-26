"""The plant-wide bio-P model (``plantwide_bio_p``) on the source's grid.

The same reconstructed SBR and 36 conditions as ``grid.py``, run with the
model's own reaction term (``CompiledModel.dCdt``) in the model's units
(g COD, g P), and compared with the source's Fig. 6 and with the faithful
reference (``grid.json``, written by ``grid.py``). Oxygen is held at 2 g/m3 in
the aerobic phase and at zero in the anaerobic one, as the source assumes;
ammonium and inorganic carbon are supplied in excess.
"""

import json
from pathlib import Path

import numpy as np
from grid import classify
from scipy.integrate import solve_ivp

import aquakin

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "models" / "plantwide_bio_p.yaml"
COD_X = 34.76  # g COD per C-mol biomass
POPS = dict(PAO="XPAO", CMP="XGAOC", ALP="XGAOA")
GLY = dict(PAO="XGLY_PAO", CMP="XGLY_GAOC", ALP="XGLY_GAOA")
FMAX_GLY = dict(PAO=0.27, CMP=0.35, ALP=0.35)

m = aquakin.load_model_from_file(MODEL)
P0 = m.default_parameters()
IX = m.species_index


def _compiled_dcdt():
    """The model's reaction term, compiled once. jax is imported here, after
    aquakin has enabled its 64-bit mode, rather than at module level."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def dcdt(C, T, pH):
        cond = {"T": jnp.reshape(T, (1,)), "pH": jnp.reshape(pH, (1,))}
        return m.dCdt(C, P0, cond, 0)

    return dcdt


_DCDT = _compiled_dcdt()


def rhs(t, y, T, pH, aerobic):
    y = np.array(y)
    y[IX["SO2"]] = 2.0 if aerobic else 0.0
    d = np.array(_DCDT(y, T, pH))
    d[IX["SO2"]] = 0.0
    return d


def run(T, pH, f_hac, days=50.0, x0=30.0, srt_d=8.0, exchange=0.5,
        t_ana=2.0, t_aer=3.5, t_idle=0.5):
    hours = 1.0 / 24.0  # the model's clock is days
    y = np.zeros(m.n_species)
    for pop, x in POPS.items():
        y[IX[x]] = x0 * COD_X
        y[IX[GLY[pop]]] = 0.9 * FMAX_GLY[pop] * x0 * 32.0
    y[IX["XPP"]] = 0.9 * 0.30 * x0 * 31.0
    y[IX["SNH4"]] = 50.0
    y[IX["SIC"]] = 100.0
    cycle = t_ana + t_aer + t_idle
    waste = cycle / (srt_d * 24.0)
    soluble = [i for i, s in enumerate(m.species) if s.startswith("S")]
    particulate = [i for i, s in enumerate(m.species) if s.startswith("X")]
    feed = np.zeros(m.n_species)
    feed[IX["SA"]] = 5.0 * f_hac * 32.0 / exchange
    feed[IX["SPRO"]] = 5.0 * (1 - f_hac) * (112 / 3) / exchange
    feed[IX["SPO4"]] = 0.40 * 31.0 / exchange
    feed[IX["SNH4"]] = 50.0
    feed[IX["SIC"]] = 100.0
    for _ in range(round(days * 24 / cycle)):
        y[soluble] = (1 - exchange) * y[soluble] + exchange * feed[soluble]
        for aerobic, dur in ((False, t_ana), (True, t_aer)):
            sol = solve_ivp(rhs, (0, dur * hours), y, method="LSODA", args=(T + 273.15, pH, aerobic),
                            rtol=1e-6, atol=1e-6)
            y = np.maximum(sol.y[:, -1], 0.0)
        y[particulate] *= 1 - waste
    xs = {p: y[IX[x]] for p, x in POPS.items()}
    tot = sum(xs.values())
    return {p: xs[p] / tot for p in POPS}


if __name__ == "__main__":
    rows = json.load(open(HERE / "grid.json"))
    worst, agree_ref, agree_paper = 0.0, 0, 0
    for r in rows:
        fr = run(r["T"], r["pH"], r["f_hac"])
        dev = max(abs(fr[p] - r[p]) for p in POPS)
        worst = max(worst, dev)
        got = classify(fr)
        agree_ref += got == r["got"]
        agree_paper += got == r["paper"]
        print(f"HAc {r['f_hac']:4.2f} {r['T']:2d}C pH{r['pH']}: model PAO {fr['PAO']:.2f} CMP {fr['CMP']:.2f}"
              f" ALP {fr['ALP']:.2f} | ref {r['PAO']:.2f} {r['CMP']:.2f} {r['ALP']:.2f}  dev {dev:.3f}"
              f"  {got:12s} paper {r['paper']}", flush=True)
    print(f"worst deviation from the faithful reference {worst:.3f}; outcome agrees with the"
          f" reference {agree_ref}/36, with the paper {agree_paper}/36")
