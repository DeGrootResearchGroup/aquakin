"""The plant-wide bio-P level (``plantwide_bio_p``) and ASM2d on the same A2O.

Both run on ``build_a2o`` (anaerobic, anoxic and aerobic zones, internal and
sludge recycles), with the ideal clarifier since the plant-wide model has no
TSS state yet, and with the same municipal influent: ASM2d's default
(``a2o_influent``), carried over species by species. ASM2d has one VFA; the
plant-wide model splits it into acetate and propionate, 75% / 25% by default
(sewage VFA carries a propionate share). Alkalinity becomes inorganic carbon
(1 mol per mol, 12 g C).

Each plant-wide case is solved for its steady state by pseudo-transient
continuation, then checked for stability: a steady state with no GAOs is an
equilibrium whether or not GAOs could invade it, so both GAO groups are seeded
at 1 g COD/m3 and the plant is run 150 days. Their growth rate over the last
75 days says whether the state stands (negative) or GAOs take over (positive).

Run as a script to print the comparison (about 10 minutes).
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp

import aquakin
from aquakin.plant import A2O_WARM_REACTOR_COMPOSITION, a2o_influent, a2o_warm_start, build_a2o
from aquakin.plant.a2o import A2O_INFLUENT, A2O_Q_AVG

MODEL = Path(__file__).resolve().parents[1] / "models" / "plantwide_bio_p.yaml"
GAOS = {"XGAOC": "XGLY_GAOC", "XGAOA": "XGLY_GAOA"}
SEED_GAO = 1.0  # g COD/m3 of each GAO group, with 0.3 of it again as glycogen
STABILITY_DAYS = 150.0

# ASM2d species the plant-wide model has under another name or not at all.
_RENAMED = {"SALK": "SIC"}
_SALK_TO_SIC = 12.0  # g C per mol of alkalinity (bicarbonate)
# A healthy mixed liquor seeds the plant-wide reactors: ASM2d's warm start, with
# its PHA as acetate-derived PAO PHA, PAO glycogen at a mid storage fraction,
# and GAOs at a few percent of the PAOs so the competition starts from both.
_PLANTWIDE_WARM_EXTRA = {
    "XGLY_PAO": 150.0,
    "XGAOC": 50.0,
    "XGLY_GAOC": 10.0,
    "XGAOA": 50.0,
    "XGLY_GAOA": 10.0,
}


def plantwide_model():
    return aquakin.load_model_from_file(MODEL)


def _carry_over(asm2d_values, model):
    """ASM2d concentrations in the plant-wide model's species: alkalinity as
    inorganic carbon, ASM2d's PHA as the PAOs' acetate-derived PHA, and the
    species the plant-wide model lacks (TSS, metal precipitates) dropped."""
    out = {}
    for s, v in asm2d_values.items():
        if s == "SALK":
            out[_RENAMED[s]] = v * _SALK_TO_SIC
        elif s == "XPHA":
            out["XPHA_PAO_AC"] = v
        elif s in model.species_index:
            out[s] = v
    return out


def plantwide_influent(model, acetate_fraction=0.75):
    """ASM2d's default A2O influent in the plant-wide model's species, its VFA
    split into acetate and propionate."""
    values = _carry_over(A2O_INFLUENT, model)
    vfa = values.pop("SA")
    values["SA"] = acetate_fraction * vfa
    values["SPRO"] = (1.0 - acetate_fraction) * vfa
    return values


def plantwide_plant(model, acetate_fraction=0.75):
    """The A2O on the plant-wide model, its influent added, and the warm start."""
    plant = build_a2o(model, use_takacs=False)
    plant.add_influent(
        "feed", model.influent(plantwide_influent(model, acetate_fraction), Q=A2O_Q_AVG)
    )
    warm = {**_carry_over(A2O_WARM_REACTOR_COMPOSITION, model), **_PLANTWIDE_WARM_EXTRA}
    mixed_liquor = model.concentrations(warm, base="zero")
    y0 = plant.initial_state(
        overrides=dict.fromkeys(plant.activated_sludge_reactors(), mixed_liquor)
    )
    return plant, y0


def _seeded(plant, model, state):
    """``state`` with each GAO group added at ``SEED_GAO`` (and glycogen) in
    every activated-sludge reactor."""
    blocks = {r: plant.states_by_unit(state)[r] for r in plant.activated_sludge_reactors()}
    for r, block in blocks.items():
        for x, gly in GAOS.items():
            block = block.at[model.species_index[x]].add(SEED_GAO)
            block = block.at[model.species_index[gly]].add(0.3 * SEED_GAO)
        blocks[r] = block
    index = plant.states_by_unit(jnp.arange(state.size))
    for r, block in blocks.items():
        state = state.at[jnp.asarray(index[r], dtype=int)].set(block)
    return state


def run_plantwide(model=None, acetate_fraction=0.75, params=None):
    """Steady state and its stability check.

    Returns
    -------
    dict
        ``plant``, ``model``, ``steady`` (the PTC result), ``solution`` (the
        seeded 150-day run, saved at 16 times over its last 75 days) and
        ``gao_growth`` (1/d per GAO group over those 75 days).
    """
    model = plantwide_model() if model is None else model
    plant, y0 = plantwide_plant(model, acetate_fraction)
    params = plant.default_parameters() if params is None else params
    steady = plant.steady_state(params=params, y0=y0, tol=1e-8, max_iter=2000)
    t_eval = jnp.linspace(STABILITY_DAYS / 2, STABILITY_DAYS, 16)
    solution = plant.solve(
        t_span=(0.0, STABILITY_DAYS),
        t_eval=t_eval,
        y0=_seeded(plant, model, steady.state),
        params=params,
        rtol=1e-6,
        atol=1e-4,
        integrator=aquakin.IntegratorConfig(max_steps=4_000_000),
    )

    def held(state, species):
        by_unit = plant.states_by_unit(state)
        return sum(
            float(by_unit[r][model.species_index[species]]) * plant.units[r].volume
            for r in plant.activated_sludge_reactors()
        )

    span = float(t_eval[-1] - t_eval[0])
    growth = {
        x: math.log(held(solution.state[-1], x) / held(solution.state[0], x)) / span for x in GAOS
    }
    return dict(plant=plant, model=model, steady=steady, solution=solution, gao_growth=growth)


def run_asm2d():
    """ASM2d's A2O at its steady state (``a2o_influent``, ``a2o_warm_start``)."""
    model = aquakin.load_model("asm2d")
    plant = build_a2o(model, use_takacs=False)
    plant.add_influent("feed", a2o_influent(model))
    steady = plant.steady_state(y0=a2o_warm_start(plant))
    return dict(plant=plant, model=model, steady=steady)


def summary(result):
    """End of the anaerobic zone (``SF``, VFA, phosphate), the aerobic zone's
    PAOs, and the effluent phosphate, at the steady state."""
    plant, model, state = result["plant"], result["model"], result["steady"].state
    by_unit = plant.states_by_unit(state)

    def at(unit, species):
        return (
            float(by_unit[unit][model.species_index[species]])
            if species in model.species_index
            else 0.0
        )

    return dict(
        SF_anaerobic=at("anaer2", "SF"),
        VFA_anaerobic=at("anaer2", "SA") + at("anaer2", "SPRO"),
        PO4_anaerobic=at("anaer2", "SPO4"),
        PAO=at("aer3", "XPAO"),
        PO4_effluent=at("aer3", "SPO4"),  # the ideal clarifier passes solubles through
    )


if __name__ == "__main__":
    rows = {"ASM2d": summary(run_asm2d())}
    growth = {}
    for label, f in (("plant-wide, VFA 75/25", 0.75), ("plant-wide, all acetate", 1.0)):
        result = run_plantwide(acetate_fraction=f)
        rows[label] = summary(result)
        growth[label] = result["gao_growth"]
    print(
        f"{'':26s} {'SF anaer':>9s} {'VFA anaer':>10s} {'PO4 anaer':>10s} {'PAO':>7s} {'PO4 eff':>8s}"
    )
    for label, r in rows.items():
        print(
            f"{label:26s} {r['SF_anaerobic']:9.2f} {r['VFA_anaerobic']:10.2f} {r['PO4_anaerobic']:10.2f}"
            f" {r['PAO']:7.1f} {r['PO4_effluent']:8.2f}"
        )
    for label, g in growth.items():
        print(
            f"{label}: seeded GAO growth, Competibacter {g['XGAOC']:+.4f}, Alpha {g['XGAOA']:+.4f} 1/d"
        )
