"""The BSM2 dynamic window the plant-sensitivity scripts measure on.

Shared by ``profile_plant_sensitivity.py`` and ``probe_newton_drift.py`` so both
measure the same case: the open-loop BSM2 plant (``build_bsm2`` defaults,
BSM2-configured ASM1 + ADM1) started from its constant-influent steady state
(150 d from ``bsm2_warm_start`` at ``rtol=1e-5`` / ``atol=1e-3``, computed once
and cached as ``.npy``), driven by the synthesised dry-weather influent
(``load_bsm2_influent("dry")``).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import aquakin
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm.bsm2 import (
    bsm2_asm1_model,
    bsm2_constant_influent,
    bsm2_parameters,
    build_bsm2,
)
from aquakin.plant.influent import load_bsm2_influent

# The solve tolerances every measurement on this window uses.
RTOL = 1e-4
ATOL = 1e-3


def _steady_state(asm1, adm1, params, cache: Path):
    if cache.exists():
        return jnp.asarray(np.load(cache))
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    sol = plant.solve(
        t_span=(0.0, 150.0),
        t_eval=jnp.array([0.0, 150.0]),
        params=params,
        y0=bsm2_warm_start(plant),
        rtol=1e-5,
        atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=500_000),
    )
    y_ss = sol.state[-1]
    np.save(cache, np.asarray(y_ss))
    return y_ss


def bsm2_window(cache: Path):
    """Build the dynamic BSM2 plant and its steady-state start.

    Parameters
    ----------
    cache : Path
        Where the steady state is cached (computed on the first call).

    Returns
    -------
    plant : Plant
        BSM2 driven by the synthesised dry-weather influent.
    params : jnp.ndarray
        The BSM2 parameter vector.
    y_ss : jnp.ndarray
        The constant-influent steady state, shape ``(ndof,)``.
    asm_params : list of str
        The plant's ``asm1.*`` parameter names, in order (the first ``k`` are
        the sensitivity parameters of a ``k``-parameter measurement).
    """
    asm1, adm1 = bsm2_asm1_model(), aquakin.load_model("adm1")
    params = bsm2_parameters(asm1, adm1)
    y_ss = _steady_state(asm1, adm1, params, cache)
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    plant.add_influent("feed", load_bsm2_influent("dry", asm1))
    asm_params = [n for n in plant.parameter_names() if n.startswith("asm1.")]
    return plant, params, y_ss, asm_params
