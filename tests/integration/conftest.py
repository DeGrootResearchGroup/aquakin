"""Shared whole-plant solves for the integration tests.

A fixture here that builds and solves a full plant must also be named in
``_SLOW_PLANT_FIXTURES`` in ``tests/conftest.py``, so every test using it is
deferred to the merge-only ``slow`` job.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    bsm2_asm1_model,
    bsm2_constant_influent,
    bsm2_parameters,
    build_bsm2,
)


@pytest.fixture(scope="session")
def bsm2_open_loop_10d():
    """The default open-loop BSM2 plant on the constant influent, solved for 10 d.

    One solve shared by every file that needs this plant at this operating point
    (the open-loop contrast in ``test_bsm2_control.py`` and the EQI / OCI
    evaluation in ``test_bsm2_evaluation.py``), so each worker compiles it once.
    The solve starts from the plant's default initial state and saves daily.

    Returns
    -------
    tuple
        ``(plant, sol, params)``.
    """
    asm1 = bsm2_asm1_model()
    adm1 = aquakin.load_model("adm1")
    plant = build_bsm2(asm1, adm1, do_control=False)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    params = bsm2_parameters(asm1, adm1)
    sol = plant.solve(
        (0.0, 10.0),
        t_eval=jnp.linspace(0.0, 10.0, 11),
        params=params,
        rtol=1e-4,
        atol=1e-3,
        integrator=aquakin.IntegratorConfig(max_steps=300_000),
    )
    return plant, sol, params
