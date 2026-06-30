"""Plant.solve(solver=...) -- override the default Kvaerno5 integrator.

A lower-order ESDIRK (Kvaerno3) does less implicit linear algebra per step, a
speed lever on large stiff plants whose per-step cost is dominated by the
Jacobian factorisation. These tests check the override runs, agrees with the
default within tolerance, keys the compiled-solve cache by solver class, and is
rejected on the paths that manage their own integrator.
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.cstr import Aeration, CSTRUnit
from aquakin.plant.plant import Plant


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def _mini_plant(asm1):
    """A single aerated CSTR fed by a constant influent -- fast to solve."""
    plant = Plant("solver-opt")
    plant.add_unit(CSTRUnit("tank", asm1, volume=1000.0,
                            input_port_names=["inlet"],
                            conditions={"T": 288.15},
                            aeration=Aeration(kla=240.0, do_sat=8.0)))
    plant.add_influent("feed", asm1.influent({"SS": 60.0, "SNH": 25.0, "XS": 80.0,
                                              "XB_H": 30.0}, Q=1000.0),
                       to="tank.inlet")
    return plant


def test_solver_override_runs_and_matches_default(asm1):
    plant = _mini_plant(asm1)
    y0 = plant.initial_state()
    t_eval = jnp.linspace(0.0, 2.0, 9)
    # The reference arm pins the old Kvaerno5/no-cap default to preserve the
    # default(K5)-vs-K3 contrast (the new Plant.solve default is K3 + factormax=3).
    base = plant.solve(t_span=(0.0, 2.0), t_eval=t_eval, y0=y0,
                       integrator=aquakin.IntegratorConfig(order=5, factormax=None))
    k3 = plant.solve(t_span=(0.0, 2.0), t_eval=t_eval, y0=y0,
                     integrator=aquakin.IntegratorConfig(solver=diffrax.Kvaerno3()))
    assert np.all(np.isfinite(np.asarray(k3.state)))
    rel = np.max(np.abs(np.asarray(k3.state[-1]) - np.asarray(base.state[-1]))
                 / (np.abs(np.asarray(base.state[-1])) + 1e-9))
    assert rel < 1e-3   # same trajectory, different integrator


def test_solver_override_keys_cache_by_class(asm1):
    plant = _mini_plant(asm1)
    y0 = plant.initial_state()
    t_eval = jnp.linspace(0.0, 1.0, 5)
    plant._jit_cache.clear()
    # Pin Kvaerno5 on the reference arm so it keys distinctly from the explicit
    # Kvaerno3 (the new default is itself K3).
    plant.solve(t_span=(0.0, 1.0), t_eval=t_eval, y0=y0,
                integrator=aquakin.IntegratorConfig(order=5, factormax=None))  # Kvaerno5
    n_after_default = len(plant._jit_cache)
    plant.solve(t_span=(0.0, 1.0), t_eval=t_eval, y0=y0,
                integrator=aquakin.IntegratorConfig(solver=diffrax.Kvaerno3()))
    assert len(plant._jit_cache) == n_after_default + 1   # distinct entry
    # A fresh instance of the same class reuses the entry (no new compile).
    plant.solve(t_span=(0.0, 1.0), t_eval=t_eval, y0=y0,
                integrator=aquakin.IntegratorConfig(solver=diffrax.Kvaerno3()))
    assert len(plant._jit_cache) == n_after_default + 1


def test_solver_override_rejected_with_events(asm1):
    plant = _mini_plant(asm1)
    y0 = plant.initial_state()
    ev = aquakin.Event(at_times=[0.5])
    with pytest.raises(ValueError, match="not supported with events="):
        plant.solve(t_span=(0.0, 1.0), t_eval=jnp.array([1.0]), y0=y0,
                    events=[ev],
                    integrator=aquakin.IntegratorConfig(solver=diffrax.Kvaerno3()))


def test_factormax_runs_and_matches_default(asm1):
    plant = _mini_plant(asm1)
    y0 = plant.initial_state()
    t_eval = jnp.linspace(0.0, 2.0, 9)
    # Reference: the old no-cap default; capped: the factormax cap.
    base = plant.solve(t_span=(0.0, 2.0), t_eval=t_eval, y0=y0,
                       integrator=aquakin.IntegratorConfig(order=5, factormax=None))
    capped = plant.solve(t_span=(0.0, 2.0), t_eval=t_eval, y0=y0,
                         integrator=aquakin.IntegratorConfig(factormax=3.0))
    assert np.all(np.isfinite(np.asarray(capped.state)))
    rel = np.max(np.abs(np.asarray(capped.state[-1]) - np.asarray(base.state[-1]))
                 / (np.abs(np.asarray(base.state[-1])) + 1e-9))
    assert rel < 1e-3


def test_factormax_keys_cache(asm1):
    plant = _mini_plant(asm1)
    y0 = plant.initial_state()
    t_eval = jnp.linspace(0.0, 1.0, 5)
    plant._jit_cache.clear()
    # Pin no-cap on the reference arm so it keys distinctly from factormax=3 (the
    # new default itself caps factormax at 3).
    plant.solve(t_span=(0.0, 1.0), t_eval=t_eval, y0=y0,
                integrator=aquakin.IntegratorConfig(order=5, factormax=None))  # no cap
    n = len(plant._jit_cache)
    plant.solve(t_span=(0.0, 1.0), t_eval=t_eval, y0=y0,
                integrator=aquakin.IntegratorConfig(factormax=3.0))   # distinct
    assert len(plant._jit_cache) == n + 1


def test_factormax_honored_with_events(asm1):
    # factormax= is threaded into the located-event segmented solve, which builds
    # the same integrator config as a plain solve -- so it runs without error and
    # the event path does not silently drift onto a different integrator. (Only
    # solver=/colored_jacobian=True remain the opt-ins the segmented solve cannot
    # honour; those stay rejected with events=.)
    plant = _mini_plant(asm1)
    y0 = plant.initial_state()
    ev = aquakin.Event(at_times=[0.5])
    sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.array([1.0]), y0=y0,
                      events=[ev],
                      integrator=aquakin.IntegratorConfig(factormax=3.0))
    assert jnp.all(jnp.isfinite(sol.state))


def test_solver_override_grad_flows(asm1):
    """jax.grad still flows through the solve with a non-default solver."""
    plant = _mini_plant(asm1)
    y0 = plant.initial_state()
    snh = asm1.species_index["SNH"]

    def loss(scale):
        sol = plant.solve(t_span=(0.0, 1.0), t_eval=jnp.array([1.0]),
                          params=plant.default_parameters() * scale, y0=y0,
                          integrator=aquakin.IntegratorConfig(solver=diffrax.Kvaerno3()),
                          diff=aquakin.DifferentiationConfig(method="through_solve"))
        return jnp.sum(sol.state[-1, snh] ** 2)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)
