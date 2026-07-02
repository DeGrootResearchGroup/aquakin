"""Integration tests for the built-in UV/H2O2 AOP model."""

import jax.numpy as jnp
import pytest

import aquakin


@pytest.fixture
def model():
    return aquakin.load_model("uv_h2o2")


def _atol_for(model):
    return model.atol({"OH": 1e-20}, default=1e-12)


def _solve(model, *, fluence_rate, t_end=600.0, n=61):
    conditions = aquakin.SpatialConditions.uniform(
        1, fluence_rate=fluence_rate, OH_scavenging=5.0e4
    )
    reactor = aquakin.BatchReactor(model, conditions, atol=_atol_for(model))
    return reactor.solve(
        model.default_concentrations(),
        params=model.default_parameters(),
        t_span=(0.0, t_end),
        t_eval=jnp.linspace(0.0, t_end, n),
    )


def test_model_shape(model):
    assert model.n_species == 4
    assert model.n_reactions == 4
    assert set(model.conditions_required) == {"fluence_rate", "OH_scavenging"}


def test_uv_on_decays_H2O2_and_target(model):
    sol = _solve(model, fluence_rate=1.0)
    h2o2 = sol.C_named("H2O2")
    target = sol.C_named("target")
    assert float(h2o2[-1]) < float(h2o2[0])
    assert float(target[-1]) < float(target[0])
    # Monotone non-increasing within tolerance.
    assert jnp.all(jnp.diff(h2o2) <= 1e-12)
    assert jnp.all(jnp.diff(target) <= 1e-12)


def test_uv_off_holds_steady(model):
    sol = _solve(model, fluence_rate=0.0)
    # Without photolysis there is no OH source, so H2O2 and target are conserved.
    assert float(sol.C_named("H2O2")[-1]) == pytest.approx(
        float(sol.C_named("H2O2")[0]), rel=1e-8
    )
    assert float(sol.C_named("target")[-1]) == pytest.approx(
        float(sol.C_named("target")[0]), rel=1e-8
    )


def test_higher_fluence_destroys_more_target(model):
    sol_low = _solve(model, fluence_rate=0.5)
    sol_high = _solve(model, fluence_rate=5.0)
    assert float(sol_high.C_named("target")[-1]) < float(sol_low.C_named("target")[-1])
