"""Sensitivity and fit smoke tests."""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


def test_autodiff_matches_finite_difference(simple_network):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_network, conditions)
    C0 = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()

    t_eval = jnp.linspace(0.0, 10.0, 11)

    result = aquakin.sensitivity(
        reactor,
        C0,
        params,
        output_fn=lambda sol: sol.C_named("B")[-1],
        solve_kwargs={"t_span": (0.0, 10.0), "t_eval": t_eval},
    )

    def _eval_k(k_val):
        p = params.at[0].set(k_val)
        sol = reactor.solve(C0, p, t_span=(0.0, 10.0), t_eval=t_eval)
        return float(sol.C_named("B")[-1])

    h = 1e-4
    fd = (_eval_k(float(params[0]) + h) - _eval_k(float(params[0]) - h)) / (2 * h)
    ad = float(result.doutput_dparams[0])
    assert ad == pytest.approx(fd, rel=1e-3, abs=1e-6)


def test_ranked_params(simple_network):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_network, conditions)
    result = aquakin.sensitivity(
        reactor,
        jnp.asarray([1.0, 0.0]),
        simple_network.default_parameters(),
        output_fn=lambda sol: sol.C_named("B")[-1],
        solve_kwargs={"t_span": (0.0, 5.0), "t_eval": jnp.linspace(0.0, 5.0, 6)},
    )
    ranked = result.ranked_params()
    assert ranked[0][0] == "A_to_B.k"


def test_sensitivity_doutput_dconditions_matches_finite_diff(simple_network):
    """Verify the conditions-override grad path is non-zero where expected."""
    # Build a tiny network rate that depends on T via Arrhenius so dF/dT != 0.
    # The shipped simple_network's rate is k*[A] only — independent of T —
    # so we exercise the override path with the ozone_bromate network instead.
    network = aquakin.load_network("ozone_bromate")
    atol = jnp.full((network.n_species,), 1e-12)
    atol = atol.at[network.species_index["OH"]].set(1e-20)
    conditions = aquakin.SpatialConditions.uniform(
        1, pH=7.5, T=293.15, OH_scavenging=5.0e4
    )
    reactor = aquakin.BatchReactor(network, conditions, atol=atol)
    C0 = network.default_concentrations()
    C0 = C0.at[network.species_index["O3"]].set(1.0e-4)
    C0 = C0.at[network.species_index["Br-"]].set(1.0e-5)
    params = network.default_parameters()
    t_eval = jnp.linspace(0.0, 300.0, 31)

    result = aquakin.sensitivity(
        reactor,
        C0,
        params,
        output_fn=lambda sol: sol.C_named("BrO3-")[-1],
        solve_kwargs={"t_span": (0.0, 300.0), "t_eval": t_eval},
    )

    # The OH scavenging field should have a *negative* gradient on bromate.
    g_scav = float(result.doutput_dconditions["OH_scavenging"][0])
    assert g_scav < 0.0
    # All gradients should be finite.
    for name, g in result.doutput_dconditions.items():
        assert jnp.all(jnp.isfinite(g)), f"non-finite grad for {name}"


def test_fit_rejects_empty_free_params(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    with pytest.raises(ValueError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.0, 0.5]),
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=[],
        )


def test_fit_rejects_unknown_method(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    with pytest.raises(ValueError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.0, 0.5]),
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=["A_to_B.k"],
            method="newton",
        )


def test_fit_rejects_unknown_param(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    with pytest.raises(KeyError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.0, 0.5]),
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=["nonexistent.k"],
        )


def test_fit_rejects_observation_shape_mismatch(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    with pytest.raises(ValueError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.zeros((2, 3)),  # 3 cols but only 1 observed_species
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=["A_to_B.k"],
            observed_species=["B"],
        )


def test_fit_rejects_descending_t_obs(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    with pytest.raises(ValueError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.5, 0.0]),
            t_obs=jnp.asarray([1.0, 0.0]),
            free_params=["A_to_B.k"],
            observed_species=["B"],
        )


def test_fit_recovers_known_rate(simple_network):
    """Generate synthetic observations from a known k, then re-fit."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_network, conditions)
    C0 = jnp.asarray([1.0, 0.0])

    true_k = 0.25
    true_params = simple_network.default_parameters().at[0].set(true_k)
    t_obs = jnp.linspace(0.5, 10.0, 20)
    sol = reactor.solve(C0, true_params, t_span=(0.0, 10.0), t_eval=t_obs)
    obs = sol.C_named("B")

    # Start from a wrong initial guess (the network default, k = 0.1).
    result = aquakin.fit(
        reactor,
        C0,
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        observed_species=["B"],
    )

    assert result.converged
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)
