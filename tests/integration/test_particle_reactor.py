"""Integration tests for ParticleTrackReactor."""

import jax
import jax.numpy as jnp
import pytest

import aquakin


def test_constant_track_matches_batch(simple_network):
    """Constant-condition track should match BatchReactor.solve to integration tol."""
    t_end = 10.0
    track = aquakin.Track(
        t=jnp.asarray([0.0, t_end]),
        fields={"T": jnp.asarray([293.15, 293.15])},
    )
    reactor = aquakin.ParticleTrackReactor(simple_network, track, n_save=11)
    C0 = jnp.asarray([1.0, 0.0])
    params = simple_network.default_parameters()
    sol_track = reactor.solve(C0, params)

    batch = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    sol_batch = batch.solve(
        C0, params, t_span=(0.0, t_end), t_eval=jnp.linspace(0.0, t_end, 11)
    )

    assert jnp.allclose(sol_track.C, sol_batch.C, atol=1e-7, rtol=1e-5)


def test_grad_through_particle_solve(simple_network):
    track = aquakin.Track(
        t=jnp.asarray([0.0, 5.0]),
        fields={"T": jnp.asarray([293.15, 293.15])},
    )
    reactor = aquakin.ParticleTrackReactor(simple_network, track, n_save=6)
    C0 = jnp.asarray([1.0, 0.0])

    def loss(params):
        sol = reactor.solve(C0, params)
        return jnp.sum(sol.C_named("B") ** 2)

    g = jax.grad(loss)(simple_network.default_parameters())
    assert jnp.all(jnp.isfinite(g))


def test_track_validates_ascending_time(simple_network):
    with pytest.raises(ValueError):
        aquakin.Track(
            t=jnp.asarray([0.0, 5.0, 3.0]),
            fields={"T": jnp.asarray([293.15, 293.15, 293.15])},
        )


def test_track_validates_field_length(simple_network):
    with pytest.raises(ValueError):
        aquakin.Track(
            t=jnp.asarray([0.0, 5.0]),
            fields={"T": jnp.asarray([293.15, 293.15, 293.15])},
        )


def test_missing_condition_rejected(simple_network):
    # simple_network requires 'T'; provide nothing.
    track = aquakin.Track(t=jnp.asarray([0.0, 1.0]), fields={})
    with pytest.raises(ValueError):
        aquakin.ParticleTrackReactor(simple_network, track)


def test_scavenging_step_affects_OH_path():
    """Step-up in OH_scavenging should suppress bromate yield on the back half."""
    network = aquakin.load_network("ozone_bromate")
    atol = jnp.full((network.n_species,), 1e-12)
    atol = atol.at[network.species_index["OH"]].set(1e-20)

    t = jnp.linspace(0.0, 600.0, 13)
    low = jnp.full_like(t, 1.0e3)
    high = jnp.full_like(t, 1.0e6)

    # Two tracks: low scavenging then high (more product), vs constant high.
    track_step = aquakin.Track(
        t=t,
        fields={
            "pH": jnp.full_like(t, 7.5),
            "T": jnp.full_like(t, 293.15),
            "OH_scavenging": jnp.where(t < 300.0, low, high),
        },
    )
    track_high = aquakin.Track(
        t=t,
        fields={
            "pH": jnp.full_like(t, 7.5),
            "T": jnp.full_like(t, 293.15),
            "OH_scavenging": high,
        },
    )

    C0 = network.default_concentrations()
    C0 = C0.at[network.species_index["O3"]].set(1e-4)
    C0 = C0.at[network.species_index["Br-"]].set(1e-5)

    sol_step = aquakin.ParticleTrackReactor(network, track_step, atol=atol).solve(
        C0, network.default_parameters()
    )
    sol_high = aquakin.ParticleTrackReactor(network, track_high, atol=atol).solve(
        C0, network.default_parameters()
    )

    bro3_step = float(sol_step.C_named("BrO3-")[-1])
    bro3_high = float(sol_high.C_named("BrO3-")[-1])
    assert bro3_step > bro3_high


def test_track_reactor_n_save_minimum(simple_network):
    track = aquakin.Track(
        t=jnp.asarray([0.0, 1.0]),
        fields={"T": jnp.asarray([293.15, 293.15])},
    )
    with pytest.raises(ValueError):
        aquakin.ParticleTrackReactor(simple_network, track, n_save=1)


def test_integrate_ensemble_distinct_tracks(simple_network):
    tracks = {
        0: aquakin.Track(t=jnp.asarray([0.0, 5.0]), fields={"T": jnp.asarray([293.15, 293.15])}),
        1: aquakin.Track(t=jnp.asarray([0.0, 10.0]), fields={"T": jnp.asarray([293.15, 293.15])}),
        2: aquakin.Track(t=jnp.asarray([0.0, 20.0]), fields={"T": jnp.asarray([293.15, 293.15])}),
    }
    results = aquakin.integrate_ensemble(
        simple_network,
        tracks,
        C0_fn=lambda pid: jnp.asarray([1.0, 0.0]),
        params=simple_network.default_parameters(),
        n_save=3,
    )
    assert set(results.keys()) == {0, 1, 2}
    # Later end times -> more conversion of A -> B.
    final_B = [float(results[i].C_named("B")[-1]) for i in (0, 1, 2)]
    assert final_B[0] < final_B[1] < final_B[2]


def test_particle_direct_adjoint_enables_forward_mode(simple_network):
    """ParticleTrackReactor now accepts adjoint=; DirectAdjoint makes the track
    solve forward-mode differentiable (jacfwd)."""
    import diffrax

    track = aquakin.Track(t=jnp.linspace(0.0, 10.0, 11),
                          fields={"T": jnp.full(11, 293.15)})
    reactor = aquakin.ParticleTrackReactor(
        simple_network, track, adjoint=diffrax.DirectAdjoint())
    C0 = jnp.asarray([1.0, 0.0])

    def out(p):
        return jnp.sum(reactor.solve(C0, p).C)

    J = jax.jacfwd(out)(simple_network.default_parameters())
    assert jnp.all(jnp.isfinite(J))


def test_integrate_ensemble_forwards_adjoint_and_dtmax(simple_network):
    """integrate_ensemble forwards adjoint/dtmax to each per-track reactor."""
    import diffrax
    from aquakin.integrate.particle import integrate_ensemble

    tracks = {
        i: aquakin.Track(t=jnp.linspace(0.0, 10.0, 11),
                         fields={"T": jnp.full(11, 293.15)})
        for i in range(2)
    }
    res = integrate_ensemble(
        simple_network, tracks, lambda pid: jnp.asarray([1.0, 0.0]),
        simple_network.default_parameters(),
        adjoint=diffrax.DirectAdjoint(), dtmax=1.0,
    )
    assert set(res) == {0, 1}
    for sol in res.values():
        assert jnp.all(jnp.isfinite(sol.C))
