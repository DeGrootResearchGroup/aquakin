"""Fake-C++-caller contract test for CFDReactor.

Mimics what an OpenFOAM ``fvOptions`` plugin (Option C, runtime coupling via
pybind11) is expected to do each timestep: assemble a ``(n_cells, n_species)``
NumPy array of cell concentrations, assemble a dict of ``(n_cells,)``
condition arrays, call :meth:`CFDReactor.step`, and feed the result back
into the next timestep.

If this test passes, the C++ side only has to match the calling convention
to obtain identical results.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture
def cfd_setup():
    """The expanded ozone/bromate network plus per-cell initial state."""
    network = aquakin.load_network("ozone_bromate")
    n_cells = 8

    # Per-species absolute tolerance: OH lives at ~1e-12 M.
    atol = np.full(network.n_species, 1e-12)
    atol[network.species_index["OH"]] = 1e-20
    reactor = aquakin.CFDReactor(network, atol=jnp.asarray(atol))

    # Initial cell state: identical inlet concentration across cells.
    C0 = np.zeros((n_cells, network.n_species), dtype=np.float64)
    C0[:, network.species_index["O3"]] = 1.0e-4
    C0[:, network.species_index["Br-"]] = 1.0e-5

    # Per-cell condition fields. Sweep OH_scavenging log-spaced across cells
    # to simulate streamlines passing through differing matrix scavenging.
    conditions = {
        "pH": np.full(n_cells, 7.5),
        "T": np.full(n_cells, 293.15),
        "OH_scavenging": np.logspace(3.0, 6.0, n_cells),
    }
    return reactor, C0, conditions


def _fake_cfd_loop(reactor, C0, conditions, *, total_time=600.0, n_steps=20):
    """Simulate ``n_steps`` transport sub-steps. Returns trajectory of shape
    ``(n_steps+1, n_cells, n_species)``."""
    dt = total_time / n_steps
    history = [C0.copy()]
    C = C0.copy()
    for _ in range(n_steps):
        C = reactor.step(C, conditions, dt)
        history.append(C.copy())
    return np.stack(history), dt


def test_full_loop_runs_and_returns_numpy(cfd_setup):
    reactor, C0, conditions = cfd_setup
    history, _ = _fake_cfd_loop(reactor, C0, conditions, total_time=300.0, n_steps=10)
    assert isinstance(history, np.ndarray)
    assert history.shape == (11, C0.shape[0], reactor.network.n_species)
    assert np.all(np.isfinite(history))


def test_jit_cache_amortises_across_timesteps(cfd_setup):
    """After warm-up the cache should remain a single entry for the loop."""
    reactor, C0, conditions = cfd_setup
    reactor.step(C0, conditions, dt=1.0)  # warm-up
    cache_size_before = len(reactor._jit_cache)
    for _ in range(10):
        reactor.step(C0, conditions, dt=2.0)
    assert len(reactor._jit_cache) == cache_size_before == 1


def test_bromide_atom_balance(cfd_setup):
    """Total Br atoms (Br- + HOBr + BrO2- + BrO3-) per cell must be conserved.

    OH is not Br-bearing, so it is excluded from the count.
    """
    reactor, C0, conditions = cfd_setup
    network = reactor.network
    history, _ = _fake_cfd_loop(reactor, C0, conditions, total_time=600.0, n_steps=12)
    br_species = ["Br-", "HOBr", "BrO2-", "BrO3-"]
    idxs = [network.species_index[s] for s in br_species]
    initial = history[0, :, idxs].sum(axis=0)
    final = history[-1, :, idxs].sum(axis=0)
    # Conservation to integration tolerance.
    assert np.allclose(final, initial, rtol=1e-4, atol=1e-12)


def test_ozone_monotone_per_cell(cfd_setup):
    """O3 decays monotonically (within solver tolerance) in every cell."""
    reactor, C0, conditions = cfd_setup
    network = reactor.network
    history, _ = _fake_cfd_loop(reactor, C0, conditions, total_time=600.0, n_steps=12)
    o3 = history[:, :, network.species_index["O3"]]
    diffs = np.diff(o3, axis=0)
    # Allow a small positive tolerance to absorb solver noise.
    assert np.all(diffs <= 1e-12)


def test_per_cell_results_differ_with_conditions(cfd_setup):
    """Cells with very different OH_scavenging must produce visibly different
    bromate yields after the simulation."""
    reactor, C0, conditions = cfd_setup
    network = reactor.network
    history, _ = _fake_cfd_loop(reactor, C0, conditions, total_time=600.0, n_steps=12)
    bro3_final = history[-1, :, network.species_index["BrO3-"]]
    # Low-scavenging cells (index 0) should yield more bromate than
    # high-scavenging cells (index -1).
    assert bro3_final[0] > bro3_final[-1]
    # And the trend should be monotonic in scavenging.
    assert np.all(np.diff(bro3_final) < 0)


def test_no_negative_concentrations(cfd_setup):
    """Final concentrations are physically non-negative (stiff solver may
    introduce sub-noise negatives; allow only that)."""
    reactor, C0, conditions = cfd_setup
    history, _ = _fake_cfd_loop(reactor, C0, conditions, total_time=600.0, n_steps=12)
    minimum = history.min()
    assert minimum >= -1e-15


def test_matches_per_cell_BatchReactor(cfd_setup):
    """One ``CFDReactor.step`` over dt should match a ``BatchReactor.solve``
    on each cell independently to integration tolerance.

    This is the core correctness check: the vmapped seam should be
    semantically identical to running ``BatchReactor`` per cell.
    """
    reactor, C0, conditions = cfd_setup
    network = reactor.network
    dt = 60.0
    out_cfd = reactor.step(C0, conditions, dt)

    out_batch = np.zeros_like(C0)
    for i in range(C0.shape[0]):
        sc = aquakin.SpatialConditions(
            fields={
                name: jnp.asarray([float(conditions[name][i])])
                for name in network.conditions_required
            }
        )
        # Same per-species atol as the CFD reactor.
        atol = np.full(network.n_species, 1e-12)
        atol[network.species_index["OH"]] = 1e-20
        br = aquakin.BatchReactor(network, sc, atol=jnp.asarray(atol))
        sol = br.solve(
            jnp.asarray(C0[i]),
            network.default_parameters(),
            t_span=(0.0, dt),
        )
        out_batch[i] = np.asarray(sol.C[-1])

    assert np.allclose(out_cfd, out_batch, rtol=1e-4, atol=1e-12)


def test_step_is_ad_clean(cfd_setup):
    """`jax.grad` through `step` should produce finite gradients w.r.t. params.

    Needed for any inverse-design / parameter-fitting workflow that goes
    through the runtime-coupled CFD path.
    """
    reactor, C0, conditions = cfd_setup
    network = reactor.network

    # Convert to JAX for the grad path; CFDReactor.step does NumPy conversion
    # internally, so we drive the underlying jit directly here.
    inner = reactor._build_step()
    cond_jax = {k: jnp.asarray(v) for k, v in conditions.items()}
    C0_jax = jnp.asarray(C0)

    def output(params):
        C_new = inner(C0_jax, cond_jax, jnp.asarray(60.0), params)
        return jnp.mean(C_new[:, network.species_index["BrO3-"]])

    g = jax.grad(output)(network.default_parameters())
    assert jnp.all(jnp.isfinite(g))
    # The bromate-formation rate constants should have positive gradient
    # contribution.
    k1_idx = network.param_index["O3_Br_direct.k1"]
    assert float(g[k1_idx]) > 0.0
