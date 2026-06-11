"""Integration tests for the forward (variational) sensitivity solver.

``reactor.solve_sensitivity`` integrates ``S = dC/dtheta`` alongside the state,
with adaptive control over both, so the sensitivity is exact and finite without
the ``dtmax`` cap that ordinary AD through a stiff solve needs. The fast tests
here check exactness against a closed-form sensitivity and against ``jax.jacfwd``
on small networks; the slow ``validation``-marked tests check the headline claim
on a genuinely stiff network (finite where uncapped AD is non-finite) and that
the JVP flows through a state-derived-pH speciation solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diffrax

import aquakin

# Slow module: forward-sensitivity solves through stiff networks. Excluded from
# the fast PR gate; runs in the merge-to-main suite (see the ``slow`` marker).
pytestmark = pytest.mark.slow


# --- Fast exactness / API tests -----------------------------------------


def test_analytic_decay_sensitivity_exact(simple_network):
    # First-order decay A -> B: A(t) = A0 e^{-kt}, so dA/dk = -t A0 e^{-kt} and
    # dB/dk = +t A0 e^{-kt} in closed form.
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    reactor = aquakin.BatchReactor(simple_network, cond, rtol=1e-11, atol=1e-13)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_network.default_parameters()
    k = float(p[0])
    t_eval = jnp.linspace(0.0, 20.0, 11)

    sol, S = reactor.solve_sensitivity(
        C0, p, (0.0, 20.0), t_eval, sens_params=["A_to_B.k"]
    )
    assert S.shape == (11, 2, 1)
    dA = -t_eval * jnp.exp(-k * t_eval)
    dB = t_eval * jnp.exp(-k * t_eval)
    assert float(jnp.max(jnp.abs(S[:, 0, 0] - dA))) < 1e-7
    assert float(jnp.max(jnp.abs(S[:, 1, 0] - dB))) < 1e-7
    # The primal state is the usual solution, uncapped.
    assert float(jnp.max(jnp.abs(sol.C_named("A") - jnp.exp(-k * t_eval)))) < 1e-6


def test_matches_jacfwd_multi_param():
    # On a small non-stiff network the augmented solve must reproduce jax.jacfwd
    # for several parameters at once.
    net = aquakin.load_network("uv_h2o2")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    names = list(net.parameters)
    t_eval = jnp.linspace(0.0, 5.0, 6)

    ref = aquakin.BatchReactor(net, cond, adjoint=diffrax.DirectAdjoint())

    def Cfn(pp):
        return ref.solve(C0, pp, (0.0, 5.0), t_eval).C

    J = jax.jacfwd(Cfn)(p)  # (n_t, n_species, n_params)

    r = aquakin.BatchReactor(net, cond)
    _, S = r.solve_sensitivity(C0, p, (0.0, 5.0), t_eval, sens_params=names)
    assert S.shape == J.shape
    assert float(jnp.max(jnp.abs(S - J))) < 1e-7


def test_sensitivity_array_is_finite_and_composes(simple_network):
    # AD-correctness analogue: the returned S is a real, finite JAX array that
    # composes in further computation (the project requires every solve path to
    # produce finite derivatives).
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    reactor = aquakin.BatchReactor(simple_network, cond)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_network.default_parameters()
    _, S = reactor.solve_sensitivity(
        C0, p, (0.0, 10.0), jnp.linspace(0.0, 10.0, 6), sens_params=["A_to_B.k"]
    )
    assert isinstance(S, jnp.ndarray)
    assert jnp.all(jnp.isfinite(S))
    # composes: contract with an arbitrary weight vector without error
    w = jnp.ones(S.shape[1])
    contracted = jnp.einsum("tsk,s->tk", S, w)
    assert jnp.all(jnp.isfinite(contracted))


def test_int_indices_equivalent_to_names(simple_network):
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    reactor = aquakin.BatchReactor(simple_network, cond)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_network.default_parameters()
    t_eval = jnp.linspace(0.0, 10.0, 6)
    _, S_name = reactor.solve_sensitivity(
        C0, p, (0.0, 10.0), t_eval, sens_params=["A_to_B.k"]
    )
    _, S_idx = reactor.solve_sensitivity(
        C0, p, (0.0, 10.0), t_eval, sens_params=[0]
    )
    assert jnp.allclose(S_name, S_idx)


def test_biofilm_sensitivity_matches_jacfwd(simple_network):
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    kw = dict(
        n_layers=4, thickness=8e-4, area_per_volume=50.0,
        diffusivity=1e-4, boundary_layer=1e-4, rtol=1e-9, atol=1e-11,
    )
    n = simple_network.n_species
    C0 = jnp.zeros((5, n)).at[0, simple_network.species_index["A"]].set(1.0)
    p = simple_network.default_parameters()
    t_eval = jnp.linspace(0.0, 5.0, 6)

    ref = aquakin.BiofilmReactor(
        simple_network, cond, adjoint=diffrax.DirectAdjoint(), **kw
    )

    def bulkC(pp):
        return ref.solve(C0, pp, (0.0, 5.0), t_eval).C  # bulk trajectory

    J = jax.jacfwd(bulkC)(p)[:, :, 0]

    r = aquakin.BiofilmReactor(simple_network, cond, **kw)
    sol, S = r.solve_sensitivity(
        C0, p, (0.0, 5.0), t_eval, sens_params=["A_to_B.k"]
    )
    assert S.shape == (6, n, 1)
    assert sol.profile.shape == (6, 5, n)
    assert float(jnp.max(jnp.abs(S[:, :, 0] - J))) < 1e-6


def test_pfr_sensitivity_matches_jacfwd(simple_network):
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_network.default_parameters()

    ref = aquakin.PlugFlowReactor(
        simple_network, cond, n_points=6, length=10.0, velocity=1.0,
        rtol=1e-10, atol=1e-12,
    )
    # The PFR uses the default (reverse-mode) adjoint, so the reference Jacobian
    # is taken with jacrev; the augmented forward solve must reproduce it.
    def Cfn(pp):
        return ref.solve(C0, pp).C

    J = jax.jacrev(Cfn)(p)[:, :, 0]
    _, S = ref.solve_sensitivity(C0, p, sens_params=["A_to_B.k"])
    assert S.shape == (6, 2, 1)
    assert float(jnp.max(jnp.abs(S[:, :, 0] - J))) < 1e-6


def test_free_function_and_accessors():
    net = aquakin.load_network("uv_h2o2")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    reactor = aquakin.BatchReactor(net, cond)
    names = list(net.parameters)
    t_eval = jnp.linspace(0.0, 5.0, 6)

    res = aquakin.forward_sensitivity(
        reactor, C0, p, sens_params=names, t_span=(0.0, 5.0), t_eval=t_eval
    )
    assert res.sens_params == names
    assert res.S.shape == (6, net.n_species, len(names))
    sp = net.species[0]
    assert res.S_named(sp).shape == (6, len(names))
    assert res.dC_dparam(sp, names[0]).shape == (6,)
    assert jnp.allclose(res.dC_dparam(sp, names[0]), res.S[:, 0, 0])
    # solution is the usual trajectory
    assert res.solution.C.shape == (6, net.n_species)


def test_shared_factor_matches_dense_single_param(simple_network):
    # The simultaneous corrector solves the same Newton system as the dense
    # path, so shared_factor=True must reproduce shared_factor=False exactly.
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    reactor = aquakin.BatchReactor(simple_network, cond, rtol=1e-11, atol=1e-13)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_network.default_parameters()
    k = float(p[0])
    t_eval = jnp.linspace(0.0, 20.0, 11)
    _, S_dense = reactor.solve_sensitivity(
        C0, p, (0.0, 20.0), t_eval, sens_params=["A_to_B.k"], shared_factor=False
    )
    _, S_shared = reactor.solve_sensitivity(
        C0, p, (0.0, 20.0), t_eval, sens_params=["A_to_B.k"], shared_factor=True
    )
    assert float(jnp.max(jnp.abs(S_dense - S_shared))) < 1e-12
    # and still exact against the closed form
    dA = -t_eval * jnp.exp(-k * t_eval)
    assert float(jnp.max(jnp.abs(S_shared[:, 0, 0] - dA))) < 1e-7


def test_shared_factor_matches_dense_multi_param():
    # Several parameters: the block-arrow forward substitution must match the
    # dense augmented solve to machine precision.
    net = aquakin.load_network("uv_h2o2")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    names = list(net.parameters)
    t_eval = jnp.linspace(0.0, 5.0, 6)
    r = aquakin.BatchReactor(net, cond)
    _, S_dense = r.solve_sensitivity(
        C0, p, (0.0, 5.0), t_eval, sens_params=names, shared_factor=False
    )
    _, S_shared = r.solve_sensitivity(
        C0, p, (0.0, 5.0), t_eval, sens_params=names, shared_factor=True
    )
    assert S_dense.shape == S_shared.shape == (6, net.n_species, len(names))
    assert float(jnp.max(jnp.abs(S_dense - S_shared))) < 1e-12


def test_simultaneous_corrector_solver_matches_dense_lu():
    # Unit test of the custom linear solver in isolation: on a hand-built
    # block-arrow operator M (identical diagonal D, S-blocks coupling only to y),
    # the simultaneous-corrector solve must equal a dense solve of M.
    import lineax as lx

    from aquakin.integrate._simultaneous_corrector import SimultaneousCorrector

    key = jax.random.PRNGKey(0)
    n, k = 4, 3
    kd, *kl = jax.random.split(key, 1 + k)
    D = jnp.eye(n) + 0.1 * jax.random.normal(kd, (n, n))   # well-conditioned
    Ls = [0.1 * jax.random.normal(kl[j], (n, n)) for j in range(k)]
    N = n * (1 + k)
    M = jnp.zeros((N, N))
    # diagonal blocks = D
    for b in range(1 + k):
        M = M.at[b * n:(b + 1) * n, b * n:(b + 1) * n].set(D)
    # off-diagonal: S_j block (row b=j+1) couples to y block (col 0) via L_j
    for j in range(k):
        M = M.at[(j + 1) * n:(j + 2) * n, 0:n].set(Ls[j])

    rng_b = jax.random.normal(jax.random.PRNGKey(1), (N,))
    operator = lx.MatrixLinearOperator(M)
    solver = SimultaneousCorrector(ndof=n, n_sens=k)
    # Forward solve M x = b via lineax (calls init then compute).
    sol = lx.linear_solve(operator, rng_b, solver)
    x_ref = jnp.linalg.solve(M, rng_b)
    assert float(jnp.max(jnp.abs(sol.value - x_ref))) < 1e-9
    # Transpose method: init -> transpose -> compute must solve M^T x = b.
    state = solver.init(operator, {})
    state_T, _ = solver.transpose(state, {})
    xT, result, _ = solver.compute(state_T, rng_b, {})
    xT_ref = jnp.linalg.solve(M.T, rng_b)
    assert float(jnp.max(jnp.abs(xT - xT_ref))) < 1e-9


def test_bad_sens_params_raise(simple_network):
    cond = aquakin.SpatialConditions.uniform(T=293.15)
    reactor = aquakin.BatchReactor(simple_network, cond)
    C0 = jnp.asarray([1.0, 0.0])
    p = simple_network.default_parameters()
    with pytest.raises(KeyError):
        reactor.solve_sensitivity(C0, p, (0.0, 5.0), sens_params=["nope.k"])
    with pytest.raises(ValueError):
        reactor.solve_sensitivity(C0, p, (0.0, 5.0), sens_params=[])
    with pytest.raises(IndexError):
        reactor.solve_sensitivity(C0, p, (0.0, 5.0), sens_params=[99])


# --- Slow validation tests (stiff network; the headline claim) ----------


@pytest.mark.validation
def test_stiff_uncapped_finite_and_matches_capped_jacfwd():
    # The canonical stiff sewer-biofilm network. Uncapped forward-mode AD through
    # the solve is non-finite (the failure the dtmax cap exists to avoid), yet the
    # augmented forward-sensitivity solve -- uncapped -- is finite and matches a
    # tightly-capped jacfwd. This network also carries a positivity limiter, so
    # the limiter is part of f and the sensitivity sees it.
    net = aquakin.load_network("wats_sewer_khalil_paper_balanced")
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    names = ["mu_h", "q_m"]
    idx = [net.param_index[n] for n in names]
    t_eval = jnp.array([0.0, 0.05, 0.1])

    # Uncapped forward-mode AD: expected to be non-finite (raises or NaNs).
    uncapped = aquakin.BatchReactor(
        net, cond, adjoint=diffrax.DirectAdjoint(), dtmax=None
    )

    def Cfn_uncapped(pp):
        return uncapped.solve(C0, pp, (0.0, 0.1), t_eval).C

    uncapped_bad = False
    try:
        J_unc = jax.jacfwd(Cfn_uncapped)(p)[:, :, idx]
        uncapped_bad = not bool(jnp.all(jnp.isfinite(J_unc)))
    except Exception:
        uncapped_bad = True
    assert uncapped_bad, "expected uncapped jacfwd to be non-finite on the stiff net"

    # Capped jacfwd reference (finite, the current workaround).
    capped = aquakin.BatchReactor(
        net, cond, adjoint=diffrax.DirectAdjoint(), dtmax=3e-3
    )

    def Cfn_capped(pp):
        return capped.solve(C0, pp, (0.0, 0.1), t_eval).C

    J_cap = jax.jacfwd(Cfn_capped)(p)[:, :, idx]
    assert bool(jnp.all(jnp.isfinite(J_cap)))

    # Augmented forward sensitivity, uncapped -- finite and matching.
    r = aquakin.BatchReactor(net, cond, dtmax=None)
    _, S = r.solve_sensitivity(C0, p, (0.0, 0.1), t_eval, sens_params=names)
    assert bool(jnp.all(jnp.isfinite(S)))
    assert float(jnp.max(jnp.abs(S - J_cap))) < 1e-6


@pytest.mark.validation
def test_shared_factor_biofilm_matches_dense():
    # On the stiff layered biofilm -- the regime the simultaneous corrector is
    # built for (ndof = n_comp * n_species is large, so sharing the diagonal-block
    # factorization across the sensitivity columns is the point) -- it must
    # reproduce the dense augmented solve to machine precision.
    net = aquakin.load_network("wats_sewer_khalil_paper_balanced")
    cond = net.default_conditions(1)
    n = net.n_species
    soluble = jnp.asarray([not s.startswith("X") for s in net.species])
    fixed = jnp.asarray([s == "X_I" for s in net.species])
    r = aquakin.BiofilmReactor(
        net, cond, n_layers=3, thickness=8e-4, area_per_volume=50.0,
        diffusivity=1e-4, boundary_layer=1e-4,
        soluble_mask=soluble, fixed_mask=fixed,
    )
    C0 = net.default_concentrations()
    p = net.default_parameters()
    names = ["mu_h", "q_m", "k_h2"]
    t_eval = jnp.array([0.0, 0.05])
    _, S_dense = r.solve_sensitivity(
        C0, p, (0.0, 0.05), t_eval, sens_params=names, shared_factor=False
    )
    _, S_shared = r.solve_sensitivity(
        C0, p, (0.0, 0.05), t_eval, sens_params=names, shared_factor=True
    )
    assert bool(jnp.all(jnp.isfinite(S_shared)))
    assert S_shared.shape == (2, n, 3)
    assert float(jnp.max(jnp.abs(S_dense - S_shared))) < 1e-8


@pytest.mark.validation
def test_state_derived_ph_jvp_flows_and_matches():
    # The WATS sewer network derives pH from the instantaneous state through a
    # charge-balance speciation solver. The JVP must flow through that solver, so
    # the augmented sensitivity matches a capped jacfwd with no special-casing.
    net = aquakin.load_network("wats_sewer")
    assert net.derived_fields == ["pH"]
    cond = net.default_conditions(1)
    C0 = net.default_concentrations()
    p = net.default_parameters()
    name = net.parameters[0]
    idx = [net.param_index[name]]
    t_eval = jnp.array([0.0, 0.05, 0.1])

    capped = aquakin.BatchReactor(
        net, cond, adjoint=diffrax.DirectAdjoint(), dtmax=1e-3
    )

    def Cfn(pp):
        return capped.solve(C0, pp, (0.0, 0.1), t_eval).C

    J = jax.jacfwd(Cfn)(p)[:, :, idx]
    r = aquakin.BatchReactor(net, cond)
    _, S = r.solve_sensitivity(C0, p, (0.0, 0.1), t_eval, sens_params=[name])
    assert bool(jnp.all(jnp.isfinite(S)))
    # Absolute agreement; scale-robust since some species barely move.
    assert float(jnp.max(jnp.abs(S - J))) < 1e-6
