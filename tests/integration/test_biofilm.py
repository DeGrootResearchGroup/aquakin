"""Integration tests for the layered biofilm reactor (diffusion--reaction).

Uses only the toy two-species decay fixture (A -> B). The biofilm reactor adds
1-D diffusion and a bulk boundary layer on top of the per-compartment chemistry,
so these check the transport operator (conservation, the well-mixed limit, and
diffusion limitation) and that gradients flow through the layered solve.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

# Slow module: stiff biofilm diffusion-reaction solves. Excluded from the fast
# PR gate; runs in the merge-to-main suite (see the ``slow`` marker).
pytestmark = pytest.mark.slow

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "simple_network.yaml"
)


@pytest.fixture(scope="module")
def net():
    return aquakin.load_network_from_file(FIXTURE)


@pytest.fixture
def cond():
    return aquakin.SpatialConditions.uniform(T=293.15)


def _reactor(net, cond, **kw):
    defaults = dict(
        n_layers=6, thickness=8e-4, area_per_volume=50.0,
        diffusivity=1e-4, boundary_layer=1e-4,
    )
    defaults.update(kw)
    return aquakin.BiofilmReactor(net, cond, **defaults)


def test_pure_diffusion_conserves_and_equilibrates(net, cond):
    # With the reaction switched off (k=0), diffusion alone must conserve the
    # volume-weighted total and relax every compartment to a common value.
    r = _reactor(net, cond, diffusivity=1e-3)
    p = net.parameter_values({"A_to_B.k": 0.0})
    n_comp = r.n_layers + 1
    # bulk full of A, biofilm empty.
    y0 = jnp.zeros((n_comp, net.n_species))
    y0 = y0.at[0, net.species_index["A"]].set(1.0)
    sol = r.solve(y0, params=p, t_span=(0.0, 2.0))
    final = np.asarray(sol.profile[-1])  # (n_comp, n_species)
    a = final[:, net.species_index["A"]]
    # all compartments equal at steady state
    assert np.ptp(a) < 1e-4
    # volume-weighted A conserved: bulk weight 1/A_V, each layer weight dz
    dz = r.thickness / r.n_layers
    w = np.array([1.0 / r.area_per_volume] + [dz] * r.n_layers)
    total0 = (1.0 / r.area_per_volume) * 1.0
    total1 = float(np.sum(w * a))
    assert total1 == pytest.approx(total0, rel=1e-4)


def test_boundary_diffusivity_sets_transfer_and_defaults(net, cond):
    # The external boundary layer is liquid: its mass-transfer coefficient k_L
    # must come from the supplied (free-water) boundary diffusivity, not the
    # density-reduced in-biofilm value. Passing it equal to ``diffusivity``
    # reproduces the default (None) behaviour; a larger value speeds bulk<->film
    # exchange.
    d_film = 1e-4
    r_default = _reactor(net, cond, diffusivity=d_film)
    r_equal = _reactor(net, cond, diffusivity=d_film, boundary_diffusivity=d_film)
    r_fast = _reactor(net, cond, diffusivity=d_film, boundary_diffusivity=10 * d_film)
    # k_L = D_bl / boundary_layer, applied to solubles only.
    sidx = net.species_index["A"]  # A is soluble in the toy network
    assert float(r_default._kL[sidx]) == pytest.approx(float(r_equal._kL[sidx]))
    assert float(r_fast._kL[sidx]) == pytest.approx(10 * float(r_default._kL[sidx]))

    # A faster boundary layer equilibrates the bulk with the film sooner. Bulk
    # full, film empty, reaction off: the bulk drains faster with the larger k_L.
    p = net.parameter_values({"A_to_B.k": 0.0})
    n_comp = r_default.n_layers + 1
    y0 = jnp.zeros((n_comp, net.n_species)).at[0, sidx].set(1.0)
    bulk_default = float(r_default.solve(y0, params=p, t_span=(0.0, 0.5)).C_named("A")[-1])
    bulk_fast = float(r_fast.solve(y0, params=p, t_span=(0.0, 0.5)).C_named("A")[-1])
    assert bulk_fast < bulk_default


def test_reactive_particulate_evolves_and_conserves(net, cond):
    # A reactive particulate does not diffuse (soluble_mask False) but must still
    # react (fixed_mask False) -- the two roles are decoupled. Freezing such a
    # species (the old default) turns it into a non-depleting source/sink and
    # breaks mass balance. Here B is made a non-diffusing reactive product of
    # A -> B; with B reactive it accumulates, frozen it cannot.
    ia, ib = net.species_index["A"], net.species_index["B"]
    soluble = jnp.array([True, False])  # A diffuses, B does not
    p = net.default_parameters()  # A_to_B.k > 0
    n_comp = cond_layers = 5
    y0 = jnp.zeros((n_comp, net.n_species)).at[:, ia].set(1.0)

    reactive = _reactor(net, cond, n_layers=n_comp - 1,
                        soluble_mask=soluble, fixed_mask=jnp.array([False, False]))
    frozen = _reactor(net, cond, n_layers=n_comp - 1,
                      soluble_mask=soluble, fixed_mask=jnp.array([False, True]))
    sr = reactive.solve(y0, params=p, t_span=(0.0, 1.0))
    sf = frozen.solve(y0, params=p, t_span=(0.0, 1.0))
    # Reactive B grows from zero; frozen B stays put.
    assert float(sr.profile[-1, 0, ib]) > 0.05
    assert float(sf.profile[-1, 0, ib]) == pytest.approx(0.0, abs=1e-12)
    # A -> B is 1:1, so the volume-weighted total A+B is conserved when B reacts.
    dz = reactive.thickness / reactive.n_layers
    w = np.array([1.0 / reactive.area_per_volume] + [dz] * reactive.n_layers)
    tot0 = float(np.sum(w * np.asarray(y0[:, ia] + y0[:, ib])))  # from the IC
    tot1 = float(np.sum(w * (np.asarray(sr.profile[-1, :, ia])
                             + np.asarray(sr.profile[-1, :, ib]))))
    assert tot1 == pytest.approx(tot0, rel=1e-5)


def test_default_fixed_mask_warns_on_reactive_particulate(net, cond):
    # The toy network A -> B; mark B a (reactive) particulate via soluble_mask.
    # The DEFAULT fixed_mask freezes every particulate, which would silently turn
    # the reactive B into a non-depleting sink -> the reactor must warn. An
    # explicit fixed_mask is a deliberate choice and must NOT warn.
    #
    # B is ONLY produced (A -> B, B never consumed) -- the same stoichiometric
    # shape as a precipitation sink like X_FeS. The detector must use "any nonzero
    # stoichiometry", NOT "appears with both signs": a both-signs test would miss
    # B (and FeS) and fail to warn about the exact mass-balance break this guards.
    # This test therefore locks the any-nonzero choice in.
    soluble = jnp.array([True, False])  # A diffuses, B is particulate
    with pytest.warns(UserWarning, match="reactive particulate"):
        _reactor(net, cond, soluble_mask=soluble)  # fixed_mask defaulted
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning would raise
        _reactor(net, cond, soluble_mask=soluble,
                 fixed_mask=jnp.array([False, False]))  # explicit -> no warning


def test_solve_validates_c0_and_params_shape(net, cond):
    # The shared _coerce_y0 / _check_params helpers must reject a wrong-shaped
    # C0 or params uniformly across the solve paths.
    r = _reactor(net, cond)
    p = net.default_parameters()
    bad_C0 = jnp.zeros(net.n_species + 1)
    with pytest.raises(ValueError, match="C0 has shape"):
        r.solve(bad_C0, params=p, t_span=(0.0, 1.0))
    with pytest.raises(ValueError, match="params has shape"):
        r.solve(jnp.zeros(net.n_species), params=jnp.zeros(net.n_params + 1), t_span=(0.0, 1.0))
    # Both the (n,) and (n_layers+1, n) C0 shapes are accepted.
    n_comp = r.n_layers + 1
    sol_flat = r.solve(jnp.zeros(net.n_species), params=p, t_span=(0.0, 0.5))
    sol_prof = r.solve(jnp.zeros((n_comp, net.n_species)), params=p, t_span=(0.0, 0.5))
    assert sol_flat.profile.shape[1:] == (n_comp, net.n_species)
    assert sol_prof.profile.shape[1:] == (n_comp, net.n_species)


def test_solve_validates_t_eval(net, cond):
    # Diffrax silently returns NaN for save times outside the span; the solve
    # must reject them up front (same guarantee as BatchReactor) rather than
    # produce a silently-corrupt BiofilmSolution.
    r = _reactor(net, cond)
    C0 = jnp.array([1.0, 0.0])
    p = net.default_parameters()
    with pytest.raises(ValueError, match="within t_span"):
        r.solve(C0, params=p, t_span=(0.0, 1.0), t_eval=jnp.array([0.0, 2.0]))
    with pytest.raises(ValueError, match="ascending"):
        r.solve(C0, params=p, t_span=(0.0, 1.0), t_eval=jnp.array([0.5, 0.2]))
    # solve_sensitivity guards too.
    with pytest.raises(ValueError, match="within t_span"):
        r.solve_sensitivity(
            C0, params=p, t_span=(0.0, 1.0), t_eval=jnp.array([0.0, 5.0]),
            sens_params=[0],
        )


def test_max_steps_is_enforced(net, cond):
    # ``max_steps`` is a construction-time attribute threaded into BOTH solve
    # paths (with and without t_eval). A budget far too small for the adaptive
    # solve makes it fail; the generous default completes finite. (This also
    # guards the no-t_eval path, where the knob was once silently dropped.)
    C0 = jnp.array([1.0, 0.0])
    p = net.default_parameters()
    assert _reactor(
        net, cond,
        integrator=aquakin.IntegratorConfig(max_steps=7)).max_steps == 7

    r_tight = _reactor(net, cond, integrator=aquakin.IntegratorConfig(max_steps=4))
    capped = False
    try:
        sol = r_tight.solve(C0, params=p, t_span=(0.0, 5.0))     # no t_eval path
        capped = not np.all(np.isfinite(np.asarray(sol.C)))
    except Exception:
        capped = True
    assert capped

    sol = _reactor(
        net, cond,
        integrator=aquakin.IntegratorConfig(max_steps=100_000)
    ).solve(C0, params=p, t_span=(0.0, 5.0))
    assert np.all(np.isfinite(np.asarray(sol.C)))


def test_attachment_detachment_conserve_particulate(net, cond):
    # Attachment (bulk -> surface) and detachment (layers -> bulk) are pure
    # transport: with the reaction off they must conserve the volume-weighted
    # total of the particulate they move. B is the particulate here.
    ib = net.species_index["B"]
    soluble = jnp.array([True, False])      # A diffuses, B is particulate
    pmask = jnp.array([False, True])        # move/transport B
    p = net.parameter_values({"A_to_B.k": 0.0})
    n_comp = 1 + 5

    def total_B(sol):
        dz = r.thickness / r.n_layers
        w = np.array([1.0 / r.area_per_volume] + [dz] * r.n_layers)
        return float(np.sum(w * np.asarray(sol.profile[-1, :, ib])))

    for kw in (dict(k_att=2.0, attach_mask=pmask),
               dict(k_det=2.0, detach_mask=pmask)):
        r = _reactor(net, cond, n_layers=5, soluble_mask=soluble,
                     fixed_mask=jnp.array([False, False]), **kw)
        # start with B only in the bulk (attachment) or only in the film (detach)
        y0 = jnp.zeros((n_comp, net.n_species))
        if "k_att" in kw:
            y0 = y0.at[0, ib].set(1.0)
        else:
            y0 = y0.at[1:, ib].set(1.0)
        sol = r.solve(y0, params=p, t_span=(0.0, 0.5))
        tot0 = (1.0 / r.area_per_volume) * 1.0 if "k_att" in kw else \
            (r.thickness / r.n_layers) * r.n_layers * 1.0
        assert total_B(sol) == pytest.approx(tot0, rel=1e-5)
        # and transport actually happened: the source compartment lost B (the bulk
        # for attachment, the layers for detachment -- the destination's *concen-
        # tration* is volume-diluted, so check the source, which is unambiguous).
        if "k_att" in kw:
            assert float(np.asarray(sol.profile[-1, 0, ib])) < 0.95   # bulk drained
        else:
            assert float(np.asarray(sol.profile[-1, 1:, ib]).max()) < 0.95  # film drained


def test_density_cap_throttles_growth(net, cond):
    # With a max-density cap, an autocatalytic A->B growth (B feeds back) cannot
    # push the capped species past the packing limit. Here cap B at rho with
    # packing 1.0 so the ceiling is rho; reaction A->B with B uncapped would run
    # to A_0; capped, B saturates at rho.
    ib = net.species_index["B"]
    rho = 0.3
    maxd = jnp.array([jnp.inf, rho])        # cap B only
    r = _reactor(net, cond, n_layers=3, max_density=maxd, packing_fraction=1.0)
    C0 = jnp.array([1.0, 0.0])              # all A; A->B
    sol = r.solve(C0, params=net.default_parameters(), t_span=(0.0, 50.0))
    b_final = np.asarray(sol.profile[-1, :, ib])
    assert np.all(b_final <= rho + 1e-6)    # never exceeds the cap
    assert np.max(b_final) > 0.5 * rho      # but does fill toward it


def test_well_mixed_limit_matches_batch(net, cond):
    # Started uniform with a uniform reaction, every compartment stays identical
    # and transport is zero, so the bulk must follow the batch first-order decay.
    r = _reactor(net, cond)
    batch = aquakin.BatchReactor(net, cond)
    C0 = jnp.array([1.0, 0.0])  # A, B
    p = net.default_parameters()
    t_eval = jnp.linspace(0.0, 5.0, 11)
    bio = r.solve(C0, params=p, t_span=(0.0, 5.0), t_eval=t_eval)  # uniform IC broadcast
    bat = batch.solve(C0, params=p, t_span=(0.0, 5.0), t_eval=t_eval)
    assert np.allclose(np.asarray(bio.C_named("A")), np.asarray(bat.C_named("A")),
                       rtol=1e-5, atol=1e-7)


def test_diffusion_limitation_starves_deep_layers(net, cond):
    # Bulk starts full, biofilm empty, reaction on. With fast diffusion A
    # penetrates and fills the deepest layer; with slow diffusion it is consumed
    # in the outer layers before reaching the wall, so the deepest layer stays
    # starved. This depth dependence is the whole point of resolving the biofilm.
    p = net.default_parameters()  # slow reaction (k=0.1): bulk stays ~full
    n_comp = 7
    y0 = jnp.zeros((n_comp, net.n_species)).at[0, net.species_index["A"]].set(1.0)
    fast = _reactor(net, cond, diffusivity=1e-3)
    slow = _reactor(net, cond, diffusivity=1e-6)
    # Short time: fast diffusion (timescale L^2/D ~ 6e-4 d) fills the deepest
    # layer; slow diffusion (timescale ~0.6 d) has barely penetrated.
    deep_fast = float(fast.solve(y0, params=p, t_span=(0.0, 0.1)).profile_named("A")[-1, -1])
    deep_slow = float(slow.solve(y0, params=p, t_span=(0.0, 0.1)).profile_named("A")[-1, -1])
    assert deep_fast > deep_slow + 0.3


def test_phase_mask_confines_reaction_to_biofilm(net, cond):
    # With A_to_B marked biofilm-only, the bulk compartment has no reaction, so
    # the (large) bulk A pool falls only by diffusion into the (small) reacting
    # biofilm -- much slower than when the reaction also runs in the bulk. And
    # bulk B is then produced only in the layers, so it lags.
    p = net.default_parameters()
    C0 = jnp.array([1.0, 0.0])  # uniform A everywhere
    everywhere = _reactor(net, cond)                       # runs in every compartment
    film_only = _reactor(net, cond, biofilm_reactions=["A_to_B"])
    se = everywhere.solve(C0, params=p, t_span=(0.0, 2.0))
    sf = film_only.solve(C0, params=p, t_span=(0.0, 2.0))
    assert float(sf.C_named("A")[-1]) > float(se.C_named("A")[-1]) + 1e-2
    assert float(sf.C_named("B")[-1]) < float(se.C_named("B")[-1])


def test_unknown_biofilm_reaction_name_raises(net, cond):
    with pytest.raises(ValueError, match="Unknown biofilm reaction"):
        _reactor(net, cond, biofilm_reactions=["not_a_reaction"])


def test_solution_shapes_and_named_accessors(net, cond):
    r = _reactor(net, cond)
    p = net.default_parameters()
    t_eval = jnp.linspace(0.0, 1.0, 5)
    sol = r.solve(jnp.array([1.0, 0.0]), params=p, t_span=(0.0, 1.0), t_eval=t_eval)
    assert sol.C.shape == (5, net.n_species)               # bulk trajectory
    assert sol.profile.shape == (5, r.n_layers + 1, net.n_species)
    assert sol.depth.shape == (r.n_layers,)
    assert sol.profile_named("A").shape == (5, r.n_layers + 1)
    # bulk is compartment 0 of the profile
    assert np.allclose(np.asarray(sol.C), np.asarray(sol.profile[:, 0, :]))


def test_grad_flows_through_layered_solve(net, cond):
    # jax.grad must flow through the diffusion-reaction solve without NaNs.
    r = _reactor(net, cond)
    ki = net.param_index["A_to_B.k"]
    n_comp = r.n_layers + 1
    y0 = jnp.zeros((n_comp, net.n_species)).at[0, net.species_index["A"]].set(1.0)

    def final_bulk_A(k):
        p = net.default_parameters().at[ki].set(k)
        return r.solve(y0, params=p, t_span=(0.0, 1.0)).C_named("A")[-1]

    g = jax.grad(final_bulk_A)(0.1)
    assert np.isfinite(float(g))
    # more decay with larger k -> final bulk A decreases
    assert float(g) < 0.0


# --- steady_state() root-find ------------------------------------------------

def _fed_reactor(net, cond, **kw):
    """A biofilm with a continuous bulk feed of A, so the bulk has a unique
    feed-driven steady state (steady_state needs the feed to drive the bulk --
    it does not work with clamp_bulk or held-fixed species)."""
    feed = jnp.zeros(net.n_species).at[net.species_index["A"]].set(1.0)
    return _reactor(net, cond, n_layers=4, feed=feed, dilution_rate=2.0, **kw)


def test_steady_state_converges_to_a_fixed_point(net, cond):
    """steady_state finds y* with RHS(y*) = 0: a short forward solve started from
    the returned profile does not move it. The bulk sits at its feed-driven
    value, and the gradient flows through the implicit-function-theorem adjoint."""
    r = _fed_reactor(net, cond)
    p = net.default_parameters()
    n_comp = r.n_layers + 1
    C0 = jnp.full((n_comp, net.n_species), 0.5)

    ss = r.steady_state(C0, p, warmup=5.0, rtol=1e-9)
    prof = ss.profile[-1]
    # A fixed point of the dynamics: advancing it barely changes it.
    advanced = r.solve(prof, params=p, t_span=(0.0, 0.5)).profile[-1]
    assert float(jnp.max(jnp.abs(advanced - prof))) < 1e-8
    # The bulk equilibrates near the feed (dilution >> the slow decay loss).
    assert 0.9 < float(prof[0, net.species_index["A"]]) < 1.0

    # Differentiable w.r.t. a rate constant via the implicit-function-theorem
    # adjoint of the pseudo-transient steady state.
    def loss(k):
        pk = p.at[net.param_index["A_to_B.k"]].set(k)
        return jnp.sum(r.steady_state(C0, pk, warmup=5.0).profile[-1])

    g = jax.grad(loss)(float(p[net.param_index["A_to_B.k"]]))
    assert np.isfinite(float(g))


def test_steady_state_can_stall_silently(net, cond):
    """The documented failure mode: the solver returns a profile without raising,
    so when it is not given enough iterations (or a good seed) the result is a
    NON-steady profile. Here a 1-iteration solve from a far seed does not reach
    the fixed point, and the returned profile is demonstrably not steady --
    advancing it moves it substantially. Callers must therefore verify
    convergence (or, for a genuinely slow/stiff maturation, integrate forward
    instead)."""
    r = _fed_reactor(net, cond)
    p = net.default_parameters()
    n_comp = r.n_layers + 1
    far_seed = jnp.full((n_comp, net.n_species), 0.01)   # far from the fed state

    stalled = r.steady_state(far_seed, p, warmup=0.0, newton_steps=1).profile[-1]
    advanced = r.solve(stalled, params=p, t_span=(0.0, 0.5)).profile[-1]
    # NOT a fixed point: the under-iterated result still evolves a lot.
    assert float(jnp.max(jnp.abs(advanced - stalled))) > 0.1


def test_64bit_precision_enabled():
    import jax
    assert jax.config.x64_enabled
