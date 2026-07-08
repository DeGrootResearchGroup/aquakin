"""Coverage of input-validation ``raise`` statements across ``aquakin.integrate``.

Each test triggers one validation error via the most natural public API (or, for
the internal helpers, a direct call with a bad shape). These all fire *before*
any expensive stiff solve, so the suite stays fast: no full integration runs.
"""

import os

import jax.numpy as jnp
import pytest

import aquakin
from aquakin import BatchReactor, SpatialConditions

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "simple_model.yaml")


@pytest.fixture(scope="module")
def model():
    return aquakin.load_model_from_file(FIXTURE)


@pytest.fixture
def cond():
    return SpatialConditions.uniform(T=293.15)


def _biofilm(model, cond, **kw):
    defaults = dict(
        n_layers=6,
        thickness=8e-4,
        area_per_volume=50.0,
        diffusivity=1e-4,
        boundary_layer=1e-4,
    )
    defaults.update(kw)
    return aquakin.BiofilmReactor(model, cond, **defaults)


# ---------------------------------------------------------------------------
# aquakin/integrate/biofilm.py
# ---------------------------------------------------------------------------


def test_biofilm_profile_named_unknown_species(model, cond):
    # BiofilmSolution.profile_named raises KeyError on an unknown species.
    r = _biofilm(model, cond, n_layers=2)
    p = model.parameter_values({"A_to_B.k": 0.0})
    sol = r.solve(jnp.array([1.0, 0.0]), params=p, t_span=(0.0, 1.0))
    with pytest.raises(KeyError, match="Unknown species 'Z'"):
        sol.profile_named("Z")


def test_biofilm_n_layers_must_be_positive(model, cond):
    with pytest.raises(ValueError, match="n_layers must be >= 1"):
        _biofilm(model, cond, n_layers=0)


def test_biofilm_geometry_must_be_positive(model, cond):
    with pytest.raises(ValueError, match="must be positive"):
        _biofilm(model, cond, thickness=0.0)


def test_biofilm_soluble_mask_wrong_shape(model, cond):
    with pytest.raises(ValueError, match="soluble_mask must have shape"):
        _biofilm(model, cond, soluble_mask=jnp.array([True, True, True]))


def test_biofilm_fixed_mask_wrong_shape(model, cond):
    with pytest.raises(ValueError, match="fixed_mask must have shape"):
        _biofilm(model, cond, fixed_mask=jnp.array([True, True, True]))


def test_biofilm_unknown_reaction_name(model, cond):
    with pytest.raises(ValueError, match="Unknown biofilm reaction names"):
        _biofilm(model, cond, biofilm_reactions=["not_a_reaction"])


def test_biofilm_reactions_mask_wrong_shape(model, cond):
    with pytest.raises(ValueError, match="biofilm_reactions mask must have shape"):
        _biofilm(model, cond, biofilm_reactions=jnp.array([True, True, True]))


def test_biofilm_detach_mask_wrong_shape(model, cond):
    with pytest.raises(ValueError, match="detach_mask must have shape"):
        _biofilm(model, cond, k_det=0.1, detach_mask=jnp.array([True, True, True]))


def test_biofilm_attach_mask_wrong_shape(model, cond):
    with pytest.raises(ValueError, match="attach_mask must have shape"):
        _biofilm(model, cond, k_att=0.1, attach_mask=jnp.array([True, True, True]))


def test_biofilm_check_params_wrong_shape(model, cond):
    r = _biofilm(model, cond, n_layers=2)
    with pytest.raises(ValueError, match="params has shape"):
        r.solve(jnp.array([1.0, 0.0]), params=jnp.array([1.0, 2.0, 3.0]), t_span=(0.0, 1.0))


def test_biofilm_coerce_y0_wrong_shape(model, cond):
    r = _biofilm(model, cond, n_layers=2)
    p = model.parameter_values({"A_to_B.k": 0.0})
    with pytest.raises(ValueError, match="C0 has shape"):
        r.solve(jnp.array([1.0, 0.0, 3.0]), params=p, t_span=(0.0, 1.0))


def test_biofilm_solve_requires_t_span(model, cond):
    r = _biofilm(model, cond, n_layers=2)
    p = model.parameter_values({"A_to_B.k": 0.0})
    with pytest.raises(ValueError, match=r"t_span=.*is required"):
        r.solve(jnp.array([1.0, 0.0]), params=p, t_span=None)


def test_biofilm_solve_t_span_end_must_exceed_start(model, cond):
    r = _biofilm(model, cond, n_layers=2)
    p = model.parameter_values({"A_to_B.k": 0.0})
    with pytest.raises(ValueError, match="t_span end must exceed start"):
        r.solve(jnp.array([1.0, 0.0]), params=p, t_span=(1.0, 1.0))


def test_biofilm_solve_sensitivity_t_span_end_must_exceed_start(model, cond):
    r = _biofilm(model, cond, n_layers=2)
    p = model.default_parameters()
    with pytest.raises(ValueError, match="t_span end must exceed start"):
        r.solve_sensitivity(
            jnp.array([1.0, 0.0]),
            t_span=(1.0, 0.5),
            params=p,
            sens_params=["A_to_B.k"],
        )


# ---------------------------------------------------------------------------
# aquakin/integrate/_common.py
# ---------------------------------------------------------------------------


def test_validate_t_eval_not_1d(model, cond):
    reactor = BatchReactor(model, cond)
    p = model.default_parameters()
    with pytest.raises(ValueError, match="t_eval must be 1-D"):
        reactor.solve(
            jnp.array([1.0, 0.0]),
            params=p,
            t_span=(0.0, 1.0),
            t_eval=jnp.array([[0.0, 0.5], [0.5, 1.0]]),
        )


def test_validate_t_eval_outside_span(model, cond):
    reactor = BatchReactor(model, cond)
    p = model.default_parameters()
    with pytest.raises(ValueError, match="t_eval must lie within t_span"):
        reactor.solve(
            jnp.array([1.0, 0.0]),
            params=p,
            t_span=(0.0, 1.0),
            t_eval=jnp.array([0.0, 2.0]),
        )


def test_validate_t_eval_not_ascending(model, cond):
    reactor = BatchReactor(model, cond)
    p = model.default_parameters()
    with pytest.raises(ValueError, match="t_eval must be strictly ascending"):
        reactor.solve(
            jnp.array([1.0, 0.0]),
            params=p,
            t_span=(0.0, 1.0),
            t_eval=jnp.array([0.0, 0.75, 0.5]),
        )


def test_coerce_atol_wrong_shape():
    from aquakin.integrate._common import _coerce_atol

    with pytest.raises(ValueError, match=r"atol array must have shape \(3,\)"):
        _coerce_atol(jnp.array([1e-6, 1e-6]), 3)


def test_validate_c0_params_bad_c0():
    from aquakin.integrate._common import validate_C0_params

    net = aquakin.load_model_from_file(FIXTURE)
    with pytest.raises(ValueError, match="C0 has shape"):
        validate_C0_params(net, jnp.array([1.0, 0.0, 0.0]), net.default_parameters())


def test_validate_c0_params_bad_params():
    from aquakin.integrate._common import validate_C0_params

    net = aquakin.load_model_from_file(FIXTURE)
    with pytest.raises(ValueError, match="params has shape"):
        validate_C0_params(net, net.default_concentrations(), jnp.array([1.0, 2.0, 3.0]))


def test_build_implicit_solver_bad_order():
    from aquakin.integrate._common import build_implicit_solver

    with pytest.raises(ValueError, match="order must be one of"):
        build_implicit_solver(1e-6, 1e-9, order=7)


# ---------------------------------------------------------------------------
# aquakin/integrate/particle.py
# ---------------------------------------------------------------------------


def test_track_t_must_be_1d():
    with pytest.raises(ValueError, match=r"Track\.t must be 1-D"):
        aquakin.Track(t=jnp.array([[0.0, 1.0], [2.0, 3.0]]))


def test_track_needs_two_samples():
    with pytest.raises(ValueError, match="at least 2 sample points"):
        aquakin.Track(t=jnp.array([0.0]))


# ---------------------------------------------------------------------------
# aquakin/integrate/batch.py
# ---------------------------------------------------------------------------


def test_batch_solve_sensitivity_t_span_end_must_exceed_start(model, cond):
    reactor = BatchReactor(model, cond)
    p = model.default_parameters()
    with pytest.raises(ValueError, match="t_span end must exceed start"):
        reactor.solve_sensitivity(
            jnp.array([1.0, 0.0]),
            t_span=(1.0, 1.0),
            params=p,
            sens_params=["A_to_B.k"],
        )


# ---------------------------------------------------------------------------
# aquakin/integrate/events.py
# ---------------------------------------------------------------------------


def test_solve_with_events_needs_an_event():
    with pytest.raises(ValueError, match="needs at least one Event"):
        aquakin.solve_with_events(
            lambda t, y, args: jnp.ones_like(y),
            jnp.array([0.0]),
            1.0,
            t0=0.0,
            t1=1.0,
            t_eval=None,
            events=[],
            rtol=1e-6,
            atol=1e-9,
        )


def test_solve_with_events_t_eval_must_be_sorted():
    ev = aquakin.Event(at_times=[0.5], apply=lambda t, y, a: y, name="bump")
    with pytest.raises(ValueError, match="t_eval must be sorted"):
        aquakin.solve_with_events(
            lambda t, y, args: jnp.ones_like(y),
            jnp.array([0.0]),
            1.0,
            t0=0.0,
            t1=1.0,
            t_eval=jnp.array([0.0, 0.75, 0.5]),
            events=[ev],
            rtol=1e-6,
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# aquakin/integrate/pfr.py
# ---------------------------------------------------------------------------


def test_pfr_solve_sensitivity_conditions_nlocations_mismatch(model):
    cond = SpatialConditions.uniform(T=293.15)  # n_locations == 1
    reactor = aquakin.PlugFlowReactor(model, cond, n_points=4, length=1.0, velocity=1.0)
    override = SpatialConditions.uniform(n_locations=3, T=293.15)
    with pytest.raises(ValueError, match="conditions override must have n_locations"):
        reactor.solve_sensitivity(
            jnp.array([1.0, 0.0]),
            params=model.default_parameters(),
            sens_params=["A_to_B.k"],
            conditions=override,
        )


# ---------------------------------------------------------------------------
# aquakin/integrate/_simultaneous_corrector.py
# ---------------------------------------------------------------------------


def test_simultaneous_corrector_rejects_wrong_size_operator():
    import lineax as lx

    from aquakin.integrate._simultaneous_corrector import SimultaneousCorrector

    # ndof=2, n_sens=1 -> expects a flat augmented operator of size 2*(1+1) = 4.
    # Hand it a 3x3 matrix operator instead so init's shape check fires.
    solver = SimultaneousCorrector(ndof=2, n_sens=1)
    operator = lx.MatrixLinearOperator(jnp.eye(3))
    with pytest.raises(ValueError, match="expects a flat augmented operator of size"):
        solver.init(operator, {})


# ---------------------------------------------------------------------------
# aquakin/integrate/fit.py  -- t_obs / observations validation (pre-solve)
# ---------------------------------------------------------------------------


def test_fit_rejects_non_1d_t_obs(model, cond):
    reactor = BatchReactor(model, cond)
    C0 = model.default_concentrations()
    obs = jnp.array([[1.0], [0.5]])
    with pytest.raises(ValueError, match="t_obs must be a non-empty 1-D array"):
        aquakin.fit(reactor, C0, obs, jnp.array([[0.0], [1.0]]), ["A_to_B.k"])


def test_fit_rejects_negative_t_obs(model, cond):
    reactor = BatchReactor(model, cond)
    C0 = model.default_concentrations()
    obs = jnp.array([[1.0], [0.5]])
    with pytest.raises(ValueError, match="t_obs must be non-negative"):
        aquakin.fit(reactor, C0, obs, jnp.array([-1.0, 1.0]), ["A_to_B.k"])


def test_fit_rejects_observations_row_mismatch(model, cond):
    reactor = BatchReactor(model, cond)
    C0 = model.default_concentrations()
    # 3 observation rows against 2 time points.
    with pytest.raises(ValueError, match="observations has 3 rows but t_obs has 2"):
        aquakin.fit(reactor, C0, jnp.array([1.0, 0.5, 0.2]), jnp.array([0.0, 1.0]), ["A_to_B.k"])


# ---------------------------------------------------------------------------
# aquakin/integrate/_qmc.py  -- sampler / output_names validation
# ---------------------------------------------------------------------------


def test_monte_carlo_rejects_unknown_sampler():
    with pytest.raises(ValueError, match="unknown sampler 'bogus'"):
        aquakin.monte_carlo(lambda x: x[0], [(0.0, 1.0)], sampler="bogus", n_samples=8)


def test_monte_carlo_rejects_output_names_length_mismatch():
    # fn returns a scalar (m=1) but two output_names are supplied.
    with pytest.raises(ValueError, match="output_names has 2 entries but fn returns m=1"):
        aquakin.monte_carlo(lambda x: x[0], [(0.0, 1.0)], output_names=["a", "b"], n_samples=8)


# ---------------------------------------------------------------------------
# aquakin/integrate/_common.py  -- friendly_solve_errors upstream-message pins
#
# friendly_solve_errors / is_forward_mode_ad_error map two opaque upstream
# failures to domain-level remedies by substring-matching the exception message.
# That is pragmatic but fragile: a Diffrax/Equinox or JAX wording change would
# silently stop the match and let the raw traceback leak. These tests provoke the
# two upstream errors directly (minimal, no reactor) and pin the exact substrings
# the matchers key on, so such a dependency bump fails here -- loudly and with a
# clear pointer to the wording that moved -- instead of degrading in the field.
# ---------------------------------------------------------------------------


def test_upstream_max_steps_message_still_matches():
    """Diffrax/Equinox still says 'maximum number of solver steps' when a solve
    exhausts ``max_steps`` (the substring ``friendly_solve_errors`` keys on)."""
    import diffrax

    term = diffrax.ODETerm(lambda t, y, args: -y)
    with pytest.raises(Exception) as excinfo:
        diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=jnp.asarray([1.0]),
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=1,
        )
    assert "maximum number of solver steps" in str(excinfo.value).lower()


def test_upstream_forward_mode_custom_vjp_message_still_matches():
    """JAX still rejects forward-mode autodiff through a ``custom_vjp`` with a
    message containing 'forward-mode autodiff' and 'custom_vjp' -- the two
    substrings ``is_forward_mode_ad_error`` requires."""
    import jax

    from aquakin.integrate._common import is_forward_mode_ad_error

    @jax.custom_vjp
    def f(x):
        return x

    f.defvjp(lambda x: (x, None), lambda _res, g: (g,))

    with pytest.raises(Exception) as excinfo:
        jax.jvp(f, (1.0,), (1.0,))
    assert is_forward_mode_ad_error(excinfo.value)
    lowered = str(excinfo.value).lower()
    assert "forward-mode autodiff" in lowered
    assert "custom_vjp" in lowered
