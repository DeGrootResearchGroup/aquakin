"""Unit tests for CFDReactor: shape validation, error semantics, return types."""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


@pytest.fixture
def reactor(simple_model):
    return aquakin.CFDReactor(simple_model)


@pytest.fixture
def simple_state(simple_model):
    """Minimal valid call shape for the simple A -> B model."""
    n_cells = 4
    C = np.zeros((n_cells, simple_model.n_species))
    C[:, simple_model.species_index["A"]] = 1.0
    conds = {"T": np.full(n_cells, 293.15)}
    return C, conds


def test_step_returns_numpy(reactor, simple_state):
    C, conds = simple_state
    out = reactor.step(C, conds, dt=1.0)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == C.shape


def test_step_preserves_mass(reactor, simple_state):
    """For A -> B the total A+B per cell is conserved."""
    C, conds = simple_state
    total_in = C.sum(axis=1)
    out = reactor.step(C, conds, dt=5.0)
    total_out = out.sum(axis=1)
    assert np.allclose(total_out, total_in, rtol=1e-8, atol=1e-10)


def test_step_default_params_uses_model_defaults(reactor, simple_state):
    C, conds = simple_state
    out_default = reactor.step(C, conds, dt=1.0)
    out_explicit = reactor.step(
        C, conds, dt=1.0, params=np.asarray(reactor.model.default_parameters())
    )
    assert np.allclose(out_default, out_explicit)


def test_step_rejects_wrong_C_shape(reactor):
    bad = np.zeros((3, 5))  # 5 species columns; model has 2
    with pytest.raises(ValueError):
        reactor.step(bad, {"T": np.full(3, 293.15)}, dt=1.0)


def test_step_rejects_1d_C(reactor):
    with pytest.raises(ValueError):
        reactor.step(np.zeros(2), {"T": np.full(2, 293.15)}, dt=1.0)


def test_step_rejects_missing_condition(reactor):
    C = np.zeros((3, 2))
    with pytest.raises(ValueError):
        reactor.step(C, {}, dt=1.0)


def test_step_rejects_wrong_condition_shape(reactor, simple_state):
    C, _ = simple_state
    bad_conds = {"T": np.full(C.shape[0] + 1, 293.15)}  # wrong cell count
    with pytest.raises(ValueError):
        reactor.step(C, bad_conds, dt=1.0)


def test_step_rejects_non_positive_dt(reactor, simple_state):
    C, conds = simple_state
    with pytest.raises(ValueError):
        reactor.step(C, conds, dt=0.0)
    with pytest.raises(ValueError):
        reactor.step(C, conds, dt=-1.0)


def test_step_rejects_wrong_params_shape(reactor, simple_state):
    C, conds = simple_state
    with pytest.raises(ValueError):
        reactor.step(C, conds, dt=1.0, params=np.zeros(99))


def test_step_rejects_zero_cells(reactor):
    C = np.zeros((0, 2))
    conds = {"T": np.zeros(0)}
    with pytest.raises(ValueError):
        reactor.step(C, conds, dt=1.0)


def test_step_jit_cache_hit(reactor, simple_state):
    C, conds = simple_state
    assert reactor._jit_cache == {}
    reactor.step(C, conds, dt=1.0)
    assert len(reactor._jit_cache) == 1
    # Same n_cells — cache should be reused, not regrown.
    for _ in range(3):
        reactor.step(C, conds, dt=2.0)
    assert len(reactor._jit_cache) == 1


def test_step_jit_cache_keyed_on_n_cells(reactor, simple_model):
    """Different cell counts get separate cache entries."""
    C2 = np.zeros((2, simple_model.n_species))
    C2[:, simple_model.species_index["A"]] = 1.0
    C5 = np.zeros((5, simple_model.n_species))
    C5[:, simple_model.species_index["A"]] = 1.0
    reactor.step(C2, {"T": np.full(2, 293.15)}, dt=1.0)
    reactor.step(C5, {"T": np.full(5, 293.15)}, dt=1.0)
    assert set(reactor._jit_cache.keys()) == {2, 5}


def test_step_raises_on_pathological_inputs(reactor, simple_state):
    """The seam re-raises whatever exception the stiff solver produces.

    Diffrax's adaptive controller hits ``max_steps`` (or similar) when the
    integration cannot make progress; the seam propagates that exception so
    the C++ caller can decide whether to retry with a smaller ``dt``.
    Either Diffrax raises (typical) or the on_nan policy raises after the
    integration completes with NaN. Both routes surface as exceptions.
    """
    C, conds = simple_state
    with pytest.raises(Exception):
        reactor.step(C, conds, dt=1.0, params=np.asarray([np.nan]))


def test_invalid_on_nan_rejected(simple_model):
    with pytest.raises(ValueError):
        aquakin.CFDReactor(simple_model, on_nan="explode")


def _inject_nonfinite_inner(reactor, n_cells, bad_row):
    """Pre-populate the jit cache with a stub that returns a non-finite cell,
    isolating step()'s post-solve finiteness/mask handling from the solver."""

    def fake(C, cond, dt, params):
        return C.at[bad_row, 0].set(jnp.nan)

    reactor._jit_cache[n_cells] = fake


def test_step_ignore_returns_nonfinite_without_raising(simple_model, simple_state):
    """on_nan='ignore' passes non-finite cells through with no signal."""
    C, conds = simple_state
    reactor = aquakin.CFDReactor(simple_model, on_nan="ignore")
    _inject_nonfinite_inner(reactor, C.shape[0], bad_row=1)
    out = reactor.step(C, conds, dt=1.0)
    assert isinstance(out, np.ndarray)
    assert not np.all(np.isfinite(out[1]))                    # the corrupted cell
    assert np.all(np.isfinite(np.delete(out, 1, axis=0)))     # the rest is finite


def test_step_return_mask_flags_nonfinite_under_ignore(simple_model, simple_state):
    """return_mask lets the caller detect dropped cells even under 'ignore'."""
    C, conds = simple_state
    reactor = aquakin.CFDReactor(simple_model, on_nan="ignore")
    _inject_nonfinite_inner(reactor, C.shape[0], bad_row=2)
    out, mask = reactor.step(C, conds, dt=1.0, return_mask=True)
    assert mask.shape == (C.shape[0],)
    assert not mask[2]
    assert mask.sum() == C.shape[0] - 1


def test_step_return_mask_all_true_on_clean(reactor, simple_state):
    """A clean step with return_mask=True returns an all-True mask."""
    C, conds = simple_state
    out, mask = reactor.step(C, conds, dt=1.0, return_mask=True)
    assert out.shape == C.shape
    assert mask.shape == (C.shape[0],)
    assert mask.all()


def test_step_raise_detects_injected_nonfinite(simple_model, simple_state):
    """on_nan='raise' (default) surfaces a non-finite result as RuntimeError."""
    C, conds = simple_state
    reactor = aquakin.CFDReactor(simple_model)   # on_nan='raise'
    _inject_nonfinite_inner(reactor, C.shape[0], bad_row=0)
    with pytest.raises(RuntimeError, match="non-finite"):
        reactor.step(C, conds, dt=1.0)


def test_species_field_order_matches_model(reactor):
    assert reactor.species_field_order == reactor.model.species


def test_condition_field_names_matches_model(reactor):
    assert reactor.condition_field_names == reactor.model.conditions_required
