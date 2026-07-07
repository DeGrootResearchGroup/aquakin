"""Fast unit tests for the steady-state DGSM sample-validity mask.

``_valid_sample_mask`` is the single home of the operating-regime exclusion
policy (near-singular-Jacobian drop AND past-fold exclusion) that
``convergence``, ``with_cond_factor`` and ``steady_state_dgsm`` all consume. It
is pure NumPy, so the policy -- non-finite drop, the ``cond_factor`` median
filter, the ``operating_point_exists`` past-fold exclusion, and their
intersection -- is checked here on hand-built arrays, no plant solve.
"""

import numpy as np

from aquakin.plant.sensitivity import _cond_mask, _valid_sample_mask


def test_no_filter_keeps_finite_drops_nonfinite():
    # cond_factor=None, no operating record -> keep every finite-cond sample.
    cond = np.array([1.0, 2.0, np.inf, np.nan, 5.0])
    mask = _valid_sample_mask(cond, None, None)
    assert mask.tolist() == [True, True, False, False, True]


def test_cond_factor_drops_near_singular():
    # median of [1,2,3,4,100] is 3; factor=2 keeps cond <= 6 -> drops the 100.
    cond = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    mask = _valid_sample_mask(cond, 2.0, None)
    assert mask.tolist() == [True, True, True, True, False]


def test_past_fold_samples_excluded():
    # operating_point_exists: False -> past-fold (excluded); True/None -> kept.
    cond = np.array([1.0, 1.0, 1.0, 1.0])
    oe = [True, False, None, True]
    mask = _valid_sample_mask(cond, None, oe)
    assert mask.tolist() == [True, False, True, True]


def test_operating_none_applies_only_conditioning():
    # operating_point_exists=None (result predates the record) -> the mask is
    # exactly the conditioning filter.
    cond = np.array([1.0, 2.0, np.inf, 4.0])
    assert np.array_equal(_valid_sample_mask(cond, None, None), _cond_mask(cond, None))


def test_combined_is_intersection_of_both_criteria():
    cond = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    oe = [True, False, True, True, True]
    mask = _valid_sample_mask(cond, 2.0, oe)
    # Sample 1 fails the past-fold test; sample 4 fails the conditioning test.
    assert mask.tolist() == [True, False, True, True, False]


def test_matches_the_inline_construction_it_replaced():
    # Guard the refactor: the helper reproduces the exact
    # ``_cond_mask(cond, factor) & operating_mask`` the call sites used inline.
    rng = np.random.default_rng(0)
    cond = rng.uniform(1.0, 50.0, size=64)
    cond[::7] = np.inf  # some non-finite samples
    oe = [False if i % 5 == 0 else (None if i % 3 == 0 else True) for i in range(64)]
    for factor in (None, 1.5, 3.0):
        operating_mask = np.array([e is not False for e in oe], dtype=bool)
        expected = _cond_mask(cond, factor) & operating_mask
        assert np.array_equal(_valid_sample_mask(cond, factor, oe), expected)


def test_prefix_call_matches_old_convergence_construction():
    # ``convergence`` feeds prefixes: it now calls the helper on cond[:k]/oe[:k]
    # instead of pre-building the operating array once and slicing op[:k]. The
    # two are equivalent because building the operating mask from oe[:k] equals
    # slicing the full operating mask -- while the conditioning filter is still
    # (correctly) recomputed on each prefix's own median.
    rng = np.random.default_rng(1)
    cond = rng.uniform(1.0, 20.0, size=32)
    oe = [False if i % 4 == 0 else True for i in range(32)]
    op_full = np.array([e is not False for e in oe], dtype=bool)  # the old precompute
    for k in (4, 8, 16, 32):
        old = _cond_mask(cond[:k], 2.0) & op_full[:k]
        assert np.array_equal(_valid_sample_mask(cond[:k], 2.0, oe[:k]), old)
