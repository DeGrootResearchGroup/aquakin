"""Fast unit tests for the compiled-solver settings cache key.

The reactor path (:func:`reactor_settings_key`, which memoises the ``atol`` key)
and the plant path (:func:`settings_cache_key`, raw ``atol``) both assemble the
same field-ordered tuple, now through the single :func:`_assemble_settings_key`.
These check the tuple shape/values and -- crucially -- that the two entry points
agree, so a compile-affecting setting added to one but not the other would fail
here rather than surface as a silent wrong-result cache hit. No solve/compile:
the reactor is a lightweight stub carrying only the settings attributes.
"""

from types import SimpleNamespace

import numpy as np

from aquakin.integrate._common import (
    atol_cache_key,
    reactor_settings_key,
    settings_cache_key,
)


def _stub_reactor(**kw):
    """A reactor stand-in exposing only the settings attributes the key reads."""
    kw.setdefault("adjoint", None)
    kw.setdefault("dtmax", None)
    kw.setdefault("order", None)
    kw.setdefault("factormax", None)
    kw.setdefault("solver", None)
    return SimpleNamespace(**kw)


def test_key_fields_and_none_normalisation():
    key = settings_cache_key(1e-6, 1e-8, None, None, 4096)
    assert key == (
        1e-6,
        atol_cache_key(1e-8),
        None,  # default adjoint
        None,  # no dtmax
        4096,
        None,  # no order
        None,  # no factormax
        None,  # no solver
    )


def test_optional_fields_are_keyed_when_supplied():
    class Kvaerno5:  # a stand-in solver; keyed by class name
        pass

    adj = object()
    key = settings_cache_key(1e-6, 1e-8, adj, 0.5, 1000, order=3, factormax=8.0, solver=Kvaerno5())
    assert key[2] == id(adj)  # adjoint keyed by identity
    assert key[3] == 0.5  # dtmax
    assert key[5] == 3  # order
    assert key[6] == 8.0  # factormax
    assert key[7] == "Kvaerno5"  # solver by class name


def test_reactor_key_matches_settings_key_for_equal_settings():
    # The property the dedup guarantees: the two entry points cannot drift.
    r = _stub_reactor(rtol=1e-6, atol=np.array([1e-8, 2e-8]), max_steps=4096)
    assert reactor_settings_key(r) == settings_cache_key(
        r.rtol, r.atol, r.adjoint, r.dtmax, r.max_steps
    )


def test_reactor_key_matches_with_all_optional_settings():
    class Kvaerno3:
        pass

    adj = object()
    r = _stub_reactor(
        rtol=3e-5,
        atol=np.array([1e-7, 5e-7, 1e-6]),
        max_steps=2000,
        adjoint=adj,
        dtmax=0.25,
        order=5,
        factormax=4.0,
        solver=Kvaerno3(),
    )
    assert reactor_settings_key(r) == settings_cache_key(
        r.rtol,
        r.atol,
        r.adjoint,
        r.dtmax,
        r.max_steps,
        order=r.order,
        factormax=r.factormax,
        solver=r.solver,
    )


def test_reactor_key_memoises_the_atol_key():
    # First call materialises and stores the atol key; a later atol swap must not
    # be re-read (the key is fixed at construction), so the stored key wins.
    r = _stub_reactor(rtol=1e-6, atol=np.array([1e-8]), max_steps=100)
    k1 = reactor_settings_key(r)
    assert r._atol_key == atol_cache_key(np.array([1e-8]))
    r.atol = np.array([9.9e-3])  # would change the key if re-read
    assert reactor_settings_key(r) == k1  # memoised -> unchanged


def test_distinct_settings_give_distinct_keys():
    base = dict(rtol=1e-6, atol=1e-8, adjoint=None, dtmax=None, max_steps=4096)
    k = settings_cache_key(**base)
    assert settings_cache_key(**{**base, "rtol": 1e-7}) != k
    assert settings_cache_key(**{**base, "atol": 1e-9}) != k
    assert settings_cache_key(**{**base, "dtmax": 0.5}) != k
    assert settings_cache_key(**{**base, "max_steps": 2048}) != k
