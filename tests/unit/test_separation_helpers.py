"""Fast unit tests for the shared separator/clarifier helpers.

The separator/clarifier family (ideal thickener/clarifier, primary clarifier,
Takacs settler, SBR settling) now shares one species-mask policy and the
Q-weighted mixing / capture-split kernels. These check the helpers directly on
stub models/streams -- no plant solve -- including the **decided missing-species
policy** (raise, not silently drop).
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from aquakin.plant._constants import (
    species_indices,
    species_mask,
    tss_concentration,
)
from aquakin.plant.streams import mixed_feed, split_by_capture, total_flow


def _model(*names):
    """A stub carrying just the two attributes the mask helpers read."""
    return SimpleNamespace(species_index={n: i for i, n in enumerate(names)}, n_species=len(names))


# ---- species_indices / species_mask -------------------------------------


def test_species_indices_resolves_in_order():
    m = _model("A", "B", "C", "D")
    assert species_indices(m, ["C", "A"]) == [2, 0]


def test_species_indices_raises_on_missing_naming_the_species():
    m = _model("A", "B")
    with pytest.raises(ValueError, match=r"\bZ\b"):
        species_indices(m, ["A", "Z"], what="settling species")


def test_species_indices_empty_is_empty():
    assert species_indices(_model("A", "B"), []) == []


def test_species_mask_places_weight_at_named_species():
    m = _model("A", "B", "C")
    mask = species_mask(m, ["A", "C"], weight=0.75)
    assert np.allclose(np.asarray(mask), [0.75, 0.0, 0.75])


def test_species_mask_default_weight_is_one():
    m = _model("A", "B", "C")
    assert np.allclose(np.asarray(species_mask(m, ["B"])), [0.0, 1.0, 0.0])


def test_species_mask_raises_on_missing():
    # The policy: a missing species is a loud construction-time error, not a
    # silent drop (which under-settles / under-counts without warning).
    with pytest.raises(ValueError):
        species_mask(_model("A", "B"), ["A", "nope"])


def test_species_mask_empty_is_all_zero():
    assert np.allclose(np.asarray(species_mask(_model("A", "B"), [])), [0.0, 0.0])


# ---- tss_concentration --------------------------------------------------


def test_tss_concentration_is_the_weighted_sum():
    tss_vec = jnp.array([0.75, 0.0, 0.75, 0.0])
    C = jnp.array([10.0, 5.0, 4.0, 8.0])
    assert float(tss_concentration(C, tss_vec)) == pytest.approx(0.75 * (10.0 + 4.0))


def test_tss_concentration_vectorises_over_leading_axis():
    tss_vec = jnp.array([1.0, 0.0])
    C = jnp.array([[3.0, 9.0], [4.0, 9.0]])  # (n_t, n_species)
    assert np.allclose(np.asarray(tss_concentration(C, tss_vec)), [3.0, 4.0])


# ---- total_flow ---------------------------------------------------------


def test_total_flow_sums_an_iterable():
    assert float(total_flow(jnp.asarray(q) for q in (1.0, 2.5, 0.5))) == pytest.approx(4.0)


def test_total_flow_empty_is_zero():
    assert float(total_flow(iter(()))) == 0.0


# ---- mixed_feed ---------------------------------------------------------


def _stream(Q, C):
    return SimpleNamespace(Q=jnp.asarray(Q), C=jnp.asarray(C))


def test_mixed_feed_is_flow_weighted():
    inputs = {"a": _stream(1.0, [10.0, 0.0]), "b": _stream(3.0, [2.0, 4.0])}
    Q_total, C_in = mixed_feed(inputs, ["a", "b"])
    assert float(Q_total) == pytest.approx(4.0)
    # (1*10 + 3*2)/4 = 4 ; (1*0 + 3*4)/4 = 3
    assert np.allclose(np.asarray(C_in), [4.0, 3.0])


def test_mixed_feed_zero_flow_is_finite():
    inputs = {"a": _stream(0.0, [5.0, 5.0])}
    Q_total, C_in = mixed_feed(inputs, ["a"])
    assert float(Q_total) == 0.0
    assert np.all(np.isfinite(np.asarray(C_in)))  # eps guard, no inf/nan


# ---- split_by_capture ---------------------------------------------------


def test_split_by_capture_conserves_mass_and_partitions():
    part_mask = jnp.array([1.0, 0.0])  # species 0 particulate, 1 soluble
    C_in = jnp.array([100.0, 20.0])
    Q_in, Q_under, Q_over = 10.0, 2.0, 8.0
    cap = 0.9
    C_under, C_over = split_by_capture(C_in, part_mask, cap, Q_in, Q_under, Q_over)
    # Particulate mass: 90% of 10*100=1000 -> 900 to underflow, 100 to overflow.
    assert float(C_under[0]) == pytest.approx(900.0 / Q_under)
    assert float(C_over[0]) == pytest.approx(100.0 / Q_over)
    # Total particulate mass conserved.
    mass_out = float(C_under[0]) * Q_under + float(C_over[0]) * Q_over
    assert mass_out == pytest.approx(Q_in * 100.0)
    # Solubles pass through at the inlet concentration into both outlets.
    assert float(C_under[1]) == pytest.approx(20.0)
    assert float(C_over[1]) == pytest.approx(20.0)


def test_split_by_capture_zero_outlet_flow_is_finite():
    part_mask = jnp.array([1.0])
    C_under, C_over = split_by_capture(jnp.array([5.0]), part_mask, 1.0, 10.0, 10.0, 0.0)
    # All captured to underflow; the empty overflow must not blow up.
    assert np.all(np.isfinite(np.asarray(C_over)))
