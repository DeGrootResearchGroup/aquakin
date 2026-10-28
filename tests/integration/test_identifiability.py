"""Integration tests for :func:`aquakin.identifiability`.

The catalysed decay ``A -> B`` on a constant catalyst ``X`` gives
``A(t) = A0 exp(-k [X]0 t)``: only the product ``k [X]0`` is identifiable, so in
log coordinates the Fisher information has an exact null direction
``(log k - log [X]0) / sqrt(2)``.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import Experiment, FreeICConfig, LaplaceConfig, OptimizerConfig

FIXTURES = Path(__file__).parent.parent / "fixtures"
T_OBS = jnp.linspace(0.5, 12.0, 25)
LOG = {"A_to_B.k": "positive_log"}


@pytest.fixture
def catalysed_fim():
    model = aquakin.load_model_from_file(FIXTURES / "catalysed_decay.yaml")
    reactor = aquakin.BatchReactor(model, aquakin.SpatialConditions.uniform(1, T=293.15))
    design = Experiment(jnp.asarray([1.5, 0.0, 2.0]), T_OBS, sigma=0.01, observed_species=["A"])
    return aquakin.fisher_information(
        reactor, design, ["A_to_B.k"], transforms=LOG, free_ic=FreeICConfig(["A", "X"])
    )


@pytest.fixture
def decay_fim(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    design = Experiment(jnp.asarray([1.5, 0.0]), T_OBS, sigma=0.01, observed_species=["A"])
    return aquakin.fisher_information(
        reactor,
        design,
        ["A_to_B.k"],
        params=simple_model.default_parameters().at[0].set(0.25),
        transforms=LOG,
        free_ic=FreeICConfig(["A"]),
    )


def test_structural_product_is_flagged(catalysed_fim):
    rep = aquakin.identifiability(catalysed_fim)
    assert rep.names == ["A_to_B.k", "ic[0].A", "ic[0].X"]

    (null,) = rep.unidentifiable
    assert null.relative_eigenvalue < 1e-10
    assert rep.directions[0] is null  # least identifiable first
    s = 2**-0.5
    assert null.loadings["A_to_B.k"] == pytest.approx(s, abs=1e-6)
    assert null.loadings["ic[0].X"] == pytest.approx(-s, abs=1e-6)
    assert null.loadings["ic[0].A"] == pytest.approx(0.0, abs=1e-6)
    assert list(null.loadings)[2] == "ic[0].A"  # ordered by magnitude

    # The sign convention is deterministic under the exact +/- tie: the first
    # name with a non-negligible loading is positive, even if V's sign flips.
    flipped = aquakin.identifiability(catalysed_fim.matrix[::-1, ::-1], names=rep.names[::-1])
    assert flipped.unidentifiable[0].loadings["ic[0].X"] == pytest.approx(s, abs=1e-6)

    i, j = rep.names.index("A_to_B.k"), rep.names.index("ic[0].X")
    assert rep.correlation[i, j] == pytest.approx(-1.0, abs=1e-6)
    np.testing.assert_allclose(np.diag(rep.correlation), 1.0)

    assert rep.std["A_to_B.k"] == np.inf and rep.std["ic[0].X"] == np.inf
    assert np.isfinite(rep.std["ic[0].A"])
    assert rep.relative_std["ic[0].A"] == pytest.approx(rep.std["ic[0].A"] / 1.5)

    assert rep.collinearity_index(["A_to_B.k", "ic[0].X"]) > 1e6
    best = rep.largest_identifiable_subset()
    assert len(best) == 2 and "ic[0].A" in best
    assert {"A_to_B.k", "ic[0].X"} not in [set(sub) for sub, _ in rep.identifiable_subsets()]


def test_well_posed_design_matches_cramer_rao(decay_fim):
    rep = aquakin.identifiability(decay_fim)
    assert rep.unidentifiable == []
    assert all(d.identifiable for d in rep.directions)
    for name, sd in decay_fim.std().items():
        assert rep.std[name] == pytest.approx(sd, rel=1e-8)

    cov = np.linalg.inv(decay_fim.matrix)
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert rep.correlation[0, 1] == pytest.approx(rho, rel=1e-6)

    # For two quantities the normalised information has off-diagonal c and
    # smallest eigenvalue 1 - |c| (and |c| = |rho| for a 2x2 inverse).
    F = decay_fim.matrix
    c = F[0, 1] / np.sqrt(F[0, 0] * F[1, 1])
    gamma = (1.0 - abs(c)) ** -0.5
    assert rep.collinearity_index(rep.names) == pytest.approx(gamma, rel=1e-8)
    assert rep.collinearity_index(["A_to_B.k"]) == pytest.approx(1.0)
    assert rep.collinearity_table() == [(tuple(rep.names), pytest.approx(gamma, rel=1e-8))]
    assert rep.largest_identifiable_subset(threshold=gamma * 1.01) == tuple(rep.names)
    assert len(rep.largest_identifiable_subset(threshold=gamma * 0.99)) == 1


def test_eig_keep_controls_flagging(decay_fim):
    rep = aquakin.identifiability(decay_fim)
    w_rel = rep.directions[0].relative_eigenvalue
    flagged = aquakin.identifiability(decay_fim, eig_keep=min(0.99, 2 * w_rel))
    assert len(flagged.unidentifiable) == 1
    assert np.all(np.isfinite(list(flagged.std.values())))  # flagged, not singular


def test_from_calibration_result(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    params = simple_model.default_parameters().at[0].set(0.25)
    obs = reactor.solve(
        jnp.asarray([1.5, 0.0]), params=params, t_span=(0.0, 12.0), t_eval=T_OBS
    ).C_named("A")
    common = dict(
        transforms=LOG,
        observed_species=["A"],
        loss="nll",
        sigma=jnp.asarray(0.01),
        free_ic=FreeICConfig(["A"], bounds=(0.1, 10.0)),
        optimizer=OptimizerConfig(method="gauss_newton"),
    )
    joint = aquakin.calibrate(
        reactor,
        jnp.asarray([1.0, 0.0]),
        obs,
        T_OBS,
        ["A_to_B.k"],
        laplace=LaplaceConfig(method="gauss_newton", include_ic=True),
        **common,
    )
    rep = aquakin.identifiability(joint)
    assert rep.names == joint.joint_names
    np.testing.assert_allclose(rep.matrix, np.asarray(joint.joint_hessian))
    assert rep.std["ic[0].A"] == pytest.approx(
        np.sqrt(np.linalg.inv(rep.matrix)[1, 1]) * joint.ic_named[0]["A"], rel=1e-6
    )

    rate_only = aquakin.calibrate(
        reactor,
        jnp.asarray([1.0, 0.0]),
        obs,
        T_OBS,
        ["A_to_B.k"],
        laplace=LaplaceConfig(method="gauss_newton"),
        **common,
    )
    assert aquakin.identifiability(rate_only).names == ["A_to_B.k"]

    no_laplace = aquakin.calibrate(
        reactor, jnp.asarray([1.0, 0.0]), obs, T_OBS, ["A_to_B.k"], laplace=False, **common
    )
    with pytest.raises(ValueError, match="laplace=True"):
        aquakin.identifiability(no_laplace)


def test_bare_matrix_and_zero_information():
    F = np.diag([4.0, 1.0, 0.0])
    rep = aquakin.identifiability(F, names=["a", "b", "c"])
    assert rep.std is None and rep.relative_std is None
    assert rep.collinearity_index(["a", "c"]) == np.inf
    assert len(rep.collinearity_table()) == 2**3 - 3 - 1
    assert len(rep.collinearity_table(max_size=2)) == 3
    assert rep.largest_identifiable_subset() == ("a", "b")
    assert ("c",) not in [s for s, _ in rep.identifiable_subsets()]

    rep = aquakin.identifiability(
        np.diag([4.0, 1.0]), names=["a", "b"], values={"a": 2.0, "b": 0.5}
    )
    assert rep.std == pytest.approx({"a": 0.5, "b": 1.0})
    assert rep.relative_std == pytest.approx({"a": 0.25, "b": 2.0})

    # A singular pair whose combination still has information gives inf for
    # the collinearity index, not an error.
    ones = aquakin.identifiability(np.ones((2, 2)), names=["a", "b"])
    assert ones.collinearity_index(["a", "b"]) == np.inf
    assert ones.largest_identifiable_subset() in [("a",), ("b",)]
    assert aquakin.identifiability(np.zeros((1, 1)) + 1.0, names=["a"]).collinearity_table() == []


def test_input_validation():
    with pytest.raises(ValueError, match="names="):
        aquakin.identifiability(np.eye(2))
    with pytest.raises(ValueError, match="does not match"):
        aquakin.identifiability(np.eye(2), names=["a"])
    with pytest.raises(ValueError, match="eig_keep"):
        aquakin.identifiability(np.eye(2), names=["a", "b"], eig_keep=1.0)
    with pytest.raises(ValueError, match="no positive eigenvalue"):
        aquakin.identifiability(np.zeros((2, 2)), names=["a", "b"])
    big = aquakin.identifiability(np.eye(21), names=[f"p{i}" for i in range(21)])
    with pytest.raises(ValueError, match="max_size"):
        big.collinearity_table()
    assert len(big.collinearity_table(max_size=2)) == 21 * 20 // 2


def test_round_off_loading_on_null_direction_is_ignored():
    """A zero-eigenvalue direction that is a alone plus a round-off-sized
    loading on b leaves b determined: finite std and no spurious correlation."""
    eps = 3e-3
    v0 = np.array([1.0, eps, 0.0]) / np.hypot(1.0, eps)
    v1 = np.array([-eps, 1.0, 0.0]) / np.hypot(1.0, eps)
    v2 = np.array([0.0, 0.0, 1.0])
    V = np.stack([v0, v1, v2], axis=1)
    F = V @ np.diag([0.0, 1.0, 4.0]) @ V.T
    values = {"a": 1.0, "b": 1.0, "c": 1.0}
    rep = aquakin.identifiability(F, names=["a", "b", "c"], values=values)
    assert rep.std["a"] == np.inf
    assert rep.std["b"] == pytest.approx(1.0, rel=1e-4)
    assert rep.std["c"] == pytest.approx(0.5)
    assert abs(rep.correlation[0, 1]) < 1e-3

    # A tolerance below the spurious loading counts b as involved.
    strict = aquakin.identifiability(F, names=["a", "b", "c"], values=values, null_loading_tol=1e-4)
    assert strict.std["b"] == np.inf
    with pytest.raises(ValueError, match="null_loading_tol"):
        aquakin.identifiability(F, names=["a", "b", "c"], null_loading_tol=0.0)


def test_at_bound_names_get_nan_std_and_others_are_conditional():
    """A quantity flagged at_bound gets a nan std; the others are the
    Cramer-Rao std with it held fixed (1/sqrt(F_aa) here), not the marginal."""
    F = np.array([[4.0, 1.5], [1.5, 1.0]])
    vals = {"a": 1.0, "b": 2.0}
    free = aquakin.identifiability(F, names=["a", "b"], values=vals)
    held = aquakin.identifiability(F, names=["a", "b"], values=vals, at_bound={"b": "lower"})
    assert held.at_bound == {"b": "lower"}
    assert np.isnan(held.std["b"]) and np.isnan(held.relative_std["b"])
    assert held.std["a"] == pytest.approx(1.0 / np.sqrt(4.0))
    assert free.std["a"] == pytest.approx(np.sqrt(np.linalg.inv(F)[0, 0]))
    assert free.at_bound == {}
    # The eigen-analysis still covers the whole matrix.
    np.testing.assert_allclose(held.eigenvalues, free.eigenvalues)
    with pytest.raises(ValueError, match="at_bound name"):
        aquakin.identifiability(F, names=["a", "b"], values=vals, at_bound={"c": "upper"})
