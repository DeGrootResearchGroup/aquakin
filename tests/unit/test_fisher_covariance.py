"""``FisherInformation.covariance`` on hand-built matrices (no solves): the
Cramer-Rao bound is the inverse, and a matrix singular up to round-off raises
``LinAlgError`` whether or not LAPACK's LU finds an exactly zero pivot."""

import numpy as np
import pytest

import aquakin
from aquakin.integrate.fisher import FisherInformation


def _fim(F):
    n = F.shape[0]
    names = [f"p{i}" for i in range(n)]
    return FisherInformation(
        matrix=F,
        names=names,
        transforms=["none"] * n,
        values=dict.fromkeys(names, 1.0),
        jacobian=np.zeros((1, n)),
    )


def _round_off_rank_one():
    """``v v^T`` with one entry nudged by one ulp: rank one up to round-off, and
    LU (``np.linalg.inv`` on Linux / OpenBLAS) finds a tiny nonzero pivot rather
    than an exact zero, so it returns a huge meaningless inverse."""
    v = np.asarray([1.0, 0.3])
    F = np.outer(v, v)
    F[0, 0] = np.nextafter(F[0, 0], 2.0)
    return F


def test_inv_does_not_flag_the_round_off_singular_matrix():
    """The premise of the regression test below: ``np.linalg.inv`` alone returns
    (an inverse of order 1/eps) instead of raising on this matrix."""
    F = _round_off_rank_one()
    assert np.abs(np.linalg.inv(F)).max() > 1e14


@pytest.mark.parametrize(
    "F",
    [_round_off_rank_one(), np.zeros((2, 2)), np.full((2, 2), np.nan)],
    ids=["round-off-rank-one", "zero", "nan"],
)
def test_covariance_raises_on_singular_matrix(F):
    fim = _fim(F)
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        fim.covariance()
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        fim.std()


def test_covariance_is_the_inverse_of_a_regular_matrix():
    """An ill-conditioned but regular matrix (smallest eigenvalue 1e-8 of the
    largest, above the null threshold) is inverted, not flagged."""
    Q = np.asarray([[0.6, -0.8], [0.8, 0.6]])
    F = Q @ np.diag([1e-8, 1.0]) @ Q.T
    cov = _fim(F).covariance()
    np.testing.assert_allclose(cov, Q @ np.diag([1e8, 1.0]) @ Q.T, rtol=1e-6)
    np.testing.assert_allclose(cov @ F, np.eye(2), atol=1e-6)


def test_singular_error_advice_names_the_null_direction():
    """Following the error's advice, ``aquakin.identifiability`` on the same
    result reports the null direction: ``v_perp = (0.3, -1) / |.|`` for
    ``F = v v^T`` with ``v = (1, 0.3)``, and both quantities load on it."""
    fim = _fim(_round_off_rank_one())
    with pytest.raises(np.linalg.LinAlgError, match=r"aquakin\.identifiability\(\)"):
        fim.covariance()
    rep = aquakin.identifiability(fim)
    (null,) = rep.unidentifiable
    assert null.relative_eigenvalue <= 1e-10
    expected = np.asarray([0.3, -1.0]) / np.hypot(0.3, 1.0)
    got = np.asarray([null.loadings["p0"], null.loadings["p1"]])
    np.testing.assert_allclose(np.abs(got), np.abs(expected), rtol=1e-8)
    assert np.isinf(rep.std["p0"]) and np.isinf(rep.std["p1"])
