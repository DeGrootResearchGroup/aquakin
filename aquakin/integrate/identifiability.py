"""Practical-identifiability report from a Fisher information matrix.

A Fisher information ``F`` (from :func:`aquakin.fisher_information` or a
calibration's Gauss-Newton Laplace Hessian) says which parameter combinations a
design can determine. This module turns it into the quantities an
identifiability study reports:

- **Eigen-directions.** ``F = V diag(w) V^T``. A direction with a small
  eigenvalue ``w_k`` is a parameter combination the data barely constrain; its
  eigenvector loadings name the parameters involved (in log coordinates, equal
  and opposite loadings on ``k`` and ``[X]0`` mean only ``k [X]0`` is seen).
- **Correlations** of the Cramer-Rao covariance ``F^{-1}``.
- **Collinearity index** of a parameter subset ``K``: with ``F~`` the
  information normalised to unit diagonal (``F~_ij = F_ij / sqrt(F_ii F_jj)``,
  the Gram matrix of the unit-length sensitivity columns),
  ``gamma_K = 1 / sqrt(min eig(F~_KK))``. It is 1 for orthogonal sensitivities
  and infinite when the columns of ``K`` are linearly dependent, so it measures
  how far changes in one parameter of ``K`` can be compensated by the others,
  independently of how sensitive the output is to each.
- **Relative standard errors** ``sd_j / |p_j|`` in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from aquakin.integrate._transforms import dphysical_dunconstrained

# Eigenvalues at or below this fraction of the largest are treated as exactly
# zero (a structurally non-identifiable direction), so the parameters loading on
# them get an infinite standard error rather than a round-off-sized finite one.
_NULL_RTOL = 1e-10
# Loadings below this fraction of a direction's largest are ignored when fixing
# the eigenvector's sign convention.
_LOADING_TOL = 1e-6
# Largest parameter count for the exhaustive subset search without a size cap.
_MAX_EXHAUSTIVE = 20


@dataclass(frozen=True)
class EigenDirection:
    """One eigen-direction of the Fisher information.

    Attributes
    ----------
    eigenvalue : float
        Information along the direction.
    relative_eigenvalue : float
        ``eigenvalue`` divided by the largest eigenvalue.
    identifiable : bool
        ``relative_eigenvalue > eig_keep`` -- the same rule by which the Laplace
        posterior in :func:`aquakin.calibrate` keeps or drops a direction.
    loadings : dict[str, float]
        Eigenvector components by name, largest magnitude first. The sign is
        fixed so the first name (in ``names`` order) with a non-negligible
        loading is positive, which stays deterministic when loadings tie in
        magnitude (as they do for an exact product such as ``k [X]0``).
    """

    eigenvalue: float
    relative_eigenvalue: float
    identifiable: bool
    loadings: dict[str, float]


@dataclass
class IdentifiabilityResult:
    """Result of :func:`identifiability`.

    Attributes
    ----------
    names : list[str]
        Row / column names of ``matrix``.
    matrix : np.ndarray
        The Fisher information analysed, ``(n, n)``.
    eigenvalues : np.ndarray
        Eigenvalues of ``matrix``, ascending.
    eigenvectors : np.ndarray
        Matching unit eigenvectors as columns, ``(n, n)``.
    directions : list[EigenDirection]
        The eigen-directions, least identifiable first.
    correlation : np.ndarray
        Correlation matrix of the Cramer-Rao covariance, ``(n, n)``. It is
        formed from ``(F + delta I)^{-1}`` with a ``delta`` negligible next to
        the largest eigenvalue, so an exactly non-identifiable pair shows as a
        correlation of +/-1 rather than making the matrix undefined.
    std : dict[str, float] or None
        Cramer-Rao lower bound on each standard deviation in physical units,
        infinite for a quantity involved in an exactly non-identifiable
        direction (see ``null_loading_tol``). ``None`` when no physical values
        were supplied.
    relative_std : dict[str, float] or None
        ``std`` divided by each quantity's magnitude.
    eig_keep : float
        The relative eigenvalue floor used to flag directions.
    """

    names: list[str]
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    directions: list[EigenDirection]
    correlation: np.ndarray
    std: dict[str, float] | None
    relative_std: dict[str, float] | None
    eig_keep: float

    @property
    def unidentifiable(self) -> list[EigenDirection]:
        """The directions flagged as not identifiable (``identifiable=False``)."""
        return [d for d in self.directions if not d.identifiable]

    def _normalised(self) -> np.ndarray:
        diag = np.diag(self.matrix)
        scale = np.where(diag > 0.0, 1.0 / np.sqrt(np.where(diag > 0.0, diag, 1.0)), 0.0)
        return self.matrix * np.outer(scale, scale)

    def collinearity_index(self, subset: list[str]) -> float:
        """Collinearity index ``1 / sqrt(min eig(F~_KK))`` of a subset of names.

        Infinite when the subset's sensitivities are linearly dependent, or when
        it contains a quantity the data carry no information about.
        """
        idx = [self.names.index(n) for n in subset]
        if any(self.matrix[i, i] <= 0.0 for i in idx):
            return float("inf")
        lam = float(np.linalg.eigvalsh(self._normalised()[np.ix_(idx, idx)])[0])
        return float("inf") if lam <= 0.0 else float(lam**-0.5)

    def collinearity_table(
        self, max_size: int | None = None
    ) -> list[tuple[tuple[str, ...], float]]:
        """Collinearity index of every subset of two or more names.

        Parameters
        ----------
        max_size : int, optional
            Largest subset size to include. ``None`` means all sizes, which is
            allowed for up to 20 names (the table has ``2^n`` entries).

        Returns
        -------
        list of (tuple of str, float)
            ``(subset, index)`` pairs, ordered by size then by name order.
        """
        n = len(self.names)
        if max_size is None:
            if n > _MAX_EXHAUSTIVE:
                raise ValueError(
                    f"{n} quantities give 2^{n} subsets; pass max_size to cap the subset size."
                )
            max_size = n
        F = self._normalised()
        info = np.diag(self.matrix) > 0.0
        table = []
        for size in range(2, min(max_size, n) + 1):
            for idx in combinations(range(n), size):
                if not info[list(idx)].all():
                    gamma = float("inf")
                else:
                    lam = float(np.linalg.eigvalsh(F[np.ix_(idx, idx)])[0])
                    gamma = float("inf") if lam <= 0.0 else float(lam**-0.5)
                table.append((tuple(self.names[i] for i in idx), gamma))
        return table

    def identifiable_subsets(
        self, threshold: float = 15.0, max_size: int | None = None
    ) -> list[tuple[tuple[str, ...], float]]:
        """Subsets whose collinearity index is below ``threshold``.

        Single quantities with nonzero information are included (index 1).
        Ordered largest first, then by increasing index, so the first entry is
        the largest identifiable subset.

        Parameters
        ----------
        threshold : float
            Collinearity index above which a subset is treated as not
            identifiable. Default 15.
        max_size : int, optional
            As for :meth:`collinearity_table`.
        """
        singles = [((n,), 1.0) for i, n in enumerate(self.names) if self.matrix[i, i] > 0.0]
        found = singles + [
            (s, g) for s, g in self.collinearity_table(max_size=max_size) if g < threshold
        ]
        return sorted(found, key=lambda sg: (-len(sg[0]), sg[1]))

    def largest_identifiable_subset(
        self, threshold: float = 15.0, max_size: int | None = None
    ) -> tuple[str, ...]:
        """The largest subset with collinearity index below ``threshold`` (the
        one with the smallest index among equal sizes); empty if none."""
        found = self.identifiable_subsets(threshold=threshold, max_size=max_size)
        return found[0][0] if found else ()


def _source_fields(source, names, values, transforms):
    """``(F, names, values, transforms)`` from a FisherInformation, a
    CalibrationResult, or a bare matrix."""
    from aquakin.integrate.calibrate import CalibrationResult
    from aquakin.integrate.fisher import FisherInformation

    if isinstance(source, FisherInformation):
        return source.matrix, source.names, source.values, source.transforms
    if isinstance(source, CalibrationResult):
        if source.joint_hessian is not None:
            F, nm = source.joint_hessian, source.joint_names
            vals = dict(source.params_named)
            for k, pools in enumerate(source.ic_named or []):
                vals.update({f"ic[{k}].{s}": v for s, v in pools.items()})
            tr = list(source.transforms) + ["positive_log"] * (len(nm) - len(source.transforms))
            return F, nm, vals, tr
        if source.hessian_unconstrained is None:
            raise ValueError("The CalibrationResult has no Hessian; calibrate with laplace=True.")
        return (
            source.hessian_unconstrained,
            source.parameter_names,
            source.params_named,
            source.transforms,
        )
    if names is None:
        raise ValueError("Pass names= when analysing a bare matrix.")
    return source, names, values, transforms


def identifiability(
    source,
    *,
    names: list[str] | None = None,
    values: dict[str, float] | None = None,
    transforms: list[str] | None = None,
    eig_keep: float = 1e-2,
    null_loading_tol: float = 1e-2,
) -> IdentifiabilityResult:
    """Practical-identifiability report from a Fisher information matrix.

    Parameters
    ----------
    source : FisherInformation, CalibrationResult or array_like
        The information to analyse: the output of
        :func:`aquakin.fisher_information`; a :class:`CalibrationResult`
        calibrated with ``laplace=`` (its joint Hessian when
        ``LaplaceConfig(include_ic=True)`` was used, otherwise the rate Hessian;
        use ``loss="nll"`` and ``method="gauss_newton"`` for the Fisher
        information); or a bare ``(n, n)`` matrix with ``names``.
    names : list of str, optional
        Row / column names, required for a bare matrix.
    values : dict, optional
        Physical values by name, for a bare matrix; enables ``std`` /
        ``relative_std``.
    transforms : list of str, optional
        Each row's transform for a bare matrix (default ``"none"``), mapping the
        matrix coordinates to physical values.
    eig_keep : float
        Relative eigenvalue floor below which a direction is flagged as not
        identifiable. The default matches :class:`LaplaceConfig`, so the flagged
        directions are the ones the Laplace posterior drops.
    null_loading_tol : float
        Smallest loading on an exactly non-identifiable direction (eigenvalue at
        or below 1e-10 of the largest) that counts as involvement. Such a
        direction's eigenvector is only determined to about the accuracy of the
        sensitivities, so a quantity the design does determine can show a tiny
        spurious loading on it. Quantities with larger loadings get an infinite
        standard deviation; the others keep the finite one from the remaining
        directions. The default 1e-2 ignores loadings carrying less than 0.01 %
        of the direction's weight.

    Returns
    -------
    IdentifiabilityResult

    Examples
    --------
    >>> fim = aquakin.fisher_information(reactor, design, free_params,
    ...                                  free_ic=aquakin.FreeICConfig(["X"]))
    >>> rep = aquakin.identifiability(fim)
    >>> rep.unidentifiable[0].loadings     # e.g. {"r1.k": 0.71, "ic[0].X": -0.71}
    >>> rep.largest_identifiable_subset()
    """
    F, nm, vals, tr = _source_fields(source, names, values, transforms)
    F = np.asarray(F, dtype=float)
    nm = list(nm)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] != len(nm):
        raise ValueError(f"matrix shape {F.shape} does not match {len(nm)} names.")
    if not (0.0 < null_loading_tol < 1.0):
        raise ValueError(f"null_loading_tol must be in (0, 1); got {null_loading_tol}.")
    if not (0.0 <= eig_keep < 1.0):
        raise ValueError(f"eig_keep must be in [0, 1); got {eig_keep}.")
    F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)
    w_max = float(w[-1])
    if not np.isfinite(w_max) or w_max <= 0.0:
        raise ValueError("The information matrix has no positive eigenvalue.")

    directions = []
    for k in range(len(w)):
        a = np.abs(V[:, k])
        lead = int(np.argmax(a > _LOADING_TOL * a.max()))
        v = V[:, k] * np.sign(V[lead, k])
        order = np.argsort(-np.abs(v))
        directions.append(
            EigenDirection(
                eigenvalue=float(w[k]),
                relative_eigenvalue=float(w[k] / w_max),
                identifiable=bool(w[k] > eig_keep * w_max),
                loadings={nm[i]: float(v[i]) for i in order},
            )
        )

    # Exactly non-identifiable directions, with their round-off loadings
    # (below null_loading_tol) cleared so they do not drag in quantities the
    # design does determine.
    null = w <= _NULL_RTOL * w_max
    V_null = np.where(np.abs(V[:, null]) > null_loading_tol, V[:, null], 0.0)
    involved = (V_null != 0.0).any(axis=1)

    # Correlation of (F + delta I)^{-1}, built from the eigen-decomposition with
    # the cleaned null directions, so an exactly non-identifiable pair reads as
    # +/-1 and an uninvolved quantity is not correlated with it by round-off.
    delta = _NULL_RTOL * w_max
    cov_reg = (V[:, ~null] / (w[~null] + delta)) @ V[:, ~null].T + (V_null / delta) @ V_null.T
    sd_reg = np.sqrt(np.diag(cov_reg))
    correlation = cov_reg / np.outer(sd_reg, sd_reg)

    std = relative_std = None
    if vals is not None:
        tr = list(tr) if tr is not None else ["none"] * len(nm)
        var = (V[:, ~null] ** 2) @ (1.0 / w[~null])
        sd = np.where(involved, np.inf, np.sqrt(var))
        dphys = np.asarray(
            [float(dphysical_dunconstrained(vals[n], t, xp=np)) for n, t in zip(nm, tr)]
        )
        sd_phys = sd * np.abs(dphys)
        std = {n: float(s) for n, s in zip(nm, sd_phys)}
        relative_std = {n: float(s / abs(vals[n])) for n, s in zip(nm, sd_phys)}

    return IdentifiabilityResult(
        names=nm,
        matrix=F,
        eigenvalues=w,
        eigenvectors=V,
        directions=directions,
        correlation=correlation,
        std=std,
        relative_std=relative_std,
        eig_keep=eig_keep,
    )
