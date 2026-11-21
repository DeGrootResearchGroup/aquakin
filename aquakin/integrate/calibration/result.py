"""The calibration result: :class:`CalibrationResult` and the posterior-predictive
:class:`PredictiveBand`."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aquakin.integrate._transforms import from_unconstrained, to_unconstrained

if TYPE_CHECKING:
    from aquakin.integrate._common import Reactor


# --- Result dataclasses ------------------------------------------------


@dataclass
class PredictiveBand:
    """Posterior-predictive band from :meth:`CalibrationResult.predictive_band`.

    Attributes
    ----------
    t : np.ndarray
        Output time grid, shape ``(n_t,)`` (the ``t_eval`` passed in).
    median : np.ndarray
        Pointwise median over the posterior draws, shape ``(n_t, n_species)``
        (or ``(n_t, n_observed)`` if ``observed_species`` was given).
    lo, hi : np.ndarray
        Lower / upper percentile envelopes, same shape as ``median``.
    percentiles : tuple[float, float]
        The ``(lo, hi)`` percentiles used.
    n_valid : int
        Number of posterior draws that solved to a finite trajectory and were
        included in the percentiles.
    species : list[str] or None
        Observed-species labels for the columns, or ``None`` for all species.
    """

    t: np.ndarray
    median: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    percentiles: tuple[float, float]
    n_valid: int
    species: list[str] | None = None


@dataclass
class CalibrationResult:
    """Result of :func:`calibrate`.

    Attributes
    ----------
    params : jnp.ndarray
        Full parameter vector after optimisation (fixed entries unchanged).
    params_named : dict[str, float]
        Free parameters by namespaced name, in physical space.
    loss : float
        The full scalar objective at the MAP: the summed data loss plus any
        prior penalty. It is the same value regardless of ``optimizer`` --- for
        ``loss="nll"`` it is the complete Gaussian negative log-likelihood
        (including the ``sum(log(sigma))`` normaliser), not the Gauss-Newton
        ``0.5*||residual||^2`` (which drops that constant).
    converged : bool
        Whether SciPy declared convergence.
    message : str
        Optimiser status message.
    n_iter : int
        Optimiser iteration count: ``nit`` for the L-BFGS-B optimiser, and the
        Jacobian-evaluation count (``njev``, ~one per trust-region iteration) for
        the Gauss-Newton least-squares optimiser, which exposes no direct
        iteration count. Either way it is an iteration-scale figure, not the raw
        residual/function-evaluation count.
    parameter_names : list[str]
        Ordered free-parameter names. Matches the rows / cols of
        ``posterior_cov`` and ``hessian_unconstrained`` when present.
    transforms : list[str]
        Transform used for each free parameter, in the same order.
    posterior_cov : jnp.ndarray, optional
        ``(d, d)`` covariance matrix of the rate parameters in *unconstrained*
        space (Laplace). With ``LaplaceConfig(include_ic=True)`` it is the rate
        block of ``joint_cov`` (marginal over the fitted pools); otherwise it is
        conditional on the pools held at their MAP. ``None`` if
        ``laplace=False`` was passed.
    posterior_std_unconstrained : jnp.ndarray, optional
        Marginal std devs in unconstrained space.
    params_named_std : dict[str, float], optional
        Marginal std devs propagated back to physical space via the
        delta method (``|dp/dtheta| * std_unconstrained``).
    hessian_unconstrained : jnp.ndarray, optional
        Rate-rate block of the loss Hessian at the MAP, unconstrained space:
        finite-difference for ``LaplaceConfig(method="fd")``, Gauss-Newton
        ``J^T J`` (the Fisher information for ``loss="nll"``) for
        ``method="gauss_newton"``.
    priors_applied : dict[str, tuple[float, float]], optional
        Gaussian priors (physical-space ``(mean, std)``) that were added to the
        objective, by free-parameter name. Empty if no priors were active.
    C0_fitted : list[jnp.ndarray], optional
        When ``free_ic`` is used, the fitted initial-state vector for each
        dataset (full species vectors, free pools set to their fitted values).
        ``None`` if no initial conditions were fit.
    ic_named : list[dict[str, float]], optional
        When ``free_ic`` is used, the fitted free initial pools per dataset, by
        species name. ``None`` if no initial conditions were fit.
    joint_cov : jnp.ndarray, optional
        With ``LaplaceConfig(include_ic=True)`` and fitted pools, the joint
        covariance of the rate parameters and the fitted pools in unconstrained
        space (pools in log space), ordered as ``joint_names``. ``None``
        otherwise.
    joint_hessian : jnp.ndarray, optional
        The joint Hessian ``joint_cov`` was built from, same ordering.
    joint_names : list[str], optional
        Row/column names of ``joint_cov``: the rate names, then
        ``"ic[k].<species>"`` for each dataset ``k``'s fitted pools.
    ic_named_std : list[dict[str, float]], optional
        Marginal std devs of the fitted pools in physical units, per dataset
        and by species (same layout as ``ic_named``). Only with
        ``include_ic=True``.
    ic_fractions : list[dict[str, float]], optional
        With ``FreeICConfig(fixed_total=True)``, the fitted split of each
        dataset's total, as fractions summing to 1, by species. ``None``
        otherwise.
    ic_fractions_std : list[dict[str, float]], optional
        Std devs of ``ic_fractions`` (delta method from the joint covariance of
        the log-ratio coordinates). Only with ``fixed_total`` and
        ``include_ic=True``.
    at_bound : dict[str, str]
        The fitted quantities that ended on an optimiser bound, mapped to
        ``"lower"`` or ``"upper"``: free rate parameters by name and fitted
        initial pools as ``"ic[k].<species>"``. Empty when nothing is on a
        bound. The bound is an active constraint there, so the local curvature
        does not describe their uncertainty: their ``params_named_std`` /
        ``ic_named_std`` and ``posterior_std_unconstrained`` entries are
        ``nan``, they have zero variance in ``posterior_cov`` / ``joint_cov``
        (posterior draws keep them on the bound), and the other parameters'
        covariance is conditional on them staying on the bound. A
        ``RuntimeWarning`` names them. Use :func:`aquakin.profile_likelihood`
        for a one-sided interval, or revisit the bound.
    sigma_estimate : np.ndarray, optional
        With ``loss="nll", sigma="estimate"``, the estimated noise standard
        deviation of each observed output, ``sqrt(SSE_j / N_j)`` at the MAP
        (pooled over the datasets), shape ``(n_observed,)``. ``None``
        otherwise.
    """

    params: jnp.ndarray
    params_named: dict[str, float]
    loss: float
    converged: bool
    message: str
    n_iter: int
    parameter_names: list[str]
    transforms: list[str]
    posterior_cov: jnp.ndarray | None = None
    posterior_std_unconstrained: jnp.ndarray | None = None
    params_named_std: dict[str, float] | None = field(default=None)
    hessian_unconstrained: jnp.ndarray | None = None
    priors_applied: dict[str, tuple[float, float]] = field(default_factory=dict)
    C0_fitted: list | None = None
    ic_named: list | None = None
    joint_cov: jnp.ndarray | None = None
    joint_hessian: jnp.ndarray | None = None
    joint_names: list[str] | None = None
    ic_named_std: list | None = None
    ic_fractions: list | None = None
    ic_fractions_std: list | None = None
    at_bound: dict[str, str] = field(default_factory=dict)
    sigma_estimate: np.ndarray | None = None

    def posterior_draws(self, model, n_draw: int = 200, *, seed: int = 0) -> np.ndarray:
        """Full parameter vectors drawn from the Laplace posterior.

        Draws the free rate parameters from ``N(MAP, posterior_cov)`` in
        unconstrained space, maps each draw to physical values, and sets it into
        the fitted parameter vector ``params``; fixed parameters keep their
        values. An exactly null (non-identifiable) direction of the Hessian is
        not in ``posterior_cov`` and is not sampled; such parameters have an
        infinite ``params_named_std``. A parameter that ended on an optimiser
        bound (``at_bound``) has zero variance and stays on the bound in every
        draw. Initial pools are not drawn. The
        rows can be passed as ``param_draws`` to the experiment-design
        functions, and :meth:`predictive_band` propagates the same draws.

        Parameters
        ----------
        model : CompiledModel
            The calibrated model (``reactor.model``), to place the free
            parameters in the full vector.
        n_draw : int
            Number of draws.
        seed : int
            Seed for the draws (reproducible).

        Returns
        -------
        np.ndarray
            Shape ``(n_draw, n_params)``.

        Raises
        ------
        ValueError
            If there is no Laplace posterior, or it has no positive-variance
            direction.
        """
        if self.posterior_cov is None:
            raise ValueError(
                "posterior_draws requires a Laplace posterior; call calibrate(..., laplace=True)."
            )
        names = self.parameter_names
        free_idx = np.asarray([model.param_index[n] for n in names])
        theta_map = np.array(
            [
                float(to_unconstrained(jnp.asarray(self.params_named[n]), t))
                for n, t in zip(names, self.transforms)
            ]
        )

        # Draw from N(theta_map, posterior_cov), the covariance that also backs
        # params_named_std. Eigen-decompose to sample; an exactly null Hessian
        # direction has zero variance in posterior_cov and is not sampled (a
        # tiny relative floor drops it and any round-off negatives).
        cov = np.asarray(self.posterior_cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        s, Q = np.linalg.eigh(cov)
        pos = s > 1e-12 * max(float(s.max()), 0.0)
        if not np.any(pos):
            raise ValueError(
                "Posterior covariance has no positive-variance directions; the "
                "Laplace posterior is degenerate."
            )
        std_k = np.sqrt(s[pos])
        rng = np.random.default_rng(seed)
        thetas = (
            theta_map[None, :]
            + (rng.standard_normal((n_draw, int(pos.sum()))) * std_k[None, :]) @ Q[:, pos].T
        )
        out = np.tile(np.asarray(self.params, dtype=float), (n_draw, 1))
        for i, t in enumerate(self.transforms):
            out[:, free_idx[i]] = np.asarray(from_unconstrained(jnp.asarray(thetas[:, i]), t))
        return out

    def predictive_band(
        self,
        reactor: Reactor,
        C0: jnp.ndarray,
        t_eval: jnp.ndarray,
        *,
        n_draw: int = 200,
        percentiles: tuple[float, float] = (2.5, 97.5),
        seed: int = 0,
        eig_keep: float | None = None,
        observed_species: list[str] | None = None,
    ) -> PredictiveBand:
        """Posterior-predictive band by propagating Laplace draws through a solve.

        Samples the calibrated parameters from the Laplace posterior with
        :meth:`posterior_draws`, integrates ``reactor``
        from ``C0`` over ``t_eval``, and returns the pointwise percentile
        envelope across draws. The draws are taken from ``self.posterior_cov``,
        the covariance behind ``params_named_std``. A parameter on an exactly
        null Hessian direction (infinite std) is not sampled, so the draws stay
        finite.

        Requires that the calibration was run with ``laplace=True`` (so a Hessian
        is available). The supplied ``C0`` may differ from the calibration
        initial state --- e.g. propagate the calibrated-rate uncertainty through a
        held-out / validation batch.

        Parameters
        ----------
        reactor : Reactor
            Reactor to integrate (forward solves only; any adjoint is fine).
        C0 : jnp.ndarray
            Initial state to propagate.
        t_eval : jnp.ndarray
            Output times; the band is reported at these points.
        n_draw : int
            Number of posterior draws.
        percentiles : tuple[float, float]
            Lower / upper band percentiles (default the central 95%).
        seed : int
            Seed for the draws (reproducible).
        eig_keep : float, optional
            Deprecated and ignored. The band uses ``self.posterior_cov``, the
            covariance behind ``params_named_std``. Passing a value here emits
            a ``DeprecationWarning``.
        observed_species : list[str], optional
            Restrict the returned band to these species. ``None`` returns all.

        Returns
        -------
        PredictiveBand
        """
        if self.posterior_cov is None:
            raise ValueError(
                "predictive_band requires a Laplace posterior; call calibrate(..., laplace=True)."
            )
        if eig_keep is not None:
            warnings.warn(
                "predictive_band(eig_keep=...) is deprecated and ignored; the band "
                "is drawn from the calibration's posterior_cov.",
                DeprecationWarning,
                stacklevel=2,
            )
        model = reactor.model
        draws = self.posterior_draws(model, n_draw, seed=seed)
        C0_j = jnp.asarray(C0)
        t_eval_j = jnp.asarray(t_eval)
        t_end = float(np.asarray(t_eval)[-1])
        curves = []
        first_error = None
        for p in draws:
            p = jnp.asarray(p)
            try:
                cc = np.asarray(
                    reactor.solve(C0_j, params=p, t_span=(0.0, t_end), t_eval=t_eval_j).C
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc  # keep the first cause, to report if all fail
                continue
            if np.all(np.isfinite(cc)):
                curves.append(cc)
        if not curves:
            # Don't swallow the cause: surface the first solve failure (or note
            # that every draw merely went non-finite) instead of a bare message.
            if first_error is not None:
                raise RuntimeError(
                    "All posterior draws failed to solve; the first error is the "
                    "direct cause above."
                ) from first_error
            raise RuntimeError(
                "All posterior draws produced a non-finite trajectory (no "
                "exception raised) -- the Laplace draws may leave the feasible "
                "region; check the fit and params_named_std for poorly identified "
                "parameters."
            )
        curves = np.array(curves)
        lo = np.percentile(curves, percentiles[0], axis=0)
        hi = np.percentile(curves, percentiles[1], axis=0)
        median = np.percentile(curves, 50.0, axis=0)
        if observed_species is not None:
            sp_idx = [model.species_index[s] for s in observed_species]
            lo, hi, median = lo[:, sp_idx], hi[:, sp_idx], median[:, sp_idx]
        return PredictiveBand(
            t=np.asarray(t_eval),
            median=median,
            lo=lo,
            hi=hi,
            percentiles=tuple(percentiles),
            n_valid=len(curves),
            species=list(observed_species) if observed_species is not None else None,
        )
