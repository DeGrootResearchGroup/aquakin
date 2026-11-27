"""The calibration result: :class:`CalibrationResult`, the posterior-predictive
:class:`PredictiveBand` and the sampled posterior :class:`PosteriorSamples`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from aquakin.integrate._transforms import from_unconstrained, to_unconstrained
from aquakin.integrate.calibration.saved import read_calibration, save_calibration

if TYPE_CHECKING:
    from aquakin.integrate._common import Reactor


# --- Result dataclasses ------------------------------------------------


class _FitRecord(NamedTuple):
    """What a fit leaves on its :class:`CalibrationResult` for the posterior
    sampler: the resolved problem, the forward model it was fitted through, and
    the MAP in the unconstrained coordinates (rates, then fitted pools)."""

    problem: object
    forward_model: object
    theta_map: np.ndarray


@dataclass
class PosteriorSamples:
    """Draws from the posterior of a calibration, from :func:`aquakin.sample_posterior`.

    Attributes
    ----------
    draws : np.ndarray
        Full parameter vectors, shape ``(n_chains * n_draws, n_params)``, laid
        out as :meth:`CalibrationResult.posterior_draws` lays them out: the free
        parameters drawn, the fixed ones at their calibrated values. Chain 0's
        draws come first. Pass it as ``draws=`` to
        :meth:`CalibrationResult.predictive_band`, or as ``param_draws`` to the
        experiment-design functions.
    parameter_names : list[str]
        The free rate parameters, in the order of ``params_named``.
    params_named : dict[str, np.ndarray]
        The draws of each free parameter in physical units, shape
        ``(n_chains * n_draws,)``, by name.
    ic_draws : list[dict[str, np.ndarray]] or None
        With fitted initial pools, the draws of each dataset's pools in physical
        units, by species (the layout of ``CalibrationResult.ic_named``).
        ``None`` when no pools were fitted.
    chains : np.ndarray
        The chains in the sampled (unconstrained) coordinates, shape
        ``(n_chains, n_draws, n_coordinates)``, columns as ``coordinate_names``.
    coordinate_names : list[str]
        The sampled coordinates: the rate names, then ``"ic[k].<species>"`` for
        the log of each fitted pool (``"ic[k].<c_j>:<c_0>"`` for the log-ratio
        coordinates of a fixed-total split).
    r_hat : dict[str, float]
        Rank-normalized split R-hat of each coordinate, the larger of its bulk
        and folded (tail) values. Close to 1 when the chains agree; above about
        1.01 the chains have not converged to the same distribution.
    ess_bulk : dict[str, float]
        Bulk effective sample size of each coordinate, over every chain: the
        number of independent draws carrying the same information about the
        centre of the distribution. The Monte Carlo error of a posterior mean is
        about ``std / sqrt(ess_bulk)``.
    acceptance_rate : np.ndarray
        Fraction of proposals each chain accepted after the warm-up, shape
        ``(n_chains,)``.
    """

    draws: np.ndarray
    parameter_names: list[str]
    params_named: dict[str, np.ndarray]
    ic_draws: list | None
    chains: np.ndarray
    coordinate_names: list[str]
    r_hat: dict[str, float]
    ess_bulk: dict[str, float]
    acceptance_rate: np.ndarray


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
        prior penalty. It is the same value regardless of ``optimizer``. For
        ``loss="nll"`` the data loss is the Gaussian negative log-likelihood
        less its constant ``(N/2) log(2 pi)``: it includes the
        ``sum(log(sigma))`` normalizer, which the Gauss-Newton
        ``0.5*||residual||^2`` drops. It is therefore not ``-log_likelihood``;
        use ``log_likelihood`` for model comparison.
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
    log_likelihood : float, optional
        With ``loss="nll"``, the Gaussian log-likelihood of the observations at
        the fitted parameters, complete with its ``-(N/2) log(2 pi)`` constant
        and without any prior penalty (with ``sigma="estimate"``, at the
        estimated noise levels). It is the maximized log-likelihood when the fit
        had no prior. ``None`` for the other losses, which are not likelihoods.
    n_observations : int, optional
        The number of observed (non-``NaN``) entries over every dataset.
    n_estimated : int, optional
        The number of estimated quantities ``k`` the information criteria
        count: the free rate parameters, the fitted initial-pool coordinates of
        every dataset, and with ``sigma="estimate"`` one noise level per
        observed output. See :func:`aquakin.aic`.
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
    log_likelihood: float | None = None
    n_observations: int | None = None
    n_estimated: int | None = None
    # What :func:`aquakin.compare_models` checks: a fingerprint of the
    # observations and their times, and whether the objective carried a prior.
    _observations_key: str | None = field(default=None, repr=False)
    _prior_in_loss: bool = field(default=False, repr=False)
    # The resolved problem and forward model the fit ran on, for
    # :func:`aquakin.sample_posterior` (``None`` on a result built by hand).
    _fit_record: _FitRecord | None = field(default=None, repr=False, compare=False)
    # On a result read by :func:`load_calibration`: the names and model
    # fingerprints it was saved with, so it can be saved again.
    _model_record: dict | None = field(default=None, repr=False, compare=False)

    def save(self, folder, *, model=None):
        """Save the fit to a folder of plain files, to read back with
        :func:`aquakin.load_calibration`.

        The folder holds ``fit.yaml`` (the summary: fitted values and their
        std, loss, optimizer status, transforms, priors, the
        information-criteria counts, fitted initial pools, the fingerprint of
        each model and the aquakin version), ``parameters.yaml`` (the full
        parameter vector by name) and, when the fit has them, the matrices
        ``covariance.csv``, ``hessian.csv``, ``joint_covariance.csv``,
        ``joint_hessian.csv`` and the fitted initial states
        ``initial_states.csv``, every row and column named. Floats are written
        with ``repr``, so every number reads back exactly. The forward model
        and the objective the fit ran on are not saved.

        Parameters
        ----------
        folder : str or pathlib.Path
            The folder to write (created if needed). It must not hold a saved
            fit already.
        model : CompiledModel, optional
            The calibrated model (``reactor.model``), which names the full
            parameter vector. Only needed for a result that came from neither
            a fit nor :func:`aquakin.load_calibration` (built by hand); such a
            result also has no model fingerprint other than this model's.

        Returns
        -------
        pathlib.Path
            The folder.

        Raises
        ------
        FileExistsError
            If the folder already holds a saved fit's files.
        ValueError
            If the result does not know its model and ``model`` is not given,
            or ``model`` does not have the result's number of parameters.
        """
        return save_calibration(self, folder, model)

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
        observed_species: list[str] | None = None,
        draws: np.ndarray | None = None,
    ) -> PredictiveBand:
        """Posterior-predictive band by propagating posterior draws through a solve.

        Samples the calibrated parameters from the Laplace posterior with
        :meth:`posterior_draws` (or takes the given ``draws``), integrates
        ``reactor`` from ``C0`` over ``t_eval``, and returns the pointwise
        percentile envelope across draws. The Laplace draws are taken from
        ``self.posterior_cov``, the covariance behind ``params_named_std``. A
        parameter on an exactly null Hessian direction (infinite std) is not
        sampled, so the draws stay finite.

        Without ``draws``, requires that the calibration was run with
        ``laplace=True`` (so a Hessian is available). The supplied ``C0`` may
        differ from the calibration initial state --- e.g. propagate the
        calibrated-rate uncertainty through a held-out / validation batch.

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
        observed_species : list[str], optional
            Restrict the returned band to these species. ``None`` returns all.
        draws : np.ndarray, optional
            Full parameter vectors to propagate, shape ``(n, n_params)``, such
            as ``PosteriorSamples.draws`` from :func:`aquakin.sample_posterior`.
            ``n_draw`` and ``seed`` are then not used. ``None`` (default) draws
            from the Laplace posterior.

        Returns
        -------
        PredictiveBand

        Raises
        ------
        ValueError
            If ``draws`` is not given and there is no Laplace posterior, or
            ``draws`` does not have one column per model parameter.
        """
        if draws is None and self.posterior_cov is None:
            raise ValueError(
                "predictive_band requires a Laplace posterior; call calibrate(..., laplace=True)."
            )
        model = reactor.model
        if draws is None:
            draws = self.posterior_draws(model, n_draw, seed=seed)
        draws = np.atleast_2d(np.asarray(draws, dtype=float))
        if draws.shape[1] != model.n_params:
            raise ValueError(
                f"draws has {draws.shape[1]} columns; the model has "
                f"{model.n_params} parameters (pass full parameter vectors, "
                "such as PosteriorSamples.draws)."
            )
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


def load_calibration(folder) -> CalibrationResult:
    """Read a fit saved by :meth:`CalibrationResult.save`.

    Every number the fit carried comes back exactly, including what the
    information criteria check, so :func:`aquakin.aic` and
    :func:`aquakin.compare_models` work on it as on the fit itself. Methods
    that take the model or reactor (:meth:`~CalibrationResult.posterior_draws`,
    :meth:`~CalibrationResult.predictive_band`) work too. The forward model the
    fit ran on is not saved, so :func:`aquakin.sample_posterior` needs the
    result a fit returns.

    Parameters
    ----------
    folder : str or pathlib.Path
        The folder written by :meth:`CalibrationResult.save`.

    Returns
    -------
    CalibrationResult

    Raises
    ------
    FileNotFoundError
        If the folder does not exist, or a file of the saved fit is missing.
    ValueError
        If a file is damaged, or the fit was saved in another format (by a
        newer aquakin).
    """
    return CalibrationResult(**read_calibration(folder))
