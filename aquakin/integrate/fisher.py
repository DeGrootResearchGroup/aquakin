"""Fisher information of a batch-reactor experiment design.

For Gaussian measurement noise the Fisher information of a set of experiments is

    F = sum_e  S_e^T  Sigma_e^{-1}  S_e,

where ``S_e`` is the sensitivity of experiment ``e``'s predicted observations to
the estimated quantities and ``Sigma_e = diag(sigma_e^2)`` its noise covariance.
``F^{-1}`` is the Cramer-Rao lower bound on the covariance of any unbiased
estimator, so ``F`` measures how well a design pins the parameters down *before*
the experiment is run. It needs only assumed parameter values and the design
(initial states, sampling times, observed species, noise levels), never data.

:func:`fisher_information` evaluates ``F`` in the same unconstrained coordinates
:func:`aquakin.calibrate` fits in -- each rate parameter through its transform
and each fitted initial pool in log space -- so at a calibration's MAP it equals
the Gauss-Newton Laplace Hessian (``loss="nll"``, ``method="gauss_newton"``).
The sensitivities come from forward-mode AD through the reactor solve: one
Jacobian-vector product per estimated quantity, which is the cheap direction
when there are many more observations than parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._common import (
    Reactor,
    check_finite_gradient,
    forward_adjoint,
    native_time_factor,
    with_adjoint,
)
from aquakin.integrate._transforms import (
    dphysical_dunconstrained,
    from_unconstrained,
    to_unconstrained,
)
from aquakin.integrate.calibrate import FreeICConfig


@dataclass(frozen=True)
class Experiment:
    """One batch experiment in a design.

    Parameters
    ----------
    C0 : jnp.ndarray
        Initial concentration vector, shape ``(n_species,)``. When initial pools
        are estimated (``free_ic``), their entries here are the assumed values
        the information is evaluated at, and must be positive.
    t_obs : jnp.ndarray
        Sampling times, strictly ascending and non-negative, shape ``(n_t,)``.
        The solve starts at ``t = 0``.
    sigma : float or jnp.ndarray
        Measurement standard deviation: a scalar, one value per observed
        species ``(n_observed,)``, or one per observation ``(n_t, n_observed)``.
    observed_species : list of str, optional
        Species that are measured. ``None`` (default) measures every species.
    reactor : Reactor, optional
        A reactor to solve this experiment with in place of the one passed to
        :func:`fisher_information` -- for an experiment run at different
        conditions (pH, temperature, ...). It must use the same model.
    time_unit : str, optional
        Unit of ``t_obs``; ``None`` means the model's native time unit.
    """

    C0: jnp.ndarray
    t_obs: jnp.ndarray
    sigma: float | jnp.ndarray
    observed_species: list[str] | None = None
    reactor: Reactor | None = None
    time_unit: str | None = None


@dataclass
class FisherInformation:
    """Result of :func:`fisher_information`.

    Attributes
    ----------
    matrix : np.ndarray
        ``(n, n)`` Fisher information in unconstrained coordinates (each rate
        parameter through its transform, each initial pool as its log), ordered
        as ``names``.
    names : list[str]
        Row / column names: the rate parameters, then ``"ic[e].<species>"`` for
        experiment ``e``'s estimated initial pools.
    transforms : list[str]
        Transform of each row: the rate parameters' transforms, then
        ``"positive_log"`` for every pool.
    values : dict[str, float]
        The physical values the information was evaluated at, by name.
    jacobian : np.ndarray
        ``(n_obs_total, n)`` noise-scaled sensitivity ``Sigma^{-1/2} S`` stacked
        over all experiments (plus prior rows if priors were included), so that
        ``matrix == jacobian.T @ jacobian``.
    """

    matrix: np.ndarray
    names: list[str]
    transforms: list[str]
    values: dict[str, float]
    jacobian: np.ndarray = field(repr=False)

    def _dphys(self) -> np.ndarray:
        return np.asarray(
            [
                float(dphysical_dunconstrained(self.values[n], t, xp=np))
                for n, t in zip(self.names, self.transforms)
            ]
        )

    def physical(self) -> np.ndarray:
        """The Fisher information with respect to the physical values,
        ``D^{-1} F D^{-1}`` with ``D = diag(d physical / d unconstrained)``."""
        inv_d = 1.0 / self._dphys()
        return self.matrix * np.outer(inv_d, inv_d)

    def covariance(self) -> np.ndarray:
        """The Cramer-Rao covariance bound ``F^{-1}`` in unconstrained coordinates.

        Raises
        ------
        np.linalg.LinAlgError
            If ``F`` is singular: some parameter combination is not identifiable
            from this design.
        """
        return np.linalg.inv(self.matrix)

    def std(self) -> dict[str, float]:
        """Lower bound on each quantity's standard deviation in physical units,
        by name: the diagonal of :meth:`covariance` mapped through the delta
        method."""
        sd = np.sqrt(np.diag(self.covariance())) * np.abs(self._dphys())
        return {n: float(v) for n, v in zip(self.names, sd)}

    def relative_std(self) -> dict[str, float]:
        """:meth:`std` divided by each quantity's magnitude."""
        return {n: s / abs(self.values[n]) for n, s in self.std().items()}


def _resolve_transforms(model, free_params, transforms) -> list[str]:
    """Each free parameter's transform: the explicit override, else the
    model-declared one, else ``"none"``."""
    transforms = dict(transforms or {})
    return [transforms.get(n) or model.parameter_transforms.get(n, "none") for n in free_params]


def fisher_information(
    reactor: Reactor,
    experiments: Experiment | list[Experiment],
    free_params: list[str],
    *,
    params: jnp.ndarray | None = None,
    transforms: dict[str, str] | None = None,
    free_ic: FreeICConfig | None = None,
    priors: dict[str, tuple[float, float]] | None = None,
    use_priors: bool = False,
) -> FisherInformation:
    """Fisher information of a design at assumed parameter values.

    Parameters
    ----------
    reactor : Reactor
        Reactor that solves each experiment (unless the experiment carries its
        own). Its integrator settings are used as-is; for a stiff model, set
        ``IntegratorConfig(dtmax=...)`` on it as for any AD through a solve.
    experiments : Experiment or list of Experiment
        The design. The information of independent experiments adds.
    free_params : list of str
        Rate parameters to estimate (namespaced names).
    params : jnp.ndarray, optional
        Full parameter vector the information is evaluated at. Defaults to
        ``model.default_parameters()``.
    transforms : dict, optional
        Per-parameter transform override (``"positive_log"``, ``"logit"``,
        ``"none"``), as in :func:`aquakin.calibrate`; otherwise the model's
        declared transform is used.
    free_ic : FreeICConfig, optional
        Initial pools to estimate alongside the rates, one set per experiment,
        evaluated at the values in each experiment's ``C0``. Only
        ``free_ic.species`` (and, with ``use_priors``, ``prior_log_std``) is used.
    priors : dict, optional
        Gaussian priors ``{name: (mean, std)}`` on rate parameters, in physical
        space. Used only with ``use_priors=True``.
    use_priors : bool
        Add the prior precision (model-declared priors, ``priors``, and the
        ``free_ic`` log-space prior) to ``F``. Default ``False``: the information
        from the data alone.

    Returns
    -------
    FisherInformation
        The matrix with named rows, plus helpers for the physical-space matrix
        and the Cramer-Rao standard deviations.

    Raises
    ------
    KeyError
        If a parameter or species name is unknown.
    ValueError
        If an experiment is malformed or an estimated pool is not positive.

    Examples
    --------
    >>> design = [aquakin.Experiment(C0, t_obs=jnp.linspace(1, 60, 20),
    ...                              sigma=0.02, observed_species=["O3"])]
    >>> fim = aquakin.fisher_information(reactor, design, ["r1.k1", "r2.k2"],
    ...                                  free_ic=aquakin.FreeICConfig(["IOD"]))
    >>> fim.std()       # Cramer-Rao lower bounds, physical units
    """
    setup = _FisherSetup(
        reactor,
        experiments,
        free_params,
        params=params,
        transforms=transforms,
        free_ic=free_ic,
        priors=priors,
        use_priors=use_priors,
    )
    J = setup.jacobian(setup.p0, setup.tobs, setup.C0)
    check_finite_gradient(
        J,
        what="the Fisher-information sensitivity",
        remedy="cap the reactor's integrator step with IntegratorConfig(dtmax=...)",
    )
    J = np.asarray(J)
    F = J.T @ J
    return FisherInformation(
        matrix=0.5 * (F + F.T),
        names=setup.names,
        transforms=setup.transforms,
        values=setup.values(setup.p0, setup.C0),
        jacobian=J,
    )


class _FisherSetup:
    """The resolved, static part of a Fisher-information evaluation.

    Validates the arguments once and keeps everything that does not change
    when the design does (reactors, observed columns, noise, transforms,
    priors). :meth:`jacobian` is then a pure JAX function of the full parameter
    vector, the per-experiment sampling times and the per-experiment initial
    states, so it can be differentiated with respect to the design or mapped
    over parameter draws. ``t_end`` (a value, or a list with one entry or
    ``None`` per experiment) overrides each experiment's integration end
    (default: its last sampling time), so sampling times can move under
    tracing while the solve interval stays fixed.
    """

    def __init__(
        self,
        reactor,
        experiments,
        free_params,
        *,
        params=None,
        transforms=None,
        free_ic=None,
        priors=None,
        use_priors=False,
        t_end=None,
    ):
        model = reactor.model
        exps = [experiments] if isinstance(experiments, Experiment) else list(experiments)
        if not exps:
            raise ValueError("experiments must contain at least one Experiment.")
        for name in free_params:
            if name not in model.param_index:
                raise KeyError(f"Unknown parameter '{name}'. Available: {model.parameters}")
        self.model = model
        self.free_params = list(free_params)
        self.rate_transforms = _resolve_transforms(model, free_params, transforms)
        self.p0 = jnp.asarray(params) if params is not None else model.default_parameters()
        self.free_idx = jnp.asarray([model.param_index[n] for n in free_params], dtype=int)

        self.ic_species = list(free_ic.species) if free_ic is not None else []
        for s in self.ic_species:
            if s not in model.species_index:
                raise KeyError(f"Unknown free_ic species '{s}'. Available: {model.species}")
        self.ic_idx = jnp.asarray([model.species_index[s] for s in self.ic_species], dtype=int)
        if not self.free_params and not self.ic_species:
            raise ValueError("Nothing to estimate: give free_params and/or free_ic.")

        # Resolve each experiment once: its forward-capable reactor, native-unit
        # sampling times, observed columns and per-observation noise.
        self.reactors, self.obs_idx, self.sigma, self.tobs, self.C0, self.t_end = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for e, ex in enumerate(exps):
            rctr = ex.reactor if ex.reactor is not None else reactor
            if list(rctr.model.parameters) != list(model.parameters):
                raise ValueError(f"experiment {e}: its reactor uses a different model.")
            C0 = jnp.asarray(ex.C0, dtype=float)
            if C0.shape != (model.n_species,):
                raise ValueError(
                    f"experiment {e}: C0 has shape {C0.shape}, expected ({model.n_species},)."
                )
            tobs = jnp.asarray(ex.t_obs, dtype=float)
            if tobs.ndim != 1 or tobs.shape[0] < 1:
                raise ValueError(f"experiment {e}: t_obs must be a non-empty 1-D array.")
            if float(tobs[0]) < 0.0 or not bool(jnp.all(jnp.diff(tobs) > 0)):
                raise ValueError(
                    f"experiment {e}: t_obs must be non-negative and strictly ascending."
                )
            tobs = tobs * native_time_factor(model.time_unit, ex.time_unit)
            obs = ex.observed_species if ex.observed_species is not None else list(model.species)
            for s in obs:
                if s not in model.species_index:
                    raise KeyError(f"experiment {e}: unknown observed species '{s}'.")
            if self.ic_species:
                pools = np.asarray(C0)[np.asarray(self.ic_idx)]
                if np.any(pools <= 0.0):
                    raise ValueError(
                        f"experiment {e}: estimated initial pools {self.ic_species} must be "
                        f"positive in C0; got {pools.tolist()}."
                    )
            self.reactors.append(with_adjoint(rctr, forward_adjoint()))
            self.obs_idx.append(jnp.asarray([model.species_index[s] for s in obs], dtype=int))
            self.sigma.append(
                jnp.broadcast_to(jnp.asarray(ex.sigma, dtype=float), (tobs.shape[0], len(obs)))
            )
            self.tobs.append(tobs)
            self.C0.append(C0)
            end = t_end[e] if isinstance(t_end, (list, tuple)) else t_end
            self.t_end.append(float(tobs[-1]) if end is None else float(end))

        # Prior rows (only their Jacobian matters, so the means drop out).
        active: dict[str, tuple[float, float]] = {}
        if use_priors:
            declared = getattr(model, "parameter_priors", {})
            active = {n: declared[n] for n in free_params if n in declared}
            for n, ms in (priors or {}).items():
                if n in free_params:
                    active[n] = (float(ms[0]), float(ms[1]))
        self.has_rate_prior = bool(active)
        self.prior_mask = jnp.asarray([1.0 if n in active else 0.0 for n in free_params])
        self.prior_std = jnp.asarray([active.get(n, (0.0, 1.0))[1] for n in free_params])
        m = len(self.ic_species)
        self.ic_prior = (
            free_ic.prior_log_std if (use_priors and free_ic is not None and m) else None
        )

        ic_names = [f"ic[{e}].{s}" for e in range(len(exps)) for s in self.ic_species]
        self.names = self.free_params + ic_names
        self.transforms = self.rate_transforms + ["positive_log"] * len(ic_names)

    def values(self, p, C0s) -> dict[str, float]:
        """Physical values of the estimated quantities at ``(p, C0s)``."""
        rate = [float(p[self.model.param_index[n]]) for n in self.free_params]
        pools = [float(C0[i]) for C0 in C0s for i in np.asarray(self.ic_idx)]
        return dict(zip(self.names, rate + pools))

    def jacobian(self, p, tobs_list, C0_list):
        """Noise-scaled sensitivity ``Sigma^{-1/2} S`` at parameters ``p`` for
        sampling times ``tobs_list`` and initial states ``C0_list`` (one per
        experiment), in unconstrained coordinates. Pure JAX."""
        d, m = len(self.free_params), len(self.ic_species)
        trs = self.rate_transforms

        def _residuals(theta):
            physical = jnp.stack([from_unconstrained(theta[i], trs[i]) for i in range(d)])
            pp = p.at[self.free_idx].set(physical)
            parts = []
            for e, (rctr, C0, tobs) in enumerate(zip(self.reactors, C0_list, tobs_list)):
                if m:
                    C0 = C0.at[self.ic_idx].set(jnp.exp(theta[d + e * m : d + (e + 1) * m]))
                C = rctr.solve(C0, params=pp, t_span=(0.0, self.t_end[e]), t_eval=tobs).C
                parts.append((C[:, self.obs_idx[e]] / self.sigma[e]).reshape(-1))
            if self.has_rate_prior:
                parts.append(self.prior_mask * physical / self.prior_std)
            if self.ic_prior:
                parts.append(theta[d:] / self.ic_prior)
            return jnp.concatenate(parts)

        rate_theta = [jnp.atleast_1d(to_unconstrained(p[i], t)) for i, t in zip(self.free_idx, trs)]
        pool_theta = [jnp.log(C0[self.ic_idx]) for C0 in C0_list] if m else []
        theta = jnp.concatenate(rate_theta + pool_theta)
        return jax.jacfwd(_residuals)(theta)

    def information(self, p, tobs_list, C0_list):
        """Fisher information ``J^T J`` at ``(p, tobs_list, C0_list)``. Pure JAX."""
        J = self.jacobian(p, tobs_list, C0_list)
        return J.T @ J
