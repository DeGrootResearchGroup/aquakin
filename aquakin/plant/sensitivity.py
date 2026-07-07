"""Plant steady-state and dynamic sensitivity / DGSM surface.

The derivative-based sensitivity and uncertainty-quantification layer of
:class:`~aquakin.plant.plant.Plant`, split out of ``plant.py``. Everything here
only *consumes* the public solve API (:meth:`Plant.steady_state`,
:meth:`Plant.solve`, :meth:`Plant.derivative`) plus the plant's parameter- and
state-layout helpers, so it is a thin layer on top of the flowsheet rather than
part of its assembly.

Two parallel stacks:

- **Steady state** -- :func:`steady_state_sensitivity` reads the exact output
  sensitivity at an operating point through the implicit function theorem (one
  ``dF/dy`` factorisation, reused across outputs and parameters), and
  :func:`steady_state_dgsm` screens it globally (scrambled-Sobol QMC) into the
  Sobol total-index bound :class:`SteadyStateDGSMResult`.
- **Dynamic** -- :func:`solve_sensitivity` integrates the augmented ``[y; S]``
  variational system, :func:`dynamic_sensitivity` differentiates a transient
  output through the stable adjoint (reverse) or that variational solve
  (forward), and :func:`dynamic_dgsm` screens it into :class:`DynamicDGSMResult`.

The five public entry points take the :class:`Plant` as their first argument and
are bound onto :class:`Plant` as methods (see ``plant.py``), so the public
``plant.steady_state_dgsm(...)`` API is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from aquakin.integrate._common import DifferentiationConfig, IntegratorConfig
from aquakin.integrate._transforms import (
    dphysical_dunconstrained,
    from_unconstrained,
    to_unconstrained,
)

# The Sobol total-index aggregation kernel is shared with the reactor DGSM
# (:func:`aquakin.dgsm`); the steady-state and dynamic screens below pass a
# ``sample_mask`` and, for Gaussian inputs, an explicit ``poincare`` constant.
# Re-exported here so ``from aquakin.plant.sensitivity import _dgsm_aggregate``
# keeps working.
from aquakin.integrate.global_sensitivity import _dgsm_aggregate


def _cond_mask(cond, cond_factor):
    """Keep-mask dropping near-singular-Jacobian samples.

    A steady-state sensitivity ``-J^{-1}(...)`` is only well-defined at a
    hyperbolic (non-marginal) operating point; near a bifurcation ``dF/dy`` becomes
    near-singular and the sensitivity blows up (finite but huge), giving the DGSM a
    heavy tail that the Monte-Carlo mean cannot resolve. This drops any sample whose
    Jacobian condition number exceeds ``cond_factor`` times the **median** over the
    sample -- i.e. that is far more ill-conditioned than the typical operating point
    -- alongside the non-finite ones. ``cond_factor=None`` keeps every sample.
    Returns a boolean ``(N,)`` mask.
    """
    import numpy as np

    cond = np.asarray(cond)
    finite = np.isfinite(cond)
    if cond_factor is None:
        return finite
    med = np.median(cond[finite]) if finite.any() else np.inf
    return finite & (cond <= cond_factor * med)


def _valid_sample_mask(cond, cond_factor, operating_point_exists):
    """Keep-mask selecting the samples that enter the steady-state DGSM aggregation.

    The single home of the operating-regime exclusion policy, otherwise re-derived
    at each call site (``convergence``, ``with_cond_factor``, ``steady_state_dgsm``).
    Combines the two exclusion criteria:

    - the near-singular-Jacobian drop (:func:`_cond_mask`, gated by ``cond_factor``),
      and
    - the **past-fold** operating-regime exclusion: a sample whose
      ``operating_point_exists`` is ``False`` folds before its parameters, so no
      operating-branch steady state exists and it is outside the viable regime.

    ``operating_point_exists`` is the raw per-sample list (``True`` / ``False`` /
    ``None`` per sample) or ``None`` when the result predates that record, in which
    case only the conditioning filter applies. Returns a boolean ``(N,)`` keep-mask
    aligned with ``cond``.
    """
    import numpy as np

    cond = np.asarray(cond)
    if operating_point_exists is None:
        operating_mask = np.ones(len(cond), dtype=bool)
    else:
        operating_mask = np.array([e is not False for e in operating_point_exists], dtype=bool)
    return _cond_mask(cond, cond_factor) & operating_mask


def _to_z(theta, kind):
    """Physical parameter -> calibration-transform space (where a prior is normal).

    ``positive_log`` -> natural log, ``logit`` -> log-odds, ``none`` -> identity.
    Host-side (NumPy) view of :func:`aquakin.integrate._transforms.to_unconstrained`.
    """
    import numpy as np

    return to_unconstrained(theta, kind, xp=np)


def _from_z(z, kind):
    """Inverse of :func:`_to_z`: transform-space variable -> physical parameter."""
    import numpy as np

    return from_unconstrained(z, kind, xp=np)


def _dtheta_dz(theta, kind):
    """``d(physical)/d(transform)`` at ``theta`` -- the DGSM chain-rule factor that
    converts a physical sensitivity ``dg/dtheta`` to the transform-space
    ``dg/dz`` in which the input is Gaussian."""
    import numpy as np

    return dphysical_dunconstrained(theta, kind, xp=np)


@dataclass
class SteadyStateDGSMResult:
    """Result of :meth:`Plant.steady_state_dgsm` -- a derivative-based global
    sensitivity (DGSM) screen of a plant steady state.

    For each screened parameter ``z_j`` (uniform on ``[a_j, b_j]``) and each output
    ``g_i``, the **Sobol total-index upper bound**

        S_ij^tot <= nu_ij (b_j - a_j)^2 / (pi^2 Var(g_i)),

    with ``nu_ij = E[(dg_i/dz_j)^2]`` the mean squared steady-state sensitivity over
    the Sobol sample (Lamboni, Sobol & Kucherenko 2013) -- the AD-accelerated
    analogue of a variance-based Sobol total index, here read from the
    implicit-function-theorem sensitivity (one ``dF/dy`` factorisation per sample).

    Attributes
    ----------
    input_names : list[str]
        The ``k`` screened parameters (columns of every array).
    output_names : list[str]
        The ``m`` outputs (rows of every array).
    sobol_total_bound : jnp.ndarray
        ``(m, k)`` Sobol total-index upper bound per output and parameter.
    std_error : jnp.ndarray
        ``(m, k)`` Monte-Carlo standard error of the bound (shrinks like
        ``1/sqrt(n_valid)``; the convergence indicator).
    nu : jnp.ndarray
        ``(m, k)`` mean squared sensitivity ``E[(dg_i/dz_j)^2]``.
    n_valid : jnp.ndarray
        ``(m,)`` finite samples per output -- the extreme Sobol corners where the
        perturbed steady state is non-finite are dropped per output (as
        :func:`~aquakin.dgsm` does); a near-constant output (zero variance) yields
        a ``NaN`` bound.
    output_variance : jnp.ndarray
        ``(m,)`` variance of each output over the sample.
    ranges : jnp.ndarray
        ``(k, 2)`` ``[a_j, b_j]`` screened ranges.
    n_samples : int
        Number of quasi-random points drawn (a power of two).
    seed : int
        Scrambled-Sobol seed (fixing it makes the result reproducible).
    grad_sq : jnp.ndarray
        ``(n_samples, m, k)`` per-sample squared sensitivities, retained so
        :meth:`convergence` can recompute the bound from any prefix of the sample.
    outputs : jnp.ndarray
        ``(n_samples, m)`` per-sample output values (for the running variance).
    cond : jnp.ndarray
        ``(n_samples,)`` condition number of the steady-state Jacobian at each
        sample (the near-singularity diagnostic the ``cond_factor`` filter uses).
    cond_factor : float or None
        The near-singular-Jacobian drop threshold applied (``None`` = no filter);
        see :meth:`Plant.steady_state_dgsm`. :meth:`with_cond_factor` re-applies a
        different value without re-solving.
    """

    input_names: list
    output_names: list
    sobol_total_bound: jnp.ndarray
    std_error: jnp.ndarray
    nu: jnp.ndarray
    output_variance: jnp.ndarray
    ranges: jnp.ndarray
    n_samples: int
    seed: int
    grad_sq: jnp.ndarray
    outputs: jnp.ndarray
    n_valid: jnp.ndarray
    cond: jnp.ndarray
    cond_factor: Optional[float] = None
    # Per-sample final scaled steady-state residual ``max_i |F_i|/max(|y_i|,floor)``
    # (shape ``(N,)``). A sample whose solve did not converge carries a large
    # residual; it can be excluded as an invalid operating point, more directly
    # than by a conditioning or output-magnitude test. ``None`` on results built
    # before this was recorded. A sample solved by the forward backstop carries
    # ``NaN`` here (the forward solve reports no PTC scaled residual).
    residual: Optional[jnp.ndarray] = None
    # Per-sample solver method from the layered solve -- ``"ptc"`` (direct),
    # ``"continuation"`` (deformed from the nominal), or ``"ptc->forward"`` (forward
    # backstop, the near-bifurcation samples a fast algebraic solve cannot tighten).
    # A Python list of length ``N``; the coverage tally is ``Counter(solve_method)``.
    # ``None`` on results built before this was recorded.
    solve_method: Optional[list] = None
    # Per-sample convergence flag (shape ``(N,)`` bool) from the layered solve.
    converged: Optional[jnp.ndarray] = None
    # Per-sample operating-point existence (length ``N`` list of ``True`` /
    # ``False`` / ``None``): ``False`` marks a ``past_fold`` sample (the operating
    # branch folds before its parameters, so no operating-branch steady state
    # exists -- past a saddle-node bifurcation). These are **excluded** from the
    # screen (``sobol_total_bound``) as outside the viable operating regime -- the
    # physical, fold-based exclusion criterion. ``None`` on results built before
    # this was recorded.
    operating_point_exists: Optional[list] = None
    # Per-parameter Poincare constant of the input measure, shape ``(k,)``. When
    # set (Gaussian/prior input sampling, ``input_dist="normal"``) the bound is
    # ``nu_ij * poincare_j / Var(g_i)`` with ``poincare_j = sigma_zj^2`` the
    # transform-space prior variance; ``None`` is the uniform default, where the
    # bound uses ``(b_j-a_j)^2/pi^2`` from ``ranges``. Consumed by ``convergence``
    # and ``with_cond_factor`` so they re-aggregate on the correct constant.
    poincare: Optional[jnp.ndarray] = None

    def ranked(self, output=0):
        """``[(input_name, bound)]`` for one output, sorted by decreasing bound.

        ``output`` is an output name or row index.
        """
        i = self.output_names.index(output) if isinstance(output, str) else int(output)
        pairs = [(n, float(b)) for n, b in zip(self.input_names, self.sobol_total_bound[i])]
        return sorted(pairs, key=lambda t: t[1], reverse=True)

    def convergence(self, counts=None):
        """Running Sobol bound and Monte-Carlo standard error versus sample count.

        Recomputes the bound from the first ``k`` samples for each ``k`` in
        ``counts`` (default: powers of two up to ``n_samples``), so plotting the
        bound -- or the standard error -- against ``k`` shows the Monte-Carlo
        convergence. This is the sample-size study, run from the retained
        per-sample data (no re-solving).

        Returns
        -------
        counts : jnp.ndarray
            ``(n_counts,)`` the sample counts used.
        bound : jnp.ndarray
            ``(n_counts, m, k)`` the Sobol bound from each prefix.
        std_error : jnp.ndarray
            ``(n_counts, m, k)`` the Monte-Carlo standard error of each.
        """
        import math

        import numpy as np

        n = int(self.outputs.shape[0])
        if counts is None:
            j0 = 4
            counts = [2**j for j in range(j0, max(j0, int(math.log2(n))) + 1)]
            if not counts or counts[-1] != n:
                counts.append(n)
        rng2 = np.asarray((self.ranges[:, 1] - self.ranges[:, 0]) ** 2)  # (k,)
        pc = None if self.poincare is None else np.asarray(self.poincare)  # (k,)
        gs = np.asarray(self.grad_sq)  # (N, m, k)
        ov = np.asarray(self.outputs)  # (N, m)
        cond = np.asarray(self.cond)  # (N,)
        oe = self.operating_point_exists
        b_list, se_list = [], []
        for k in counts:
            oe_k = None if oe is None else oe[:k]
            mask = _valid_sample_mask(cond[:k], self.cond_factor, oe_k)
            bound, se, *_ = _dgsm_aggregate(gs[:k], ov[:k], rng2, sample_mask=mask, poincare=pc)
            b_list.append(bound)
            se_list.append(se)
        return (jnp.asarray(counts), jnp.asarray(np.stack(b_list)), jnp.asarray(np.stack(se_list)))

    def with_cond_factor(self, cond_factor):
        """Re-aggregate with a different near-singular-Jacobian threshold.

        Returns a new :class:`SteadyStateDGSMResult` whose bounds, standard errors
        and ``n_valid`` are recomputed from the retained per-sample data with the
        given ``cond_factor`` -- so the filter can be tuned (and the convergence
        re-read) without re-solving. See :meth:`Plant.steady_state_dgsm`.
        """
        from dataclasses import replace

        import numpy as np

        rng2 = np.asarray((self.ranges[:, 1] - self.ranges[:, 0]) ** 2)
        mask = _valid_sample_mask(self.cond, cond_factor, self.operating_point_exists)
        pc = None if self.poincare is None else np.asarray(self.poincare)
        bound, se, nu, var, n_valid = _dgsm_aggregate(
            np.asarray(self.grad_sq), np.asarray(self.outputs), rng2, sample_mask=mask, poincare=pc
        )
        return replace(
            self,
            sobol_total_bound=jnp.asarray(bound),
            std_error=jnp.asarray(se),
            nu=jnp.asarray(nu),
            output_variance=jnp.asarray(var),
            n_valid=jnp.asarray(n_valid),
            cond_factor=cond_factor,
        )


def _dynamic_value_and_jacobian(f, x, mode):
    """``(value, jacobian)`` of ``f`` at ``x`` in one primal pass.

    Reverse mode forms the value and the ``(m, k)`` Jacobian from a single
    ``jax.vjp`` (one primal solve, ``m`` transposed passes); forward mode from a
    single ``jax.linearize`` (one primal solve, ``k`` tangent pushes). ``f`` must
    already carry the adjoint matching ``mode`` (reverse: the cap-free stable
    adjoint; forward: a forward-capable adjoint).
    """
    if mode == "reverse":
        g, vjp_fn = jax.vjp(f, x)
        S = jax.vmap(lambda c: vjp_fn(c)[0])(jnp.eye(int(g.shape[0])))  # (m, k)
        return g, S
    g, jvp_fn = jax.linearize(f, x)
    S = jax.vmap(jvp_fn)(jnp.eye(int(x.shape[0]))).T  # (m, k)
    return g, S


def _dynamic_dgsm_bounds(grad_sq, outputs, rng2, poincare=None):
    """Sobol total-index bound from per-sample squared sensitivities of a transient
    output, dropping (per output) any non-finite sample. The dynamic analogue of
    the steady-state aggregation; there is no Jacobian-conditioning filter because
    a transient sensitivity is differentiated through the solve, not via a
    steady-state Jacobian.
    """
    import numpy as np

    grad_sq = np.asarray(grad_sq)
    outputs = np.asarray(outputs)
    _, m, k = grad_sq.shape
    valid = np.isfinite(outputs) & np.isfinite(grad_sq).all(axis=2)
    nu = np.full((m, k), np.nan)
    se = np.full((m, k), np.nan)
    var = np.full(m, np.nan)
    n_valid = valid.sum(axis=0).astype(int)
    for i in range(m):
        v = valid[:, i]
        if int(v.sum()) < 2:
            continue
        g = grad_sq[v, i, :]
        nu[i] = g.mean(axis=0)
        se[i] = g.std(axis=0) / np.sqrt(int(v.sum()))
        var[i] = outputs[v, i].var()
    # Poincare constant of the input measure: uniform default ``(b_j-a_j)^2/pi^2``;
    # for a Gaussian-distributed input the bound instead carries ``poincare_j =
    # std_j^2`` (Sobol & Kucherenko 2010, Sec. 8; Lamboni et al. 2013, Thm 3.1),
    # passed in directly. ``bound = nu * C_j / Var(g)`` either way.
    const = np.asarray(poincare)[None, :] if poincare is not None else rng2[None, :] / (np.pi**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where((var > 0)[:, None], const / var[:, None], np.nan)
    return nu * scale, se * scale, nu, var, n_valid


@dataclass
class DynamicDGSMResult:
    """Result of :meth:`Plant.dynamic_dgsm` -- a derivative-based global sensitivity
    (DGSM) screen of a transient (time-window) plant output. Same Sobol
    total-index bound as the steady-state screen, but each sample's sensitivity is
    differentiated through the dynamic solve. Fields mirror
    :class:`SteadyStateDGSMResult` (without the steady-state Jacobian conditioning).
    """

    input_names: list
    output_names: list
    sobol_total_bound: jnp.ndarray
    std_error: jnp.ndarray
    nu: jnp.ndarray
    output_variance: jnp.ndarray
    ranges: jnp.ndarray
    n_samples: int
    seed: int
    grad_sq: jnp.ndarray
    outputs: jnp.ndarray
    n_valid: jnp.ndarray

    def ranked(self, output=0):
        """``[(input_name, bound)]`` for one output, sorted by decreasing bound."""
        i = self.output_names.index(output) if isinstance(output, str) else int(output)
        pairs = [(n, float(b)) for n, b in zip(self.input_names, self.sobol_total_bound[i])]
        return sorted(pairs, key=lambda t: t[1], reverse=True)

    def convergence(self, counts=None):
        """Running Sobol bound and Monte-Carlo standard error versus sample count,
        recomputed from the retained per-sample data (no re-solving).
        """
        import math

        import numpy as np

        n = int(self.outputs.shape[0])
        if counts is None:
            j0 = 4
            counts = [2**j for j in range(j0, max(j0, int(math.log2(n))) + 1)]
            if not counts or counts[-1] != n:
                counts.append(n)
        rng2 = np.asarray((self.ranges[:, 1] - self.ranges[:, 0]) ** 2)
        gs = np.asarray(self.grad_sq)
        ov = np.asarray(self.outputs)
        b_list, se_list = [], []
        for k in counts:
            bound, se, *_ = _dynamic_dgsm_bounds(gs[:k], ov[:k], rng2)
            b_list.append(bound)
            se_list.append(se)
        return (jnp.asarray(counts), jnp.asarray(np.stack(b_list)), jnp.asarray(np.stack(se_list)))


def steady_state_sensitivity(
    plant,
    params: Optional[jnp.ndarray] = None,
    y0: Optional[jnp.ndarray] = None,
    *,
    state: Optional[jnp.ndarray] = None,
    output_fn: Optional[Callable] = None,
    wrt: Optional[Sequence] = None,
    operating: Optional[Sequence] = None,
    mode: str = "auto",
    elasticity: bool = False,
    return_jacobian: bool = False,
    **steady_kwargs,
) -> jnp.ndarray:
    """Exact steady-state output sensitivities via the implicit function theorem.

    Solves the plant to steady state once, then returns ``d(output)/d(params)``
    directly from the right-hand-side Jacobians -- no time integration and no
    differentiation through the solve. A single steady-state solve and a single
    ``dF/dy`` factorisation are reused for every output and parameter, so this
    is far cheaper than ``jacfwd``/``jacrev`` through :meth:`steady_state`
    (which re-solves per call).

    The two AD directions are the two ways to read the same sensitivity:
    forward mode (one solve per parameter, all outputs follow) is efficient
    when the outputs outnumber the parameters; reverse mode (one transposed
    solve plus a vector--Jacobian product per output, all parameters follow) is
    efficient when the parameters outnumber the outputs.

    Parameters
    ----------
    params : jnp.ndarray, optional
        Plant parameters (defaults to :meth:`default_parameters`).
    y0 : jnp.ndarray, optional
        Warm start for the steady-state solve.
    state : jnp.ndarray, optional
        A pre-solved steady state to evaluate the sensitivity at, skipping the
        internal solve. Use this when the operating point is already known (e.g.
        a warm-start steady state) and to read the sensitivity in several ways
        without re-solving each time. Must satisfy ``F(state, params) = 0``.
    output_fn : callable, optional
        Maps the flat plant state ``(total_state_size,)`` to a length-``m``
        vector of scalar outputs. Defaults to the identity (the full state, so
        the result is the ``(n_states, n_params)`` sensitivity ``dy*/dtheta``).
    wrt : sequence of int or str, optional
        The parameters to differentiate with respect to -- flat indices or
        ``"<model>.<param>"`` names (resolved by :meth:`parameter_index`).
        Defaults to all parameters. Restricting to a subset of ``k`` parameters
        makes **forward** mode cost ``k`` solves (one per chosen parameter)
        rather than ``n_params``; reverse mode returns all parameters from the
        per-output solve regardless and simply selects this subset.
    mode : {"auto", "forward", "reverse"}
        AD direction. ``"auto"`` picks ``"forward"`` when the number of selected
        parameters ``<= m`` and ``"reverse"`` otherwise. Both give the same exact
        sensitivity.
    elasticity : bool
        If ``True`` return the dimensionless elasticity
        ``(dg/dtheta)(theta/g)`` instead of the raw derivative.
    return_jacobian : bool
        If ``True`` also return the steady-state Jacobian ``dF/dy`` (so a
        caller can assess its conditioning without recomputing it, as
        :meth:`steady_state_dgsm` does to flag near-singular operating points).
    **steady_kwargs
        Forwarded to :meth:`steady_state` (e.g. ``max_iter``).

    Returns
    -------
    jnp.ndarray
        ``(m, k)`` sensitivity of the ``m`` outputs to the ``k`` selected
        parameters (``k == n_params`` when ``wrt`` is ``None``).

    Notes
    -----
    Exact when the steady Jacobian ``dF/dy`` is full rank (true for the shipped
    models at their operating point; see :func:`solve_steady_state`).
    """
    params = plant.default_parameters() if params is None else jnp.asarray(params)
    if state is not None:
        y_star = jax.lax.stop_gradient(jnp.asarray(state))
    else:
        y_star = jax.lax.stop_gradient(plant.steady_state(params, y0=y0, **steady_kwargs).state)

    def F(y, p):
        return plant.derivative(y, params=p)

    out_fn = output_fn if output_fn is not None else (lambda y: y)
    g0 = jnp.atleast_1d(out_fn(y_star))
    G = jax.jacfwd(out_fn)(y_star)  # (m, n) or (n,)
    if G.ndim == 1:
        G = G[None, :]  # scalar output -> (1, n)
    m, n_par = G.shape[0], int(params.shape[0])

    if wrt is None:
        wrt_idx = jnp.arange(n_par)
    else:
        wrt_idx = jnp.asarray(
            [plant.parameter_index(w) if isinstance(w, str) else int(w) for w in wrt]
        )
    n_wrt = int(wrt_idx.shape[0])

    # Operating-condition inputs (the influent scales, nominal 1.0): their IFT
    # columns are appended after the kinetic-parameter columns below, through
    # the same ``design`` override the steady-state solve uses. The
    # no-operating path is byte-for-byte unchanged.
    op_meta = plant._parse_operating(operating)
    n_op = len(op_meta)

    J_y = jax.jacfwd(lambda y: F(y, params))(y_star)

    chosen = ("forward" if (n_wrt + n_op) <= m else "reverse") if mode == "auto" else mode
    if chosen == "forward":
        # Differentiate only the selected parameters: k forward passes, not
        # n_params. (k == n_params when wrt is None.)
        def F_sub(theta_sub):
            return F(y_star, params.at[wrt_idx].set(theta_sub))

        J_theta = jax.jacfwd(F_sub)(params[wrt_idx])  # (n, n_wrt)
        S = jnp.linalg.solve(J_y, -J_theta)  # (n, n_wrt)
        dgdth = G @ S  # (m, n_wrt)
    elif chosen == "reverse":
        # Each output's adjoint returns the sensitivity to every parameter;
        # select the requested subset afterwards.
        J_yT = J_y.T
        _, vjp_F = jax.vjp(lambda p: F(y_star, p), params)

        def _row(g_i):
            lam = jnp.linalg.solve(J_yT, -g_i)  # adjoint
            return vjp_F(lam)[0][wrt_idx]  # (n_wrt,)

        dgdth = jax.vmap(_row)(G)  # (m, n_wrt)
    else:
        raise ValueError(f"mode must be 'auto', 'forward', or 'reverse'; got {mode!r}.")

    # Append the operating-condition columns: the same IFT, with
    # ``dF/d(scale)`` taken through the influent ``design`` override, in the
    # chosen AD direction. The constant influent is sampled at ``t=0`` (the
    # derivative's evaluation time).
    if n_op:
        plant._build_state_layout()
        plant._build_parameter_layout()
        t0 = jnp.asarray(0.0)
        op0 = jnp.ones(n_op)
        pf = plant._coerce_params(params)

        def F_op(op_vals):
            return plant._rhs(t0, y_star, pf, design=plant._operating_design(op_meta, op_vals))

        if chosen == "forward":
            J_op = jax.jacfwd(F_op)(op0)  # (n, n_op)
            dg_op = G @ jnp.linalg.solve(J_y, -J_op)  # (m, n_op)
        else:
            _, vjp_op = jax.vjp(F_op, op0)
            dg_op = jax.vmap(lambda g_i: vjp_op(jnp.linalg.solve(J_y.T, -g_i))[0])(G)
        dgdth = jnp.concatenate([dgdth, dg_op], axis=1)  # (m, n_wrt+n_op)

    if elasticity:
        theta_all = jnp.concatenate([params[wrt_idx], jnp.ones(n_op)]) if n_op else params[wrt_idx]
        dgdth = dgdth * (theta_all[None, :] / g0[:, None])
    if return_jacobian:
        return dgdth, J_y
    return dgdth


def steady_state_dgsm(
    plant,
    ranges,
    *,
    output_fn: Callable,
    output_names: Optional[Sequence] = None,
    wrt: Optional[Sequence] = None,
    y0: Optional[jnp.ndarray] = None,
    n_samples: int = 256,
    seed: int = 0,
    mode: str = "auto",
    cond_factor: Optional[float] = None,
    input_dist: str = "uniform",
    input_transforms: Optional[Sequence[str]] = None,
    continuation: bool = True,
    progress: Optional[int] = None,
    **steady_kwargs,
) -> "SteadyStateDGSMResult":
    """Derivative-based global sensitivity (DGSM) of the plant steady state.

    Samples the screened parameters over their ranges (scrambled-Sobol QMC),
    solves the steady state at each sample, and reads each output's sensitivity
    to those parameters through the implicit-function-theorem helper
    (:meth:`steady_state_sensitivity`) -- reusing **one** ``dF/dy``
    factorisation per sample. That makes it markedly cheaper than a generic
    :func:`~aquakin.dgsm` over ``steady_state`` (whose ``jacfwd``/``jacrev``
    recompute the steady-state structure for every input tangent / output).
    Aggregates to the Sobol total-index upper bound per (output, parameter); the
    retained per-sample data lets :meth:`SteadyStateDGSMResult.convergence`
    report the sample-size study without re-solving.

    Parameters
    ----------
    ranges : array-like, shape ``(k, 2)``
        ``[a_j, b_j]`` sampling range for each screened parameter, aligned with
        ``wrt``. For ``input_dist="uniform"`` (default) inputs are sampled
        uniformly within. For ``input_dist="normal"`` the range is read as the
        ``+/-2 sigma`` band of the input's prior *in its calibration-transform
        space*, i.e. ``[a_j, b_j] = [t^{-1}(m_j - 2 sigma_j), t^{-1}(m_j +
        2 sigma_j)]``, so ``m_j = t(nominal_j)`` and ``sigma_j = (t(b_j) -
        t(a_j))/4`` with ``t`` the parameter's transform.
    output_fn : callable
        Maps the flat plant state to a length-``m`` vector of scalar outputs.
    output_names : sequence of str, optional
        Names for the ``m`` outputs (default ``"output0"`` ...).
    wrt : sequence of int or str, optional
        The screened parameters -- flat indices or ``"<model>.<param>"``
        names (default: all parameters; usually pass an explicit subset).
    y0 : jnp.ndarray, optional
        Warm start for each steady-state solve.
    n_samples : int, optional
        Sobol points (rounded to a power of two). Increase until
        :meth:`SteadyStateDGSMResult.convergence` flattens.
    seed : int, optional
        Scrambled-Sobol seed (fixing it makes the screen reproducible).
    mode : {"auto", "forward", "reverse"}, optional
        AD direction for the per-sample sensitivity (passed to
        :meth:`steady_state_sensitivity`). ``"auto"`` is reverse for the usual
        many-parameter / few-output screen.
    cond_factor : float, optional
        Drop any sample whose steady-state Jacobian ``dF/dy`` is more than
        ``cond_factor`` times as ill-conditioned as the sample median -- a
        near-singular (near-bifurcation) operating point, where the sensitivity
        is not well-defined and would give the DGSM a heavy tail. ``None``
        (default) keeps every sample (then the bounds match :func:`~aquakin.dgsm`
        exactly). A value around ``1e2`` removes the marginal-stability outliers
        of a stiff plant; the dropped count is reported in ``n_valid``.
    input_dist : {"uniform", "normal"}, optional
        Input sampling distribution. ``"uniform"`` (default) draws uniformly
        over ``ranges`` and bounds the Sobol total index with the uniform
        Poincare constant ``(b_j-a_j)^2/pi^2``. ``"normal"`` draws each input
        from its Gaussian prior in the calibration-transform space (a
        log-normal for a positive rate, logit-normal for a fraction), reads the
        sensitivity in that space (chain-ruling ``dg/dz = (dg/dtheta)
        (dtheta/dz)``), and uses the Gaussian Poincare constant ``sigma_j^2`` --
        the established generalization of the DGSM bound to non-uniform inputs
        (Sobol & Kucherenko 2010, Sec. 8; Lamboni et al. 2013, Thm 3.1). It
        samples the prior we believe rather than a box, so an improbable
        corner is weighted by its prior density. Requires ``input_transforms``.
    input_transforms : sequence of str, optional
        Required for ``input_dist="normal"``: the calibration transform of each
        screened parameter (``"positive_log"`` / ``"logit"`` / ``"none"``),
        aligned with ``wrt`` -- the space in which its prior is Gaussian.
        Ignored for ``"uniform"``.
    progress : int, optional
        If set, print a progress line every ``progress`` samples.
    **steady_kwargs
        Forwarded to :meth:`steady_state` (e.g. ``max_iter``).

    Returns
    -------
    SteadyStateDGSMResult
        The ``(m, k)`` Sobol total-index bounds, standard errors, and the
        retained per-sample data.
    """
    import numpy as np

    from ..integrate._qmc import _sobol_sample

    base = plant.default_parameters()
    if wrt is None:
        wrt_idx = list(range(int(base.shape[0])))
    else:
        wrt_idx = [plant.parameter_index(w) if isinstance(w, str) else int(w) for w in wrt]
    k = len(wrt_idx)
    ranges_arr = jnp.asarray(ranges, dtype=base.dtype)
    if ranges_arr.shape != (k, 2):
        raise ValueError(
            f"ranges must have shape ({k}, 2) aligned with the {k} screened "
            f"parameters; got {tuple(ranges_arr.shape)}."
        )
    lo, hi = ranges_arr[:, 0], ranges_arr[:, 1]
    wrt_j = jnp.asarray(wrt_idx)

    # The steady-state solve uses the LAYERED fallback (direct PTC ->
    # parameter continuation from the nominal -> forward integration), which is
    # eager Python control flow and so cannot run inside a jit (under a trace
    # the fallback branches are skipped and only the un-converged direct PTC
    # result survives). The nominal steady state is solved once as the
    # continuation known point -- and as a better per-sample warm start than the
    # cold default. Only the implicit-function-theorem sensitivity at the found
    # state is jitted (compiled once, reused across samples).
    known = None
    if continuation:
        nom = plant.steady_state(base, y0=y0, **steady_kwargs)
        known = (base, nom.state)
        if y0 is None:
            y0 = nom.state

    @jax.jit
    def _sens_at(p, ss):
        S, J_y = plant.steady_state_sensitivity(
            p, state=ss, output_fn=output_fn, wrt=wrt_idx, mode=mode, return_jacobian=True
        )  # (m, k), (n, n)
        return jnp.atleast_1d(output_fn(ss)), S, jnp.linalg.cond(J_y)

    if input_dist == "normal":
        from ..integrate._qmc import _sobol_normal_sample

        if input_transforms is None or len(input_transforms) != k:
            raise ValueError(
                "input_dist='normal' requires input_transforms aligned with "
                f"the {k} screened parameters (got "
                f"{None if input_transforms is None else len(input_transforms)})."
            )
        nominal = np.asarray(base[wrt_j])
        lo_np, hi_np = np.asarray(lo), np.asarray(hi)
        # The +/-2 sigma band [lo, hi] is read in the calibration-transform
        # space (where the prior is Gaussian): mean = transform(nominal), and
        # the band spans +/-2 sigma there, so sigma_z = (t(hi) - t(lo)) / 4.
        m_z = np.array([_to_z(nominal[j], input_transforms[j]) for j in range(k)])
        s_z = np.array(
            [
                (_to_z(hi_np[j], input_transforms[j]) - _to_z(lo_np[j], input_transforms[j])) / 4.0
                for j in range(k)
            ]
        )
        Zz, n_drawn = _sobol_normal_sample(m_z, s_z, k, n_samples, seed)
        Z = np.column_stack(
            [_from_z(Zz[:, j], input_transforms[j]) for j in range(k)]
        )  # physical samples
        poincare_const = s_z**2  # Gaussian Poincare C_j
    elif input_dist == "uniform":
        Z, n_drawn = _sobol_sample(lo, hi, k, n_samples, seed)
        poincare_const = None
    else:
        raise ValueError(f"input_dist must be 'uniform' or 'normal', got {input_dist!r}.")
    outs, grads, conds, resids, smethods, sconv, sexist = ([], [], [], [], [], [], [])
    for i in range(n_drawn):
        p = base.at[wrt_j].set(jnp.asarray(Z[i]))
        ssr = plant.steady_state(p, y0=y0, continuation_from=known, **steady_kwargs)
        o, S, c = _sens_at(p, ssr.state)
        S = np.asarray(S)
        if input_dist == "normal":
            # Chain rule dg/dz = (dg/dtheta) * (dtheta/dz): express the
            # sensitivity in the transform space where the input is Gaussian,
            # so the DGSM and its Poincare constant are consistent.
            mult = np.array([_dtheta_dz(float(Z[i, j]), input_transforms[j]) for j in range(k)])
            S = S * mult[None, :]
        outs.append(np.asarray(o))
        grads.append(S)
        conds.append(float(c))
        resids.append(float(ssr.residual) if ssr.residual is not None else float("nan"))
        smethods.append(ssr.method)
        sconv.append(bool(ssr.converged))
        sexist.append(ssr.operating_point_exists)  # True / False / None
        if progress and (i + 1) % progress == 0:
            print(
                f"  [steady_state_dgsm] {i + 1}/{n_drawn} samples (last: {ssr.method})",
                flush=True,
            )

    outputs = np.stack(outs)  # (N, m)
    grad_sq = np.stack(grads) ** 2  # (N, m, k)
    cond = np.asarray(conds)  # (N,)
    residual = np.asarray(resids)  # (N,) final scaled residual
    rng2 = np.asarray((hi - lo) ** 2)  # (k,)
    # Keep-mask: drop, per output, any non-finite sample, plus (if cond_factor is
    # set) any near-singular-Jacobian operating point where the sensitivity blows
    # up, plus the past-fold (non-operating) samples. See _valid_sample_mask.
    sample_mask = _valid_sample_mask(cond, cond_factor, sexist)
    bound, std_error, nu, var, n_valid = _dgsm_aggregate(
        grad_sq, outputs, rng2, sample_mask=sample_mask, poincare=poincare_const
    )

    names = plant.parameter_names()
    input_names = [names[i] for i in wrt_idx]
    m = int(outputs.shape[1])
    if output_names is None:
        output_names = [f"output{i}" for i in range(m)]
    return SteadyStateDGSMResult(
        input_names=input_names,
        output_names=list(output_names),
        sobol_total_bound=jnp.asarray(bound),
        std_error=jnp.asarray(std_error),
        nu=jnp.asarray(nu),
        output_variance=jnp.asarray(var),
        ranges=ranges_arr,
        n_samples=int(n_drawn),
        seed=int(seed),
        grad_sq=jnp.asarray(grad_sq),
        outputs=jnp.asarray(outputs),
        n_valid=jnp.asarray(n_valid),
        cond=jnp.asarray(cond),
        cond_factor=cond_factor,
        residual=jnp.asarray(residual),
        solve_method=smethods,
        converged=jnp.asarray(sconv),
        operating_point_exists=sexist,
        poincare=(None if poincare_const is None else jnp.asarray(poincare_const)),
    )


def solve_sensitivity(
    plant,
    params,
    wrt,
    *,
    operating=None,
    t_span,
    t_eval=None,
    y0=None,
    rtol: float = 1e-6,
    atol=None,
    sens_rtol=None,
    factormax=None,
    dtmax=None,
    max_steps: int = 1_000_000,
):
    """Stable forward (variational) sensitivity ``dy/dtheta`` of the plant.

    Integrates the augmented system ``z = [y; S]`` (the state plus its
    sensitivity ``S = dy/dtheta``) with the step controller's error norm
    bounding ``S`` directly, so the sensitivity stays finite over long
    horizons where ``jacfwd`` through the stiff plant solve goes non-finite.
    This is the plant counterpart of the reactors' ``solve_sensitivity`` --
    the forward-mode sensitivity tool the dynamic plant lacked.

    The augmented ``[y; S]`` solve uses the SAME enhanced solver config the
    plant's forward / discrete-adjoint solves use -- a decoupled Newton root
    finder, the ``factormax`` step-growth cap, and the cached recycle / flow
    maps -- with a lean ``Kvaerno3`` base solver and the block-arrow
    :class:`~aquakin.integrate._simultaneous_corrector.SimultaneousCorrector`
    for the per-stage linear algebra (factor the shared diagonal block
    ``D = I - gamma.dt.J`` once, reuse it across every sensitivity column).

    Parameters
    ----------
    params : jnp.ndarray
        Plant parameter vector.
    wrt : sequence of str or int
        The sensitivity parameters, as ``"<model>.<param>"`` names or flat
        indices. The cached recycle map drops the ``dM/dtheta`` term, so this
        is exact for kinetic parameters; a flow-setpoint sensitivity would
        need the per-call map (not wired here).
    operating : sequence of dict, optional
        Operating-parameter sensitivities computed alongside ``wrt``, each a
        differentiable multiplicative scale on an influent (nominal 1.0, so
        the sensitivity is ``d output / d(scale)`` at the recorded influent).
        Each spec is ``{"kind": "influent_flow", "port": p}`` (scale that
        influent's flow) or ``{"kind": "influent_concentration", "port": p,
        "species": s}`` (scale that species' load). They append columns to
        ``S`` after the ``wrt`` columns, in order. Exact through the
        variational solve -- the cached recycle/flow maps are
        influent-independent, so no term is dropped (unlike a flow setpoint).
    t_span : tuple of float
        ``(t0, t1)`` integration interval (plant time units).
    t_eval : jnp.ndarray, optional
        Times to record. ``None`` records the endpoint only.
    y0 : jnp.ndarray, optional
        Initial state (defaults to :meth:`initial_state`); warm-start a stiff
        plant.
    rtol, atol : float or array, optional
        State step tolerances (``atol`` defaults to the per-component plant
        floor).
    sens_rtol : float, optional
        Relative tolerance on ``S`` (defaults to ``rtol``).
    factormax, dtmax : float, optional
        Step-growth cap and maximum step, passed to the augmented solve.
    max_steps : int
        Step budget for the augmented solve.

    Returns
    -------
    ts : jnp.ndarray
        Save times, shape ``(n_t,)``.
    ys : jnp.ndarray
        State trajectory, shape ``(n_t, ndof)`` (split with
        :meth:`states_by_unit`).
    S : jnp.ndarray
        Sensitivity ``dy/dtheta``, shape ``(n_t, ndof, k)`` for the ``k``
        ``wrt`` parameters.

    Notes
    -----
    The block-arrow ``SimultaneousCorrector`` exploits the augmented system's
    structure (each sensitivity column couples only to ``y``); it is specific
    to this ``[y; S]`` solve and does not apply to the single-state forward /
    adjoint solves, whose per-step lever is the colored Jacobian.
    """
    from aquakin.integrate._common import default_atol
    from aquakin.integrate.forward_sensitivity import (
        augmented_forward_sensitivity,
    )

    plant._build_state_layout()
    plant._build_parameter_layout()
    params_full = plant._coerce_params(jnp.asarray(params))
    y0 = plant.initial_state() if y0 is None else jnp.asarray(y0)
    free_idx = jnp.asarray(
        [plant.parameter_index(w) if isinstance(w, str) else int(w) for w in wrt], dtype=int
    )  # int even when wrt is empty (operating-only)
    t0, t1 = float(t_span[0]), float(t_span[1])
    t0a = jnp.asarray(t0)

    # Cached recycle / flow maps (state-invariant for fixed-pump plants): run
    # the same one-time concrete check plant.solve does, then reuse the maps
    # per RHS so the augmented solve's f_flat does not re-resolve the recycle
    # every call. The maps are built from the concrete ``params_full`` and
    # closed over, so the augmented linearisation w.r.t. the wrt parameters
    # drops ``dM/dtheta`` -- exact for kinetic parameters (M depends only on
    # the flow setpoints).
    if plant._recycle._recycle_map_constant is None:
        plant._recycle._check_recycle_map_constant(t0a, y0, params_full)
    if plant._recycle._flow_map_constant is None:
        plant._recycle._check_flow_map_constant(t0a, y0, params_full)
    states0 = plant._split_state(y0)
    rmap = plant._recycle._maybe_recycle_map(t0a, states0, params_full)
    fmap = plant._recycle._maybe_flow_map(t0a, states0, params_full)

    # Operating-parameter sensitivity: alongside the kinetic ``wrt`` params,
    # differentiate w.r.t. multiplicative scales on the influent (a
    # differentiable operating input, nominal 1.0). Each spec appends one
    # scalar column to the augmented parameter vector, threaded into the RHS
    # through the ``design`` influent override -- the same path the
    # steady-state IFT uses -- so the cached recycle/flow maps, which are
    # influent-independent, stay exact. The spec + design build are the shared
    # ``_parse_operating`` / ``_operating_design`` (see above), so the steady
    # and dynamic stacks accept an identical operating-input description.
    n_params = int(params_full.shape[0])
    op_meta = plant._parse_operating(operating)
    n_op = len(op_meta)

    if n_op:
        theta_aug = jnp.concatenate([params_full, jnp.ones(n_op)])
        free_aug = jnp.concatenate([free_idx, n_params + jnp.arange(n_op)])
    else:
        theta_aug, free_aug = params_full, free_idx

    def f_flat(t, y, theta):
        design = plant._operating_design(op_meta, theta[n_params:]) if n_op else None
        return plant._rhs(t, y, theta[:n_params], recycle_map=rmap, flow_map=fmap, design=design)

    atol_arr = default_atol(y0) if atol is None else jnp.asarray(atol)
    te = None if t_eval is None else jnp.asarray(t_eval)
    return augmented_forward_sensitivity(
        f_flat,
        y0,
        theta_aug,
        free_aug,
        t0=t0,
        t1=t1,
        t_eval=te,
        rtol=rtol,
        atol_y=atol_arr,
        sens_rtol=sens_rtol,
        dtmax=dtmax,
        max_steps=max_steps,
        shared_factor=int(free_aug.shape[0]) > 1,
        order=3,
        factormax=factormax,
    )


def _dynamic_adjoint_kwargs(mode):
    """The ``solve`` keyword that pins the adjoint matching the AD ``mode``.

    Reverse mode uses the cap-free stable adjoint; forward mode a
    forward-capable (direct) adjoint. This is the choice a user otherwise has
    to make by hand -- the default reverse adjoint goes non-finite on a stiff
    dynamic plant, and forward mode needs a non-``custom_vjp`` adjoint.
    """
    if mode == "reverse":
        return {"diff": DifferentiationConfig(mode="reverse", method="stable")}
    if mode == "forward":
        return {"diff": DifferentiationConfig(mode="forward", method="through_solve")}
    raise ValueError(f"mode must be 'reverse', 'forward', or 'auto'; got {mode!r}.")


def _dynamic_value_jac(
    plant,
    params,
    wrt_idx,
    theta,
    *,
    output_fn,
    t_span,
    t_eval,
    y0,
    mode,
    solve_kwargs,
    operating=None,
):
    """``(value, (m, k) Jacobian)`` of ``output_fn(solve(theta))`` using the
    STABLE method for ``mode``, shared by :meth:`dynamic_sensitivity` and
    :meth:`dynamic_dgsm`.

    Reverse mode differentiates the solve through the cap-free stable discrete
    adjoint. Forward mode integrates the augmented ``[y; S]`` variational system
    (:meth:`solve_sensitivity`) -- finite over long horizons where forward-mode
    ``jacfwd`` through the stiff solve goes non-finite -- and chains the
    full-state sensitivity through ``output_fn`` with one ``jax.linearize`` of
    the output map (no extra solve). ``theta`` is the value of the screened
    parameters at the evaluation point (the nominal vector for a single
    sensitivity, or a Sobol sample for the DGSM screen).
    """
    wrt_j = jnp.asarray(wrt_idx)
    if mode == "forward":
        extra = set(solve_kwargs) - {
            "max_steps",
            "dtmax",
            "factormax",
            "rtol",
            "atol",
            "sens_rtol",
        }
        if extra:
            raise TypeError(
                f"forward-mode dynamic sensitivity runs the solve through "
                f"solve_sensitivity, which does not accept {sorted(extra)}; "
                f"it accepts max_steps / dtmax / factormax / rtol / atol / "
                f"sens_rtol."
            )
        full = params.at[wrt_j].set(theta)
        ts, ys, S_state = plant.solve_sensitivity(
            full,
            list(wrt_idx),
            t_span=t_span,
            t_eval=t_eval,
            y0=y0,
            operating=operating,
            **solve_kwargs,
        )

        from aquakin.plant.plant import PlantSolution

        def g_of_ys(ys_traj):
            return jnp.atleast_1d(output_fn(PlantSolution(t=ts, state=ys_traj, plant=plant)))

        # One linearization of the output map at the saved trajectory (no
        # solve), pushed across the k parameter columns of dy/dtheta -> (m, k).
        g, jvp_g = jax.linearize(g_of_ys, ys)
        S = jax.vmap(jvp_g, in_axes=2, out_axes=1)(S_state)
        return g, S
    if mode == "reverse":
        # Differentiate the solve directly through the cap-free stable adjoint,
        # without an enclosing jit (the primal runs with concrete parameters,
        # which some plant setup needs); the solve's compiled-solve cache reuses
        # the integrator compile across calls.
        solve_adj = _dynamic_adjoint_kwargs("reverse")
        # solve_kwargs may carry the legacy integrator primitives (max_steps /
        # dtmax / factormax); fold them into an IntegratorConfig for Plant.solve,
        # leaving the plain solve tolerances (rtol/atol) and any other kwargs.
        sk = dict(solve_kwargs)
        int_kw = {k: sk.pop(k) for k in ("max_steps", "dtmax", "factormax") if k in sk}
        if int_kw:
            solve_adj = {**solve_adj, "integrator": IntegratorConfig(**int_kw)}

        def f(th):
            p = params.at[wrt_j].set(th)
            sol = plant.solve(t_span, t_eval=t_eval, params=p, y0=y0, **solve_adj, **sk)
            return jnp.atleast_1d(output_fn(sol))

        return _dynamic_value_and_jacobian(f, theta, "reverse")
    raise ValueError(f"mode must be 'reverse', 'forward', or 'auto'; got {mode!r}.")


def dynamic_sensitivity(
    plant,
    params: Optional[jnp.ndarray] = None,
    *,
    output_fn: Callable,
    t_span: tuple,
    t_eval: Optional[jnp.ndarray] = None,
    wrt: Optional[Sequence] = None,
    operating: Optional[Sequence] = None,
    mode: str = "reverse",
    y0: Optional[jnp.ndarray] = None,
    elasticity: bool = False,
    **solve_kwargs,
) -> jnp.ndarray:
    """Sensitivity of a transient (time-window) output to the parameters.

    The sensitivity of a transient output -- the dynamic counterpart of
    :meth:`steady_state_sensitivity`, for outputs that depend on the trajectory
    rather than the operating point (an effluent time series, a window average,
    a peak). Unlike the steady-state case there is no implicit-function-theorem
    shortcut, so the cost is one stiff solve per direction; this wrapper's value
    is that it uses the **stable method for each AD direction** (the footgun it
    removes -- a naive differentiation of :meth:`solve` is non-finite on a stiff
    plant). **Reverse** mode differentiates the solve through the cap-free stable
    discrete adjoint. **Forward** mode integrates the augmented ``[y; S]``
    variational system (:meth:`solve_sensitivity`), whose step controller bounds
    ``S`` so it stays finite over long horizons where forward-mode ``jacfwd``
    through the solve goes non-finite, then chains the full-state sensitivity
    through ``output_fn``. Forward is also the **memory-light** direction over a
    long horizon: ``solve_sensitivity`` carries the parameter tangents in
    lockstep with the state, so its memory is independent of the integration
    length, whereas reverse mode stores the trajectory to replay it. Neither is
    wrapped in an enclosing jit, so the primal runs with concrete parameters
    (which some plant setup requires); the solve's compiled-solve cache reuses
    the integrator compile across calls.

    Parameters
    ----------
    params : jnp.ndarray, optional
        Plant parameters (defaults to :meth:`default_parameters`).
    output_fn : callable
        Maps the :class:`PlantSolution` to a length-``m`` vector of scalar
        outputs (e.g. ``lambda sol: sol.C_named("tank5", "SNO")`` for the
        effluent-nitrate trajectory, or a window average of it).
    t_span, t_eval :
        Integration interval and the times at which the trajectory is saved
        (passed straight to :meth:`solve`).
    wrt : sequence of int or str, optional
        Parameters to differentiate (flat indices or ``"<model>.<param>"``
        names; default all).
    mode : {"reverse", "forward", "auto"}
        AD direction. ``"reverse"`` (default; ``"auto"`` resolves to it) suits
        many parameters / few outputs and uses the stable discrete adjoint;
        ``"forward"`` suits few parameters / many outputs and uses the augmented
        variational solve (:meth:`solve_sensitivity`) -- finite and memory-light
        over long horizons. In forward mode ``**solve_kwargs`` are forwarded to
        :meth:`solve_sensitivity` (``max_steps``, ``dtmax``, ``factormax``,
        ``rtol``, ``atol``, ``sens_rtol``), not to :meth:`solve`.
    y0 : jnp.ndarray, optional
        Initial plant state (e.g. a warm-start steady state).
    elasticity : bool
        If ``True`` return the dimensionless ``(dg/dtheta)(theta/g)``.
    **solve_kwargs
        Forwarded to :meth:`solve` (e.g. ``max_steps``).

    Returns
    -------
    jnp.ndarray
        ``(m, k)`` sensitivity of the ``m`` outputs to the ``k`` parameters.
    """
    params = plant.default_parameters() if params is None else jnp.asarray(params)
    if wrt is None:
        wrt_idx = list(range(int(params.shape[0])))
    else:
        wrt_idx = [plant.parameter_index(w) if isinstance(w, str) else int(w) for w in wrt]
    wrt_j = jnp.asarray(wrt_idx)
    theta0 = params[wrt_j]
    n_op = len(plant._parse_operating(operating))
    # Operating-condition sensitivity rides the augmented variational solve, so
    # it is forward-mode only; default to forward when it is requested.
    chosen = ("forward" if n_op else "reverse") if mode == "auto" else mode
    if n_op and chosen != "forward":
        raise ValueError(
            "operating-condition sensitivity is available only in forward "
            "mode (the augmented variational solve); pass mode='forward'."
        )
    g, S = _dynamic_value_jac(
        plant,
        params,
        wrt_idx,
        theta0,
        output_fn=output_fn,
        t_span=t_span,
        t_eval=t_eval,
        y0=y0,
        mode=chosen,
        solve_kwargs=solve_kwargs,
        operating=operating,
    )
    if elasticity:
        theta_all = jnp.concatenate([theta0, jnp.ones(n_op)]) if n_op else theta0
        S = S * (theta_all[None, :] / g[:, None])
    return S


def dynamic_dgsm(
    plant,
    ranges,
    *,
    output_fn: Callable,
    t_span: tuple,
    t_eval: Optional[jnp.ndarray] = None,
    wrt: Optional[Sequence] = None,
    mode: str = "reverse",
    y0: Optional[jnp.ndarray] = None,
    n_samples: int = 256,
    seed: int = 0,
    output_names: Optional[Sequence] = None,
    progress: Optional[int] = None,
    **solve_kwargs,
) -> "DynamicDGSMResult":
    """Derivative-based global sensitivity (DGSM) of a transient plant output.

    The dynamic counterpart of :meth:`steady_state_dgsm`: scrambled-Sobol QMC
    over the screened parameters, each sample's sensitivity from
    :meth:`dynamic_sensitivity` (differentiated through the dynamic solve, with
    the adjoint selected by ``mode``), aggregated to the Sobol total-index upper
    bound. The solve's compiled-solve cache amortizes the integrator compile
    across samples, but each sample is a full differentiated solve, so a dynamic
    screen is far heavier per sample than the steady-state one; use a modest
    ``n_samples`` / parameter count.

    Parameters mirror :meth:`steady_state_dgsm` (``ranges`` aligned with
    ``wrt``, ``output_fn`` mapping the :class:`PlantSolution` to ``m`` outputs)
    plus ``t_span`` / ``t_eval`` for the window. Returns a
    :class:`DynamicDGSMResult` (``sobol_total_bound`` / ``std_error`` shape
    ``(m, k)``, ``.ranked(output)``, ``.convergence()``).
    """
    import numpy as np

    from ..integrate._qmc import _sobol_sample

    params = plant.default_parameters()
    if wrt is None:
        wrt_idx = list(range(int(params.shape[0])))
    else:
        wrt_idx = [plant.parameter_index(w) if isinstance(w, str) else int(w) for w in wrt]
    k = len(wrt_idx)
    ranges_arr = jnp.asarray(ranges, dtype=params.dtype)
    if ranges_arr.shape != (k, 2):
        raise ValueError(
            f"ranges must have shape ({k}, 2) aligned with the {k} screened "
            f"parameters; got {tuple(ranges_arr.shape)}."
        )
    lo, hi = ranges_arr[:, 0], ranges_arr[:, 1]
    chosen = "reverse" if mode == "auto" else mode

    def _per_sample(z):
        # The stable value+Jacobian at the Sobol sample z (reverse: stable
        # adjoint; forward: the augmented [y; S] variational solve). Bare (no
        # enclosing jit): the primal runs with concrete parameters, so plant
        # setup that needs a concrete value does not see a tracer; the solve's
        # compiled-solve cache amortizes the integrator compile across samples.
        return _dynamic_value_jac(
            plant,
            params,
            wrt_idx,
            z,
            output_fn=output_fn,
            t_span=t_span,
            t_eval=t_eval,
            y0=y0,
            mode=chosen,
            solve_kwargs=solve_kwargs,
        )

    Z, n_drawn = _sobol_sample(lo, hi, k, n_samples, seed)
    outs, grads = [], []
    for i in range(n_drawn):
        g, S = _per_sample(Z[i])
        outs.append(np.asarray(g))
        grads.append(np.asarray(S))
        if progress and (i + 1) % progress == 0:
            print(f"  [dynamic_dgsm] {i + 1}/{n_drawn} samples", flush=True)

    outputs = np.stack(outs)  # (N, m)
    grad_sq = np.stack(grads) ** 2  # (N, m, k)
    rng2 = np.asarray((hi - lo) ** 2)  # (k,)
    bound, se, nu, var, n_valid = _dynamic_dgsm_bounds(grad_sq, outputs, rng2)

    names = plant.parameter_names()
    input_names = [names[i] for i in wrt_idx]
    m = int(outputs.shape[1])
    if output_names is None:
        output_names = [f"output{i}" for i in range(m)]
    return DynamicDGSMResult(
        input_names=input_names,
        output_names=list(output_names),
        sobol_total_bound=jnp.asarray(bound),
        std_error=jnp.asarray(se),
        nu=jnp.asarray(nu),
        output_variance=jnp.asarray(var),
        ranges=ranges_arr,
        n_samples=int(n_drawn),
        seed=int(seed),
        grad_sq=jnp.asarray(grad_sq),
        outputs=jnp.asarray(outputs),
        n_valid=jnp.asarray(n_valid),
    )
