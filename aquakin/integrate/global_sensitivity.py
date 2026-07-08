"""Derivative-based global sensitivity (DGSM) via autodiff + Sobol QMC.

Estimates each input's derivative-based global sensitivity measure
``nu_j = E[(d output / d z_j)^2]`` by averaging the squared partial derivative
over scrambled-Sobol quasi-random points, which bounds the Sobol total-order
index (Lamboni, Sobol & Kucherenko 2013) -- the AD analogue of a variance-based
Sobol total index. The scrambled-Sobol samplers live in
:mod:`aquakin.integrate._qmc` (shared with the design-of-experiments workflows),
and the Sobol total-index aggregation kernel (:func:`_dgsm_aggregate`) is shared
with the plant steady-state / dynamic screens in
:mod:`aquakin.plant.sensitivity`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._common import DifferentiationConfig, is_forward_mode_ad_error
from aquakin.integrate._qmc import _sobol_sample


@dataclass
class DGSMResult:
    """Result of :func:`dgsm`.

    Attributes
    ----------
    input_names : list[str]
        Names of the uncertain inputs, matching the rows of every array.
    dgsm : jnp.ndarray
        The derivative-based global sensitivity measure
        ``nu_j = E[(d output / d z_j)^2]``, shape ``(d,)``.
    sobol_total_bound : jnp.ndarray
        Upper bound on the Sobol total-order index of each input,
        ``S_j^tot <= nu_j (b_j - a_j)^2 / (pi^2 Var(f))`` for ``z_j`` uniform on
        ``[a_j, b_j]`` (Lamboni, Sobol & Kucherenko 2013). Dimensionless and
        directly comparable across inputs -- the AD-accelerated replacement for
        a variance-based Sobol total index.
    std_error : jnp.ndarray
        Monte-Carlo standard error of ``sobol_total_bound`` (convergence
        indicator). Shrinks like ``1/sqrt(n_valid)``.
    output_variance : float
        Variance of the scalar output over the sample.
    n_samples : int
        Number of quasi-random points actually drawn (a power of two).
    n_valid : int
        Number of points with a finite output and gradient (others skipped). For
        a vector-valued ``fn`` this is counted **per output** -- a sample
        non-finite in another output is not dropped from this one -- so different
        outputs may report different ``n_valid``.
    seed : int
        Seed of the scrambled-Sobol sampler -- fixing it makes the result
        bit-for-bit reproducible.
    ranges : jnp.ndarray
        The ``(d, 2)`` input ranges used.
    """

    input_names: list[str]
    dgsm: jnp.ndarray
    sobol_total_bound: jnp.ndarray
    std_error: jnp.ndarray
    output_variance: float
    n_samples: int
    n_valid: int
    seed: int
    ranges: jnp.ndarray
    output_name: str | None = None

    def ranked(self) -> list[tuple[str, float]]:
        """Return ``(name, sobol_total_bound)`` pairs sorted by decreasing bound."""
        pairs = [(n, float(b)) for n, b in zip(self.input_names, self.sobol_total_bound)]
        return sorted(pairs, key=lambda kv: kv[1], reverse=True)


# Guidance raised when a forward-mode screen hits the default reactor adjoint's
# custom_vjp (which rejects jvp). Shared by the batched and per-sample paths.
_DGSM_FORWARD_HINT = (
    "ad_mode='forward' requires forward-mode autodiff through the solve. Build "
    "the reactor inside fn with adjoint=aquakin.forward_adjoint() (dgsm cannot "
    "set the adjoint for you -- your fn constructs the reactor); the default "
    "RecursiveCheckpointAdjoint registers a custom_vjp that rejects forward mode."
)


def _validate_dgsm_ranges(ranges, input_names):
    """Coerce/validate ``ranges`` and ``input_names``.

    Returns ``(ranges_np, lo, hi, d, input_names)`` with ``input_names`` filled
    in (``z0, z1, ...``) when not supplied.
    """
    ranges_np = np.asarray(ranges, dtype=float)
    if ranges_np.ndim != 2 or ranges_np.shape[1] != 2:
        raise ValueError(f"ranges must have shape (d, 2); got {ranges_np.shape}.")
    d = ranges_np.shape[0]
    lo, hi = ranges_np[:, 0], ranges_np[:, 1]
    if not np.all(hi > lo):
        raise ValueError("each range must satisfy upper > lower.")
    if input_names is None:
        input_names = [f"z{j}" for j in range(d)]
    elif len(input_names) != d:
        raise ValueError(f"input_names has {len(input_names)} entries but ranges has d={d}.")
    return ranges_np, lo, hi, d, list(input_names)


def _make_dgsm_value_and_jac(fn, z0, mode):
    """Build the jitted ``(value, Jacobian)`` callable for the requested mode.

    Probes the output rank once (via :func:`jax.eval_shape`, no solve) to choose
    between scalar (``value_and_grad`` / ``jacfwd``) and vector
    (``jacrev`` / ``jacfwd``) Jacobians. Returns
    ``(value_and_jac, vector, m_out)``; the Jacobian is shape ``(d,)`` for a
    scalar output and ``(m, d)`` for a vector output.
    """
    f_arr = lambda z: jnp.asarray(fn(z))
    out_shape = jax.eval_shape(f_arr, jnp.asarray(z0)).shape
    vector = len(out_shape) == 1
    m_out = int(out_shape[0]) if vector else 1
    if mode == "reverse":
        if vector:
            value_and_jac = jax.jit(lambda z: (f_arr(z), jax.jacrev(f_arr)(z)))
        else:
            value_and_jac = jax.jit(jax.value_and_grad(f_arr))
    else:  # forward
        value_and_jac = jax.jit(lambda z: (f_arr(z), jax.jacfwd(f_arr)(z)))
    return value_and_jac, vector, m_out


def _dgsm_aggregate(grad_sq, outputs, rng2, sample_mask=None, poincare=None):
    """Sobol total-index bound from per-sample squared sensitivities, robustly.

    Drops, **per output**, any sample with a non-finite output or sensitivity --
    the extreme Sobol corners where the output is hard to resolve -- and, when
    ``sample_mask`` is given, any sample it marks ``False`` (e.g. a
    near-singular-Jacobian steady-state operating point). Then forms
    ``nu_ij = mean_s (dg_i/dz_j)^2``, ``Var(g_i)``, the Sobol total-index bound
    ``nu_ij (b_j-a_j)^2 / (pi^2 Var(g_i))`` and its Monte-Carlo standard error. An
    output that does not vary (zero variance) has an undefined bound -- returned
    as ``NaN`` (the public :func:`dgsm` entry point reports it as ``0`` with a
    warning; the plant screens leave it ``NaN``).

    Shared by :func:`dgsm` (uniform inputs, ``poincare=None`` ->
    ``(b_j-a_j)^2/pi^2``) and the plant steady-state / dynamic DGSM screens (which
    pass a ``sample_mask`` and, for Gaussian inputs, an explicit ``poincare``).

    Parameters
    ----------
    grad_sq : ndarray, shape ``(N, m, k)`` ; outputs : ndarray, shape ``(N, m)`` ;
    rng2 : ndarray, shape ``(k,)`` -- the squared screened ranges ``(b_j-a_j)^2``.
    sample_mask : ndarray bool, shape ``(N,)``, optional -- samples to keep.
    poincare : ndarray, shape ``(k,)``, optional -- the per-input Poincare constant
        ``C_j`` (Gaussian inputs); defaults to the uniform ``rng2 / pi^2``.

    Returns
    -------
    bound, std_error, nu : ndarray, shape ``(m, k)``
    var : ndarray, shape ``(m,)`` ; n_valid : ndarray int, shape ``(m,)``
    """
    grad_sq = np.asarray(grad_sq)
    outputs = np.asarray(outputs)
    _, m, k = grad_sq.shape
    valid = np.isfinite(outputs) & np.isfinite(grad_sq).all(axis=2)  # (N, m)
    if sample_mask is not None:
        valid = valid & np.asarray(sample_mask)[:, None]
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


def _evaluate_dgsm_samples(value_and_jac, Z, mode, batched):
    """Evaluate the value/Jacobian over every sample; return the full stacked
    arrays (non-finite rows **included**).

    Finiteness is filtered downstream *per output* (see :func:`_dgsm_aggregate`),
    so this returns every drawn row -- a sample whose value/gradient is non-finite
    in one output must still contribute to the others. ``batched=True`` dispatches
    the whole sample through one :func:`jax.vmap` (a single device->host
    transfer); ``batched=False`` is the per-sample fallback (one host transfer
    each, lower peak memory). Both return identical ``(vals, jacs)`` NumPy arrays:
    ``vals`` is ``(N,)``/``(N, m)`` and ``jacs`` is ``(N, d)``/``(N, m, d)``.
    """
    if batched:
        try:
            vals, jacs = jax.vmap(value_and_jac)(jnp.asarray(Z))
        except Exception as exc:
            if mode == "forward" and is_forward_mode_ad_error(exc):
                raise RuntimeError(_DGSM_FORWARD_HINT) from exc
            raise
        return np.asarray(vals), np.asarray(jacs)

    v_list: list[np.ndarray] = []
    j_list: list[np.ndarray] = []
    for z in Z:
        try:
            v, J = value_and_jac(jnp.asarray(z))
        except Exception as exc:
            if mode == "forward" and is_forward_mode_ad_error(exc):
                raise RuntimeError(_DGSM_FORWARD_HINT) from exc
            raise
        v_list.append(np.asarray(v))
        j_list.append(np.asarray(J))
    return np.asarray(v_list), np.asarray(j_list)


def dgsm(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    ranges: Any,
    *,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    n_samples: int = 64,
    seed: int = 0,
    diff: DifferentiationConfig = DifferentiationConfig(),
    batched: bool = True,
) -> Any:
    """Derivative-based global sensitivity measure via autodiff + Sobol QMC.

    Estimates, for each uncertain input ``z_j``,

        ``nu_j = E_z[ (d fn / d z_j)^2 ]``

    by averaging the squared partial derivative over scrambled-Sobol
    quasi-random points in the input ranges. ``nu_j`` bounds the Sobol
    total-order index (see :attr:`DGSMResult.sobol_total_bound`), so it is the
    AD analogue of a variance-based Sobol total index, obtained from
    derivatives rather than a variance decomposition.

    The derivatives are exact (no finite-difference truncation) and reuse the
    differentiable model, so the same machinery serves the calibration and
    identifiability analyses. The cost depends on ``ad_mode`` and on the number
    of outputs ``m`` and inputs ``d``:

    - ``ad_mode="reverse"`` (default) forms the per-sample sensitivities with
      ``m`` reverse-mode passes (one per output), each independent of ``d``.
      Best when there are few outputs relative to inputs **and** the adjoint is
      cheap. Works with any reactor adjoint.
    - ``ad_mode="forward"`` forms them with ``d`` forward-mode tangents pushed
      through a single solve, independent of ``m``. Best when there are many
      outputs, or when the reverse adjoint is expensive -- e.g. a stiff solve
      whose differentiated step must be capped (``dtmax``), which inflates the
      reverse pass. **The reactor inside ``fn`` must then be built with**
      ``adjoint=aquakin.forward_adjoint()`` (``dgsm`` cannot set the adjoint
      for you, because ``fn`` constructs the reactor): the default
      ``RecursiveCheckpointAdjoint`` registers a ``custom_vjp`` that rejects
      forward-mode autodiff.

    Both modes return identical sensitivities (to machine precision);
    ``ad_mode`` is purely a performance choice. For a single scalar output
    ``reverse`` is
    almost always cheaper; the ``forward`` advantage appears for multi-output
    screening of a stiff model.

    Parameters
    ----------
    fn : callable
        Maps an input vector (shape ``(d,)``) to either a scalar JAX value or a
        vector of ``m`` outputs (shape ``(m,)``). Must be ``jax``-differentiable
        in the requested ``mode``. For a reactor study, ``fn`` typically maps the
        uncertain inputs into a parameter vector / initial state, calls
        ``reactor.solve`` and reduces the solution to the output(s). If the
        model is stiff, build the reactor with a suitable ``dtmax`` so the
        differentiated solve stays finite. ``dgsm`` does not own the solve (your
        ``fn`` builds the reactor and chooses the ``t_eval``), so it cannot apply
        a ``time_unit`` conversion for you: any ``t_eval`` / ``t_span`` inside
        ``fn`` must be in the model's **native** time unit, or ``fn`` must pass
        ``time_unit=`` to its own ``reactor.solve`` call.
    ranges : array-like, shape (d, 2)
        ``[lower, upper]`` bound for each input; sampling is uniform within.
    input_names : list[str], optional
        Names for reporting; defaults to ``["z0", "z1", ...]``.
    output_names : list[str], optional
        Names for the ``m`` outputs when ``fn`` is vector-valued; defaults to
        ``["output0", ...]``. Ignored for a scalar ``fn``.
    n_samples : int, optional
        Target number of quasi-random points; rounded to the nearest power of
        two (Sobol sequences are balanced at powers of two). Increase until
        ``std_error`` is small relative to the ranking gaps.
    seed : int, optional
        Seed for the scrambled-Sobol sampler. Fixing it (the default ``0``)
        makes the analysis exactly reproducible.
    diff : DifferentiationConfig, optional
        Autodiff configuration. ``mode`` ({"reverse", "forward"}) selects the
        direction used to form the per-sample sensitivities (see above).
    batched : bool, optional
        When ``True`` (default) the whole sample is pushed through one
        ``jax.vmap`` dispatch and finiteness is filtered once on the stacked
        result -- one device->host transfer instead of one per point. Set
        ``False`` to evaluate point-by-point (lower peak memory for a large
        screen). Both give identical results.

    Returns
    -------
    DGSMResult or list[DGSMResult]
        A single :class:`DGSMResult` when ``fn`` is scalar-valued, or a list of
        results (one per output, in order, each carrying its ``output_name``)
        when ``fn`` is vector-valued.

    Examples
    --------
    >>> def fn(z):                       # output sensitive to z0, not z1
    ...     return 3.0 * z[0] + 0.0 * z[1]
    >>> res = aquakin.dgsm(fn, [(0.0, 1.0), (0.0, 1.0)], input_names=["a", "b"])
    >>> res.ranked()[0][0]
    'a'
    """
    diff.validated()
    mode = diff.mode

    ranges_np, lo, hi, d, input_names = _validate_dgsm_ranges(ranges, input_names)
    Z, n_drawn = _sobol_sample(lo, hi, d, n_samples, seed)
    value_and_jac, vector, m_out = _make_dgsm_value_and_jac(fn, Z[0], mode)
    vals, jacs = _evaluate_dgsm_samples(value_and_jac, Z, mode, batched)

    # Reshape to the (N, m, d) / (N, m) layout the shared aggregator expects, and
    # resolve the output names.
    if vector:
        if output_names is None:
            output_names = [f"output{i}" for i in range(m_out)]
        elif len(output_names) != m_out:
            raise ValueError(
                f"output_names has {len(output_names)} entries but fn returns m={m_out} outputs."
            )
        names: list[str | None] = list(output_names)
        grad_sq = jacs**2  # (N, m, d)
        out_col = vals  # (N, m)
    else:
        names = [None]
        grad_sq = (jacs**2)[:, None, :]  # (N, 1, d)
        out_col = vals[:, None]  # (N, 1)

    # Sobol total-index bound per output via the shared aggregator. Uniform inputs
    # (poincare=None -> (b-a)^2/pi^2); finiteness is filtered PER OUTPUT inside it,
    # so a sample non-finite in one output is dropped only for that one and each
    # output's nu_j / n_valid stay unbiased by the others' failures.
    rng2 = (hi - lo) ** 2
    bound, bound_se, nu, var, n_valid = _dgsm_aggregate(grad_sq, out_col, rng2)

    results = []
    for i, name in enumerate(names):
        n = int(n_valid[i])
        if n < 2:
            raise RuntimeError(
                f"DGSM needs >= 2 finite samples"
                f"{f' for output {name!r}' if name else ''}; got {n}/{n_drawn}. "
                "The output or its gradient is non-finite over the sampled ranges "
                "-- for a stiff model, cap the integrator step via the "
                "reactor's dtmax."
            )
        var_f = float(var[i])
        if var_f > 0:
            bnd = np.asarray(bound[i])
            se = np.asarray(bound_se[i])
        else:
            # Every (finite) sample produced an identical output (e.g. a saturated
            # or clipped response): the Sobol total-index bound is undefined (0/0).
            # Report an all-zero bound but warn, so an empty ranking is not
            # silently read as "no input matters".
            warnings.warn(
                f"DGSM output{f' {name!r}' if name else ''} has zero variance "
                f"over the sampled ranges; the Sobol total-index bound is "
                f"undefined and reported as 0. The output may be saturated, "
                f"clipped, or insensitive to every input over these ranges.",
                stacklevel=2,
            )
            bnd = np.zeros(d)
            se = np.zeros(d)
        results.append(
            DGSMResult(
                input_names=list(input_names),
                dgsm=jnp.asarray(nu[i]),
                sobol_total_bound=jnp.asarray(bnd),
                std_error=jnp.asarray(se),
                output_variance=var_f,
                n_samples=n_drawn,
                n_valid=n,
                seed=seed,
                ranges=jnp.asarray(ranges_np),
                output_name=name,
            )
        )
    return results if vector else results[0]
