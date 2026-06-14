"""Shared internals for the integrator submodules.

Not part of the public API. Reactors depend on this; this module depends only
on JAX and Diffrax.
"""

from __future__ import annotations

import contextlib
import copy
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, runtime_checkable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from aquakin.core.network import CompiledNetwork

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.conditions import SpatialConditions


# --- AD-mode helpers (hide the diffrax adjoint plumbing) ---------------------


def forward_adjoint() -> "diffrax.AbstractAdjoint":
    """Return the diffrax adjoint that supports forward-mode autodiff.

    A thin, dependency-free alias for ``diffrax.DirectAdjoint()`` so a user
    script that needs forward-mode AD through a reactor solve (e.g. the reactor
    inside a ``dgsm(ad_mode="forward")`` ``fn``) can write
    ``adjoint=aquakin.forward_adjoint()`` without importing ``diffrax`` or
    knowing that the default ``RecursiveCheckpointAdjoint`` registers a
    ``custom_vjp`` that rejects forward mode.
    """
    return diffrax.DirectAdjoint()


def with_adjoint(reactor, adjoint):
    """Return a shallow copy of ``reactor`` with its adjoint strategy replaced.

    Reactors are stateless after construction and read ``self.adjoint`` at solve
    time, so a shallow copy with a swapped ``adjoint`` is a valid forward-/
    reverse-capable variant of the same reactor. Used by ``calibrate`` /
    ``sensitivity`` to build the right adjoint internally from an ``ad_mode``
    string, so ``diffrax`` never appears in user code.
    """
    clone = copy.copy(reactor)
    clone.adjoint = adjoint
    return clone


def check_finite_gradient(value, *, what: str, remedy: str) -> None:
    """Raise a friendly ``RuntimeError`` if ``value`` is non-finite.

    The silent-NaN footgun of differentiating a stiff solve: the gradient comes
    back ``NaN``/``Inf`` and nothing says why. Call this on a freshly computed
    gradient/Jacobian to convert that into an actionable error.

    Parameters
    ----------
    value : array-like
        The gradient or Jacobian to check.
    what : str
        Short noun for the message (e.g. ``"calibration gradient"``).
    remedy : str
        The concrete fix to suggest.
    """
    if not bool(np.isfinite(np.asarray(value)).all()):
        raise RuntimeError(
            f"The {what} is non-finite (NaN/Inf). This is almost always the "
            f"reverse-mode adjoint of a stiff solve overflowing, not a bug in "
            f"your model. {remedy}"
        )


class GradientCheckMixin:
    """Give every reactor a finiteness check for a hand-rolled reverse gradient.

    A reverse-mode gradient (``jax.grad``/``jax.jacrev``) taken *directly*
    through a stiff network's :meth:`solve` -- a user's own loss + optimizer,
    outside :func:`aquakin.calibrate` / :func:`aquakin.sensitivity`, which guard
    this internally -- can return silent ``NaN``/``Inf`` when the integrator step
    is uncapped (the backward accumulation overflows; see the ``dtmax`` note on
    the reactor). Nothing raises, so the garbage gradient flows into the
    optimizer and the run "works" but never converges.

    This mixin adds :meth:`check_gradient_finite`, a one-call guard the DIY user
    wraps their gradient in. The remedy it suggests is tailored to whether the
    reactor already caps ``dtmax``.
    """

    def check_gradient_finite(self, grad_value, *, what: str = "gradient"):
        """Raise a friendly error if a reverse-mode gradient is non-finite.

        Wrap a freshly computed ``jax.grad`` result through this reactor's
        ``solve`` to turn a silent non-finite gradient into an actionable error::

            g = reactor.check_gradient_finite(jax.grad(loss)(params))

        Parameters
        ----------
        grad_value : array-like
            The gradient/Jacobian to check (returned unchanged when finite, so
            the call composes inline).
        what : str, optional
            Short noun for the message (default ``"gradient"``).

        Returns
        -------
        The ``grad_value`` argument, unchanged.

        Raises
        ------
        RuntimeError
            If ``grad_value`` contains any non-finite entry.
        """
        if getattr(self, "dtmax", None) is None:
            remedy = (
                "A reverse-mode gradient through a stiff solve overflows in the "
                "backward pass when the integrator step is uncapped. Build the "
                "reactor with a dtmax cap (a small multiple of the fastest "
                "reaction timescale), or differentiate in forward mode "
                "(jax.jacfwd with adjoint=aquakin.forward_adjoint()). "
                "aquakin.calibrate (gradient='stable_adjoint', cap-free) and "
                "aquakin.sensitivity handle this for you."
            )
        else:
            remedy = (
                f"The reactor already caps dtmax={self.dtmax}; if the gradient is "
                "still non-finite, tighten dtmax further, or check the model, "
                "data and parameter ranges."
            )
        check_finite_gradient(grad_value, what=what, remedy=remedy)
        return grad_value


# --- Tabular export helpers (optional pandas) --------------------------------


def require_pandas():
    """Import and return pandas, with a helpful message if it is missing.

    pandas is an optional dependency, used only by the ``to_dataframe()`` /
    ``to_csv()`` result exporters.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "to_dataframe() / to_csv() require pandas, an optional dependency. "
            "Install it with `pip install pandas` or `pip install "
            "aquakin[dataframe]`."
        ) from e
    return pd


def require_matplotlib():
    """Import and return ``matplotlib.pyplot``, with a helpful message if missing.

    matplotlib is an optional dependency, used only by the ``plot()`` result
    helpers.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "plot() requires matplotlib, an optional dependency. Install it with "
            "`pip install matplotlib` or `pip install aquakin[plot]`."
        ) from e
    return plt


def build_dataframe(
    index,
    columns,
    *,
    index_name=None,
    units=None,
    units_in_columns=False,
    extra=None,
):
    """Assemble a pandas ``DataFrame`` from an index and named value columns.

    Shared by every result exporter so the column/units conventions stay
    consistent.

    Parameters
    ----------
    index : array-like or pandas.Index
        The row index. A plain array is wrapped in a ``pd.Index`` named
        ``index_name``; a pre-built ``Index``/``MultiIndex`` is used as-is.
    columns : list of (str, array)
        ``(species_name, 1-D values)`` pairs, in display order.
    index_name : str, optional
        Name for the index when ``index`` is a plain array.
    units : dict, optional
        ``{species_name: unit_string}``; stored in ``df.attrs["units"]`` and,
        when ``units_in_columns`` is set, appended to the column labels.
    units_in_columns : bool, optional
        Append ``" [unit]"`` to each column label.
    extra : list of (str, array), optional
        Additional non-species columns to place before the value columns (e.g.
        a flow ``Q`` or a ``depth`` column). Never relabelled with units.

    Returns
    -------
    pandas.DataFrame
    """
    pd = require_pandas()
    units = units or {}
    if isinstance(index, (pd.Index, pd.MultiIndex)):
        idx = index
    else:
        idx = pd.Index(np.asarray(index), name=index_name)
    data = {}
    for name, arr in extra or []:
        data[name] = np.asarray(arr)
    for name, arr in columns:
        unit = units.get(name, "")
        label = f"{name} [{unit}]" if (units_in_columns and unit) else name
        data[label] = np.asarray(arr)
    df = pd.DataFrame(data, index=idx)
    df.attrs["units"] = dict(units)
    return df


# --- Cross-instance compiled-solver cache ------------------------------------
#
# Each ``reactor.solve(...)`` jit-compiles an inner ``_solve`` closure that
# captures the network and the solver settings. Compilation (trace + lower +
# XLA) dominates the cost of a solve -- the run itself is comparatively free --
# so rebuilding that closure for every reactor instance means every fresh
# reactor pays a full from-scratch compile, even for an identical network and
# settings. That is the dominant cost of the test suite (build a reactor, solve
# once) and of any code that constructs many short-lived reactors.
#
# This module-level cache shares the compiled solver across *all* reactor
# instances keyed by everything the compiled computation depends on: the
# network identity, the solver settings, and the call signature. A repeat solve
# of the same (network, settings, signature) then reuses the compiled graph and
# runs in milliseconds. The cache holds a reference to the network (and any
# settings objects, e.g. a custom adjoint) so their ``id()`` cannot be reused by
# a later object while the entry is live -- keying on ``id`` is therefore safe.
#
# Built-in networks are themselves cached by name (``load_network``), so the
# same network object is reused across calls and the ``id()`` key is stable
# across the whole process. The key NEVER omits anything that changes the
# compiled result, so a hit always returns a solver compiled for the exact same
# computation (no false hits); argument shapes/dtypes (C0, params, conditions,
# t_eval) are handled by JAX's own per-function cache and need not be keyed here.
_SOLVER_CACHE: dict = {}


def atol_cache_key(atol):
    """A hashable, value-based key for an (array or scalar) absolute tolerance.

    ``atol`` is often a per-component array derived from the network, so a fresh
    array object each reactor; key on its *values* so identical tolerances share
    a cache entry.
    """
    a = jnp.asarray(atol)
    return (tuple(a.shape), tuple(float(x) for x in np.asarray(a).reshape(-1)))


def settings_cache_key(rtol, atol, adjoint, dtmax, max_steps):
    """Hashable key for the solver settings that affect the compiled solve.

    ``adjoint`` is keyed by identity (``None`` for the default, which is shared);
    a custom adjoint object keys by ``id`` -- safe (never a false hit), and it
    shares whenever the same object is reused.
    """
    return (
        float(rtol),
        atol_cache_key(atol),
        None if adjoint is None else id(adjoint),
        None if dtmax is None else float(dtmax),
        int(max_steps),
    )


def concrete_settings_key(rtol, atol, adjoint, dtmax, max_steps):
    """Return :func:`settings_cache_key`, or ``None`` if a value is traced.

    The key materialises ``atol`` to Python floats, which is impossible when
    ``solve`` runs *under tracing* (e.g. a calibration loss differentiating
    through the solve). In that case the shared cache gives no benefit anyway --
    the solve is being traced into an outer computation that JAX compiles as a
    whole -- so return ``None`` to signal "build without caching".
    """
    try:
        return settings_cache_key(rtol, atol, adjoint, dtmax, max_steps)
    except jax.errors.TracerArrayConversionError:
        return None


def cached_jitted_solver(key, build, *keep_alive):
    """Return the cached compiled solver for ``key``, building it once.

    ``build`` is a zero-arg factory returning the jitted ``_solve``; it is called
    only on a cache miss. ``keep_alive`` objects (the network, a custom adjoint)
    are retained by the cache so their ``id()`` stays valid for the lifetime of
    the entry. A ``key`` of ``None`` (an un-cacheable, traced call) bypasses the
    cache and just builds.
    """
    if key is None:
        return build()
    entry = _SOLVER_CACHE.get(key)
    if entry is None:
        fn = build()
        _SOLVER_CACHE[key] = (fn, *keep_alive)
        return fn
    return entry[0]


def validate_t_eval(t_eval_arr: jnp.ndarray, t0: float, t1: float) -> None:
    """Validate output times before handing them to ``SaveAt(ts=...)``.

    Diffrax silently returns NaN for save times outside ``[t0, t1]`` or a
    non-ascending sequence, so check here for a clear error instead. Value
    checks run only for concrete (non-traced) ``t_eval``; a traced array
    (e.g. differentiating with respect to the save times) skips them.

    Parameters
    ----------
    t_eval_arr : jnp.ndarray
        Candidate save times.
    t0, t1 : float
        Integration interval bounds.

    Raises
    ------
    ValueError
        If ``t_eval`` is not 1-D, lies outside ``[t0, t1]``, or is not
        strictly ascending.
    """
    if t_eval_arr.ndim != 1:
        raise ValueError(
            f"t_eval must be 1-D; got shape {tuple(t_eval_arr.shape)}."
        )
    if isinstance(t_eval_arr, jax.core.Tracer):
        return
    t_eval_np = np.asarray(t_eval_arr)
    if t_eval_np.size == 0:
        return
    lo, hi = float(t_eval_np.min()), float(t_eval_np.max())
    if lo < t0 or hi > t1:
        raise ValueError(
            f"t_eval must lie within t_span [{t0}, {t1}]; got values in "
            f"[{lo}, {hi}]."
        )
    if t_eval_np.size > 1 and not np.all(np.diff(t_eval_np) > 0):
        raise ValueError("t_eval must be strictly ascending.")


class _HasNamedSpecies:
    """Mixin: provides ``C_named`` given a ``.C`` array and ``.network``.

    Solution dataclasses inherit from this to share the species-by-name
    accessor without duplicating the implementation.
    """

    C: jnp.ndarray  # set by the dataclass subclass
    network: CompiledNetwork  # set by the dataclass subclass

    def C_named(self, species: str) -> jnp.ndarray:
        """Return the trajectory of a single species by name."""
        if species not in self.network.species_index:
            import difflib
            hint = difflib.get_close_matches(species, self.network.species, n=3)
            suffix = f" Did you mean: {', '.join(hint)}?" if hint else ""
            raise KeyError(
                f"Unknown species '{species}'. Available: "
                f"{self.network.species}.{suffix}"
            )
        return self.C[:, self.network.species_index[species]]

    def C_named_many(self, species) -> "dict[str, jnp.ndarray]":
        """Trajectories of several species by name, as ``{name: array}``.

        The multi-species companion to :meth:`C_named` -- read a handful of
        species in one call (``sol.C_named_many(["SNH", "SNO", "SO"])``) instead
        of a separate slice each. Each value has the same shape ``C_named``
        returns. An unknown name raises the same hinted ``KeyError``.
        """
        return {sp: self.C_named(sp) for sp in species}

    def final_named(self, species=None) -> "dict[str, float]":
        """Values at the **last** recorded point, as ``{name: float}``.

        The reporting shortcut for a steady-state / end-of-run value: instead of
        ``float(sol.C_named("SNH")[-1])`` per species, ``sol.final_named(["SNH",
        "SNO"])`` returns them in one dict. With ``species=None`` (default) every
        network species is returned. Values are plain Python floats (this is a
        post-processing read on an already-solved solution -- use
        ``C_named(sp)[-1]`` if you need a differentiable last value).
        """
        names = list(self.network.species) if species is None else list(species)
        return {sp: float(self.C_named(sp)[-1]) for sp in names}

    @property
    def final(self) -> "dict[str, float]":
        """Every species' value at the last recorded point (``{name: float}``).

        The no-argument attribute form of :meth:`final_named` -- ``sol.final``."""
        return self.final_named()

    def units_named(self, species: str) -> str:
        """Return the declared units of a species (for axis/column labels).

        Convenience for plotting and tabulating results without re-deriving
        units by string-matching species names. Equivalent to
        ``self.network.units_of(species)``.
        """
        return self.network.units_of(species)

    @property
    def time_unit(self) -> "str | None":
        """The time unit of ``self.t`` (``"s"``, ``"d"``, ... or ``None``).

        Defaults to the network's native unit (:attr:`CompiledNetwork.time_unit`),
        surfaced here so a plot or table can label the time axis unambiguously.
        When the solve was called with an explicit ``time_unit=`` the result times
        are reported in *that* unit, and this returns it."""
        override = getattr(self, "_requested_time_unit", None)
        return override if override is not None else self.network.time_unit

    def _table_index(self) -> "tuple[str, jnp.ndarray]":
        """Return ``(name, array)`` for the dataframe index. Time by default;
        space-indexed solutions (PFR) override this."""
        return "t", self.t

    def _independent_axis_label(self) -> str:
        """Label for the plot's independent axis. Time (with the network's
        time unit) by default; a space-indexed PFR overrides this."""
        unit = self.time_unit
        return f"time [{unit}]" if unit else "time"

    def plot(self, species=None, *, ax=None, **kwargs):
        """Plot one or more species against the independent axis (time, or axial
        position for a PFR).

        A thin wrapper over matplotlib so "plot SNH over time" needs no manual
        ``C_named`` / unit / axis-label boilerplate. The x-axis is labelled with
        the network's time unit (or position for a PFR), and a single-species
        plot labels the y-axis with that species' units.

        Parameters
        ----------
        species : str or iterable of str, optional
            Species to plot. A single name plots one line; an iterable plots
            several with a legend; ``None`` (default) plots every species.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If ``None``, a new figure/axes is created.
        **kwargs
            Forwarded to ``ax.plot`` (e.g. ``lw``, ``ls``, ``color``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on (for further customisation / saving).

        Raises
        ------
        ImportError
            If matplotlib is not installed (optional dependency; install with
            ``pip install aquakin[plot]``).
        KeyError
            If a species name is unknown (with a "did you mean?" hint).
        """
        plt = require_matplotlib()
        names = ([species] if isinstance(species, str)
                 else list(self.network.species) if species is None
                 else list(species))
        if ax is None:
            _, ax = plt.subplots()
        _, x = self._table_index()
        x = np.asarray(x)
        for sp in names:
            ax.plot(x, np.asarray(self.C_named(sp)), label=sp, **kwargs)
        ax.set_xlabel(self._independent_axis_label())
        if len(names) == 1:
            ax.set_ylabel(f"{names[0]} [{self.units_named(names[0])}]")
        else:
            ax.set_ylabel("concentration")
            ax.legend()
        return ax

    def to_dataframe(self, *, units_in_columns: bool = False):
        """Return the solution as a pandas ``DataFrame``.

        One row per recorded point, one column per species (in network
        ordering), indexed by the independent axis (time ``t`` for batch /
        track / biofilm solutions, axial position ``x`` for a PFR).

        Parameters
        ----------
        units_in_columns : bool, optional
            If ``True``, append ``" [unit]"`` to each species column label
            (e.g. ``"SNH [g_N/m³]"``). If ``False`` (default), columns are bare
            species names and the per-species units are stored in
            ``df.attrs["units"]`` instead, which keeps columns selectable by
            species name.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ImportError
            If pandas is not installed (it is an optional dependency; install
            with ``pip install aquakin[dataframe]``).
        """
        network = self.network
        columns = [(sp, self.C[:, j]) for j, sp in enumerate(network.species)]
        units = {sp: network.units_of(sp) for sp in network.species}
        name, index = self._table_index()
        return build_dataframe(
            index, columns, index_name=name, units=units,
            units_in_columns=units_in_columns,
        )

    def to_csv(self, path_or_buf=None, *, units_in_columns: bool = True, **kwargs):
        """Write the solution to CSV (delegates to :meth:`to_dataframe`).

        Parameters
        ----------
        path_or_buf : str or path or file-like, optional
            Destination passed to ``DataFrame.to_csv``. If ``None``, the CSV is
            returned as a string.
        units_in_columns : bool, optional
            Defaults to ``True`` here (unlike :meth:`to_dataframe`): a CSV
            cannot carry ``df.attrs``, so the units are embedded in the column
            headers by default so the written file is self-describing.
        **kwargs
            Forwarded to ``pandas.DataFrame.to_csv``.
        """
        return self.to_dataframe(units_in_columns=units_in_columns).to_csv(
            path_or_buf, **kwargs
        )


@runtime_checkable
class Reactor(Protocol):
    """Structural type for the solve-based reactors that ``sensitivity`` /
    ``fit`` / ``calibrate`` / ``profile_likelihood`` consume.

    Declares the contract those consumers actually rely on: the compiled
    ``network``, the five solver settings every reactor exposes (set uniformly
    by :func:`init_solver_settings` + :func:`resolve_state_atol`), and a
    ``solve(C0, params=None, ...)`` whose extra arguments are reactor-specific
    (a batch/biofilm reactor takes ``t_span`` / ``t_eval``; a PFR fixes its grid
    at construction; a particle reactor takes neither). ``CFDReactor`` is
    deliberately **not** a ``Reactor`` -- it exposes ``step()``, not ``solve()``.

    A reactor that also carries spatially-varying ``conditions`` (batch / PFR /
    biofilm) satisfies the narrower :class:`ConditionedReactor`; a particle
    reactor carries a ``track`` instead and does not.
    """

    network: CompiledNetwork
    rtol: float
    atol: "float | jnp.ndarray"
    adjoint: "diffrax.AbstractAdjoint | None"
    dtmax: "float | None"
    max_steps: int

    def solve(self, C0, params=None, *args, **kwargs):  # pragma: no cover
        ...


@runtime_checkable
class ConditionedReactor(Reactor, Protocol):
    """A :class:`Reactor` that exposes spatially-varying ``conditions``.

    The batch / PFR / biofilm reactors carry a :class:`SpatialConditions`, so a
    consumer that differentiates through condition fields (e.g.
    :func:`aquakin.sensitivity`) requires this narrower type; a particle reactor,
    which carries a ``track`` instead, is a :class:`Reactor` but not a
    ``ConditionedReactor``.
    """

    conditions: "SpatialConditions"


def _coerce_atol(atol, n_species: int):
    """Validate and normalise an ``atol`` argument.

    Returns either the original scalar or a ``(n_species,)`` JAX array.
    Raises ``ValueError`` if an array of the wrong shape is supplied.
    """
    arr = jnp.asarray(atol)
    if arr.ndim == 0:
        # A concrete scalar (the reactor-construction path) is returned as a
        # Python float, but a traced value is returned as the 0-d array: calling
        # ``float()`` on a tracer raises a concretization error, which would
        # otherwise prevent jitting a solve whose ``atol`` flows in under tracing.
        if isinstance(arr, jax.core.Tracer):
            return arr
        return float(arr)
    if arr.shape != (n_species,):
        raise ValueError(
            f"atol array must have shape ({n_species},), got {arr.shape}"
        )
    return arr


@contextlib.contextmanager
def friendly_solve_errors(max_steps, *, what: str = "solve"):
    """Re-raise two opaque solve-time failures as domain-level errors.

    Both failures surface from JAX/Diffrax/Equinox internals with messages a
    process engineer cannot map to a fix. Wrap the *execution* of a solve in
    this context manager -- the call to the jitted solve / ``diffeqsolve``, where
    each failure actually surfaces -- to re-raise it as a plain ``RuntimeError``
    naming the remedy, with the noisy traceback suppressed (``from None``). Any
    other exception propagates unchanged.

    1. **Step-budget exhaustion.** An adaptive solve that exhausts ``max_steps``
       raises a verbose ``EquinoxRuntimeError`` ("The maximum number of solver
       steps was reached") wrapped in JAX/Equinox debugging chatter
       (``EQX_ON_ERROR``, ``kidger.site``, ...). Re-raised naming the
       warm-start / loosen-rtol / raise-max_steps remedies.
    2. **Forward-mode AD through the reverse-only adjoint.** The default
       ``RecursiveCheckpointAdjoint`` registers a ``custom_vjp``, so a
       ``jax.jacfwd`` / ``jax.jvp`` through the solve fails with JAX's
       "can't apply forward-mode autodiff (jvp) to a custom_vjp function" --
       which never mentions the cure. Re-raised naming
       ``aquakin.forward_adjoint()``.

    Parameters
    ----------
    max_steps : int
        The step budget that was hit (quoted back in the step-budget message).
    what : str
        A short label for the failing solve (e.g. ``"plant solve"``), used in
        the message.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 -- re-interpret two specific failures
        msg = str(exc).lower()
        if "maximum number of solver steps" in msg:
            raise RuntimeError(
                f"The {what} hit its integrator step budget (max_steps={max_steps}) "
                "before completing. This is almost always a stiff transient, not a "
                "bug. Try, in order: (1) warm-start from a settled state -- for a "
                "plant, pass y0 from plant.run_to_steady_state (or a previous run); "
                "(2) loosen rtol (the default atol already auto-scales to the state "
                "magnitudes); or (3) raise max_steps. If none help, the model may be "
                "genuinely unstable at these parameters/inputs."
            ) from None
        if "forward-mode autodiff" in msg and "custom_vjp" in msg:
            raise RuntimeError(
                f"Forward-mode autodiff (jax.jacfwd / jax.jvp) cannot flow through "
                f"the {what}: the default adjoint (RecursiveCheckpointAdjoint) is "
                "reverse-mode only -- it registers a custom_vjp that rejects forward "
                "mode. Build the reactor with adjoint=aquakin.forward_adjoint() to "
                "use a forward-mode-capable solve, or take a reverse-mode gradient "
                "(jax.grad / jax.jacrev) instead. (aquakin.sensitivity and "
                "aquakin.dgsm accept ad_mode='forward' and set this adjoint for you.)"
            ) from None
        raise


def default_atol(scale_like, reference=None, *, atol_factor: float = 1e-6,
                 floor_frac: float = 1e-6):
    """Per-component absolute tolerance scaled off the states' operating magnitudes.

    The error test every adaptive solver uses weights each component by
    ``atol_i + rtol*|y_i|``; ``atol_i`` is the **noise floor** below which
    component ``i`` is treated as negligible. When components span very different
    scales (e.g. ADM1 from ~1e-13 to ~17, or an OH radical at ~1e-12 beside a
    bulk reactant at ~1e-4) a single scalar floor is wrong for most of them, so
    the floor is set **per component** -- the SUNDIALS "vector atol" guidance and
    Hairer & Wanner's rule of ``atol_i`` proportional to the typical magnitude of
    component ``i``.

    Returns ``atol_i = atol_factor * max(|scale_i|, |reference_i|, floor_frac*char)``,
    where ``char = max_j(typical_j)`` is the system's bulk magnitude (so a
    component whose typical value is ~0, e.g. a product not present initially,
    gets a small floor tied to the system scale rather than ``atol_i = 0``, which
    the solver literature explicitly warns against).

    Parameters
    ----------
    scale_like : array
        A representative state vector -- typically the initial condition ``C0`` /
        ``y0`` (the operating magnitudes).
    reference : array, optional
        A second magnitude source merged in via elementwise max -- typically the
        network's ``default_concentrations`` (the YAML reference values), so a
        component that starts at 0 but has a nonzero reference is still floored
        sensibly.
    atol_factor : float
        Fraction of each component's typical magnitude used as its noise floor.
    floor_frac : float
        Floor for near-zero components, as a fraction of the bulk scale ``char``.

    Returns
    -------
    jnp.ndarray
        Per-component absolute tolerance, same shape as ``scale_like``.
    """
    typ = jnp.abs(jnp.asarray(scale_like, dtype=float))
    if reference is not None:
        typ = jnp.maximum(typ, jnp.abs(jnp.asarray(reference, dtype=float)))
    char = jnp.max(typ)
    # When every magnitude is zero there is no scale to floor against, so the
    # near-zero floor floor_frac*char would itself be 0 -- returning atol_i = 0,
    # which the solver literature warns against (the invariant this floor exists
    # to uphold). Fall back to unit scale in that case. Identity for any input
    # with a nonzero magnitude (the common path), since char > 0 there.
    char = jnp.where(char > 0.0, char, 1.0)
    typ = jnp.maximum(typ, floor_frac * char)
    return atol_factor * typ


# Seconds per time unit, for converting a user-supplied ``time_unit`` to a
# network's native (rate-constant) time unit. The keys match
# ``CompiledNetwork.time_unit`` / ``utils.units._TIME_TOKENS``.
_TIME_UNIT_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}


def native_time_factor(network_time_unit, requested_unit) -> float:
    """Factor ``f`` such that ``t_native = f * t_requested``.

    Converts a time expressed in ``requested_unit`` into the network's native
    (rate-constant) time unit, so a caller can pass ``t_span`` / ``t_eval`` in a
    convenient unit (e.g. hours) and have the solve run with the network's
    unchanged rate constants. Returns ``1.0`` when ``requested_unit is None``
    (no conversion -- the default, native-unit path).

    Parameters
    ----------
    network_time_unit : str or None
        The network's native time unit (``CompiledNetwork.time_unit``).
    requested_unit : str or None
        The unit the caller's times are in. ``None`` means "native".

    Raises
    ------
    ValueError
        If ``requested_unit`` is unknown, or it is given but the network's own
        time unit could not be inferred (``None``) and so there is no native
        unit to convert to.
    """
    if requested_unit is None:
        return 1.0
    if requested_unit not in _TIME_UNIT_SECONDS:
        raise ValueError(
            f"Unknown time_unit {requested_unit!r}; expected one of "
            f"{sorted(_TIME_UNIT_SECONDS)}."
        )
    if network_time_unit is None:
        raise ValueError(
            f"Cannot convert times to time_unit={requested_unit!r}: this "
            "network's own time unit could not be inferred from its "
            "rate-constant units (network.time_unit is None), so there is no "
            "native unit to convert to. Pass t_span / t_eval already in the "
            "network's rate-constant time unit and omit time_unit."
        )
    # network_time_unit comes from CompiledNetwork.time_unit, which only ever
    # returns a known token, so it is guaranteed to be in the table.
    return _TIME_UNIT_SECONDS[requested_unit] / _TIME_UNIT_SECONDS[network_time_unit]


def to_native_time(network_time_unit, requested_unit, t_span, t_eval):
    """Scale ``(t_span, t_eval)`` from ``requested_unit`` into native time.

    Returns ``(t_span_native, t_eval_native, factor)`` where ``factor`` is the
    :func:`native_time_factor` (``t_native = factor * t_requested``); divide a
    native-unit result time by it to report back in ``requested_unit``. A
    ``factor`` of ``1.0`` (the ``requested_unit is None`` default, or a unit
    equal to the native one) leaves the inputs untouched.
    """
    factor = native_time_factor(network_time_unit, requested_unit)
    if factor == 1.0:
        return t_span, t_eval, factor
    if t_span is not None:
        t_span = (t_span[0] * factor, t_span[1] * factor)
    if t_eval is not None:
        t_eval = jnp.asarray(t_eval) * factor
    return t_span, t_eval, factor


def resolve_state_atol(network, atol):
    """Resolve the ``atol`` for a reactor whose state is one ``(n_species,)``
    concentration vector.

    ``atol=None`` -> the per-component :func:`default_atol` noise floor scaled off
    the network's reference concentrations (so a g/m³ ASM network and a mol/L
    ozone network each get sensible tolerances without hand-tuning, instead of a
    fixed scalar that is ~9 orders too tight for g/m³ states). An explicit scalar
    or ``(n_species,)`` array is validated and returned verbatim. Shared by every
    reactor with a single-concentration-vector state (Batch / PFR / Particle /
    CFD); the layered :class:`~aquakin.BiofilmReactor`, whose state spans several
    compartments, sets its own scalar ``atol`` instead.
    """
    return (
        default_atol(network.default_concentrations())
        if atol is None else _coerce_atol(atol, network.n_species)
    )


def init_solver_settings(reactor, network, *, rtol, adjoint, dtmax, max_steps):
    """Store the solver settings every reactor shares, on ``reactor``.

    Sets ``network``, ``rtol``, ``adjoint``, ``dtmax`` and ``max_steps`` -- the
    five settings common to every reactor constructor. ``atol`` is **not** set
    here because its resolution depends on the reactor's state shape (see
    :func:`resolve_state_atol` for the single-vector case); the caller sets
    ``reactor.atol`` itself.
    """
    reactor.network = network
    reactor.rtol = float(rtol)
    reactor.adjoint = adjoint
    reactor.dtmax = dtmax
    reactor.max_steps = int(max_steps)


def validate_C0_params(network, C0, params):
    """Raise ``ValueError`` if ``C0`` / ``params`` do not match the network.

    The shared shape check every single-vector-state reactor runs at the top of
    ``solve`` (``C0`` is ``(n_species,)``, ``params`` is ``(n_params,)``).
    """
    if C0.shape != (network.n_species,):
        raise ValueError(
            f"C0 has shape {C0.shape}, expected ({network.n_species},)"
        )
    if params.shape != (network.n_params,):
        raise ValueError(
            f"params has shape {params.shape}, expected ({network.n_params},)"
        )


def _run_diffeqsolve(
    rhs: Callable,
    *,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    args,
    saveat: diffrax.SaveAt,
    rtol: float,
    atol,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int = 100_000,
    dtmax: float | None = None,
    event: "diffrax.Event | None" = None,
):
    """Wrapper around the canonical Kvaerno5 + PIDController + adjoint setup.

    All reactors call this with their own ``rhs``. Adjusting the default
    solver, controller, or adjoint here changes behaviour for every reactor.

    ``dtmax`` caps the integrator step size. It is ``None`` (uncapped) by
    default, which is fastest for plain forward solves. For *reverse-mode*
    differentiation of a stiff network it must be set. An L-stable solver may
    take steps far larger than the fastest reaction timescale and simply damp
    the unresolved fast modes in the primal (which stays accurate). The two AD
    modes then diverge: **forward mode** (``jax.jvp`` / ``jax.jacfwd``) stays
    finite at any step, losing only accuracy when the fast modes are
    unresolved; **reverse mode** (``jax.grad``, the discrete adjoint) returns
    **non-finite** values above a step-size threshold, an overflow in the
    backward accumulation governed by the per-step stiffness ``gamma*dt*||J||``
    (not by operator conditioning). Capping ``dtmax`` to a small multiple of
    the fastest reaction timescale bounds that product; the resulting reverse
    gradient is finite and matches both forward mode and finite differences.
    This is reverse-mode-specific and independent of the adjoint flavour. See
    the "Differentiating stiff networks" discussion in CLAUDE.md.
    """
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
        event=event,
    )


def solve_chemistry(
    network: CompiledNetwork,
    C0: jnp.ndarray,
    params: jnp.ndarray,
    *,
    cond_fn: Callable[[jnp.ndarray], Mapping[str, jnp.ndarray]],
    saveat: diffrax.SaveAt,
    t0,
    t1,
    rtol: float,
    atol,
    adjoint: diffrax.AbstractAdjoint | None = None,
    dtmax: float | None = None,
    max_steps: int = 100_000,
    rate_scale=None,
):
    """The canonical chemistry sub-solve shared by every reactor.

    Hoists the (parameter-dependent) stoichiometry out of the per-step RHS ---
    so dynamic coefficients are evaluated once per solve, not per step --- builds
    the right-hand side ``dC/dt = rate_scale * dCdt(C, params, cond_fn(t))`` and
    runs the Kvaerno5 + ``PIDController`` solve via :func:`_run_diffeqsolve`.

    The reactors differ only in three traced-time choices, passed in here:

    - ``cond_fn(t)`` returns the condition arrays at independent-variable value
      ``t``. A batch / CFD cell passes a constant dict; a PFR or particle track
      passes an interpolation of its spatially / temporally varying fields.
    - ``rate_scale`` (``None`` = identity) multiplies the rate, e.g. ``1/velocity``
      for the steady-state PFR whose independent variable is axial position.
    - ``saveat`` / ``t0`` / ``t1`` select the output points and the span.

    Returns the diffrax ``Solution``; callers read ``sol.ts`` / ``sol.ys`` (or
    ``sol.ys[-1]`` for a single-endpoint step).
    """
    stoich = network.compute_stoich(params)

    if rate_scale is None:
        def rhs(t, C, args):
            return network.dCdt(C, args, cond_fn(t), 0, stoich=stoich)
    else:
        def rhs(t, C, args):
            return network.dCdt(C, args, cond_fn(t), 0, stoich=stoich) * rate_scale

    return _run_diffeqsolve(
        rhs,
        t0=t0,
        t1=t1,
        y0=C0,
        args=params,
        saveat=saveat,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        dtmax=dtmax,
        max_steps=max_steps,
    )


def _interp_fields_to_scalar(
    t: jnp.ndarray,
    t_grid: jnp.ndarray,
    fields: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Linearly interpolate every field to a single-location array at ``t``.

    Returns a fresh dict whose values are shape-``(1,)`` arrays. Reactors
    pair this with ``loc_idx=0`` to satisfy the canonical rate-callable
    signature.
    """
    return {
        name: jnp.asarray([jnp.interp(t, t_grid, arr)])
        for name, arr in fields.items()
    }
