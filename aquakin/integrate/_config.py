"""The solver and autodiff configuration objects, the AD-mode helpers, the
finite-gradient check, and the structural ``Reactor`` protocols.

Not part of the public API; ``aquakin.integrate._common`` re-exports every name.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import diffrax
import jax.numpy as jnp
import numpy as np

from aquakin.core.model import CompiledModel

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.core.conditions import SpatialConditions


# --- Public solver / autodiff configuration objects --------------------------


@dataclass(frozen=True)
class IntegratorConfig:
    """The integrator / step-size configuration for a solve.

    A single frozen value object centralising the per-step solver settings that
    used to be scattered as loose keyword arguments on ``Plant.solve`` and the
    reactor constructors. Passed as ``integrator=IntegratorConfig(...)``; the
    default flips the public solve stack to the fast configuration (Kvaerno3 +
    a capped step-growth factor + an auto colored Jacobian).

    ``rtol`` / ``atol`` are deliberately **not** here -- they stay separate
    arguments of ``Plant.solve`` / the reactor constructors (a tolerance is the
    accuracy contract, not the integrator machinery).

    Parameters
    ----------
    order : int
        ESDIRK order: ``3`` -> ``Kvaerno3`` (the new fast default, fewer stages),
        ``5`` -> ``Kvaerno5`` (the robust higher-order method). Ignored when
        ``solver`` is given.
    factormax : float, optional
        Cap on the PID controller's per-step growth factor (diffrax default 10).
        A smaller cap (default 3) damps the overshoot-then-reject oscillation on a
        stiff forced system. ``None`` keeps the diffrax default.
    colored_jacobian : {"auto", True, False}
        Sparse (colored-AD) materialisation of the per-step implicit Jacobian.
        See ``Plant.solve`` for the full semantics. Plant-only -- reactors ignore
        it.
    dtmax : float, optional
        Maximum integrator step size (``None`` = uncapped). Set it for a
        reverse-mode gradient *through* a stiff solve (the ``through_solve``
        differentiation method); the cap-free ``stable`` method does not need it.
    max_steps : int
        Maximum number of internal solver steps.
    solver : diffrax.AbstractSolver, optional
        An explicit solver object that overrides ``order`` and is honoured
        verbatim (the one escape hatch). ``None`` builds the canonical solver of
        ``order``.
    """

    order: int = 3
    factormax: float | None = 3.0
    colored_jacobian: bool | str = "auto"
    dtmax: float | None = None
    max_steps: int = 100_000
    solver: diffrax.AbstractSolver | None = None


@dataclass(frozen=True)
class DifferentiationConfig:
    """How a gradient / sensitivity flows through a solve.

    A single frozen value object replacing the old AD selectors (``adjoint=``,
    ``gradient=``, ``ad_mode=``, ``check_finite=``, ``stable_adjoint_max_steps=``).
    Passed as ``diff=DifferentiationConfig(...)``.

    Parameters
    ----------
    mode : {"reverse", "forward"}
        Autodiff direction. ``"reverse"`` (default) is the calibration-gradient
        direction (``jax.grad`` / a discrete adjoint); ``"forward"`` is the
        sensitivity-screen direction (``jax.jacfwd`` / a variational solve).
    method : {"stable", "through_solve"}
        How the adjoint / sensitivity is *formed* for the chosen ``mode``.

        - ``mode="reverse", method="stable"`` (the default) -- a plant uses the
          cap-free hand-written discrete adjoint (the old
          ``gradient="stable_adjoint"`` / ``"auto"`` routing: a concrete forward
          solve still takes the fast cached path, a differentiated solve takes the
          discrete adjoint). For a **reactor** it depends on the entry point. A
          bare ``reactor.solve`` under ``jax.grad`` has no stable adjoint: it
          always differentiates through the solve with
          ``RecursiveCheckpointAdjoint``, whatever ``method`` is, so a stiff model
          needs a ``dtmax`` cap there. :func:`~aquakin.calibrate` with
          ``method="stable"`` does not call ``reactor.solve``: it fits a batch
          reactor through the cap-free ESDIRK discrete adjoint
          (:func:`~aquakin.esdirk_adjoint_solve`) on the reactor's model, with the
          reactor's integrator order.
        - ``mode="reverse", method="through_solve"`` -- differentiate *through*
          the diffrax solve (``RecursiveCheckpointAdjoint``; the old
          ``gradient="jax_adjoint"``). Needs a ``dtmax`` cap for a stiff model.
        - ``mode="forward", method="stable"`` -- the augmented ``[y; S]``
          variational solve (cap-free, exact). Routed through
          ``solve_sensitivity`` / ``augmented_forward_sensitivity``.
        - ``mode="forward", method="through_solve"`` -- ``jax.jvp`` / ``jacfwd``
          via a forward-capable adjoint (``DirectAdjoint`` / ``ForwardMode``).
    check_finite : bool
        Raise a friendly error if a freshly computed gradient/Jacobian is
        non-finite, instead of returning silent ``NaN`` values.
    adjoint_max_steps : int
        Buffer length for the discrete-adjoint backward scan (the old
        ``stable_adjoint_max_steps``); set it to a tight upper bound on the
        forward step count.
    adjoint_low_memory : bool
        Low-memory mode for the ``method="stable"`` discrete adjoint. ``False``
        (default) saves each forward step's dense-output stage increments, so the
        backward reconstructs the stages cheaply -- but the saved buffer is
        ``~n_stages`` × the trajectory. ``True`` stores only the step states and
        **recomputes** each step's stages in the backward pass (a per-step Newton
        scan), so the dense buffer is never allocated -- trading compute for
        memory when that buffer is the binding constraint on a long, large-state
        plant solve. The recompute matches the saved-stage gradient (it is the
        same well-conditioned per-stage solve), so it is a pure
        memory/compute trade. Reverse ``method="stable"`` only -- ignored for a
        ``through_solve`` or forward solve.
    """

    mode: str = "reverse"
    method: str = "stable"
    check_finite: bool = True
    adjoint_max_steps: int = 100_000
    adjoint_low_memory: bool = False

    # -- Decoding the (mode, method) vocabulary -----------------------------
    #
    # The public (mode, method) meaning is owned here rather than re-decoded in
    # each entry point (calibrate / sensitivity / dgsm / Plant.solve), so a change
    # to the config's semantics -- or a new (mode, method) pairing -- lands in one
    # place instead of being mirrored across every consumer.

    def validated(self) -> DifferentiationConfig:
        """Validate the ``mode`` / ``method`` vocabulary; return ``self``.

        Raises ``ValueError`` for an out-of-range ``mode`` or ``method``. This is
        the vocabulary check only -- whether a *particular* consumer serves a
        given valid pairing (e.g. ``Plant.solve`` rejecting the ``forward`` +
        ``stable`` augmented solve) is that consumer's own guard.
        """
        if self.mode not in ("reverse", "forward"):
            raise ValueError(f"diff.mode must be 'reverse' or 'forward'; got {self.mode!r}.")
        if self.method not in ("stable", "through_solve"):
            raise ValueError(
                f"diff.method must be 'stable' or 'through_solve'; got {self.method!r}."
            )
        return self

    def forms_jacfwd(self) -> bool:
        """Whether the AD direction is forward (``jax.jacfwd`` / a variational
        solve) rather than reverse (``jax.grad`` / a discrete adjoint)."""
        return self.mode == "forward"

    def gradient_backend(self) -> str:
        """The reverse-adjoint backend this config selects.

        ``"stable_adjoint"`` for the cap-free hand-written discrete adjoint
        (``method="stable"``), or ``"jax_adjoint"`` for differentiating *through*
        the diffrax solve (``method="through_solve"``). This is the canonical
        ``method`` -> backend meaning the calibration entry points consume; the
        plant *solve* routes ``method="stable"`` through its own ``"auto"``
        dispatch instead (see :meth:`Plant._resolve_diff_config`).
        """
        return "stable_adjoint" if self.method == "stable" else "jax_adjoint"

    def reactor_adjoint(self):
        """The diffrax adjoint object a reactor / plant solve should carry.

        No diffrax adjoint object is the hand-written ("stable") reverse adjoint,
        so reverse mode always carries the default ``RecursiveCheckpointAdjoint``
        (signalled by ``None``); the discrete adjoint is invoked explicitly by the
        callers that use it (``Plant.solve`` under reverse-mode AD and
        :func:`~aquakin.calibrate` with ``method="stable"``). ``method`` matters
        only for *forward* mode: ``through_solve`` -> a forward-capable
        ``DirectAdjoint`` (:func:`forward_adjoint`); ``stable`` -> the augmented
        ``solve_sensitivity`` backend, invoked explicitly (not via this adjoint),
        so ``None`` is still returned.
        """
        if self.mode == "forward" and self.method == "through_solve":
            return forward_adjoint()
        return None


# --- The differentiation settings that error messages advise ------------------
#
# Every remedy that tells a user how to change the AD direction or adjoint names
# one of these, so the advice cannot drift from the real keyword arguments.

#: The forward-mode setting of an entry point that builds the forward-capable
#: adjoint itself (``sensitivity``, ``dgsm``).
FORWARD_DIFF = 'diff=aquakin.DifferentiationConfig(mode="forward")'
#: The forward-mode setting of a reactor (or ``Plant.solve``) or of ``calibrate``:
#: forward-mode autodiff *through* the solve. ``method="stable"`` is the reverse
#: discrete adjoint, so a bare ``mode="forward"`` does not give this.
FORWARD_THROUGH_SOLVE_DIFF = (
    'diff=aquakin.DifferentiationConfig(mode="forward", method="through_solve")'
)
#: The cap-free reverse-mode setting (the hand-written discrete adjoint).
STABLE_REVERSE_DIFF = 'diff=aquakin.DifferentiationConfig(method="stable")'
#: How to make a solve the user builds forward-mode differentiable.
FORWARD_REACTOR_REMEDY = (
    f"build the reactor with {FORWARD_THROUGH_SOLVE_DIFF} (or pass that diff= to "
    "Plant.solve), which gives the solve a forward-capable adjoint"
)


@dataclass(frozen=True)
class SolverSettings:
    """The tolerances and solver configuration of a reactor, in one value.

    The settings the other reactors take as four loose keyword arguments
    (``rtol``, ``atol``, ``integrator``, ``diff``), bundled for a reactor whose
    constructor already carries its geometry and transport parameters
    (:class:`~aquakin.DispersedFlowReactor`, :class:`~aquakin.TanksInSeries`).

    Parameters
    ----------
    rtol : float
        Relative tolerance for the ODE solver.
    atol : float or array-like, optional
        Absolute tolerance: a scalar, a ``(n_species,)`` array applied in every
        cell, or one value per state. ``None`` (default) uses the per-species
        noise floor scaled off the model's reference concentrations.
    integrator : IntegratorConfig
        Integrator / step-size configuration.
    diff : DifferentiationConfig
        Autodiff configuration.
    """

    rtol: float = 1e-6
    atol: object = None
    integrator: IntegratorConfig = IntegratorConfig()
    diff: DifferentiationConfig = DifferentiationConfig()


# --- AD-mode helpers (hide the diffrax adjoint plumbing) ---------------------


def forward_adjoint() -> diffrax.AbstractAdjoint:
    """Return the diffrax adjoint that supports forward-mode autodiff.

    A thin, dependency-free alias for ``diffrax.DirectAdjoint()``, the adjoint a
    reactor carries when built with
    ``diff=aquakin.DifferentiationConfig(mode="forward", method="through_solve")``
    (see :meth:`DifferentiationConfig.reactor_adjoint`). The default
    ``RecursiveCheckpointAdjoint`` registers a ``custom_vjp`` that rejects
    forward mode. Reactors take the ``diff=`` config, not an adjoint object; use
    this alias only where a raw diffrax adjoint is wanted (a hand-written
    ``diffrax.diffeqsolve``, or :func:`with_adjoint`).
    """
    return diffrax.DirectAdjoint()


def with_adjoint(reactor, adjoint):
    """Return a shallow copy of ``reactor`` with its adjoint strategy replaced.

    Reactors are stateless after construction and read ``self.adjoint`` at solve
    time, so a shallow copy with a swapped ``adjoint`` is a valid forward-/
    reverse-capable variant of the same reactor. Used by ``calibrate`` /
    ``sensitivity`` to build the right adjoint internally from the
    ``diff.mode`` of their :class:`DifferentiationConfig`, so ``diffrax`` never
    appears in user code.

    The clone gets fresh, empty per-instance compile caches (every ``*_cache``
    dict attribute, e.g. ``_jit_cache`` / ``_sens_jit_cache``). A shallow copy
    would otherwise share those dicts with the original, so a solve compiled
    with the old adjoint could be reused by the clone (and vice versa).
    """
    clone = copy.copy(reactor)
    clone.adjoint = adjoint
    for name, value in list(vars(clone).items()):
        if name.endswith("_cache") and isinstance(value, dict):
            setattr(clone, name, {})
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
    through a stiff model's :meth:`solve` -- a user's own loss + optimizer,
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
                "reactor with a dtmax cap (integrator=aquakin.IntegratorConfig("
                "dtmax=...), a small multiple of the fastest reaction timescale), "
                "or differentiate in forward mode (jax.jacfwd): "
                f"{FORWARD_REACTOR_REMEDY}. aquakin.calibrate (cap-free with "
                f"{STABLE_REVERSE_DIFF}) and aquakin.sensitivity (with "
                f"{FORWARD_DIFF}) handle this for you."
            )
        else:
            remedy = (
                f"The reactor already caps dtmax={self.dtmax}; if the gradient is "
                "still non-finite, tighten dtmax further, or check the model, "
                "data and parameter ranges."
            )
        check_finite_gradient(grad_value, what=what, remedy=remedy)
        return grad_value


@runtime_checkable
class Reactor(Protocol):
    """Structural type for the solve-based reactors that ``sensitivity`` /
    ``fit`` / ``calibrate`` / ``profile_likelihood`` consume.

    Declares the contract those consumers actually rely on: the compiled
    ``model``, the five solver settings every reactor exposes (set uniformly
    by :func:`init_solver_settings` + :func:`resolve_state_atol`), and a
    ``solve(C0, params=None, ...)`` whose extra arguments are reactor-specific
    (a batch/biofilm reactor takes ``t_span`` / ``t_eval``; a PFR fixes its grid
    at construction; a particle reactor takes neither). ``CFDReactor`` is
    deliberately **not** a ``Reactor`` -- it exposes ``step()``, not ``solve()``.

    A reactor that also carries spatially-varying ``conditions`` (batch / PFR /
    biofilm) satisfies the narrower :class:`ConditionedReactor`; a particle
    reactor carries a ``track`` instead and does not.
    """

    model: CompiledModel
    rtol: float
    atol: float | jnp.ndarray
    adjoint: diffrax.AbstractAdjoint | None
    dtmax: float | None
    max_steps: int
    order: int
    factormax: float | None
    solver: diffrax.AbstractSolver | None

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

    conditions: SpatialConditions
