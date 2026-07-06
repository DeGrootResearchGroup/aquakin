"""Compiled-solve and kernel memoization for :class:`Plant`.

A ``Plant`` compiles its stiff monolithic solve once and reuses it across
repeated solves -- a parameter sweep / Monte Carlo that builds the plant once and
solves it many times, or a warm-started steady-state-then-dynamic run. That
memoization -- the jitted forward solves, the PTC steady-state solves, and the
natural-parameter / pseudo-arclength continuation kernels -- is solver-run state,
not part of the flowsheet definition.

:class:`SolveCache` owns it in one place so a single :meth:`invalidate` drops
*every* compiled artefact, and a condition / topology mutator never has to
remember which named dict to clear. That is the bug this collaborator closes: a
compiled solve bakes in the reactor conditions, so a mutator that clears some
caches but not others (e.g. the forward + steady solves but not the continuation
kernels) leaves a solve that silently reuses the old conditions -- a stale-result
bug with no error.

The colored-Jacobian builders are *structural* sparsity caches (keyed to the
state layout, not the compiled solve), so they live in
:class:`~aquakin.plant.colored.ColoredJacobianManager` and are reset only on a
state-layout change. :meth:`invalidate` here covers the compiled solves, which go
stale on **any** condition-value *or* layout change -- so a value-only change
(``set_temperature``) invalidates the solves while leaving the still-valid
colored pattern in place, and a layout change (``set_temperature_model``)
invalidates both.

It is a plain value object: a ``Plant`` holds exactly one (``plant._solve_cache``)
and reads/populates its dicts from the solve paths. Unlike ``RecycleResolver`` /
``ColoredJacobianManager`` it needs no back-reference to the plant -- it owns
storage, not plant-dependent logic.
"""

from __future__ import annotations


class SolveCache:
    """Owns a :class:`Plant`'s compiled-solve and kernel memoization, with one
    :meth:`invalidate`. See the module docstring."""

    def __init__(self):
        # Jitted forward solves, keyed by call signature + solver settings. The
        # plant RHS closes over the (static) unit graph, so once compiled the same
        # solve is reused across repeated solves -- without it every solve rebuilds
        # the RHS closure and Diffrax recompiles the whole stiff plant (~tens of
        # seconds) each call.
        self.jit: dict = {}
        # Compiled PTC steady-state forward solves, keyed by settings. The eager
        # ``jax.lax.while_loop`` in ``ptc_forward`` re-traces and recompiles on
        # every call (~12-17 s for BSM2), so a persisted jitted solver lets a
        # repeated concrete ``steady_state`` (a sweep / multistart / figure regen)
        # pay that compile once and reuse it (~40 ms run thereafter).
        self.steady_jit: dict = {}
        # Jitted natural-parameter continuation kernels (the PTC-corrector
        # predictor-corrector), keyed by PTC settings, so a screen over many
        # targets from one known point compiles once.
        self.continuation_kernels: dict = {}
        # Jitted pseudo-arclength continuation kernels, keyed by settings.
        self.arclength_kernels: dict = {}

    def invalidate(self) -> None:
        """Drop every compiled solve / kernel.

        Call whenever a change would make a previously-compiled solve stale -- a
        reactor condition value (the compiled RHS bakes it in) or the flat state
        length. Cheap: the artefacts recompile lazily on the next solve. This is
        the single invalidation point so a mutator cannot clear one cache and
        forget another.
        """
        self.jit.clear()
        self.steady_jit.clear()
        self.continuation_kernels.clear()
        self.arclength_kernels.clear()
