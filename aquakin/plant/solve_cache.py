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
storage, not plant-dependent logic. :func:`structure_changed` is the one place
a structure change (a unit, connection or influent added) clears this cache
together with the plant's other structure-derived state.

**Sharing between plants built from one spec.** A plant rebuilt from the same
flowsheet spec compiles the same programs, and a study run or the editor's
worker rebuilds the plant on every run. So a plant built by
:meth:`Plant.from_spec` is given an identity (:meth:`SolveCache.share_as`: its
class and the fingerprint of the spec *as built*, the loaded models' and
influent tables' own digests included), and its forward and steady-state
solves are also kept in a process-wide store under that identity, where every
other plant of the same identity finds them. Equal identities mean the same
units, settings, wiring, options, model content and influent data, hence the
same compiled program. The identity lasts only while the plant is exactly as
built: any :meth:`SolveCache.invalidate` (a condition or structure mutator)
drops it and withdraws what the plant published, because a published program
closes over the plant that built it and a retrace would read its new state. A
plant reads a shared program without copying it, so a withdrawn one is never
reused. The store keeps the :data:`SHARED_LIMIT` most recently used programs:
each holds its plant and its executables alive.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

#: The most compiled programs the process-wide store keeps; the least recently
#: used is dropped first. A study run uses up to about three (an initial steady
#: state, the solve, a sensitivity), and each holds its plant and executables
#: (some 80 MB for a BSM1 steady-state solve), so the bound is kept small.
SHARED_LIMIT = 4

# ``(identity, store, key) -> (value, *keep_alive)``, most recently used last.
_SHARED: OrderedDict = OrderedDict()
# Plants of one process may solve on several threads (the editor's tests run
# studies on threads).
_SHARED_LOCK = threading.Lock()

# The stores a plant shares; the continuation and arclength kernels close over
# values taken from one call (a known solution's scale) that their keys omit, so
# they stay with the plant that built them.
_SHAREABLE = ("jit", "steady_jit")


def clear_shared() -> None:
    """Drop every compiled program in the process-wide store.

    The plants keep their own; a plant built afterwards compiles afresh.
    """
    with _SHARED_LOCK:
        _SHARED.clear()


def shared_size() -> int:
    """The number of compiled programs in the process-wide store."""
    with _SHARED_LOCK:
        return len(_SHARED)


class SolveCache:
    """Owns a :class:`Plant`'s compiled-solve and kernel memoization, with one
    :meth:`invalidate`. See the module docstring."""

    def __init__(self):
        # Jitted forward solves (and the ``solve_sensitivity`` [y; S] solves,
        # under a "solve_sensitivity" key), keyed by call signature + solver
        # settings. The plant RHS closes over the (static) unit graph, so once
        # compiled the same solve is reused across repeated solves -- without it
        # every solve rebuilds the RHS closure and Diffrax recompiles the whole
        # stiff plant (~tens of seconds) each call.
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
        # The identity under which this plant's solves are shared, or ``None``.
        self.shared_as = None
        # The process-wide keys this plant published, withdrawn on invalidation.
        self._published: set = set()

    def share_as(self, identity) -> None:
        """Share this plant's compiled solves with every plant of ``identity``.

        Parameters
        ----------
        identity : hashable
            Equal only for plants that compile identical programs (see the
            module docstring): :meth:`Plant.from_spec` passes the plant class
            and the fingerprint of the spec as built.
        """
        self.shared_as = identity

    def get(self, store: str, key):
        """The compiled program for ``key`` in ``store``, or ``None``.

        Looks in this plant's own ``store`` (``"jit"`` or ``"steady_jit"``),
        then, while the plant is shared, in the process-wide store.

        Parameters
        ----------
        store : str
            ``"jit"`` or ``"steady_jit"``.
        key : hashable or None
            The settings key; ``None`` (an uncacheable call) finds nothing.

        Returns
        -------
        object or None
            The cached value, or ``None`` on a miss.
        """
        if key is None:
            return None
        own = getattr(self, store).get(key)
        if own is not None or self.shared_as is None:
            return own
        full = (self.shared_as, store, key)
        with _SHARED_LOCK:
            entry = _SHARED.get(full)
            if entry is None:
                return None
            _SHARED.move_to_end(full)
        return entry[0]

    def put(self, store: str, key, value, *keep_alive) -> None:
        """Keep ``value`` for ``key`` in ``store``, and publish it while shared.

        Parameters
        ----------
        store : str
            ``"jit"`` or ``"steady_jit"``.
        key : hashable
            The settings key.
        value : object
            The compiled program.
        *keep_alive : object
            Objects the key names by ``id`` (a custom adjoint), held with the
            published entry so their ``id`` cannot be reused while it lives.
        """
        if store not in _SHAREABLE:
            raise ValueError(f"no shared store {store!r}; expected one of {_SHAREABLE}")
        getattr(self, store)[key] = value
        if self.shared_as is None:
            return
        full = (self.shared_as, store, key)
        with _SHARED_LOCK:
            _SHARED[full] = (value, *keep_alive)
            _SHARED.move_to_end(full)
            while len(_SHARED) > SHARED_LIMIT:
                _SHARED.popitem(last=False)
        self._published.add(full)

    def invalidate(self, *, keep_shared: bool = False) -> None:
        """Drop every compiled solve / kernel.

        Call whenever a change would make a previously-compiled solve stale -- a
        reactor condition value (the compiled RHS bakes it in) or the flat state
        length. Cheap: the artefacts recompile lazily on the next solve. This is
        the single invalidation point so a mutator cannot clear one cache and
        forget another.

        Parameters
        ----------
        keep_shared : bool, optional
            Keep sharing (see the module docstring). Only for work every plant
            of the identity does the same way before it compiles: the plant's
            clock, resolved from its models, units and influents at the first
            solve. A change to the plant never passes it: the plant is then no
            longer the one its identity describes, so it stops sharing and
            withdraws what it published.
        """
        self.jit.clear()
        self.steady_jit.clear()
        self.continuation_kernels.clear()
        self.arclength_kernels.clear()
        if keep_shared:
            return
        self.shared_as = None
        with _SHARED_LOCK:
            for full in self._published:
                _SHARED.pop(full, None)
        self._published.clear()


def structure_changed(plant) -> None:
    """Drop everything a :class:`Plant` derived from its flowsheet structure.

    Bound onto ``Plant`` as ``_structure_changed`` and called by every method
    that adds a unit, a connection or a wired influent, so a plant edited after
    a solve is solved, read and differentiated as if it had been built that
    way. It clears the compiled solves and kernels (:meth:`SolveCache.invalidate`),
    the colored-Jacobian builders, the recycle map-constancy verdicts, the
    per-unit parameter slices, the one-time recycle diagnostics and the
    controller-materialization guards; the state layout, topology and clock
    are rebuilt by ``Plant._build_state_layout`` on every solve anyway. A new
    memo derived from the structure is cleared here, not by each mutator.

    Parameters
    ----------
    plant : Plant
        The plant whose structure changed.
    """
    plant._solve_cache.invalidate()
    plant._colored.reset()
    plant._recycle.reset()
    plant._params_unit_cache = None
    plant._flow_affinity_checked = False
    plant._recycle_convergence_checked = False
    plant._aeration_materialized = False
    plant._dosing_materialized = False
