"""Colored-Jacobian sparsity management for :class:`Plant`.

A plant flowsheet has a large, block-sparse Jacobian: each unit couples to its
own state and to the inlet concentrations of the streams feeding it, and the
recycle loops add a few off-diagonal blocks. Coloring that sparsity lets the
implicit integrator, the ``stable_adjoint`` backward pass, and the PTC
steady-state solve each build the Jacobian in one Jacobian-vector product per
*color* rather than one per column (BSM2: ~45 vs 167), while reconstructing the
identical matrix on the pattern's support -- exact, since the colored matrix
equals the dense one when the pattern is a superset.

:class:`ColoredJacobianManager` owns that subsystem for one :class:`Plant`:

- :meth:`_structural_plant_pattern` -- assemble the plant's structural Jacobian
  sparsity from each unit's emitted couplings (the complete superset a numeric
  probe alone would miss for a saturated warm start).
- :meth:`forward_root_finder` / :meth:`jacobian_solver` -- the forward
  implicit-solve colored root finder, and a diffrax solver configured with it.
- :meth:`adjoint_jacobian_builder` -- the ``stable_adjoint`` backward ``df/dy``
  colored builder.
- :meth:`steady_jacobian_builder` -- the PTC steady residual ``dF/dy`` colored
  builder.

The three builders share the same scaffold -- probe the sparsity at a
representative state, union it with the structural pattern, build a colored root
finder, and **guard** the result against the dense Jacobian (falling back to
dense on a mismatch) -- factored into :meth:`_build_and_guard` /
:meth:`_colored_from_probe`. Each builder caches its result and is reused across
later solves; :meth:`reset` clears the caches when the state layout changes
(e.g. a temperature block is appended).

The manager holds a back-reference to its :class:`Plant` for the state layout,
unit topology, translator couplings, RHS and recycle maps. It is a behaviour
collaborator, not a value object: a ``Plant`` creates exactly one
(``plant._colored``) and drives it from its solve / adjoint / steady paths. The
size-based *decision* of whether to use the colored backward
(:attr:`Plant._COLORED_BACKWARD_MIN_STATES`) stays on the plant's solve routing;
this manager only builds the machinery when asked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from aquakin.integrate._common import default_atol

if TYPE_CHECKING:
    from aquakin.plant.plant import Plant


def _output_state_pattern(plant, unit, cp) -> np.ndarray:
    """Which of ``unit``'s state entries its outlet concentrations read.

    A boolean ``(n_species, state_size)`` block: entry ``(i, j)`` is true when the
    outlet concentration of species ``i`` (on any port) can depend on state entry
    ``j``. Taken from the unit's :class:`~aquakin.plant.coupling.CouplingPattern`
    ``output_pattern`` when it declares one. Otherwise: the identity for a unit
    whose state is its concentration vector; the identity on the concentration
    head plus every trailing scalar (a volume, a filtered flow) for a unit whose
    state leads with its concentrations; and every state entry for any other
    layout -- a dense block is a safe superset.
    """
    n = unit.model.n_species
    m = int(unit.state_size)
    out = getattr(cp, "output_pattern", None) if cp is not None else None
    if out is not None:
        return np.asarray(out, dtype=bool)
    if m == n:
        return np.eye(n, dtype=bool)
    if plant._is_concentration_unit(unit):
        return np.concatenate([np.eye(n, dtype=bool), np.ones((n, m - n), dtype=bool)], axis=1)
    return np.ones((n, m), dtype=bool)


class ColoredJacobianManager:
    """Owns the colored-Jacobian sparsity subsystem (structural pattern + the
    forward / adjoint / steady builder caches) for one :class:`Plant`. See the
    module docstring."""

    def __init__(self, plant: Plant):
        self.plant = plant
        # Colored-Jacobian root finder (built once, concretely, on the first
        # colored_jacobian=True solve): (root_finder, n_colors, ok). ``ok`` False
        # means the setup guard found the colored Jacobian disagreed with the
        # dense one at the start state, so the solve falls back to the dense path.
        self._root_finder: tuple | None = None
        # Colored-Jacobian builder for the stable_adjoint BACKWARD pass (built
        # once, concretely): (builder_or_None, n_colors, ok, n_states, rf).
        # Distinct from _root_finder (the forward root finder) -- this colors the
        # AUGMENTED (time-carrying, n+1) primal rhs the discrete adjoint
        # differentiates. ``ok`` False => the guard found a colored/dense mismatch
        # at the start state, so the backward falls back to dense jacfwd.
        self._adjoint_builder: tuple | None = None
        # Colored-Jacobian builder for the PTC STEADY-STATE iteration (built once,
        # concretely): (builder_or_None, n_colors, ok). Colors the autonomous
        # steady residual Jacobian dF/dy. The PTC operating-point neighbourhood is
        # narrow, so the start-state pattern stays valid throughout (unlike a wide
        # dynamic run).
        self._steady_builder: tuple | None = None

    def reset(self) -> None:
        """Drop the three builder caches.

        The colored-Jacobian builders cache seed matrices / sparsity patterns
        sized for the concrete state. When the state length changes (e.g. a
        temperature block is appended by ``use_temperature_model``), a
        previously-built builder is stale and would be dimension-mismatched on
        the next colored solve, so it must be rebuilt.
        """
        self._root_finder = None
        self._adjoint_builder = None
        self._steady_builder = None

    # --- structural sparsity -------------------------------------------------

    def _structural_plant_pattern(self, coupling_mask=None) -> np.ndarray:
        """Assemble the plant's structural Jacobian sparsity from each unit's
        emitted couplings, for the colored pattern (issue #388).

        The numerical probe at the start state captures the plant's **linear,
        always-on** couplings but drops every **nonlinear** coupling that is
        saturated at the warm-start operating point and only switches on once a
        dynamic influent drives the plant off it -- reaction kinetics (Monod / pH
        switches), the Takacs settling velocity, and the ASM<->ADM interface
        branches. Those are the stiff couplings, so a stale pattern wrecks the
        chord-Newton convergence (a ~6x slowdown).

        Each unit emits its own structural sparsity (:class:`CouplingAware`,
        :meth:`coupling_pattern`): a ``self`` block (d rhs / d own state) and an
        ``inlet`` block (d rhs / d inlet concentration). This assembler places the
        ``self`` blocks on the diagonal and composes each ``inlet`` block with the
        species coupling of the stream feeding it -- identity for a same-model
        feed, the translator's emitted ``coupling_pattern()`` for an ASM<->ADM
        feed -- to form the off-diagonal blocks, restricted to the unit pairs the
        probe shows actually coupled (the recycle's real reach). The result is a
        structural superset that cannot go stale for any influent. ``coupling_mask``
        is the probe pattern (used only to restrict off-diagonal placement to real
        couplings, keeping the coloring tight).
        """

        from aquakin.plant.translators import translator_coupling_pattern

        plant = self.plant
        N = plant._total_state_size
        P = np.zeros((N, N), dtype=bool)

        # Each unit emits its structural Jacobian sparsity (CouplingAware): a
        # self block (d rhs / d own state) and an inlet block (d rhs / d inlet
        # concentration). Reactors derive self from the rate AST (saturated Monod
        # terms are numerically invisible to a probe), the Takacs settler by AD
        # over diverse solids profiles, stateless units are empty.
        cps = {}
        for name, unit in plant.units.items():
            fn = getattr(unit, "coupling_pattern", None)
            if fn is not None:
                cps[name] = (unit, fn())

        # Diagonal: each unit's self_pattern, placed on its own state block.
        for name, (unit, cp) in cps.items():
            sp = np.asarray(cp.self_pattern, dtype=bool)
            if sp.size == 0:
                continue
            off, _ = plant._state_layout[name]
            k = sp.shape[0]
            P[off : off + k, off : off + k] |= sp

        # Off-diagonal: J[A][B] = inlet_pattern_A composed with the species
        # coupling of the stream feeding A from B, composed with B's output
        # pattern (which of B's states its outlet concentrations read). For a
        # same-model feed the species coupling is the identity; for a cross-model
        # feed (ASM<->ADM) it is the translator's emitted pattern. B ranges over
        # every stateful unit that emits a stream -- a reactor (outlet == state),
        # a tank with trailing scalars, the layered settler (outlet == a layer) --
        # so no source's columns are left to the probe alone. Placement is
        # restricted to the unit pairs the probe shows actually coupled (the
        # recycle's real reach), keeping the pattern tight. Couplings that run
        # through the plant's flows rather than a concentration path (a
        # TSS-dependent thickener flow changing every downstream dilution) are not
        # modelled here; the probe supplies them.
        tcache: dict[tuple, np.ndarray] = {}

        def _translator(src_net, tgt_net):
            key = (id(src_net), id(tgt_net))
            if key not in tcache:
                tcache[key] = None
                for conn in plant.connections:
                    T = getattr(conn, "translator", None)
                    if T is not None and T.source_model is src_net and T.target_model is tgt_net:
                        tcache[key] = np.asarray(translator_coupling_pattern(T), dtype=bool)
                        break
            return tcache[key]

        sources = {
            nm: (u, _output_state_pattern(plant, u, cps.get(nm, (None, None))[1]))
            for nm, u in plant.units.items()
            if getattr(u, "model", None) is not None
            and getattr(u, "state_size", 0) > 0
            and list(getattr(u, "output_ports", []))
        }
        for aname, (aunit, cp) in cps.items():
            if cp.inlet_pattern is None:
                continue
            ip = np.asarray(cp.inlet_pattern, dtype=bool)  # (a_rows, a_nsp)
            a_off, _ = plant._state_layout[aname]
            a_rows = ip.shape[0]
            a_net = getattr(aunit, "model", None)
            for bname, (bunit, out_pat) in sources.items():
                if bname == aname:
                    continue
                b_net = bunit.model
                b_off, b_cols = plant._state_layout[bname]
                if (
                    coupling_mask is not None
                    and not coupling_mask[a_off : a_off + a_rows, b_off : b_off + b_cols].any()
                ):
                    continue  # not really coupled
                if a_net is b_net:
                    coupling = np.eye(a_net.n_species, dtype=bool)
                else:
                    coupling = _translator(b_net, a_net)  # (a_nsp, b_nsp)
                    if coupling is None:
                        continue
                species_block = (ip.astype(np.int8) @ coupling.astype(np.int8)) > 0
                block = (species_block.astype(np.int8) @ out_pat.astype(np.int8)) > 0
                P[a_off : a_off + a_rows, b_off : b_off + b_cols] |= block
        return P

    # --- shared build scaffold ----------------------------------------------

    def _build_and_guard(self, rhs_y, y0, *, rtol, atol, probe_pattern, extra_pattern, context):
        """Build a colored root finder for ``rhs_y`` at ``y0`` and guard it
        against the dense Jacobian. Returns ``(rf, n_colors, ok)``.

        The single point where all three builders (forward / adjoint / steady)
        share the ``build_colored_root_finder`` -> ``colored_jacobian_guard``
        tail: the colored matrix equals the dense one on a superset pattern, so
        ``ok`` False (a guard mismatch) is the signal to fall back to dense.
        """
        from aquakin.integrate.colored_jacobian import (
            build_colored_root_finder,
            colored_jacobian_guard,
        )

        rf, n_colors = build_colored_root_finder(
            rhs_y,
            y0,
            rtol=rtol,
            atol=atol,
            probe_pattern=probe_pattern,
            extra_pattern=extra_pattern,
        )
        ok = colored_jacobian_guard(rhs_y, y0, rf, context=context)
        return rf, n_colors, ok

    def _colored_from_probe(self, rhs_y, y0, *, rtol, atol, context):
        """Probe ``rhs_y``'s sparsity at ``y0``, union it with the structural
        pattern, and build + guard the colored root finder. Returns
        ``(rf, n_colors, ok)``.

        The scaffold the forward and steady builders share: both derive the
        pattern from a live probe of the *same* rhs they will solve, then union
        it with the complete per-component structural superset.
        """
        from aquakin.integrate.colored_jacobian import jacobian_sparsity_pattern

        probe = jacobian_sparsity_pattern(rhs_y, y0) > 0
        structural = self._structural_plant_pattern(coupling_mask=probe)
        return self._build_and_guard(
            rhs_y,
            y0,
            rtol=rtol,
            atol=atol,
            probe_pattern=probe,
            extra_pattern=structural,
            context=context,
        )

    # --- forward implicit solve ---------------------------------------------

    def forward_root_finder(self, t0, y0, params, rtol, atol):
        """The guarded colored root finder for the forward implicit solve, or
        ``None`` when the colored path is unavailable.

        Built **once per plant** (concretely): derive the plant Jacobian sparsity
        pattern, color it, pack a :class:`ColoredVeryChord`, and **guard** it by
        comparing the colored and dense Jacobians at the start state -- returning
        ``None`` (with a warning) if they disagree. Reused on later solves.
        Skipped under tracing if not yet built (the probe/guard need concrete
        arrays), so a first traced solve gets ``None``. The colored matrix equals
        the dense one when the pattern is a superset; a pattern miss only costs
        solver steps, never accuracy.

        Returns
        -------
        ColoredVeryChord or None
            Its ``seed_matrix`` / ``color_of`` / ``pattern`` materialize the
            per-step Jacobian
            (:func:`~aquakin.integrate.colored_jacobian.state_jacobian`).
        """
        plant = self.plant
        if self._root_finder is None:
            if isinstance(params, jax.core.Tracer) or isinstance(y0, jax.core.Tracer):
                return None  # can't build under trace
            t0a = jnp.asarray(float(t0))
            states0 = plant._split_state(y0)
            rmap = plant._recycle._maybe_recycle_map(t0a, states0, params)
            fmap = plant._recycle._maybe_flow_map(t0a, states0, params)

            def rhs_y(y):
                return plant._rhs(t0a, y, params, recycle_map=rmap, flow_map=fmap)

            atol_arr = jnp.asarray(atol)
            self._root_finder = self._colored_from_probe(
                rhs_y,
                y0,
                rtol=10.0 * rtol,
                atol=10.0 * atol_arr,
                context="colored_jacobian=True",
            )

        rf, n_colors, ok = self._root_finder
        return rf if ok else None

    def jacobian_solver(self, solver, t0, y0, params, rtol, atol):
        """Return ``solver`` reconfigured to use the colored-AD Jacobian root
        finder of :meth:`forward_root_finder`, or ``solver`` unchanged when the
        colored path is unavailable (the dense path)."""
        rf = self.forward_root_finder(t0, y0, params, rtol, atol)
        if rf is None:
            return solver
        # Build through the single-source-of-truth helper: it injects the colored
        # root finder into the user-supplied solver, or into the canonical Kvaerno5
        # when none is given -- so the colored path constructs no solver of its own.
        from aquakin.integrate._common import build_implicit_solver

        return build_implicit_solver(rtol, atol, solver=solver, colored_root_finder=rf)

    # --- stable_adjoint backward pass ---------------------------------------

    # Perturbed states the stable-adjoint pattern is checked at, besides the
    # default state and (when concrete) the solve's own start state: each
    # component scaled by 10**U[-spread, spread], from a fixed seed.
    _ADJOINT_GUARD_N_PERTURBED = 3
    _ADJOINT_GUARD_SPREAD = 0.5

    def adjoint_jacobian_builder(self, t0, rtol, atol, y0=None):
        """Derive (once) the sparsity-colored ``df/dy`` Jacobian builder for the
        stable_adjoint backward pass, from the plant's **own default operating
        point**.

        The cap-free reverse-mode backward is dominated (~82% on BSM2) by per-step
        dense ``df/dy`` Jacobian builds -- one Jacobian-vector product per state.
        For a large block-sparse plant, coloring the Jacobian computes it in one
        JVP per *color* (BSM2: ~45 vs 167), cutting that cost while staying exact
        (the colored matrix equals the dense one when the pattern is a superset).

        The colored pattern is a **structural** property of the plant -- the unit
        coupling graph plus the recycle block structure -- so it is built from the
        plant's **default** state/params, NOT the solve's ``y0``/``params``. That
        decouples it from the solve routing and the AD trace: it builds lazily on
        first use even under a gradient trace, because it touches only concrete
        plant defaults (wrapped in ``ensure_compile_time_eval`` so the probe is not
        staged into the gradient). Whether to *use* the result is a separate,
        size-based decision (:attr:`Plant._COLORED_BACKWARD_MIN_STATES`), so no
        build-time benchmark is needed -- which is what removes the old dependency
        on a concrete ``stable_adjoint`` solve to trigger and time the build.

        Caches ``(builder_or_None, n_colors, ok, n_states, rf)`` on
        ``_adjoint_builder``: the augmented colored builder, the color count,
        whether the default-state guard passed, the plant state count (for the
        size gate), and the ``ColoredVeryChord``.

        Built for the **augmented** (time-carrying, ``n+1``) primal right-hand side
        the discrete adjoint actually differentiates. A missed coupling corrupts
        the gradient rather than slowing the solve, so the guard **fails closed**:
        besides the colored/dense comparison at the default state, the pattern
        must cover every dense-Jacobian entry above a per-row threshold
        (:func:`~aquakin.integrate.colored_jacobian.pattern_coverage_misses`) at
        the default state, a few perturbed copies of it and -- when ``y0`` is a
        concrete start state -- the solve's own operating point, which is also
        probed. Any miss falls back to the dense backward (with a warning).
        """
        from aquakin.integrate.colored_jacobian import (
            jacobian_sparsity_pattern,
            materialize_colored_jacobian,
            pattern_coverage_misses,
        )
        from aquakin.integrate.discrete_adjoint import _autonomize

        plant = self.plant
        if self._adjoint_builder is not None:
            return self._adjoint_builder[0]
        y_solve = None if y0 is None or isinstance(y0, jax.core.Tracer) else np.asarray(y0)

        # Build from the plant's OWN default operating point -- a concrete,
        # always-available representative state -- so the pattern (a structural
        # property) is independent of the solve's inputs and computable even when
        # the only caller is a gradient (whose y0/params are tracers).
        # ``ensure_compile_time_eval`` keeps the concrete probe from being staged
        # into the enclosing gradient computation.
        with jax.ensure_compile_time_eval():
            y0 = plant.initial_state()
            params = plant.default_parameters()
            t0f = float(t0)
            rmap = plant._recycle._maybe_recycle_map(
                jnp.asarray(t0f), plant._split_state(y0), params
            )
            fmap = plant._recycle._maybe_flow_map(jnp.asarray(t0f), plant._split_state(y0), params)

            def primal(t, y, p):
                return plant._rhs(t, y, p, recycle_map=rmap, flow_map=fmap)

            # The discrete adjoint integrates the autonomized [y; tau] state, so
            # the backward Jacobians -- and hence the coloring -- are of that.
            rhs_aug, y0_aug = _autonomize(primal, y0, t0f)

            def rhs_aug_y(ya):
                return rhs_aug(0.0, ya, params)

            # The backward feeds J directly into ``I - dt.gamma.J^T`` and the
            # transposed solve, so a missed coupling **silently corrupts the
            # gradient**. So the pattern must be a *complete* structural superset.
            # Use the per-component structural pattern (each unit's equation-derived
            # ``coupling_pattern()``, assembled by :meth:`_structural_plant_pattern`)
            # for the plant ``df/dy`` block, embedded in the augmented ``[y; tau]``
            # layout's top-left block; the probe at the default state supplies the
            # recycle block structure (connectivity-based, so any representative
            # state reveals it -- which is why the default state serves as well as
            # the real y0).
            n = int(y0.shape[0])

            def primal_y(y):
                return primal(jnp.asarray(t0f), y, params)

            plain_probe = jacobian_sparsity_pattern(primal_y, y0) > 0
            if y_solve is not None:
                plain_probe |= jacobian_sparsity_pattern(primal_y, jnp.asarray(y_solve)) > 0
            structural = self._structural_plant_pattern(coupling_mask=plain_probe)
            aug_extra = np.zeros((n + 1, n + 1), dtype=bool)
            aug_extra[:n, :n] = plain_probe | structural

            # rtol/atol only set the (unused) chord tolerances on the returned
            # object; the seed matrix / coloring / pattern is all we use. A scalar
            # atol avoids augmenting the per-component vector for the probe. A
            # traced atol (derived from a traced y0 under a gradient) has no
            # concrete value, so fall back to the default operating point's.
            atol_ref = default_atol(y0, y0) if isinstance(atol, jax.core.Tracer) else atol
            atol_s = float(jnp.max(jnp.asarray(atol_ref)))
            rf, n_colors, ok = self._build_and_guard(
                rhs_aug_y,
                y0_aug,
                rtol=10.0 * rtol,
                atol=10.0 * atol_s,
                probe_pattern=None,
                extra_pattern=aug_extra,
                context="colored_jacobian (stable_adjoint)",
            )
            if ok:
                ok = self._adjoint_pattern_covers(
                    rhs_aug_y, np.asarray(y0_aug), y_solve, rf.pattern, pattern_coverage_misses
                )

        # ``builder`` is applied later in the backward to the real (traced) f/y;
        # it closes over the concrete ``rf`` (seed/coloring/pattern) built above.
        def builder(f, y):
            return materialize_colored_jacobian(rf, f, y)

        # ``rf`` is cached too: it is reused as the *forward* solve's root finder
        # so the adjoint's forward pass can color its per-step Jacobian as well.
        self._adjoint_builder = (
            builder if ok else None,
            n_colors,
            ok,
            n,
            rf if ok else None,
        )
        return self._adjoint_builder[0]

    def _adjoint_pattern_covers(self, rhs_aug_y, y0_aug, y_solve, pattern, misses_fn) -> bool:
        """Whether the stable-adjoint pattern covers the dense Jacobian at the
        default state, its perturbed copies and the concrete solve start state.

        Warns and returns ``False`` on any miss, so the caller falls back to the
        dense backward.
        """
        import warnings

        tau = y0_aug[-1:]
        rng = np.random.default_rng(0)
        states = [y0_aug]
        for _ in range(self._ADJOINT_GUARD_N_PERTURBED):
            scale = 10.0 ** rng.uniform(
                -self._ADJOINT_GUARD_SPREAD, self._ADJOINT_GUARD_SPREAD, y0_aug.size - 1
            )
            states.append(np.concatenate([y0_aug[:-1] * scale, tau]))
        if y_solve is not None:
            states.append(np.concatenate([y_solve, tau]))
        count, worst = misses_fn(rhs_aug_y, states, pattern)
        if count:
            warnings.warn(
                f"colored_jacobian (stable_adjoint): the sparsity pattern misses "
                f"{count} Jacobian entries (largest {worst:.2e} of its row) at the "
                f"default or start state; using the dense backward Jacobian. This "
                f"indicates the structural pattern missed a coupling -- please "
                f"report it.",
                RuntimeWarning,
                stacklevel=4,
            )
            return False
        return True

    # --- PTC steady-state solve ---------------------------------------------

    def steady_jacobian_builder(self, rhs, y0, theta, *, tol):
        """Derive (once, concretely) a colored-AD materializer for the PTC steady
        residual Jacobian ``dF/dy``, or ``None`` to fall back to dense ``jacfwd``.

        The PTC iteration (:func:`aquakin.plant.steady.ptc_forward`) forms the
        full plant Jacobian ``dF/dy`` -- the same block-sparse object the
        integrator's implicit-stage and the stable_adjoint backward color -- once
        per Newton step (~tens of times for BSM2). Coloring builds it in one
        Jacobian-vector product per *color* (BSM2: ~45 vs 167 columns) instead of
        ``n``, while reconstructing the identical matrix on the pattern's support.

        Unlike the dynamic solve, this is well suited to coloring: PTC marches to
        a single operating point in a narrow neighbourhood, so the warm-start
        probe usually suffices on its own (the dynamic run's wide load excursion
        is what makes start-state-missed couplings a problem). The builder unions
        it with the complete per-component structural pattern regardless, matching
        the forward and stable_adjoint builders. Built and **guarded** against the
        dense Jacobian at ``y0`` once (falling back to dense with a warning on a
        mismatch), then cached and reused -- a parameter/design sweep keeps the
        same structural pattern.

        Returns a builder ``(F, y) -> dF/dy``, or ``None`` (dense) when called
        under a trace (the probe needs concrete arrays) or when the guard fails.
        """
        from aquakin.integrate.colored_jacobian import materialize_colored_jacobian

        if self._steady_builder is None:
            if any(
                isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves((theta, y0))
            ):
                return None  # can't build under trace; fall back

            def rhs_y(y):
                return rhs(y, theta)

            tol_s = float(jnp.max(jnp.asarray(tol)))
            # Use the per-component structural pattern (each unit's equation-derived
            # coupling_pattern, assembled by _structural_plant_pattern) unioned with
            # the warm-start probe, matching the forward and stable_adjoint builders.
            # PTC stays in a narrow neighbourhood so the start-state probe is usually
            # enough on its own, but the structural superset is complete regardless
            # of how far the warm start sits from the steady state.
            rf, n_colors, ok = self._colored_from_probe(
                rhs_y,
                y0,
                rtol=tol_s,
                atol=tol_s,
                context="colored_jacobian=True (steady_state)",
            )

            def builder(f, y):
                return materialize_colored_jacobian(rf, f, y)

            self._steady_builder = (builder if ok else None, n_colors, ok)

        return self._steady_builder[0]
