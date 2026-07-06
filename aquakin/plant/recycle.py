"""Recycle-flow and recycle-concentration resolution for :class:`Plant`.

A plant flowsheet has back-edges (RAS / internal recycle / reject loops): a
unit's output stream depends on a downstream unit's output. :class:`RecycleResolver`
owns the machinery that closes those loops **exactly and gain-independently** per
RHS evaluation, plus the per-solve caches that let it skip redundant work:

- **Flows** (:meth:`_resolve_flows`) -- probe the affine recycle-flow map ``A``
  and solve ``(I - A)x = b`` for the back-edge flows.
- **Concentrations** (:meth:`_resolve_recycle_concentrations`) -- one forward
  output sweep at fixed flows is an affine map ``c -> M c + d``; probe ``M``/``d``
  and solve ``(I - M)c = d`` (species-decoupled, grouped by model, with an
  optional decoupled temperature channel), falling back to an adaptive
  Gauss-Seidel refine (:meth:`_adaptive_recycle_refine`) for a genuinely
  non-affine in-cycle unit.

The resolver holds a back-reference to its :class:`Plant` for the topology, state
layout, signal bus and output sweep, and owns the tri-state map-constant caches
(``_recycle_map_constant`` / ``_recycle_T_map_constant`` / ``_flow_map_constant``)
that decide whether the affine maps are state-invariant and can be precomputed
once per solve. It is a behaviour collaborator, not a value object: a ``Plant``
creates exactly one (``plant._recycle``) and drives it from its RHS and solve
paths.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from aquakin.plant.streams import Stream, make_scalars
from aquakin.plant.units import FlowContext

if TYPE_CHECKING:
    from aquakin.plant.plant import Plant


class RecycleResolver:
    """Owns recycle-flow / recycle-concentration resolution and its per-solve
    map-constant caches for one :class:`Plant`. See the module docstring."""

    def __init__(self, plant: "Plant"):
        self.plant = plant
        # Tri-state: None until checked, then True/False -- whether the recycle
        # *concentration* map M is state-independent (so it can be precomputed
        # once per solve and reused, skipping the per-RHS per-species probes).
        # Determined by :meth:`_check_recycle_map_constant`.
        self._recycle_map_constant: Optional[bool] = None
        # Same, for the *temperature* map MT. Constant in heat-balance mode (the
        # reactor T-state breaks the loop coupling) and trivially when no T is
        # carried; NOT constant in algebraic mode (T passes through reactors, so
        # MT rides on the concentration-dependent recycle flows) -- then MT is
        # re-probed every RHS (cheap scalar) while M stays cached.
        self._recycle_T_map_constant: Optional[bool] = None
        self._flow_map_constant: Optional[bool] = None

    def _seed_recycle_streams(
        self,
        resolved_flows: dict[tuple[Optional[str], str], jnp.ndarray],
    ) -> dict[tuple[str, str], Stream]:
        """Pre-seed recycle back-edges with their resolved flow + initial conc.

        A recycle edge (downstream -> upstream) must be readable before its
        source unit has been visited in the sweep. We seed it with the
        exactly-resolved flow (from :meth:`_resolve_flows`) and a seed
        concentration -- the edge's explicit ``initial_value`` when given, else
        the zero-flow auto-seed from :meth:`_finalize_topology`; the
        concentration is then corrected within a few sweep passes, as recycle
        concentrations are read directly from the source unit's state.
        """
        seeded: dict[tuple[str, str], Stream] = {}
        for conn in self.plant._recycle_conns:
            key = (conn.from_unit, conn.from_port)
            iv = (
                conn.initial_value
                if conn.initial_value is not None
                else self.plant._recycle_seeds[key]
            )
            q = resolved_flows[key]
            seeded[key] = Stream(Q=q, C=iv.C, model=iv.model, scalars=dict(iv.scalars))
        return seeded

    def _resolve_recycle_concentrations(
        self,
        t: jnp.ndarray,
        states: dict[str, jnp.ndarray],
        params_full: jnp.ndarray,
        resolved_flows: dict[tuple[Optional[str], str], jnp.ndarray],
        signals: Optional[dict] = None,
        recycle_map: Optional[list] = None,
    ) -> dict[tuple[str, str], Stream]:
        """Pre-solve the recycle back-edge concentrations exactly, as a seed.

        One forward output sweep, with the recycle back-edge concentrations held
        at trial values, is an **affine** map ``c -> M·c + d`` on those
        concentrations (mixers / splitters / clarifiers are linear in
        concentration at the already-resolved flows; stateful units output their
        state, a constant). So instead of iterating it (gain-limited), we recover
        ``M`` and ``d`` by probing -- one pass at ``c = 0`` (``d``), one pass per
        recycle edge set to a unit concentration (the columns of ``M``) -- and
        solve ``(I − M) c = d`` directly. This is exact and **gain-independent**,
        the same closed-form trick :meth:`_resolve_flows` uses for the flows.

        The map is **species-decoupled** (a mixer/splitter/clarifier maps each
        species independently; the only species-coupling unit, an ASM↔ADM
        translator, is fed by a digester *state* and so never enters the cyclic
        map), so one probe per edge yields its whole column across all species,
        and the solve is ``n_species`` independent ``n_edge × n_edge`` systems --
        ``n_recycle_edges + 1`` cheap forward passes total, like the flow probe.
        Edges of different models do not couple (the translator that would
        couple them is broken by the digester state), so the solve is grouped by
        model. Temperature, when carried, is one more decoupled scalar channel.

        Returned as the recycle seed for :meth:`_sweep_outputs`. For the linear
        case (every shipped topology) it **is** the exact fixed point, so the
        subsequent mop-up passes leave it unchanged; the iterative sweep then only
        has to refine the residual of a genuinely *non*-affine in-cycle unit (a
        translator inside a pure-stateless loop -- not constructible from the
        shipped units), with :meth:`_check_recycle_convergence` as the backstop.

        The *concentration* map ``M`` is determined purely by the resolved recycle
        flows and the topology (the mixer/splitter/clarifier ratios) and -- for the
        BSM plants, whose recycle flows are fixed pumps -- it is invariant to the
        state and time; only ``d = forward(0)`` varies. So when a precomputed
        ``recycle_map`` (the per-group concentration ``M``) is supplied, this skips
        the ``n_recycle_edges`` per-species concentration probes and computes only
        ``d`` (one sweep), collapsing the dominant per-RHS cost. The concentration
        result is bit-identical to probing. The *temperature* channel is the
        exception: its mixing weights ride on the concentration-dependent reject
        flows, so ``MT`` is **not** state-invariant; it is always re-probed here
        (a cheap scalar T-only sweep -- the per-species part is CSE-shared with
        ``d``), keeping the temperature exact.
        """
        keys = self.plant._recycle_keys
        if not keys:
            return {}
        ctx = self._recycle_context(t, states, params_full, resolved_flows, signals)
        seed_net, group_lists, forward, zeroC, zeroT, forward_full = ctx

        dC, dT = forward(zeroC, zeroT)  # constant part d
        resolve_T = all(dT[k] is not None for k in keys)

        # Concentration map M: cached (state-invariant) or probed. The cached
        # ``recycle_map`` also carries MT when it is state-invariant (heat-balance
        # / no-T modes); ``MT_cached is None`` signals per-RHS re-probing (the
        # algebraic mode, where MT rides on the concentration-dependent flows).
        if recycle_map is None:
            colC = self._probe_recycle_C(forward, keys, zeroC, zeroT, dC)
            M_groups = self._assemble_recycle_M(group_lists, colC)
            MT_cached = None
        else:
            M_groups, MT_cached = recycle_map
        # Temperature map MT: cached when state-invariant, else re-probed (cheap
        # scalar -- the per-species part, at c=0, is CSE-shared with ``d``).
        if not resolve_T:
            MT_groups = [None] * len(group_lists)
        elif MT_cached is not None:
            MT_groups = MT_cached
        else:
            colT = self._probe_recycle_T(forward, keys, zeroC, zeroT, dT)
            MT_groups = self._assemble_recycle_MT(group_lists, colT)

        # Solve (I - M) c = d per group, with d fresh and M fresh-or-cached.
        solved_C: dict = {}
        solved_T: dict = {}
        for gkeys, M, MT in zip(group_lists, M_groups, MT_groups):
            m = len(gkeys)
            eye = jnp.eye(m)
            d = jnp.stack([dC[gkeys[j]] for j in range(m)], axis=1)  # (nsp, m_j)
            cs = jnp.linalg.solve(eye[None] - M, d[..., None])[..., 0]  # (nsp, m_j)
            for j, kj in enumerate(gkeys):
                solved_C[kj] = cs[:, j]
            if MT is not None:
                dT_g = jnp.stack([dT[gkeys[j]] for j in range(m)])
                cT_g = jnp.linalg.solve(eye - MT, dT_g)
                for j, kj in enumerate(gkeys):
                    solved_T[kj] = cT_g[j]
            else:
                for kj in gkeys:
                    solved_T[kj] = None

        # Opt-in adaptive refinement. The affine solve above is the exact fixed
        # point only for a *linear* topology (mixers/splitters/clarifiers linear
        # in concentration at the resolved flows). When a recycle-loop flow is
        # concentration-dependent (a thickener/dewatering ``%TSS`` underflow on a
        # reject loop), the true map ``c -> forward(c)`` is nonlinear and the
        # affine ``c`` is off by that nonlinear residual. The fixed ``recycle_passes``
        # mop-up removes it in ``log(tol)/log(rho)`` passes (``rho`` = the loop's
        # spectral radius); calibrated to BSM's ~0.0066, ample at 3 passes, but
        # topology-dependent. With ``recycle_tol`` set, iterate the nonlinear map
        # to tolerance instead (AD-safe via :meth:`_adaptive_recycle_refine`), so
        # convergence is guaranteed for any ``rho < 1``.
        if self.plant.recycle_tol is not None:
            seed_Q = {k: resolved_flows[k] for k in keys}
            solved_Q, solved_C, solved_T = self._adaptive_recycle_refine(
                forward_full, keys, seed_Q, solved_C, solved_T, resolve_T
            )
            return {
                k: Stream(
                    Q=solved_Q[k],
                    C=solved_C[k],
                    model=seed_net[k],
                    scalars=make_scalars(T=solved_T[k]),
                )
                for k in keys
            }

        return {
            k: Stream(
                Q=resolved_flows[k],
                C=solved_C[k],
                model=seed_net[k],
                scalars=make_scalars(T=solved_T[k]),
            )
            for k in keys
        }

    def _adaptive_recycle_refine(self, forward_full, keys, seed_Q, seed_C, seed_T, resolve_T):
        """Iterate the nonlinear recycle map to ``recycle_tol``, AD-safe.

        The recycle back-edge *streams* -- flow ``Q``, concentration ``C`` and
        (when carried) temperature ``T`` -- are the fixed point ``x = G(x)`` of
        one forward output sweep (``forward_full``), read back on the recycle
        edges. ``Q`` is part of the variable because a concentration-dependent
        in-loop flow (a thickener/dewatering ``%TSS`` underflow on a reject loop)
        is recomputed from the seeded streams each sweep, so the true fixed point
        couples ``Q`` and ``C``; iterating ``C`` alone solves the wrong problem.
        The affine pre-solve (:meth:`_resolve_recycle_concentrations`) plus the
        resolved flows give the exact *linear*-topology fixed point as a warm
        start; this closes the residual of the nonlinear flow<->concentration
        coupling.

        Mirrors the charge-balance pH solver (``core/ph_solver.py``): the forward
        solve is an adaptive :func:`jax.lax.while_loop` that stops once the
        relative step falls below ``recycle_tol`` (capped at ``recycle_max_passes``
        for the worst case), wrapped in :func:`jax.lax.custom_root` so the
        sensitivity is the exact implicit-function-theorem tangent -- AD (forward
        and reverse) is O(1) in the iteration count rather than differentiating
        through every sweep. The map is contractive (spectral radius ``rho < 1``),
        so both the forward fixed-point iteration ``x <- G(x)`` and the tangent's
        Neumann inversion converge geometrically.

        Parameters
        ----------
        forward_full : callable
            ``forward_full(q, c, T) -> (C_dict, T_dict, Q_dict)`` -- one sweep,
            read back on the recycle edges (from :meth:`_recycle_context`).
        keys : list
            The recycle back-edge keys.
        seed_Q, seed_C, seed_T : dict
            The warm start: per-edge resolved flows ``Q``, affine-solve ``C``,
            and ``T`` scalars (``seed_T`` entries are ``None`` when no temperature
            is carried).
        resolve_T : bool
            Whether a temperature channel is part of the fixed point.

        Returns
        -------
        (dict, dict, dict)
            The refined per-edge ``Q``, ``C`` and ``T`` (``T`` passed through
            unchanged when ``resolve_T`` is False).
        """
        tol = self.plant.recycle_tol
        max_passes = self.plant.recycle_max_passes

        # The fixed-point variable is (Q, C) per edge, plus T when carried;
        # T=None channels are kept out of the differentiated pytree and threaded
        # through ``forward_full`` separately.
        none_T = dict.fromkeys(keys)

        if resolve_T:

            def G(x):
                Q, C, T = x
                Cn, Tn, Qn = forward_full(Q, C, T)
                return (Qn, Cn, Tn)

            x0 = (seed_Q, seed_C, seed_T)
        else:

            def G(x):
                Q, C = x
                Cn, _, Qn = forward_full(Q, C, none_T)
                return (Qn, Cn)

            x0 = (seed_Q, seed_C)

        def F(x):
            gx = G(x)
            return jax.tree_util.tree_map(lambda a, b: a - b, x, gx)

        def _relerr(x, xn):
            # Per-leaf relative step (each channel -- a Q ~ 1e3, a C ~ 1e0 -- has
            # its own scale), then the worst across leaves.
            rel = jnp.array(0.0)
            for a, b in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(xn)):
                step = jnp.max(jnp.abs(b - a))
                scale = jnp.max(jnp.abs(a)) + 1e-9
                rel = jnp.maximum(rel, step / scale)
            return rel

        def solve_root(f, init):
            def body(carry):
                x, _err, i = carry
                # G(x) = x - f(x): one nonlinear sweep, read back recycle edges.
                fx = f(x)
                xn = jax.tree_util.tree_map(lambda a, b: a - b, x, fx)
                return xn, _relerr(x, xn), i + 1

            def cond(carry):
                _x, err, i = carry
                return (err > tol) & (i < max_passes)

            xf, _, _ = jax.lax.while_loop(cond, body, (init, jnp.inf, jnp.array(0)))
            return xf

        def tangent_solve(g, y):
            # g is the linearisation z -> z - dG.z of F at the root -- a small
            # linear operator on the recycle-edge variable (a few edges x ~tens of
            # channels). Solve g(z) = y by materialising its dense matrix (one
            # jacobian of the linear map, constant) and a direct dense solve: the
            # exact implicit-function-theorem inverse, and the vector generalisation
            # of the pH solver's scalar ``y / g(1)``. ``jnp.linalg.solve`` is
            # cleanly transposable, so custom_root composes with the outer autodiff
            # (reverse and forward) without differentiating any iteration.
            y_flat, unravel = ravel_pytree(y)

            def g_flat(z_flat):
                return ravel_pytree(g(unravel(z_flat)))[0]

            J = jax.jacobian(g_flat)(jnp.zeros_like(y_flat))
            z_flat = jnp.linalg.solve(J, y_flat)
            return unravel(z_flat)

        xr = jax.lax.custom_root(F, x0, solve_root, tangent_solve)
        if resolve_T:
            Qr, Cr, Tr = xr
            return Qr, Cr, Tr
        Qr, Cr = xr
        return Qr, Cr, seed_T

    def _recycle_context(self, t, states, params_full, resolved_flows, signals):
        """Shared setup for the recycle affine solve.

        Returns ``(seed_net, group_lists, forward, zeroC, zeroT, forward_full)``:
        the per-edge seed models, the recycle edges grouped by model (no
        cross-model coupling), the one-pass ``forward(c, T) -> (C_out, T_out)``
        sweep closure (Q held at ``resolved_flows`` -- the affine probe), the zero
        seeds, and ``forward_full(q, c, T) -> (C, T, Q)`` (Q varying -- the
        adaptive solver's true fixed-point map). Used by the live solve and
        :meth:`_compute_recycle_map`.
        """
        keys = self.plant._recycle_keys
        seed_net: dict = {}
        for conn in self.plant._recycle_conns:
            key = (conn.from_unit, conn.from_port)
            iv = (
                conn.initial_value
                if conn.initial_value is not None
                else self.plant._recycle_seeds[key]
            )
            seed_net[key] = iv.model
        nsp = {k: seed_net[k].n_species for k in keys}
        # Temperature propagates only when an influent carries it (the mixer
        # T-gate needs every inlet to have T); otherwise the whole plant is
        # T=None regardless of the nominal recycle-seed T. So resolve a T channel
        # only in that case -- a number around the loop ignites the gate.
        # influents are InfluentSeries (their own T trajectory field), not Streams.
        carries_T = any(s.T is not None for s in self.plant.influents.values())
        t_seed = jnp.zeros(()) if carries_T else None

        # The influent streams are independent of the probe trial values, so
        # interpolate them once rather than on every one of the ``n_edges + 1``
        # forward probe passes.
        influent = {(None, pn): s.at(t) for pn, s in self.plant.influents.items()}

        def forward(c_by_key, T_by_key):
            seeded = {
                k: Stream(
                    Q=resolved_flows[k],
                    C=c_by_key[k],
                    model=seed_net[k],
                    scalars=make_scalars(T=T_by_key[k]),
                )
                for k in keys
            }
            out = self.plant._sweep_outputs(
                t, states, influent, seeded, params_full, passes=1, signals=signals
            )
            return ({k: out[k].C for k in keys}, {k: out[k].scalars.get("T") for k in keys})

        # Q-varying one-pass map for the adaptive solver: the recycle back-edge
        # Q is itself a fixed-point variable, because a concentration-dependent
        # in-loop flow (a thickener/dewatering ``%TSS`` underflow on a reject
        # loop) is recomputed in ``compute_outputs`` from the seeded streams. The
        # affine probe above holds Q fixed at ``resolved_flows`` (the linear part);
        # this closure lets it vary so :meth:`_adaptive_recycle_refine` iterates
        # the true (Q, C, T) fixed point. Returns (C, T, Q) read back on the edges.
        def forward_full(q_by_key, c_by_key, T_by_key):
            seeded = {
                k: Stream(
                    Q=q_by_key[k],
                    C=c_by_key[k],
                    model=seed_net[k],
                    scalars=make_scalars(T=T_by_key[k]),
                )
                for k in keys
            }
            out = self.plant._sweep_outputs(
                t, states, influent, seeded, params_full, passes=1, signals=signals
            )
            return (
                {k: out[k].C for k in keys},
                {k: out[k].scalars.get("T") for k in keys},
                {k: out[k].Q for k in keys},
            )

        groups: dict = {}
        for k in keys:
            groups.setdefault(id(seed_net[k]), []).append(k)
        group_lists = list(groups.values())
        zeroC = {k: jnp.zeros((nsp[k],)) for k in keys}
        zeroT = dict.fromkeys(keys, t_seed)
        return seed_net, group_lists, forward, zeroC, zeroT, forward_full

    @staticmethod
    def _probe_recycle_C(forward, keys, zeroC, zeroT, dC):
        """Probe each edge with a unit *concentration* -> the concentration
        columns ``colC[i][j] = M[:, j, i]`` (response of edge j to a unit at i).
        These per-species probes are the dominant cost; the affine map they form
        is state-invariant, so they can be precomputed once (:meth:`_compute_recycle_map`)."""
        nsp = {k: zeroC[k].shape[0] for k in keys}
        colC: dict = {}
        for ki in keys:
            cC = dict(zeroC)
            cC[ki] = jnp.ones((nsp[ki],))
            fC, _ = forward(cC, zeroT)
            colC[ki] = {kj: fC[kj] - dC[kj] for kj in keys}
        return colC

    @staticmethod
    def _probe_recycle_T(forward, keys, zeroC, zeroT, dT):
        """Probe each edge with a unit *temperature* -> the scalar temperature
        columns. Cheap (the per-species concentration part, at ``c = 0``, is
        CSE-shared with the ``d`` sweep). Re-run every RHS: ``MT`` is *not*
        state-invariant (its mixing weights ride on the concentration-dependent
        reject flows)."""
        colT: dict = {}
        for ki in keys:
            cT = dict(zeroT)
            cT[ki] = jnp.ones(())
            _, fT = forward(zeroC, cT)
            colT[ki] = {kj: fT[kj] - dT[kj] for kj in keys}
        return colT

    @staticmethod
    def _assemble_recycle_M(group_lists, colC):
        """Stack the concentration columns into per-group ``M`` (``(nsp, m, m)``)."""
        M_groups = []
        for gkeys in group_lists:
            m = len(gkeys)
            M = jnp.stack(
                [jnp.stack([colC[gkeys[i]][gkeys[j]] for i in range(m)], axis=1) for j in range(m)],
                axis=1,
            )  # (nsp, m_j, m_i)
            M_groups.append(M)
        return M_groups

    @staticmethod
    def _assemble_recycle_MT(group_lists, colT):
        """Stack the temperature columns into per-group ``MT`` (``(m, m)``)."""
        MT_groups = []
        for gkeys in group_lists:
            m = len(gkeys)
            MT = jnp.stack(
                [jnp.stack([colT[gkeys[i]][gkeys[j]] for i in range(m)]) for j in range(m)]
            )  # (m_j, m_i)
            MT_groups.append(MT)
        return MT_groups

    def _compute_recycle_map(self, t, states, params_full, resolved_flows, signals=None):
        """Probe and assemble the state-invariant recycle map(s) once, for reuse.

        Returns ``(M_groups, MT_groups)``: the per-group concentration map ``M``
        (always cached -- it is fixed by the recycle flows + topology, so for a
        fixed-pump plant it is invariant to the state and time) and the per-group
        temperature map ``MT`` *only when it too is state-invariant*
        (``_recycle_T_map_constant`` -- heat-balance / no-T modes), else ``None``
        to signal per-RHS re-probing (algebraic mode). Run once per solve and
        passed back as ``recycle_map=``, this skips the ``n_recycle_edges``
        per-species concentration probes -- the dominant per-RHS cost -- on every
        call. ``None`` if there are no recycle edges. Guarded by
        :meth:`_check_recycle_map_constant`.
        """
        keys = self.plant._recycle_keys
        if not keys:
            return None
        ctx = self._recycle_context(t, states, params_full, resolved_flows, signals)
        _, group_lists, forward, zeroC, zeroT, _ = ctx
        dC, dT = forward(zeroC, zeroT)
        colC = self._probe_recycle_C(forward, keys, zeroC, zeroT, dC)
        M_groups = self._assemble_recycle_M(group_lists, colC)
        MT_groups = None
        if self._recycle_T_map_constant and all(dT[k] is not None for k in keys):
            colT = self._probe_recycle_T(forward, keys, zeroC, zeroT, dT)
            MT_groups = self._assemble_recycle_MT(group_lists, colT)
        return (M_groups, MT_groups)

    def _maybe_recycle_map(self, t, states, params_full):
        """Precompute the cached recycle affine map, or ``None`` to probe per-call.

        Returns the :meth:`_compute_recycle_map` result when the map is known to
        be state-invariant (``_recycle_map_constant`` is ``True``, set by
        :meth:`_check_recycle_map_constant` on the first concrete solve), else
        ``None`` -- the signal for the resolver to probe ``M`` every call. The map
        is built from ``params_full`` (so a downstream gradient still flows through
        it and a parameter sweep stays correct) at the supplied ``(t, states)``,
        which the caller uses as the once-per-solve reference point. Shared by the
        forward RHS, the located-event segments and the stream reconstruction so
        all three reuse one definition.
        """
        if self._recycle_map_constant is not True:
            return None
        t = jnp.asarray(t)
        signals = self.plant._compute_signals(t, states, params_full)
        flows = self._resolve_flows(t, params_full, states)
        return self._compute_recycle_map(t, states, params_full, flows, signals)

    def _check_recycle_convergence(
        self,
        t: jnp.ndarray,
        states: dict[str, jnp.ndarray],
        params_full: jnp.ndarray,
        *,
        extra_passes: int = 4,
        rtol: float = 1e-3,
        atol: float = 1e-6,
    ) -> None:
        """Warn if the fixed-pass recycle concentration sweep has not converged.

        The recycle *flows* are solved exactly, but the inter-unit
        *concentrations* are a fixed ``recycle_passes`` Gauss-Seidel sweep (see
        :meth:`_sweep_outputs`) -- enough for the BSM topologies, but an atypical
        recycle-heavy topology could need more, in which case the steady state is
        silently wrong. This re-runs the sweep to ``recycle_passes + extra_passes``
        and warns if any output-stream concentration still moves by more than the
        ``atol + rtol*|C|`` tolerance, naming the worst-converging stream. It is a
        diagnostic only -- it never blocks the solve, and runs once per plant on
        concrete inputs (skipped under tracing). A plant with no recycle edges has
        nothing to converge and is skipped.
        """
        if not self.plant._recycle_conns:
            return

        signals = self.plant._compute_signals(t, states, params_full)

        def sweep(passes):
            streams: dict[tuple[Optional[str], str], Stream] = {}
            for port_name, series in self.plant.influents.items():
                streams[(None, port_name)] = series.at(t)
            resolved_flows = self._resolve_flows(t, params_full, states)
            # Seed from the exact affine pre-solve, exactly as _resolve_streams
            # does -- so the check measures the *residual the mop-up still has to
            # remove* (the non-affine-in-cycle part), not the gain-limited
            # convergence of the bare zero-seed the pre-solve replaced.
            seeded = self._resolve_recycle_concentrations(
                t, states, params_full, resolved_flows, signals
            )
            return self.plant._sweep_outputs(
                t, states, streams, seeded, params_full, passes=passes, signals=signals
            )

        base = sweep(self.plant.recycle_passes)
        deep = sweep(self.plant.recycle_passes + extra_passes)
        worst, worst_key = 0.0, None
        for key, sb in base.items():
            sd = deep.get(key)
            if sd is None:
                continue
            cb, cd = jnp.asarray(sb.C), jnp.asarray(sd.C)
            if cb.size == 0:
                continue
            # Scale the residual by the *stream's* characteristic magnitude (its
            # largest concentration), not per-species: a trace species at ~0 with
            # a negligible absolute change must not dominate the relative metric.
            scale = float(jnp.max(jnp.abs(cb)))
            resid = float(jnp.max(jnp.abs(cd - cb)))
            rel = resid / (atol + rtol * scale)
            if rel > worst:
                worst, worst_key = rel, key
        if worst > 1.0 and worst_key is not None:
            unit, port = worst_key
            deeper = self.plant.recycle_passes + extra_passes
            warnings.warn(
                f"Plant recycle concentration sweep may not have converged at "
                f"recycle_passes={self.plant.recycle_passes}: stream '{unit}.{port}' "
                f"still changes by ~{worst:.0f}x the rtol={rtol}/atol={atol} "
                f"tolerance after {deeper} passes. On a recycle-heavy topology "
                f"the steady state may be wrong; raise recycle_passes (e.g. "
                f"Plant(..., recycle_passes={deeper})) until this warning clears.",
                stacklevel=3,
            )

    def _check_recycle_map_constant(
        self,
        t: jnp.ndarray,
        y0: jnp.ndarray,
        params_full: jnp.ndarray,
        *,
        rtol: float = 1e-9,
    ) -> None:
        """Set ``_recycle_map_constant`` / ``_recycle_T_map_constant`` -- are the
        recycle concentration map ``M`` and temperature map ``MT`` state-fixed?

        Both maps are fixed by the recycle flows + topology (mixer/splitter/
        clarifier ratios). For a fixed-pump plant ``M`` is invariant to the state
        and time, so it can be precomputed once per solve and reused -- skipping
        the per-RHS per-species probes. ``MT`` is invariant too in heat-balance
        mode (the reactor T-state breaks the loop coupling) and when no T is
        carried, but NOT in algebraic mode (T passes through reactors, so ``MT``
        rides on the concentration-dependent recycle flows). This guard detects
        each by comparing the map at ``y0`` and a perturbed state, enabling reuse
        of whichever is constant; a varying map is re-probed every RHS. Concrete-
        only (the comparison is data-dependent); runs once per plant. No recycle
        edges -> trivially constant.
        """
        if not self.plant._recycle_conns:
            self._recycle_map_constant = True
            self._recycle_T_map_constant = True
            return

        keys = self.plant._recycle_keys

        def probe(y):
            states = self.plant._split_state(y)
            sig = self.plant._compute_signals(t, states, params_full)
            flows = self._resolve_flows(t, params_full, states)
            ctx = self._recycle_context(t, states, params_full, flows, sig)
            _, group_lists, forward, zeroC, zeroT, _ = ctx
            dC, dT = forward(zeroC, zeroT)
            colC = self._probe_recycle_C(forward, keys, zeroC, zeroT, dC)
            M = self._assemble_recycle_M(group_lists, colC)
            MT = None
            if all(dT[k] is not None for k in keys):
                colT = self._probe_recycle_T(forward, keys, zeroC, zeroT, dT)
                MT = self._assemble_recycle_MT(group_lists, colT)
            return M, MT

        # Probe THREE materially-different states, not two: a state-dependent map
        # that coincidentally agrees at a single perturbed pair (e.g. a split
        # ratio invariant under one affine shift) would be mis-cached as constant,
        # and there is no mid-solve re-guard. Three distinct affine perturbations
        # make that coincidence far less likely (a genuinely state-dependent map
        # must agree under all three to slip through). Residual limitation: a map
        # that is constant on all three probes but varies elsewhere is still
        # mis-cached -- not observed for any shipped topology, where the only
        # state-dependent splits (clarifier %TSS) vary under these perturbations.
        probes = [probe(y0), probe(1.3 * y0 + 1.0), probe(0.5 * y0 + 3.0)]
        Ma, MTa = probes[0]

        def _max_dev(idx):
            base = probes[0][idx]
            return max(
                (
                    float(jnp.max(jnp.abs(a - b)))
                    for other in probes[1:]
                    for a, b in zip(base, other[idx])
                ),
                default=0.0,
            )

        self._recycle_map_constant = bool(_max_dev(0) <= rtol)
        if MTa is None:
            self._recycle_T_map_constant = True  # no T carried
        else:
            self._recycle_T_map_constant = bool(_max_dev(1) <= rtol)

    def _flow_one_pass(self, t, params_full, states, design):
        """Build the affine recycle-FLOW forward pass.

        Returns ``(one_pass, n)``: ``one_pass(recycle_Qs) -> (recycled_stack,
        full_flows_dict)`` is one topological sweep of every unit's
        ``flow_outputs`` with the ``n`` recycle back-edges held at the trial
        flows. Shared by :meth:`_resolve_flows` and :meth:`_compute_flow_map`
        (the cached-``A`` path) so the two cannot drift.
        """
        influent_override = (design or {}).get("influent", {})
        base: dict[tuple[Optional[str], str], jnp.ndarray] = {}
        for port_name, series in self.plant.influents.items():
            if port_name in influent_override:
                ov = influent_override[port_name]
                # Absolute override and/or multiplicative flow scale (see
                # _resolve_streams); both default to identity.
                base[(None, port_name)] = ov.get("Q", series.at(t).Q) * ov.get("Q_scale", 1.0)
            else:
                base[(None, port_name)] = series.at(t).Q
        recycle_keys = self.plant._recycle_keys
        n = len(recycle_keys)

        def one_pass(recycle_Qs):
            flows = dict(base)
            for k, q in zip(recycle_keys, recycle_Qs):
                flows[k] = q
            for name in self.plant._unit_order:
                in_flows: dict[str, jnp.ndarray] = {}
                for conn in self.plant._inputs_by_unit.get(name, ()):
                    src = (
                        (None, conn.from_port)
                        if conn.from_unit is None
                        else (conn.from_unit, conn.from_port)
                    )
                    in_flows[conn.to_port] = flows[src]
                unit = self.plant.units[name]
                params_unit = self.plant._params_for_unit(name, params_full)
                # Every unit gets the same flow_outputs signature; the context
                # carries the unit's own state and the current time, both held
                # fixed across the affine recycle-flow probe (a state- or
                # time-dependent split is constant in the recycle flows, so the
                # probe stays exact). A fixed-split unit ignores the context.
                ctx = FlowContext(
                    state=None if states is None else states[name],
                    t=t,
                )
                out = unit.flow_outputs(in_flows, params_unit, ctx)
                for port, q in out.items():
                    flows[(name, port)] = q
            recycled = jnp.stack([flows[k] for k in recycle_keys]) if n else jnp.zeros((0,))
            return recycled, flows

        return one_pass, n

    def _compute_flow_map(self, t, params_full, states, design=None):
        """Probe the recycle-FLOW affine map ``A`` (the ``n x n`` back-edge
        response). State/time-invariant for a fixed-pump plant, so cacheable
        (:meth:`_maybe_flow_map`) -- the flow analogue of
        :meth:`_compute_recycle_map`. ``params_full`` is coerced so the
        flow-setpoint block (the RAS/wastage/primary-sludge split flows, appended
        after the kinetic blocks) is present even from a kinetic-only vector --
        the cached ``A`` must match the hot path, which runs on coerced params.
        Idempotent on a full vector, so a flow-setpoint gradient still flows."""
        params_full = self.plant._coerce_params(params_full)
        one_pass, n = self._flow_one_pass(t, params_full, states, design)
        if n == 0:
            return jnp.zeros((0, 0))
        b, _ = one_pass(jnp.zeros((n,)))
        eye = jnp.eye(n)
        cols = [one_pass(eye[i])[0] - b for i in range(n)]
        return jnp.stack(cols, axis=1)

    def _maybe_flow_map(self, t, states, params_full):
        """The flow map ``A`` when it is state-invariant (``_flow_map_constant``
        True, set by :meth:`_check_flow_map_constant`), else ``None`` (re-probe).
        Same params handling as :meth:`_maybe_recycle_map`."""
        if self._flow_map_constant is not True:
            return None
        return self._compute_flow_map(t, params_full, states)

    def _check_flow_map_constant(self, t, y0, params_full, *, rtol: float = 1e-9):
        """Set ``_flow_map_constant`` -- is the recycle flow map ``A``
        state-fixed? ``A`` is fixed by the recycle flows + topology; for a
        fixed-pump plant it is invariant to the state, so it can be precomputed
        once per solve and reused, skipping the per-RHS flow probe. A flow split
        riding on a unit state (a level-gated storage bypass) makes it vary, then
        it is re-probed. Compares ``A`` over THREE materially-different states
        (not a single pair, which a split invariant under one affine shift could
        slip through; there is no mid-solve re-guard). Concrete-only; runs once
        per plant. No recycle edges -> trivially constant. (The flow analogue of
        :meth:`_check_recycle_map_constant`.)"""
        if not self.plant._recycle_conns:
            self._flow_map_constant = True
            return
        A0 = self._compute_flow_map(t, params_full, self.plant._split_state(y0))
        if not A0.size:
            self._flow_map_constant = True
            return
        wA = max(
            float(
                jnp.max(
                    jnp.abs(A0 - self._compute_flow_map(t, params_full, self.plant._split_state(s)))
                )
            )
            for s in (1.3 * y0 + 1.0, 0.5 * y0 + 3.0)
        )
        self._flow_map_constant = bool(wA <= rtol)

    def _resolve_flows(
        self,
        t: jnp.ndarray,
        params_full: jnp.ndarray,
        states: Optional[dict[str, jnp.ndarray]] = None,
        *,
        design: Optional[dict] = None,
        check_affine: bool = False,
        flow_map: Optional[jnp.ndarray] = None,
    ) -> dict[tuple[Optional[str], str], jnp.ndarray]:
        """Solve the recycle FLOW network exactly (decoupled from concentration).

        Every unit exposes a linear ``flow_outputs`` rule (output port flows from
        input port flows). One topological pass over those rules, with the
        recycle back-edges held at trial values, is an affine map
        ``x -> A x + b`` on the back-edge flows. We recover ``A`` and ``b`` by
        probing (one pass at zero, one per back-edge) and solve
        ``(I - A) x = b`` for the consistent recycle flows -- exact and
        gain-independent, in ``n_recycle + 1`` cheap scalar passes. Returns the
        resolved flow for every stream key.

        Every unit's ``flow_outputs`` receives the same :class:`FlowContext`
        (its own current state and the time), so a unit whose split depends on
        its *own internal state* (a variable-volume storage tank, whose overflow
        bypass is gated by the liquid level) or on time (a scheduled pump) reads
        it from there; the rest ignore it. The affine probe stays exact as long
        as that dependence does not couple to the recycle-flow variables -- true
        here because such a unit's *inlet* flow comes from the fixed-pump sludge
        line (constant during the probe), so at fixed state/time its outputs are
        constant in the recycle flows.

        The self-consistency of this assumption is checked once, at ``(t0, y0)``,
        by ``check_affine`` (a warning, never a block). That t0 check does NOT
        catch a unit with a *piecewise-linear* flow rule (a
        ``ThresholdSplitter`` bypass, a level-gated ``StorageTank`` bypass) that is on
        one side of its kink at ``t0`` but crosses it later in a dynamic run: the
        probe then linearises across the kink at that time silently. The resolved
        flows are exact only while every unit stays in the affine regime it
        occupied at ``t0`` (issue #255).
        """
        one_pass, n = self._flow_one_pass(t, params_full, states, design)
        if n == 0:
            return one_pass(jnp.zeros((0,)))[1]
        b, _ = one_pass(jnp.zeros((n,)))
        eye = jnp.eye(n)
        if flow_map is not None:
            A = flow_map
        else:
            cols = [one_pass(eye[i])[0] - b for i in range(n)]
            A = jnp.stack(cols, axis=1)
        x = jnp.linalg.solve(eye - A, b)
        recycled_at_x, flows = one_pass(x)
        if check_affine:
            # ``x`` solves the *linearised* (probed) system; re-evaluating the
            # forward pass at ``x`` reproduces it iff the flow rules are truly
            # affine in the recycle flows. A mismatch means a piecewise-linear
            # rule (a threshold split / storage bypass with a recycle-dependent
            # inlet crossing its kink) was linearised across the kink -- the
            # resolved flows are then inaccurate. Concrete-only (see solve()).
            self.plant._warn_if_flow_nonaffine(recycled_at_x, x)
        return flows
