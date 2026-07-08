"""Runtime-compiled reaction model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from aquakin.core.context import CompileContext
from aquakin.core.hints import did_you_mean
from aquakin.core.nodes import (
    ASTNode,
    ConditionNode,
    ConstantNode,
    NegateNode,
    ParamNode,
    RateCallable,
    SpeciesNode,
    _BinaryNode,
)
from aquakin.core.parser import parse_rate_expression
from aquakin.core.stoich_resolve import resolve_auto_coefficients
from aquakin.core.temperature import arrhenius_factor
from aquakin.core.units import prettify_units

# Used to detect references-to-other-expressions during AST inspection.
_LEAF_TYPES = (ConstantNode, SpeciesNode, ConditionNode, ParamNode)


# Stoichiometry-coefficient expressions may reference only constants,
# parameters, and arithmetic / negation. Species, conditions, named
# expressions, and domain functions are forbidden — stoichiometry must be
# state-independent so we can evaluate it once per ``solve`` call.
_ALLOWED_STOICH_NODES = (ConstantNode, ParamNode, NegateNode, _BinaryNode)


def _validate_stoich_ast(ast: ASTNode, rxn_name: str, species: str) -> None:
    """Reject stoich ASTs that reference state, conditions, or functions.

    Generic over the AST via ``children()``: an allowed node's children are
    validated recursively, so a new node type cannot slip through unchecked.
    """
    if not isinstance(ast, _ALLOWED_STOICH_NODES):
        kind = type(ast).__name__
        raise ValueError(
            f"Stoichiometric coefficient for '{rxn_name}' / '{species}' uses an "
            f"unsupported expression element ({kind}). Stoich expressions may only "
            f"reference parameters, numeric constants, and arithmetic / negation; "
            f"species, conditions, named expressions, and domain functions "
            f"(arrhenius, pH_switch, monod, ...) are not allowed."
        )
    for child in ast.children():
        _validate_stoich_ast(child, rxn_name, species)


def _collect_param_refs(node: ASTNode) -> set[str]:
    """Walk an AST and return every ParamNode name encountered.

    Driven by ``ASTNode.children()`` so it descends into *every* node type,
    including ones added later -- a hand-enumerated walk would silently treat an
    unrecognised node as a leaf and miss its parameter references.
    """
    if isinstance(node, ParamNode):
        return {node.name}
    refs: set[str] = set()
    for child in node.children():
        refs |= _collect_param_refs(child)
    return refs


def _substitute(node: ASTNode, expr_asts: dict[str, ASTNode]) -> ASTNode:
    """Return a new AST with ParamNode references to named expressions replaced
    by the corresponding (already-resolved) expression AST.

    Driven by ``ASTNode.map_children()``: every node type's children are
    rewritten generically (and identity is preserved where nothing changed), so
    a new node type cannot silently skip inlining.
    """
    if isinstance(node, ParamNode) and node.name in expr_asts:
        return expr_asts[node.name]
    return node.map_children(lambda child: _substitute(child, expr_asts))


def _topo_sort_expressions(
    expr_names: list[str],
    expr_deps: dict[str, set[str]],
) -> list[str]:
    """Topologically sort named expressions; raise on cycle."""
    visited: dict[str, str] = {}  # name -> "visiting" | "done"
    order: list[str] = []

    def _visit(name: str, stack: list[str]) -> None:
        state = visited.get(name)
        if state == "done":
            return
        if state == "visiting":
            cycle = stack[stack.index(name) :] + [name]
            raise ValueError(f"Cycle in named expressions: {' -> '.join(cycle)}")
        visited[name] = "visiting"
        for dep in expr_deps.get(name, set()):
            _visit(dep, stack + [name])
        visited[name] = "done"
        order.append(name)

    for name in expr_names:
        _visit(name, [])
    return order


if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass
class CompiledModel:
    """
    Runtime representation of a reaction model.

    Attributes
    ----------
    name : str
        Model identifier (e.g. ``"ozone_bromate"``).
    description : str
        Free-text description of the model.
    references : list[str]
        Literature citations associated with the model.
    species : list[str]
        Ordered species names. Index in this list is the index used in ``C``.
    parameters : list[str]
        Ordered, namespaced parameter names (e.g. ``"O3_Br_direct.k1"``).
        Index in this list is the index used in ``params``.
    conditions_required : list[str]
        Names of condition fields the model needs at runtime.
    stoich_matrix : jnp.ndarray
        Shape ``(n_reactions, n_species)``. Stoichiometric coefficient of
        species *j* in reaction *i*.
    reaction_names : list[str]
        Ordered reaction names corresponding to rows of ``stoich_matrix``.
    rate_callables : list[Callable]
        Per-reaction compiled rate functions with the canonical signature
        ``(C, params, condition_arrays, loc_idx) -> scalar``.
    rate_asts : list[ASTNode]
        Per-reaction parsed AST roots. Retained for inspection (``to_latex``)
        and not used in the runtime hot path.
    param_index : dict[str, int]
        Map from namespaced parameter name to its position in ``params``.
    species_index : dict[str, int]
        Map from species name to its position in ``C``.
    _default_concentrations : jnp.ndarray
        Default initial concentrations, shape ``(n_species,)``.
    _default_parameters : jnp.ndarray
        Default parameter values, shape ``(n_params,)``.
    parameter_bounds : dict[str, tuple[float, float]]
        ``(low, high)`` bounds per namespaced parameter name. Parameters
        without declared bounds are absent from the mapping (no ``None``
        sentinel values).
    """

    name: str
    description: str
    references: list[str]
    species: list[str]
    parameters: list[str]
    conditions_required: list[str]
    stoich_matrix: jnp.ndarray
    reaction_names: list[str]
    rate_callables: list[RateCallable]
    rate_asts: list[ASTNode]
    param_index: dict[str, int]
    species_index: dict[str, int]
    _default_concentrations: jnp.ndarray
    _default_parameters: jnp.ndarray
    _condition_defaults: dict[str, float] = field(default_factory=dict)
    # Per-species metadata carried verbatim from the YAML ``species:`` block,
    # keyed by species name. ``species_units`` gives the concentration units
    # (e.g. ``"g_COD/m3"``, ``"g_N/m3"``, ``"mol/L"``) and
    # ``species_descriptions`` the human-readable label. Surfaced in
    # :meth:`summary`, :meth:`units_of` / :meth:`description_of`, and on the
    # solution objects (``units_named``), so results no longer have to
    # re-derive units by string-matching species names.
    species_units: dict[str, str] = field(default_factory=dict)
    species_descriptions: dict[str, str] = field(default_factory=dict)
    # Optional per-species content of conserved quantities (COD / N / P / S / Fe /
    # charge / ...), in the species' own measure, declared in the YAML
    # ``species[].composition`` block. Empty unless declared. Consumed by
    # :meth:`composition` / :meth:`check_conservation`, so a model can carry its
    # own conservation table instead of one hand-maintained elsewhere.
    species_composition: dict[str, dict[str, float]] = field(default_factory=dict)
    # Declared units for the rate-constant parameters (keyed by namespaced name,
    # e.g. ``"O3_Br_direct.k1"``) and the condition fields (keyed by field name).
    # Advisory metadata carried verbatim from the YAML; consumed only by the
    # opt-in :meth:`check_units` dimensional-consistency check. A blank ``units:``
    # is kept as ``""`` and treated as "unknown" (skipped) by the check.
    parameter_units: dict[str, str] = field(default_factory=dict)
    condition_units: dict[str, str] = field(default_factory=dict)
    parameter_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    parameter_transforms: dict[str, str] = field(default_factory=dict)
    # Gaussian priors per namespaced parameter name, as ``(mean, std)`` in
    # physical space. Parameters without a declared prior are absent. Consumed
    # by ``aquakin.calibrate`` to regularise the fit toward literature values.
    parameter_priors: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Parameter-dependent stoichiometry. Each tuple is (row, col, callable)
    # where ``callable(params) -> scalar`` computes the coefficient from the
    # current parameter vector. ``stoich_matrix`` holds zeros at these
    # (row, col) cells; ``compute_stoich(params)`` scatters the dynamic
    # values onto the static base.
    stoich_dynamic: list[tuple[int, int, "Callable"]] = field(default_factory=list)
    _stoich_dynamic_rows: "jnp.ndarray | None" = None
    _stoich_dynamic_cols: "jnp.ndarray | None" = None
    # Optional state-derived condition fields (e.g. a charge-balance pH).
    # ``derived_condition_fn(C, params, condition_arrays, loc_idx)`` returns a
    # mapping of extra condition-field name -> scalar, computed from the
    # instantaneous state. These are merged into ``condition_arrays`` before
    # the rate callables run, so ordinary ``{pH}`` / ``pH_switch`` expressions
    # see the derived value. ``derived_fields`` lists the names it produces.
    derived_condition_fn: "Callable | None" = None
    derived_fields: list[str] = field(default_factory=list)
    # Optional projection onto the precipitation equilibrium, present when a
    # ``precipitation:`` block declares any ``mode: equilibrium`` minerals.
    # ``precipitation_equilibrium_fn(C, condition_arrays, loc_idx) -> C_eq``
    # snaps a composition onto the algebraic mineral equilibrium (see
    # :meth:`CompiledModel.precipitation_equilibrium`).
    precipitation_equilibrium_fn: "Callable | None" = None
    # Optional positivity limiter on the net reaction term. When set, each
    # species' net reaction rate is throttled as its concentration approaches
    # zero, so consumption cannot drive a state negative. Applied to the
    # reaction term only (transport is added by the reactor afterwards).
    positivity_threshold: "float | None" = None
    # Optional clamp of the concentration vector to >= 0 when evaluating the
    # reaction rates (and any state-derived condition such as pH). This protects
    # the nonlinear kinetics (Monod / ratio terms) from evaluating at a
    # transiently-negative state, where they produce large/garbage rates and a
    # stiff blow-up. The clamp applies ONLY to rate evaluation: the raw state is
    # still what the reactor's (linear) transport term and the unit outputs see,
    # so the linear washout stays self-correcting and the inter-unit mass balance
    # stays exact. Concretely, the rate inputs are clamped with ``max(x, 0)``
    # before the process rates. Identity at feasible
    # (non-negative) states, so it does not change the physical solution.
    # Consequence for users: because the clamp is not applied to the saved state,
    # a returned trajectory (``solution.C`` / a plant ``state``) may contain small
    # transient negatives -- a normal numerical artefact of the stiff solve, not a
    # solver/model error (the rates were computed on the clamped values). The
    # Solution docstrings flag this; clip for display with ``jnp.maximum(C, 0)``.
    clip_negative_states: bool = False

    # Optional per-parameter temperature corrections. Each entry is
    # ``(param_idx, ln_theta, ref_T, condition_field)``: when evaluating the
    # rates, the parameter at ``param_idx`` is multiplied by
    # ``exp(ln_theta * (T - ref_T))`` (a ``theta^(T - ref_T)`` Arrhenius-style
    # factor), where ``T`` is read from ``condition_arrays[condition_field]``.
    # The correction is applied to the rate constants only (it is confined to
    # :meth:`rates`); ``compute_stoich`` always uses the raw parameters. Unity
    # at ``T == ref_T``, so a model whose conditions sit at the reference
    # temperature behaves exactly as if uncorrected.
    temperature_corrections: list = field(default_factory=list)

    # Precomputed, vectorised form of ``temperature_corrections``, grouped by
    # condition field: a list of ``(condition_field, idx_array, ln_theta_array,
    # ref_T_array)``. Built once in :meth:`__post_init__` so :meth:`_apply_temperature`
    # applies one scatter-multiply per condition field, not one per correction.
    _temp_groups: list = field(default_factory=list, init=False, repr=False)

    # Vectorized rate kernel: evaluates all reaction rates in batched ops with
    # a much smaller traced jaxpr than the per-reaction scalar stack (the same
    # values -- bit-identical -- but a faster compile, which dominates the cost
    # of stiff solves and especially their reverse-mode adjoint). Built once in
    # __post_init__; ``None`` falls back to the scalar path (an unsupported AST
    # node type, or a model with no reactions).
    _rate_kernel: "Any | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        grouped: dict[str, tuple[list, list, list]] = {}
        for idx, ln_theta, ref_T, cond in self.temperature_corrections:
            idxs, ln_thetas, ref_Ts = grouped.setdefault(cond, ([], [], []))
            idxs.append(idx)
            ln_thetas.append(ln_theta)
            ref_Ts.append(ref_T)
        self._temp_groups = [
            (cond, jnp.asarray(idxs), jnp.asarray(ln_thetas), jnp.asarray(ref_Ts))
            for cond, (idxs, ln_thetas, ref_Ts) in grouped.items()
        ]
        self._build_rate_kernel()

    def _build_rate_kernel(self) -> None:
        """Build the vectorized rate kernel, or leave it ``None`` to fall back
        to the scalar per-reaction path (unsupported node type / no reactions)."""
        from aquakin.core.vector_kernel import (
            UnsupportedNode,
            build_vectorized_rates,
        )

        if not self.rate_asts:
            self._rate_kernel = None
            return
        try:
            self._rate_kernel = build_vectorized_rates(
                self.rate_asts,
                self.reaction_names,
                self.species_index,
                self.param_index,
            )
        except UnsupportedNode:
            self._rate_kernel = None

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_names)

    @property
    def n_params(self) -> int:
        return len(self.parameters)

    # ----- Construction vectors (build C / params / conditions / atol) --------

    def default_concentrations(self) -> jnp.ndarray:
        """Return a copy of the default initial-concentration vector.

        This is the model's **reference state** -- every species at its YAML
        ``default_concentration`` -- *not* a blank slate. For most models that
        means many species are nonzero (e.g. a full biomass inventory). It is a
        sensible starting initial condition, but it is the **wrong base for a
        feed composition**: an influent species you do not list should be
        *absent*, not sitting at its reference value. Build feeds with
        :meth:`influent` or ``concentrations(..., base="zero")`` instead -- see
        :meth:`concentrations` for the trap in full.
        """
        return jnp.asarray(self._default_concentrations)

    def default_parameters(self) -> jnp.ndarray:
        """Return a copy of the default parameter vector."""
        return jnp.asarray(self._default_parameters)

    def _override_vector(self, base, index_map, overrides, kwargs, kind):
        """Return ``base`` with named entries replaced via ``index_map``.

        ``overrides`` is a dict (the robust form -- many species/parameter names
        are not valid Python identifiers, e.g. ``"Br-"`` or the namespaced
        ``"O3_Br_direct.k1"``); ``kwargs`` adds identifier-safe convenience
        overrides. Unknown names raise a ``KeyError`` with a close-match hint.
        """
        merged: dict[str, float] = {}
        if overrides is not None:
            if not isinstance(overrides, dict):
                raise TypeError(
                    f"overrides must be a dict of {kind} name -> value; got "
                    f"{type(overrides).__name__}."
                )
            merged.update(overrides)
        merged.update(kwargs)
        if not merged:
            return base
        idxs, vals = [], []
        for name, value in merged.items():
            if name not in index_map:
                suffix = did_you_mean(name, index_map)
                raise KeyError(f"Unknown {kind} '{name}' for model '{self.name}'.{suffix}")
            idxs.append(index_map[name])
            vals.append(jnp.asarray(value))
        # Assemble with jnp.stack rather than float()-coercing each value, so a
        # traced/JAX override (building C0 / params inside jax.grad / vmap to
        # differentiate w.r.t. an initial concentration or rate constant) flows
        # through unconcretised -- the convenience builder stays AD-clean.
        stacked = jnp.stack(vals).astype(base.dtype)
        return base.at[jnp.asarray(idxs)].set(stacked)

    def concentrations(self, overrides=None, /, *, base: str = "defaults", **kwargs) -> jnp.ndarray:
        """Initial-concentration vector with named species set.

        A by-name builder that avoids manual
        ``default_concentrations().at[species_index[name]].set(value)`` chains.

        .. warning::
           **Building a feed? Use** :meth:`influent` **or** ``base="zero"``.
           With the default ``base="defaults"`` every species you do *not* list
           keeps its YAML reference value, so ``concentrations({"SS": 100.0})``
           silently carries a full biomass/inert inventory (``XB_H``, ``XS``,
           ``XI``, ...) into the result. That is correct for an **initial
           condition** (the reactor starts from the reference state with a few
           species adjusted) but wrong for an **influent** (an unlisted species
           should be absent). For a feed, pass ``base="zero"`` so the vector
           contains *only* what you list, or use :meth:`influent`, which
           defaults to the zero base.

        Parameters
        ----------
        overrides : dict[str, float], optional
            Species name -> concentration. Positional-only. Use the dict for
            names that are not valid Python identifiers (``"Br-"``, ``"BrO3-"``).
        base : {"defaults", "zero"}, optional
            Starting point for unlisted species. ``"defaults"`` (the default)
            keeps each unspecified species at its YAML reference value;
            ``"zero"`` starts every species at 0, so the result contains *only*
            what was passed -- the correct base for building a feed composition
            (where an unspecified species means "absent", not "at its reference
            value"). A species literally named ``base`` must be passed via the
            ``overrides`` dict.
        **kwargs : float
            Convenience overrides for identifier-safe species names (``O3=1e-4``).

        Returns
        -------
        jnp.ndarray
            Concentration vector of shape ``(n_species,)``.

        Examples
        --------
        >>> model.concentrations({"O3": 1e-4, "Br-": 1e-5})
        >>> model.concentrations(SS=50.0)
        >>> model.concentrations({"SS": 50.0, "SNH": 25.0}, base="zero")
        """
        if base == "defaults":
            vec = self.default_concentrations()
        elif base == "zero":
            vec = jnp.zeros_like(self._default_concentrations)
        else:
            raise ValueError(f"base must be 'defaults' or 'zero', got {base!r}.")
        return self._override_vector(
            vec,
            self.species_index,
            overrides,
            kwargs,
            "species",
        )

    def influent(self, overrides=None, /, *, Q: float, base: str = "zero", T=None, **kwargs):
        """Build a constant-in-time influent stream from a feed composition.

        Convenience for the common "constant feed of known composition" case:
        a one-call, **zero-based** :class:`~aquakin.plant.influent.InfluentSeries`
        so an unspecified species is absent from the feed rather than sitting at
        its YAML reference value. The returned series is constant in time, so it
        can be passed straight to ``plant.add_influent(...)``.

        Parameters
        ----------
        overrides : dict[str, float], optional
            Species name -> feed concentration. Positional-only.
        Q : float
            Volumetric flow rate of the feed (constant), required.
        base : {"zero", "defaults"}, optional
            Composition base, defaulting to ``"zero"`` (see :meth:`concentrations`).
        T : float, optional
            Constant feed temperature (Kelvin). ``None`` (default) leaves the
            influent temperature-agnostic.
        **kwargs : float
            Convenience overrides for identifier-safe species names.

        Returns
        -------
        InfluentSeries
            A constant-in-time influent.

        Examples
        --------
        >>> net.influent({"SS": 60.0, "SNH": 25.0}, Q=18446.0)
        >>> net.influent(SS=400.0, Q=2.0)            # carbon dose
        """
        from aquakin.plant.influent import InfluentSeries

        return InfluentSeries.constant(self, overrides, Q=Q, base=base, T=T, **kwargs)

    def parameter_values(self, overrides=None, /, **kwargs) -> jnp.ndarray:
        """Parameter vector: defaults with named (namespaced) parameters set.

        The parameter analogue of :meth:`concentrations`. Names are the
        namespaced keys (``"O3_Br_direct.k1"``), so the dict form is the usual
        one; ``kwargs`` works for the rare bare model-level parameter.

        Examples
        --------
        >>> model.parameter_values({"O3_Br_direct.k1": 175.0})
        """
        return self._override_vector(
            self.default_parameters(),
            self.param_index,
            overrides,
            kwargs,
            "parameter",
        )

    def atol(self, overrides=None, /, default=None, **kwargs) -> jnp.ndarray:
        """Per-species absolute-tolerance vector for a reactor's ``atol=``.

        ``default`` everywhere, with named species overridden -- the by-name
        replacement for ``jnp.full((n_species,), d).at[species_index[s]].set(v)``
        when a trace species needs a tighter tolerance.

        ``default=None`` (the default) starts from the per-component noise floor
        :func:`~aquakin.integrate._common.default_atol` scaled off the model's
        reference concentrations -- the same floor a reactor built with
        ``atol=None`` uses -- rather than a fixed scalar, which is ~9 orders too
        tight for g/m3 ASM/ADM states. Pass an explicit scalar ``default=`` for a
        uniform floor.

        Examples
        --------
        >>> reactor = BatchReactor(net, conds, atol=net.atol({"OH": 1e-20}))
        >>> reactor = BatchReactor(net, conds, atol=net.atol({"OH": 1e-20}, default=1e-12))
        """
        from aquakin.integrate._common import default_atol

        base = (
            default_atol(self.default_concentrations())
            if default is None
            else jnp.full((self.n_species,), float(default))
        )
        return self._override_vector(base, self.species_index, overrides, kwargs, "species")

    def default_conditions(self, n_locations: int = 1):
        """Build a :class:`SpatialConditions` from the model's declared defaults.

        Convenience for the common case of "use the defaults as written in
        the YAML". Each required condition is broadcast to ``n_locations``.
        """
        from aquakin.core.conditions import SpatialConditions

        return SpatialConditions.uniform(n_locations, **self._condition_defaults)

    def condition_defaults(self) -> "dict[str, float]":
        """The model's declared default condition values as a plain scalar dict.

        The scalar-dict counterpart to :meth:`default_conditions` (which returns a
        location-broadcast :class:`SpatialConditions`): ``{field: default}`` for
        each condition the YAML declares a default for. The public accessor for
        the raw defaults -- e.g. a plant builder seeding a reactor unit's scalar
        ``conditions=`` from the model's declared operating point. A copy, so the
        caller cannot mutate the model's stored defaults.
        """
        return dict(self._condition_defaults)

    # ----- Runtime hot path (state -> rates -> dCdt; differentiable) ----------

    def precipitation_equilibrium(
        self,
        C: "jnp.ndarray | None" = None,
        conditions=None,
        *,
        loc_idx: int = 0,
    ) -> jnp.ndarray:
        """Project a composition onto its mineral precipitation equilibrium.

        For a ``precipitation:`` model with ``mode: equilibrium`` minerals,
        solve the coupled algebraic equilibrium -- every precipitated mineral on
        its solubility (``IAP = Ksp``), every absent mineral undersaturated, mass
        balanced across the shared ions -- and return the equilibrium-projected
        state: each equilibrium solid set to its equilibrium amount and the
        dissolved ions rebalanced. This is the differentiable, non-stiff
        alternative to integrating an ultra-insoluble mineral's kinetics (whose
        ``~1e13`` rate Jacobian defeats every sensitivity method): the solve is
        well conditioned and ``jax.grad`` flows through it via the
        implicit-function-theorem sensitivity, so it composes with
        :func:`~aquakin.sensitivity` / :func:`~aquakin.calibrate` w.r.t. the dose
        (the composition) and the operating conditions (pH, T).

        Parameters
        ----------
        C : jnp.ndarray, optional
            Composition to project, shape ``(n_species,)``. Defaults to the
            model's ``default_concentrations()``.
        conditions : SpatialConditions, optional
            Conditions supplying pH / T. Defaults to ``default_conditions()``.
        loc_idx : int, optional
            Spatial location index into ``conditions`` (default 0).

        Returns
        -------
        jnp.ndarray
            The equilibrium-projected composition, shape ``(n_species,)``. Read a
            residual dissolved ion or a solid amount with
            :meth:`species_index`-based indexing or a ``BatchSolution``-style
            accessor.

        Raises
        ------
        ValueError
            If the model declares no ``mode: equilibrium`` minerals.
        """
        if self.precipitation_equilibrium_fn is None:
            raise ValueError(
                "precipitation_equilibrium() requires a precipitation: block with "
                "at least one 'mode: equilibrium' mineral; this model has none."
            )
        C = self.default_concentrations() if C is None else jnp.asarray(C)
        if conditions is None:
            conditions = self.default_conditions()
        return self.precipitation_equilibrium_fn(C, conditions.fields, loc_idx)

    def rates(
        self,
        C: jnp.ndarray,
        params: jnp.ndarray,
        condition_arrays: dict[str, jnp.ndarray],
        loc_idx,
    ) -> jnp.ndarray:
        """
        Evaluate all reaction rates at the given state.

        Parameters
        ----------
        C : jnp.ndarray
            Concentration vector, shape ``(n_species,)``.
        params : jnp.ndarray
            Flat parameter vector, shape ``(n_params,)``.
        condition_arrays : dict[str, jnp.ndarray]
            Mapping ``field_name -> (n_locations,) array``.
        loc_idx : int or jnp.ndarray
            Spatial location index.

        Returns
        -------
        jnp.ndarray
            Reaction rate vector, shape ``(n_reactions,)``.
        """
        if self.clip_negative_states:
            # Clamp to >= 0 for rate evaluation only; the reactor's transport
            # term and the unit outputs still use the raw, un-clamped state.
            C = jnp.maximum(C, 0.0)
        if self.derived_condition_fn is not None:
            condition_arrays = self._augment_conditions(C, params, condition_arrays, loc_idx)
        if self.temperature_corrections:
            params = self._apply_temperature(params, condition_arrays, loc_idx)
        if self._rate_kernel is not None:
            return self._rate_kernel(C, params, condition_arrays, loc_idx)
        return jnp.stack([f(C, params, condition_arrays, loc_idx) for f in self.rate_callables])

    def _apply_temperature(
        self, params: jnp.ndarray, condition_arrays: dict, loc_idx
    ) -> jnp.ndarray:
        """Multiply temperature-corrected rate constants by ``theta^(T-ref_T)``.

        Returns a new parameter vector with each corrected entry scaled by its
        Arrhenius-style factor. Confined to rate evaluation, so the
        stoichiometry parameters are untouched. Corrections are grouped by
        condition field (see ``_temp_groups``) and applied as one vectorised
        scatter-multiply per field rather than one scatter per correction.
        """
        for cond, idxs, ln_thetas, ref_Ts in self._temp_groups:
            T = condition_arrays[cond][loc_idx]
            factors = arrhenius_factor(T, ref_Ts, ln_thetas)
            # ``idxs`` are distinct (a parameter carries at most one temperature
            # correction), so the scatter-multiply has unique indices -- required
            # for its reverse-mode gradient (scatter_mul VJP needs it).
            params = params.at[idxs].multiply(factors, unique_indices=True)
        return params

    def _augment_conditions(
        self,
        C: jnp.ndarray,
        params: jnp.ndarray,
        condition_arrays: dict[str, jnp.ndarray],
        loc_idx,
    ) -> dict[str, jnp.ndarray]:
        """Merge state-derived condition fields into ``condition_arrays``.

        Each derived scalar is broadcast across all spatial locations (via the
        shared :func:`_merge_derived`) so that the existing
        ``condition_arrays[name][loc_idx]`` indexing used by
        :class:`~aquakin.core.nodes.ConditionNode` returns it unchanged. The
        derived value is computed from the local state ``C`` (the state at
        ``loc_idx``), so the entry read at ``loc_idx`` is the correct one;
        other indices are never consulted within this call.
        """
        derived = self.derived_condition_fn(C, params, condition_arrays, loc_idx)
        return _merge_derived(condition_arrays, derived)

    def compute_stoich(self, params: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the stoichiometry matrix at the given parameter vector.

        For models whose stoichiometry is purely numeric this returns
        the cached ``stoich_matrix`` unchanged. For models with
        parameter-dependent coefficients (the dynamic entries listed in
        ``stoich_dynamic``), this scatters the per-call values onto the
        static base.

        Reactors typically call this once per ``solve()`` and hoist the
        result as a closure constant for the duration of the integration
        — see the rhs builders in :mod:`aquakin.integrate.batch` etc.
        """
        if not self.stoich_dynamic:
            return self.stoich_matrix
        values = jnp.stack([fn(params) for (_, _, fn) in self.stoich_dynamic])
        return self.stoich_matrix.at[self._stoich_dynamic_rows, self._stoich_dynamic_cols].set(
            values
        )

    def dCdt(
        self,
        C: jnp.ndarray,
        params: jnp.ndarray,
        condition_arrays: dict[str, jnp.ndarray],
        loc_idx,
        *,
        stoich: "jnp.ndarray | None" = None,
        rate_scale: "jnp.ndarray | None" = None,
    ) -> jnp.ndarray:
        """Return the fully post-processed chemistry RHS for one location.

        This is the single canonical reaction term — ``stoich.T @ rates(...)``
        with ``clip_negative_states`` applied to the rate inputs (inside
        :meth:`rates`) and the ``positivity_limiter`` applied to the net term.
        Reactors that add their own transport build their RHS as
        ``dCdt(...) + transport`` rather than re-deriving this, so the clip and
        limiter are applied identically everywhere.

        ``stoich`` may be precomputed via :meth:`compute_stoich` and passed
        in to avoid re-evaluating parameter-dependent coefficients on every
        ODE step. If omitted, it is computed from ``params`` here.

        ``rate_scale`` is an optional per-reaction multiplier applied to the
        rate vector **before** ``stoich.T @ r`` — for a reaction-level throttle
        such as the biofilm density cap, where uptake and production must scale
        together (a post-multiply on the net term would not conserve mass).
        """
        r = self.rates(C, params, condition_arrays, loc_idx)
        if rate_scale is not None:
            r = r * rate_scale
        if stoich is None:
            stoich = self.compute_stoich(params)
        R = stoich.T @ r
        if self.positivity_threshold is not None:
            R = self._apply_positivity_limiter(R, C)
        return R

    def _apply_positivity_limiter(self, R: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
        """Throttle net consumption as a species approaches zero.

            R_lim = max(R, 0) + min(R, 0) * C / max(C, threshold)

        Positive (production) terms pass through unchanged; negative
        (consumption) terms are scaled by ``C / max(C, threshold)``, which is
        1 well away from zero and decays to 0 as ``C`` falls below
        ``threshold`` — preventing the state from crossing into negative
        values and removing the associated stiffness. Smooth enough for AD
        (uses only ``maximum``/``minimum``).
        """
        thr = self.positivity_threshold
        pos = jnp.maximum(R, 0.0)
        neg = jnp.minimum(R, 0.0)
        scale = C / jnp.maximum(C, thr)
        return pos + neg * scale

    # ----- Species / model metadata accessors ---------------------------------

    def units_of(self, species: str) -> str:
        """Return the declared concentration units of a species.

        Parameters
        ----------
        species : str
            Species name.

        Returns
        -------
        str
            The units string from the YAML ``species:`` block (e.g.
            ``"g_COD/m3"``, ``"g_N/m3"``, ``"mol/L"``).

        Raises
        ------
        KeyError
            If ``species`` is not a declared species.
        """
        if species not in self.species_index:
            raise KeyError(f"Unknown species '{species}'. Available: {self.species}")
        return self.species_units.get(species, "")

    def description_of(self, species: str) -> str:
        """Return the human-readable description of a species.

        Parameters
        ----------
        species : str
            Species name.

        Returns
        -------
        str
            The description string from the YAML ``species:`` block (``""`` if
            none was declared).

        Raises
        ------
        KeyError
            If ``species`` is not a declared species.
        """
        if species not in self.species_index:
            raise KeyError(f"Unknown species '{species}'. Available: {self.species}")
        return self.species_descriptions.get(species, "")

    @property
    def time_unit(self) -> str | None:
        """Integration time unit, inferred from the rate-constant units.

        aquakin has **no global time unit**: ``t_span`` and ``t_eval`` are
        interpreted in whatever time unit the model's rate constants are
        written in, and that differs by model. The chemistry models (ozone,
        UV/H₂O₂) use **seconds** (rate constants in ``M-1 s-1``); the biological
        models (ASM1/2d/3, ADM1, WATS) use **days** (``1/d``). So
        ``reactor.solve(C0, t_span=(0, 600))`` integrates 600 *seconds* for
        ``ozone_bromate`` but 600 *days* for ``asm1`` — same code, no warning.

        This property recovers that unit by parsing the declared parameter
        ``units:`` strings and reading the inverse-time token the rate constants
        share (the ``s`` in ``M-1 s-1``, the ``d`` in ``1/d``), so a caller can
        check it before choosing a ``t_span``.

        Returns
        -------
        str or None
            The shared inverse-time token (``"s"``, ``"d"``, ``"h"`` or
            ``"min"``), or ``None`` when it cannot be determined unambiguously —
            no parameter declares a time unit, or different rate constants
            disagree.
        """
        from aquakin.utils.units import _TIME_TOKENS, parse_units

        found = set()
        for unit in self.parameter_units.values():
            dim = parse_units(unit)
            if dim is None:
                continue
            found |= {tok for tok, exp in dim.tokens if tok in _TIME_TOKENS and exp < 0}
        if len(found) == 1:
            return next(iter(found))
        return None

    # ----- Advisory / introspection surface -----------------------------------
    #
    # Presentation (summary / to_latex) and scientific audit (dimensional
    # consistency, COD/N/P/S conservation, nitrogen balance) delegated to
    # ``core.introspect`` free functions so this dataclass stays focused on the
    # differentiable hot path. These thin methods are the public API; the module
    # keeps the lazy ``utils.*`` import boundary.

    def summary(self) -> str:
        """Return a human-readable table summarising the model."""
        from aquakin.core import introspect

        return introspect.format_model_summary(self)

    def to_latex(self) -> dict[str, str]:
        """Return a mapping ``reaction_name -> LaTeX rate expression``."""
        from aquakin.core import introspect

        return introspect.model_to_latex(self)

    def check_units(self, *, check_root: bool = True) -> list:
        """Check the rate expressions for dimensional ("unit") consistency.

        A currency-aware dimensional analysis of every ``rate:`` expression: it
        catches a dropped concentration factor, a wrong rate-constant exponent,
        or a Monod term that compares two different "currencies"
        (``g_COD/m3`` vs ``g_N/m3``), which a plain SI dimension check waves
        through because both are mass/volume. Units are taken from the species,
        parameter, and condition ``units:`` declarations.

        The check is **advisory**: a blank or unparseable unit is treated as
        unknown and skipped, so an empty result is "no inconsistency among the
        declared, parseable units", not a proof of correctness. Stoichiometry
        (a conservation question) is out of scope -- use
        :func:`aquakin.check_conservation` for that.

        Parameters
        ----------
        check_root : bool, default True
            Also assert each rate resolves to ``currency / volume / time`` (e.g.
            ``g_COD/m3/d`` or ``mol/L/s``). Set ``False`` to run only the local
            operand- and Monod-consistency rules.

        Returns
        -------
        list of aquakin.utils.units.UnitWarning
            One entry per finding, as ``(reaction, location, detail)`` named
            tuples (empty when nothing is flagged).

        Examples
        --------
        >>> net = aquakin.load_model("asm1")
        >>> for w in net.check_units():
        ...     print(w)
        """
        from aquakin.core import introspect

        return introspect.check_units(self, check_root=check_root)

    def composition(
        self, *, params=None, electron_acceptor_cod: bool = True
    ) -> dict[str, dict[str, float]]:
        """The per-species conserved-quantity content table for this model.

        Returns ``{species: {quantity: content}}`` -- the content of each
        conserved quantity (``COD`` / ``N`` / ``P`` / ``S`` / ``Fe`` / ...) per
        unit of the species' own measure. This is the table a conservation check
        dots against the stoichiometry (:meth:`check_conservation`) and that a
        results-level balance dots against concentrations.

        Resolution order:

        1. the model's own declared ``species[].composition`` metadata, if any;
        2. otherwise the shipped role-based table
           (:func:`aquakin.composition_table`) for the ASM / ADM families, which
           reads the model's composition *parameters* (so a calibrated N / P
           fraction flows through);
        3. otherwise an empty table (no metadata available).

        Parameters
        ----------
        params : array-like, optional
            Parameter vector passed to the shipped role-based fallback so the
            table tracks a calibrated / run composition. Ignored when the model
            declares its own (literal) ``composition:`` metadata.
        electron_acceptor_cod : bool, default True
            Passed to the shipped role-based fallback (``False`` selects the
            lab-COD convention; see :func:`aquakin.composition_table`). Ignored
            for declared metadata.
        """
        from aquakin.core import introspect

        return introspect.model_composition(
            self, params=params, electron_acceptor_cod=electron_acceptor_cod
        )

    def check_conservation(
        self,
        *,
        tol: float = 1e-2,
        params=None,
        quantities=None,
        composition=None,
        electron_acceptor_cod: bool = True,
    ) -> list:
        """Conservation violations ``(reaction, quantity, residual)`` above ``tol``.

        Dots each reaction's stoichiometry against the per-species
        :meth:`composition` table; a quantity whose stoichiometry-weighted content
        does not sum to zero (beyond ``tol``) is reported. Catches the
        conservation-determined-coefficient errors that are otherwise easy to miss
        -- a wrong electron-acceptor (O2 / NO3) demand breaks the COD balance, a
        wrong product split breaks an elemental (S / N / P / Fe) balance.

        **Advisory and opt-in**, like :meth:`check_units`: it never runs at load
        and never raises on a violation -- it returns the list for you to inspect.
        Restrict to specific ``quantities`` (e.g. ``["COD"]``) if desired, or pass
        an explicit ``composition`` to override the model's table.

        Raises ``ValueError`` only if no composition table is available (the
        model declares none and there is no shipped fallback).
        """
        from aquakin.core import introspect

        return introspect.check_conservation(
            self,
            tol=tol,
            params=params,
            quantities=quantities,
            composition=composition,
            electron_acceptor_cod=electron_acceptor_cod,
        )

    def check_nitrogen(
        self,
        *,
        tol: float = 1e-2,
        params=None,
        composition=None,
        nitrate: str = "S_NO",
        n_key: str = "N",
    ) -> list:
        """Nitrogen-balance violations ``(reaction, residual)`` above ``tol``.

        The nitrogen analogue of :meth:`check_conservation`, accounting for the
        nitrate reduced to (untracked) N2 gas: a reaction conserves nitrogen when
        its tracked-species N content plus the gassed-off nitrate is zero. Exact
        for both nitrification (no nitrate consumed) and denitrification. Uses the
        model's :meth:`composition` table unless ``composition`` is passed.
        """
        from aquakin.core import introspect

        return introspect.check_nitrogen(
            self, tol=tol, params=params, composition=composition, nitrate=nitrate, n_key=n_key
        )


# --- compile_model stages --------------------------------------------


def _compile_speciation(spec, species_index, declared_conditions):
    """Wire an optional ``speciation:`` block (state-derived pH).

    Returns ``(derived_condition_fn, derived_fields, condition_fields)`` where
    ``condition_fields`` is ``declared_conditions`` plus the produced field (so
    rate-expression validation sees it).
    """
    derived_condition_fn = None
    derived_fields: list[str] = []
    condition_fields = declared_conditions
    speciation_cfg = getattr(spec, "speciation", None)
    if speciation_cfg is None:
        return derived_condition_fn, derived_fields, condition_fields

    from aquakin.core.speciation import build_ph_derived_fn

    # Accept either a plain dict or a Pydantic model (duck-typed).
    cfg = speciation_cfg if isinstance(speciation_cfg, dict) else speciation_cfg.model_dump()
    derived_condition_fn, produced_fields, required_fields = build_ph_derived_fn(cfg, species_index)
    missing = sorted(required_fields - declared_conditions)
    if missing:
        raise ValueError(
            f"speciation block reads condition field(s) {missing} that are "
            f"not declared in the model's 'conditions:' block."
        )
    clash = sorted(set(produced_fields) & declared_conditions)
    if clash:
        raise ValueError(
            f"speciation produces condition field(s) {clash}, which must not "
            f"also be declared in 'conditions:' (they are computed, not supplied)."
        )
    derived_fields = list(produced_fields)
    condition_fields = declared_conditions | set(produced_fields)
    return derived_condition_fn, derived_fields, condition_fields


# Shared empty state / condition placeholders for a dynamic stoichiometric
# coefficient's ``(params,)``-only callable: the coefficient reads only ``params``
# (no species, conditions or location), so it is evaluated with these dummies.
# Module-level constants rather than rebuilt each loop iteration.
_EMPTY_C = jnp.zeros(0)
_EMPTY_COND: dict[str, jnp.ndarray] = {}


def _params_only_callable(inner):
    """Adapt a full rate callable to a ``(params,)``-only callable.

    A dynamic stoichiometric coefficient is compiled as an ordinary rate AST but
    reads only ``params``; this wraps it to be called with just the parameter
    vector, supplying the shared empty state/conditions. A module-level factory
    (rather than a closure defined in the compile loop) so the captured ``inner``
    binds per call — no late-binding-over-loop-variable trap and no default-arg
    trick to work around one.
    """

    def _params_only(p):
        return inner(_EMPTY_C, p, _EMPTY_COND, 0)

    return _params_only


def _merge_derived(conditions: dict, produced: dict) -> dict:
    """Return ``conditions`` augmented with each ``produced`` derived scalar,
    broadcast across the spatial-location axis (the originals untouched).

    A derived field is a per-location scalar computed from the local state; it is
    broadcast to the conditions' shape so the existing
    ``condition_arrays[name][loc_idx]`` indexing (used by
    :class:`~aquakin.core.nodes.ConditionNode`) returns it unchanged. The one
    place this broadcast-merge lives, shared by :meth:`CompiledModel._augment_conditions`
    and :func:`_compose_derived`.
    """
    shape = jnp.shape(next(iter(conditions.values()))) if conditions else (1,)
    merged = dict(conditions)
    for name, value in produced.items():
        merged[name] = jnp.broadcast_to(jnp.asarray(value), shape)
    return merged


def _compose_derived(first, second):
    """Compose two derived-condition functions: run ``first``, inject its outputs
    into the condition arrays, then run ``second`` on the augmented conditions;
    return both outputs merged. This lets the precipitation saturation-index
    computation read the pH that the speciation block produced in the same RHS
    call (they share one ``derived_condition_fn`` slot)."""

    def composed(C, params, conditions, loc_idx):
        out1 = first(C, params, conditions, loc_idx)
        merged = _merge_derived(conditions, out1)
        out2 = second(C, params, merged, loc_idx)
        return {**out1, **out2}

    return composed


def _compile_precipitation(spec, species_index, condition_fields, derived_fields, speciation_fn):
    """Wire an optional ``precipitation:`` block (SI-driven mineral precipitation).

    Composes after the speciation pH: the precipitation derived-fn needs the
    system pH, which may be a declared condition OR produced by ``speciation:``.
    Returns ``(derived_condition_fn, derived_fields, condition_fields)`` with the
    valid condition set extended by each mineral's ``SI_<name>`` / ``R_<name>``.
    """
    precip_cfg = getattr(spec, "precipitation", None)
    if precip_cfg is None:
        return speciation_fn, derived_fields, condition_fields, None

    from aquakin.core.precipitation import build_precipitation_derived_fn
    from aquakin.core.precipitation_equilibrium import (
        build_precipitation_equilibrium_derived_fn,
    )

    cfg = precip_cfg if isinstance(precip_cfg, dict) else precip_cfg.model_dump()

    fn = speciation_fn
    new_fields = list(derived_fields)
    cond = condition_fields
    equilibrium_project = None

    def _add(stage_fn, produced, required, what):
        nonlocal fn, new_fields, cond
        missing = sorted(required - cond)
        if missing:
            raise ValueError(
                f"precipitation ({what}) reads condition field(s) {missing} that "
                f"are neither declared in 'conditions:' nor produced by "
                f"'speciation:'."
            )
        clash = sorted(set(produced) & cond)
        if clash:
            raise ValueError(
                f"precipitation ({what}) produces field(s) {clash} that collide "
                f"with declared conditions or other derived fields."
            )
        fn = stage_fn if fn is None else _compose_derived(fn, stage_fn)
        new_fields = new_fields + list(produced)
        cond = cond | set(produced)

    # Kinetic minerals -> SI_/R_ (skips equilibrium-mode minerals).
    kin_fn, kin_produced, kin_required = build_precipitation_derived_fn(cfg, species_index)
    if kin_produced:
        _add(kin_fn, kin_produced, kin_required, "kinetic")

    # Equilibrium-mode minerals -> Xeq_ (the algebraic equilibrium amount), plus
    # a projection fn that snaps a composition onto the precipitation equilibrium.
    eq = build_precipitation_equilibrium_derived_fn(cfg, species_index)
    if eq is not None:
        eq_fn, eq_produced, eq_required, equilibrium_project = eq
        _add(eq_fn, eq_produced, eq_required, "equilibrium")

    return fn, new_fields, cond, equilibrium_project


def _build_param_index(spec):
    """Build the flat parameter index by walking model-level then
    reaction-local parameters in declaration order (so ordering is
    deterministic).

    Returns ``(parameters, param_index, defaults, bounds, transforms, priors,
    units, temperature_corrections)``.
    """
    parameters: list[str] = []
    param_index: dict[str, int] = {}
    defaults: list[float] = []
    bounds: dict[str, tuple[float, float]] = {}
    transforms: dict[str, str] = {}
    priors: dict[str, tuple[float, float]] = {}
    units: dict[str, str] = {}
    temperature: list = []

    def record(key: str, pspec) -> None:
        idx = len(parameters)
        param_index[key] = idx
        parameters.append(key)
        defaults.append(float(pspec.value))
        if pspec.bounds is not None:
            bounds[key] = (float(pspec.bounds[0]), float(pspec.bounds[1]))
        transforms[key] = pspec.transform
        units[key] = getattr(pspec, "units", "") or ""
        prior = getattr(pspec, "prior", None)
        if prior is not None:
            priors[key] = prior.resolved()
        tc = getattr(pspec, "temperature", None)
        if tc is not None:
            temperature.append((idx, math.log(float(tc.theta)), float(tc.ref_T), tc.condition))

    for local_name, pspec in getattr(spec, "parameters", {}).items():
        record(local_name, pspec)
    for rxn in spec.reactions:
        for local_name, pspec in rxn.parameters.items():
            record(f"{rxn.name}.{local_name}", pspec)
    return (parameters, param_index, defaults, bounds, transforms, priors, units, temperature)


def _resolve_expressions(spec) -> dict[str, ASTNode]:
    """Parse the model's named expressions, topologically sort them by
    inter-expression references, and inline them so each resolved AST contains
    only leaf / parameter / species / condition references.
    """
    raw_expressions = getattr(spec, "expressions", {})
    expression_asts_raw: dict[str, ASTNode] = {}
    for name, formula in raw_expressions.items():
        try:
            expression_asts_raw[name] = parse_rate_expression(formula)
        except Exception as exc:
            raise ValueError(f"Failed to parse named expression '{name}': {exc}") from exc

    # (An expression name colliding with a parameter name is already rejected by
    # the schema validator in schema/model_spec.py, so a collision never
    # reaches the silent ``_substitute`` shadowing path.)
    expr_names = list(expression_asts_raw.keys())
    expr_name_set = set(expr_names)
    expr_deps: dict[str, set[str]] = {}
    for name, ast in expression_asts_raw.items():
        refs = _collect_param_refs(ast) & expr_name_set
        refs.discard(name)  # self-loops caught below
        expr_deps[name] = refs

    expression_asts: dict[str, ASTNode] = {}
    for name in _topo_sort_expressions(expr_names, expr_deps):
        raw = expression_asts_raw[name]
        if name in _collect_param_refs(raw):
            raise ValueError(f"Named expression '{name}' references itself.")
        expression_asts[name] = _substitute(raw, expression_asts)
    return expression_asts


def _collect_species_refs(node: ASTNode) -> set[str]:
    """Every ``SpeciesNode`` name in ``ast``, via ``children()`` (so a new node
    type cannot hide a species reference)."""
    if isinstance(node, SpeciesNode):
        return {node.name}
    refs: set[str] = set()
    for child in node.children():
        refs |= _collect_species_refs(child)
    return refs


def _validate_expression_refs(expression_asts, species_index, condition_fields):
    """Validate every named expression's species / condition references.

    A reaction's rate is validated against the species / condition / parameter
    indices when it is compiled, which also catches bad references inside the
    expressions it inlines -- but an expression that **no reaction consumes** is
    never compiled, so a typo in it would otherwise load silently. Check the
    species and condition references of *every* expression here.

    Parameter references are deliberately not checked: an expression's bare
    parameter resolves against the *consuming* reaction's namespace (reaction-
    local, then model-level), so it has no well-defined meaning standalone.
    """
    for name, ast in expression_asts.items():
        for sp in sorted(_collect_species_refs(ast)):
            if sp not in species_index:
                raise KeyError(
                    f"Named expression '{name}' references undeclared species '{sp}'."
                    f"{did_you_mean(sp, species_index)}"
                )
        for cond in sorted(ast.condition_names()):
            if cond not in condition_fields:
                raise ValueError(
                    f"Named expression '{name}' references unknown condition field '{cond}'."
                    f"{did_you_mean(cond, condition_fields)}"
                )


def _unresolved_params(ast: ASTNode, rxn_name: str, param_index: dict) -> list[str]:
    """ParamNode names in ``ast`` resolving to neither a reaction-local
    (``<rxn>.<name>``) nor a model-level parameter."""
    return [
        p
        for p in ast.param_names()
        if f"{rxn_name}.{p}" not in param_index and p not in param_index
    ]


def _compile_reaction(rxn, species_index, param_index, condition_fields, expression_asts):
    """Compile one reaction.

    Returns ``(static_coeffs, dynamic_entries, rate_callable, rate_ast)`` where
    ``static_coeffs`` is ``[(species_col, coef)]`` and ``dynamic_entries`` is
    ``[(species_col, params_callable)]`` for parameter-expression coefficients.
    """
    static_coeffs: list[tuple[int, float]] = []
    dynamic_entries: list[tuple[int, Any]] = []

    for sp_name, coef in rxn.stoichiometry.items():
        if sp_name not in species_index:
            raise KeyError(
                f"Reaction '{rxn.name}' references undeclared species '{sp_name}' "
                f"in stoichiometry.{did_you_mean(sp_name, species_index)}"
            )
        j = species_index[sp_name]
        if isinstance(coef, (int, float)):
            static_coeffs.append((j, float(coef)))
            continue
        # Coefficient is a parameter expression: parse, validate, resolve, compile.
        try:
            raw_stoich_ast = parse_rate_expression(coef)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse stoichiometric coefficient '{rxn.name}'/{sp_name!r}: {exc}"
            ) from exc
        _validate_stoich_ast(raw_stoich_ast, rxn.name, sp_name)
        bad = _unresolved_params(raw_stoich_ast, rxn.name, param_index)
        if bad:
            raise KeyError(
                f"Stoichiometric coefficient '{rxn.name}'/{sp_name!r} "
                f"references identifier '{bad[0]}' which is not a "
                f"reaction-local nor model-level parameter."
                f"{did_you_mean(bad[0], param_index)}"
            )
        stoich_ctx = CompileContext(
            species_index={},  # species not allowed
            param_index=param_index,
            condition_fields=frozenset(),  # conditions not allowed
            reaction_name=rxn.name,
        )
        inner = raw_stoich_ast.compile(stoich_ctx)
        dynamic_entries.append((j, _params_only_callable(inner)))

    # Rate expression: inline named-expression references, then validate refs.
    ast = _substitute(parse_rate_expression(rxn.rate), expression_asts)
    for sp in ast.species():
        if sp not in species_index:
            raise KeyError(
                f"Reaction '{rxn.name}' rate expression references undeclared "
                f"species '{sp}'.{did_you_mean(sp, species_index)}"
            )
    for cf in ast.condition_names():
        if cf not in condition_fields:
            raise KeyError(
                f"Reaction '{rxn.name}' rate expression references undeclared "
                f"condition '{cf}'.{did_you_mean(cf, condition_fields)}"
            )
    bad = _unresolved_params(ast, rxn.name, param_index)
    if bad:
        raise KeyError(
            f"Reaction '{rxn.name}' rate expression references identifier "
            f"'{bad[0]}' which is not a reaction-local parameter, a model-level "
            f"parameter, or a named expression.{did_you_mean(bad[0], param_index)}"
        )
    ctx = CompileContext(
        species_index=species_index,
        param_index=param_index,
        condition_fields=condition_fields,
        reaction_name=rxn.name,
    )
    return static_coeffs, dynamic_entries, ast.compile(ctx), ast


def compile_model(spec: "Any") -> CompiledModel:
    """
    Build a :class:`CompiledModel` from a validated :class:`ModelSpec`.

    Parameters
    ----------
    spec : ModelSpec
        A Pydantic-validated model specification.

    Returns
    -------
    CompiledModel
    """
    species_names = [s.name for s in spec.species]
    species_index = {name: i for i, name in enumerate(species_names)}
    declared_conditions = frozenset(c.name for c in spec.conditions)

    # Stage 1: optional state-derived-pH speciation (extends the valid
    # condition-field set with any produced field).
    derived_condition_fn, derived_fields, condition_fields = _compile_speciation(
        spec, species_index, declared_conditions
    )
    # Stage 1b: optional mineral precipitation, composed after the speciation pH
    # (it adds each mineral's SI_<name> / R_<name> to the valid condition set).
    (derived_condition_fn, derived_fields, condition_fields, precipitation_equilibrium_fn) = (
        _compile_precipitation(
            spec, species_index, condition_fields, derived_fields, derived_condition_fn
        )
    )

    # Stage 2: the flat parameter index (model-level then reaction-local).
    (
        parameters,
        param_index,
        parameter_defaults,
        parameter_bounds,
        parameter_transforms,
        parameter_priors,
        parameter_units,
        temperature_corrections,
    ) = _build_param_index(spec)

    # Stage 3: parse + topo-sort + inline the named expressions, and validate the
    # species/condition references of every one (even those no reaction consumes).
    expression_asts = _resolve_expressions(spec)
    _validate_expression_refs(expression_asts, species_index, condition_fields)

    # Stage 3b: per-species conserved-quantity content (declared `composition:`),
    # then resolve any `auto`/`?` stoichiometric coefficients from the declared
    # conservation laws (issue #291), so the stoichiometry read in Stage 4 is fully
    # numeric. Mutates spec.reactions in place; a no-op when no reaction uses `auto`.
    species_composition = {
        s.name: {q: float(v) for q, v in s.composition.items()}
        for s in spec.species
        if getattr(s, "composition", None)
    }
    resolve_auto_coefficients(
        spec.reactions, species_composition, getattr(spec, "conserved_for", None)
    )

    # Stage 4: compile each reaction's stoichiometry + rate.
    n_species = len(species_names)
    n_reactions = len(spec.reactions)
    stoich_np = np.zeros((n_reactions, n_species), dtype=np.float64)
    stoich_dynamic: list[tuple[int, int, Any]] = []
    reaction_names: list[str] = []
    rate_callables: list[RateCallable] = []
    rate_asts: list[ASTNode] = []
    for i, rxn in enumerate(spec.reactions):
        reaction_names.append(rxn.name)
        static_coeffs, dynamic_entries, rate_callable, ast = _compile_reaction(
            rxn, species_index, param_index, condition_fields, expression_asts
        )
        for j, coef in static_coeffs:
            stoich_np[i, j] = coef
        for j, fn in dynamic_entries:
            stoich_dynamic.append((i, j, fn))
        rate_callables.append(rate_callable)
        rate_asts.append(ast)

    default_concentrations = jnp.asarray([float(s.default_concentration) for s in spec.species])
    default_parameters = jnp.asarray(parameter_defaults)

    # Per-species metadata carried through to the runtime model and results.
    # Units are prettified for display (plain-ASCII ``m3`` -> ``m³``); the YAML
    # keeps the easy-to-type ASCII form.
    species_units = {s.name: prettify_units(s.units) for s in spec.species}
    species_descriptions = {s.name: s.description for s in spec.species}
    # species_composition was built in Stage 3b (above) for the auto-coefficient
    # resolver; it is carried verbatim onto the runtime model here.

    # conditions_required = declared conditions; reactors validate runtime
    # SpatialConditions against this list.
    conditions_required = [c.name for c in spec.conditions]
    condition_defaults = {c.name: float(c.default) for c in spec.conditions}
    # Advisory units per condition field (blank when undeclared), for check_units.
    condition_units = {c.name: getattr(c, "units", "") or "" for c in spec.conditions}

    # --- Optional positivity limiter --------------------------------
    positivity_threshold = None
    limiter_cfg = getattr(spec, "positivity_limiter", None)
    if limiter_cfg is not None:
        positivity_threshold = float(
            limiter_cfg["threshold"] if isinstance(limiter_cfg, dict) else limiter_cfg.threshold
        )

    stoich = jnp.asarray(stoich_np)
    if stoich_dynamic:
        dyn_rows = jnp.asarray([i for (i, _, _) in stoich_dynamic])
        dyn_cols = jnp.asarray([j for (_, j, _) in stoich_dynamic])
    else:
        dyn_rows = None
        dyn_cols = None

    return CompiledModel(
        name=spec.model.name,
        description=spec.model.description or "",
        references=list(spec.model.references or []),
        species=species_names,
        parameters=parameters,
        conditions_required=conditions_required,
        stoich_matrix=stoich,
        reaction_names=reaction_names,
        rate_callables=rate_callables,
        rate_asts=rate_asts,
        param_index=param_index,
        species_index=species_index,
        _default_concentrations=default_concentrations,
        _default_parameters=default_parameters,
        _condition_defaults=condition_defaults,
        species_units=species_units,
        species_descriptions=species_descriptions,
        species_composition=species_composition,
        parameter_units=parameter_units,
        condition_units=condition_units,
        parameter_bounds=parameter_bounds,
        parameter_transforms=parameter_transforms,
        parameter_priors=parameter_priors,
        stoich_dynamic=stoich_dynamic,
        _stoich_dynamic_rows=dyn_rows,
        _stoich_dynamic_cols=dyn_cols,
        derived_condition_fn=derived_condition_fn,
        derived_fields=derived_fields,
        precipitation_equilibrium_fn=precipitation_equilibrium_fn,
        positivity_threshold=positivity_threshold,
        clip_negative_states=bool(getattr(spec, "clip_negative_states", False)),
        temperature_corrections=temperature_corrections,
    )
