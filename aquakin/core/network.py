"""Runtime-compiled reaction network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from aquakin.core.context import CompileContext
from aquakin.core.nodes import (
    AddNode,
    ArrheniusNode,
    ASTNode,
    ConditionNode,
    ConstantNode,
    DivideNode,
    MonodInhibitionNode,
    MonodInhibitionRatioNode,
    MonodNode,
    MonodRatioNode,
    MultiplyNode,
    NegateNode,
    ParamNode,
    PowerNode,
    RateCallable,
    SpeciesNode,
    SubtractNode,
    pHInhibitNode,
    pHSwitchNode,
    _BinaryNode,
)
from aquakin.core.parser import parse_rate_expression


# Used to detect references-to-other-expressions during AST inspection.
_LEAF_TYPES = (ConstantNode, SpeciesNode, ConditionNode, ParamNode)


# Stoichiometry-coefficient expressions may reference only constants,
# parameters, and arithmetic / negation. Species, conditions, named
# expressions, and domain functions are forbidden — stoichiometry must be
# state-independent so we can evaluate it once per ``solve`` call.
_ALLOWED_STOICH_NODES = (ConstantNode, ParamNode, NegateNode, _BinaryNode)


def _validate_stoich_ast(ast: ASTNode, rxn_name: str, species: str) -> None:
    """Reject stoich ASTs that reference state, conditions, or functions."""
    if isinstance(ast, _ALLOWED_STOICH_NODES):
        if isinstance(ast, NegateNode):
            _validate_stoich_ast(ast.operand, rxn_name, species)
        elif isinstance(ast, _BinaryNode):
            _validate_stoich_ast(ast.left, rxn_name, species)
            _validate_stoich_ast(ast.right, rxn_name, species)
        return
    kind = type(ast).__name__
    raise ValueError(
        f"Stoichiometric coefficient for '{rxn_name}' / '{species}' uses an "
        f"unsupported expression element ({kind}). Stoich expressions may only "
        f"reference parameters, numeric constants, and arithmetic / negation; "
        f"species, conditions, named expressions, and domain functions "
        f"(arrhenius, pH_switch, monod, ...) are not allowed."
    )


def _collect_param_refs(node: ASTNode) -> set[str]:
    """Walk an AST and return every ParamNode name encountered."""
    if isinstance(node, ParamNode):
        return {node.name}
    if isinstance(node, _BinaryNode):
        return _collect_param_refs(node.left) | _collect_param_refs(node.right)
    if isinstance(node, NegateNode):
        return _collect_param_refs(node.operand)
    if isinstance(node, ArrheniusNode):
        return _collect_param_refs(node.A) | _collect_param_refs(node.Ea)
    if isinstance(node, pHSwitchNode):
        return _collect_param_refs(node.pKa)
    if isinstance(node, pHInhibitNode):
        return _collect_param_refs(node.pH_LL) | _collect_param_refs(node.pH_UL)
    if isinstance(node, (MonodNode, MonodInhibitionNode)):
        return _collect_param_refs(node.X) | _collect_param_refs(node.K)
    if isinstance(node, (MonodRatioNode, MonodInhibitionRatioNode)):
        return (
            _collect_param_refs(node.A)
            | _collect_param_refs(node.B)
            | _collect_param_refs(node.K)
        )
    return set()  # ConstantNode, SpeciesNode, ConditionNode


def _substitute(node: ASTNode, expr_asts: dict[str, ASTNode]) -> ASTNode:
    """Return a new AST with ParamNode references to named expressions
    replaced by the corresponding (already-resolved) expression AST.
    """
    if isinstance(node, ParamNode):
        if node.name in expr_asts:
            return expr_asts[node.name]
        return node
    if isinstance(node, _BinaryNode):
        new_left = _substitute(node.left, expr_asts)
        new_right = _substitute(node.right, expr_asts)
        if new_left is node.left and new_right is node.right:
            return node
        return type(node)(new_left, new_right)
    if isinstance(node, NegateNode):
        new_op = _substitute(node.operand, expr_asts)
        if new_op is node.operand:
            return node
        return NegateNode(new_op)
    if isinstance(node, ArrheniusNode):
        new_A = _substitute(node.A, expr_asts)
        new_Ea = _substitute(node.Ea, expr_asts)
        if new_A is node.A and new_Ea is node.Ea:
            return node
        return ArrheniusNode(new_A, new_Ea)
    if isinstance(node, pHSwitchNode):
        new_pKa = _substitute(node.pKa, expr_asts)
        if new_pKa is node.pKa:
            return node
        return pHSwitchNode(new_pKa)
    if isinstance(node, pHInhibitNode):
        new_ll = _substitute(node.pH_LL, expr_asts)
        new_ul = _substitute(node.pH_UL, expr_asts)
        if new_ll is node.pH_LL and new_ul is node.pH_UL:
            return node
        return pHInhibitNode(new_ll, new_ul)
    if isinstance(node, (MonodNode, MonodInhibitionNode)):
        new_X = _substitute(node.X, expr_asts)
        new_K = _substitute(node.K, expr_asts)
        if new_X is node.X and new_K is node.K:
            return node
        return type(node)(new_X, new_K)
    if isinstance(node, (MonodRatioNode, MonodInhibitionRatioNode)):
        new_A = _substitute(node.A, expr_asts)
        new_B = _substitute(node.B, expr_asts)
        new_K = _substitute(node.K, expr_asts)
        if new_A is node.A and new_B is node.B and new_K is node.K:
            return node
        return type(node)(new_A, new_B, new_K)
    return node  # leaves we don't rewrite


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
            cycle = stack[stack.index(name):] + [name]
            raise ValueError(
                f"Cycle in named expressions: {' -> '.join(cycle)}"
            )
        visited[name] = "visiting"
        for dep in expr_deps.get(name, set()):
            _visit(dep, stack + [name])
        visited[name] = "done"
        order.append(name)

    for name in expr_names:
        _visit(name, [])
    return order

if TYPE_CHECKING:  # pragma: no cover
    from aquakin.schema.network_spec import NetworkSpec


@dataclass
class CompiledNetwork:
    """
    Runtime representation of a reaction network.

    Attributes
    ----------
    name : str
        Network identifier (e.g. ``"ozone_bromate"``).
    description : str
        Free-text description of the network.
    references : list[str]
        Literature citations associated with the network.
    species : list[str]
        Ordered species names. Index in this list is the index used in ``C``.
    parameters : list[str]
        Ordered, namespaced parameter names (e.g. ``"O3_Br_direct.k1"``).
        Index in this list is the index used in ``params``.
    conditions_required : list[str]
        Names of condition fields the network needs at runtime.
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
    # stays exact. This mirrors the reference IWA/BSM S-function convention
    # (``xtemp = max(x, 0)`` before the process rates). Identity at feasible
    # (non-negative) states, so it does not change the physical solution.
    clip_negative_states: bool = False

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_names)

    @property
    def n_params(self) -> int:
        return len(self.parameters)

    def default_concentrations(self) -> jnp.ndarray:
        """Return a copy of the default initial-concentration vector."""
        return jnp.asarray(self._default_concentrations)

    def default_parameters(self) -> jnp.ndarray:
        """Return a copy of the default parameter vector."""
        return jnp.asarray(self._default_parameters)

    def default_conditions(self, n_locations: int = 1):
        """Build a :class:`SpatialConditions` from the network's declared defaults.

        Convenience for the common case of "use the defaults as written in
        the YAML". Each required condition is broadcast to ``n_locations``.
        """
        from aquakin.core.conditions import SpatialConditions

        return SpatialConditions.uniform(n_locations, **self._condition_defaults)

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
            condition_arrays = self._augment_conditions(
                C, params, condition_arrays, loc_idx
            )
        return jnp.stack(
            [f(C, params, condition_arrays, loc_idx) for f in self.rate_callables]
        )

    def _augment_conditions(
        self,
        C: jnp.ndarray,
        params: jnp.ndarray,
        condition_arrays: dict[str, jnp.ndarray],
        loc_idx,
    ) -> dict[str, jnp.ndarray]:
        """Merge state-derived condition fields into ``condition_arrays``.

        Each derived scalar is broadcast across all spatial locations so that
        the existing ``condition_arrays[name][loc_idx]`` indexing used by
        :class:`~aquakin.core.nodes.ConditionNode` returns it unchanged. The
        derived value is computed from the local state ``C`` (the state at
        ``loc_idx``), so the entry read at ``loc_idx`` is the correct one;
        other indices are never consulted within this call.
        """
        derived = self.derived_condition_fn(C, params, condition_arrays, loc_idx)
        if condition_arrays:
            template = next(iter(condition_arrays.values()))
            shape = jnp.shape(template)
        else:
            shape = (1,)
        merged = dict(condition_arrays)
        for name, value in derived.items():
            merged[name] = jnp.broadcast_to(jnp.asarray(value), shape)
        return merged

    def compute_stoich(self, params: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the stoichiometry matrix at the given parameter vector.

        For networks whose stoichiometry is purely numeric this returns
        the cached ``stoich_matrix`` unchanged. For networks with
        parameter-dependent coefficients (the dynamic entries listed in
        ``stoich_dynamic``), this scatters the per-call values onto the
        static base.

        Reactors typically call this once per ``solve()`` and hoist the
        result as a closure constant for the duration of the integration
        — see the rhs builders in :mod:`aquakin.integrate.batch` etc.
        """
        if not self.stoich_dynamic:
            return self.stoich_matrix
        values = jnp.stack(
            [fn(params) for (_, _, fn) in self.stoich_dynamic]
        )
        return self.stoich_matrix.at[
            self._stoich_dynamic_rows, self._stoich_dynamic_cols
        ].set(values)

    def dCdt(
        self,
        C: jnp.ndarray,
        params: jnp.ndarray,
        condition_arrays: dict[str, jnp.ndarray],
        loc_idx,
        *,
        stoich: "jnp.ndarray | None" = None,
    ) -> jnp.ndarray:
        """Return ``stoich.T @ rates(...)`` — the chemistry RHS.

        ``stoich`` may be precomputed via :meth:`compute_stoich` and passed
        in to avoid re-evaluating parameter-dependent coefficients on every
        ODE step. If omitted, it is computed from ``params`` here.
        """
        r = self.rates(C, params, condition_arrays, loc_idx)
        if stoich is None:
            stoich = self.compute_stoich(params)
        R = stoich.T @ r
        if self.positivity_threshold is not None:
            R = self._apply_positivity_limiter(R, C)
        return R

    def _apply_positivity_limiter(
        self, R: jnp.ndarray, C: jnp.ndarray
    ) -> jnp.ndarray:
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

    def summary(self) -> str:
        """Return a human-readable table summarising the network."""
        lines = [
            f"Network: {self.name}",
            f"  Description: {self.description}",
            f"  Species ({self.n_species}): {', '.join(self.species)}",
            f"  Conditions required: {', '.join(self.conditions_required) or '(none)'}",
            f"  Reactions ({self.n_reactions}):",
        ]
        for i, rname in enumerate(self.reaction_names):
            stoich_terms = []
            for j, sp in enumerate(self.species):
                coef = float(self.stoich_matrix[i, j])
                if coef == 0:
                    continue
                sign = "+" if coef > 0 else "-"
                mag = abs(coef)
                term = f"{sign} {mag:g} {sp}" if mag != 1 else f"{sign} {sp}"
                stoich_terms.append(term)
            lines.append(f"    [{i}] {rname}: " + " ".join(stoich_terms).lstrip("+ "))
        lines.append(f"  Parameters ({self.n_params}):")
        for p in self.parameters:
            bounds = self.parameter_bounds.get(p)
            bounds_s = f" bounds={bounds}" if bounds is not None else ""
            lines.append(f"    {p} = {float(self._default_parameters[self.param_index[p]]):g}{bounds_s}")
        if self.references:
            lines.append("  References:")
            for ref in self.references:
                lines.append(f"    - {ref}")
        return "\n".join(lines)

    def to_latex(self) -> dict[str, str]:
        """Return a mapping ``reaction_name -> LaTeX rate expression``."""
        from aquakin.utils.latex import to_latex as _to_latex

        return {
            name: _to_latex(ast)
            for name, ast in zip(self.reaction_names, self.rate_asts, strict=True)
        }


def compile_network(spec: "Any") -> CompiledNetwork:
    """
    Build a :class:`CompiledNetwork` from a validated :class:`NetworkSpec`.

    Parameters
    ----------
    spec : NetworkSpec
        A Pydantic-validated network specification.

    Returns
    -------
    CompiledNetwork
    """
    species_names = [s.name for s in spec.species]
    species_index = {name: i for i, name in enumerate(species_names)}

    declared_conditions = frozenset(c.name for c in spec.conditions)
    condition_fields = declared_conditions

    # --- Optional speciation / state-derived pH ---------------------
    derived_condition_fn = None
    derived_fields: list[str] = []
    speciation_cfg = getattr(spec, "speciation", None)
    if speciation_cfg is not None:
        from aquakin.core.speciation import build_ph_derived_fn

        # Accept either a plain dict or a Pydantic model (duck-typed).
        cfg = (
            speciation_cfg
            if isinstance(speciation_cfg, dict)
            else speciation_cfg.model_dump()
        )
        derived_condition_fn, produced_field, required_fields = build_ph_derived_fn(
            cfg, species_index
        )
        missing = sorted(required_fields - declared_conditions)
        if missing:
            raise ValueError(
                f"speciation block reads condition field(s) {missing} that are "
                f"not declared in the network's 'conditions:' block."
            )
        if produced_field in declared_conditions:
            raise ValueError(
                f"speciation produces condition field '{produced_field}', which "
                f"must not also be declared in 'conditions:' (it is computed, "
                f"not supplied)."
            )
        derived_fields = [produced_field]
        # Make the derived field visible to rate-expression validation.
        condition_fields = declared_conditions | {produced_field}

    # Build parameter index by walking reactions in declaration order so that
    # parameter ordering is deterministic.
    parameters: list[str] = []
    param_index: dict[str, int] = {}
    parameter_defaults: list[float] = []
    parameter_bounds: dict[str, tuple[float, float]] = {}
    parameter_transforms: dict[str, str] = {}
    parameter_priors: dict[str, tuple[float, float]] = {}

    def _record_param(key: str, pspec) -> None:
        param_index[key] = len(parameters)
        parameters.append(key)
        parameter_defaults.append(float(pspec.value))
        if pspec.bounds is not None:
            parameter_bounds[key] = (
                float(pspec.bounds[0]),
                float(pspec.bounds[1]),
            )
        parameter_transforms[key] = pspec.transform
        prior = getattr(pspec, "prior", None)
        if prior is not None:
            parameter_priors[key] = prior.resolved()

    # Network-level shared parameters first; reaction-local parameters next.
    for local_name, pspec in getattr(spec, "parameters", {}).items():
        _record_param(local_name, pspec)
    for rxn in spec.reactions:
        for local_name, pspec in rxn.parameters.items():
            _record_param(f"{rxn.name}.{local_name}", pspec)

    # --- Named expressions ------------------------------------------
    raw_expressions = getattr(spec, "expressions", {})
    expression_asts_raw: dict[str, ASTNode] = {}
    for name, formula in raw_expressions.items():
        try:
            expression_asts_raw[name] = parse_rate_expression(formula)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse named expression '{name}': {exc}"
            ) from exc

    # Dependency graph: expression -> set of other expressions it references.
    expr_names = list(expression_asts_raw.keys())
    expr_name_set = set(expr_names)
    expr_deps: dict[str, set[str]] = {}
    for name, ast in expression_asts_raw.items():
        refs = _collect_param_refs(ast) & expr_name_set
        refs.discard(name)  # don't count self-loops; they'll be caught below
        expr_deps[name] = refs

    order = _topo_sort_expressions(expr_names, expr_deps)

    # Resolve expressions in dependency order: substitute references to
    # already-resolved expressions inline. The resulting AST contains only
    # leaves and parameter / species / condition references.
    expression_asts: dict[str, ASTNode] = {}
    for name in order:
        raw = expression_asts_raw[name]
        # Self-reference (e.g. `rho = rho + 1`) would loop forever; the
        # topological sort treats it as a cycle on itself.
        if name in _collect_param_refs(raw):
            raise ValueError(
                f"Named expression '{name}' references itself."
            )
        expression_asts[name] = _substitute(raw, expression_asts)

    n_species = len(species_names)
    n_reactions = len(spec.reactions)

    stoich_np = np.zeros((n_reactions, n_species), dtype=np.float64)
    stoich_dynamic: list[tuple[int, int, Any]] = []
    reaction_names: list[str] = []
    rate_callables: list[RateCallable] = []
    rate_asts: list[ASTNode] = []

    for i, rxn in enumerate(spec.reactions):
        reaction_names.append(rxn.name)
        for sp_name, coef in rxn.stoichiometry.items():
            if sp_name not in species_index:
                raise KeyError(
                    f"Reaction '{rxn.name}' references undeclared species "
                    f"'{sp_name}' in stoichiometry."
                )
            j = species_index[sp_name]
            if isinstance(coef, (int, float)):
                # Static coefficient — fill into the base matrix directly.
                stoich_np[i, j] = float(coef)
                continue
            # Coefficient is an expression in parameters. Parse, validate,
            # resolve, and compile into a (params,) -> scalar callable.
            try:
                raw_stoich_ast = parse_rate_expression(coef)
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse stoichiometric coefficient "
                    f"'{rxn.name}'/{sp_name!r}: {exc}"
                ) from exc
            _validate_stoich_ast(raw_stoich_ast, rxn.name, sp_name)
            # Resolve param references the same way rate expressions do
            # (reaction-local first, then network-level).
            for pname in raw_stoich_ast.param_names():
                if f"{rxn.name}.{pname}" in param_index:
                    continue
                if pname in param_index:
                    continue
                raise KeyError(
                    f"Stoichiometric coefficient '{rxn.name}'/{sp_name!r} "
                    f"references identifier '{pname}' which is not a "
                    f"reaction-local nor network-level parameter."
                )
            stoich_ctx = CompileContext(
                species_index={},  # species not allowed
                param_index=param_index,
                condition_fields=frozenset(),  # conditions not allowed
                reaction_name=rxn.name,
            )
            inner = raw_stoich_ast.compile(stoich_ctx)
            # Wrap to a (params,)-only callable; pass dummy C/cond/loc.
            _empty_C = jnp.zeros(0)
            _empty_cond: dict[str, jnp.ndarray] = {}

            def _params_only(p, _f=inner):
                return _f(_empty_C, p, _empty_cond, 0)

            stoich_dynamic.append((i, j, _params_only))

        raw_ast = parse_rate_expression(rxn.rate)
        # Inline named-expression references first (substitute returns a new
        # AST containing only species/param/condition leaves).
        ast = _substitute(raw_ast, expression_asts)
        ctx = CompileContext(
            species_index=species_index,
            param_index=param_index,
            condition_fields=condition_fields,
            reaction_name=rxn.name,
        )
        # Validate species/conditions referenced in the resolved AST are declared.
        for sp in ast.species():
            if sp not in species_index:
                raise KeyError(
                    f"Reaction '{rxn.name}' rate expression references undeclared "
                    f"species '{sp}'."
                )
        for cf in ast.condition_names():
            if cf not in condition_fields:
                raise KeyError(
                    f"Reaction '{rxn.name}' rate expression references undeclared "
                    f"condition '{cf}'."
                )
        for pname in ast.param_names():
            # Resolution order: reaction-local first, then network-level.
            if f"{rxn.name}.{pname}" in param_index:
                continue
            if pname in param_index:
                continue
            raise KeyError(
                f"Reaction '{rxn.name}' rate expression references identifier "
                f"'{pname}' which is not a reaction-local parameter, a network-level "
                f"parameter, or a named expression."
            )
        rate_callables.append(ast.compile(ctx))
        rate_asts.append(ast)

    default_concentrations = jnp.asarray(
        [float(s.default_concentration) for s in spec.species]
    )
    default_parameters = jnp.asarray(parameter_defaults)

    # conditions_required = declared conditions; reactors validate runtime
    # SpatialConditions against this list.
    conditions_required = [c.name for c in spec.conditions]
    condition_defaults = {c.name: float(c.default) for c in spec.conditions}

    # --- Optional positivity limiter --------------------------------
    positivity_threshold = None
    limiter_cfg = getattr(spec, "positivity_limiter", None)
    if limiter_cfg is not None:
        positivity_threshold = float(
            limiter_cfg["threshold"]
            if isinstance(limiter_cfg, dict)
            else limiter_cfg.threshold
        )

    stoich = jnp.asarray(stoich_np)
    if stoich_dynamic:
        dyn_rows = jnp.asarray([i for (i, _, _) in stoich_dynamic])
        dyn_cols = jnp.asarray([j for (_, j, _) in stoich_dynamic])
    else:
        dyn_rows = None
        dyn_cols = None

    return CompiledNetwork(
        name=spec.network.name,
        description=spec.network.description or "",
        references=list(spec.network.references or []),
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
        parameter_bounds=parameter_bounds,
        parameter_transforms=parameter_transforms,
        parameter_priors=parameter_priors,
        stoich_dynamic=stoich_dynamic,
        _stoich_dynamic_rows=dyn_rows,
        _stoich_dynamic_cols=dyn_cols,
        derived_condition_fn=derived_condition_fn,
        derived_fields=derived_fields,
        positivity_threshold=positivity_threshold,
        clip_negative_states=bool(getattr(spec, "clip_negative_states", False)),
    )
