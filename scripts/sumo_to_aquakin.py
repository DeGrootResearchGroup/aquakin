"""Convert a SUMO-importer JSON model dump to aquakin YAML.

Reads the JSON format produced by WastewaterAD's ``wastewaterad.tools.sumo_import``
(``models/asm1_full.json`` is the canonical example) and emits an aquakin
network YAML using:

- network-level shared ``parameters:`` (every kinetic constant once)
- named ``expressions:`` for SUMO auxiliaries that reference state variables
- ``monod`` / ``monod_inh`` / ``monod_ratio`` helpers when the auxiliary
  matches the standard Monod patterns
- precomputed numeric ``stoichiometry:`` coefficients (yield, N-content,
  charge factors evaluated at their literature defaults)

The SUMO export's ``units`` field carries internal sentinels (``"0"`` /
``"SmallNumber"`` / ``"-BigNumber"``) for parameters and a dotted dialect
(``g COD.m-3``) for species, neither of which aquakin can parse. Real units are
stamped afterwards by the post-processor
[`aquakin/networks/_fix_sumo_units.py`](../aquakin/networks/_fix_sumo_units.py)
(run it after regenerating), which assigns each unit from the network's own
structure -- so ``network.time_unit`` resolves and ``check_units`` works.

Usage::

    python scripts/sumo_to_aquakin.py /tmp/sumo_json/asm3.json aquakin/networks/asm3.yaml [network_name]
    (cd aquakin/networks && python _fix_sumo_units.py)   # stamp real units
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Pattern: v<process_number>_<species> — SUMO's cross-reference between
# stoichiometric coefficients in the Petersen matrix.
_V_PATTERN = re.compile(r"^v(\d+)_(.+)$")

# Pattern: MRinh<A>_<B>_<K> — inhibition-ratio Monod that the upstream
# SUMO importer left as an opaque variable name. We expand it into
# monod_inh_ratio(A, B, K) since aquakin has a domain node for it.
_MRINH_PATTERN = re.compile(r"^MRinh([A-Za-z0-9]+)_([A-Za-z0-9]+)_(.+)$")


# ---------- AST helpers ----------


_OP_FN = {
    "add": lambda parts: sum(parts),
    "sub": lambda parts: parts[0] - sum(parts[1:]),
    "mul": lambda parts: _reduce_mul(parts),
    "div": lambda parts: _reduce_div(parts),
    "pow": lambda parts: parts[0] ** parts[1],
}


def _reduce_mul(parts: list[float]) -> float:
    out = 1.0
    for x in parts:
        out *= x
    return out


def _reduce_div(parts: list[float]) -> float:
    out = parts[0]
    for x in parts[1:]:
        out /= x
    return out


def asts_equal(a: dict, b: dict) -> bool:
    """Deep structural equality for AST nodes."""
    if a.keys() != b.keys():
        return False
    for k, v in a.items():
        if isinstance(v, list):
            if len(v) != len(b[k]):
                return False
            for ai, bi in zip(v, b[k]):
                if not asts_equal(ai, bi):
                    return False
        elif v != b[k]:
            return False
    return True


def evaluate_constant(
    ast: dict,
    params: dict[str, dict],
    aux_consts: dict[str, float],
    aux_asts: dict[str, dict],
    v_lookup: dict[tuple[int, str], float] | None = None,
) -> float:
    """Evaluate an AST that should resolve to a pure constant.

    Raises ``ValueError`` if the AST references a state variable, an
    unresolved name, or a ``v<n>_<species>`` cross-reference that is not
    yet in ``v_lookup``.
    """
    if v_lookup is None:
        v_lookup = {}
    if "const" in ast:
        return float(ast["const"])
    if "var" in ast:
        name = ast["var"]
        m = _V_PATTERN.match(name)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in v_lookup:
                return v_lookup[key]
            raise ValueError(f"Unresolved Petersen cross-reference: {name}")
        if name in aux_consts:
            return aux_consts[name]
        if name in params:
            return float(params[name]["default"])
        if name in aux_asts:
            return evaluate_constant(
                aux_asts[name], params, aux_consts, aux_asts, v_lookup
            )
        raise ValueError(f"Cannot evaluate '{name}' as a constant")
    if "ref" in ast:
        name = ast["ref"]
        if name in aux_consts:
            return aux_consts[name]
        if name in aux_asts:
            return evaluate_constant(
                aux_asts[name], params, aux_consts, aux_asts, v_lookup
            )
        raise ValueError(f"Cannot evaluate aux '{name}' as a constant")
    if "op" in ast:
        args = [
            evaluate_constant(a, params, aux_consts, aux_asts, v_lookup)
            for a in ast["args"]
        ]
        return _OP_FN[ast["op"]](args)
    raise ValueError(f"Unknown AST node: {ast}")


def _mrinh_states_and_param(
    name: str, state_names: set[str], params: set[str]
) -> tuple[str, str, str] | None:
    """If ``name`` matches the MRinh<A>_<B>_<K> pattern and A, B are state
    names, return (A, B, K). The K may have underscores in it."""
    m = _MRINH_PATTERN.match(name)
    if not m:
        return None
    A, B, K = m.groups()
    if A in state_names and B in state_names:
        return (A, B, K)
    return None


def collect_state_refs(
    ast: dict,
    state_names: set[str],
    aux_asts: dict[str, dict],
    visited: set[str] | None = None,
    params: set[str] | None = None,
) -> set[str]:
    """Collect every state-name reference in an AST (recursively through aux)."""
    if visited is None:
        visited = set()
    if params is None:
        params = set()
    refs: set[str] = set()
    if "const" in ast:
        return refs
    if "var" in ast:
        name = ast["var"]
        if name in state_names:
            refs.add(name)
        elif name in aux_asts and name not in visited:
            refs |= collect_state_refs(
                aux_asts[name], state_names, aux_asts, visited | {name}, params
            )
        else:
            mr = _mrinh_states_and_param(name, state_names, params)
            if mr:
                refs.add(mr[0])
                refs.add(mr[1])
        return refs
    if "ref" in ast:
        name = ast["ref"]
        if name in aux_asts and name not in visited:
            refs |= collect_state_refs(
                aux_asts[name], state_names, aux_asts, visited | {name}, params
            )
        return refs
    for a in ast.get("args", []):
        refs |= collect_state_refs(a, state_names, aux_asts, visited, params)
    return refs


def collect_param_refs(
    ast: dict,
    params: set[str],
    aux_asts: dict[str, dict],
    visited: set[str] | None = None,
    state_names: set[str] | None = None,
) -> set[str]:
    """Collect every parameter-name reference (recursively through aux)."""
    if visited is None:
        visited = set()
    if state_names is None:
        state_names = set()
    refs: set[str] = set()
    if "var" in ast:
        name = ast["var"]
        if name in params:
            refs.add(name)
        elif name in aux_asts and name not in visited:
            refs |= collect_param_refs(
                aux_asts[name], params, aux_asts, visited | {name}, state_names
            )
        else:
            mr = _mrinh_states_and_param(name, state_names, params)
            if mr and mr[2] in params:
                refs.add(mr[2])
        return refs
    if "ref" in ast:
        name = ast["ref"]
        if name in aux_asts and name not in visited:
            refs |= collect_param_refs(
                aux_asts[name], params, aux_asts, visited | {name}, state_names
            )
        return refs
    for a in ast.get("args", []):
        refs |= collect_param_refs(a, params, aux_asts, visited, state_names)
    return refs


# ---------- Monod pattern detection ----------


def detect_monod_pattern(ast: dict, state_names: set[str]) -> tuple | None:
    """Recognise ``X/(K+X)`` / ``K/(K+X)`` / ``(A/B)/(K+A/B)`` patterns.

    Returns one of:
      ('monod',       X_ast, K_ast)
      ('monod_inh',   X_ast, K_ast)
      ('monod_ratio', A_ast, B_ast, K_ast)
    or None if the AST does not match.
    """
    if ast.get("op") != "div" or len(ast.get("args", [])) != 2:
        return None
    num, denom = ast["args"]
    if denom.get("op") != "add" or len(denom["args"]) != 2:
        return None
    # The numerator must appear as one of the two add operands.
    matched_idx = None
    for i in (0, 1):
        if asts_equal(denom["args"][i], num):
            matched_idx = i
            break
    if matched_idx is None:
        return None
    other = denom["args"][1 - matched_idx]

    # Ratio form: numerator is itself a division (A/B).
    if num.get("op") == "div" and len(num["args"]) == 2:
        return ("monod_ratio", num["args"][0], num["args"][1], other)

    # Monod vs Monod-inhibition: decide by which side references states.
    num_states = collect_state_refs(num, state_names, {})
    other_states = collect_state_refs(other, state_names, {})
    if num_states and not other_states:
        return ("monod", num, other)
    if other_states and not num_states:
        return ("monod_inh", other, num)
    # Ambiguous (both or neither reference states); leave as raw arithmetic.
    return None


# ---------- AST → aquakin formula string ----------


_OP_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}
_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "pow": 4}
# Unary minus sits between div and pow.


def emit(
    ast: dict,
    state_names: set[str],
    expr_names: set[str],
    parent_prec: int = 0,
) -> str:
    """Convert an AST to an aquakin rate-expression string."""
    if "const" in ast:
        v = ast["const"]
        if float(v) < 0 and parent_prec > 0:
            return f"({v:g})"
        return f"{v:g}"
    if "var" in ast:
        name = ast["var"]
        if name in state_names:
            return f"[{name}]"
        # Catch SUMO's unexpanded MRinh<A>_<B>_<K> pattern: the upstream
        # importer recognises Msat / Minh / MRsat but not MRinh, leaving
        # these as opaque variable names. Expand them here.
        mrinh = _MRINH_PATTERN.match(name)
        if mrinh:
            A, B, K = mrinh.groups()
            if A in state_names and B in state_names:
                return f"monod_inh_ratio([{A}], [{B}], {K})"
        # Bare identifier — parameter or expression. Either resolves
        # automatically in aquakin's compile step.
        return name
    if "ref" in ast:
        # Expression reference — must match an `expressions:` entry.
        return ast["ref"]

    # Monod pattern recognition.
    pat = detect_monod_pattern(ast, state_names)
    if pat:
        if pat[0] == "monod":
            _, X, K = pat
            return f"monod({emit(X, state_names, expr_names)}, {emit(K, state_names, expr_names)})"
        if pat[0] == "monod_inh":
            _, X, K = pat
            return f"monod_inh({emit(X, state_names, expr_names)}, {emit(K, state_names, expr_names)})"
        _, A, B, K = pat
        return (
            f"monod_ratio({emit(A, state_names, expr_names)}, "
            f"{emit(B, state_names, expr_names)}, "
            f"{emit(K, state_names, expr_names)})"
        )

    op = ast["op"]
    sym = _OP_SYM[op]
    prec = _PREC[op]
    args = ast["args"]
    # Subtraction and division are left-associative and non-commutative, so a
    # right operand at *equal* precedence must be parenthesised: ``a / (b * c)``
    # is NOT ``a / b * c`` (which evaluates as ``a * c / b``), and ``a - (b - c)``
    # is NOT ``a - b - c``. The first operand keeps this op's precedence; the
    # rest are emitted as if under a tighter parent so an equal-precedence
    # subexpression gets its parentheses. Addition / multiplication are
    # associative, so all operands keep ``prec``.
    right_prec = prec + 1 if op in ("sub", "div") else prec
    parts = [emit(args[0], state_names, expr_names, prec)]
    parts += [emit(a, state_names, expr_names, right_prec) for a in args[1:]]
    result = f" {sym} ".join(parts)
    if prec < parent_prec:
        return f"({result})"
    return result


# ---------- Conversion ----------


def _expand_state_aliases(ast: dict, state_names: set[str]) -> dict:
    """Rewrite ``{"var": "X_Y"}`` where both X and Y are state names
    into ``{"op": "add", "args": [X, Y]}``.

    The SUMO importer occasionally leaves comma-separated state sums as
    underscore-joined opaque variable names (seen in ASM2D_TUD with
    ``XH_XPAO`` standing for ``XH + XPAO``). This pass restores the
    intended addition.
    """
    if "var" in ast:
        name = ast["var"]
        if name not in state_names:
            parts = name.split("_")
            if len(parts) >= 2 and all(p in state_names for p in parts):
                expanded = {"var": parts[0]}
                for p in parts[1:]:
                    expanded = {"op": "add", "args": [expanded, {"var": p}]}
                return expanded
        return ast
    if "op" in ast:
        return {
            "op": ast["op"],
            "args": [_expand_state_aliases(a, state_names) for a in ast["args"]],
        }
    return ast


def _ast_substitute(
    ast: dict,
    aux_asts: dict[str, dict],
    v_lookup: "dict[tuple[int, str], dict]",
    visited_aux: set[str] | None = None,
) -> dict:
    """Recursively inline every auxiliary and Petersen cross-reference.

    Used to build stoichiometric expressions for the symbolic-stoich
    output path. After substitution the resulting AST should contain only
    constants, parameter references, and arithmetic / negation.

    Raises ``KeyError`` on an unresolved ``v<i>_<sp>`` reference (caller
    catches and retries during the fixed-point loop). Raises
    ``ValueError`` on a cyclic auxiliary reference.
    """
    if visited_aux is None:
        visited_aux = set()
    if "const" in ast:
        return ast
    if "var" in ast:
        name = ast["var"]
        m = _V_PATTERN.match(name)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key not in v_lookup:
                raise KeyError(f"Unresolved Petersen cross-reference: {name}")
            return v_lookup[key]
        if name in aux_asts:
            if name in visited_aux:
                raise ValueError(f"Cyclic auxiliary reference: {name}")
            return _ast_substitute(
                aux_asts[name], aux_asts, v_lookup, visited_aux | {name}
            )
        return ast
    if "ref" in ast:
        name = ast["ref"]
        if name not in aux_asts:
            raise KeyError(f"Unknown auxiliary reference: {name}")
        if name in visited_aux:
            raise ValueError(f"Cyclic auxiliary reference: {name}")
        return _ast_substitute(
            aux_asts[name], aux_asts, v_lookup, visited_aux | {name}
        )
    if "op" in ast:
        return {
            "op": ast["op"],
            "args": [
                _ast_substitute(a, aux_asts, v_lookup, visited_aux)
                for a in ast["args"]
            ],
        }
    return ast


def _ast_contains_state(ast: dict, state_names: set[str]) -> bool:
    """True if any leaf is a state-variable reference."""
    if "const" in ast:
        return False
    if "var" in ast:
        return ast["var"] in state_names
    if "op" in ast:
        return any(_ast_contains_state(a, state_names) for a in ast["args"])
    return False


def _try_constant_fold(ast: dict) -> "float | None":
    """Reduce an AST to a numeric constant if every leaf is a ``const``;
    return ``None`` otherwise (any ``var`` leaf means a parameter remains)."""
    if "const" in ast:
        return float(ast["const"])
    if "var" in ast:
        return None
    if "op" in ast:
        children = [_try_constant_fold(a) for a in ast["args"]]
        if any(c is None for c in children):
            return None
        return _OP_FN[ast["op"]](children)
    return None


def _canonicalise_case(ast: dict, case_map: dict[str, str]) -> dict:
    """Rewrite ``{"var": "Foo"}`` → ``{"var": "foo"}`` when ``Foo`` is a
    case-only variant of a known parameter.

    SUMO XLSX occasionally has case mismatches between parameter
    declarations and uses (e.g. ASM2D_TUD declares ``Kac`` but the
    kinetic expression references ``KAc``).
    """
    if "var" in ast:
        name = ast["var"]
        canonical = case_map.get(name.lower())
        if canonical and canonical != name:
            return {"var": canonical}
        return ast
    if "op" in ast:
        return {
            "op": ast["op"],
            "args": [_canonicalise_case(a, case_map) for a in ast["args"]],
        }
    return ast


def convert(spec: dict, network_name: str | None = None) -> str:
    """Convert a SUMO JSON spec to an aquakin YAML string."""

    state_names_list = [s["name"] for s in spec["states"]]
    state_names = set(state_names_list)
    params = {p["name"]: p for p in spec["parameters"]}
    aux_asts: dict[str, dict] = dict(spec.get("auxiliaries", {}))

    # Pre-process: SUMO importer sometimes leaves "X_Y" (state-sum) as an
    # opaque variable name; rewrite to a proper addition.
    aux_asts = {n: _expand_state_aliases(a, state_names) for n, a in aux_asts.items()}
    for proc in spec["processes"]:
        proc["rate"] = _expand_state_aliases(proc["rate"], state_names)

    # Pre-process: case-canonicalise param/state references against the
    # declared parameter/state names, so case mismatches in the source
    # XLSX (Kac vs KAc, etc.) resolve.
    case_map: dict[str, str] = {}
    for n in params:
        case_map[n.lower()] = n
    for n in state_names:
        case_map[n.lower()] = n
    aux_asts = {n: _canonicalise_case(a, case_map) for n, a in aux_asts.items()}
    for proc in spec["processes"]:
        proc["rate"] = _canonicalise_case(proc["rate"], case_map)
    for proc_name, stoich in spec["stoichiometry"].items():
        for sp, ast in stoich.items():
            stoich[sp] = _canonicalise_case(ast, case_map)

    # Auxiliaries that reference state variables stay symbolic and become
    # entries in aquakin's ``expressions:`` block (rate-expression scope).
    # Those that don't are inlined wherever they're referenced.
    aux_kept: dict[str, dict] = {}
    for name, ast in aux_asts.items():
        refs = collect_state_refs(ast, state_names, aux_asts)
        if refs:
            aux_kept[name] = ast

    # Every parameter that's referenced anywhere — rate, kept aux, or
    # stoichiometry — becomes an aquakin network-level parameter. With
    # symbolic stoichiometry there's no longer any "frozen at default"
    # category; yield / N-content / fraction parameters are now calibratable.
    used_params: set[str] = set()
    for proc in spec["processes"]:
        used_params |= collect_param_refs(
            proc["rate"], set(params), aux_asts, state_names=state_names
        )
    for ast in aux_kept.values():
        used_params |= collect_param_refs(
            ast, set(params), aux_asts, state_names=state_names
        )
    for proc_name, stoich in spec["stoichiometry"].items():
        for ast in stoich.values():
            used_params |= collect_param_refs(
                ast, set(params), aux_asts, state_names=state_names
            )

    # ----- Build YAML -----
    out: list[str] = []
    meta = spec.get("metadata", {})
    out.append("network:")
    out.append(f"  name: {network_name or spec['name'].lower()}")
    out.append(f'  version: "1.0"')
    desc = (meta.get("description") or "").strip()
    if desc:
        out.append(f"  description: >")
        for line in desc.split("\n"):
            out.append(f"    {line.strip()}")
    if "reference" in meta:
        out.append("  references:")
        out.append(f"    - {json.dumps(meta['reference'])}")
    elif "references" in meta:
        out.append("  references:")
        for ref in meta["references"]:
            out.append(f"    - {json.dumps(ref)}")
    out.append("")

    out.append("species:")
    for s in spec["states"]:
        units = s.get("units", "")
        desc = (s.get("description") or "").replace('"', "'")
        out.append(
            f'  - {{name: {s["name"]}, default_concentration: 1.0, '
            f'units: "{units}", description: "{desc}"}}'
        )
    out.append("")

    out.append("conditions:")
    out.append("  - {name: T, default: 293.15, description: \"Temperature (K).\"}")
    out.append("")

    # Network-level shared parameters. Includes every parameter that
    # appears anywhere (rate, aux, or stoichiometric expression).
    out.append("parameters:")
    for name in sorted(used_params):
        p = params[name]
        default = float(p["default"])
        transform = p.get("transform", "none")
        units = p.get("units", "")
        # Bounds: use ±2 decades around default for positive_log; (0,1)
        # for logit; ±50% for none.
        if transform == "positive_log" and default > 0:
            lo, hi = default / 100.0, default * 100.0
        elif transform == "logit":
            lo, hi = 0.01, 0.99
        else:
            lo, hi = default - abs(default), default + abs(default) + 1e-9
        out.append(
            f"  {name}: {{value: {default!r}, transform: {transform}, "
            f"bounds: [{lo!r}, {hi!r}], units: \"{units}\"}}"
        )
    out.append("")

    # Named expressions (auxiliaries that reference state vars).
    if aux_kept:
        out.append("expressions:")
        for name, ast in aux_kept.items():
            formula = emit(ast, state_names, set(aux_kept.keys()))
            out.append(f'  {name}: "{formula}"')
        out.append("")

    # Fixed-point resolution of every stoich coefficient. Each entry is
    # AST-substituted (inlining auxiliaries and Petersen cross-references)
    # so the resulting AST contains only constants, parameter references,
    # and arithmetic / negation. We do this in passes until every entry
    # resolves (or none can progress, which signals a real cycle).
    process_order = [p["name"] for p in spec["processes"]]
    v_lookup: dict[tuple[int, str], dict] = {}
    pending: list[tuple[int, str, str, dict]] = []  # (proc_num, proc_name, sp_name, ast)
    for proc_num, proc_name in enumerate(process_order, start=1):
        for sp_name, ast in spec["stoichiometry"].get(proc_name, {}).items():
            if sp_name not in state_names:
                raise ValueError(
                    f"Stoichiometry of '{proc_name}' references unknown state '{sp_name}'"
                )
            pending.append((proc_num, proc_name, sp_name, ast))

    while True:
        progress = False
        still_pending: list[tuple[int, str, str, dict]] = []
        for entry in pending:
            proc_num, proc_name, sp_name, ast = entry
            try:
                substituted = _ast_substitute(ast, aux_asts, v_lookup)
            except KeyError:
                still_pending.append(entry)
                continue
            # Any state reference in stoichiometry is a hard error — the
            # schema requires stoich to be state-independent.
            if _ast_contains_state(substituted, state_names):
                raise ValueError(
                    f"Stoichiometric coefficient '{proc_name}'/{sp_name!r} "
                    f"depends on a state variable after auxiliary substitution; "
                    f"aquakin only supports state-independent stoichiometry."
                )
            v_lookup[(proc_num, sp_name)] = substituted
            progress = True
        pending = still_pending
        if not pending:
            break
        if not progress:
            unresolved = [
                f"{proc_name}/{sp_name}" for _, proc_name, sp_name, _ in pending
            ]
            raise ValueError(
                f"Could not resolve stoichiometric coefficients: {unresolved}"
            )

    # Reactions. SUMO reuses names like "Lysis" across different organism
    # processes; disambiguate by appending a 1-based numeric suffix to any
    # duplicates so aquakin's uniqueness check passes.
    seen_names: dict[str, int] = {}
    renamed: list[str] = []
    for proc in spec["processes"]:
        nm = proc["name"]
        count = seen_names.get(nm, 0) + 1
        seen_names[nm] = count
    name_counter: dict[str, int] = {}
    for proc in spec["processes"]:
        nm = proc["name"]
        if seen_names[nm] > 1:
            name_counter[nm] = name_counter.get(nm, 0) + 1
            renamed.append(f"{nm}_{name_counter[nm]}")
        else:
            renamed.append(nm)

    out.append("reactions:")
    for proc_num, (proc, unique_name) in enumerate(
        zip(spec["processes"], renamed, strict=True), start=1
    ):
        name = proc["name"]  # original (used for stoich lookup)
        rate_str = emit(proc["rate"], state_names, set(aux_kept.keys()))
        out.append(f"  - name: {unique_name}")
        if proc.get("description"):
            d = proc["description"].replace('"', "'")
            out.append(f'    description: "{d}"')
        out.append(f'    rate: "{rate_str}"')
        out.append(f"    stoichiometry:")
        for sp_name in spec["stoichiometry"].get(name, {}):
            ast = v_lookup[(proc_num, sp_name)]
            const_val = _try_constant_fold(ast)
            if const_val is not None:
                out.append(f"      {sp_name}: {const_val!r}")
            else:
                expr = emit(ast, state_names=set(), expr_names=set())
                # Stoich strings should not contain a leading `-` followed
                # by anything ambiguous; the parser handles unary minus but
                # YAML may interpret a leading dash as a list marker, so
                # always quote.
                out.append(f"      {sp_name}: \"{expr}\"")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    json_path = Path(sys.argv[1])
    yaml_path = Path(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else None
    spec = json.loads(json_path.read_text())
    yaml_text = convert(spec, network_name=name)
    yaml_path.write_text(yaml_text)
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
